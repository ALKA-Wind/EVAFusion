import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
import torch
from test_support import scratch_directory

ROOT = Path(__file__).resolve().parents[1]


class SpatialReward(torch.nn.Module):
    """An analytic reward gives a known signal without a multi-GB dependency in unit tests."""
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, fused, ir, vi):
        score = fused.mean((1, 2, 3)) * self.scale
        return {'scores': {k: score for k in ('Artifacts', 'Texture_Preservation', 'Thermal_Retention', 'Sharpness')},
                'heatmaps': {'artifact_heatmap': fused.mean(1)}}


class SamGrpoTraining(unittest.TestCase):
    def test_standalone_training_entry_works_outside_the_project_directory(self):
        with scratch_directory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / 'training/finetune.py'), '--help'],
                                    cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--samples-per-input', result.stdout)
            self.assertIn('--target-kl', result.stdout)
            self.assertNotIn('--group-size', result.stdout)
            self.assertIn('--policy-std', result.stdout)
            self.assertNotIn('--ratio-alpha', result.stdout)
            self.assertNotIn('--fusion-weight', result.stdout)

    def test_literal_first_updates_report_zero_gradient_and_keep_frozen_state(self):
        from utils.fusion import FusionModel
        from utils.grpo import collect_rollout, update_policy
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = FusionModel.from_checkpoint(ROOT / 'ckpt/DCEvo_fusion.pth')
        reference = copy.deepcopy(model).eval().requires_grad_(False)
        reward = SpatialReward()
        ir, vis = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
        mask = torch.zeros(2, 16, 16, dtype=torch.bool)
        mask[0, :8] = True; mask[1, 8:12] = True
        rng_before = torch.random.get_rng_state().clone()
        rollout = collect_rollout(model, reference, reward, ir, vis,
                                  mask_batches=[mask], reward_size=32)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        self.assertFalse(rollout.rewards.requires_grad)
        self.assertEqual(tuple(rollout.rewards.shape), (1, 2))
        self.assertEqual(rollout.regions.sam_counts.tolist(), [2])
        self.assertGreater(rollout.advantages.abs().sum().item(), 0)
        old_fused = rollout.old_fused.clone()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        reference_before = reference.Encoder.patch_embed.proj.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(2):
            metrics = update_policy(model, optimizer, ir, vis, rollout)
            self.assertTrue(np.isfinite(metrics['loss']))
            self.assertEqual(metrics['grad_norm'], 0.)
            self.assertTrue(metrics['zero_objective_gradient'])
            self.assertTrue(metrics['policy_at_old'])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        torch.testing.assert_close(reference.Encoder.patch_embed.proj.weight, reference_before)
        torch.testing.assert_close(rollout.old_fused, old_fused)
        self.assertFalse(hasattr(rollout, 'actions'))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))
        self.assertIsNone(reward.scale.grad)

    def test_cli_precomputed_masks_train_and_save_an_inference_compatible_checkpoint(self):
        import h5py
        from transformers import ViTConfig, ViTModel
        from reward_model.model import RAHF
        from utils.fusion import FusionModel
        torch.set_num_threads(2)
        with scratch_directory() as tmp:
            root = Path(tmp)
            config = ViTConfig(image_size=32, patch_size=16, hidden_size=64,
                               num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
            ViTModel(config).save_pretrained(root / 'vit')
            reward = RAHF(vit_model=str(root / 'vit'), image_size=32)
            torch.save(reward.state_dict(), root / 'reward.pt')
            with h5py.File(root / 'data.h5', 'w') as file:
                group = file.create_group('sample')
                group['vis'] = np.ones((16, 16), dtype=np.float32)
                group['ir'] = np.zeros((16, 16), dtype=np.float32)
                mask = np.zeros((16, 16), dtype=bool); mask[:8] = True
                group['mask_0'] = mask
                mask2 = np.zeros_like(mask); mask2[8:12] = True
                group['mask_1'] = mask2
            command = [sys.executable, '-m', 'training.finetune', '--device', 'cpu',
                       '--mask-source', 'h5', '--h5', str(root / 'data.h5'), '--batch-size', '1',
                       '--vit-model', str(root / 'vit'), '--reward-checkpoint', str(root / 'reward.pt'),
                       '--epochs', '1', '--max-steps', '1', '--update-epochs', '2', '--weight-decay', '0',
                       '--output-dir', str(root / 'run')]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            records = [json.loads(x) for x in (root / 'run/train.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(x['semantic_regions'] == [2] for x in records))
            self.assertTrue(all(not x['zero_policy_gradient'] for x in records))
            self.assertTrue(all(x['update_accepted'] for x in records))
            self.assertTrue(all(x['parameter_delta_norm'] > 0 for x in records))
            self.assertEqual(records[0]['clip_fraction'], 0.)
            self.assertEqual(records[0]['kl'], 0.)
            self.assertEqual(records[0]['ratio_min'], 1.)
            self.assertGreater(records[0]['policy_grad_norm'], 0.)
            self.assertLessEqual(records[0]['post_old_kl'], .02)
            self.assertIn('deterministic_full_image_reward_after', records[-1])
            config = json.loads((root / 'run/run_config.json').read_text())
            self.assertEqual(config['method'], 'sam_stochastic_regional_grpo')
            self.assertEqual(config['advantage_axis'], 'semantic_regions')
            self.assertEqual(config['reward_preprocess'], 'processor')
            self.assertEqual(config['samples_per_input'], 1)
            self.assertEqual(config['full_image_role'], 'monitoring_only')
            self.assertNotIn('group_size', config)
            model = FusionModel.from_checkpoint(root / 'run/final.pth').eval()
            with torch.no_grad():
                fused = model(torch.zeros(1, 1, 16, 16), torch.ones(1, 1, 16, 16))
            self.assertEqual(tuple(fused.shape), (1, 1, 16, 16))

    def test_full_image_monitoring_and_scoring_microbatches_do_not_change_region_advantages(self):
        from utils.grpo import collect_rollout
        class IdentityFusion(torch.nn.Module):
            def forward(self, ir, vis):
                return (ir + vis) / 2
        class Segmenter:
            def generate(self, images):
                self.seen = images.clone()
                return [[np.array([[1, 1, 0, 0]], dtype=bool),
                         np.array([[0, 0, 1, 0]], dtype=bool)]]
        ir = torch.zeros(1, 1, 1, 4)
        vis = torch.tensor([[[[.2, .4, .6, .8]]]])
        segmenter = Segmenter()
        model = IdentityFusion()
        a = collect_rollout(model, copy.deepcopy(model), SpatialReward(), ir, vis,
                            segmenter=segmenter, reward_size=4, score_batch_size=1)
        b = collect_rollout(model, copy.deepcopy(model), SpatialReward(), ir, vis,
                            segmenter=segmenter, reward_size=4, score_batch_size=3)
        torch.testing.assert_close(segmenter.seen, (ir + vis) / 2)
        torch.testing.assert_close(a.rewards, b.rewards)
        torch.testing.assert_close(a.advantages, b.advantages)
        self.assertEqual(tuple(a.full_image_rewards.shape), (1,))
        self.assertEqual(tuple(a.advantages.shape), (1, 2))

    def test_old_output_is_a_snapshot_even_when_model_returns_parameter_storage(self):
        from utils.grpo import collect_rollout, update_policy
        class PixelPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image = torch.nn.Parameter(torch.tensor([[[[.2, .4, .6, .8]]]]))
            def forward(self, ir, vis):
                return self.image
        model = PixelPolicy()
        reference = copy.deepcopy(model)
        ir = torch.zeros(1, 1, 1, 4)
        masks = [[np.array([[1, 1, 0, 0]], dtype=bool), np.array([[0, 0, 1, 1]], dtype=bool)]]
        rollout = collect_rollout(model, reference, SpatialReward(), ir, ir,
                                  mask_batches=masks, reward_size=4)
        old = rollout.old_fused.clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.01)
        first = update_policy(model, optimizer, ir, ir, rollout)
        self.assertTrue(first['zero_objective_gradient'])
        self.assertTrue(first['policy_at_old'])
        self.assertTrue(first['weight_decay_active'])
        self.assertGreater((model.image.detach() - old).abs().sum().item(), 0.)
        torch.testing.assert_close(rollout.old_fused, old, rtol=0, atol=0)
        second = update_policy(model, optimizer, ir, ir, rollout)
        self.assertFalse(second['policy_at_old'])
        self.assertGreater(second['grad_norm'], 0.)


if __name__ == '__main__':
    unittest.main()
