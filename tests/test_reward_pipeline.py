"""Offline reward-network checks using a tiny real ViT, not downloaded weights."""
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel
from test_support import scratch_directory


class RewardPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        scratch = Path(__file__).resolve().parents[1] / 'tmp/restoration_tests'
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = scratch_directory()
        cls.path = Path(cls.temp.__enter__())
        config = ViTConfig(image_size=32, patch_size=16, hidden_size=64,
                           num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
        ViTModel(config).save_pretrained(cls.path, safe_serialization=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.__exit__(None, None, None)

    def make_model(self):
        from reward_model.model import RAHF
        return RAHF(vit_model=str(self.path), image_size=32)

    def test_single_sample_keeps_the_score_batch_dimension(self):
        model = self.make_model().eval()
        with torch.no_grad():
            result = model(*(torch.rand(1, 3, 32, 32) for _ in range(3)))
        self.assertEqual(set(result['scores']),
                         {'Thermal_Retention', 'Texture_Preservation', 'Artifacts', 'Sharpness', 'Overall_Score'})
        for score in result['scores'].values():
            self.assertEqual(tuple(score.shape), (1,))
        self.assertEqual(tuple(result['heatmaps']['artifact_heatmap'].shape), (1, 32, 32))

    def test_imported_evaluation_accepts_empty_artifact_maps(self):
        from RichHF.train import evaluate_model
        model = self.make_model()
        batch = {k: torch.rand(2, 3, 32, 32) for k in ('fused', 'ir', 'vi')}
        batch['artifact_heatmap'] = torch.zeros(2, 32, 32)
        for i, key in enumerate(model.score_types):
            batch[key] = torch.tensor([.5, .5])
        metrics = evaluate_model(model, [batch], torch.nn.MSELoss(), 'cpu')
        self.assertGreater(metrics['loss/val'], 0)
        self.assertTrue(torch.isfinite(torch.tensor(metrics['avg_metric'])))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))

    def test_inference_exports_all_samples_including_last_single_sample(self):
        import json
        from RichHF.inference import infer
        from test_support import scratch_directory
        model = self.make_model()
        batches = []
        for count in (2, 1):
            batch = {k: torch.rand(count, 3, 32, 32) for k in ('fused', 'ir', 'vi')}
            for key in model.score_types:
                batch[key] = torch.full((count,), .5)
            batches.append(batch)
        with scratch_directory() as tmp:
            records = infer(model, batches, 'cpu', tmp)
            self.assertEqual(len(records), 3)
            self.assertEqual(len(json.loads((Path(tmp) / 'scores.json').read_text())), 3)
            self.assertEqual(len(list(Path(tmp).glob('*_artifact_heatmap.png'))), 3)

    def test_reward_preprocess_preserves_gradients_and_processor_range(self):
        from utils.reward import prepare_reward_input
        x = torch.tensor([[[[0., 1.], [.5, .5]]]], requires_grad=True)
        out = prepare_reward_input(x, size=2, mode='processor')
        self.assertEqual(tuple(out.shape), (1, 3, 2, 2))
        self.assertEqual(out.min().item(), -1.)
        self.assertEqual(out.max().item(), 1.)
        out.sum().backward()
        self.assertTrue(torch.all(x.grad != 0))


if __name__ == '__main__':
    unittest.main()
