"""Probability and startup contracts for SAM regional stochastic GRPO."""
import copy
import math
import unittest

import numpy as np
import torch

from utils.sam_regions import build_regions


def two_regions():
    return build_regions([[np.array([[1, 1, 0, 0]], dtype=bool),
                           np.array([[0, 0, 1, 1]], dtype=bool)]], 1, 4)


class ProbabilityMath(unittest.TestCase):
    def test_fusion_logits_preserve_default_output_and_checkpoint_keys(self):
        from utils.fusion import FusionModel
        torch.set_num_threads(2)
        model = FusionModel().eval()
        ir, vi = torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16)
        keys = set(model.state_dict())
        with torch.no_grad():
            image = model(ir, vi)
            logits = model(ir, vi, return_logits=True)
            logits_with_features, features = model(ir, vi, return_features=True, return_logits=True)
        torch.testing.assert_close(image, logits.sigmoid(), rtol=0, atol=0)
        torch.testing.assert_close(logits, logits_with_features)
        self.assertEqual(len(features), 4)
        self.assertEqual(set(model.state_dict()), keys)

    def test_masked_joint_ratios_and_kl_match_scalar_oracles(self):
        from utils.stochastic_grpo import regional_log_ratios, regional_gaussian_kl
        old = torch.zeros(1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        cur = torch.tensor([[[[.1, -.2, .2, -.1]]]], dtype=torch.float64, requires_grad=True)
        action = torch.tensor([[[[.3, -.1, -.3, .4]]]], dtype=torch.float64, requires_grad=True)
        log_ratio = regional_log_ratios(cur, old, action, two_regions(), .5)
        oracle = []
        for indices in ((0, 1), (2, 3)):
            oracle.append(sum(-((float(action.detach().flatten()[i])-float(cur.detach().flatten()[i]))**2
                                - float(action.detach().flatten()[i])**2) / (2*.5**2) for i in indices))
        torch.testing.assert_close(log_ratio, torch.tensor([oracle], dtype=torch.float64))
        self.assertGreater(float(log_ratio[0, 0].detach()), 0)
        self.assertLess(float(log_ratio[0, 1].detach()), 0)
        kl = regional_gaussian_kl(cur, old, two_regions(), .5)
        torch.testing.assert_close(kl, torch.tensor([[.1, .1]], dtype=torch.float64))
        log_ratio.sum().backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(action.grad)
        self.assertGreater(cur.grad.abs().sum().item(), 0)

    def test_clipping_uses_both_signs_of_advantage_and_real_probability_ratio(self):
        from utils.stochastic_grpo import stochastic_grpo_loss
        old = torch.zeros(1, 1, 1, 4, dtype=torch.float64)
        cur = torch.full_like(old, .1, requires_grad=True)
        # Per-region log ratios 0.4 and -0.4 for sigma=1.
        action = torch.tensor([[[[2.05, 2.05, -1.95, -1.95]]]], dtype=torch.float64)
        a = torch.tensor([[1., -1.]], dtype=torch.float64, requires_grad=True)
        loss, stats = stochastic_grpo_loss(cur, old, action, a, two_regions(), old,
                                            policy_std=1., beta=0.)
        expected = -.5*(1.2-.8)
        self.assertAlmostEqual(float(loss.detach()), expected, places=12)
        self.assertEqual(float(stats['clip_fraction']), 1.)
        loss.backward()
        self.assertIsNone(a.grad)

    def test_identical_policy_has_nonzero_reward_gradient_at_ratio_one(self):
        from utils.stochastic_grpo import stochastic_grpo_loss
        cur = torch.zeros(1, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        action = torch.tensor([[[[.2, .2, -.1, -.1]]]], dtype=torch.float64)
        a = torch.tensor([[1., -1.]], dtype=torch.float64)
        loss, stats = stochastic_grpo_loss(cur, cur.detach(), action, a, two_regions(), cur.detach())
        self.assertEqual(float(stats['ratio_min']), 1.)
        self.assertEqual(float(stats['ratio_max']), 1.)
        self.assertEqual(float(stats['kl']), 0.)
        loss.backward()
        torch.testing.assert_close(cur.grad, torch.tensor([[[[-10., -10., -5., -5.]]]], dtype=torch.float64))

    def test_invalid_shapes_and_distributions_are_rejected(self):
        from utils.stochastic_grpo import regional_log_ratios
        x = torch.zeros(1, 1, 1, 4)
        for std in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                regional_log_ratios(x, x, x, two_regions(), std)
        with self.assertRaises(ValueError):
            regional_log_ratios(x, x[..., :2], x, two_regions(), .1)
        with self.assertRaises(ValueError):
            regional_log_ratios(x, x, x+float('nan'), two_regions(), .1)


class PixelPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[[[.2, .4, -.1, -.2]]]]))

    def forward(self, ir, vis, return_logits=False):
        values = self.logits.expand(ir.shape[0], -1, -1, -1)
        return values if return_logits else values.sigmoid()


class SpatialReward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, fused, ir, vis):
        score = fused.mean((1, 2, 3))*self.scale
        return {'scores': {key: score for key in ('Artifacts', 'Texture_Preservation', 'Thermal_Retention', 'Sharpness')},
                'heatmaps': {'artifact_heatmap': torch.zeros_like(fused[:, 0])}}


class RolloutAndUpdates(unittest.TestCase):
    def setup_rollout(self, samples_per_input=1, batch_size=1, microbatch=1):
        from utils.stochastic_grpo import collect_rollout
        torch.manual_seed(9)
        model = PixelPolicy()
        reference, reward = copy.deepcopy(model), SpatialReward()
        ir, vi = torch.zeros(batch_size, 1, 1, 4), torch.ones(batch_size, 1, 1, 4)
        class Segmenter:
            def generate(self, images):
                self.seen = images.detach().clone()
                return [[np.array([[1, 1, 0, 0]], dtype=bool),
                         np.array([[0, 0, 1, 0]], dtype=bool)] for _ in images]
        segmenter = Segmenter()
        rollout = collect_rollout(model, reference, reward, ir, vi, segmenter=segmenter,
                                  reward_size=(1, 4), preprocess='legacy', score_batch_size=microbatch,
                                  samples_per_input=samples_per_input)
        return model, reference, reward, ir, vi, segmenter, rollout

    def test_sampled_sam_images_and_regional_advantages_never_pool_inputs_or_samples(self):
        from utils.grpo import normalize_region_advantages
        a = self.setup_rollout(samples_per_input=2, batch_size=2)
        b = self.setup_rollout(samples_per_input=2, batch_size=2, microbatch=3)
        model, reference, reward, ir, vi, segmenter, rollout = a
        self.assertEqual(rollout.source_indices.tolist(), [0, 0, 1, 1])
        self.assertEqual(tuple(rollout.rewards.shape), (4, 2))
        torch.testing.assert_close(segmenter.seen, rollout.actions.sigmoid())
        self.assertFalse(torch.equal(segmenter.seen, model(ir, vi).repeat_interleave(2, 0)))
        torch.testing.assert_close(rollout.advantages, normalize_region_advantages(rollout.rewards, rollout.regions.valid))
        torch.testing.assert_close(rollout.advantages.mean(1), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(rollout.rewards, b[-1].rewards)
        torch.testing.assert_close(rollout.actions, b[-1].actions)
        self.assertFalse(rollout.actions.requires_grad)
        self.assertFalse(rollout.advantages.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in reference.parameters()))
        self.assertTrue(all(not p.requires_grad for p in reward.parameters()))

    def test_zero_decay_update_changes_policy_only_and_keeps_rollout_fixed(self):
        from utils.stochastic_grpo import update_policy
        model, reference, reward, ir, vi, _, rollout = self.setup_rollout()
        before = model.logits.detach().clone()
        frozen = [rollout.old_logits.clone(), rollout.reference_logits.clone(), rollout.actions.clone(), rollout.advantages.clone()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.)
        stats = update_policy(model, optimizer, ir, vi, rollout)
        self.assertGreater(stats['policy_grad_norm'], 0.)
        self.assertFalse(stats['zero_objective_gradient'])
        self.assertEqual(stats['ratio_min'], 1.)
        self.assertEqual(stats['kl'], 0.)
        self.assertTrue(stats['update_accepted'])
        self.assertGreater((model.logits.detach()-before).abs().sum().item(), 0.)
        torch.testing.assert_close(reference.logits, before)
        self.assertIsNone(reward.scale.grad)
        for actual, expected in zip((rollout.old_logits, rollout.reference_logits, rollout.actions, rollout.advantages), frozen):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejected_adam_step_restores_parameters_state_and_learning_rate(self):
        from utils.stochastic_grpo import update_policy
        model, _, _, ir, vi, _, rollout = self.setup_rollout()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.)
        # Populate Adam's moment and step state, then collect a new old snapshot.
        model.logits.square().sum().backward(); optimizer.step(); optimizer.zero_grad()
        from utils.stochastic_grpo import collect_rollout
        rollout = collect_rollout(model, copy.deepcopy(model), SpatialReward(), ir, vi,
                                  mask_batches=[[np.array([[1,1,0,0]], dtype=bool),
                                                 np.array([[0,0,1,0]], dtype=bool)]],
                                  reward_size=(1,4), preprocess='legacy')
        before = model.logits.detach().clone()
        state = copy.deepcopy(optimizer.state_dict())
        with self.assertRaisesRegex(RuntimeError, 'accept'):
            update_policy(model, optimizer, ir, vi, rollout, target_kl=1e-20, max_backtracks=0)
        torch.testing.assert_close(model.logits, before, rtol=0, atol=0)
        actual_state = optimizer.state_dict()
        self.assertEqual(actual_state['param_groups'], state['param_groups'])
        for index, values in state['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(actual_state['state'][index][key], value, rtol=0, atol=0)
        stats = update_policy(model, optimizer, ir, vi, rollout, target_kl=.01, max_backtracks=16)
        self.assertGreater(stats['backtracks'], 0)
        self.assertLessEqual(stats['post_old_kl'], .01)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1.)
        self.assertGreater((model.logits.detach()-before).abs().sum().item(), 0.)

    def test_controlled_regional_policy_gradient_moves_in_known_direction(self):
        from utils.stochastic_grpo import update_policy
        model, _, _, ir, vi, _, rollout = self.setup_rollout()
        rollout.actions = rollout.old_logits + torch.tensor([[[[.2, .2, -.1, 0.]]]])
        rollout.advantages = torch.tensor([[1., -1.]])
        before = model.logits.detach().clone()
        stats = update_policy(model, torch.optim.SGD(model.parameters(), lr=1e-4), ir, vi, rollout)
        # Positive-reward action goes up; negative-reward sampled action goes
        # down, so its probability falls: both regional means increase here.
        self.assertTrue((model.logits.detach().flatten()[:3] > before.flatten()[:3]).all())
        self.assertEqual(float(model.logits.detach().flatten()[3]), float(before.flatten()[3]))
        self.assertGreater(stats['policy_grad_norm'], 0.)

    def test_single_region_has_zero_advantage_and_zero_decay_cannot_fake_startup(self):
        from utils.stochastic_grpo import collect_rollout, update_policy
        model = PixelPolicy(); ir = torch.zeros(1,1,1,4)
        rollout = collect_rollout(model, copy.deepcopy(model), SpatialReward(), ir, ir,
                                  mask_batches=[[np.array([[1,1,0,0]], dtype=bool)]], reward_size=(1,4))
        self.assertEqual(float(rollout.advantages.abs().sum()), 0.)
        before = model.logits.detach().clone()
        stats = update_policy(model, torch.optim.Adam(model.parameters(), lr=1e-4), ir, ir, rollout)
        self.assertEqual(stats['policy_grad_norm'], 0.)
        self.assertTrue(stats['zero_objective_gradient'])
        torch.testing.assert_close(model.logits, before, rtol=0, atol=0)

    def test_multiseed_short_runs_improve_analytic_expected_reward(self):
        from utils.stochastic_grpo import collect_rollout, update_policy
        improvements = []
        ir = torch.zeros(1,1,1,4)
        masks = [[np.array([[1,1,0,0]], dtype=bool), np.array([[0,0,1,1]], dtype=bool)]]
        for seed in (1, 7, 19):
            torch.manual_seed(seed)
            model = PixelPolicy()
            with torch.no_grad():
                model.logits.zero_()
            reference = copy.deepcopy(model)
            initial = float(model(ir, ir).detach().mean())
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            for _ in range(8):
                rollout = collect_rollout(model, reference, SpatialReward(), ir, ir,
                                          mask_batches=masks, samples_per_input=16,
                                          reward_size=(1,4), preprocess='legacy', score_batch_size=32)
                update_policy(model, optimizer, ir, ir, rollout, beta=0.)
            # For symmetric Gaussian logit noise, increasing each pixel mean
            # increases E[sigmoid(z)] as well as this deterministic rendering.
            self.assertTrue((model.logits.detach() > 0).all())
            improvements.append(float(model(ir, ir).detach().mean())-initial)
        self.assertGreater(min(improvements), .005)


if __name__ == '__main__':
    unittest.main()
