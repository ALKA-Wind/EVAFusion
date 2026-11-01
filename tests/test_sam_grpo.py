"""Contracts for the literal regional objective in paper equations (3)-(5)."""
import math
import unittest
import numpy as np
import torch


def two_regions():
    from utils.sam_regions import build_regions
    return build_regions([[np.array([[1, 1, 0, 0]], dtype=bool),
                           np.array([[0, 0, 1, 1]], dtype=bool)]], 1, 4)


class RegionAndPolicyMath(unittest.TestCase):
    def test_advantages_compare_regions_per_image_and_ignore_padding(self):
        from utils.grpo import normalize_region_advantages
        rewards = torch.tensor([[1., 3., 999.], [9., 9., -999.]], requires_grad=True)
        valid = torch.tensor([[True, True, False], [True, True, False]])
        actual = normalize_region_advantages(rewards, valid, eps=0.)
        torch.testing.assert_close(actual, torch.tensor([[-1., 1., 0.], [0., 0., 0.]]))
        self.assertFalse(actual.requires_grad)

    def test_single_region_and_constant_rewards_have_zero_advantages(self):
        from utils.grpo import normalize_region_advantages
        torch.testing.assert_close(normalize_region_advantages(torch.tensor([[7.]])), torch.zeros(1, 1))
        torch.testing.assert_close(normalize_region_advantages(torch.ones(2, 3)), torch.zeros(2, 3))

    def test_regions_are_semantic_only_deduplicated_and_padded(self):
        from utils.sam_regions import build_regions
        large = np.zeros((4, 4), dtype=bool); large[:2] = True
        small = np.zeros_like(large); small[2, :2] = True
        regions = build_regions([[large, large.copy(), small], [small]], 4, 4)
        self.assertEqual(regions.sam_counts.tolist(), [2, 1])
        self.assertEqual(regions.valid.tolist(), [[True, True], [True, False]])
        self.assertEqual(tuple(regions.masks.shape), (2, 2, 1, 4, 4))
        torch.testing.assert_close(regions.weights, torch.tensor([[.5, .5], [1., 0.]]))
        area = build_regions([[large, small]], 4, 4, weighting='area')
        torch.testing.assert_close(area.weights, torch.tensor([[.8, .2]]))

    def test_empty_regions_default_to_error_and_explicit_fallback_has_zero_advantage(self):
        from utils.sam_regions import build_regions
        from utils.grpo import normalize_region_advantages
        with self.assertRaisesRegex(ValueError, 'region'):
            build_regions([[]], 4, 4)
        regions = build_regions([[]], 4, 4, empty_policy='full-image')
        self.assertEqual(regions.sam_counts.tolist(), [0])
        self.assertTrue(regions.masks.all())
        torch.testing.assert_close(normalize_region_advantages(torch.ones(1, 1), regions.valid), torch.zeros(1, 1))

    def test_ratio_uses_region_l1_change_over_old_l1_and_guards_zero_denominator(self):
        from utils.grpo import region_change_ratios
        old = torch.tensor([[[[.5, .5, 0., 0.]]]], requires_grad=True)
        current = torch.tensor([[[[.6, .4, .1, .1]]]], requires_grad=True)
        ratios = region_change_ratios(current, old, two_regions(), alpha=.5, denominator_eps=.1)
        torch.testing.assert_close(ratios, torch.tensor([[1.1, 2.]]))
        ratios.sum().backward()
        self.assertIsNone(old.grad)
        torch.testing.assert_close(current.grad, torch.tensor([[[[.5, -.5, 5., 5.]]]]))

    def test_clipped_regional_objective_and_forward_kl_match_scalar_equations(self):
        from utils.grpo import regional_grpo_loss
        current = torch.tensor([[[[.7, .7, .55, .55]]]], dtype=torch.float64, requires_grad=True)
        old = torch.full_like(current, .5, requires_grad=True)
        advantage = torch.tensor([[1., -1.]], dtype=torch.float64, requires_grad=True)
        loss, stats = regional_grpo_loss(current, old, advantage, two_regions(), old,
                                         alpha=1., clip_epsilon=.2, beta=.1)
        # Ratios [1.4, 1.1]; positive advantage clips, negative advantage does not.
        expected_policy = -.5 * (1.2 - 1.1)
        logits = [.7, .7, .55, .55]
        probs = [math.exp(x) / sum(math.exp(y) for y in logits) for x in logits]
        expected_kl = sum(p * math.log(p / .25) for p in probs)
        self.assertAlmostEqual(float(stats['policy_loss']), expected_policy, places=10)
        self.assertAlmostEqual(float(stats['kl']), expected_kl, places=10)
        self.assertAlmostEqual(float(loss.detach()), expected_policy + .1 * expected_kl, places=10)
        self.assertEqual(float(stats['clip_fraction']), .5)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)
        self.assertGreater(current.grad.abs().sum().item(), 0.)

    def test_identical_current_old_reference_has_zero_gradient_without_surrogate_tricks(self):
        from utils.grpo import regional_grpo_loss
        current = torch.full((1, 1, 1, 4), .5, requires_grad=True)
        loss, stats = regional_grpo_loss(current, current.detach(), torch.tensor([[-1., 1.]]),
                                         two_regions(), current.detach())
        loss.backward()
        self.assertEqual(current.grad.abs().sum().item(), 0.)
        self.assertEqual(float(stats['kl']), 0.)
        self.assertEqual(float(stats['ratio_min']), 1.)
        self.assertEqual(float(stats['ratio_max']), 1.)

    def test_kl_direction_temperature_and_shift_invariance(self):
        from utils.grpo import categorical_image_kl
        current = torch.tensor([[[[0., 2.]]]], dtype=torch.float64, requires_grad=True)
        ref = torch.tensor([[[[0., 0.]]]], dtype=torch.float64, requires_grad=True)
        p = math.exp(1) / (1 + math.exp(1))
        expected = p * math.log(p / .5) + (1-p) * math.log((1-p) / .5)
        kl = categorical_image_kl(current, ref, temperature=2.)
        self.assertAlmostEqual(float(kl.detach()), expected, places=12)
        torch.testing.assert_close(kl, categorical_image_kl(current + 100, ref, temperature=2.))
        kl.backward()
        self.assertIsNone(ref.grad)

    def test_nonfinite_inputs_and_invalid_hyperparameters_are_rejected(self):
        from utils.grpo import normalize_region_advantages, region_change_ratios, categorical_image_kl
        with self.assertRaises(ValueError):
            normalize_region_advantages(torch.tensor([[float('nan'), 1.]]))
        with self.assertRaises(ValueError):
            normalize_region_advantages(torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.bool))
        current = torch.ones(1, 1, 1, 4)
        for alpha, eps in ((-1., 1e-6), (1., 0.), (float('nan'), 1e-6)):
            with self.assertRaises(ValueError):
                region_change_ratios(current, current, two_regions(), alpha=alpha, denominator_eps=eps)
        with self.assertRaises(ValueError):
            categorical_image_kl(current, current, temperature=0.)

    def test_batched_overlapping_regions_padding_and_multichannel_gradient_match_scalar_oracle(self):
        from utils.grpo import normalize_region_advantages, regional_grpo_loss
        from utils.sam_regions import build_regions
        masks = [[np.array([[1, 1], [0, 0]], dtype=bool), np.array([[0, 1], [1, 1]], dtype=bool)],
                 [np.array([[1, 0], [1, 0]], dtype=bool)]]
        regions = build_regions(masks, 2, 2, weighting='area')
        old = torch.full((2, 2, 2, 2), .5, dtype=torch.float64)
        current = (old + torch.tensor([.01, -.02, .03, -.04, .02, .01, -.01, .03,
                                       .02, .04, -.01, -.02, .03, .01, -.03, .02],
                                      dtype=torch.float64).reshape_as(old)).requires_grad_()
        advantage = normalize_region_advantages(torch.tensor([[2., 5.], [9., 999.]], dtype=torch.float64),
                                                 regions.valid, eps=0.)
        expected, gradient = 0., torch.zeros_like(current)
        for b in range(2):
            for k in range(2):
                if not regions.valid[b, k]:
                    continue
                mask = regions.masks[b, k, 0]
                denom = sum(float(old[b, c, y, x]) for c in range(2) for y in range(2) for x in range(2) if mask[y, x])
                delta = sum(abs(float(current.detach()[b, c, y, x] - old[b, c, y, x]))
                            for c in range(2) for y in range(2) for x in range(2) if mask[y, x])
                a, w = float(advantage[b, k]), float(regions.weights[b, k])
                expected -= w * (1 + .5 * delta / denom) * a / 2
                for c in range(2):
                    for y in range(2):
                        for x in range(2):
                            if mask[y, x]:
                                sign = 1 if current.detach()[b, c, y, x] > old[b, c, y, x] else -1
                                gradient[b, c, y, x] -= w * a * .5 * sign / denom / 2
        loss, stats = regional_grpo_loss(current, old, advantage, regions, old, alpha=.5, beta=0.)
        self.assertAlmostEqual(float(loss.detach()), expected, places=12)
        self.assertEqual(float(stats['clip_fraction']), 0.)
        loss.backward()
        torch.testing.assert_close(current.grad, gradient)


if __name__ == '__main__':
    unittest.main()
