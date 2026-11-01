"""Stochastic SAM regional GRPO; advantages compare regions of ONE image.

The paper's displacement ratio is explicitly revised to a Gaussian action
likelihood ratio. Literal-paper diagnostics remain in utils.grpo.
"""
import copy
from dataclasses import dataclass
import math

import torch
from utils.grpo import normalize_region_advantages
from utils.reward import prepare_reward_input, reward_value
from utils.sam_regions import RegionBatch, build_regions


def _validate_images(current, other, regions, policy_std):
    if not math.isfinite(policy_std) or policy_std <= 0:
        raise ValueError('policy_std must be finite and positive')
    if current.ndim != 4 or current.shape != other.shape or not current.is_floating_point():
        raise ValueError('Policy images must have matching floating BCHW shapes')
    if not torch.isfinite(current).all() or not torch.isfinite(other).all():
        raise ValueError('Nonfinite policy images')
    expected = (current.shape[0], regions.valid.shape[1], 1, *current.shape[-2:])
    if (regions.masks.shape != expected or regions.valid.shape != regions.weights.shape
            or not regions.valid.any(1).all()):
        raise ValueError('Invalid policy region shapes or empty group')


def _regional_sum(values, regions):
    # Sum, never mean: this is a joint density on the region's action dimensions.
    return (values[:, None] * regions.masks.detach()).sum((2, 3, 4))


def regional_log_ratios(current, old, actions, regions, policy_std=.1):
    """Log pi_current(z[M])/pi_old(z[M]), with fixed detached latent actions."""
    _validate_images(current, old, regions, policy_std)
    if actions.shape != current.shape or not torch.isfinite(actions).all():
        raise ValueError('Latent actions must be finite with matching image shapes')
    cur, prev, action = current.double(), old.detach().double(), actions.detach().double()
    # Difference of squares in factored form; exactly zero at current==old while
    # its derivative is (action-current)/sigma^2, rather than abs(0)'s zero.
    values = (cur-prev) * (2*action-cur-prev) / (2*policy_std**2)
    return _regional_sum(values, regions)


def regional_gaussian_kl(current, reference, regions, policy_std=.1):
    """Exact fixed-variance regional KL(current || reference), per region."""
    _validate_images(current, reference, regions, policy_std)
    values = (current.double()-reference.detach().double()).square() / (2*policy_std**2)
    return _regional_sum(values, regions)


def stochastic_grpo_loss(current, old, actions, advantages, regions, reference,
                         policy_std=.1, clip_epsilon=.2, beta=.1):
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError('Require finite 0<clip_epsilon<1 and beta>=0')
    if advantages.shape != regions.valid.shape or not torch.isfinite(advantages[regions.valid]).all():
        raise ValueError('Advantages must match valid regions and be finite')
    log_ratios = regional_log_ratios(current, old, actions, regions, policy_std)
    ratios = log_ratios.exp()
    if not torch.isfinite(ratios[regions.valid]).all():
        raise ValueError('Nonfinite regional likelihood ratio; reduce the policy update')
    a = advantages.detach().double().masked_fill(~regions.valid, 0)
    w = regions.weights.detach().double().masked_fill(~regions.valid, 0)
    clipped = ratios.clamp(1-clip_epsilon, 1+clip_epsilon)
    policy_loss = -(w * torch.minimum(ratios*a, clipped*a)).sum(1).mean()
    kl = (w * regional_gaussian_kl(current, reference, regions, policy_std)).sum(1).mean()
    valid_ratios = ratios[regions.valid]
    return policy_loss+beta*kl, {
        'policy_loss': policy_loss, 'kl': kl,
        'ratio_min': valid_ratios.min(), 'ratio_max': valid_ratios.max(),
        'clip_fraction': ((valid_ratios < 1-clip_epsilon) | (valid_ratios > 1+clip_epsilon)).double().mean(),
    }


@dataclass
class Rollout:
    old_logits: torch.Tensor
    reference_logits: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    full_image_rewards: torch.Tensor
    advantages: torch.Tensor
    regions: RegionBatch
    source_indices: torch.Tensor
    policy_std: float


@torch.no_grad()
def score_full_images(reward_model, fused, ir, vis, reward_size=384,
                      preprocess='processor', score_batch_size=1):
    if score_batch_size < 1:
        raise ValueError('score_batch_size must be positive')
    reward_model.eval().requires_grad_(False)
    values = []
    for start in range(0, fused.shape[0], score_batch_size):
        inputs = [prepare_reward_input(x[start:start+score_batch_size], reward_size, preprocess)
                  for x in (fused, ir, vis)]
        values.append(reward_value(reward_model(*inputs)))
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite full image reward')
    return result


@torch.no_grad()
def collect_rollout(model, reference, reward_model, ir, vis, segmenter=None,
                    mask_batches=None, reward_size=384, preprocess='processor',
                    score_batch_size=1, region_options=None, advantage_eps=1e-4,
                    policy_std=.1, samples_per_input=1):
    """Sample images, then compare SAM regions within each sampled image only."""
    if model is reference:
        raise ValueError('The initial reference must be a separate frozen policy')
    if (not isinstance(samples_per_input, int) or samples_per_input < 1
            or score_batch_size < 1 or not math.isfinite(policy_std) or policy_std <= 0):
        raise ValueError('Require positive sample counts, microbatch size and finite policy std')
    if ir.ndim != 4 or ir.shape != vis.shape or ir.shape[0] < 1:
        raise ValueError('IR and visible images must be matching nonempty BCHW batches')
    model.eval()
    reference.eval().requires_grad_(False)
    reward_model.eval().requires_grad_(False)
    indices = torch.arange(ir.shape[0], device=ir.device).repeat_interleave(samples_per_input)
    old = model(ir, vis, return_logits=True).detach().clone()
    ref = reference(ir, vis, return_logits=True).detach().clone()
    if old.shape != ir.shape or ref.shape != old.shape or not torch.isfinite(old).all() or not torch.isfinite(ref).all():
        raise ValueError('Policy logits must be finite and match source image shape')
    full_rewards = score_full_images(reward_model, old.sigmoid(), ir, vis, reward_size,
                                     preprocess, score_batch_size).index_select(0, indices)
    old = old.index_select(0, indices)
    ref = ref.index_select(0, indices)
    actions = (old + policy_std*torch.randn_like(old)).detach()
    sampled = actions.sigmoid()
    if mask_batches is None:
        if segmenter is None:
            raise ValueError('Provide online SAM or explicit precomputed semantic masks')
        mask_batches = segmenter.generate(sampled)
    else:
        if len(mask_batches) != ir.shape[0]:
            raise ValueError('Precomputed mask and input batch sizes differ')
        mask_batches = [mask_batches[int(index)] for index in indices.cpu()]
    if len(mask_batches) != sampled.shape[0]:
        raise ValueError('SAM must return one region group per sampled image')
    regions = build_regions(mask_batches, *sampled.shape[-2:], device=sampled.device,
                            **(region_options or {}))
    rewards = sampled.new_zeros(regions.valid.shape)
    for b, source in enumerate(indices.tolist()):
        valid_indices = regions.valid[b].nonzero(as_tuple=True)[0]
        for start in range(0, len(valid_indices), score_batch_size):
            selected = valid_indices[start:start+score_batch_size]
            masks = regions.masks[b, selected].to(sampled.dtype)
            inputs = [prepare_reward_input(x * masks, reward_size, preprocess)
                      for x in (sampled[b:b+1], ir[source:source+1], vis[source:source+1])]
            rewards[b, selected] = reward_value(reward_model(*inputs))
    if not torch.isfinite(rewards[regions.valid]).all():
        raise ValueError('Nonfinite semantic region reward')
    advantages = normalize_region_advantages(rewards, regions.valid, advantage_eps)
    return Rollout(old, ref, actions, rewards, full_rewards, advantages, regions, indices, policy_std)


def _weighted_kl(current, other, rollout):
    values = regional_gaussian_kl(current, other, rollout.regions, rollout.policy_std)
    return (values * rollout.regions.weights.detach().double()).sum(1).mean()


def update_policy(model, optimizer, ir, vis, rollout, clip_epsilon=.2, beta=.1,
                  max_grad_norm=1., target_kl=.02, max_backtracks=16):
    """Update using frozen actions; bound exact old-policy KL via full rollback.

    Backtracking changes only this step's learning rate. The scheduled optimizer
    rates are restored on return, and rejected Adam moments/steps are discarded.
    """
    if (not math.isfinite(max_grad_norm) or max_grad_norm <= 0
            or not math.isfinite(target_kl) or target_kl <= 0
            or not isinstance(max_backtracks, int) or max_backtracks < 0):
        raise ValueError('Invalid gradient bound, KL target or backtracking count')
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    current = model(ir, vis, return_logits=True).index_select(0, rollout.source_indices)
    loss, stats = stochastic_grpo_loss(current, rollout.old_logits, rollout.actions,
                                      rollout.advantages, rollout.regions, rollout.reference_logits,
                                      rollout.policy_std, clip_epsilon, beta)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite stochastic regional objective')
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('The current policy has no trainable parameters')
    policy_gradients = torch.autograd.grad(stats['policy_loss'], parameters, retain_graph=True, allow_unused=True)
    policy_grad_norm = math.sqrt(sum(float(g.detach().double().square().sum())
                                    for g in policy_gradients if g is not None))
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    metrics = {key: float(value.detach()) for key, value in stats.items()}
    metrics.update({'loss': float(loss.detach()), 'grad_norm': float(total_norm),
                    'policy_grad_norm': policy_grad_norm})
    # Release the forward/backward graph before trial steps and diagnostics.
    del policy_gradients, loss, stats, current
    model_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rates = [group['lr'] for group in optimizer.param_groups]
    accepted = False
    try:
        for attempt in range(max_backtracks+1):
            if attempt:
                model.load_state_dict(model_before)
                optimizer.load_state_dict(copy.deepcopy(optimizer_before))
            scale = .5**attempt
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate*scale
            optimizer.step()
            with torch.no_grad():
                post = model(ir, vis, return_logits=True).index_select(0, rollout.source_indices)
                try:
                    post_kl = float(_weighted_kl(post, rollout.old_logits, rollout))
                    _, post_stats = stochastic_grpo_loss(post, rollout.old_logits, rollout.actions,
                                                        rollout.advantages, rollout.regions,
                                                        rollout.reference_logits, rollout.policy_std,
                                                        clip_epsilon, beta)
                    finite = all(torch.isfinite(v).all() for v in post_stats.values())
                except ValueError:
                    finite, post_kl = False, math.inf
            if finite and post_kl <= target_kl:
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f'Unable to accept policy update within KL target after {max_backtracks+1} attempts '
                               f'(last old KL={post_kl}, target={target_kl}, finite={finite})')
    finally:
        if not accepted:
            model.load_state_dict(model_before)
            optimizer.load_state_dict(optimizer_before)
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate
    delta = math.sqrt(sum(float((v.detach().double()-model_before[k].double()).square().sum())
                          for k, v in model.state_dict().items() if v.is_floating_point()))
    valid, count = rollout.regions.valid, rollout.regions.valid.sum(1)
    means = (rollout.rewards*valid).sum(1)/count
    std = (((rollout.rewards-means[:, None]).square()*valid).sum(1)/count).sqrt()
    metrics.update({'update_accepted': True, 'backtracks': attempt,
                    'accepted_lr_scale': scale, 'accepted_lr': min(rates)*scale,
                    'post_old_kl': post_kl, 'post_reference_kl': float(post_stats['kl']),
                    'post_ratio_min': float(post_stats['ratio_min']),
                    'post_ratio_max': float(post_stats['ratio_max']),
                    'parameter_delta_norm': delta,
                    'reward_mean': float(means.mean()), 'reward_std': float(std.mean()),
                    'full_image_reward_mean': float(rollout.full_image_rewards.mean()),
                    'advantage_abs_mean': float((rollout.advantages.abs().sum(1)/count).mean()),
                    'zero_objective_gradient': float(total_norm) <= 1e-12,
                    'zero_policy_gradient': policy_grad_norm <= 1e-12,
                    'weight_decay_active': any(g.get('weight_decay', 0) != 0 for g in optimizer.param_groups)})
    return metrics
