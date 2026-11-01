"""Literal regional objective from EVAFusion equations (3)-(5).

The paper leaves norm, scalar rewards, weights and policy KL unspecified;
our explicit computational conventions are listed in docs/sam_grpo.md.
"""
from dataclasses import dataclass
import math
import torch
from utils.sam_regions import RegionBatch, build_regions
from utils.reward import prepare_reward_input, reward_value


def normalize_region_advantages(rewards, valid=None, eps=1e-4):
    """Equation (3): normalize across valid semantic regions of each image."""
    if rewards.ndim != 2 or min(rewards.shape) < 1 or not rewards.is_floating_point():
        raise ValueError('Rewards must be floating [batch, regions] tensors')
    if not math.isfinite(eps) or eps < 0:
        raise ValueError('Advantage epsilon must be finite and nonnegative')
    rewards = rewards.detach()
    if valid is None:
        valid = torch.ones_like(rewards, dtype=torch.bool)
    if valid.shape != rewards.shape or valid.dtype != torch.bool:
        raise ValueError('Valid regions must be a boolean tensor matching rewards')
    count = valid.sum(dim=1, keepdim=True)
    if (count == 0).any() or not torch.isfinite(rewards[valid]).all():
        raise ValueError('Each image needs finite rewards for at least one valid region')
    safe = rewards.masked_fill(~valid, 0)
    mean = safe.sum(dim=1, keepdim=True) / count
    centered = (safe - mean).masked_fill(~valid, 0)
    std = (centered.square().sum(dim=1, keepdim=True) / count).sqrt()
    return centered / (std + eps).clamp_min(torch.finfo(rewards.dtype).eps)


def region_change_ratios(current, old, regions, alpha=1., denominator_eps=1e-6):
    """Equation (5): 1 + alpha * ||(F-F_old) M_k||_1 / ||F_old M_k||_1."""
    if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(denominator_eps) or denominator_eps <= 0:
        raise ValueError('Require finite alpha>=0 and denominator_eps>0')
    if current.ndim != 4 or current.shape != old.shape or not current.is_floating_point():
        raise ValueError('Current and old fused images must have matching BCHW shapes')
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError('Nonfinite fused images')
    expected = (current.shape[0], regions.valid.shape[1], 1, *current.shape[-2:])
    if regions.masks.shape != expected or regions.valid.shape != regions.weights.shape:
        raise ValueError('Region and image shapes differ')
    mask = regions.masks.detach().to(current.dtype)
    old = old.detach()
    delta = ((current - old).abs()[:, None] * mask).sum(dim=(2, 3, 4))
    denominator = (old.abs()[:, None] * mask).sum(dim=(2, 3, 4)).clamp_min(denominator_eps)
    return 1 + alpha * delta / denominator


def categorical_image_kl(current, reference, temperature=1.):
    """Explicit KL convention: D(softmax(F/T) || softmax(F_ref/T)).

    Softmax is over all C*H*W pixels per image; reduction is batch mean.
    This distribution is a computational choice, not specified by the paper.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('KL temperature must be finite and positive')
    if current.ndim != 4 or current.shape != reference.shape:
        raise ValueError('Current and reference images must have matching BCHW shapes')
    reference = reference.detach()
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite images in KL')
    log_p = (current.flatten(1) / temperature).log_softmax(dim=1)
    log_q = (reference.flatten(1) / temperature).log_softmax(dim=1)
    per_image = (log_p.exp() * (log_p - log_q)).sum(dim=1)
    # At exactly identical images the analytical KL and its derivative are zero.
    # Return that exact result, avoiding spurious softmax roundoff gradients.
    identical = (current.detach() == reference).flatten(1).all(dim=1)
    per_image = torch.where(identical, (current - reference).flatten(1).sum(1) * 0, per_image)
    return per_image.mean()


def regional_grpo_loss(current, old, advantages, regions, reference,
                       alpha=1., clip_epsilon=.2, beta=.1,
                       denominator_eps=1e-6, kl_temperature=1.):
    """Minimize negative equation (5) plus beta times equation (4)'s KL."""
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1 or not math.isfinite(beta) or beta < 0:
        raise ValueError('Require finite 0<clip_epsilon<1 and beta>=0')
    if advantages.shape != regions.valid.shape or not torch.isfinite(advantages[regions.valid]).all():
        raise ValueError('Advantages must match valid regions and be finite')
    ratio = region_change_ratios(current, old, regions, alpha, denominator_eps)
    advantage = advantages.detach().masked_fill(~regions.valid, 0)
    weights = regions.weights.detach().to(current.dtype).masked_fill(~regions.valid, 0)
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -(weights * torch.minimum(ratio * advantage, clipped * advantage)).sum(1).mean()
    kl = categorical_image_kl(current, reference, kl_temperature)
    loss = policy_loss + beta * kl
    valid_ratios = ratio[regions.valid]
    return loss, {'policy_loss': policy_loss.detach(), 'kl': kl.detach(),
                  'clip_fraction': ((valid_ratios < 1 - clip_epsilon) | (valid_ratios > 1 + clip_epsilon)).float().mean(),
                  'ratio_min': valid_ratios.min().detach(), 'ratio_max': valid_ratios.max().detach()}


@dataclass
class Rollout:
    old_fused: torch.Tensor
    reference_fused: torch.Tensor
    rewards: torch.Tensor             # [B, K], semantic regions only
    full_image_rewards: torch.Tensor  # [B], monitoring only
    advantages: torch.Tensor          # [B, K], detached
    regions: RegionBatch


@torch.no_grad()
def collect_rollout(model, reference, reward_model, ir, vis,
                    segmenter=None, mask_batches=None, reward_size=384,
                    preprocess='legacy', score_batch_size=1, region_options=None,
                    advantage_eps=1e-4):
    """Segment and score one deterministic fusion per input, with no candidates."""
    if score_batch_size < 1:
        raise ValueError('score_batch_size must be positive')
    if model is reference:
        raise ValueError('The reference must be a separate frozen model')
    model.eval()
    reference.eval().requires_grad_(False)
    reward_model.eval().requires_grad_(False)
    # clone() also protects snapshots if a policy returns a parameter view.
    old_fused = model(ir, vis).detach().clone()
    reference_fused = reference(ir, vis).detach().clone()
    if mask_batches is None:
        if segmenter is None:
            raise ValueError('Provide online SAM or explicit precomputed masks')
        mask_batches = segmenter.generate(old_fused)
    if len(mask_batches) != old_fused.shape[0]:
        raise ValueError('Mask and image batch sizes differ')
    regions = build_regions(mask_batches, *old_fused.shape[-2:], device=old_fused.device,
                            **(region_options or {}))
    rewards = old_fused.new_zeros(regions.valid.shape)
    full_rewards = old_fused.new_zeros(old_fused.shape[0])
    for b in range(old_fused.shape[0]):
        indices = regions.valid[b].nonzero(as_tuple=True)[0]
        # The full image is scored separately, outside the K-region advantage group.
        masks = torch.cat((regions.masks[b, indices], torch.ones_like(regions.masks[b, :1])), dim=0)
        values = old_fused.new_empty(masks.shape[0])
        for start in range(0, len(masks), score_batch_size):
            end = min(start + score_batch_size, len(masks))
            mask = masks[start:end].to(old_fused.dtype)
            inputs = [prepare_reward_input(x[b:b+1] * mask, reward_size, preprocess)
                      for x in (old_fused, ir, vis)]
            values[start:end] = reward_value(reward_model(*inputs))
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite regional or full-image reward')
        rewards[b, indices] = values[:-1]
        full_rewards[b] = values[-1]
    advantages = normalize_region_advantages(rewards, regions.valid, advantage_eps)
    return Rollout(old_fused, reference_fused, rewards, full_rewards, advantages, regions)


def update_policy(model, optimizer, ir, vis, rollout, alpha=1., clip_epsilon=.2,
                  beta=.1, denominator_eps=1e-6, kl_temperature=1., max_grad_norm=1.):
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('max_grad_norm must be finite and positive')
    # eval() disables stochastic layers and buffer drift; parameter gradients remain enabled.
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    current = model(ir, vis)
    policy_at_old = torch.equal(current.detach(), rollout.old_fused)
    loss, metrics = regional_grpo_loss(current, rollout.old_fused, rollout.advantages,
                                      rollout.regions, rollout.reference_fused, alpha,
                                      clip_epsilon, beta, denominator_eps, kl_temperature)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite regional objective')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    valid = rollout.regions.valid
    count = valid.sum(1)
    mean_reward = (rollout.rewards * valid).sum(1) / count
    reward_std = (((rollout.rewards - mean_reward[:, None]).square() * valid).sum(1) / count).sqrt()
    metrics.update({'loss': loss.detach(), 'grad_norm': norm.detach(),
                    'reward_mean': mean_reward.mean(), 'reward_std': reward_std.mean(),
                    'full_image_reward_mean': rollout.full_image_rewards.mean(),
                    'advantage_abs_mean': (rollout.advantages.abs().sum(1) / count).mean()})
    result = {key: float(value) for key, value in metrics.items()}
    result.update({'zero_objective_gradient': float(norm) <= 1e-12,
                   'policy_at_old': policy_at_old,
                   'weight_decay_active': any(group.get('weight_decay', 0) != 0 for group in optimizer.param_groups)})
    return result
