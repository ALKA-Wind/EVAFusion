"""SAM + stochastic GRPO: same-image regional advantages and true policy ratios."""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import random
import sys
import warnings

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import H5Dataset
from utils.fusion import FusionModel
from utils.stochastic_grpo import collect_rollout, update_policy, score_full_images
from utils.paths import ROOT, load_paths
from utils.reward import load_reward_model
from utils.sam_regions import SAMSegmenter


def collate_images_and_masks(samples):
    vis, ir, masks = zip(*samples)
    return torch.stack(vis), torch.stack(ir), list(masks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--h5')
    parser.add_argument('--checkpoint')
    parser.add_argument('--reward-checkpoint')
    parser.add_argument('--vit-model')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mask-source', choices=('sam', 'h5'), default='sam')
    parser.add_argument('--sam-checkpoint')
    parser.add_argument('--sam-model', choices=('vit_b', 'vit_l', 'vit_h'), default='vit_b')
    parser.add_argument('--sam-points-per-side', type=int, default=12)
    parser.add_argument('--sam-pred-iou-thresh', type=float, default=.88)
    parser.add_argument('--sam-stability-thresh', type=float, default=.92)
    parser.add_argument('--max-regions', type=int, default=3)
    parser.add_argument('--min-area-ratio', type=float, default=.01)
    parser.add_argument('--max-area-ratio', type=float, default=.95)
    parser.add_argument('--mask-dedup-iou', type=float, default=.95)
    parser.add_argument('--region-weighting', choices=('area', 'uniform'), default='uniform')
    parser.add_argument('--empty-mask-policy', choices=('full-image', 'error'), default='error')
    parser.add_argument('--policy-std', type=float, default=.1, help='Fixed Gaussian std in decoder logit units')
    parser.add_argument('--samples-per-input', type=int, default=1,
                        help='Independent image samples; advantages still compare regions within EACH image')
    parser.add_argument('--advantage-eps', type=float, default=1e-4)
    parser.add_argument('--target-kl', type=float, default=.02, help='Maximum weighted regional KL to old policy after a step')
    parser.add_argument('--max-backtracks', type=int, default=16)
    parser.add_argument('--update-epochs', type=int, default=2)
    parser.add_argument('--clip-epsilon', type=float, default=.2)
    parser.add_argument('--beta-kl', type=float, default=.1)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--score-batch-size', type=int, default=1)
    parser.add_argument('--reward-preprocess', choices=('legacy', 'processor'), default='processor')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--max-steps', type=int, help='Limit rollout batches, each with update-epochs optimizer steps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    positive = (args.epochs, args.batch_size, args.update_epochs, args.score_batch_size,
                args.max_regions, args.num_threads, args.sam_points_per_side, args.samples_per_input)
    if min(positive) < 1 or args.num_workers < 0 or args.max_backtracks < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Positive counts and nonnegative num-workers are required')
    floats = (args.policy_std, args.advantage_eps, args.target_kl,
              args.clip_epsilon, args.beta_kl, args.max_grad_norm, args.lr, args.weight_decay,
              args.min_area_ratio, args.max_area_ratio, args.mask_dedup_iou,
              args.sam_pred_iou_thresh, args.sam_stability_thresh)
    if not all(math.isfinite(x) for x in floats):
        parser.error('All numeric settings must be finite')
    if args.policy_std <= 0 or args.advantage_eps <= 0 or args.target_kl <= 0:
        parser.error('Require positive policy std, advantage eps and target KL')
    if not 0 < args.clip_epsilon < 1 or args.beta_kl < 0:
        parser.error('Require 0<clip-epsilon<1 and beta-kl>=0')
    if args.max_grad_norm <= 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('Invalid optimizer settings')
    if not 0 <= args.min_area_ratio <= args.max_area_ratio <= 1 or not 0 < args.mask_dedup_iou <= 1:
        parser.error('Invalid region area/IoU thresholds')
    if not 0 <= args.sam_pred_iou_thresh <= 1 or not 0 <= args.sam_stability_thresh <= 1:
        parser.error('SAM thresholds must be in [0,1]')
    output = Path(args.output_dir or ROOT / 'outputs/sam_grpo' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}')
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    paths = load_paths(args.config)
    h5_path = args.h5 or paths['fusion_h5']
    checkpoint = args.checkpoint or paths['fusion_checkpoint']
    reward_checkpoint = args.reward_checkpoint or paths['reward_checkpoint']
    vit_model = args.vit_model or paths['vit_model']
    sam_checkpoint = args.sam_checkpoint or paths['sam_checkpoint']
    dataset = H5Dataset(h5_path, require_masks=args.mask_source == 'h5')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_images_and_masks)
    segmenter = (SAMSegmenter(sam_checkpoint, args.sam_model, args.device, args.sam_points_per_side,
                              args.sam_pred_iou_thresh, args.sam_stability_thresh)
                 if args.mask_source == 'sam' else None)
    model = FusionModel.from_checkpoint(checkpoint, args.device)
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    reward = load_reward_model(reward_checkpoint, vit_model, args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    region_options = {'top_k': args.max_regions, 'min_area_ratio': args.min_area_ratio,
                      'max_area_ratio': args.max_area_ratio, 'duplicate_iou': args.mask_dedup_iou,
                      'weighting': args.region_weighting,
                      'empty_policy': args.empty_mask_policy}
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {'h5': h5_path, 'checkpoint': checkpoint,
                           'reward_checkpoint': reward_checkpoint, 'vit_model': vit_model,
                           'sam_checkpoint': sam_checkpoint if segmenter else None,
                           'samples': len(dataset), 'method': 'sam_stochastic_regional_grpo',
                           'reference_policy': 'frozen_initial_checkpoint',
                           'paper': 'arXiv:2603.03871v1 section 3.3 equations 3-5',
                           'advantage_axis': 'semantic_regions', 'full_image_role': 'monitoring_only',
                           'advantage_group': 'SAM regions within one sampled fused image; never inputs or candidates',
                           'policy_distribution': 'z~Normal(decoder_logits,sigma^2*I); fused=sigmoid(z)',
                           'ratio_definition': 'exp(sum_region(log pi_current(z)-log pi_old(z)))',
                           'kl_definition': 'exact fixed-variance regional Gaussian KL(current||initial_reference), weighted region sum and batch mean',
                           'paper_alignment': {'equation_3': 'same-image semantic-region normalization preserved',
                                               'equation_5': 'clipped form preserved; displacement ratio replaced with action likelihood ratio'},
                           'reward_weights': {'Artifacts': .5, 'Texture_Preservation': 2., 'Thermal_Retention': 1., 'Sharpness': 2.},
                           'reward_artifact_factor': '1+fraction(heatmap>=0.5)',
                           'implementation_choices': ['scalar_reward', 'region_weights', 'Gaussian_logit_policy',
                                                      'policy_std', 'old_policy_refresh_schedule', 'KL_backtracking'],
                           'startup_limitations': 'Single-region or constant-region-reward groups have zero advantage; startup does not establish quality improvement.'}
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    step = 0
    warned_zero_gradient = False
    with (output / 'train.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(args.epochs):
            for vis, ir, masks in loader:
                vis, ir = vis.to(args.device), ir.to(args.device)
                rollout = collect_rollout(model, reference, reward, ir, vis,
                                          segmenter=segmenter, mask_batches=masks if args.mask_source == 'h5' else None,
                                          reward_size=reward.vit.config.image_size, preprocess=args.reward_preprocess,
                                          score_batch_size=args.score_batch_size, region_options=region_options,
                                          advantage_eps=args.advantage_eps, policy_std=args.policy_std,
                                          samples_per_input=args.samples_per_input)
                step += 1
                for update in range(args.update_epochs):
                    metrics = update_policy(model, optimizer, ir, vis, rollout,
                                            clip_epsilon=args.clip_epsilon, beta=args.beta_kl,
                                            max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
                                            max_backtracks=args.max_backtracks)
                    if metrics['zero_policy_gradient'] and not warned_zero_gradient:
                        warnings.warn('Zero regional policy gradient: check single-region groups or equal regional rewards. '
                                      'Weight decay, optimizer momentum and KL changes alone do not establish reward-driven startup.',
                                      RuntimeWarning)
                        warned_zero_gradient = True
                    with torch.no_grad():
                        deterministic_after = model(ir, vis)
                        after_reward = score_full_images(reward, deterministic_after, ir, vis,
                                                         reward.vit.config.image_size,
                                                         args.reward_preprocess, args.score_batch_size)
                    record = {'epoch': epoch+1, 'rollout_step': step, 'update': update+1,
                              'semantic_regions': rollout.regions.sam_counts.tolist(),
                              'single_region_images': int((rollout.regions.valid.sum(1) == 1).sum()),
                              'fallback_images': int((rollout.regions.sam_counts == 0).sum()),
                              'source_indices': rollout.source_indices.tolist(),
                              'deterministic_full_image_reward_before': float(rollout.full_image_rewards.mean()),
                              'deterministic_full_image_reward_after': float(after_reward.mean()),
                              'startup_verified': (not metrics['weight_decay_active']
                                                   and not metrics['zero_policy_gradient']
                                                   and metrics['parameter_delta_norm'] > 0
                                                   and bool((rollout.regions.sam_counts >= 2).all())), **metrics}
                    log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                    print(record, flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
            scheduler.step()
            model.save_checkpoint(output / f'epoch_{epoch+1}.pth')
            if args.max_steps and step >= args.max_steps:
                break
    model.save_checkpoint(output / 'final.pth')
    print(f'Saved fusion model: {output / "final.pth"}')


if __name__ == '__main__':
    main()
