"""SAM automatic regions and padded region batches for image-policy training."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


@dataclass
class RegionBatch:
    masks: torch.Tensor       # [B, K, 1, H, W], semantic regions only (unless explicit fallback)
    weights: torch.Tensor     # [B, R], sums to one per image
    valid: torch.Tensor       # [B, R], excludes padding
    sam_counts: torch.Tensor  # [B], selected semantic regions (excluding whole image)


def build_regions(mask_batches, height, width, device='cpu', top_k=3,
                  min_area_ratio=.01, max_area_ratio=.95, duplicate_iou=.95,
                  weighting='uniform', empty_policy='error'):
    if not mask_batches or height < 1 or width < 1 or top_k < 1:
        raise ValueError('A nonempty batch, positive dimensions and top_k are required')
    if not 0 <= min_area_ratio <= max_area_ratio <= 1 or not 0 < duplicate_iou <= 1:
        raise ValueError('Invalid region area or duplicate IoU thresholds')
    if weighting not in ('area', 'uniform'):
        raise ValueError('Invalid region weights')
    if empty_policy not in ('full-image', 'error'):
        raise ValueError('Unknown empty region policy')
    selected = []
    for batch_index, raw in enumerate(mask_batches):
        candidates = []
        for item in raw:
            array = item['segmentation'] if isinstance(item, dict) else item
            mask = torch.as_tensor(array, device=device)
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if tuple(mask.shape) != (height, width) or not torch.isfinite(mask).all():
                raise ValueError(f'Invalid region shape/values in image {batch_index}')
            mask = mask > .5
            area = int(mask.sum())
            if area and min_area_ratio <= area / (height * width) <= max_area_ratio:
                candidates.append((area, mask))
        keep = []
        for area, mask in sorted(candidates, key=lambda x: x[0], reverse=True):
            duplicate = any(float((mask & other).sum()) / float((mask | other).sum()) >= duplicate_iou
                            for _, other in keep)
            if not duplicate:
                keep.append((area, mask))
            if len(keep) >= top_k:
                break
        if not keep and empty_policy == 'error':
            raise ValueError(f'No valid semantic region in image {batch_index}')
        selected.append(keep)
    batch, count = len(selected), max(1, max(len(x) for x in selected))
    masks = torch.zeros(batch, count, 1, height, width, dtype=torch.bool, device=device)
    valid = torch.zeros(batch, count, dtype=torch.bool, device=device)
    weights = torch.zeros(batch, count, device=device)
    for b, keep in enumerate(selected):
        if keep:
            area_weights = torch.tensor([a if weighting == 'area' else 1 for a, _ in keep],
                                        dtype=torch.float32, device=device)
            area_weights = area_weights / area_weights.sum()
            for k, ((_, mask), weight) in enumerate(zip(keep, area_weights)):
                masks[b, k, 0], valid[b, k], weights[b, k] = mask, True, weight
        else:
            # Explicitly requested fallback; one region gives zero relative advantage.
            masks[b, 0], valid[b, 0], weights[b, 0] = True, True, 1.
    return RegionBatch(masks, weights, valid,
                       torch.tensor([len(x) for x in selected], device=device))


class SAMSegmenter:
    """Frozen SAM, loaded only when online semantic masks are requested."""
    def __init__(self, checkpoint, model_type='vit_b', device='cpu', points_per_side=12,
                 pred_iou_thresh=.88, stability_score_thresh=.92):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'SAM checkpoint not found: {checkpoint}')
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise ImportError('Install requirements-sam.txt to use online SAM masks') from exc
        self.model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        self.model.to(device).eval().requires_grad_(False)
        self.generator = SamAutomaticMaskGenerator(
            self.model, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh, box_nms_thresh=.7,
            crop_n_layers=0, output_mode='binary_mask')

    @torch.no_grad()
    def generate(self, images):
        if images.ndim != 4 or images.shape[1] not in (1, 3) or not torch.isfinite(images).all():
            raise ValueError('SAM requires finite grayscale/RGB BCHW images')
        results = []
        for image in images.detach().clamp(0, 1).cpu():
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            results.append(self.generator.generate(np.ascontiguousarray(rgb)))
        return results
