"""Frozen reward scoring with differentiable image preprocessing."""
import torch
import torch.nn.functional as F
from reward_model.model import RAHF


def prepare_reward_input(image, size=384, mode='legacy'):
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    if image.shape[1] != 3:
        raise ValueError('Reward model expects grayscale or RGB tensors')
    shape = (size, size) if isinstance(size, int) else size
    image = F.interpolate(image, size=shape, mode='bilinear', align_corners=False)
    if mode == 'processor':
        # Original local preprocessor_config.json: mean=std=[0.5, 0.5, 0.5].
        return (image - .5) / .5
    if mode != 'legacy':
        raise ValueError(f'Unknown preprocessing mode: {mode}')
    return image


def compute_white_ratio_batch(heatmaps, threshold=.5):
    return torch.stack([(hm >= threshold).float().flatten(1).mean(1)
                        for hm in heatmaps.values()]).mean(0)


def reward_value(result, weights=(.5, 2., 1., 2.)):
    scores = result['scores']
    artifact, texture, thermal, sharp = weights
    ratio = compute_white_ratio_batch(result['heatmaps'])
    return (artifact * (1 + ratio) * scores['Artifacts']
            + texture * scores['Texture_Preservation']
            + thermal * scores['Thermal_Retention'] + sharp * scores['Sharpness'])


def load_reward_model(checkpoint, vit_model, device='cpu'):
    # Detect the actual head structure in the state dict, rather than trusting old args files.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    multi_heads = any(k.startswith('score_predictor.Thermal_Retention.') for k in state)
    from transformers import ViTConfig
    config = ViTConfig.from_pretrained(vit_model)
    model = RAHF(vit_model=vit_model, multi_heads=multi_heads,
                 image_size=config.image_size, patch_size=config.patch_size)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False)
