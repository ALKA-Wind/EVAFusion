"""Project-local checkpoints and configurable external experiment resources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_paths(config=None):
    overrides = json.loads(Path(config).read_text(encoding='utf-8')) if config else {}
    source = Path(overrides.pop('source_root', ROOT.parent / 'RLHF-Fusion')).expanduser()
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    paths = {
        'vit_model': source / 'RichHF/vit-large-patch16-384',
        'train_data': source / 'RichHF/data/train',
        'dev_data': source / 'RichHF/data/dev',
        'test_data': source / 'RichHF/data/test',
        'reward_checkpoint': ROOT / 'reward_model/20250503_232931/model_best.pt',
        'fusion_h5': source / 'DCEvo-main/data/MSRS_train_imgsize_128_stride_200.h5',
        'dataset_root': source / 'DCEvo-main/datasets',
        'fusion_checkpoint': ROOT / 'ckpt/DCEvo_fusion.pth',
        'sam_checkpoint': ROOT / 'ckpt/sam_vit_b_01ec64.pth',
        'final_checkpoint': ROOT / 'models/DCEvo/DCEvo_epoch10_2025-07-25_07-47-21.pth',
    }
    for key, value in overrides.items():
        if key not in paths:
            raise ValueError(f'Unknown resource key: {key}')
        path = Path(value).expanduser()
        paths[key] = path if path.is_absolute() else ROOT / path
    return {k: str(v) for k, v in paths.items()}
