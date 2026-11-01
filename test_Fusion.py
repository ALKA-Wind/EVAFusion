"""Fuse paired IR/VIS images using restored DCEvo/EVAFusion checkpoints."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from Metric.metrics import PSNR_MODES, calculate_cc, calculate_psnr, calculate_ssim, fusion_metrics
from utils.fusion import FusionModel
from utils.img_read_save import image_read_cv2, img_save
from utils.paths import load_paths

__test__ = False


def test(pth_path=None, out_path='outputs/fusion', dataset_root=None,
         datasets=('M3FD', 'TNO', 'RoadScene'), device=None, limit=None,
         ir_dir=None, vi_dir=None, resize=None, psnr_mode='legacy-normalized-255'):
    paths = load_paths()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionModel.from_checkpoint(pth_path or paths['final_checkpoint'], device).eval()
    output = Path(out_path)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(dataset_root or paths['dataset_root'])
    pairs = {'custom': (Path(ir_dir), Path(vi_dir))} if ir_dir and vi_dir else {
        name: (root / name / 'ir', root / name / 'vi') for name in datasets}
    results = {}
    for name, (ir_folder, vi_folder) in pairs.items():
        files = sorted(p for p in ir_folder.iterdir()
                       if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
        files = files[:limit] if limit is not None else files
        if not files:
            raise ValueError(f'No input images in {ir_folder}')
        per_image = []
        for path in files:
            vis_path = vi_folder / path.name
            if not vis_path.exists():
                raise FileNotFoundError(f'Missing visible image paired with {path}: {vis_path}')
            ir, vis = image_read_cv2(path, 'GRAY'), image_read_cv2(vis_path, 'GRAY')
            if ir.shape != vis.shape:
                raise ValueError(f'Unregistered pair: {path.name}, {ir.shape} vs {vis.shape}')
            if resize:
                ir, vis = [cv2.resize(x, (resize[1], resize[0]), interpolation=cv2.INTER_AREA) for x in (ir, vis)]
            tensors = [torch.as_tensor(x.astype(np.float32) / 255.)[None, None].to(device) for x in (ir, vis)]
            with torch.no_grad():
                fused = model(*tensors)[0, 0].cpu().numpy() * 255.
            metrics = fusion_metrics(fused, ir, vis, psnr_mode=psnr_mode)
            per_image.append({'image': path.name, **metrics})
            img_save(fused, path.stem, output / name)
        mean = {key: float(np.mean([x[key] for x in per_image])) for key in ('CC', 'PSNR', 'SSIM', 'Qabf')}
        results[name] = {'count': len(per_image), 'psnr_mode': psnr_mode,
                         'mean': mean, 'per_image': per_image}
        print(name, mean, flush=True)
    def json_finite(value):
        if isinstance(value, dict):
            return {k: json_finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_finite(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    (output / 'metrics.json').write_text(json.dumps(json_finite(results), indent=2, allow_nan=False), encoding='utf-8')
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--dataset-root')
    parser.add_argument('--datasets', nargs='+', default=['M3FD', 'TNO', 'RoadScene'])
    parser.add_argument('--ir-dir')
    parser.add_argument('--vi-dir')
    parser.add_argument('--output-dir', default='outputs/fusion')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resize', nargs=2, type=int, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--psnr-mode', choices=PSNR_MODES, default='legacy-normalized-255',
                        help='Default reproduces original Metric_torch normalized-MSE/peak-255 protocol')
    args = parser.parse_args(argv)
    if bool(args.ir_dir) != bool(args.vi_dir):
        parser.error('--ir-dir and --vi-dir must be supplied together')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.resize and min(args.resize) < 7:
        parser.error('--resize dimensions must be at least 7 for SSIM')
    paths = load_paths(args.config)
    torch.set_num_threads(args.num_threads)
    test(args.checkpoint or paths['final_checkpoint'], args.output_dir,
         args.dataset_root or paths['dataset_root'], args.datasets, args.device,
         args.limit, args.ir_dir, args.vi_dir, args.resize, args.psnr_mode)


if __name__ == '__main__':
    main()
