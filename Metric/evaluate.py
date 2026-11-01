"""Evaluate saved fusion images; default: CC, PSNR, Qabf, SSIM, NIQE, BRISQUE."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from Metric.metrics import FusionMetrics, PAPER_METRICS, SUPPORTED_METRICS, FUSED_ONLY, LOWER_BETTER, PSNR_MODES
from utils.img_read_save import image_read_cv2
from utils.paths import ROOT, load_paths

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _files(directory):
    result = {}
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in result:
            raise ValueError(f'Duplicate image stem: {result[key]} and {path}')
        result[key] = path
    if not result:
        raise ValueError(f'No images found in {directory}')
    return result


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    valid = array[~np.isnan(array)]
    finite = valid[np.isfinite(valid)]
    if not len(valid):
        mean = None
    elif np.isposinf(valid).any():
        mean = float('inf')
    elif np.isneginf(valid).any():
        mean = float('-inf')
    else:
        mean = float(finite.mean())
    return {'mean': mean, 'std': float(finite.std()) if len(finite) == len(valid) and len(valid) else None,
            'valid_count': len(valid), 'finite_count': len(finite), 'invalid_count': int(np.isnan(array).sum()),
            'positive_infinity_count': int(np.isposinf(array).sum()),
            'negative_infinity_count': int(np.isneginf(array).sum())}


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def evaluate_directory(engine, fused_dir, ir_dir=None, vi_dir=None, allow_subset=False, limit=None):
    fused = _files(fused_dir)
    references = ir_dir is not None
    missing = {}
    if references:
        ir, vi = _files(ir_dir), _files(vi_dir)
        missing = {'without_fused': sorted((set(ir) | set(vi)) - set(fused)),
                   'without_ir': sorted((set(fused) | set(vi)) - set(ir)),
                   'without_vi': sorted((set(fused) | set(ir)) - set(vi))}
        if any(missing.values()) and not allow_subset:
            raise ValueError(f'Image stems do not match; use --allow-subset for an explicit intersection: '
                             f'{ {k: len(v) for k, v in missing.items()} }')
        keys = sorted(set(fused) & set(ir) & set(vi))
    else:
        keys = sorted(fused)
    if limit is not None:
        keys = keys[:limit]
    if not keys:
        raise ValueError('No matched image triplets')
    records, errors = [], []
    for index, key in enumerate(keys, start=1):
        try:
            f = image_read_cv2(fused[key], 'GRAY')
            a, b = (image_read_cv2(ir[key], 'GRAY'), image_read_cv2(vi[key], 'GRAY')) if references else (None, None)
            values, failures = engine.evaluate(f, a, b, on_error='record')
        except (ValueError, OSError) as exc:
            values, failures = {name: float('nan') for name in engine.names}, {'image': str(exc)}
        records.append({'image': fused[key].name, **values})
        errors.extend({'image': fused[key].name, 'metric': name, 'reason': reason} for name, reason in failures.items())
        print(f'[{index}/{len(keys)}] {fused[key].name}', flush=True)
    return {'count': len(records), 'per_image': records, 'errors': errors, 'excluded_stems': missing,
            'summary': {name: summarize([record[name] for record in records]) for name in engine.names}}


def _write_csv(path, fields, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir-dir')
    parser.add_argument('--vi-dir')
    parser.add_argument('--fused-dir', help='A single fusion-image directory')
    parser.add_argument('--dataset-root', help='Dataset root containing DATASET/ir and DATASET/vi')
    parser.add_argument('--fused-root', help='Fusion result root containing DATASET directories')
    parser.add_argument('--datasets', nargs='+', default=['M3FD', 'TNO', 'RoadScene'])
    parser.add_argument('--config')
    parser.add_argument('--metrics', nargs='+', choices=SUPPORTED_METRICS, default=list(PAPER_METRICS))
    parser.add_argument('--method', default='EVAFusion')
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cpu', help='NIQE always runs on CPU; torch metrics may use CUDA')
    parser.add_argument('--assets-dir')
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--psnr-mode', choices=PSNR_MODES, default='mean-db')
    parser.add_argument('--allow-subset', action='store_true', help='Evaluate only explicit stem intersection and report exclusions')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--strict', action='store_true', help='Return nonzero if any image/metric is undefined; still write reports')
    args = parser.parse_args(argv)
    if args.num_threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error('num-threads and limit must be positive')
    if bool(args.ir_dir) != bool(args.vi_dir):
        parser.error('--ir-dir and --vi-dir must be supplied together')
    if args.fused_dir and args.fused_root:
        parser.error('Choose --fused-dir or --fused-root')
    if args.fused_dir:
        if args.ir_dir is None and set(args.metrics) - FUSED_ONLY:
            parser.error('The selected metrics require --ir-dir and --vi-dir')
        directories = {'custom': (args.fused_dir, args.ir_dir, args.vi_dir)}
    else:
        if not args.fused_root or args.ir_dir or args.vi_dir:
            parser.error('Supply --fused-root for dataset mode, or --fused-dir for custom mode')
        root = Path(args.dataset_root or load_paths(args.config)['dataset_root'])
        directories = {name: (Path(args.fused_root) / name, root / name / 'ir', root / name / 'vi')
                       for name in args.datasets}
    output = Path(args.output_dir or ROOT / 'outputs/metrics' / datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output directory is not empty: {output}')
    if set(args.metrics) & {'NIQE', 'BRISQUE', 'VIF', 'MS_SSIM'}:
        import torch
        torch.set_num_threads(args.num_threads)
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            parser.error('CUDA was requested but is unavailable')
    try:
        engine = FusionMetrics(args.metrics, args.device, args.assets_dir, args.psnr_mode)
        datasets = {name: evaluate_directory(engine, f, a, b, args.allow_subset, args.limit)
                    for name, (f, a, b) in directories.items()}
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    result = {'method': args.method, 'configuration': vars(args) | {'metrics': list(engine.names),
               'channel': 'OpenCV RGB-to-gray, 8-bit', 'resize': 'none',
               'inputs': {name: {'fused_dir': str(Path(f).resolve()),
                                 'ir_dir': str(Path(a).resolve()) if a else None,
                                 'vi_dir': str(Path(b).resolve()) if b else None}
                          for name, (f, a, b) in directories.items()}}, 'datasets': datasets}
    (output / 'results.json').write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding='utf-8')
    rows, summary = [], []
    for dataset, data in datasets.items():
        rows.extend({'dataset': dataset, 'method': args.method, **_json_safe(row)} for row in data['per_image'])
        for name, values in data['summary'].items():
            summary.append({'dataset': dataset, 'method': args.method, 'metric': name,
                            'direction': 'lower' if name in LOWER_BETTER else 'higher', **_json_safe(values)})
        print(dataset, {k: v['mean'] for k, v in data['summary'].items()})
        if data['errors']:
            print(f'{dataset}: {len(data["errors"])} undefined/image errors; see results.json', file=sys.stderr)
    _write_csv(output / 'per_image.csv', ['dataset', 'method', 'image', *engine.names], rows)
    _write_csv(output / 'summary.csv', ['dataset', 'method', 'metric', 'direction', 'mean', 'std',
               'valid_count', 'finite_count', 'invalid_count', 'positive_infinity_count', 'negative_infinity_count'], summary)
    print(f'Saved evaluation to {output}')
    return int(args.strict and any(data['errors'] for data in datasets.values()))


if __name__ == '__main__':
    sys.exit(main())
