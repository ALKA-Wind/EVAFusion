"""Reward-model evaluation and per-sample score/heatmap export."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_from_disk
from RichHF.train import HuggingFaceDataset, evaluate_model
from utils.paths import load_paths
from utils.reward import load_reward_model


def infer(model, dataloader, device, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            result = model(tensors['fused'], tensors['ir'], tensors['vi'])
            for i in range(len(batch['fused'])):
                index = len(records)
                row = {'index': index, 'image_id': batch.get('image_id', [None] * len(batch['fused']))[i]}
                for key, score in result['scores'].items():
                    row[key] = float(score[i])
                    row['gt_' + key] = float(batch[key][i])
                records.append(row)
                for name, heatmap in result['heatmaps'].items():
                    image = (heatmap[i].cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(image).save(output / f'{index:05d}_{name}.png')
    (output / 'scores.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', '--ckpt', dest='checkpoint')
    parser.add_argument('--vit-model', '--vit_model', dest='vit_model')
    parser.add_argument('--dataset-path')
    parser.add_argument('--split', choices=('train', 'dev', 'test'), default='test')
    parser.add_argument('--output-dir', default='outputs/reward')
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--best', action='store_true')
    args = parser.parse_args(argv)
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error('batch-size and limit must be positive')
    paths = load_paths(args.config)
    torch.set_num_threads(args.num_threads)
    vit_model = args.vit_model or paths['vit_model']
    checkpoint = args.checkpoint or (str(Path(args.log_dir) / ('model_best.pt' if args.best else 'model.pt'))
                                     if args.log_dir else paths['reward_checkpoint'])
    model = load_reward_model(checkpoint, vit_model, args.device)
    dataset = load_from_disk(args.dataset_path or paths[args.split + '_data'])
    if isinstance(dataset, DatasetDict):
        dataset = dataset[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    data = HuggingFaceDataset(dataset, vit_model, image_size=model.feature_size * model.vit.config.patch_size)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.num_workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.eval:
        metrics = evaluate_model(model, loader, torch.nn.MSELoss(), args.device, phase=args.split)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
        print(metrics)
    if args.infer or not args.eval:
        records = infer(model, loader, args.device, output)
        print(f'Exported {len(records)} samples to {output}')


if __name__ == '__main__':
    main()
