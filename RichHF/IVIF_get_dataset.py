"""Save the original Annotation/ir/vi/fused/heatmap layout as a HF DatasetDict."""
import argparse
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Image, Value


def build_dataset(dataset_dir):
    root = Path(dataset_dir)
    records = []
    for path in sorted((root / 'Annotation').glob('*.json')):
        annotation = json.loads(path.read_text(encoding='utf-8'))
        name = path.stem + '.png'
        row = {'image_id': path.stem, 'scores': annotation['scores']}
        for key, folder in [('infrared', 'ir'), ('visible', 'vi'), ('fused', 'fused'),
                            ('artifact_heatmap', 'heatmap/Artifacts')]:
            image = root / folder / name
            if not image.exists():
                raise FileNotFoundError(image)
            row[key] = str(image.resolve())
        records.append(row)
    if not records:
        raise ValueError(f'No annotations in {root / "Annotation"}')
    features = Features({key: Image() for key in ('infrared', 'visible', 'fused', 'artifact_heatmap')} |
                        {'image_id': Value('string'), 'scores': {
                            'Thermal Retention': Value('int64'), 'Texture Preservation': Value('int64'),
                            'Artifacts': Value('int64'), 'Sharpness': Value('int64'), 'Overall Score': Value('float64')}})
    return Dataset.from_list(records, features=features)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--split', choices=('train', 'dev', 'test'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new directory')
    dataset = build_dataset(args.dataset_dir)
    DatasetDict({args.split: dataset}).save_to_disk(args.output)
    print(f'Saved {len(dataset)} {args.split} samples to {args.output}')


if __name__ == '__main__':
    main()
