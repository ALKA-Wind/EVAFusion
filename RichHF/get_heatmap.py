"""Artifact circles -> binary heatmaps, adapted from the original annotation tool."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def generate_heatmaps_from_json(json_dir, image_dir, save_dir_artifacts):
    output = Path(save_dir_artifacts)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    for label_path in sorted(Path(json_dir).glob('*.json')):
        record = json.loads(label_path.read_text(encoding='utf-8'))
        name = Path(record.get('imagePath', label_path.stem + '.png')).name
        with Image.open(Path(image_dir) / name) as image:
            width, height = image.size
        heatmap = np.zeros((height, width), dtype=np.uint8)
        for shape in record.get('shapes', []):
            if shape.get('shape_type') != 'circle' or len(shape.get('points', [])) < 2:
                continue
            (cx, cy), (ex, ey) = shape['points'][:2]
            radius = int(round(((ex-cx)**2 + (ey-cy)**2)**.5))
            cv2.circle(heatmap, (int(round(cx)), int(round(cy))), radius, 255, thickness=-1)
        # Empty annotations are valid negative examples and need a black heatmap.
        Image.fromarray(heatmap).save(output / name)
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print('Generated', generate_heatmaps_from_json(args.annotations, args.images, args.output), 'heatmaps')


if __name__ == '__main__':
    main()
