import json
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from test_support import scratch_directory


class Preparation(unittest.TestCase):
    def test_heatmap_writes_empty_and_circle_annotations(self):
        from RichHF.get_heatmap import generate_heatmaps_from_json
        with scratch_directory() as tmp:
            root = Path(tmp)
            labels, images = root / 'labels', root / 'images'
            labels.mkdir(); images.mkdir()
            for name, shapes in [('a', []), ('b', [{'shape_type': 'circle', 'points': [[4, 4], [6, 4]]}])]:
                Image.new('RGB', (16, 16)).save(images / (name + '.png'))
                (labels / (name + '.json')).write_text(json.dumps({'imagePath': name + '.png', 'shapes': shapes}))
            generate_heatmaps_from_json(labels, images, root / 'heatmaps')
            self.assertEqual(np.asarray(Image.open(root / 'heatmaps/a.png')).sum(), 0)
            heatmap = np.asarray(Image.open(root / 'heatmaps/b.png'))
            self.assertEqual(heatmap[4, 4], 255)
            self.assertEqual(heatmap[0, 0], 0)

    def test_h5_preparation_preserves_normalized_patch_values(self):
        from prepare_fusion_data import create_h5
        from utils.dataset import H5Dataset
        with scratch_directory() as tmp:
            root = Path(tmp)
            for folder, color in [('ir', 0), ('vi', 255)]:
                (root / folder).mkdir()
                Image.new('L', (16, 16), color).save(root / folder / 'a.png')
            create_h5(root / 'ir', root / 'vi', root / 'patches.h5', patch_size=8, stride=8)
            dataset = H5Dataset(root / 'patches.h5')
            self.assertEqual(len(dataset), 4)
            self.assertEqual(dataset[0][0].max().item(), 1.)
            self.assertEqual(dataset[0][1].max().item(), 0.)


if __name__ == '__main__':
    unittest.main()
