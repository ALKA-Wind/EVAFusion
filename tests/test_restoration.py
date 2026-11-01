"""Regression tests for the restored research pipeline (offline CPU fixtures)."""
import unittest
from pathlib import Path

import numpy as np
from test_support import scratch_directory

ROOT = Path(__file__).resolve().parents[1]
TEST_TMP = ROOT / 'tmp/restoration_tests'
TEST_TMP.mkdir(parents=True, exist_ok=True)


class CoreRestoration(unittest.TestCase):
    def test_h5_integer_dark_pixels_are_scaled_as_bytes(self):
        from utils.dataset import _image_tensor
        image = _image_tensor(np.ones((8, 8), dtype=np.uint8))
        self.assertAlmostEqual(image.max().item(), 1. / 255, places=7)

    def test_decoder_does_not_mix_images_across_the_batch(self):
        import torch
        from utils.fusion import FusionModel
        torch.set_num_threads(2)
        torch.manual_seed(1)
        model = FusionModel.from_checkpoint(ROOT / 'ckpt/DCEvo_fusion.pth').eval()
        ir, vi = torch.rand(2, 1, 16, 24), torch.rand(2, 1, 16, 24)
        with torch.no_grad():
            batched = model(ir, vi)
            singles = torch.cat([model(ir[i:i+1], vi[i:i+1]) for i in range(2)])
        torch.testing.assert_close(batched, singles, atol=1e-5, rtol=1e-5)

    def test_fusion_loss_backpropagates_on_cpu(self):
        import torch
        from utils.loss import Fusionloss
        fused = torch.rand(1, 1, 16, 16, requires_grad=True)
        loss, _, _ = Fusionloss()(torch.zeros_like(fused), torch.ones_like(fused), fused)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(fused.grad.abs().sum().item(), 0)

    def test_original_checkpoint_loads_without_dataparallel_on_cpu(self):
        import torch
        from utils.fusion import FusionModel
        model = FusionModel.from_checkpoint(ROOT / 'ckpt/DCEvo_fusion.pth', 'cpu')
        model.eval()
        with torch.no_grad():
            result = model(torch.rand(1, 1, 16, 24), torch.rand(1, 1, 16, 24))
        self.assertEqual(tuple(result.shape), (1, 1, 16, 24))
        self.assertTrue(torch.isfinite(result).all())

    def test_h5_grouped_samples_support_masks_and_spawn_pickle(self):
        import h5py
        import pickle
        from utils.dataset import H5Dataset
        with scratch_directory() as tmp:
            path = Path(tmp) / 'grouped.h5'
            with h5py.File(path, 'w') as f:
                g = f.create_group('sample')
                g['vis'] = np.full((8, 8), 255, dtype=np.uint8)
                g['ir'] = np.zeros((8, 8), dtype=np.uint8)
                for i in range(3):
                    g[f'mask_{i}'] = np.ones((8, 8), dtype=bool)
            data = H5Dataset(path)
            vis, ir, masks = data[0]
            self.assertEqual(tuple(vis.shape), (1, 8, 8))
            self.assertEqual(vis.max().item(), 1.0)
            self.assertEqual(ir.max().item(), 0.0)
            self.assertEqual(tuple(masks.shape), (3, 8, 8))
            reloaded = pickle.loads(pickle.dumps(data))
            self.assertEqual(reloaded[0][0].max().item(), 1.0)

    def test_h5_patch_groups_do_not_invent_semantic_masks(self):
        import h5py
        from utils.dataset import H5Dataset
        with scratch_directory() as tmp:
            path = Path(tmp) / 'patches.h5'
            with h5py.File(path, 'w') as f:
                f.create_group('vis_patchs')['0'] = np.ones((1, 8, 8), dtype=np.float32)
                f.create_group('ir_patchs')['0'] = np.zeros((1, 8, 8), dtype=np.float32)
            _, _, masks = H5Dataset(path)[0]
            self.assertEqual(tuple(masks.shape), (0, 8, 8))
            with self.assertRaisesRegex(ValueError, 'mask'):
                H5Dataset(path, require_masks=True)


if __name__ == '__main__':
    unittest.main()
