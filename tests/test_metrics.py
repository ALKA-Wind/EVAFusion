"""Small checks for metric arithmetic and the batch evaluation CLI."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
import warnings
import numpy as np
from PIL import Image
from test_support import scratch_directory

ROOT = Path(__file__).resolve().parents[1]


class MetricEvaluation(unittest.TestCase):
    def test_legacy_normalized_psnr_matches_original_formula_and_changes_no_other_metric(self):
        from Metric.metrics import FusionMetrics, fusion_metrics
        fused = np.full((8,8), 51., dtype=np.float32)
        ir = np.zeros((8,8), dtype=np.uint8)
        vi = np.full((8,8), 102, dtype=np.uint8)
        # Both normalized errors are 0.2; original code uses peak 255:
        # 20*log10(255/0.2) = 62.1102036954 dB (float32 Torch arithmetic).
        legacy = FusionMetrics(['PSNR'], psnr_mode='legacy-normalized-255').evaluate(fused, ir, vi)[0]
        self.assertAlmostEqual(legacy['PSNR'], 62.1102037, places=5)
        current = fusion_metrics(fused, ir, vi)
        actual = fusion_metrics(fused, ir, vi, psnr_mode='legacy-normalized-255')
        self.assertAlmostEqual(actual['PSNR'], 62.1102037, places=5)
        for name in ('CC', 'SSIM', 'Qabf'):
            self.assertEqual(actual[name], current[name])
        self.assertAlmostEqual(current['PSNR'], 13.97940009, places=6)

    def test_legacy_psnr_averages_mse_before_log_and_preserves_infinite_case(self):
        from Metric.metrics import FusionMetrics
        zero = np.zeros((8,8))
        one = np.full((8,8),255.)
        engine = FusionMetrics(['PSNR'], psnr_mode='legacy-normalized-255')
        self.assertTrue(np.isinf(engine.evaluate(zero, zero, zero)[0]['PSNR']))
        # MSEs are 0 and 1, average is .5. Averaging PSNR dB would give inf.
        self.assertAlmostEqual(engine.evaluate(zero, zero, one)[0]['PSNR'], 51.1411036, places=5)

    def test_cli_can_explicitly_reproduce_legacy_psnr_protocol(self):
        with scratch_directory() as tmp:
            root = Path(tmp)
            for name, value in (('ir',0), ('vi',102), ('fused',51)):
                (root/name).mkdir()
                Image.new('L',(8,8),value).save(root/name/'a.png')
            result = subprocess.run([sys.executable, '-m', 'Metric.evaluate',
                                     '--ir-dir', str(root/'ir'), '--vi-dir', str(root/'vi'),
                                     '--fused-dir', str(root/'fused'), '--metrics', 'PSNR',
                                     '--psnr-mode', 'legacy-normalized-255', '--output-dir', str(root/'out')],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root/'out/results.json').read_text())
            self.assertEqual(data['configuration']['psnr_mode'], 'legacy-normalized-255')
            actual = data['datasets']['custom']['summary']['PSNR']['mean']
            self.assertAlmostEqual(actual, 62.1102037, places=5)

    def test_core_metrics_use_float_and_handle_flat_images(self):
        from Metric.metrics import calculate_psnr, fusion_metrics
        black = np.zeros((8, 8), dtype=np.uint8)
        white = np.full((8, 8), 255, dtype=np.uint8)
        self.assertAlmostEqual(calculate_psnr(black, white), 0.)
        self.assertTrue(np.isinf(calculate_psnr(white, white)))
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            metrics = fusion_metrics(black, black, black)
        self.assertEqual(metrics['Qabf'], 0.)
        self.assertEqual(metrics['SSIM'], 1.)
        with self.subTest(metric='NIQE partial AGGD features'):
            from Metric.no_reference import _aggd
            alpha, left, right = _aggd(np.ones((8, 8), dtype=np.float32))
            self.assertEqual(alpha, .2)
            self.assertTrue(np.isnan(left))
            self.assertGreater(right, 0.)
        with self.subTest(metric='VIF input/reference order'):
            import torch
            import piq
            import cv2
            from Metric.metrics import FusionMetrics
            torch.set_num_threads(2)
            reference = np.random.default_rng(7).uniform(0, 255, (64, 64))
            distorted = cv2.GaussianBlur(reference, (5, 5), 1.2)
            x, y = [torch.from_numpy(a.copy()).float()[None, None] for a in (distorted, reference)]
            expected = 2 * float(piq.vif_p(x, y, data_range=255))
            actual = FusionMetrics(['VIF']).evaluate(distorted, reference, reference)[0]['VIF']
            self.assertAlmostEqual(actual, expected, places=6)

    def test_cli_pairs_stems_and_writes_per_image_and_summary(self):
        with scratch_directory() as tmp:
            root = Path(tmp)
            for name in ('ir', 'vi', 'fused'):
                (root / name).mkdir()
            image = np.arange(256, dtype=np.uint8).reshape(16, 16)
            Image.fromarray(image).save(root / 'ir/a.bmp')
            Image.fromarray(image).save(root / 'vi/a.bmp')
            Image.fromarray(image).save(root / 'fused/a.png')
            result = subprocess.run([sys.executable, '-m', 'Metric.evaluate',
                                     '--ir-dir', str(root / 'ir'), '--vi-dir', str(root / 'vi'),
                                     '--fused-dir', str(root / 'fused'), '--metrics', 'CC', 'PSNR', 'SSIM', 'Qabf',
                                     '--output-dir', str(root / 'out')],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / 'out/results.json').read_text())
            self.assertEqual(data['datasets']['custom']['count'], 1)
            self.assertEqual(data['datasets']['custom']['summary']['SSIM']['mean'], 1.)
            self.assertEqual(data['datasets']['custom']['summary']['PSNR']['positive_infinity_count'], 1)
            self.assertTrue((root / 'out/per_image.csv').exists())
            self.assertTrue((root / 'out/summary.csv').exists())

    def test_cli_reports_undefined_niqe_and_other_metrics_still_finish(self):
        with scratch_directory() as tmp:
            root = Path(tmp)
            Image.new('L', (16, 16), 42).save(root / 'small.png')
            result = subprocess.run([sys.executable, '-m', 'Metric.evaluate', '--fused-dir', str(root),
                                     '--metrics', 'NIQE', 'EN', '--output-dir', str(root / 'out')],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((root / 'out/results.json').read_text())
            summary = data['datasets']['custom']['summary']
            self.assertEqual(summary['NIQE']['valid_count'], 0)
            self.assertIsNone(summary['NIQE']['mean'])
            self.assertEqual(summary['EN']['valid_count'], 1)
            self.assertTrue(data['datasets']['custom']['errors'])


if __name__ == '__main__':
    unittest.main()
