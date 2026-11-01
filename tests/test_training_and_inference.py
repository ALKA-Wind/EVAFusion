import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import ViTConfig, ViTModel
from test_support import scratch_directory

ROOT = Path(__file__).resolve().parents[1]


class TrainingAndInference(unittest.TestCase):
    def run_reward_training_fixture(self, root, dataparallel=False):
        from datasets import Dataset, DatasetDict
        from transformers import ViTImageProcessor
        vit_dir = root / 'vit'
        config = ViTConfig(image_size=32, patch_size=16, hidden_size=64,
                           num_hidden_layers=1, num_attention_heads=4, intermediate_size=128)
        ViTModel(config).save_pretrained(vit_dir)
        ViTImageProcessor(size={'height': 32, 'width': 32}).save_pretrained(vit_dir)
        rows = []
        for i in range(2):
            row = {key: Image.new('RGB', (32, 32), color=20)
                   for key in ('infrared', 'visible', 'fused')}
            row['artifact_heatmap'] = Image.new('L', (32, 32))
            row['scores'] = {key: 5 for key in ('Thermal Retention', 'Texture Preservation',
                                                'Artifacts', 'Sharpness', 'Overall Score')}
            row['image_id'] = str(i)
            rows.append(row)
        dataset = Dataset.from_list(rows)
        for split in ('train', 'dev'):
            DatasetDict({split: dataset}).save_to_disk(root / split)
        command = [sys.executable, str(ROOT / 'train_reward.py'), '--device', 'cpu',
                   '--vit_model', str(vit_dir), '--train_path', str(root / 'train'),
                   '--dev_path', str(root / 'dev'), '--num_epochs', '1', '--batch_size', '1',
                   '--log_dir', str(root / 'logs')]
        if dataparallel:
            from reward_model.model import RAHF
            reward = RAHF(vit_model=str(vit_dir), image_size=32)
            checkpoint = root / 'parallel.pt'
            torch.save({'module.' + key: value for key, value in reward.state_dict().items()}, checkpoint)
            command += ['--ckpt', str(checkpoint)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((root / 'logs').glob('*/model_best.pt'))), 1)

    def test_reward_training_cli_saves_best_even_when_correlation_is_zero(self):
        with scratch_directory() as tmp:
            self.run_reward_training_fixture(Path(tmp))

    def test_reward_training_cli_accepts_dataparallel_checkpoint_on_cpu(self):
        with scratch_directory() as tmp:
            self.run_reward_training_fixture(Path(tmp), dataparallel=True)



    def test_inference_cli_honors_checkpoint_and_output_paths(self):
        with scratch_directory() as tmp:
            tmp = Path(tmp)
            ir, vi = tmp / 'ir', tmp / 'vi'
            ir.mkdir(); vi.mkdir()
            Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(ir / 'a.png')
            Image.fromarray(np.flipud(np.arange(256, dtype=np.uint8).reshape(16, 16))).save(vi / 'a.png')
            output = tmp / 'nested/output'
            command = [sys.executable, str(ROOT / 'test_Fusion.py'), '--device', 'cpu',
                       '--checkpoint', str(ROOT / 'ckpt/DCEvo_fusion.pth'),
                       '--ir-dir', str(ir), '--vi-dir', str(vi), '--output-dir', str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / 'custom/a.png').exists())
            metrics = json.loads((output / 'metrics.json').read_text())
            self.assertEqual(metrics['custom']['count'], 1)
            self.assertEqual(set(metrics['custom']['mean']), {'CC', 'PSNR', 'SSIM', 'Qabf'})
            self.assertEqual(metrics['custom']['psnr_mode'], 'legacy-normalized-255')
            from Metric.metrics import fusion_metrics
            with Image.open(output / 'custom/a.png') as file:
                fused = np.array(file.convert('L'))
            a = np.arange(256, dtype=np.uint8).reshape(16,16)
            standard = fusion_metrics(fused, a, np.flipud(a))
            for name in ('CC', 'SSIM', 'Qabf'):
                self.assertAlmostEqual(metrics['custom']['mean'][name], standard[name], places=8)
            # Compare against the independent original Torch formula.
            f, ir_tensor, vi_tensor = [torch.from_numpy(x.copy()).float()/255.
                                      for x in (fused, a, np.flipud(a))]
            mse = .5*((f-ir_tensor)**2).mean()+.5*((f-vi_tensor)**2).mean()
            expected_psnr = float(20*torch.log10(255/torch.sqrt(mse)))
            self.assertAlmostEqual(metrics['custom']['mean']['PSNR'], expected_psnr, places=6)


if __name__ == '__main__':
    unittest.main()
