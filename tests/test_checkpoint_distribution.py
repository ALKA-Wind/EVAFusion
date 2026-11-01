"""Lossless reward-checkpoint distribution through bounded-size LFS parts."""
import hashlib
import json
from pathlib import Path
import unittest

from test_support import scratch_directory


class CheckpointDistribution(unittest.TestCase):
    def fixture(self, root):
        pieces = [b'checkpoint\x00\xff', b'weights\x01\x02']
        entries = []
        for index, data in enumerate(pieces, 1):
            name = f'model_best.pt.part{index:02d}'
            (root/name).write_bytes(data)
            entries.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        combined = b''.join(pieces)
        manifest = {'file':'model_best.pt','bytes':len(combined),
                    'sha256':hashlib.sha256(combined).hexdigest(),'parts':entries}
        (root/'checkpoint_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return combined

    def test_reassembles_exact_bytes_and_accepts_existing_verified_checkpoint(self):
        from reward_model.restore_checkpoint import restore_checkpoint
        with scratch_directory() as tmp:
            root=Path(tmp); expected=self.fixture(root)
            path=restore_checkpoint(root)
            self.assertEqual(path.read_bytes(), expected)
            self.assertEqual(restore_checkpoint(root), path)
            self.assertFalse(list(root.glob('*.assembling')))

    def test_corrupt_part_does_not_publish_or_leave_partial_checkpoint(self):
        from reward_model.restore_checkpoint import restore_checkpoint
        with scratch_directory() as tmp:
            root=Path(tmp); self.fixture(root)
            path=root/'model_best.pt.part02'; path.write_bytes(b'x'*path.stat().st_size)
            with self.assertRaisesRegex(ValueError,'checksum'):
                restore_checkpoint(root)
            self.assertFalse((root/'model_best.pt').exists())
            self.assertFalse(list(root.glob('*.assembling')))

    def test_missing_lfs_part_is_reported_without_a_partial_checkpoint(self):
        from reward_model.restore_checkpoint import restore_checkpoint
        with scratch_directory() as tmp:
            root=Path(tmp); self.fixture(root)
            (root/'model_best.pt.part02').unlink()
            with self.assertRaises(FileNotFoundError):
                restore_checkpoint(root)
            self.assertFalse((root/'model_best.pt').exists())
            self.assertFalse(list(root.glob('*.assembling')))

    def test_existing_different_weight_is_not_overwritten(self):
        from reward_model.restore_checkpoint import restore_checkpoint
        with scratch_directory() as tmp:
            root=Path(tmp); self.fixture(root)
            (root/'model_best.pt').write_bytes(b'user weight')
            with self.assertRaises(ValueError):
                restore_checkpoint(root)
            self.assertEqual((root/'model_best.pt').read_bytes(),b'user weight')
