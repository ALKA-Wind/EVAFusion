"""Original grouped/masked and CDDFuse-style HDF5 patches, safe with Windows workers."""
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _image_tensor(array):
    array = np.asarray(array)
    integer_pixels = np.issubdtype(array.dtype, np.integer)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(f'Expected a grayscale [H,W] or [1,H,W] patch, got {array.shape}')
    array = array.astype(np.float32)
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 255:
        raise ValueError('Image patches must contain finite values in [0,1] or [0,255]')
    if integer_pixels or array.max() > 1:
        array = array / 255.0
    return torch.from_numpy(np.ascontiguousarray(array))


class H5Dataset(Dataset):
    def __init__(self, h5_path, require_masks=False):
        self.path = str(Path(h5_path).resolve())
        self.require_masks = require_masks
        with h5py.File(self.path, 'r') as file:
            self.patch_groups = 'ir_patchs' in file and 'vis_patchs' in file
            if self.patch_groups:
                self.keys = sorted(file['ir_patchs'].keys())
                if set(self.keys) != set(file['vis_patchs'].keys()):
                    raise ValueError('Infrared and visible patch keys do not match')
                self.has_masks = False
            else:
                self.keys = sorted(file.keys())
                self.has_masks = bool(self.keys) and 'mask_0' in file[self.keys[0]]
            if not self.keys:
                raise ValueError(f'No image patches in {self.path}')
            if require_masks and not self.has_masks:
                raise ValueError('Region training requires real mask_* datasets; this HDF5 has no masks')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # No persistent h5py.File object: the Dataset remains picklable under spawn.
        with h5py.File(self.path, 'r') as file:
            key = self.keys[idx]
            if self.patch_groups:
                vis = _image_tensor(file['vis_patchs'][key][()])
                ir = _image_tensor(file['ir_patchs'][key][()])
                masks = torch.empty((0, *vis.shape[-2:]), dtype=torch.bool)
            else:
                group = file[key]
                vis, ir = _image_tensor(group['vis'][()]), _image_tensor(group['ir'][()])
                names = sorted((k for k in group if k.startswith('mask_')),
                               key=lambda k: int(k.split('_')[-1]))
                masks = (torch.from_numpy(np.stack([group[k][()] for k in names]).astype(bool))
                         if names else torch.empty((0, *vis.shape[-2:]), dtype=torch.bool))
            if vis.shape != ir.shape or masks.shape[-2:] != vis.shape[-2:]:
                raise ValueError(f'Image/mask shapes differ for patch {key}')
            if self.require_masks and not masks.shape[0]:
                raise ValueError(f'No masks for patch {key}')
            return vis, ir, masks
