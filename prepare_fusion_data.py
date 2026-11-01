"""Build normalized grayscale HDF5 patches; no semantic masks are invented."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from utils.img_read_save import image_read_cv2


def create_h5(ir_dir, vi_dir, output, patch_size=128, stride=200):
    if patch_size < 2 or stride < 1:
        raise ValueError('patch_size must be >=2 and stride must be positive')
    files = sorted(p for p in Path(ir_dir).iterdir()
                   if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'})
    if not files:
        raise ValueError(f'No images in {ir_dir}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(output, 'x') as file:
        ir_group, vi_group = file.create_group('ir_patchs'), file.create_group('vis_patchs')
        file.attrs['patch_size'], file.attrs['stride'] = patch_size, stride
        file.attrs['recipe'] = 'fixed grid; drop incomplete border patches; normalize to [0,1]'
        for path in files:
            ir = image_read_cv2(path, 'GRAY').astype(np.float32) / 255.
            vis = image_read_cv2(Path(vi_dir) / path.name, 'GRAY').astype(np.float32) / 255.
            if ir.shape != vis.shape:
                raise ValueError(f'Unregistered image pair: {path.name}')
            for y in range(0, ir.shape[0] - patch_size + 1, stride):
                for x in range(0, ir.shape[1] - patch_size + 1, stride):
                    for group, image in [(ir_group, ir), (vi_group, vis)]:
                        group.create_dataset(str(count), data=image[None, y:y+patch_size, x:x+patch_size], compression='lzf')
                    count += 1
        if not count:
            raise ValueError('No patches fit the images; lower --patch-size')
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir-dir', required=True)
    parser.add_argument('--vi-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=200)
    args = parser.parse_args(argv)
    print('Saved', create_h5(args.ir_dir, args.vi_dir, args.output, args.patch_size, args.stride), 'patches')


if __name__ == '__main__':
    main()
