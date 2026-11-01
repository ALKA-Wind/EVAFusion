"""Image IO adapted from original DCEvo utilities, including grayscale input."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def image_read_cv2(path, mode='RGB'):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    if mode == 'GRAY':
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mode == 'YCrCb':
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    if mode != 'RGB':
        raise ValueError(f'Unsupported image mode: {mode}')
    return rgb


def img_save(image, imagename, savepath):
    path = Path(savepath)
    path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(image), 0, 255).astype(np.uint8)).save(path / f'{imagename}.png')
