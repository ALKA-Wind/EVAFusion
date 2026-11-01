"""NIQE adapted from BasicSR (Apache-2.0); BRISQUE features from pinned PIQ 0.8.0.

Changes: local assets, cached AGGD table, explicit invalid-image handling and PIQ's
MATLAB-compatible bicubic resize. Attribution/license: assets/BasicSR_LICENSE.txt.
"""
from functools import lru_cache
import importlib
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma

ASSETS = Path(__file__).resolve().parent / 'assets'


@lru_cache(maxsize=1)
def _aggd_table():
    alpha = np.arange(.2, 10.001, .001)
    ratio = gamma(2 / alpha)**2 / (gamma(1 / alpha) * gamma(3 / alpha))
    return alpha, ratio


def _aggd(block):
    values = block.flatten()
    left, right = values[values < 0], values[values > 0]
    ls = np.sqrt(np.mean(left**2)) if left.size else np.nan
    rs = np.sqrt(np.mean(right**2)) if right.size else np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        gammahat = ls / rs
        rhat = np.mean(np.abs(values))**2 / np.mean(values**2)
        normalized = rhat * (gammahat**3 + 1) * (gammahat + 1) / (gammahat**2 + 1)**2
    table, ratios = _aggd_table()
    # BasicSR's argmin of an all-NaN distance selects alpha=.2. Keep its finite
    # partial features so the subsequent per-column nanmean remains compatible.
    alpha = table[np.argmin((ratios - normalized)**2)] if np.isfinite(normalized) else table[0]
    factor = np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return alpha, ls * factor, rs * factor


def _niqe_features(block):
    alpha, left, right = _aggd(block)
    result = [alpha, (left + right) / 2]
    for shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        alpha, left, right = _aggd(block * np.roll(block, shift, axis=(0, 1)))
        mean = (right - left) * gamma(2 / alpha) / gamma(1 / alpha)
        result.extend((alpha, mean, left, right))
    return result


def niqe(image, assets_dir=ASSETS):
    """Gray float [0,255], 96x96 non-overlapping blocks, two scales."""
    image = np.asarray(image, dtype=np.float32).round()
    rows, cols = image.shape[0] // 96, image.shape[1] // 96
    if rows * cols < 2:
        raise ValueError('NIQE requires at least two 96x96 blocks; image is too small')
    with np.load(Path(assets_dir) / 'niqe_pris_params.npz', allow_pickle=False) as model:
        prior_mean = model['mu_pris_param'].copy()
        prior_cov = model['cov_pris_param'].copy()
        window = model['gaussian_window'].copy()
    image = image[:rows * 96, :cols * 96]
    features = []
    for scale in (1, 2):
        mean = convolve(image, window, mode='nearest')
        std = np.sqrt(np.abs(convolve(image**2, window, mode='nearest') - mean**2))
        normalized = (image - mean) / (std + 1)
        size = 96 // scale
        features.append(np.asarray([_niqe_features(normalized[y*size:(y+1)*size, x*size:(x+1)*size])
                                    for x in range(cols) for y in range(rows)]))
        if scale == 1:
            import torch
            from piq.functional import imresize
            tensor = torch.from_numpy(image / 255.)[None, None]
            image = imresize(tensor, scale=.5, antialiasing=True)[0, 0].numpy() * 255.
    features = np.concatenate(features, axis=1)
    valid = features[np.isfinite(features).all(axis=1)]
    if len(valid) < 2:
        raise ValueError('NIQE is undefined: fewer than two nondegenerate feature blocks')
    # Match BasicSR: per-feature nanmean for the mean, complete rows for covariance.
    if np.isnan(features).all(axis=0).any():
        raise ValueError('NIQE is undefined for degenerate image statistics')
    mean = np.nanmean(features, axis=0)
    cov = np.cov(valid, rowvar=False)
    delta = prior_mean - mean
    value = float(np.squeeze(delta @ np.linalg.pinv((prior_cov + cov) / 2) @ delta.T))
    if not np.isfinite(value):
        raise ValueError('NIQE produced nonfinite statistics')
    return float(np.sqrt(max(value, 0.)))


class NoReferenceMetrics:
    def __init__(self, device='cpu', assets_dir=ASSETS):
        self.device, self.assets_dir = device, Path(assets_dir)
        self._svm = None

    def brisque(self, image):
        import torch
        from piq.functional import imresize
        if min(image.shape) < 16:
            raise ValueError('BRISQUE requires image dimensions of at least 16 pixels')
        module = importlib.import_module('piq.brisque')
        with torch.no_grad():
            value = torch.as_tensor(np.ascontiguousarray(image), device=self.device, dtype=torch.float32)[None, None]
            value = value / 255. * 255.  # Match PIQ's public data_range=255 preprocessing exactly.
            features = []
            for _ in range(2):
                features.append(module._natural_scene_statistics(value, 7, 7 / 6))
                value = imresize(value, sizes=(value.shape[-2] // 2, value.shape[-1] // 2))
            features = module._scale_features(torch.cat(features, dim=-1))
            if self._svm is None:
                coefficient, vectors = torch.load(self.assets_dir / 'brisque_svm_weights.pt',
                                                   map_location=self.device, weights_only=True)
                # Preserve PIQ's transposed stride: SVR cancellation is sensitive to reduction order.
                self._svm = coefficient.to(features), vectors.to(features).T
            coefficient, vectors = self._svm
            kernel = torch.exp(-.05 * (features[:, :, None] - vectors[None]).square().sum(1))
            score = float((kernel @ coefficient + 153.591).item())
        if not np.isfinite(score):
            raise ValueError('BRISQUE produced nonfinite statistics')
        return score
