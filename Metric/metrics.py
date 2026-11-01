"""Single source of fusion metrics. Arrays are grayscale, in the [0,255] range."""
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
from Metric.Qabf import get_Qabf

PAPER_METRICS = ('CC', 'PSNR', 'Qabf', 'SSIM', 'NIQE', 'BRISQUE')
CORE_METRICS = ('CC', 'PSNR', 'SSIM', 'Qabf')
SUPPORTED_METRICS = PAPER_METRICS + ('EN', 'SD', 'SF', 'AG', 'MI', 'SCD', 'MSE', 'VIF', 'MS_SSIM')
FUSED_ONLY = frozenset(('NIQE', 'BRISQUE', 'EN', 'SD', 'SF', 'AG'))
LOWER_BETTER = frozenset(('NIQE', 'BRISQUE', 'MSE'))
PSNR_MODES = ('mean-db', 'mean-mse', 'legacy-normalized-255')


def _image(array):
    image = np.asarray(array, dtype=np.float64)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError('Expected a nonempty finite grayscale image [H,W]')
    if image.min() < 0 or image.max() > 255:
        raise ValueError('Image values must be in [0,255]; normalized arrays must be scaled explicitly')
    return image


def calculate_cc(img1, img2):
    a, b = np.asarray(img1, dtype=np.float64).ravel(), np.asarray(img2, dtype=np.float64).ravel()
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a.dot(b) / denominator) if denominator else 0.


def calculate_psnr(img1, img2):
    mse = np.mean((np.asarray(img1, dtype=np.float64) - np.asarray(img2, dtype=np.float64))**2)
    return float(10 * np.log10(255.**2 / mse)) if mse else float('inf')


def calculate_legacy_fusion_psnr(fused, ir, vi):
    """Original DCEvo Metric_torch.PSNR_function, including its peak convention.

    Inputs are 0..255. The original normalizes pixels to 0..1, averages
    the two MSEs, but retains peak 255. Keep float32 Torch arithmetic for
    historical comparison; this is not conventional range-consistent PSNR.
    """
    import torch
    a, b, f = [torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)/255.
               for x in (ir, vi, fused)]
    mse = .5*torch.mean((f-a)**2) + .5*torch.mean((f-b)**2)
    return float(20*torch.log10(255/torch.sqrt(mse)))


def calculate_ssim(img1, img2):
    if min(img1.shape) < 7:
        raise ValueError('SSIM requires image dimensions of at least 7 pixels')
    return float(structural_similarity(img1, img2, data_range=255))


def _mi(a, b):
    counts = np.histogram2d(a.round().ravel(), b.round().ravel(), bins=256, range=((0, 256), (0, 256)))[0]
    joint = counts / counts.sum()
    independent = joint.sum(1)[:, None] * joint.sum(0)[None]
    valid = joint > 0
    return float((joint[valid] * np.log2(joint[valid] / independent[valid])).sum())


class FusionMetrics:
    def __init__(self, names=PAPER_METRICS, device='cpu', assets_dir=None, psnr_mode='mean-db'):
        self.names = tuple(dict.fromkeys(names))
        if not self.names or set(self.names) - set(SUPPORTED_METRICS):
            raise ValueError(f'Unknown/empty metric selection: {names}')
        if psnr_mode not in PSNR_MODES:
            raise ValueError(f'PSNR mode must be one of {PSNR_MODES}')
        self.device, self.psnr_mode = device, psnr_mode
        self.assets_dir = Path(assets_dir or Path(__file__).resolve().parent / 'assets')
        self._no_reference = None
        if set(self.names) & {'NIQE', 'BRISQUE', 'VIF', 'MS_SSIM'}:
            try:
                import piq  # fail once, never download weights at runtime
            except ImportError as exc:
                raise ImportError('Install metric dependencies: python -m pip install -r Metric/requirements.txt') from exc
        for name, filename in [('NIQE', 'niqe_pris_params.npz'), ('BRISQUE', 'brisque_svm_weights.pt')]:
            if name in self.names and not (self.assets_dir / filename).is_file():
                raise FileNotFoundError(self.assets_dir / filename)

    def _score(self, name, fused, ir, vi):
        if name not in FUSED_ONLY and ir is None:
            raise ValueError(f'{name} requires infrared and visible reference images')
        if name == 'CC':
            return (calculate_cc(fused, ir) + calculate_cc(fused, vi)) / 2
        if name == 'PSNR':
            if self.psnr_mode == 'legacy-normalized-255':
                return calculate_legacy_fusion_psnr(fused, ir, vi)
            if self.psnr_mode == 'mean-db':
                return (calculate_psnr(fused, ir) + calculate_psnr(fused, vi)) / 2
            mse = (np.mean((fused-ir)**2) + np.mean((fused-vi)**2)) / 2
            return float(10 * np.log10(255.**2 / mse)) if mse else float('inf')
        if name == 'SSIM':
            return (calculate_ssim(fused, ir) + calculate_ssim(fused, vi)) / 2
        if name == 'Qabf':
            return float(get_Qabf(ir, vi, fused))
        if name == 'EN':
            histogram = np.bincount(fused.round().astype(np.uint8).ravel(), minlength=256)
            p = histogram[histogram > 0] / fused.size
            return float(-(p * np.log2(p)).sum())
        if name == 'SD':
            return float(fused.std())
        if name in ('SF', 'AG'):
            if min(fused.shape) < 2:
                raise ValueError(f'{name} requires both dimensions >=2')
            if name == 'SF':
                return float(np.sqrt(np.mean(np.diff(fused, axis=0)**2) + np.mean(np.diff(fused, axis=1)**2)))
            dy, dx = np.gradient(fused)
            return float(np.sqrt((dx**2 + dy**2) / 2).mean())
        if name == 'MI':
            return _mi(fused, ir) + _mi(fused, vi)
        if name == 'SCD':
            return calculate_cc(fused-ir, vi) + calculate_cc(fused-vi, ir)
        if name == 'MSE':
            return float((np.mean((fused-ir)**2) + np.mean((fused-vi)**2)) / (2 * 255.**2))
        if name == 'NIQE':
            from Metric.no_reference import niqe
            return niqe(fused, self.assets_dir)
        if name == 'BRISQUE':
            if self._no_reference is None:
                from Metric.no_reference import NoReferenceMetrics
                self._no_reference = NoReferenceMetrics(self.device, self.assets_dir)
            return self._no_reference.brisque(fused)
        import torch
        import piq
        with torch.no_grad():
            f, a, b = [torch.as_tensor(np.ascontiguousarray(x), device=self.device, dtype=torch.float32)[None, None]
                       for x in (fused, ir, vi)]
            if name == 'VIF':
                return float(piq.vif_p(f, a, data_range=255) + piq.vif_p(f, b, data_range=255))
            return float((piq.multi_scale_ssim(a, f, data_range=255)
                          + piq.multi_scale_ssim(b, f, data_range=255)) / 2)

    def evaluate(self, fused, ir=None, vi=None, on_error='raise'):
        if on_error not in ('raise', 'record'):
            raise ValueError('on_error must be raise or record')
        fused = _image(fused)
        if (ir is None) != (vi is None):
            raise ValueError('Both reference images must be supplied together')
        if ir is not None:
            ir, vi = _image(ir), _image(vi)
            if not fused.shape == ir.shape == vi.shape:
                raise ValueError(f'Image shapes must match: F={fused.shape}, IR={ir.shape}, VI={vi.shape}')
        values, errors = {}, {}
        for name in self.names:
            try:
                values[name] = self._score(name, fused, ir, vi)
                if np.isnan(values[name]) or (name != 'PSNR' and not np.isfinite(values[name])):
                    raise ValueError('Metric returned nonfinite statistics')
            except (ValueError, AssertionError, FloatingPointError, np.linalg.LinAlgError) as exc:
                if on_error == 'raise':
                    raise ValueError(f'{name}: {exc}') from exc
                values[name], errors[name] = float('nan'), str(exc)
        return values, errors


def fusion_metrics(img_fused, img_ir, img_vis, psnr_mode='mean-db'):
    """Fast four-metric interface used by inference, quantized like the saved PNG."""
    images = [np.clip(np.round(x), 0, 255) for x in (img_fused, img_ir, img_vis)]
    return FusionMetrics(CORE_METRICS, psnr_mode=psnr_mode).evaluate(*images)[0]
