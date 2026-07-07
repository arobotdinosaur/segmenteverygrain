# Extended degradation model: signal blur (PSF) + contrast loss + the noise
# model from synthetic_noise.py. Standalone — does not modify synthetic_noise.py.
#
# Forward model (in normalized [0, 1] space):
#   x -> GaussianBlur(x, sigma_psf) -> contrast scaling -> + noise (a, b, sigma_r, l, k)
#
# theta layout (for optimization / BO):
#   [sigma_psf, contrast, a, b, sigma_r, l, k]
#
# Usage:
#   Fit parameters to a folder of real blurry images:
#     python synthetic_degradation.py --clear-folder ./testcleanimages/ \
#         --blurry-folder ./testnoisyimages/ --maxiter 20
#   Preview degraded images at given/fitted parameters:
#     python synthetic_degradation.py --preview-only --preview-dir ./degraded_preview/
from dataclasses import dataclass
import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import differential_evolution

from synthetic_noise import (
    sample_signal_dependent_noise,
    sample_row_noise,
    sample_correlated_noise,
    get_noise_loss,
    load_images_from_folder,
    downsample_image,
    downsample_mask,
    percentile_normalize,
    percentile_normalize_params,
    invert_percentile_normalize,
    extract_noise,
    get_noise_stats,
)


@dataclass
class DegradationParams:
    sigma_psf: float   # Gaussian PSF sigma applied to the signal (defocus blur)
    contrast: float    # contrast factor around the image mean (1 = unchanged)
    a: float           # signal-dependent noise: var = a*x + b
    b: float
    sigma_r: float     # per-row offset noise
    l: float           # correlation length of correlated noise
    k: float           # amplitude of correlated noise


THETA_NAMES = ["sigma_psf", "contrast", "a", "b", "sigma_r", "l", "k"]

# Bounds for theta = [sigma_psf, contrast, a, b, sigma_r, l, k].
# Noise bounds match synthetic_noise.py so fits stay comparable.
BOUNDS = [
    (0.0, 6.0),     # sigma_psf
    (0.3, 1.2),     # contrast
    (1e-6, 0.2),    # a
    (1e-6, 0.05),   # b
    (1e-6, 0.05),   # sigma_r
    (0.3, 8.0),     # l
    (1e-6, 0.1),    # k
]


def params_from_theta(theta) -> DegradationParams:
    theta = np.asarray(theta, dtype=float)
    return DegradationParams(*[float(v) for v in theta])


def theta_from_params(params: DegradationParams) -> np.ndarray:
    return np.array([getattr(params, name) for name in THETA_NAMES], dtype=float)


def apply_psf_blur(x: np.ndarray, sigma_psf: float) -> np.ndarray:
    """Defocus blur of the signal itself (unlike l/k, which blur the noise field)."""
    if sigma_psf <= 1e-3:
        return x
    return cv2.GaussianBlur(x, ksize=(0, 0), sigmaX=sigma_psf, sigmaY=sigma_psf)


def apply_contrast(x: np.ndarray, contrast: float) -> np.ndarray:
    """Scale contrast around the image mean; defocused SEM images lose contrast."""
    mean = float(np.mean(x))
    return mean + contrast * (x - mean)


def synthetic_degradation(x: np.ndarray, params: DegradationParams, rng: np.random.Generator) -> np.ndarray:
    """Full forward model on a normalized [0, 1] image: blur -> contrast -> noise."""
    x = apply_psf_blur(x, params.sigma_psf)
    x = apply_contrast(x, params.contrast)
    n_signal = sample_signal_dependent_noise(np.clip(x, 0.0, 1.0), params.a, params.b, rng)
    n_row = sample_row_noise(x.shape, params.sigma_r, rng)
    n_corr = sample_correlated_noise(x.shape, params.l, params.k, rng)
    return x + n_signal + n_row + n_corr


def synthetic_degradation_model_input(x_raw: np.ndarray, params: DegradationParams, rng: np.random.Generator) -> np.ndarray:
    """Degrade an unnormalized image: normalize, degrade, map back to original range."""
    x_norm, p_lo, p_hi = percentile_normalize_params(x_raw)
    y_norm = synthetic_degradation(x_norm, params, rng)
    y_raw = invert_percentile_normalize(y_norm, p_lo, p_hi)
    return y_raw.astype(np.float32)


def make_degraded_training_pair(
    clean_img: np.ndarray,
    clean_mask: np.ndarray,
    target_shape: tuple[int, int],
    params: DegradationParams,
    rng: np.random.Generator,
):
    """Drop-in analogue of synthetic_noise.make_noisy_training_pair.

    Blur/contrast/noise do not move grain boundaries, so the mask carries over.
    """
    img_ds = downsample_image(clean_img, target_shape)
    mask_ds = downsample_mask(clean_mask, target_shape)
    degraded_img = synthetic_degradation_model_input(img_ds, params, rng)
    return degraded_img, mask_ds, img_ds


def gradient_hist_loss(synthetic_img: np.ndarray, real_img: np.ndarray, num_bins=64, grad_range=(0.0, 0.5)) -> float:
    """L1 distance between gradient-magnitude histograms.

    The histogram+pixel loss in get_noise_loss is nearly blind to blur (a blurred
    and a sharp image can share an intensity histogram); gradient magnitudes
    collapse under defocus, so this term is what makes sigma_psf identifiable.
    """
    losses = []
    for img_a, img_b in [(synthetic_img, real_img)]:
        gy_a, gx_a = np.gradient(img_a)
        gy_b, gx_b = np.gradient(img_b)
        mag_a = np.sqrt(gx_a**2 + gy_a**2)
        mag_b = np.sqrt(gx_b**2 + gy_b**2)

        hist_a, _ = np.histogram(mag_a.ravel(), bins=num_bins, range=grad_range, density=True)
        hist_b, _ = np.histogram(mag_b.ravel(), bins=num_bins, range=grad_range, density=True)
        hist_a = hist_a / (hist_a.sum() + 1e-8)
        hist_b = hist_b / (hist_b.sum() + 1e-8)
        losses.append(float(np.mean(np.abs(hist_a - hist_b))))
    return float(np.mean(losses))


def get_degradation_loss(
    synthetic_img: np.ndarray,
    real_img: np.ndarray,
    noise_weight=1.0,
    grad_weight=2.0,
) -> float:
    """Composite realism loss: intensity/pixel statistics + blur-sensitive gradient term."""
    synthetic_img = percentile_normalize(synthetic_img.astype(np.float32))
    real_img = percentile_normalize(real_img.astype(np.float32))
    noise_term = get_noise_loss(synthetic_img, real_img)
    grad_term = gradient_hist_loss(synthetic_img, real_img)
    return noise_weight * noise_term + grad_weight * grad_term


# --- Analytic (closed-form) parameter estimation -------------------------------
#
# Instead of searching for theta with DE/BO, estimate each component directly from
# image statistics. All estimates operate on percentile-normalized [0, 1] images,
# matching the space the forward model works in.
#   sigma_psf : ratio of radially-averaged power spectra, blurry vs clean; for a
#               Gaussian PSF, log(P_b/P_c) = const - 4*pi^2*sigma^2*f^2, so a linear
#               fit of the log-ratio against f^2 over the signal band gives sigma.
#   contrast  : ratio of the low-frequency (heavily smoothed) standard deviations.
#   a, b      : variance of the high-pass noise binned by intensity, least-squares
#               fit of var = a*x + b (get_noise_stats); the intensity-independent
#               parts contributed by row/correlated noise are subtracted from b.
#   sigma_r   : variance of noise row-means minus the white-noise contribution.
#   l, k      : radial autocorrelation of the noise field; the correlated component
#               has acf(r) ~ k^2 * exp(-r^2 / (4 l^2)), fit log-linearly in r^2.
# Caveats: the spectral ratio assumes clean and blurry sets share texture statistics
# (different specimens bias sigma_psf), and extract_noise's high-pass attenuates
# long-range correlation (biases l, k low). These are starting points to refine
# with BO if needed, not ground truth.

def _radial_power_spectrum(img, n_bins=60):
    img = img - img.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    h, w = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bins = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.clip(np.digitize(r.ravel(), bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=p.ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    ps = sums / np.maximum(counts, 1)
    freqs = 0.5 * (bins[:-1] + bins[1:])
    return freqs, ps


def _mean_spectrum(images, n_bins=60):
    specs = []
    for img in images:
        f, ps = _radial_power_spectrum(percentile_normalize(img), n_bins)
        specs.append(ps)
    return f, np.mean(specs, axis=0)


def estimate_sigma_psf(clean_imgs, blurry_imgs, f_lo=0.01, f_hi=0.2, floor_factor=5.0):
    """Gaussian-PSF sigma from the blurry/clean power-spectrum ratio (linear fit).

    The fit band is restricted adaptively to frequencies where BOTH spectra sit
    well above their high-frequency noise floors, so added noise (which dominates
    the blurry spectrum at mid/high f) cannot flatten the slope.
    """
    f, p_clean = _mean_spectrum(clean_imgs)
    _, p_blur = _mean_spectrum(blurry_imgs)
    floor_c = np.nanmean(p_clean[int(0.9 * len(f)):])
    floor_b = np.nanmean(p_blur[int(0.9 * len(f)):])
    num = np.maximum(p_blur - floor_b, 1e-12)
    den = np.maximum(p_clean - floor_c, 1e-12)
    band = ((f >= f_lo) & (f <= f_hi)
            & (p_blur > floor_factor * floor_b) & (p_clean > floor_factor * floor_c))
    if band.sum() < 3:
        band = (f >= f_lo) & (f <= 0.08)
    x = f[band] ** 2
    y = np.log(num[band] / den[band])
    slope, intercept = np.polyfit(x, y, 1)
    sigma = np.sqrt(max(-slope, 0.0)) / (2 * np.pi)
    return float(np.clip(sigma, 0.0, 8.0)), dict(f=f, ratio=num / den, band=band,
                                                 slope=slope, intercept=intercept)


def _edge_profile_width(img, detect_sigma=2.0, half=12, grad_percentile=99.5, max_profiles=4000):
    """Mean second-moment width of intensity profiles across strong edges.

    Profiles are sampled along the gradient direction at the strongest-gradient
    pixels of a detect_sigma-smoothed image; their averaged derivative (the line
    spread function) has second moment sigma_total^2 = sigma_psf^2 + intrinsic^2
    + detect_sigma^2. Comparing blurry vs clean cancels the shared terms.
    """
    im = cv2.GaussianBlur(percentile_normalize(np.asarray(img, np.float32)),
                          (0, 0), detect_sigma).astype(np.float32)
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    h, w = im.shape
    m = np.zeros_like(mag, bool)
    m[half + 1:h - half - 1, half + 1:w - half - 1] = True
    thresh = np.percentile(mag[m], grad_percentile)
    ys, xs = np.where((mag >= thresh) & m)
    if len(ys) > max_profiles:
        sel = np.random.default_rng(0).choice(len(ys), max_profiles, replace=False)
        ys, xs = ys[sel], xs[sel]
    if len(ys) < 50:
        return np.nan

    t = np.arange(-half, half + 1, dtype=np.float32)
    nx = gx[ys, xs] / np.maximum(mag[ys, xs], 1e-8)
    ny = gy[ys, xs] / np.maximum(mag[ys, xs], 1e-8)
    px = xs[:, None] + t[None, :] * nx[:, None]
    py = ys[:, None] + t[None, :] * ny[:, None]
    x0 = np.clip(px.astype(int), 0, w - 2)
    y0 = np.clip(py.astype(int), 0, h - 2)
    fx, fy = px - x0, py - y0
    prof = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy)
            + im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)

    lsf = np.abs(np.gradient(prof.mean(axis=0)))
    lsf = np.maximum(lsf - lsf[[0, 1, -2, -1]].mean(), 0.0)      # strip noise baseline
    if lsf.sum() <= 0:
        return np.nan
    mu = (t * lsf).sum() / lsf.sum()
    return float(np.sqrt(((t - mu) ** 2 * lsf).sum() / lsf.sum()))


def estimate_sigma_psf_edges(clean_imgs, blurry_imgs, **kwargs):
    """PSF sigma from edge widths: sqrt(width_blurry^2 - width_clean^2).

    Robust to added noise (thousands of profiles are averaged), but assumes the
    two sets have comparably sharp intrinsic edges -- different specimens with
    different boundary/groove widths bias the estimate.
    """
    w_clean = np.nanmean([_edge_profile_width(i, **kwargs) for i in clean_imgs])
    w_blur = np.nanmean([_edge_profile_width(i, **kwargs) for i in blurry_imgs])
    if np.isnan(w_clean) or np.isnan(w_blur):
        return 0.0, dict(w_clean=w_clean, w_blur=w_blur)
    sigma = np.sqrt(max(w_blur ** 2 - w_clean ** 2, 0.0))
    return float(np.clip(sigma, 0.0, 8.0)), dict(w_clean=float(w_clean), w_blur=float(w_blur))


def estimate_sigma_psf_multiscale(clean_imgs, blurry_imgs, taus=(4, 5, 6, 8, 10, 12),
                                  sigma_grid=np.arange(0.0, 8.01, 0.05), noise_params=None):
    """PSF sigma by matching multiscale variance decay curves.

    For images I smoothed by G_tau, Var(G_tau * blur_sigma(I)) = Var(G_sqrt(tau^2+sigma^2) * I),
    and a global contrast factor c scales the whole curve by c^2. So with
    V_clean(tau) measured densely on the clean set, sigma is the grid value whose
    predicted curve c^2 * V_clean(sqrt(tau^2+sigma^2)) best matches V_blurry(tau)
    in log space (c^2 solved in closed form per candidate). Whole-image statistic:
    no edge selection bias, and noise contributes negligibly at tau >= 4.
    """
    def var_at(img, tau):
        im = percentile_normalize(np.asarray(img, np.float32)).astype(np.float32)
        return float(np.var(cv2.GaussianBlur(im, (0, 0), float(tau))))

    taus = np.asarray(taus, float)
    dense = np.arange(taus.min(), np.sqrt(taus.max() ** 2 + sigma_grid.max() ** 2) + 1.0, 0.5)
    v_clean_dense = np.array([np.mean([var_at(i, t) for i in clean_imgs]) for t in dense])
    v_blur = np.array([np.mean([var_at(i, t) for i in blurry_imgs]) for t in taus])

    if noise_params is not None:
        # subtract the predicted noise variance that survives G_tau smoothing:
        # white/signal-dependent ~ 1/(4 pi tau^2); correlated ~ l^2/(l^2+tau^2);
        # row offsets are smoothed only along columns ~ 1/(2 sqrt(pi) tau)
        p = noise_params
        x_mean = float(np.mean([percentile_normalize(np.asarray(i, np.float32)).mean()
                                for i in blurry_imgs]))
        v_noise = ((p["a"] * x_mean + p["b"]) / (4 * np.pi * taus ** 2)
                   + p["k"] ** 2 * p["l"] ** 2 / (p["l"] ** 2 + taus ** 2)
                   + p["sigma_r"] ** 2 / (2 * np.sqrt(np.pi) * taus))
        v_blur = np.maximum(v_blur - v_noise, 0.05 * v_blur)

    best_sigma, best_res, best_c2 = 0.0, np.inf, 1.0
    for sigma in sigma_grid:
        v_hat = np.interp(np.sqrt(taus ** 2 + sigma ** 2), dense, v_clean_dense)
        # optimal log-offset (= log c^2) and residual, both in log space
        d = np.log(v_blur) - np.log(np.maximum(v_hat, 1e-12))
        res = float(np.var(d))
        if res < best_res:
            best_sigma, best_res, best_c2 = float(sigma), res, float(np.exp(d.mean()))
    return best_sigma, dict(residual=best_res, c2=best_c2, taus=taus, v_blur=v_blur)


def estimate_contrast(clean_imgs, blurry_imgs, smooth_sigma=8.0):
    """Low-frequency contrast ratio between the blurry and clean sets.

    Measured on RAW intensities (not percentile-normalized -- normalization would
    re-stretch the contrast away). Confound to be aware of: different exposure
    settings between the two sets also land in this ratio.
    """
    def lf_std(imgs):
        return np.mean([np.std(cv2.GaussianBlur(np.asarray(i, np.float32), (0, 0), smooth_sigma))
                        for i in imgs])
    c = lf_std(blurry_imgs) / max(lf_std(clean_imgs), 1e-8)
    return float(np.clip(c, 0.2, 1.5))


def _noise_field(img, filter_strength=3.0):
    return extract_noise(percentile_normalize(img), filter_strength=filter_strength)


def estimate_row_sigma(blurry_imgs):
    """Row-offset noise: variance of noise row-means minus the white-noise share."""
    vals = []
    for img in blurry_imgs:
        n = _noise_field(img)
        var_rows = np.var(n.mean(axis=1))
        vals.append(max(var_rows - n.var() / n.shape[1], 0.0))
    return float(np.sqrt(np.mean(vals)))


def estimate_correlated(blurry_imgs, max_lag=8):
    """(l, k) of the correlated noise from the radial noise autocorrelation.

    Lag 1 is skipped: the high-pass extraction filter itself correlates adjacent
    pixels of white noise, which would inflate k.
    """
    ls, k2s = [], []
    for img in blurry_imgs:
        n = _noise_field(img)
        n = n - n.mean()
        f = np.fft.fft2(n)
        acf = np.fft.ifft2(np.abs(f) ** 2).real / n.size          # Wiener-Khinchin
        lags = np.arange(2, max_lag + 1)
        acf_r = np.array([(acf[0, lag] + acf[lag, 0]) / 2 for lag in lags])
        if acf_r[0] <= 0:
            continue
        valid = acf_r > 0
        if valid.sum() < 2:
            continue
        slope, intercept = np.polyfit(lags[valid] ** 2, np.log(acf_r[valid]), 1)
        if slope >= 0:
            continue
        ls.append(np.sqrt(1.0 / (-4.0 * slope)))
        k2s.append(np.exp(intercept))
    if not ls:
        return 0.5, 0.0
    l = float(np.clip(np.mean(ls), 0.3, 8.0))
    k = float(np.clip(np.sqrt(np.mean(k2s)), 0.0, 0.3))
    return l, k


def _white_capture_fraction(filter_strength=1.0, size=512, seed=0):
    """Fraction of white-noise variance the high-pass extraction filter retains."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    return float(np.var(extract_noise(w, filter_strength=filter_strength)))


def estimate_noise_ab(blurry_imgs, num_bins=10):
    """Least-squares fit of noise variance vs intensity: var = a*x + b.

    The high-pass filter only captures part of the noise variance; the fitted
    coefficients are corrected by the filter's white-noise capture fraction.
    """
    xs, vs = [], []
    for img in blurry_imgs:
        mean_list, var_list = get_noise_stats(percentile_normalize(img), num_bins=num_bins)
        xs += list(mean_list)
        vs += list(var_list)
    a, b = np.polyfit(np.array(xs), np.array(vs), 1)
    cap = _white_capture_fraction()
    return float(max(a / cap, 1e-6)), float(max(b / cap, 1e-6))


def estimate_degradation_analytic(clean_imgs, blurry_imgs, verbose=True):
    """Closed-form estimate of all degradation parameters. Returns (params, diagnostics).

    sigma_psf comes from multiscale variance matching (no selection bias, noise
    robust); edge-profile and spectral-ratio estimates land in the diagnostics
    as cross-checks.
    """
    contrast = estimate_contrast(clean_imgs, blurry_imgs)
    sigma_r = estimate_row_sigma(blurry_imgs)
    l, k = estimate_correlated(blurry_imgs)
    a, b_total = estimate_noise_ab(blurry_imgs)
    # Row and correlated noise contribute intensity-independent variance. k tends to
    # be over-estimated when grain texture leaks into the extracted noise field, so
    # keep at least 20% of the fitted intercept as true white-noise b.
    b = max(b_total - k ** 2 - sigma_r ** 2, 0.2 * b_total, 1e-6)
    noise = dict(a=a, b=b, sigma_r=sigma_r, l=l, k=k)
    sigma_psf_ms, ms_diag = estimate_sigma_psf_multiscale(clean_imgs, blurry_imgs, noise_params=noise)
    sigma_psf_edge, edge_diag = estimate_sigma_psf_edges(clean_imgs, blurry_imgs)
    sigma_psf_spec, psf_diag = estimate_sigma_psf(clean_imgs, blurry_imgs)
    # Both estimators fail toward UNDER-estimation (multiscale: coarser blurry
    # content mimics "less blur"; edge-profile: noise spikes select sharp-looking
    # profiles), so the max of the two is the defensible combination.
    sigma_psf = max(sigma_psf_ms, sigma_psf_edge)
    params = DegradationParams(sigma_psf=sigma_psf, contrast=contrast,
                               a=a, b=b, sigma_r=sigma_r, l=l, k=k)
    if verbose:
        for name in THETA_NAMES:
            print(f"  {name:>10} = {getattr(params, name):.5g}")
        print(f"  (sigma_psf estimates -- multiscale: {sigma_psf_ms:.2f}, "
              f"edge-profile: {sigma_psf_edge:.2f}, spectral-ratio: {sigma_psf_spec:.2f}; "
              f"using max of first two)")
    return params, dict(psf=psf_diag, edges=edge_diag, multiscale=ms_diag,
                        sigma_psf_edge=sigma_psf_edge,
                        sigma_psf_spectral=sigma_psf_spec, b_total=b_total)


_obj_call_count = 0


def objective(theta_vec, clear_imgs, real_imgs, seed=2):
    """Average degradation-matching loss across clear/real image pairs."""
    global _obj_call_count
    _obj_call_count += 1
    params = params_from_theta(theta_vec)
    rng = np.random.default_rng(seed)

    n = min(len(clear_imgs), len(real_imgs))
    start = time.time()
    losses = []
    for i in range(n):
        clean_lowres = downsample_image(clear_imgs[i], real_imgs[i].shape)
        clean_lowres = percentile_normalize(clean_lowres)
        real_img = percentile_normalize(real_imgs[i])
        syn_img = synthetic_degradation(clean_lowres, params, rng)
        losses.append(get_degradation_loss(syn_img, real_img))

    avg_loss = float(np.mean(losses))
    print(f"  OBJ#{_obj_call_count}: loss={avg_loss:.3e} ({time.time() - start:.1f}s, {n} images)")
    return avg_loss


class OptimizationLogger:
    def __init__(self, maxiter):
        self.maxiter = maxiter
        self.start_time = time.time()
        self.last_log = self.start_time

    def __call__(self, xk, convergence):
        elapsed = time.time() - self.start_time
        iters = int(convergence * self.maxiter) if convergence else 0
        since_last = time.time() - self.last_log
        param_str = ", ".join(f"{name}={v:.3g}" for name, v in zip(THETA_NAMES, xk))
        print(f"[{timedelta(seconds=int(elapsed))}] iter={iters}/{self.maxiter} | {param_str} | +{since_last:.1f}s")
        self.last_log = time.time()
        return False


def fit_parameters(clear_folder, blurry_folder, maxiter=20, popsize=10, n_images=10, seed=2):
    """Fit degradation parameters so degraded clear images match real blurry ones."""
    print(f"Loading clear images from {clear_folder}...")
    clear_imgs, _ = load_images_from_folder(clear_folder)
    print(f"Loading blurry reference images from {blurry_folder}...")
    real_imgs, _ = load_images_from_folder(blurry_folder)
    print(f"Loaded {len(clear_imgs)} clear, {len(real_imgs)} blurry images")
    if not clear_imgs or not real_imgs:
        raise ValueError("Both folders must contain at least one readable image.")

    result = differential_evolution(
        objective,
        bounds=BOUNDS,
        args=(clear_imgs[:n_images], real_imgs[:n_images], seed),
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        seed=seed,
        callback=OptimizationLogger(maxiter),
        disp=True,
    )

    best_params = params_from_theta(result.x)
    print("\nOptimized parameters:")
    for name in THETA_NAMES:
        print(f"  {name} = {getattr(best_params, name):.6g}")
    print(f"Best loss: {float(result.fun):.6f}")
    return best_params, float(result.fun)


def save_preview(params, clear_folder, preview_dir, target_shape=None, n_images=5, seed=42):
    """Save side-by-side clean/degraded previews for visual sanity-checking."""
    clear_imgs, paths = load_images_from_folder(clear_folder)
    rng = np.random.default_rng(seed)
    preview_dir = Path(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    for img, path in list(zip(clear_imgs, paths))[:n_images]:
        if target_shape is not None:
            img = downsample_image(img, target_shape)
        degraded = synthetic_degradation_model_input(img, params, rng)
        side_by_side = np.hstack([img, np.clip(degraded, 0.0, 1.0)])
        out_path = preview_dir / f"{Path(path).stem}_preview.png"
        cv2.imwrite(str(out_path), (side_by_side * 255).astype(np.uint8))
        print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit blur+contrast+noise degradation parameters to real blurry images.")
    parser.add_argument("--clear-folder", default="./testcleanimages/")
    parser.add_argument("--blurry-folder", default="./testnoisyimages/")
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--n-images", type=int, default=10, help="Max image pairs used in the objective")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--preview-dir", default=None, help="If set, save clean|degraded previews here after fitting")
    parser.add_argument("--preview-only", action="store_true", help="Skip fitting; preview using --theta (or defaults)")
    parser.add_argument("--theta", type=float, nargs=len(THETA_NAMES), default=None,
                        metavar=tuple(name.upper() for name in THETA_NAMES),
                        help="Explicit parameter values for --preview-only")
    args = parser.parse_args()

    if args.preview_only:
        theta = args.theta if args.theta is not None else [2.0, 0.8, 0.0024, 0.0017, 0.0007, 0.76, 0.074]
        params = params_from_theta(theta)
        preview_dir = args.preview_dir or "./degraded_preview/"
        save_preview(params, args.clear_folder, preview_dir, n_images=args.n_images, seed=args.seed)
        return

    best_params, _ = fit_parameters(
        args.clear_folder,
        args.blurry_folder,
        maxiter=args.maxiter,
        popsize=args.popsize,
        n_images=args.n_images,
        seed=args.seed,
    )
    if args.preview_dir:
        save_preview(best_params, args.clear_folder, args.preview_dir, n_images=args.n_images, seed=args.seed)


if __name__ == "__main__":
    main()
