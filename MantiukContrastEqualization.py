"""
Contrast equalization in the contrast domain (after Mantiuk et al. 2006).

Instead of deriving a tone curve from a spatial window, this operator decomposes
log-luminance into multiresolution contrasts, equalizes the histogram of contrast
MAGNITUDES, and reconstructs. Because it manipulates differences rather than absolute
levels, there is no spatially varying offset that could form a halo.

Deviation from the paper: contrasts here are Laplacian pyramid coefficients rather than
contrasts between neighbouring pixels, so reconstruction is an exact pyramid collapse
instead of a least-squares (Poisson) solve.
"""
import numpy as np
import cv2


def _laplacian_pyramid(image, levels):
    """Decompose into band-pass levels plus a low-pass residual."""
    pyramid = []
    current = image
    for _ in range(levels):
        smaller = cv2.pyrDown(current)
        upsampled = cv2.pyrUp(smaller, dstsize=(current.shape[1], current.shape[0]))
        pyramid.append(current - upsampled)
        current = smaller
    pyramid.append(current)
    return pyramid


def _collapse(pyramid):
    """Exact inverse of _laplacian_pyramid."""
    current = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        current = cv2.pyrUp(current, dstsize=(band.shape[1], band.shape[0])) + band
    return current


def _equalization_gains(pyramid, max_gain, num_bins=1024):
    """
    Build the magnitude remapping curve from the histogram of contrast magnitudes.

    Equalizing that histogram spreads magnitudes uniformly across the available range,
    which lifts the many small contrasts and compresses the few large ones -- histogram
    equalization, but on contrast rather than luminance. Returned as a lookup table of
    per-magnitude gains so it can be applied to every band.
    """
    magnitudes = np.concatenate([np.abs(band).ravel() for band in pyramid[:-1]])

    # Ignore the extreme tail so a handful of coefficients cannot set the scale
    high = float(np.percentile(magnitudes, 99.9))
    histogram, edges = np.histogram(magnitudes, bins=num_bins, range=(0.0, high))

    cdf = np.cumsum(histogram).astype(np.float32)
    cdf /= cdf[-1]

    # The equalized magnitude for a bin is its CDF position scaled to the same range
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    gains = np.divide(cdf * high, centers, out=np.zeros_like(centers), where=centers > 0)

    return centers, np.clip(gains, 0.0, max_gain)


def enhance(log_luminance, strength=1.0, max_gain=8.0, levels=None):
    """
    Equalize contrast. Takes and returns log-luminance in the same range.

    Parameters:
        strength (float): 0 leaves the image untouched, 1 applies full equalization.
        max_gain (float): Ceiling on how much any single contrast may be amplified,
            which is what keeps the smallest contrasts (i.e. noise) from exploding.
        levels (int): Pyramid depth; defaults to as deep as the image allows.
    """
    if levels is None:
        levels = max(int(np.log2(min(log_luminance.shape))) - 3, 1)

    pyramid = _laplacian_pyramid(log_luminance, levels)
    centers, gains = _equalization_gains(pyramid, max_gain)

    # Blend each band's gain toward 1.0 so `strength` fades the effect out continuously
    for i, band in enumerate(pyramid[:-1]):
        gain = np.interp(np.abs(band), centers, gains).astype(np.float32)
        pyramid[i] = band * (1.0 + strength * (gain - 1.0))

    equalized = _collapse(pyramid)

    # Reconstruction changes the range, so map it back onto the input's log range
    out_min, out_max = float(equalized.min()), float(equalized.max())
    log_min, log_max = float(log_luminance.min()), float(log_luminance.max())

    return (equalized - out_min) / (out_max - out_min) * (log_max - log_min) + log_min
