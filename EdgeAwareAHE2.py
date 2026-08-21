"""
Edge-aware local histogram equalization built from local CDFs, at full resolution.

Where EdgeAwareAHE stores local histograms in a downsampled bilateral grid and slices
them back out, this builds the local CDF directly: for a set of threshold levels t_k
over log-luminance,

    C_k = DTF( 1[L < t_k],  guide = L )

is the edge-aware local fraction of pixels below t_k -- the local CDF sampled at t_k,
with weights that decay across edges rather than across space. The output at a pixel is
C interpolated at t = L(p).

Two properties of the domain transform filter (Gastal & Oliveira 2011) make this work
without any of the corrections the grid version needs: its weights are non-negative and
normalized, so every C_k lands in [0, 1], and the indicator functions are nested, so
C_k is automatically monotone in k. The mapping is therefore a valid CDF at every pixel
for free -- no clamping, no trilinear slicing, and no grid resolution to trade off.
"""
import numpy as np

SQRT2 = float(np.sqrt(2.0))


def _domain_transform(guide, sigma_spatial, sigma_range):
    """
    Accumulated distances of Gastal & Oliveira's domain transform (their Eq. 11).

    Walking along a row, each step costs 1 plus the guide's gradient scaled by
    sigma_spatial / sigma_range, so crossing an edge covers a lot of ground and the
    filter's weights fall off sharply there.
    """
    ratio = sigma_spatial / sigma_range

    horizontal = np.ones_like(guide)
    horizontal[:, 1:] += ratio * np.abs(np.diff(guide, axis=1))

    vertical = np.ones_like(guide)
    vertical[1:, :] += ratio * np.abs(np.diff(guide, axis=0))

    return horizontal, vertical


def _recursive_filter_axis(planes, distances, alpha):
    """
    One left-to-right then right-to-left pass along the last axis, in place.

    `planes` is (K, h, w): every threshold level is filtered in the same sweep, so the
    Python loop runs once per column rather than once per column per level. The filter
    interpolates towards the running value with weight a^distance, which has unit DC
    gain -- filtering a constant returns that constant, which is what makes each C_k a
    normalized weighted average.
    """
    weights = alpha ** distances

    for column in range(1, planes.shape[2]):
        planes[:, :, column] += weights[:, column] * (planes[:, :, column - 1] - planes[:, :, column])

    for column in range(planes.shape[2] - 2, -1, -1):
        planes[:, :, column] += weights[:, column + 1] * (planes[:, :, column + 1] - planes[:, :, column])

    return planes


def domain_transform_filter(planes, guide, sigma_spatial, sigma_range, iterations=3):
    """
    Edge-aware normalized filter, applied to every plane at once (RF mode).

    Successive iterations use a geometrically shrinking sigma, which is how the recursive
    filter approximates a Gaussian-shaped kernel rather than an exponential one.
    """
    horizontal, vertical = _domain_transform(guide, sigma_spatial, sigma_range)

    filtered = np.ascontiguousarray(planes, dtype=np.float32)
    for iteration in range(iterations):
        sigma_i = (sigma_spatial * SQRT2 ** (iterations - iteration - 1)
                   * np.sqrt(3.0 / (4.0 ** iterations - 1.0)))
        alpha = float(np.exp(-SQRT2 / sigma_i))

        _recursive_filter_axis(filtered, horizontal, alpha)

        # The vertical pass is the same sweep on the transpose
        transposed = np.ascontiguousarray(filtered.transpose(0, 2, 1))
        _recursive_filter_axis(transposed, vertical.T, alpha)
        filtered = np.ascontiguousarray(transposed.transpose(0, 2, 1))

    return filtered


def local_cdf_equalize(image, num_levels=48, sigma_spatial=112.0, sigma_range=0.6,
                       iterations=3):
    """
    Local histogram equalization by edge-aware local CDF.

    Parameters:
        image (ndarray): Grayscale float32 image normalized between 0.0 and 1.0.
        num_levels (int): Number of threshold levels the CDF is sampled at.
        sigma_spatial (float): Neighbourhood size in pixels.
        sigma_range (float): How much of a jump in the image counts as an edge, in the
            same [0, 1] units as the image. This is the character knob: below ~0.3 the
            filter follows every texture and the result turns grungy, above ~1 it stops
            respecting edges and halos come back.
        iterations (int): Recursive filter passes; 3 is the paper's recommendation.
    """
    # Levels sit at bin centres, so the first and last carry half a bin like the rest
    levels = (np.arange(num_levels, dtype=np.float32) + 0.5) / num_levels

    indicators = (image[None, :, :] < levels[:, None, None]).astype(np.float32)
    cdf = domain_transform_filter(indicators, image, sigma_spatial, sigma_range, iterations)

    # Guard the CDF's shape against filter round-off rather than against the method
    np.clip(cdf, 0.0, 1.0, out=cdf)
    np.maximum.accumulate(cdf, axis=0, out=cdf)

    # Evaluate the per-pixel CDF at the pixel's own value
    position = np.clip(image * num_levels - 0.5, 0.0, num_levels - 1.0)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, num_levels - 1)
    fraction = (position - lower).astype(np.float32)

    below = np.take_along_axis(cdf, lower[None], axis=0)[0]
    above = np.take_along_axis(cdf, upper[None], axis=0)[0]

    return below * (1.0 - fraction) + above * fraction


def enhance(log_luminance, num_levels=48, sigma_spatial=112.0, sigma_range=0.6,
            iterations=3):
    """
    Equalize log-luminance locally. Takes and returns log-luminance in the same range.

    The operator works on [0, 1], so the log range is normalized on the way in and the
    equalized result is mapped back onto that same range on the way out -- luminance is
    redistributed within the scene's dynamic range rather than rescaled.
    """
    log_min = float(log_luminance.min())
    log_range = float(log_luminance.max()) - log_min

    normalized = ((log_luminance - log_min) / log_range).astype(np.float32)
    equalized = local_cdf_equalize(normalized, num_levels=num_levels,
                                   sigma_spatial=sigma_spatial, sigma_range=sigma_range,
                                   iterations=iterations)

    return equalized * log_range + log_min
