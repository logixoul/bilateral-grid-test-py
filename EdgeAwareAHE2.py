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
import cv2

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


def shadow_weight(normalized, shadow_guard, smooth_px=4.0, bins=256):
    """
    How much of the equalized result to keep, per pixel: 0 in the shadows, 1 above them.

    Equalization maps every region's own distribution onto the whole output range, so a
    region's median comes out near mid grey however many stops down it started. That is
    the mechanism working as designed, but for the darkest parts of a picture it is the
    wrong answer twice over: they stop reading as shadows at all, and lifting them by
    several stops multiplies their noise by the same factor -- chroma noise included,
    since chroma rides along as a ratio to luminance.

    Fading back to the original tone there keeps shadows dark and leaves their noise
    where it was. The fade is placed by RANK rather than by tone value, so shadow_guard
    is the fraction of the image treated as shadow and lands in the same place on every
    image; log-normalizing pins the darkest pixels at 0 and packs the rest into the top
    of the range, so a threshold on the value itself does not transfer between images.

    The tone is blurred first so the fade follows the exposure of a region rather than of
    a single pixel, which also stops the noise from modulating its own protection.
    """
    tone = normalized.astype(np.float32)
    if smooth_px > 0.0:
        tone = cv2.GaussianBlur(tone, (0, 0), sigmaX=smooth_px, sigmaY=smooth_px)

    # Rank of each pixel in the tone distribution, via its cumulative histogram
    histogram, edges = np.histogram(tone, bins=bins, range=(0.0, 1.0))
    cumulative = np.cumsum(histogram).astype(np.float32)
    cumulative /= cumulative[-1]
    rank = np.interp(tone, (edges[:-1] + edges[1:]) / 2.0, cumulative).astype(np.float32)

    ramp = np.clip(rank / max(shadow_guard, 1e-6), 0.0, 1.0)
    return (ramp * ramp * (3.0 - 2.0 * ramp)).astype(np.float32)   # smoothstep


def _limit_contrast(cdf, clip_limit, passes=3):
    """
    Cap how steep the local transfer function is allowed to get (CLAHE's clip limit).

    The differences between neighbouring CDF levels ARE the local histogram, and the
    slope they describe is exactly the factor by which the operator will amplify
    contrast at that pixel. In a flat region nearly all the local weight lands in one
    level, that level's step approaches 1, and the slope explodes -- which is the same
    thing as saying the noise in the region gets stretched over the whole output range.

    Capping each step and handing the excess back to the other levels bounds that
    amplification directly, rather than inferring it from a proxy like local variance.
    The cap is relative to a flat histogram, so clip_limit=1 forces every step to 1/K --
    a straight ramp, i.e. the identity mapping -- and a large limit never binds and
    leaves full equalization. Redistribution can lift a step back over the cap, so a few
    passes settle it.
    """
    steps = np.diff(cdf, axis=0, prepend=0.0)

    # clip_limit is either one number for the whole image or a per-pixel map, in which
    # case it broadcasts across the levels of each pixel's CDF
    cap = np.asarray(clip_limit, dtype=np.float32) / cdf.shape[0]
    if cap.ndim == 2:
        cap = cap[None, :, :]

    for _ in range(passes):
        excess = np.maximum(steps - cap, 0.0).sum(axis=0, keepdims=True)
        if float(excess.max()) <= 0.0:
            break
        # Every level gets an equal share back, so the total still comes to 1
        steps = np.minimum(steps, cap) + excess / cdf.shape[0]

    return np.cumsum(steps, axis=0, dtype=np.float32)


def local_cdf_equalize(image, num_levels=48, sigma_spatial=112.0, sigma_range=0.6,
                       clip_limit=None, iterations=3):
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
        clip_limit (float): Ceiling on local contrast amplification, as a multiple of the
            identity mapping. 1 is the identity, None leaves equalization unlimited.
        iterations (int): Recursive filter passes; 3 is the paper's recommendation.
    """
    # Levels sit at bin centres, so the first and last carry half a bin like the rest
    levels = (np.arange(num_levels, dtype=np.float32) + 0.5) / num_levels

    indicators = (image[None, :, :] < levels[:, None, None]).astype(np.float32)
    cdf = domain_transform_filter(indicators, image, sigma_spatial, sigma_range, iterations)

    # Guard the CDF's shape against filter round-off rather than against the method
    np.clip(cdf, 0.0, 1.0, out=cdf)
    np.maximum.accumulate(cdf, axis=0, out=cdf)

    if clip_limit is not None:
        cdf = _limit_contrast(cdf, clip_limit)

    # Evaluate the per-pixel CDF at the pixel's own value
    position = np.clip(image * num_levels - 0.5, 0.0, num_levels - 1.0)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, num_levels - 1)
    fraction = (position - lower).astype(np.float32)

    below = np.take_along_axis(cdf, lower[None], axis=0)[0]
    above = np.take_along_axis(cdf, upper[None], axis=0)[0]

    return below * (1.0 - fraction) + above * fraction


def enhance(log_luminance, num_levels=48, sigma_spatial=112.0, sigma_range=0.6,
            clip_limit=None, shadow_guard=0.0, iterations=3):
    """
    Equalize log-luminance locally. Takes and returns log-luminance in the same range.

    The operator works on [0, 1], so the log range is normalized on the way in and the
    equalized result is mapped back onto that same range on the way out -- luminance is
    redistributed within the scene's dynamic range rather than rescaled.

    With shadow_guard above 0 the darkest part of the image keeps its original tone
    instead of being equalized up into the midtones -- see shadow_weight.
    """
    log_min = float(log_luminance.min())
    log_range = float(log_luminance.max()) - log_min

    normalized = ((log_luminance - log_min) / log_range).astype(np.float32)

    equalized = local_cdf_equalize(normalized, num_levels=num_levels,
                                   sigma_spatial=sigma_spatial, sigma_range=sigma_range,
                                   clip_limit=clip_limit, iterations=iterations)

    if shadow_guard > 0.0:
        keep = shadow_weight(normalized, shadow_guard)
        equalized = normalized + keep * (equalized - normalized)

    return equalized * log_range + log_min
