"""
Contrast equalization after Mantiuk, Myszkowski & Seidel 2006,
"A Perceptual Framework for Contrast Processing of High Dynamic Range Images".

The image is described not by its luminances but by the contrasts between neighbouring
pixels, taken at every level of a Gaussian pyramid (Eq. 6):

    G[k][i,j] = x[k][i] - x[k][j]      x = log10 luminance, j a neighbour of i

Those contrasts are modified (here: their magnitude histogram is equalized, Eq. 17-19),
which leaves a set of contrasts that no image satisfies exactly. The output image is the
one that fits them best in the weighted least-squares sense (Eq. 7), found by setting the
derivatives to zero (Eq. 21) and solving the resulting system A x = B (Eq. 22). The
weights (Eq. 8) come from contrast discrimination thresholds, so a mismatch costs more
where the eye is more sensitive.

Because contrasts are manipulated rather than absolute levels, there is no spatially
varying offset that could form a halo.

The paper solves A x = B with the biconjugate gradient method (Numerical Recipes 2.7),
chosen over multigrid because considering contrast at all pyramid levels leaves A dense.
That is what `_biconjugate_gradient` below implements.
"""
import numpy as np
import cv2

LOG10 = float(np.log(10.0))


def _discrimination_threshold(contrast):
    """
    Simplified contrast discrimination threshold (Eq. 5), fitted to Whittle's data.

    The smallest change in contrast the eye can detect at a given contrast. Eq. 9 asks
    for this simplified fit rather than the full model of Eq. 4 precisely because it
    overestimates sensitivity at low contrast: that keeps the weight p large near zero
    contrast, which is what stops the solution from reversing contrast polarity (the
    failure mode that produces halos in gradient-domain methods, the paper's Figure 7).
    """
    return 0.038737 * contrast ** 0.537756

TRANSDUCER_EPS = 0.0160000008
TRANSDUCER_EXPONENT = 0.418500036
TRANSDUCER_MUL = 0.646000028

def _transducer(contrast):
    """
    Response of the visual system to a contrast, in JND units (Eq. 14).

    MY OWN analytical approximation to the transducer of Eq. 13, whose defining property is
    that the response changes by one unit per just noticeable difference. Equalizing in
    this space rather than in contrast is what makes "equal magnitude" mean "equally
    visible", and it bounds the gain of Eq. 19 without any need for a cap.
    """
    return ((np.abs(contrast) + TRANSDUCER_EPS) ** TRANSDUCER_EXPONENT - TRANSDUCER_EPS ** TRANSDUCER_EXPONENT) * TRANSDUCER_MUL
    #acontrast = np.abs(contrast)
    #return acontrast / (acontrast + 1.0)
    #return np.log(1.0+acontrast / TRANSDUCER_EPS) * TRANSDUCER_MUL

def _inverse_transducer(response):
    """Back from response to contrast; the inverse of Eq. 14."""
    return (np.abs(response) / TRANSDUCER_MUL + TRANSDUCER_EPS ** TRANSDUCER_EXPONENT) ** (1.0 / TRANSDUCER_EXPONENT) - TRANSDUCER_EPS
    #aresponse = np.abs(response)
    #return aresponse / (1.0 - aresponse)
    #return np.expm1(aresponse / TRANSDUCER_MUL) * TRANSDUCER_EPS

def _contrast_weights(gx, gy):
    """
    Per-contrast weighting p from Eq. 8: the sensitivity, i.e. 1 / threshold.

    A mismatch between the contrast we asked for and the contrast the solution actually
    has is penalized in proportion to how visible it would be. High contrasts sit where
    the eye discriminates poorly, so they are cheap to violate and end up compressed --
    which is what stops reconstruction from blowing the dynamic range wide open.
    """
    threshold = lambda g: _discrimination_threshold(np.maximum(np.abs(g), 0.001))
    return 1.0 / threshold(gx), 1.0 / threshold(gy)


def _downsample(image):
    """One Gaussian pyramid step, as a 2x2 average so the adjoint below is exact."""
    h, w = image.shape
    h, w = h - h % 2, w - w % 2
    return 0.25 * (image[0:h:2, 0:w:2] + image[1:h:2, 0:w:2] +
                   image[0:h:2, 1:w:2] + image[1:h:2, 1:w:2])


def _downsample_adjoint(coarse, shape):
    """Transpose of _downsample: spread each value back over its 2x2 block."""
    out = np.zeros(shape, dtype=np.float32)
    h, w = shape
    h, w = h - h % 2, w - w % 2

    quarter = 0.25 * coarse
    out[0:h:2, 0:w:2] += quarter
    out[1:h:2, 0:w:2] += quarter
    out[0:h:2, 1:w:2] += quarter
    out[1:h:2, 1:w:2] += quarter
    return out


def _contrasts(level):
    """Differences to the right and lower neighbour -- one value per pixel pair."""
    return level[:, 1:] - level[:, :-1], level[1:, :] - level[:-1, :]


def _contrasts_adjoint(gx, gy, shape):
    """Transpose of _contrasts: accumulate each pair's value onto both its pixels."""
    out = np.zeros(shape, dtype=np.float32)
    out[:, :-1] -= gx
    out[:, 1:] += gx
    out[:-1, :] -= gy
    out[1:, :] += gy
    return out


def _pyramid(image, levels):
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(_downsample(pyramid[-1]))
    return pyramid


def _accumulate_up(per_level, shapes):
    """
    Sum per-level terms at full resolution through the transposed pyramid.

    The unknowns are the finest level only; coarser levels are functions of it, so each
    level's contribution reaches the solution through the transpose of the downsampling
    chain. Accumulating coarse-to-fine keeps that linear in the number of levels.
    """
    total = per_level[-1]
    for level in range(len(per_level) - 2, -1, -1):
        total = per_level[level] + _downsample_adjoint(total, shapes[level])
    return total


def _apply_A(x, weights):
    """Left hand side of Eq. 22: weighted pyramid Laplacian, summed over levels."""
    pyramid = _pyramid(x, len(weights))
    terms = []
    for level, (px, py) in zip(pyramid, weights):
        gx, gy = _contrasts(level)
        terms.append(_contrasts_adjoint(px * gx, py * gy, level.shape))

    return _accumulate_up(terms, [level.shape for level in pyramid])


def _diagonal(weights, shapes):
    """
    Diagonal of A, for Jacobi preconditioning.

    Probing A with a single pixel gives a closed form: at level k that pixel lands in one
    cell with coefficient (1/4)^k, and only the edges touching that cell contribute, so
    its diagonal entry is (1/16)^k times the sum of their weights. Pre-scaling each level
    by (1/4)^k lets the same coarse-to-fine accumulation as everywhere else supply the
    other (1/4)^k.
    """
    terms = []
    for level, (px, py), shape in zip(range(len(weights)), weights, shapes):
        incident = np.zeros(shape, dtype=np.float32)
        incident[:, :-1] += px
        incident[:, 1:] += px
        incident[:-1, :] += py
        incident[1:, :] += py
        terms.append(incident * (0.25 ** level))

    return _accumulate_up(terms, shapes)


def _right_hand_side(modified, weights, shapes):
    """Right hand side of Eq. 22, built from the modified contrasts."""
    terms = [_contrasts_adjoint(px * gx, py * gy, shape)
             for (gx, gy), (px, py), shape in zip(modified, weights, shapes)]
    return _accumulate_up(terms, shapes)


def _biconjugate_gradient(apply_A, b, diagonal, iterations, tolerance):
    """
    Preconditioned biconjugate gradient (Numerical Recipes 2.7), as specified in the paper.

    A here is symmetric -- it is the Hessian of a quadratic objective -- so the shadow
    residual tracks the true one and this reduces to conjugate gradients. The two
    sequences are kept anyway so the method matches the paper's.

    The Eq. 9 weights span nearly two orders of magnitude, which leaves A badly
    conditioned; dividing by its diagonal (Jacobi preconditioning, the same choice
    Numerical Recipes makes) is what makes the iteration converge in practice.

    A is singular: adding a constant to every pixel changes no contrast. B is orthogonal
    to that null space, so the iteration converges to a solution defined up to an offset,
    which the caller normalizes away.
    """
    precondition = lambda v: v / diagonal

    x = np.zeros_like(b)
    residual = b - apply_A(x)
    shadow = residual.copy()

    direction = shadow_direction = None
    rho_previous = 1.0
    b_norm = float(np.linalg.norm(b)) + 1e-12

    for iteration in range(iterations):
        z = precondition(residual)
        # The preconditioner is diagonal, so it is its own transpose
        shadow_z = precondition(shadow)

        rho = float(np.sum(z * shadow))
        if abs(rho) < 1e-30:
            break

        if direction is None:
            direction = z.copy()
            shadow_direction = shadow_z.copy()
        else:
            beta = rho / rho_previous
            direction = z + beta * direction
            shadow_direction = shadow_z + beta * shadow_direction

        q = apply_A(direction)
        denominator = float(np.sum(shadow_direction * q))
        if abs(denominator) < 1e-30:
            break

        alpha = rho / denominator
        x += alpha * direction
        residual -= alpha * q
        # A is symmetric, so the transposed product is the same operator
        shadow -= alpha * apply_A(shadow_direction)

        if float(np.linalg.norm(residual)) / b_norm < tolerance:
            break

        rho_previous = rho

    return x, iteration + 1, float(np.linalg.norm(residual)) / b_norm


def _equalize_contrasts(pyramid, strength, max_gain, target_contrast, num_bins=1024):
    """
    Equalize the histogram of contrast magnitudes in response space (Eq. 17-19).

    Contrasts are first mapped through the transducer, so the histogram is one of
    perceived magnitudes. Each pixel's magnitude is the norm of its responses to all its
    neighbours (Eq. 18); the cumulative histogram evaluated at that magnitude becomes its
    new magnitude (Eq. 19). Small responses are the numerous ones, so they occupy the
    steep part of the CDF and are amplified; large ones are compressed. The result is
    mapped back to contrast by the inverse transducer.
    """
    levels = []
    for level in pyramid:
        gx, gy = _contrasts(level)
        rx = np.sign(gx) * _transducer(gx)
        ry = np.sign(gy) * _transducer(gy)

        # Norm over the four neighbours: each pixel owns the pairs on either side of it
        squares = np.zeros_like(level)
        squares[:, :-1] += rx * rx
        squares[:, 1:] += rx * rx
        squares[:-1, :] += ry * ry
        squares[1:, :] += ry * ry
        levels.append((rx, ry, np.sqrt(squares)))

    magnitudes = np.concatenate([m.ravel() for _, _, m in levels])

    # Ignore the extreme tail so a handful of pixels cannot set the scale
    high = float(np.percentile(magnitudes, 99.9))
    histogram, edges = np.histogram(magnitudes, bins=num_bins, range=(0.0, high))
    cdf = np.cumsum(histogram).astype(np.float32)
    cdf /= cdf[-1]

    # Eq. 19 rescales each pixel's responses to the new magnitude: a gain of
    # CPDF(|R|) / |R|. Responses are in JND units and never far below 1, so unlike the
    # same expression in contrast space this stays bounded on its own.
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(np.float32)
    gain_table = np.divide(cdf, centers, out=np.zeros_like(centers), where=centers > 0)
    gain_table = 1.0 + strength * (np.clip(gain_table, 0.0, max_gain) - 1.0)
    
    modified = []
    for rx, ry, magnitude in levels:
        gain = np.interp(magnitude, centers, gain_table).astype(np.float32)

        # A pair is shared by two pixels with two different gains. Both appear in the
        # objective, and a least-squares fit to both equals a single fit to their mean.
        rx = target_contrast * rx * (gain[:, :-1] + gain[:, 1:]) / 2.0
        ry = target_contrast * ry * (gain[:-1, :] + gain[1:, :]) / 2.0

        modified.append((np.sign(rx) * _inverse_transducer(rx),
                         np.sign(ry) * _inverse_transducer(ry)))

    return modified

def enhance(log_luminance, strength=1.0, max_gain=8.0, levels=None,
            iterations=150, tolerance=1e-4,
            target_contrast=50.0, brightness=0.5, verbose=False):
    """
    Equalize contrast. Takes and returns log-luminance.

    Parameters:
        strength (float): 0 leaves the image untouched, 1 applies full equalization.
        max_gain (float): Ceiling on the amplification of any single contrast.
        levels (int): Pyramid depth; defaults to as deep as the image allows.
        iterations (int): Cap on biconjugate gradient iterations.
        tolerance (float): Relative residual at which the solver stops early.
        target_contrast (float): Target contrast for the equalization.
        verbose (bool): Whether to print progress information.
    """
    # The paper works in log10 units, and Eq. 19 puts the output on that scale too
    x = (log_luminance / LOG10).astype(np.float32)

    if levels is None:
        levels = max(int(np.log2(min(x.shape))) - 2, 1)

    pyramid = _pyramid(x, levels)
    shapes = [level.shape for level in pyramid]

    # Weights come from the ORIGINAL contrasts, as in Eq. 8
    weights = [_contrast_weights(*_contrasts(level)) for level in pyramid]

    modified = _equalize_contrasts(pyramid, strength, max_gain, target_contrast)
    b = _right_hand_side(modified, weights, shapes)

    solution, used, residual = _biconjugate_gradient(
        lambda v: _apply_A(v, weights), b, _diagonal(weights, shapes), iterations, tolerance)
    if verbose:
        print(f"  biconjugate gradient: {used} iterations, relative residual {residual:.2e}")

    return solution
