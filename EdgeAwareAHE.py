"""
Edge-aware adaptive histogram equalization on a bilateral grid.

Local histogram equalization in the sense of Chen et al.'s bilateral grid paper, with
one deviation: the per-slice spatial blur can be an edge-aware guided filter instead of
a Gaussian, which suppresses the halos that window-based local histogram methods produce
at strong luminance boundaries.
"""
import numpy as np
import cv2


def _guided_slice_blur(grid, guide, radius, eps):
    """
    Edge-aware replacement for the per-slice Gaussian blur (He et al. guided filter).

    Every intensity slice is filtered with the SAME guide (the log-luminance at grid
    resolution), so a cell's histogram is gathered mostly from cells on its own side
    of an edge instead of from a fixed round window. `eps` plays the role of a range
    sigma: small values follow edges tightly, large values degrade to a box blur.
    """
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    box = lambda a: cv2.boxFilter(a, -1, ksize)

    mean_g = box(guide)
    var_g = box(guide * guide) - mean_g * mean_g

    blurred = np.empty_like(grid)
    for z in range(grid.shape[0]):
        mean_p = box(grid[z])
        cov_gp = box(guide * grid[z]) - mean_g * mean_p
        a = cov_gp / (var_g + eps)
        b = mean_p - a * mean_g
        blurred[z] = box(a) * guide + box(b)

    # Guided-filter weights can go slightly negative; clamp so the CDF stays monotonic
    return np.maximum(blurred, 0.0)


def bilateral_grid_lhe(image, s_spatial=16, num_levels=16, guided_eps=None, window_px=None):
    """
    Performs Local Histogram Equalization using a 3D Bilateral Grid.

    Parameters:
        image (ndarray): Grayscale float32 input image normalized between 0.0 and 1.0.
        s_spatial (int): Spatial downsampling factor (Grid width/height step).
        num_levels (int): Number of intensity bins in the grid (Grid depth).
        guided_eps (float): If set, blur the slices with an edge-aware guided filter
            of this regularization instead of a Gaussian (halo suppression).
        window_px (float): Size of the histogram window in PIXELS. Independent of
            s_spatial, which is then purely a grid resolution knob.
    """
    h, w = image.shape

    # 1. Determine Grid Dimensions
    grid_h = int(np.ceil(h / s_spatial))
    grid_w = int(np.ceil(w / s_spatial))
    grid_d = num_levels

    # Floating point grid coordinates of every pixel, shared by splat and slice
    fx = np.arange(w, dtype=np.float32) / s_spatial
    fy = np.arange(h, dtype=np.float32) / s_spatial
    fz = image * grid_d

    # 2. Splatting: Populate the 3D local histogram grid
    # Each grid cell (z, y, x) acts as a local histogram bin
    grid = np.zeros((grid_d, grid_h, grid_w), dtype=np.float32)

    gx = np.minimum(fx.astype(np.int32), grid_w - 1)
    gy = np.minimum(fy.astype(np.int32), grid_h - 1)
    gz = np.minimum(fz.astype(np.int32), grid_d - 1)
    np.add.at(grid, (gz, gy[:, None], gx[None, :]), 1.0)

    # 3. Blurring: Smooth the histograms spatially to prevent blocking artifacts
    # The window is specified in pixels, so s_spatial only controls grid resolution
    if window_px is None:
        window_px = s_spatial * s_spatial / 2.0
    sigma = max(window_px / s_spatial, 1.0)

    if guided_eps is None:
        # We blur each intensity slice independently using a Gaussian filter
        blurred_grid = np.zeros_like(grid)
        for z in range(grid_d):
            blurred_grid[z, :, :] = cv2.GaussianBlur(grid[z, :, :], (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        # Same support, but the window now follows edges instead of being round
        guide = cv2.resize(image, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        blurred_grid = _guided_slice_blur(grid, guide, radius=sigma, eps=guided_eps)

    # 4. Compute Cumulative Histograms (CDF) along the intensity axis (Z)
    cdf_grid = np.cumsum(blurred_grid, axis=0)

    # Normalize the CDF profiles within each local spatial column
    total_pixels_grid = cdf_grid[-1, :, :] + 1e-5 # Avoid zero-division
    normalized_cdf_grid = (cdf_grid / total_pixels_grid).astype(np.float32)

    # 5. Slicing: Sample the 3D CDF back onto the 2D image coordinates
    # Truncate index boundaries safely for trilinear interpolation
    x0 = fx.astype(np.int32)
    x1 = np.minimum(x0 + 1, grid_w - 1)
    y0 = fy.astype(np.int32)
    y1 = np.minimum(y0 + 1, grid_h - 1)
    z0 = np.minimum(fz.astype(np.int32), grid_d - 1)
    z1 = np.minimum(fz.astype(np.int32) + 1, grid_d - 1)

    # Compute interpolation weights
    wx = (fx - x0)[None, :]
    wy = (fy - y0)[:, None]
    wz = fz - z0

    # Broadcast the per-axis indices into full (h, w) gather patterns
    x0, x1 = x0[None, :], x1[None, :]
    y0, y1 = y0[:, None], y1[:, None]

    def sample(volume):
        """Trilinear interpolation of one grid volume at every pixel."""
        c000 = volume[z0, y0, x0]
        c001 = volume[z0, y0, x1]
        c010 = volume[z0, y1, x0]
        c011 = volume[z0, y1, x1]
        c100 = volume[z1, y0, x0]
        c101 = volume[z1, y0, x1]
        c110 = volume[z1, y1, x0]
        c111 = volume[z1, y1, x1]

        # Interpolate X axis
        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx

        # Interpolate Y axis
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy

        # Interpolate Z axis
        return c0 * (1 - wz) + c1 * wz

    output_image = sample(normalized_cdf_grid)

    return np.clip(output_image, 0.0, 1.0).astype(np.float32)


def enhance(log_luminance, s_spatial=8, num_levels=32, guided_eps=1e-4, window_px=112):
    """
    Equalize log-luminance locally. Takes and returns log-luminance in the same range.

    The grid works on [0, 1], so the log range is normalized on the way in and the
    equalized result is mapped back onto that same range on the way out -- the operator
    redistributes luminance within the scene's dynamic range rather than rescaling it.
    """
    log_min = float(log_luminance.min())
    log_range = float(log_luminance.max()) - log_min

    normalized = (log_luminance - log_min) / log_range
    equalized = bilateral_grid_lhe(normalized, s_spatial=s_spatial, num_levels=num_levels,
                                   guided_eps=guided_eps, window_px=window_px)

    return equalized * log_range + log_min
