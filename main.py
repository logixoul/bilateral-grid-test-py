import numpy as np
import cv2
import OpenEXR

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


def bilateral_grid_lhe(image, s_spatial=16, num_levels=16, guided_eps=None):
    """
    Performs Local Histogram Equalization using a 3D Bilateral Grid.

    Parameters:
        image (ndarray): Grayscale float32 input image normalized between 0.0 and 1.0.
        s_spatial (int): Spatial downsampling factor (Grid width/height step).
        num_levels (int): Number of intensity bins in the grid (Grid depth).
        guided_eps (float): If set, blur the slices with an edge-aware guided filter
            of this regularization instead of a Gaussian (halo suppression).
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
    # Match support to the spatial sampling rate to keep the window uniform
    sigma = s_spatial / 2.0

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
    z1 = np.minimum(z0 + 1, grid_d - 1)

    # Compute interpolation weights
    wx = (fx - x0)[None, :]
    wy = (fy - y0)[:, None]
    wz = fz - z0

    # Broadcast the per-axis indices into full (h, w) gather patterns
    x0, x1 = x0[None, :], x1[None, :]
    y0, y1 = y0[:, None], y1[:, None]

    # Trilinear Interpolation of CDF values
    c000 = normalized_cdf_grid[z0, y0, x0]
    c001 = normalized_cdf_grid[z0, y0, x1]
    c010 = normalized_cdf_grid[z0, y1, x0]
    c011 = normalized_cdf_grid[z0, y1, x1]
    c100 = normalized_cdf_grid[z1, y0, x0]
    c101 = normalized_cdf_grid[z1, y0, x1]
    c110 = normalized_cdf_grid[z1, y1, x0]
    c111 = normalized_cdf_grid[z1, y1, x1]

    # Interpolate X axis
    c00 = c000 * (1 - wx) + c001 * wx
    c01 = c010 * (1 - wx) + c011 * wx
    c10 = c100 * (1 - wx) + c101 * wx
    c11 = c110 * (1 - wx) + c111 * wx

    # Interpolate Y axis
    c0 = c00 * (1 - wy) + c01 * wy
    c1 = c10 * (1 - wy) + c11 * wy

    # Interpolate Z axis: the normalized CDF (0.0 to 1.0) is already the output value
    output_image = c0 * (1 - wz) + c1 * wz

    return np.clip(output_image, 0.0, 1.0).astype(np.float32)

# --- Example Usage Run ---
if __name__ == "__main__":
    # Load a linear HDR image and equalize its log-luminance only
    rgb_img = OpenEXR.File("run/ndk.exr").channels()["RGB"].pixels.astype(np.float32)

    # Normalize by the single brightest R, G or B sample so every channel lands in [0, 1]
    rgb_img = np.clip(rgb_img / rgb_img.max(), 0.0, 1.0)

    # Luminance via NTSC weights, equalized in the log domain (Mantiuk-style chroma handling)
    NTSC_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    luminance = np.maximum(rgb_img @ NTSC_WEIGHTS, 1e-6)
    log_luminance = np.log(luminance)

    # The grid expects [0, 1]; remember the log range so the result maps back onto it
    log_min = float(log_luminance.min())
    log_range = float(log_luminance.max()) - log_min
    norm_log_luminance = (log_luminance - log_min) / log_range

    # Interactive controls: trackbars store (value - 1), so the minimum is 1 and 2
    WINDOW = "Bilateral Grid LHE"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("s_spatial", WINDOW, 16 - 1, 64 - 1, lambda v: None)
    cv2.createTrackbar("num_levels", WINDOW, 16 - 2, 64 - 2, lambda v: None)
    # Guided blur off at 0, otherwise eps sweeps 1e-4 .. 1e-1 logarithmically
    cv2.createTrackbar("guided_eps", WINDOW, 0, 100, lambda v: None)

    params = None
    while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
        s_spatial = cv2.getTrackbarPos("s_spatial", WINDOW) + 1
        num_levels = cv2.getTrackbarPos("num_levels", WINDOW) + 2
        eps_pos = cv2.getTrackbarPos("guided_eps", WINDOW)
        guided_eps = None if eps_pos == 0 else 10.0 ** (-4.0 + 3.0 * (eps_pos - 1) / 99.0)

        # Recompute only when a slider actually moved
        if (s_spatial, num_levels, eps_pos) != params:
            params = (s_spatial, num_levels, eps_pos)
            blur = "gaussian" if guided_eps is None else f"guided eps={guided_eps:.5f}"
            print(f"Computing s_spatial={s_spatial}, num_levels={num_levels}, {blur}...")

            # Run Local Histogram Equalization on the bilateral grid
            enhanced_norm = bilateral_grid_lhe(norm_log_luminance, s_spatial=s_spatial,
                                               num_levels=num_levels, guided_eps=guided_eps)

            # Undo the log: back to a linear luminance spanning the original range
            enhanced_luminance = np.exp(enhanced_norm * log_range + log_min)

            # Reapply chroma by keeping each pixel's original ratio to its own luminance
            enhanced_rgb = rgb_img * (enhanced_luminance / luminance)[:, :, None]

            enhanced_img = np.clip(np.ascontiguousarray(enhanced_rgb[:, :, ::-1]), 0.0, 1.0)
            cv2.imshow(WINDOW, enhanced_img)

        # Esc or 'q' quits; 's' saves the currently displayed result
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            cv2.imwrite("run/output_enhanced.png", np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8))
            print("Saved 'run/output_enhanced.png'")

    cv2.destroyAllWindows()
