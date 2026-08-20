import numpy as np
import cv2

def bilateral_grid_lhe(image, s_spatial=16, num_levels=16):
    """
    Performs Local Histogram Equalization using a 3D Bilateral Grid.

    Parameters:
        image (ndarray): Grayscale float32 input image normalized between 0.0 and 1.0.
        s_spatial (int): Spatial downsampling factor (Grid width/height step).
        num_levels (int): Number of intensity bins in the grid (Grid depth).
    """
    h, w = image.shape

    # 1. Determine Grid Dimensions
    grid_h = int(np.ceil(h / s_spatial))
    grid_w = int(np.ceil(w / s_spatial))
    grid_d = num_levels

    # 2. Splatting: Populate the 3D local histogram grid
    # Each grid cell (z, y, x) acts as a local histogram bin
    grid = np.zeros((grid_d, grid_h, grid_w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            intensity = image[y, x]
            # Map pixel coordinates to grid coordinates
            gx = min(int(x / s_spatial), grid_w - 1)
            gy = min(int(y / s_spatial), grid_h - 1)
            gz = min(int(intensity * grid_d), grid_d - 1)
            grid[gz, gy, gx] += 1.0

    # 3. Blurring: Smooth the histograms spatially to prevent blocking artifacts
    # We blur each intensity slice independently using a Gaussian filter
    blurred_grid = np.zeros_like(grid)
    for z in range(grid_d):
        # Match sigma to spatial sampling rate to maintain uniform support
        sigma = s_spatial / 2.0
        blurred_grid[z, :, :] = cv2.GaussianBlur(grid[z, :, :], (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 4. Compute Cumulative Histograms (CDF) along the intensity axis (Z)
    cdf_grid = np.cumsum(blurred_grid, axis=0)

    # Normalize the CDF profiles within each local spatial column
    total_pixels_grid = cdf_grid[-1, :, :] + 1e-5 # Avoid zero-division
    normalized_cdf_grid = (cdf_grid / total_pixels_grid).astype(np.float32)

    # 5. Slicing: Sample the 3D CDF back onto the 2D image coordinates
    output_image = np.zeros_like(image, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            intensity = image[y, x]

            # Find the floating point positions in the grid
            fx = x / s_spatial
            fy = y / s_spatial
            fz = intensity * grid_d

            # Truncate index boundaries safely for trilinear interpolation
            x0 = int(np.floor(fx))
            x1 = min(x0 + 1, grid_w - 1)
            y0 = int(np.floor(fy))
            y1 = min(y0 + 1, grid_h - 1)
            z0 = min(int(np.floor(fz)), grid_d - 1)
            z1 = min(z0 + 1, grid_d - 1)

            # Compute interpolation weights
            wx = fx - x0
            wy = fy - y0
            wz = fz - z0

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

            # Interpolate Z axis
            interpolated_cdf = c0 * (1 - wz) + c1 * wz

            # The normalized CDF (0.0 to 1.0) is already the output value
            output_image[y, x] = interpolated_cdf

    return np.clip(output_image, 0.0, 1.0)

# --- Example Usage Run ---
if __name__ == "__main__":
    # Load an RGB image and equalize only its "value" channel in HSV space
    bgr_img = cv2.imread("run/test.jpg", cv2.IMREAD_COLOR)
    bgr_img = cv2.resize(bgr_img, dsize=None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)  # Resize for faster processing
    if bgr_img is None:
        raise FileNotFoundError("Could not load 'run/test.jpg'")

    # Work in float32 throughout: BGR in [0, 1], HSV with H in [0, 360], S/V in [0, 1]
    bgr_img = bgr_img.astype(np.float32) / 255.0

    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv_img)

    # Run Local Histogram Equalization on the bilateral grid
    enhanced_val = bilateral_grid_lhe(val, s_spatial=16, num_levels=16)

    # Reapply the original hue and saturation
    enhanced_img = cv2.cvtColor(cv2.merge([hue, sat, enhanced_val]), cv2.COLOR_HSV2BGR)

    # Save output to inspect contrast adaptation near boundaries
    cv2.imshow("result", np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()