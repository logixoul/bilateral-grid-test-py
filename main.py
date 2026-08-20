"""
Interactive comparison of two local contrast enhancers on an HDR image.

Everything common lives here: loading the linear image, reducing it to log-luminance,
reapplying chroma afterwards, and the trackbar UI. The two operators are independent
implementations that share only the log-luminance in / log-luminance out interface.
"""
import numpy as np
import cv2
import OpenEXR

import EdgeAwareAHE
import MantiukContrastEqualization

NTSC_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load_linear_rgb(path):
    """Load a linear HDR image, normalized by its single brightest R, G or B sample."""
    rgb = OpenEXR.File(path).channels()["RGB"].pixels.astype(np.float32)
    scale_factor = 1000 / rgb.shape[1]  # scale to 1000px wide for speed
    rgb = cv2.resize(rgb, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return np.clip(rgb / rgb.max(), 0.0, 1.0)


def to_log_luminance(rgb):
    """NTSC-weighted luminance in the log domain, plus the linear luminance itself."""
    luminance = np.maximum(rgb @ NTSC_WEIGHTS, 1e-6)
    return np.log(luminance), luminance

def _smoothstep(edge0, edge1, x):
    # Scale, clamp and check limits
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Apply smoothstep formula
    return x * x * (3 - 2 * x)

def reapply_chroma(rgb, luminance, enhanced_log_luminance):
    """
    Undo the log and put chroma back, as BGR ready for display.

    Each pixel keeps its original ratio to its own luminance, so only brightness is
    changed by the operator -- hue and saturation ride along untouched.

    Fine tuned normalization behavior for maximal visual impact - do not change unless you know what you are doing.
    """
    low, high = np.percentile(enhanced_log_luminance, [0.0, 100.0])
    enhanced_log_luminance01 = (enhanced_log_luminance - low) / (high - low)
    enhanced_log_luminance01 = np.clip(enhanced_log_luminance01, 0.0, 1.0)

    enhanced_luminance = np.exp(enhanced_log_luminance01 * 4.0)
    trim_percent = 1.0
    low, high = np.percentile(enhanced_luminance, [trim_percent, 100.0 - trim_percent])
    enhanced_luminance01 = (enhanced_luminance - low) / (high - low)
    enhanced_luminance01 = np.clip(enhanced_luminance01, 0.0, 1.0)
    enhanced_rgb = rgb * (enhanced_luminance01 / luminance)[:, :, None]

    return np.clip(np.ascontiguousarray(enhanced_rgb[:, :, ::-1]), 0.0, 1.0)


if __name__ == "__main__":
    rgb_img = load_linear_rgb("run/test.exr")
    log_luminance, luminance = to_log_luminance(rgb_img)

    WINDOW = "Local Contrast"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # 0 = edge-aware AHE on a bilateral grid, 1 = Mantiuk-style contrast equalization
    cv2.createTrackbar("operator", WINDOW, 1, 1, lambda v: None)

    # AHE: trackbars store (value - 1), so the minimum is 1 and 2
    cv2.createTrackbar("ahe s_spatial", WINDOW, 8 - 1, 64 - 1, lambda v: None)
    cv2.createTrackbar("ahe num_levels", WINDOW, 32 - 2, 64 - 2, lambda v: None)
    # Guided blur off at 0, otherwise eps sweeps 1e-4 .. 1e-1 logarithmically
    cv2.createTrackbar("ahe guided_eps", WINDOW, 1, 100, lambda v: None)
    cv2.createTrackbar("ahe window_px", WINDOW, 112, 400, lambda v: None)

    # Mantiuk: strength as a percentage, gain ceiling stored as (value - 1)
    cv2.createTrackbar("mantiuk strength", WINDOW, 100, 100, lambda v: None)
    # The solver is the slow part; fewer iterations trade accuracy for responsiveness
    cv2.createTrackbar("mantiuk iters", WINDOW, 50, 500, lambda v: None)
    cv2.createTrackbar("mantiuk target contrast", WINDOW, 0, 500, lambda v: None)

    params = None
    while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
        operator = cv2.getTrackbarPos("operator", WINDOW)
        s_spatial = cv2.getTrackbarPos("ahe s_spatial", WINDOW) + 1
        num_levels = cv2.getTrackbarPos("ahe num_levels", WINDOW) + 2
        eps_pos = cv2.getTrackbarPos("ahe guided_eps", WINDOW)
        window_px = max(cv2.getTrackbarPos("ahe window_px", WINDOW), 8)
        strength = cv2.getTrackbarPos("mantiuk strength", WINDOW) / 10.0
        iterations = max(cv2.getTrackbarPos("mantiuk iters", WINDOW), 10)
        target_contrast = (cv2.getTrackbarPos("mantiuk target contrast", WINDOW) + 1) / (5000.0 + 1.0)
        # Recompute only when a slider actually moved
        current = (operator, s_spatial, num_levels, eps_pos, window_px,
                   strength, iterations, target_contrast)
        if current != params:
            params = current

            if operator == 0:
                guided_eps = None if eps_pos == 0 else 10.0 ** (-4.0 + 3.0 * (eps_pos - 1) / 99.0)
                blur = "gaussian" if guided_eps is None else f"guided eps={guided_eps:.5f}"
                print(f"AHE: s_spatial={s_spatial}, num_levels={num_levels}, "
                      f"window_px={window_px}, {blur}...")
                enhanced_log = EdgeAwareAHE.enhance(log_luminance, s_spatial=s_spatial,
                                                    num_levels=num_levels, guided_eps=guided_eps,
                                                    window_px=window_px)
            else:
                print(f"Mantiuk: strength={strength:.2f}, "
                      f"iterations={iterations}, target_contrast={target_contrast:.2f}...")
                enhanced_log = MantiukContrastEqualization.enhance(log_luminance,
                                                                   strength=strength,
                                                                   iterations=iterations,
                                                                   target_contrast=target_contrast,
                                                                   verbose=True)

            enhanced_img = reapply_chroma(rgb_img, luminance, enhanced_log)
            
            cv2.imshow(WINDOW, enhanced_img)

        # Esc or 'q' quits; 's' saves the currently displayed result
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            cv2.imwrite("run/output_enhanced.png", np.clip(enhanced_img * 255.0, 0, 255).astype(np.uint8))
            print("Saved 'run/output_enhanced.png'")

    cv2.destroyAllWindows()
