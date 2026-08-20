"""
Interactive comparison of two local contrast enhancers on an HDR image.

Everything common lives here: loading the linear image, reducing it to log-luminance,
reapplying chroma afterwards, and the Dear PyGui interface. The two operators are
independent implementations that share only the log-luminance in / log-luminance out
interface.
"""
import time

import numpy as np
import cv2
import OpenEXR
import dearpygui.dearpygui as dpg

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

def reapply_chroma(rgb, luminance, enhanced_log_luminance, brightness=0.5):
    """
    Undo the log and put chroma back, as BGR ready for display.

    Each pixel keeps its original ratio to its own luminance, so only brightness is
    changed by the operator -- hue and saturation ride along untouched.

    Fine tuned normalization behavior for maximal visual impact - do not change unless you know what you are doing.
    """
    low, high = np.percentile(enhanced_log_luminance, [0.0, 100.0])
    enhanced_log_luminance01 = (enhanced_log_luminance - low) / (high - low)
    enhanced_log_luminance01 = np.clip(enhanced_log_luminance01, 0.0, 1.0)

    enhanced_luminance = np.exp(enhanced_log_luminance01 / (1.0 - brightness))
    trim_percent = 1.0
    low, high = np.percentile(enhanced_luminance, [trim_percent, 100.0 - trim_percent])
    enhanced_luminance01 = (enhanced_luminance - low) / (high - low)
    enhanced_luminance01 = np.clip(enhanced_luminance01, 0.0, 1.0)
    enhanced_rgb = rgb * (enhanced_luminance01 / luminance)[:, :, None]

    return np.clip(np.ascontiguousarray(enhanced_rgb[:, :, ::-1]), 0.0, 1.0)


# --- Interface ---------------------------------------------------------------------

IMAGE_PATH = "run/test.exr"
TEXTURE = "preview_texture"
PANEL_WIDTH = 340

# Transducer sliders scale these, so capture them before anything is overwritten
TRANSDUCER_BASE = (MantiukContrastEqualization.TRANSDUCER_EPS,
                   MantiukContrastEqualization.TRANSDUCER_EXPONENT,
                   MantiukContrastEqualization.TRANSDUCER_MUL)

state = {}


def build_controls():
    """Every knob, grouped. Values are read off the widgets when recomputing."""
    dpg.add_radio_button(("Mantiuk contrast equalization", "Edge-aware AHE"),
                         tag="operator", default_value="Mantiuk contrast equalization")

    with dpg.collapsing_header(label="Mantiuk", default_open=True):
        dpg.add_slider_float(label="strength", tag="strength", default_value=8.0,
                             min_value=0.0, max_value=10.0)
        dpg.add_slider_int(label="iterations", tag="iterations", default_value=50,
                           min_value=10, max_value=500)
        dpg.add_slider_float(label="target contrast", tag="target_contrast",
                             default_value=0.0002, min_value=0.0002, max_value=0.1,
                             format="%.4f")
        # 1.0 would divide by zero in reapply_chroma, so the slider stops short of it
        dpg.add_slider_float(label="brightness", tag="brightness", default_value=0.002,
                             min_value=0.0, max_value=0.95, format="%.3f")

    with dpg.collapsing_header(label="Transducer (multiples of base)", default_open=True):
        for tag, label in (("t_eps", "eps"), ("t_exponent", "exponent"), ("t_mul", "mul")):
            dpg.add_slider_float(label=label, tag=tag, default_value=1.0,
                                 min_value=0.5, max_value=2.0)
        dpg.add_text("", tag="transducer_values", wrap=PANEL_WIDTH - 30)

    with dpg.collapsing_header(label="Edge-aware AHE", default_open=False):
        dpg.add_slider_int(label="s_spatial", tag="s_spatial", default_value=8,
                           min_value=1, max_value=64)
        dpg.add_slider_int(label="num_levels", tag="num_levels", default_value=32,
                           min_value=2, max_value=64)
        dpg.add_slider_int(label="window px", tag="window_px", default_value=112,
                           min_value=8, max_value=400)
        dpg.add_checkbox(label="edge-aware blur", tag="guided", default_value=True)
        dpg.add_slider_float(label="log10 eps", tag="guided_eps", default_value=-4.0,
                             min_value=-4.0, max_value=-1.0)

    dpg.add_separator()
    dpg.add_button(label="Save PNG", callback=save_png, width=-1)
    dpg.add_text("", tag="status", wrap=PANEL_WIDTH - 30)


CONTROLS = ("operator", "strength", "iterations", "target_contrast", "brightness",
            "t_eps", "t_exponent", "t_mul", "s_spatial", "num_levels", "window_px",
            "guided", "guided_eps")


def current_parameters():
    """Everything the result depends on, as one comparable tuple."""
    return tuple(dpg.get_value(tag) for tag in CONTROLS)


def any_control_active():
    """True while a slider is being dragged. Dear PyGui only reports this per item."""
    return any(dpg.is_item_active(tag) for tag in CONTROLS)


def recompute():
    """Run the selected operator and push the result into the preview texture."""
    started = time.time()

    if dpg.get_value("operator").startswith("Mantiuk"):
        scales = (dpg.get_value("t_eps"), dpg.get_value("t_exponent"), dpg.get_value("t_mul"))
        (MantiukContrastEqualization.TRANSDUCER_EPS,
         MantiukContrastEqualization.TRANSDUCER_EXPONENT,
         MantiukContrastEqualization.TRANSDUCER_MUL) = [
            base * scale for base, scale in zip(TRANSDUCER_BASE, scales)]
        dpg.set_value("transducer_values", "eps %.5f    exponent %.5f    mul %.5f" % (
            MantiukContrastEqualization.TRANSDUCER_EPS,
            MantiukContrastEqualization.TRANSDUCER_EXPONENT,
            MantiukContrastEqualization.TRANSDUCER_MUL))

        enhanced_log = MantiukContrastEqualization.enhance(
            state["log_luminance"],
            strength=dpg.get_value("strength"),
            iterations=dpg.get_value("iterations"),
            target_contrast=dpg.get_value("target_contrast"),
            verbose=True)
    else:
        guided_eps = 10.0 ** dpg.get_value("guided_eps") if dpg.get_value("guided") else None
        enhanced_log = EdgeAwareAHE.enhance(
            state["log_luminance"],
            s_spatial=dpg.get_value("s_spatial"),
            num_levels=dpg.get_value("num_levels"),
            guided_eps=guided_eps,
            window_px=dpg.get_value("window_px"))

    bgr = reapply_chroma(state["rgb"], state["luminance"], enhanced_log,
                         dpg.get_value("brightness"))
    state["bgr"] = bgr

    # The texture wants RGBA floats; reapply_chroma hands back BGR for OpenCV's benefit
    rgba = np.empty((*bgr.shape[:2], 4), dtype=np.float32)
    rgba[:, :, :3] = bgr[:, :, ::-1]
    rgba[:, :, 3] = 1.0
    dpg.set_value(TEXTURE, rgba.ravel())

    dpg.set_value("status", "recomputed in %.2fs" % (time.time() - started))


def save_png():
    if state.get("bgr") is not None:
        cv2.imwrite("run/output_enhanced.png",
                    np.clip(state["bgr"] * 255.0, 0, 255).astype(np.uint8))
        dpg.set_value("status", "saved run/output_enhanced.png")


if __name__ == "__main__":
    rgb_img = load_linear_rgb(IMAGE_PATH)
    log_luminance, luminance = to_log_luminance(rgb_img)
    height, width = rgb_img.shape[:2]
    state.update(rgb=rgb_img, log_luminance=log_luminance, luminance=luminance, bgr=None)

    dpg.create_context()
    dpg.create_viewport(title="Local Contrast", width=width + PANEL_WIDTH + 40,
                        height=height + 60)

    with dpg.texture_registry():
        dpg.add_raw_texture(width, height, np.zeros(width * height * 4, dtype=np.float32),
                            format=dpg.mvFormat_Float_rgba, tag=TEXTURE)

    with dpg.window(tag="root"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=PANEL_WIDTH, autosize_y=True):
                build_controls()
            dpg.add_image(TEXTURE)

    dpg.set_primary_window("root", True)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    parameters = None
    while dpg.is_dearpygui_running():
        # Recompute once the user lets go of a slider, not on every frame of a drag
        if parameters != current_parameters() and not any_control_active():
            parameters = current_parameters()
            recompute()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()
