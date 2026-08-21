"""
Loading linear RGB from whatever the user drops on the window.

Every loader returns float32 RGB in scene-linear light, unnormalized -- HDR formats keep
whatever scale they were authored at, RAW files come back in sensor-linear units, and LDR
files are un-gamma'd so they at least sit in the same kind of space. Normalizing is the
caller's job, since that is a decision about display rather than about the file.
"""
import os

import numpy as np
import cv2

# LibRaw handles far more extensions than these, but these are the ones worth advertising
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".srf", ".sr2", ".dng", ".raf", ".orf",
                  ".rw2", ".pef", ".raw", ".mrw", ".dcr", ".kdc", ".x3f", ".erf", ".3fr",
                  ".mef", ".mos", ".nrw", ".iiq"}
HDR_EXTENSIONS = {".hdr", ".pic", ".rgbe"}
EXR_EXTENSIONS = {".exr"}
LDR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".ppm"}

SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | HDR_EXTENSIONS | EXR_EXTENSIONS | LDR_EXTENSIONS


def _load_exr(path):
    """
    OpenEXR, via the bindings -- this OpenCV build reports 'OpenEXR: NO'.

    Files store colour either as one interleaved RGB channel or as separate R, G and B,
    so handle both rather than assuming the layout of the files at hand.
    """
    import OpenEXR

    channels = OpenEXR.File(path).channels()

    for name in ("RGB", "RGBA"):
        if name in channels:
            return np.ascontiguousarray(channels[name].pixels[:, :, :3]).astype(np.float32)

    if all(name in channels for name in "RGB"):
        return np.stack([channels[name].pixels for name in "RGB"], axis=-1).astype(np.float32)

    # A single-channel or oddly named file: fall back to the first channel, greyscale
    first = next(iter(channels.values())).pixels.astype(np.float32)
    if first.ndim == 3:
        return np.ascontiguousarray(first[:, :, :3])
    return np.repeat(first[:, :, None], 3, axis=2)


def _load_radiance(path):
    """Radiance .hdr / .pic. OpenCV reads these natively (build reports 'HDR: YES')."""
    bgr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not read {os.path.basename(path)}")
    return np.ascontiguousarray(bgr[:, :, ::-1]).astype(np.float32)


def _load_raw(path):
    """
    Camera RAW through LibRaw.

    Demosaiced but otherwise untouched: gamma 1.0 and no auto-brightening, so what comes
    back is sensor-linear and the operators see the real scene contrast. Camera white
    balance is applied because without it the result has a strong colour cast.
    """
    import rawpy

    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(gamma=(1.0, 1.0), no_auto_bright=True, output_bps=16,
                              use_camera_wb=True)

    return rgb.astype(np.float32) / 65535.0


def _load_ldr(path):
    """Ordinary 8/16 bit images, undone from sRGB so they are linear like the rest."""
    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f"OpenCV could not read {os.path.basename(path)}")

    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = np.ascontiguousarray(bgr[:, :, :3][:, :, ::-1]).astype(np.float32)
    rgb /= 65535.0 if bgr.dtype == np.uint16 else 255.0

    # sRGB transfer curve, inverted
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load(path):
    """Load any supported file as linear float32 RGB, chosen by extension."""
    extension = os.path.splitext(path)[1].lower()

    if extension in EXR_EXTENSIONS:
        rgb = _load_exr(path)
    elif extension in HDR_EXTENSIONS:
        rgb = _load_radiance(path)
    elif extension in RAW_EXTENSIONS:
        rgb = _load_raw(path)
    elif extension in LDR_EXTENSIONS:
        rgb = _load_ldr(path)
    else:
        # Unknown extension: RAW is the likely intent, since LibRaw knows more formats
        # than the list above, but fall back to OpenCV if it turns out not to be one
        try:
            rgb = _load_raw(path)
        except Exception:
            rgb = _load_ldr(path)

    # Negative samples are meaningless here and break the log, whatever the source
    return np.maximum(np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
