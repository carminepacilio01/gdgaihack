"""Helpers for converting (pixel u, v, depth_mm) → real-world (X, Y, Z) in mm
using OAK camera intrinsics. Used by HypomimiaNode to measure facial amplitudes
in millimeters rather than pixels.
"""
import numpy as np


def get_intrinsics(calib, socket, width, height):
    """Returns 3x3 intrinsic matrix for the given socket at the given resolution."""
    K = np.array(calib.getCameraIntrinsics(socket, width, height))
    return K


def pixel_to_world_mm(u, v, depth_mm, K):
    """Back-project a single pixel (u, v) with known depth (mm) to 3D mm.

    Returns (X, Y, Z) in millimeters in the camera frame.
    """
    if depth_mm <= 0 or np.isnan(depth_mm):
        return None
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * depth_mm / fx
    Y = (v - cy) * depth_mm / fy
    return (float(X), float(Y), float(depth_mm))


def euclid_mm(p1, p2):
    """Euclidean distance between two 3D mm points. Returns None if either is None."""
    if p1 is None or p2 is None:
        return None
    return float(
        np.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
        )
    )


def safe_depth_at(depth_frame, u, v, window=3):
    """Median depth in a small window around (u, v). Robust to noisy single pixels.

    depth_frame: HxW numpy array of uint16 depth values in millimeters.
    Returns float depth in mm, or NaN if no valid pixels in the window.
    """
    if depth_frame is None:
        return float("nan")
    h, w = depth_frame.shape[:2]
    u, v = int(round(u)), int(round(v))
    if u < 0 or v < 0 or u >= w or v >= h:
        return float("nan")
    u0, u1 = max(0, u - window), min(w, u + window + 1)
    v0, v1 = max(0, v - window), min(h, v + window + 1)
    patch = depth_frame[v0:v1, u0:u1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))
