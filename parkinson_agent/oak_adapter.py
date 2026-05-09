"""Capture session data structures (face-only).

The capture layer is decoupled from depthai: it produces a `CaptureSession`
that signal_processing and the agent operate on. This means tests, demos and
the live OAK pipeline all share the same downstream code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaskWindow:
    """A labeled time window inside a capture session."""

    name: str
    start: float  # seconds from session start
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class FaceTimeSeries:
    """Per-frame MediaPipe FaceMesh landmarks (478 points, 3D normalized)."""

    timestamps: np.ndarray            # shape (N,)
    landmarks: np.ndarray             # shape (N, 478, 3)
    blink_events: list[float] = field(default_factory=list)  # blink onset times

    def coverage(self) -> float:
        """Fraction of frames with non-zero landmarks (i.e. face detected)."""
        if len(self.landmarks) == 0:
            return 0.0
        non_zero = np.any(self.landmarks != 0, axis=(1, 2))
        return float(np.mean(non_zero))

    def slice_window(self, start: float, end: float) -> "FaceTimeSeries":
        """Return a new FaceTimeSeries restricted to [start, end] seconds."""
        mask = (self.timestamps >= start) & (self.timestamps <= end)
        blinks = [b for b in self.blink_events if start <= b <= end]
        return FaceTimeSeries(
            timestamps=self.timestamps[mask],
            landmarks=self.landmarks[mask],
            blink_events=blinks,
        )


@dataclass
class CaptureSession:
    """One end-to-end capture run for a single patient."""

    session_id: str
    patient_id: str
    duration_s: float
    capture_fps: float
    tasks: list[TaskWindow] = field(default_factory=list)
    face: FaceTimeSeries | None = None

    def task(self, name: str) -> TaskWindow | None:
        for t in self.tasks:
            if t.name == name:
                return t
        return None

    def face_in_task(self, name: str) -> FaceTimeSeries | None:
        """Get the face time series clipped to the named task window."""
        if self.face is None:
            return None
        window = self.task(name)
        if window is None:
            return None
        return self.face.slice_window(window.start, window.end)
