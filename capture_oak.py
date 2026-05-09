"""OAK capture skeleton (face-only).

Drives the Luxonis OAK device through the patient task protocol, accumulates
FaceMesh landmark frames, and packages them into a CaptureSession the agent
can consume. The hand and pose pipelines from the original scaffold have
been removed — this hackathon track screens Parkinson's from the face only.

Wire-up depends on:
- Which OAK model you have (OAK-1, OAK-D, OAK-D Pro, OAK-D Lite, ...).
- Which neural blob you load on-device (MediaPipe FaceMesh recommended).
- Whether you do landmark inference on-device or on the host.

Recommended starting points:
- Luxonis examples: https://github.com/luxonis/depthai-experiments
- FaceMesh on OAK: https://github.com/geaxgx/depthai_blazepose

What this scaffold does:
1. Defines the task script (which task, in what order, for how long).
2. Provides a `record_session()` driver that, while `depthai` runs, prompts
   the patient through tasks, timestamps each task window, and accumulates
   face landmarks into a buffer.
3. Packs the buffer into a CaptureSession ready for `run_screening_agent`.

You wire in:
- The actual depthai pipeline (camera + FaceMesh NN node).
- A `parse_face` function that, given a queue message, returns
  `(timestamp, (478, 3) landmarks, blink_event_or_None)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from parkinson_agent.oak_adapter import (
    CaptureSession,
    FaceTimeSeries,
    TaskWindow,
)


# ---------------------------------------------------------------------------
# Task protocol — face-only subset of MDS-UPDRS Part III.
# ---------------------------------------------------------------------------

@dataclass
class TaskScript:
    name: str
    duration_s: float
    instruction_text: str  # what the UI shows the patient


DEFAULT_PROTOCOL: list[TaskScript] = [
    TaskScript(
        "rest_seated",
        10.0,
        "Sit relaxed and look straight at the camera. Stay still and keep a neutral face.",
    ),
    TaskScript(
        "facial_expression",
        10.0,
        "On the prompt: smile broadly, then make a surprised face, then a neutral face. "
        "Repeat twice.",
    ),
    TaskScript(
        "speech",
        8.0,
        "Read this sentence aloud at normal volume: "
        "'The early bird catches the worm but the second mouse gets the cheese.'",
    ),
]


# ---------------------------------------------------------------------------
# Frame buffer — accumulated during capture, then frozen into a FaceTimeSeries.
# ---------------------------------------------------------------------------

@dataclass
class _FaceBuffer:
    timestamps: list[float] = field(default_factory=list)
    landmarks: list[np.ndarray] = field(default_factory=list)  # each (478, 3)
    blink_events: list[float] = field(default_factory=list)

    def to_timeseries(self) -> FaceTimeSeries:
        return FaceTimeSeries(
            timestamps=np.asarray(self.timestamps, dtype=np.float64),
            landmarks=(
                np.stack(self.landmarks)
                if self.landmarks
                else np.zeros((0, 478, 3))
            ),
            blink_events=list(self.blink_events),
        )


# ---------------------------------------------------------------------------
# Driver — orchestrates the protocol while a depthai pipeline runs in the bg.
# ---------------------------------------------------------------------------

def record_session(
    patient_id: str,
    protocol: list[TaskScript] | None = None,
    pipeline_factory=None,    # () -> depthai.Pipeline; supply when wiring real OAK
    on_prompt=None,           # callable(task_name, instruction_text); UI hook
) -> CaptureSession:
    """Drive the patient through the protocol and return a CaptureSession.

    The function is intentionally NOT depthai-coupled in its signature.
    Inject `pipeline_factory` to use real hardware. Without it, this raises
    so you don't accidentally collect zero-data sessions.

    Sketch of the real implementation (pseudocode you fill in):

        with dai.Device(pipeline_factory()) as device:
            face_q = device.getOutputQueue("face_landmarks", maxSize=4, blocking=False)

            t0 = time.time()
            for task in protocol:
                if on_prompt: on_prompt(task.name, task.instruction_text)
                task_start = time.time() - t0
                while time.time() - t0 - task_start < task.duration_s:
                    msg = face_q.tryGet()
                    if msg is not None:
                        ts, lm, blink = parse_face(msg)  # YOU implement parse_face
                        buf.timestamps.append(ts)
                        buf.landmarks.append(lm)
                        if blink is not None:
                            buf.blink_events.append(blink)
                tasks.append(TaskWindow(task.name, task_start, time.time() - t0))
    """
    if pipeline_factory is None:
        raise NotImplementedError(
            "Wire in a depthai pipeline_factory. See module docstring for "
            "starting points (Luxonis examples, depthai_blazepose)."
        )
    raise NotImplementedError("record_session: implement the depthai capture loop.")


# ---------------------------------------------------------------------------
# Convenience: read/write a CaptureSession to .npz for replay during demos.
# ---------------------------------------------------------------------------

def load_session(path: str) -> CaptureSession:
    """Load a CaptureSession previously dumped via `save_session`."""
    with np.load(path, allow_pickle=True) as data:
        face = None
        if "face_timestamps" in data.files:
            face = FaceTimeSeries(
                timestamps=data["face_timestamps"],
                landmarks=data["face_landmarks"],
                blink_events=(
                    list(data["face_blink_events"])
                    if "face_blink_events" in data.files
                    else []
                ),
            )
        tasks = [
            TaskWindow(name=str(n), start=float(s), end=float(e))
            for n, s, e in data["tasks"].tolist()
        ]
        return CaptureSession(
            session_id=str(data["session_id"]),
            patient_id=str(data["patient_id"]),
            duration_s=float(data["duration_s"]),
            capture_fps=float(data["capture_fps"]),
            tasks=tasks,
            face=face,
        )


def save_session(session: CaptureSession, path: str) -> None:
    """Dump a CaptureSession to .npz for replay during demos / debugging."""
    payload = {
        "session_id": session.session_id,
        "patient_id": session.patient_id,
        "duration_s": session.duration_s,
        "capture_fps": session.capture_fps,
        "tasks": np.array(
            [(t.name, t.start, t.end) for t in session.tasks],
            dtype=object,
        ),
    }
    if session.face is not None:
        payload["face_timestamps"] = session.face.timestamps
        payload["face_landmarks"] = session.face.landmarks
        payload["face_blink_events"] = np.array(
            session.face.blink_events, dtype=np.float64
        )
    np.savez_compressed(path, **payload)
