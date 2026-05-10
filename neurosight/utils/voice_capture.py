"""Voice analysis subsystem for NeuroVista.

The OAK 4 D has no microphone, so voice capture runs on the host machine in a
parallel thread for the visit duration. After capture, audio is processed
LOCALLY:
  - parselmouth (Praat bindings) → jitter, shimmer, HNR, F0 stats, intensity
  - openai-whisper tiny (LOCAL inference, no network) → speech rate, pause ratio

The temporary WAV is deleted after feature extraction. Whisper runs locally
to keep the on-device privacy narrative intact (audio never leaves the machine).

References:
  - Rusz et al., J Acoust Soc Am 2011 — jitter / shimmer in early PD
  - Goberman & Coelho 2002 — HNR reduction in PD dysphonia
  - Skodda et al., Mov Disord 2011 — speech rate slowing in PD
"""
from __future__ import annotations

import threading
import tempfile
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 1


class VoiceCaptureSession:
    """Thread-based microphone capture for a fixed duration.

    Usage:
        sess = VoiceCaptureSession(duration_s=60)
        if sess.start():
            ... pipeline runs in parallel ...
            wav = sess.stop()           # also called automatically after duration
            features = extract_voice_features(wav)
    """

    def __init__(self, duration_s: float, output_dir: Path | None = None):
        self.duration_s = float(duration_s)
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list = []
        self._stream = None
        self._stop_event = threading.Event()
        self._timer: threading.Thread | None = None
        self._wav_path: Path | None = None
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def wav_path(self) -> Path | None:
        return self._wav_path

    def start(self) -> bool:
        """Start mic capture. Returns False if mic / audio stack unavailable."""
        try:
            import sounddevice as sd
        except ImportError as e:
            self._error = f"sounddevice not installed: {e}"
            return False

        def _callback(indata, frames, t_info, status):
            if status:
                pass  # overflow/underflow non-fatal
            self._buffer.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                callback=_callback,
                dtype="float32",
            )
            self._stream.start()
        except Exception as e:
            self._error = f"mic open failed: {e}"
            return False

        self._timer = threading.Thread(target=self._auto_stop, daemon=True)
        self._timer.start()
        print(f"[voice] mic capture started ({self.duration_s:.0f}s)")
        return True

    def _auto_stop(self):
        if self._stop_event.wait(self.duration_s):
            return  # external stop fired before timer
        self.stop()

    def stop(self) -> Path | None:
        """Stop the stream and write the captured audio to a WAV file."""
        if self._stop_event.is_set():
            return self._wav_path
        self._stop_event.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if not self._buffer:
            self._error = "no audio captured"
            return None

        try:
            import soundfile as sf
        except ImportError as e:
            self._error = f"soundfile not installed: {e}"
            return None

        audio = np.concatenate(self._buffer, axis=0).flatten()
        wav_path = self.output_dir / f"neurovista_voice_{int(time.time())}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)
        self._wav_path = wav_path
        print(f"[voice] wrote {wav_path}  ({len(audio) / SAMPLE_RATE:.1f}s)")
        return wav_path


def extract_voice_features(
    wav_path: Path | str | None, run_whisper: bool = True
) -> dict:
    """Extract voice features from a WAV. Graceful — never raises.

    Returns a dict with all known fields; missing values are None and any
    failures are accumulated in `errors`.
    """
    out: dict = {
        "wav_seconds": None,
        "jitter_local_pct": None,
        "shimmer_local_pct": None,
        "hnr_db": None,
        "f0_mean_hz": None,
        "f0_std_hz": None,
        "intensity_db": None,
        "speech_rate_wpm": None,
        "pause_ratio": None,
        "transcript_word_count": None,
        "transcript_excerpt": None,
        "errors": [],
    }
    if wav_path is None:
        out["errors"].append("no wav supplied")
        return out
    wav_path = Path(wav_path)
    if not wav_path.exists():
        out["errors"].append(f"wav not found: {wav_path}")
        return out

    # ─── Acoustic features via Praat (parselmouth) ──────────────
    try:
        import parselmouth
        snd = parselmouth.Sound(str(wav_path))
        out["wav_seconds"] = round(float(snd.duration), 2)

        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        if f0.size > 0:
            out["f0_mean_hz"] = round(float(np.mean(f0)), 1)
            out["f0_std_hz"] = round(float(np.std(f0)), 1)

        try:
            point_process = parselmouth.praat.call(
                snd, "To PointProcess (periodic, cc)", 75, 600
            )
            jitter = parselmouth.praat.call(
                point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )
            shimmer = parselmouth.praat.call(
                [snd, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
            if jitter is not None and not np.isnan(jitter):
                out["jitter_local_pct"] = round(float(jitter) * 100.0, 3)
            if shimmer is not None and not np.isnan(shimmer):
                out["shimmer_local_pct"] = round(float(shimmer) * 100.0, 3)
        except Exception as e:
            out["errors"].append(f"jitter/shimmer failed: {e}")

        try:
            harm = snd.to_harmonicity_cc(time_step=0.01)
            hnr_vals = harm.values[harm.values != -200]
            if hnr_vals.size > 0:
                out["hnr_db"] = round(float(np.mean(hnr_vals)), 2)
        except Exception as e:
            out["errors"].append(f"hnr failed: {e}")

        try:
            intensity = snd.to_intensity()
            iv = intensity.values
            iv = iv[~np.isnan(iv)]
            if iv.size > 0:
                out["intensity_db"] = round(float(np.mean(iv)), 2)
        except Exception as e:
            out["errors"].append(f"intensity failed: {e}")

    except ImportError:
        out["errors"].append("parselmouth not installed")
    except Exception as e:
        out["errors"].append(f"parselmouth failed: {e}")

    # ─── Transcription via Whisper-tiny LOCAL ───────────────────
    if run_whisper:
        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(str(wav_path), language=None, fp16=False)
            text = (result.get("text") or "").strip()
            segments = result.get("segments", []) or []
            wc = len(text.split())
            out["transcript_word_count"] = wc
            out["transcript_excerpt"] = text[:160]
            duration_s = out.get("wav_seconds") or 60.0
            out["speech_rate_wpm"] = round(wc / max(duration_s / 60.0, 1e-3), 1)
            speech_time = sum(
                max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
                for s in segments
            )
            if duration_s > 0:
                out["pause_ratio"] = round(
                    1.0 - min(1.0, speech_time / float(duration_s)), 3
                )
        except ImportError:
            out["errors"].append("whisper not installed")
        except Exception as e:
            out["errors"].append(f"whisper failed: {e}")

    return out
