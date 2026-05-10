"""NeuroVista clinician dashboard.

Reads visit JSONs from data/visits/<patient_id>/visit_*.json and renders
trend lines for each biomarker. Highlights deltas vs. baseline (first visit).

Run:
  python3 backend/app.py [--port 8080] [--data-dir data/visits]

Then open http://localhost:8080
"""
import argparse
import csv
import io
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file, abort, Response, request

from backend.pdf_report import build_pdf_report

NEUROSIGHT_DIR = Path(__file__).resolve().parent.parent  # neurosight/
PYTHON_BIN = os.environ.get("NEUROSIGHT_PYTHON", "/usr/bin/python3")

_recording_state = {
    "active": False,
    "patient_id": None,
    "visit_id": None,
    "duration": 0,
    "started_at": None,
    "csv_path": None,
    "json_path": None,
    "last_error": None,
    "proc_pid": None,
}
_record_lock = threading.Lock()


def _cleanup_partial_visit_files(patient_id, visit_id):
    """Best-effort delete of files that a cancelled recording may have left
    half-written. Safe to call even if files don't exist."""
    if not patient_id or not visit_id:
        return []
    pdir = NEUROSIGHT_DIR / "data" / "visits" / patient_id
    if not pdir.exists():
        return []
    deleted = []
    for prefix in ("visit_", "embed_"):
        for suffix in (".csv", ".json", ".run.log"):
            p = pdir / f"{prefix}{visit_id}{suffix}"
            try:
                if p.exists():
                    p.unlink()
                    deleted.append(p.name)
            except Exception:
                pass
    return deleted


def _kill_existing_pipeline():
    """Stop any running main.py so a new pipeline can grab the OAK device.

    Uses SIGTERM first (gives DepthAI a chance to tear down the DeviceGate
    session cleanly) and only escalates to SIGKILL if needed. SIGKILL leaves
    the gate in an orphan state which makes the *next* pipeline fail with
    X_LINK_DEVICE_NOT_FOUND — so we strongly prefer the graceful path.
    """
    # Step 1: graceful SIGTERM
    subprocess.run(["pkill", "-TERM", "-f", "main.py"], check=False)
    # Wait up to 4s for the process to exit
    for _ in range(8):
        time.sleep(0.5)
        result = subprocess.run(
            ["pgrep", "-f", "main.py"], capture_output=True, text=True,
        )
        # Filter out our own shell wrappers (they contain 'main.py' literally)
        pids = [p for p in result.stdout.strip().split("\n") if p]
        if not pids:
            break
    else:
        # Step 2: escalate to SIGKILL only if SIGTERM failed
        subprocess.run(["pkill", "-9", "-f", "main.py"], check=False)
        time.sleep(1.0)
    # Extra grace period for the OS to release the OAK device gate
    time.sleep(2.0)


def _spawn_pipeline(args, log_path=None):
    """Launch python3 main.py … in NEUROSIGHT_DIR. Returns the Popen handle."""
    cmd = [PYTHON_BIN, "main.py"] + list(args)
    out = open(log_path, "w") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        cmd, cwd=str(NEUROSIGHT_DIR),
        stdout=out, stderr=subprocess.STDOUT,
    )


def _spawn_debug_pipeline():
    """Restart the no-save debug pipeline (background, non-blocking).

    By default we now spawn it in **clinician mode**: HUD off + clean mask
    (only salient landmarks, thinner). The 8082 visualizer is the one
    embedded in the doctor portal session page, so it must look clean
    even when the page is opened between visits.
    """
    log_path = NEUROSIGHT_DIR / "data" / "debug_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return _spawn_pipeline(
        ["--no_save", "--no_voice", "--debug", "--no_hud", "--clean_mask"],
        log_path=str(log_path),
    )

app = Flask(__name__, template_folder="templates", static_folder="static")
DATA_DIR = Path("data/visits")


def _load_patient_visits(patient_id):
    pdir = DATA_DIR / patient_id
    if not pdir.exists():
        return []
    visits = []
    for vp in sorted(pdir.glob("visit_*.json")):
        # Skip the per-frame JSON mirrors (arrays, not summary dicts)
        if vp.stem.endswith("_frames"):
            continue
        try:
            with open(vp) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            visits.append(data)
        except Exception as e:
            print(f"[dashboard] failed to load {vp}: {e}")
    visits.sort(key=lambda v: v.get("visit_id", ""))
    return visits


def _list_patients():
    if not DATA_DIR.exists():
        return []
    return sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])


@app.route("/")
def index():
    patients = _list_patients()
    default = patients[0] if patients else None
    return render_template("dashboard.html", patients=patients, current=default)


@app.route("/api/patients")
def api_patients():
    return jsonify(_list_patients())


FACIAL_METRIC_KEYS = [
    "blink_rate",
    "smile_amplitude_mm",
    "brow_amplitude_mm",
    "asymmetry",
    "gaze_x_std",
    "gaze_y_std",
    "tremor_chin_power_4_6hz",
    "tremor_chin_dominant_hz",
    "tremor_lip_power_4_6hz",
]
VOICE_METRIC_KEYS = [
    "jitter_local_pct",
    "shimmer_local_pct",
    "hnr_db",
    "f0_mean_hz",
    "f0_std_hz",
    "intensity_db",
    "speech_rate_wpm",
    "pause_ratio",
]


def _safe_pct_delta(b, l):
    if b is None or l is None:
        return None
    try:
        b = float(b)
        l = float(l)
    except (TypeError, ValueError):
        return None
    if b == 0:
        return None
    return round((l - b) / abs(b) * 100, 1)


@app.route("/api/patient/<patient_id>")
def api_patient(patient_id):
    visits = _load_patient_visits(patient_id)
    if not visits:
        return jsonify({"patient_id": patient_id, "visits": [], "series": {}})

    series = {k: [] for k in FACIAL_METRIC_KEYS + VOICE_METRIC_KEYS}
    labels = []
    for v in visits:
        labels.append(v.get("visit_id", "?"))
        metrics = v.get("metrics", {}) or {}
        voice = v.get("voice", {}) or {}
        for k in FACIAL_METRIC_KEYS:
            entry = metrics.get(k) or {}
            series[k].append(entry.get("mean"))
        for k in VOICE_METRIC_KEYS:
            series[k].append(voice.get(k))

    base_metrics = visits[0].get("metrics", {}) or {}
    base_voice = visits[0].get("voice", {}) or {}
    last_metrics = visits[-1].get("metrics", {}) or {}
    last_voice = visits[-1].get("voice", {}) or {}
    deltas = {}
    for k in FACIAL_METRIC_KEYS:
        deltas[k] = _safe_pct_delta(
            (base_metrics.get(k) or {}).get("mean"),
            (last_metrics.get(k) or {}).get("mean"),
        )
    for k in VOICE_METRIC_KEYS:
        deltas[k] = _safe_pct_delta(base_voice.get(k), last_voice.get(k))

    ranges = visits[0].get("ranges", {})

    return jsonify(
        {
            "patient_id": patient_id,
            "labels": labels,
            "series": series,
            "deltas_pct_vs_baseline": deltas,
            "ranges": ranges,
            "n_visits": len(visits),
            "latest_visit": labels[-1] if labels else None,
            "latest_recorded_at": visits[-1].get("recorded_at"),
            "has_voice": any(
                visits[-1].get("voice", {}).get(k) is not None
                for k in VOICE_METRIC_KEYS
            ),
            "synthetic": bool(visits[-1].get("synthetic", False)),
        }
    )


@app.route("/api/patient/<patient_id>/report.pdf")
def api_report(patient_id):
    visits = _load_patient_visits(patient_id)
    if not visits:
        abort(404)
    pdf_path = build_pdf_report(patient_id, visits)
    return send_file(pdf_path, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"NeuroVista_{patient_id}_{datetime.now():%Y%m%d}.pdf")


def _start_record_subprocess(patient_id, visit_id, duration, use_voice,
                             sex=None, age=None, label=None, state="visit",
                             no_hud=False):
    """Internal helper used by /api/record and /api/baseline."""
    pipeline_args = [
        "--patient_id", patient_id,
        "--visit_id", visit_id,
        "--duration", str(duration),
        "--state", state,
        "--debug",
    ]
    if not use_voice:
        pipeline_args.append("--no_voice")
    if no_hud:
        pipeline_args.append("--no_hud")
        pipeline_args.append("--clean_mask")
    if sex is not None:
        pipeline_args += ["--sex", str(sex)]
    if age is not None:
        pipeline_args += ["--age", str(age)]
    if label is not None:
        pipeline_args += ["--label", str(label)]

    log_path = NEUROSIGHT_DIR / "data" / "visits" / patient_id / \
               f"visit_{visit_id}.run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return _spawn_pipeline(pipeline_args, log_path=str(log_path))


@app.route("/api/record", methods=["POST"])
def api_record():
    """Stop debug pipeline, run a saved visit, then resume debug pipeline."""
    body = request.get_json(silent=True) or {}
    patient_id = (body.get("patient_id") or "demo_live").strip()
    duration = max(5, min(int(body.get("duration") or 60), 600))
    visit_id = (body.get("visit_id") or
                datetime.now().strftime("%Y-%m-%d_%H%M%S")).strip()
    use_voice = bool(body.get("voice", False))
    no_hud = bool(body.get("no_hud", False))
    sex = body.get("sex")
    age = body.get("age")
    label = body.get("label")
    state = (body.get("state") or "visit").strip()

    with _record_lock:
        if _recording_state["active"]:
            return jsonify({"error": "another recording is already in progress",
                            "state": dict(_recording_state)}), 409
        _recording_state.update({
            "active": True,
            "patient_id": patient_id,
            "visit_id": visit_id,
            "duration": duration,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "csv_path": f"data/visits/{patient_id}/visit_{visit_id}.csv",
            "json_path": f"data/visits/{patient_id}/visit_{visit_id}.json",
            "last_error": None,
        })

    _kill_existing_pipeline()
    proc = _start_record_subprocess(
        patient_id, visit_id, duration, use_voice,
        sex=sex, age=age, label=label, state=state, no_hud=no_hud,
    )
    with _record_lock:
        _recording_state["proc_pid"] = proc.pid

    def _wait_and_resume():
        try:
            # Wait up to duration + 60s for the JSON to be written
            deadline = time.monotonic() + duration + 60
            json_path = NEUROSIGHT_DIR / _recording_state["json_path"]
            while time.monotonic() < deadline:
                if json_path.exists() or proc.poll() is not None:
                    break
                time.sleep(0.5)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            if not json_path.exists():
                with _record_lock:
                    _recording_state["last_error"] = (
                        "visit JSON was not produced; check run.log"
                    )
        finally:
            with _record_lock:
                _recording_state["active"] = False
            # Resume the debug pipeline so the visualizer keeps working
            try:
                _kill_existing_pipeline()
                _spawn_debug_pipeline()
            except Exception as e:
                print(f"[record] failed to restart debug pipeline: {e}")

    threading.Thread(target=_wait_and_resume, daemon=True).start()

    return jsonify({"started": True, "state": dict(_recording_state)})


@app.route("/api/recording_status")
def api_recording_status():
    with _record_lock:
        return jsonify(dict(_recording_state))


@app.route("/api/device/reboot", methods=["POST"])
def api_device_reboot():
    """Soft-reboot the OAK device via oakctl. Useful when the DeviceGate
    session is stuck (X_LINK_DEVICE_NOT_FOUND on subsequent connects).
    Kills the debug pipeline first so it doesn't fight the reboot, waits for
    the device to come back, then restarts the debug pipeline."""
    body = request.get_json(silent=True) or {}
    ip = body.get("ip", "169.254.232.53")
    _kill_existing_pipeline()
    try:
        result = subprocess.run(
            ["oakctl", "device", "reboot", "-d", ip],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"error": "oakctl reboot timed out"}), 500
    if result.returncode != 0:
        return jsonify({
            "error": "oakctl reboot failed",
            "stdout": result.stdout, "stderr": result.stderr,
        }), 500

    def _wait_then_restart():
        # Wait up to 60s for the device to come back online
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                r = subprocess.run(
                    ["oakctl", "device", "list"],
                    capture_output=True, text=True, timeout=5,
                )
                if ip in r.stdout:
                    break
            except Exception:
                pass
            time.sleep(2.0)
        time.sleep(2.0)
        try:
            _spawn_debug_pipeline()
        except Exception as e:
            print(f"[device_reboot] failed to restart debug pipeline: {e}")

    threading.Thread(target=_wait_then_restart, daemon=True).start()
    return jsonify({"rebooted": True, "ip": ip})


@app.route("/api/record/cancel", methods=["POST"])
def api_record_cancel():
    """Stop the running record subprocess and delete its partial files.
    Restarts the debug pipeline so the visualizer keeps working."""
    body = request.get_json(silent=True) or {}
    delete_partial = bool(body.get("delete", True))
    with _record_lock:
        active = _recording_state["active"]
        pid = _recording_state.get("proc_pid")
        patient_id = _recording_state.get("patient_id")
        visit_id = _recording_state.get("visit_id")
        baseline_active = _baseline_state["active"]
    if not active and not baseline_active:
        return jsonify({"error": "nothing to cancel"}), 400
    if pid:
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(0.5)
            try:
                os.kill(pid, 9)  # SIGKILL fallback
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
    deleted = []
    if delete_partial:
        deleted = _cleanup_partial_visit_files(patient_id, visit_id)
    with _record_lock:
        _recording_state["active"] = False
        _recording_state["last_error"] = "cancelled by user"
    with _baseline_lock:
        if _baseline_state["active"]:
            _baseline_state["active"] = False
            _baseline_state["last_error"] = "cancelled by user"
    # Restart debug pipeline (fire-and-forget; the _wait_and_resume thread may
    # also try, but pipeline kill is idempotent)
    threading.Thread(target=lambda: (
        time.sleep(2.0), _kill_existing_pipeline(), _spawn_debug_pipeline(),
    ), daemon=True).start()
    return jsonify({"cancelled": True, "deleted": deleted})


@app.route("/api/patient_index")
def api_patient_index():
    """Expose the patient_index.json registry so the UI can pre-fill labels."""
    from utils.visit_writer import _load_patient_index
    return jsonify(_load_patient_index())


_baseline_state = {
    "active": False,
    "patient_id": None,
    "current_step": None,    # "rest" | "talk" | "smile" | None
    "completed": [],         # list of state strings already done
    "started_at": None,
    "last_error": None,
}
_baseline_lock = threading.Lock()


@app.route("/api/baseline", methods=["POST"])
def api_baseline():
    """Run a 3-stage baseline: rest → talk → smile.
    Each stage is a fresh saved visit with state=<stage>. Same patient/age/sex
    metadata for all three. Returns immediately; poll /api/baseline_status."""
    body = request.get_json(silent=True) or {}
    patient_id = (body.get("patient_id") or "baseline_subject").strip()
    duration = max(5, min(int(body.get("duration") or 10), 60))
    sex = body.get("sex")
    age = body.get("age")
    label = body.get("label")
    use_voice = bool(body.get("voice", False))

    with _baseline_lock:
        if _baseline_state["active"]:
            return jsonify({"error": "baseline already in progress",
                            "state": dict(_baseline_state)}), 409
        _baseline_state.update({
            "active": True,
            "patient_id": patient_id,
            "current_step": None,
            "completed": [],
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "last_error": None,
        })

    _kill_existing_pipeline()

    def _run_baseline():
        stages = ["rest", "talk", "smile"]
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        try:
            for stage in stages:
                with _baseline_lock:
                    _baseline_state["current_step"] = stage
                with _record_lock:
                    _recording_state.update({
                        "active": True,
                        "patient_id": patient_id,
                        "visit_id": f"{ts}_{stage}",
                        "duration": duration,
                        "started_at": datetime.now().isoformat(timespec="seconds"),
                        "csv_path": f"data/visits/{patient_id}/visit_{ts}_{stage}.csv",
                        "json_path": f"data/visits/{patient_id}/visit_{ts}_{stage}.json",
                        "last_error": None,
                    })
                proc = _start_record_subprocess(
                    patient_id, f"{ts}_{stage}", duration, use_voice,
                    sex=sex, age=age, label=label, state=stage,
                )
                with _record_lock:
                    _recording_state["proc_pid"] = proc.pid
                deadline = time.monotonic() + duration + 30
                json_path = NEUROSIGHT_DIR / f"data/visits/{patient_id}/visit_{ts}_{stage}.json"
                while time.monotonic() < deadline:
                    if json_path.exists() or proc.poll() is not None:
                        break
                    time.sleep(0.5)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                with _record_lock:
                    _recording_state["active"] = False
                if not json_path.exists():
                    with _baseline_lock:
                        _baseline_state["last_error"] = f"{stage} JSON not produced"
                    break
                with _baseline_lock:
                    _baseline_state["completed"].append(stage)
                # Small pause between stages so the OAK gate is fully released
                time.sleep(2.0)
        finally:
            with _baseline_lock:
                _baseline_state["active"] = False
                _baseline_state["current_step"] = None
            try:
                _kill_existing_pipeline()
                _spawn_debug_pipeline()
            except Exception as e:
                print(f"[baseline] failed to restart debug pipeline: {e}")

    threading.Thread(target=_run_baseline, daemon=True).start()
    return jsonify({"started": True, "state": dict(_baseline_state)})


@app.route("/api/baseline_status")
def api_baseline_status():
    with _baseline_lock:
        return jsonify(dict(_baseline_state))


@app.route("/api/patient/<patient_id>/visits")
def api_list_visits(patient_id):
    """Concise list of all recordings for a patient (for the table view)."""
    visits = _load_patient_visits(patient_id)
    out = []
    for v in visits:
        m = v.get("metrics", {}) or {}
        def get_mean(k):
            e = m.get(k) or {}
            return e.get("mean")
        out.append({
            "visit_id": v.get("visit_id"),
            "recorded_at": v.get("recorded_at"),
            "duration_s": v.get("duration_s"),
            "n_frames": v.get("n_frames"),
            "state": v.get("state"),
            "sex": v.get("sex"),
            "age": v.get("age"),
            "label": v.get("label"),
            "synthetic": bool(v.get("synthetic")),
            "blink_rate": get_mean("blink_rate"),
            "smile_amplitude_mm": get_mean("smile_amplitude_mm"),
            "asymmetry": get_mean("asymmetry"),
            "tremor_chin_pd_likelihood": get_mean("tremor_chin_pd_likelihood"),
        })
    return jsonify(out)


@app.route("/api/patient/<patient_id>/visit/<visit_id>")
def api_get_visit(patient_id, visit_id):
    path = DATA_DIR / patient_id / f"visit_{visit_id}.json"
    if not path.exists():
        abort(404)
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/patient/<patient_id>/visit/<visit_id>", methods=["DELETE"])
def api_delete_visit(patient_id, visit_id):
    pdir = DATA_DIR / patient_id
    if not pdir.exists():
        abort(404)
    safe = visit_id.replace("/", "").replace("..", "")
    targets = [
        pdir / f"visit_{safe}.json",
        pdir / f"visit_{safe}.csv",
        pdir / f"visit_{safe}_frames.json",
        pdir / f"embed_{safe}.csv",
        pdir / f"embed_{safe}.json",
        pdir / f"visit_{safe}.run.log",
    ]
    deleted = []
    for p in targets:
        if p.exists():
            try:
                p.unlink()
                deleted.append(p.name)
            except Exception as e:
                return jsonify({"error": f"failed to delete {p.name}: {e}"}), 500
    if not deleted:
        return jsonify({"error": "no files found for that visit"}), 404
    # If patient dir is now empty, remove it too
    try:
        if not any(pdir.iterdir()):
            pdir.rmdir()
    except Exception:
        pass
    return jsonify({"deleted": deleted})


@app.route("/api/patient/<patient_id>/log.csv")
def api_log_csv(patient_id):
    """Cross-session CSV: one row per visit with summary mean + delta vs baseline,
    plus a per-frame block (concatenated visit traces with a visit_id column).

    Single CSV download, suitable for pandas / Excel / R."""
    visits = _load_patient_visits(patient_id)
    if not visits:
        abort(404)

    buf = io.StringIO()
    w = csv.writer(buf)

    summary_cols = ["visit_id", "recorded_at", "n_frames", "duration_s"] + \
                   FACIAL_METRIC_KEYS + VOICE_METRIC_KEYS
    w.writerow(["# section: per-visit summary (mean across visit)"])
    w.writerow(summary_cols)
    for v in visits:
        metrics = v.get("metrics", {}) or {}
        voice = v.get("voice", {}) or {}
        row = [
            v.get("visit_id", ""),
            v.get("recorded_at", ""),
            v.get("n_frames", ""),
            v.get("duration_s", ""),
        ]
        for k in FACIAL_METRIC_KEYS:
            row.append((metrics.get(k) or {}).get("mean", ""))
        for k in VOICE_METRIC_KEYS:
            row.append(voice.get(k, ""))
        w.writerow(row)

    w.writerow([])
    w.writerow(["# section: per-frame trace (concatenated across visits)"])
    trace_cols = ["visit_id", "frame_idx_in_visit"] + FACIAL_METRIC_KEYS
    w.writerow(trace_cols)
    for v in visits:
        vid = v.get("visit_id", "")
        metrics = v.get("metrics", {}) or {}
        # Aligned by index — assume traces share the same length per visit
        traces = {k: (metrics.get(k) or {}).get("trace", []) for k in FACIAL_METRIC_KEYS}
        n = max((len(t) for t in traces.values()), default=0)
        for i in range(n):
            row = [vid, i]
            for k in FACIAL_METRIC_KEYS:
                t = traces.get(k, [])
                row.append(t[i] if i < len(t) else "")
            w.writerow(row)

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f'attachment; filename="NeuroVista_{patient_id}_'
                f'{datetime.now():%Y%m%d_%H%M}.csv"'
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data-dir", type=str, default="data/visits")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    global DATA_DIR
    DATA_DIR = Path(args.data_dir)
    print(f"[dashboard] serving from data dir: {DATA_DIR.resolve()}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
