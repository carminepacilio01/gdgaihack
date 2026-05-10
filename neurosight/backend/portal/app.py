"""Doctor Portal — clinician-facing UI on port 8090.

Demo mockup: no login. Reads patient data from data/visits/, proxies the
recording / baseline / cancel actions to the 8080 tech dashboard so the
OAK pipeline subprocess management stays in one place.
"""
import json as _json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, Response,
)

# Make `backend.*` and `utils.*` importable from neurosight/
NEUROSIGHT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(NEUROSIGHT_DIR))
# Make `models.*` and `parkinson_agent.*` importable from gdgaihack/ root
GDGAIHACK_ROOT = NEUROSIGHT_DIR.parent
sys.path.insert(0, str(GDGAIHACK_ROOT))

from backend.app import _list_patients, _load_patient_visits  # type: ignore
from utils.visit_writer import _load_patient_index  # type: ignore


app = Flask(__name__)
TECH_DASHBOARD = os.environ.get("NEUROVISTA_TECH_URL", "http://localhost:8080")

# Paths to the upstream-model assets (TCN ONNX + scaler + feature names)
MODEL_PATH    = GDGAIHACK_ROOT / "models" / "pd_tcn_model_acc_80.19.onnx"
SCALER_PATH   = GDGAIHACK_ROOT / "models" / "preprocessed" / "scaler.pkl"
FEATURES_PATH = GDGAIHACK_ROOT / "models" / "preprocessed" / "feature_names.txt"


# ───────────────────────────────────────────────────────────────────────
# Background AI prefetch — 1 worker so CPU + Ollama don't get hammered.
# Idempotent: skips visits whose result+explanation are already on disk.
# ───────────────────────────────────────────────────────────────────────

_AI_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-prefetch")
_AI_INFLIGHT: set = set()
_AI_LOCK = threading.Lock()


def _ai_paths(patient_id: str, visit_id: str) -> dict:
    base = NEUROSIGHT_DIR / "data" / "visits" / patient_id
    return {
        "embed":       base / f"embed_{visit_id}.json",
        "result":      base / f"embed_{visit_id}_result.json",
        "explanation": base / f"embed_{visit_id}_explanation.txt",
    }


def _ai_status(patient_id: str, visit_id: str) -> dict:
    p = _ai_paths(patient_id, visit_id)
    return {
        "has_embed":       p["embed"].exists(),
        "has_result":      p["result"].exists(),
        "has_explanation": p["explanation"].exists(),
    }


def _ai_run_one(patient_id: str, visit_id: str) -> None:
    """Runs inference + explanation for a single visit. Idempotent."""
    p = _ai_paths(patient_id, visit_id)
    if not p["embed"].exists():
        return

    if not p["result"].exists():
        cmd = [
            "/usr/bin/python3", "-m", "models.inference",
            "--model",    str(MODEL_PATH),
            "--scaler",   str(SCALER_PATH),
            "--features", str(FEATURES_PATH),
            "--input",    str(p["embed"]),
            "--output",   str(p["result"]),
        ]
        try:
            subprocess.run(cmd, cwd=str(GDGAIHACK_ROOT),
                           capture_output=True, text=True, timeout=120)
        except Exception as e:
            print(f"[ai-prefetch] inference {patient_id}/{visit_id} failed: {e}")
            return

    if p["result"].exists() and not p["explanation"].exists():
        try:
            from parkinson_agent._agent import explain_prediction  # type: ignore
            text = explain_prediction(str(p["result"]))
            if text:
                p["explanation"].write_text(text)
        except Exception as e:
            print(f"[ai-prefetch] explain {patient_id}/{visit_id} failed: {e}")


def _ai_dispatch(patient_id: str, visit_id: str) -> None:
    """Submits one job to the background worker, deduped by (patient, visit)."""
    key = (patient_id, visit_id)
    with _AI_LOCK:
        if key in _AI_INFLIGHT:
            return
        st = _ai_status(patient_id, visit_id)
        if st["has_result"] and st["has_explanation"]:
            return
        if not st["has_embed"]:
            return
        _AI_INFLIGHT.add(key)

    def _job() -> None:
        try:
            _ai_run_one(patient_id, visit_id)
        finally:
            with _AI_LOCK:
                _AI_INFLIGHT.discard(key)

    _AI_EXECUTOR.submit(_job)


def _ai_prefetch_top_n(cards: list, n: int = 10) -> None:
    """For each of the top-N patients (by last_at), enqueue inference for
    their most recent visit."""
    for c in cards[:n]:
        pid = c.get("patient_id")
        if not pid or not c.get("last_at"):
            continue
        visits = _load_patient_visits(pid)
        if not visits:
            continue
        last = max(visits, key=lambda v: v.get("visit_id", ""))
        vid = last.get("visit_id")
        if vid:
            _ai_dispatch(pid, vid)


# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────

def _patient_card(patient_id):
    visits = _load_patient_visits(patient_id)
    n = len(visits)
    last = visits[-1] if visits else None
    last_at = last.get("recorded_at") if last else None
    states = sorted({v.get("state") for v in visits if v.get("state")})
    has_baseline = all(s in states for s in ("rest", "talk", "smile"))
    label = sex = age = patient_num = None
    for v in reversed(visits):
        if label is None and v.get("label") is not None:
            label = v.get("label")
        if sex is None and v.get("sex") is not None:
            sex = v.get("sex")
        if age is None and v.get("age") is not None:
            age = v.get("age")
        if patient_num is None and v.get("patient_num") is not None:
            patient_num = v.get("patient_num")
    last_metrics = (last or {}).get("metrics", {}) or {}

    def m(k):
        e = last_metrics.get(k) or {}
        return e.get("mean")

    return {
        "patient_id": patient_id,
        "patient_num": patient_num,
        "n_visits": n,
        "last_at": last_at,
        "has_baseline": has_baseline,
        "states": states,
        "label": label,
        "sex": sex,
        "age": age,
        "last_blink": m("blink_rate"),
        "last_smile": m("smile_amplitude_mm"),
        "last_asym": m("asymmetry"),
        "last_pd": m("tremor_chin_pd_likelihood"),
    }


# ───────────────────────────────────────────────────────────────────────
# Routes
# ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    patients = _list_patients()
    cards = [_patient_card(p) for p in patients]
    cards.sort(key=lambda c: (c["last_at"] or ""), reverse=True)
    # Pre-warm AI screening reports for the 10 most-recent patients so the
    # neurologist's "Generate report" click is instant. Runs serially in a
    # background thread, idempotent (skips cached visits).
    _ai_prefetch_top_n(cards, n=10)
    n_baseline = sum(1 for c in cards if c["has_baseline"])
    n_pd = sum(1 for c in cards if c["label"] == 1)
    n_no_pd = sum(1 for c in cards if c["label"] == 0)
    total_visits = sum(c["n_visits"] for c in cards)
    return render_template(
        "dashboard.html",
        cards=cards,
        n_patients=len(cards),
        n_baseline=n_baseline,
        n_pd=n_pd,
        n_no_pd=n_no_pd,
        total_visits=total_visits,
        tech_url=TECH_DASHBOARD,
    )


@app.route("/session/<patient_id>")
def session_view(patient_id):
    """Active recording page — cam feed + live notes + probability ring.
    Reads `duration`, `sex`, `age`, `label` from query string and POSTs
    /api/record on page load."""
    duration = request.args.get("duration", "60")
    sex = request.args.get("sex", "")
    age = request.args.get("age", "")
    label = request.args.get("label", "")
    card = _patient_card(patient_id) if patient_id in _list_patients() else {
        "patient_id": patient_id, "patient_num": None, "n_visits": 0,
        "last_at": None, "has_baseline": False, "states": [],
        "label": int(label) if label else None,
        "sex": int(sex) if sex else None,
        "age": int(age) if age else None,
        "last_blink": None, "last_smile": None, "last_asym": None, "last_pd": None,
    }
    return render_template(
        "session.html",
        card=card,
        patient_id=patient_id,
        duration=int(duration) if duration.isdigit() else 60,
        sex=sex, age=age, label=label,
    )


@app.route("/patient/<patient_id>")
def patient_view(patient_id):
    visits = _load_patient_visits(patient_id)
    visits.sort(key=lambda v: v.get("visit_id", ""), reverse=True)
    card = _patient_card(patient_id)
    # Kick off prefetch for this patient's most recent visit too (covers the
    # case where the user lands here without going through /dashboard first).
    if visits:
        vid = visits[0].get("visit_id")
        if vid:
            _ai_dispatch(patient_id, vid)
    # Annotate visits with AI cache status so the template can render badges
    # and the dropdown can default to a ready-cached visit.
    for v in visits:
        vid = v.get("visit_id")
        if vid:
            v["ai_status"] = _ai_status(patient_id, vid)
        else:
            v["ai_status"] = {"has_embed": False, "has_result": False, "has_explanation": False}
    return render_template(
        "patient.html",
        card=card,
        visits=visits,
        tech_url=TECH_DASHBOARD,
        patient_id=patient_id,
    )


# ───────────────────────────────────────────────────────────────────────
# Read-only API
# ───────────────────────────────────────────────────────────────────────

@app.route("/api/patient_index")
def api_patient_index():
    return jsonify(_load_patient_index())


@app.route("/api/ai_status/<patient_id>/<visit_id>")
def api_ai_status(patient_id, visit_id):
    with _AI_LOCK:
        in_flight = (patient_id, visit_id) in _AI_INFLIGHT
    return jsonify({
        **_ai_status(patient_id, visit_id),
        "in_flight": in_flight,
    })


# ───────────────────────────────────────────────────────────────────────
# Proxy to 8080 (the OAK pipeline owner)
# ───────────────────────────────────────────────────────────────────────

def _proxy(method, path, json_body=None, stream=False, timeout=10):
    url = f"{TECH_DASHBOARD}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=timeout, stream=stream)
        else:
            r = requests.post(url, json=json_body or {}, timeout=timeout)
    except requests.RequestException as e:
        return jsonify({"error": f"tech dashboard unreachable: {e}"}), 503

    if stream:
        return Response(
            r.iter_content(chunk_size=8192),
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/octet-stream"),
            headers={"Content-Disposition": r.headers.get("Content-Disposition", "")},
        )
    return Response(
        r.content, status=r.status_code,
        content_type=r.headers.get("Content-Type", "application/json"),
    )


@app.route("/api/record", methods=["POST"])
def proxy_record():
    return _proxy("POST", "/api/record", request.get_json(silent=True) or {})


@app.route("/api/recording_status")
def proxy_status():
    return _proxy("GET", "/api/recording_status", timeout=5)


@app.route("/api/record/cancel", methods=["POST"])
def proxy_cancel():
    return _proxy("POST", "/api/record/cancel", request.get_json(silent=True) or {})


@app.route("/api/baseline", methods=["POST"])
def proxy_baseline():
    return _proxy("POST", "/api/baseline", request.get_json(silent=True) or {})


@app.route("/api/baseline_status")
def proxy_baseline_status():
    return _proxy("GET", "/api/baseline_status", timeout=5)


@app.route("/api/device/reboot", methods=["POST"])
def proxy_reboot():
    return _proxy("POST", "/api/device/reboot", request.get_json(silent=True) or {})


@app.route("/patient/<patient_id>/report.pdf")
def proxy_report(patient_id):
    return _proxy("GET", f"/api/patient/{patient_id}/report.pdf",
                  stream=True, timeout=30)


@app.route("/patient/<patient_id>/log.csv")
def proxy_log(patient_id):
    return _proxy("GET", f"/api/patient/{patient_id}/log.csv",
                  stream=True, timeout=30)


# ───────────────────────────────────────────────────────────────────────
# AI screening report — TCN inference + Ollama explanation
# ───────────────────────────────────────────────────────────────────────

@app.route("/api/patient/<patient_id>/visit/<visit_id>/ai_report",
           methods=["POST", "GET"])
def api_ai_report(patient_id, visit_id):
    """
    Pipeline: embed_<visit_id>.json (camera output)
              → models.inference  → embed_<visit_id>_result.json
              → parkinson_agent._agent.explain_prediction → text report
    Returns the inference JSON + the LLM text in a single response so the
    portal can render a unified report box.
    """
    paths = _ai_paths(patient_id, visit_id)
    if not paths["embed"].exists():
        return jsonify({
            "error": "embed JSON not found for this visit",
            "expected_path": str(paths["embed"]),
        }), 404

    # Asset checks — fail clearly if model files missing
    for p, name in [(MODEL_PATH, "model"), (SCALER_PATH, "scaler"),
                    (FEATURES_PATH, "features")]:
        if not p.exists():
            return jsonify({
                "error": f"{name} file missing — cannot run inference",
                "expected_path": str(p),
            }), 500

    cache_hit_result = paths["result"].exists()
    cache_hit_explanation = paths["explanation"].exists()

    # 1) Inference (skip if cached)
    if not cache_hit_result:
        cmd = [
            "/usr/bin/python3", "-m", "models.inference",
            "--model",    str(MODEL_PATH),
            "--scaler",   str(SCALER_PATH),
            "--features", str(FEATURES_PATH),
            "--input",    str(paths["embed"]),
            "--output",   str(paths["result"]),
        ]
        proc = subprocess.run(
            cmd, cwd=str(GDGAIHACK_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not paths["result"].exists():
            return jsonify({
                "error": "inference failed",
                "stderr_tail": (proc.stderr or "")[-800:],
                "stdout_tail": (proc.stdout or "")[-300:],
                "cmd": " ".join(cmd),
            }), 500

    # 2) Read the inference result
    try:
        with open(paths["result"]) as f:
            result = _json.load(f)
    except Exception as e:
        return jsonify({"error": f"could not parse inference result: {e}"}), 500

    # 3) Explanation — from cache, else Ollama on demand (~1-3 s on qwen2.5:0.5B)
    explanation = None
    agent_error = None
    if cache_hit_explanation:
        try:
            explanation = paths["explanation"].read_text()
        except Exception as e:
            agent_error = f"cache read failed: {e}"
    if explanation is None:
        try:
            from parkinson_agent._agent import explain_prediction  # type: ignore
            explanation = explain_prediction(str(paths["result"]))
            if explanation:
                try:
                    paths["explanation"].write_text(explanation)
                except Exception:
                    pass  # cache is best-effort
        except Exception as e:
            agent_error = str(e)

    return jsonify({
        "patient_id":         result.get("patient_id", patient_id),
        "visit_id":           result.get("visit_id", visit_id),
        "prediction":         result.get("prediction"),
        "feature_importance": result.get("feature_importance"),
        "preprocessing":      result.get("preprocessing"),
        "ground_truth":       result.get("ground_truth"),
        "inference_time_ms":  result.get("inference_time_ms"),
        "explanation_text":   explanation,
        "agent_error":        agent_error,
        "cached": {
            "result":      cache_hit_result,
            "explanation": cache_hit_explanation,
        },
        "result_path":        str(paths["result"].relative_to(NEUROSIGHT_DIR)),
    })


# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    print(f"[portal] NeuroVista Doctor Portal on port {port}")
    print(f"[portal] Proxies to tech dashboard at {TECH_DASHBOARD}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
