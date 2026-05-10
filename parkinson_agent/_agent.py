import json
import ollama
import os

def explain_prediction(json_filepath):
    # 1. Read the JSON file
    if not os.path.exists(json_filepath):
        return f"Error: Could not find {json_filepath}"
        
    with open(json_filepath, 'r') as f:
        data = json.load(f)

    # Convert the JSON to a formatted string so the LLM can read it easily
    target_data_str = json.dumps(data, indent=2)

    print(f"\nSending {data.get('patient_id', 'Unknown')} data to local Ollama (Qwen2.5:0.5b)...\n")

    # 2. Few-Shot Chat History
    messages = [
        {
        """REPORT GUIDANCE:
    - `overall_risk_level`:
        'low'        — no convincing motor signs, normal expressivity.
        'borderline' — one weak/ambiguous sign or quality issues.
        'elevated'   — at least one strong sign (e.g. composite score clearly
                       reduced AND in-band jaw tremor) or convergent signs.
    - Add a `MotorSign` entry only if you have a numeric tool output that
      supports it. Cite the tool name(s) in `evidence_tool_calls`.
    - If a tool returns `valid: false`, log a `quality_issues` entry and avoid
      asserting that sign with high confidence.
    - Be conservative. This is screening for clinician review, not diagnosis.
    - Write `clinician_notes` so a busy doctor can read it in 15 seconds."""
        },
        # --- ACTUAL DATA: The JSON from your TCN ---
        {
            "role": "user",
            "content": target_data_str
        }
    ]

    # 3. Call the Ollama Python API
    try:
        response = ollama.chat(
            model='qwen2.5:0.5b',
            messages=messages,
            options={
                "temperature": 0.1,    # Extremely low temperature so it sticks strictly to the data
                "num_predict": 150     # Max tokens to keep the output concise
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Failed to connect to Ollama. Make sure the Ollama app is running and the model is pulled. Error: {e}"

if __name__ == "__main__":
    # Point this to your actual JSON file
    json_filename = "embed_2026-05-10_035241_result.json"
    
    explanation = explain_prediction(json_filename)
    
    print("="*70)
    print(" AI CLINICAL EXPLANATION (Powered by Qwen2.5:0.5B via Ollama) ")
    print("="*70)
    print(explanation)
    print("="*70)