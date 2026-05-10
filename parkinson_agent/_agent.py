import json
import ollama
import os
import re


def _strip_truncated_tail(text: str) -> str:
    """qwen2.5:0.5b often hits the token cap mid-bullet, leaving fragments
    like `**What` or trailing `- ` with nothing after. Trim back to the last
    line that looks complete (ends with `.`, `)`, `]`, or a closed `**...**`)
    so the rendered report doesn't show a dangling stub.
    """
    if not text:
        return text
    lines = text.rstrip().splitlines()
    # Drop incomplete trailing lines, but only while we still have other
    # lines to keep. If we'd reduce the report to nothing, fall through to
    # a character-level cleanup of the remaining single line instead.
    while len(lines) > 1:
        last = lines[-1].rstrip()
        if not last:
            lines.pop()
            continue
        unclosed_bold = (last.count("**") % 2) != 0
        bare_header = (
            re.match(r"^\s*[-*]?\s*\*\*[^*]*\*\*?\s*:?\s*$", last) is not None
            and not last.endswith(".")
        )
        ends_open = last.endswith((",", ":", "-", "—", "–", "(", "["))
        if unclosed_bold or bare_header or ends_open:
            lines.pop()
            continue
        break
    out = "\n".join(lines).rstrip()
    # Final char-level: if everything still ends on dangling punctuation,
    # strip it so the report doesn't trail off.
    out = re.sub(r"[,\-–—:]+\s*$", "", out).rstrip()
    return out


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
    system_prompt = (
        "You are a clinical screening assistant for early Parkinson's "
        "disease. Read the JSON below from an upstream TCN classifier and "
        "produce a clinical report for the neurologist.\n\n"
        "OUTPUT FORMAT — strict:\n"
        "- Plain prose, NO markdown, NO bullet points, NO bold (**), "
        "NO backticks, NO headers.\n"
        "- Three short paragraphs separated by a blank line:\n"
        "  1. Risk assessment (low / borderline / elevated) with the "
        "     pd_probability value and confidence pct.\n"
        "  2. Key findings: cite the top 1–2 tremor regions with their "
        "     band ratio and dominant frequency in Hz, and any standout "
        "     velocity finding.\n"
        "  3. Clinical recommendation: 1 sentence on whether deeper "
        "     assessment is warranted and what to look for clinically.\n"
        "- Use real em-dashes (—), not the literal escape \\u2014.\n"
        "- Total length: 80–120 words.\n"
        "- Be conservative — this is screening, not diagnosis.\n"
        "- Write so a busy doctor reads it in 15 seconds.\n"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        # --- ACTUAL DATA: The JSON from your TCN ---
        {"role": "user", "content": target_data_str},
    ]

    # 3. Call the Ollama Python API
    try:
        response = ollama.chat(
            model='qwen2.5:0.5b',
            messages=messages,
            options={
                "temperature": 0.1,
                # qwen2.5:0.5b is verbose with bullet lists. Generous budget
                # so we naturally hit a stop token; trailing-fragment guard
                # below cleans up the rare overflow.
                "num_predict": 500,
            }
        )
        raw = response['message']['content']
        return _strip_truncated_tail(raw)
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