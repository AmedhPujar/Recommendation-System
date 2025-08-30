import json, re

def extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None
