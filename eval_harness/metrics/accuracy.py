import re

def normalize_numeric_string(s: str) -> str:
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        val = match.group(0)
        if "." in val:
            f = float(val)
            return str(int(f)) if f.is_integer() else f"{f:.4f}".rstrip("0").rstrip(".")
        return str(int(val))
    return s.strip()

def exact_match_score(pred: str, target: str, numeric: bool = True) -> float:
    if numeric:
        return 1.0 if normalize_numeric_string(pred) == normalize_numeric_string(target) else 0.0
    return 1.0 if pred.strip().lower() == target.strip().lower() else 0.0
