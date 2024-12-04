import re
from typing import Optional

def normalize_numeric_string(s: str) -> str:
    """Extract and normalize numerical answer string (strip commas, spaces, currency symbols)."""
    if not s:
        return ""
    # Strip currency, spaces, percent signs
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", s)
    if match:
        val = match.group(0)
        try:
            if "." in val:
                f_val = float(val)
                if f_val.is_integer():
                    return str(int(f_val))
                return f"{f_val:.4f}".rstrip("0").rstrip(".")
            return str(int(val))
        except ValueError:
            return val
    return s.strip()

def exact_match_score(prediction: str, target: str, numeric: bool = True) -> float:
    """Compute binary exact match between prediction and target."""
    if numeric:
        pred_norm = normalize_numeric_string(prediction)
        target_norm = normalize_numeric_string(target)
        return 1.0 if pred_norm and pred_norm == target_norm else 0.0
    return 1.0 if prediction.strip().lower() == target.strip().lower() else 0.0
