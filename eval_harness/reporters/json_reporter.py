import json
from typing import List, Dict, Any
from ..tasks.base import EvalResult

def export_json(results: List[EvalResult], filepath: str, model_name: str):
    """Export structured evaluation results to disk."""
    payload: Dict[str, Any] = {
        "model": model_name,
        "results": {}
    }
    for r in results:
        payload["results"][r.task_name] = {
            "total_samples": r.total_samples,
            "metrics": r.metrics,
            "details": r.sample_outputs
        }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
