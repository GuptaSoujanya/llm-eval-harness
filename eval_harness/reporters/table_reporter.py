from typing import List
from ..tasks.base import EvalResult

def render_table(results: List[EvalResult]) -> str:
    """Format evaluation results into a clean markdown table."""
    lines = [
        "| Benchmark Task | Total Samples | Metric | Value |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        for metric_name, val in r.metrics.items():
            formatted_val = f"{val * 100:.2f}%" if "acc" in metric_name or "pass" in metric_name or "rate" in metric_name else f"{val:.4f}"
            lines.append(f"| `{r.task_name}` | {r.total_samples} | `{metric_name}` | **{formatted_val}** |")
    return "\n".join(lines)
