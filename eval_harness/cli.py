import argparse
import sys
from .config import EvalConfig
from .runner import EvaluationRunner
from .reporters.table_reporter import render_table
from .reporters.json_reporter import export_json

def main():
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="llm-eval-harness: Extensible evaluation for open-source & API LLMs."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct", help="Target model identifier")
    parser.add_argument("--provider", type=str, choices=["mock", "hf"], default="mock", help="Inference backend provider")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "mmlu"], help="List of benchmark tasks (gsm8k, mmlu, humaneval)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap evaluation samples per task")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write JSON results report")

    args = parser.parse_args()

    config = EvalConfig(
        model_name=args.model,
        provider=args.provider,
        tasks=args.tasks,
        max_samples=args.max_samples,
        temperature=args.temperature
    )

    print(f"\n[llm-eval-harness] Initiating evaluation for '{config.model_name}' on {config.tasks}...")
    runner = EvaluationRunner(config)
    results = runner.run()

    print("\n" + "="*50)
    print("           BENCHMARK RESULTS")
    print("="*50)
    table = render_table(results)
    print(table)
    print("="*50 + "\n")

    if args.output_json:
        export_json(results, args.output_json, model_name=args.model)
        print(f"[llm-eval-harness] Results exported to {args.output_json}")

if __name__ == "__main__":
    main()
