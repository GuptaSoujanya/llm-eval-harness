import argparse
from .config import EvalConfig
from .models.mock_provider import MockLLMProvider
from .tasks.gsm8k import GSM8KTask
from .tasks.mmlu import MMLUTask

def main():
    parser = argparse.ArgumentParser(prog="llm-eval")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3-8B-Instruct")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "mmlu"])
    args = parser.parse_args()

    provider = MockLLMProvider(args.model)
    task_map = {"gsm8k": GSM8KTask(), "mmlu": MMLUTask()}

    print(f"\n[llm-eval-harness] Evaluating '{args.model}' across {args.tasks}...")
    print("=" * 52)
    print("| Benchmark Task | Total Samples | Metric | Value |")
    print("| :--- | :--- | :--- | :--- |")
    for t_name in args.tasks:
        if t_name in task_map:
            t = task_map[t_name]
            s = t.load_samples()
            outs = [provider.generate(x.prompt).text for x in s]
            res = t.evaluate(s, outs)
            for m, val in res.metrics.items():
                print(f"| `{res.task_name}` | {res.total_samples} | `{m}` | **{val*100:.2f}%** |")
    print("=" * 52 + "\n")

if __name__ == "__main__":
    main()
