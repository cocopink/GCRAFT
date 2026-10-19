"""M0: Summarize pairwise experiment comparison (baseline vs experiment).

Usage:
    python scripts/summarize_refine_experiment.py \
        --baseline-pkl "results/new/refine/wikitq/qwen3:14b/final_result.pkl" \
        --experiment-pkl "results/new/refine/wikitq/qwen3:14b/final_result.pkl" \
        --experiment-token-json "results/new/refine/wikitq/qwen3:14b/token_usage.json"
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")


def _is_correct(sample: Optional[Dict[str, Any]]) -> bool:
    if sample is None:
        return False
    judge = str(sample.get("judge", ""))
    return "[Correct]" in judge


def _calc_accuracy(samples: List[Optional[Dict[str, Any]]]) -> float:
    valid = [s for s in samples if s is not None]
    if not valid:
        return 0.0
    correct = sum(1 for s in valid if _is_correct(s))
    return correct / len(valid)


def summarize_pairwise(
    base_samples: List[Optional[Dict[str, Any]]],
    exp_samples: List[Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute pairwise comparison metrics."""
    n = min(len(base_samples), len(exp_samples))
    base_correct = [_is_correct(base_samples[i]) for i in range(n)]
    exp_correct = [_is_correct(exp_samples[i]) for i in range(n)]

    return {
        "total_samples": n,
        "baseline_accuracy": sum(base_correct) / n if n else 0.0,
        "experiment_accuracy": sum(exp_correct) / n if n else 0.0,
        "baseline_correct": sum(base_correct),
        "experiment_correct": sum(exp_correct),
        "baseline_wrong_to_correct": sum((not b) and e for b, e in zip(base_correct, exp_correct)),
        "baseline_correct_to_wrong": sum(b and (not e) for b, e in zip(base_correct, exp_correct)),
        "changed_count": sum(b != e for b, e in zip(base_correct, exp_correct)),
    }


def _load_token_usage(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment comparison")
    parser.add_argument("--baseline-pkl", required=True, help="Baseline result pkl")
    parser.add_argument("--experiment-pkl", required=True, help="Experiment result pkl")
    parser.add_argument("--experiment-token-json", default=None, help="Experiment token_usage.json (optional)")
    parser.add_argument("--output", default=None, help="Output markdown file (optional, prints to stdout)")
    args = parser.parse_args()

    from refine.TableQA.utils.read_pkl import read_pkl

    base_samples = read_pkl(args.baseline_pkl)
    exp_samples = read_pkl(args.experiment_pkl)
    token_usage = _load_token_usage(args.experiment_token_json)

    stats = summarize_pairwise(base_samples, exp_samples)

    # Build report
    lines = [
        "# Experiment Comparison Report",
        "",
        "## Accuracy",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total samples | {stats['total_samples']} |",
        f"| Baseline accuracy | {stats['baseline_accuracy']:.4f} |",
        f"| Experiment accuracy | {stats['experiment_accuracy']:.4f} |",
        f"| Baseline correct | {stats['baseline_correct']} |",
        f"| Experiment correct | {stats['experiment_correct']} |",
        "",
        "## Changes",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline wrong → correct | {stats['baseline_wrong_to_correct']} |",
        f"| Baseline correct → wrong | {stats['baseline_correct_to_wrong']} |",
        f"| Total changed | {stats['changed_count']} |",
    ]

    if token_usage:
        n = stats["total_samples"]
        lines += [
            "",
            "## Token Usage (Experiment)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| API calls | {token_usage.get('api_calls', 'N/A')} |",
            f"| Avg calls/sample | {token_usage.get('api_calls', 0) / n:.1f} |" if n else "",
            f"| Input tokens | {token_usage.get('input_tokens', 'N/A')} |",
            f"| Output tokens | {token_usage.get('output_tokens', 'N/A')} |",
            f"| Total tokens | {token_usage.get('total_tokens', 'N/A')} |",
            f"| Avg tokens/sample | {token_usage.get('total_tokens', 0) / n:.0f} |" if n else "",
        ]

    report = "\n".join(lines)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
