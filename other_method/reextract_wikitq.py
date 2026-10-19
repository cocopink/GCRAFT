"""
Re-extract answers from existing few_shot/e2e pkl results and re-evaluate accuracy.
Saves extra_ans.json alongside summary.json for each method.

Usage:
    python other_method/reextract_wikitq.py
    python other_method/reextract_wikitq.py --method few_shot --model gpt-5.4
"""

import argparse
import json
import os
import pickle
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from thought.TableQA.utils.evaluate import wikitq_match_func_for_samples


# ===========================================================================
# Smart answer extraction
# ===========================================================================

# Patterns ordered by specificity (most specific first)
ANSWER_PATTERNS = [
    r"[Tt]herefore,\s*the answer is:\s*(.+)",
    r"[Tt]he answer is:\s*(.+)",
    r"[Aa]nswer:\s*(.+)",
]


def _clean_answer(raw: str) -> str:
    """Post-process extracted answer: strip period, markdown bold, whitespace."""
    ans = raw.strip()
    # Remove trailing period
    if ans.endswith("."):
        ans = ans[:-1].strip()
    # Remove markdown bold **xxx**
    ans = re.sub(r"\*\*(.+?)\*\*", r"\1", ans)
    return ans.strip()


def smart_extract(raw_text: str):
    """
    Extract short answer from raw model output.

    Returns:
        (answer, strategy) where strategy is one of:
            "pattern"     - matched a known answer pattern
            "last_line"   - fell back to last non-empty line
            "full_text"   - returned entire text as-is
    """
    text = raw_text.strip()
    if not text:
        return text, "full_text"

    # Strategy 1: regex pattern match (search from end to prefer last occurrence)
    for pattern in ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, text, re.MULTILINE))
        if matches:
            answer = _clean_answer(matches[-1].group(1))
            if answer:
                return answer, "pattern"

    # Strategy 2: last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return _clean_answer(lines[-1]), "last_line"

    return text, "full_text"


# ===========================================================================
# Per-method processing
# ===========================================================================

def process_method(model_dir: str, tagged_path: str):
    """
    Load pkl, re-extract answers, evaluate accuracy.
    Returns dict with results or None if pkl not found.
    """
    pkl_path = os.path.join(model_dir, "final_result.pkl")
    if not os.path.exists(pkl_path):
        return None

    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    stats = {"pattern": 0, "last_line": 0, "full_text": 0}

    for sample in results:
        raw = sample["chain"][-1]["parameter_and_conf"][0][0]
        new_answer, strategy = smart_extract(raw)
        sample["chain"][-1]["parameter_and_conf"][0] = (new_answer, 0.0)
        stats[strategy] = stats.get(strategy, 0) + 1

    acc = wikitq_match_func_for_samples(
        results, strategy="top", tagged_dataset_path=tagged_path
    )

    return {
        "accuracy": acc,
        "total_samples": len(results),
        "extraction_stats": stats,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Re-extract and re-evaluate WikiTQ answers")
    parser.add_argument("--method", type=str, default=None,
                        help="Specific method to process (default: all)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to process (default: all)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base results directory (default: results/other_method/wikitq)")
    args = parser.parse_args()

    base_dir = args.base_dir or os.path.join(PROJECT_ROOT, "results", "other_method", "wikitq")
    tagged_path = os.path.join(PROJECT_ROOT, "thought/TableQA/data/wikitq/tagged_data")

    methods = [args.method] if args.method else ["e2e", "few_shot", "cot", "cot_consist"]

    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if not os.path.isdir(method_dir):
            print(f"[SKIP] {method}/ directory not found")
            continue

        for model_name in sorted(os.listdir(method_dir)):
            if args.model and model_name != args.model:
                continue

            model_dir = os.path.join(method_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            summary_path = os.path.join(model_dir, "summary.json")
            if not os.path.exists(summary_path):
                print(f"[SKIP] {method}/{model_name}/summary.json not found")
                continue

            with open(summary_path) as f:
                original = json.load(f)

            print(f"Processing {method}/{model_name} ...", flush=True)
            result = process_method(model_dir, tagged_path)
            if result is None:
                print(f"  [SKIP] No final_result.pkl")
                continue

            extra = {
                "dataset": "wikitq",
                "method": method,
                "model": model_name,
                "accuracy": result["accuracy"],
                "original_accuracy": original.get("accuracy"),
                "total_samples": result["total_samples"],
                "extraction_stats": result["extraction_stats"],
            }

            extra_path = os.path.join(model_dir, "extra_ans.json")
            with open(extra_path, "w") as f:
                json.dump(extra, f, indent=2)

            orig_acc = original.get("accuracy", float("nan"))
            print(f"  original accuracy:    {orig_acc:.4f}")
            print(f"  re-extracted accuracy: {result['accuracy']:.4f}")
            print(f"  extraction stats:      {result['extraction_stats']}")
            print(f"  saved -> {extra_path}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
