"""M0: Build a fixed pilot index set from thought/refine result pkl files.

Usage:
    python scripts/build_refine_pilot_set.py \
        --thought-pkl "results/new/thought/wikitq/qwen3:14b/final_result.pkl" \
        --output "results/pilot/wikitq/pilot_indices_100.json" \
        --size 100
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")


def _is_correct(sample: Optional[Dict[str, Any]]) -> bool:
    """Judge sample correctness by checking the 'judge' field."""
    if sample is None:
        return False
    judge = str(sample.get("judge", ""))
    return "[Correct]" in judge


def build_pilot_indices(
    thought_samples: List[Optional[Dict[str, Any]]],
    baseline_samples: Optional[List[Optional[Dict[str, Any]]]] = None,
    size: int = 100,
    seed: int = 42,
) -> List[int]:
    """Build a deterministic pilot index list.

    Priority: baseline wrong cases first, then correct cases.
    If baseline_samples is None, use thought_samples for correctness.
    """
    rng = __import__("random").Random(seed)

    wrong_indices: List[int] = []
    correct_indices: List[int] = []

    for idx in range(len(thought_samples)):
        sample = thought_samples[idx]
        if sample is None:
            continue
        if baseline_samples is not None and idx < len(baseline_samples):
            baseline = baseline_samples[idx]
            is_correct = _is_correct(baseline)
        else:
            is_correct = _is_correct(sample)

        if is_correct:
            correct_indices.append(idx)
        else:
            wrong_indices.append(idx)

    rng.shuffle(wrong_indices)
    rng.shuffle(correct_indices)

    merged = wrong_indices + correct_indices
    return sorted(merged[:size])


def main():
    parser = argparse.ArgumentParser(description="Build pilot index set for experiments")
    parser.add_argument("--thought-pkl", required=True, help="Path to thought stage final_result.pkl")
    parser.add_argument("--baseline-pkl", default=None, help="Path to baseline refine final_result.pkl (optional)")
    parser.add_argument("--output", required=True, help="Output path for pilot_indices.json")
    parser.add_argument("--size", type=int, default=100, help="Number of samples in pilot (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Read thought pkl
    from refine.TableQA.utils.read_pkl import read_pkl
    thought_samples = read_pkl(args.thought_pkl)

    # Read baseline pkl (optional)
    baseline_samples = None
    if args.baseline_pkl:
        baseline_samples = read_pkl(args.baseline_pkl)

    indices = build_pilot_indices(
        thought_samples,
        baseline_samples=baseline_samples,
        size=args.size,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(indices, f, indent=2)

    print(f"Saved {len(indices)} indices to {args.output}")


if __name__ == "__main__":
    main()
