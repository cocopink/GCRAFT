"""只针对 error_results.pkl 重跑 thought，并注入 ATG context。

示例:
    python scripts/rerun_error_results_with_atg_context.py \
        --dataset wikitq \
        --input-pkl test/results/thought/wikitq/gpt-5.4/error_results.pkl \
        --output-dir test/results/thought/wikitq/gpt-5.4_error_atgctx \
        --mode final

    python scripts/rerun_error_results_with_atg_context.py \
        --dataset tabfact \
        --input-pkl test/results/thought/tabfact/gpt-5.4/error_results.pkl \
        --output-dir test/results/thought/tabfact/gpt-5.4_error_atgctx \
        --mode full
"""

import argparse
import json
import os
import pickle
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from table_structure.context import build_atg_context


def _drop_trailing_simple_query(sample: Dict) -> Dict:
    sample = deepcopy(sample)
    chain = list(sample.get("chain", []))
    if chain and chain[-1].get("operation_name") == "simple_query":
        sample["chain"] = chain[:-1]
    return sample


def _prepare_samples(input_pkl: Path, limit: int) -> List[Dict]:
    samples = [s for s in pickle.load(open(input_pkl, "rb")) if s is not None]
    if limit > 0:
        samples = samples[:limit]
    prepared = []
    for sample in samples:
        sample = _drop_trailing_simple_query(sample)
        context = build_atg_context(sample, merge_similar_values=True)
        if context:
            sample["atg_context"] = context
        prepared.append(sample)
    return prepared


def _write_report(output_dir: Path, report: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "repair_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _run_tabfact(args, samples: List[Dict]) -> Dict:
    sys.path.insert(0, str(PROJECT_ROOT / "thought/TableFV"))
    sys.path.insert(0, str(PROJECT_ROOT / "critic/TableFV"))

    from operations import simple_query
    from utils.chain import dynamic_chain_exec_with_cache_mp, fixed_chain_exec_mp
    from utils.evaluate import tabfact_match_func_for_samples
    from utils.llm import LLM

    llm = LLM(model_name=args.model_name, key=args.openai_api_key, base=args.base_url)
    llm.set_token_log_dir(str(args.output_dir), run_tag="tabfact_error_atgctx")

    if args.mode == "full":
        chain_inputs = []
        for sample in samples:
            reset_sample = deepcopy(sample)
            reset_sample["chain"] = []
            chain_inputs.append(reset_sample)
        proc_samples, _ = dynamic_chain_exec_with_cache_mp(
            chain_inputs,
            llm=llm,
            llm_options=llm.get_model_options(
                temperature=0.0,
                per_example_max_decode_steps=2048,
                per_example_top_p=1.0,
            ),
            strategy="top",
            cache_dir=str(args.output_dir / "cache"),
            n_proc=args.n_proc,
            chunk_size=args.chunk_size,
        )
    else:
        proc_samples = samples

    fixed_chain = [
        (
            "Simple query",
            simple_query,
            dict(use_demo=True),
            dict(temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0),
        )
    ]
    final_result, _ = fixed_chain_exec_mp(
        llm, proc_samples, fixed_chain, n_proc=args.n_proc, chunk_size=args.chunk_size
    )
    acc = tabfact_match_func_for_samples(final_result)
    return {"final_result": final_result, "accuracy": acc}


def _run_wikitq(args, samples: List[Dict]) -> Dict:
    sys.path.insert(0, str(PROJECT_ROOT / "thought/TableQA"))
    sys.path.insert(0, str(PROJECT_ROOT / "critic/TableQA"))

    from operations import simple_query
    from utils.chain import dynamic_chain_exec_with_cache_mp, fixed_chain_exec_mp
    from utils.evaluate import wikitq_match_func_for_samples
    from utils.llm import LLM

    llm = LLM(model_name=args.model_name, key=args.openai_api_key, base=args.base_url)
    llm.set_token_log_dir(str(args.output_dir), run_tag="wikitq_error_atgctx")

    if args.mode == "full":
        chain_inputs = []
        for sample in samples:
            reset_sample = deepcopy(sample)
            reset_sample["chain"] = []
            chain_inputs.append(reset_sample)
        proc_samples, _ = dynamic_chain_exec_with_cache_mp(
            chain_inputs,
            llm=llm,
            llm_options=llm.get_model_options(
                temperature=0.0,
                per_example_max_decode_steps=2048,
                per_example_top_p=1.0,
            ),
            strategy="top",
            cache_dir=str(args.output_dir / "cache"),
            n_proc=args.n_proc,
            chunk_size=args.chunk_size,
        )
    else:
        proc_samples = samples

    fixed_chain = [
        (
            "Simple query",
            simple_query,
            dict(use_demo=True),
            dict(temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0),
        )
    ]
    final_result, _ = fixed_chain_exec_mp(
        llm, proc_samples, fixed_chain, n_proc=args.n_proc, chunk_size=args.chunk_size
    )
    acc = wikitq_match_func_for_samples(final_result)
    return {"final_result": final_result, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tabfact", "wikitq"])
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["final", "full"], default="final")
    parser.add_argument("--base-url", default="https://113.44.247.131:47851/v1")
    parser.add_argument("--openai-api-key", default=os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--n-proc", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = _prepare_samples(args.input_pkl, args.limit)
    prepared_path = args.output_dir / "prepared_error_inputs.pkl"
    pickle.dump(samples, open(prepared_path, "wb"))

    if args.dataset == "tabfact":
        result = _run_tabfact(args, samples)
    else:
        result = _run_wikitq(args, samples)

    output_pkl = args.output_dir / "final_result.pkl"
    pickle.dump(result["final_result"], open(output_pkl, "wb"))
    report = {
        "dataset": args.dataset,
        "input_pkl": str(args.input_pkl),
        "output_pkl": str(output_pkl),
        "mode": args.mode,
        "num_samples": len(samples),
        "accuracy_on_error_results": result["accuracy"],
        "fixed_ratio_target": 0.5,
        "passes_50_percent_target": result["accuracy"] >= 0.5,
    }
    _write_report(args.output_dir, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
