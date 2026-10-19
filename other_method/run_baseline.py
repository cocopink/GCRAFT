"""
Baseline methods runner for WikiTQ and TabFact.
Supports: End-to-End QA, Few-Shot QA, Chain-of-Thought, CoT-Consist.

Usage:
    python other_method/run_baseline.py \
        --dataset wikitq --method e2e --model glm-5 \
        --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --first_n 100 --n_proc 8

    python other_method/run_baseline.py \
        --dataset tabfact --method cot_consist --n_sample 5 \
        --model glm-5 --base_url https://dashscope.aliyuncs.com/compatible-mode/v1
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from other_method.prompts import build_prompt

# Reuse existing data loaders and table formatters
from thought.TableQA.utils.load_data import load_wikitq_dataset
from thought.TableQA.utils.helper import table2string as wikitq_table2string
from thought.TableQA.utils.evaluate import wikitq_match_func_for_samples

from thought.TableFV.utils.load_data import load_tabfact_dataset
from thought.TableFV.utils.helper import table2string as tabfact_table2string
from thought.TableFV.utils.evaluate import tabfact_match_func_for_samples


# ===========================================================================
# LLM wrapper (reuses the project's OpenAI-compatible API pattern)
# ===========================================================================

class LLM:
    def __init__(self, model_name, key, base_url, token_log_dir=None, run_tag=""):
        self.model_name = model_name
        self.key = key
        self.base_url = base_url
        self._token_log_dir = token_log_dir
        self._run_tag = run_tag
        if token_log_dir:
            os.makedirs(token_log_dir, exist_ok=True)

    def _log_token_usage(self, prompt_tokens, completion_tokens):
        """Append token usage to log file named by method+dataset+pid."""
        if self._token_log_dir is None:
            return
        tag = self._run_tag or "unknown"
        log_path = os.path.join(self._token_log_dir, f"token_{tag}_{os.getpid()}.jsonl")
        entry = {
            "tag": tag,
            "pid": os.getpid(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def generate(self, prompt, temperature=0, max_tokens=512, n_sample=1):
        """Call LLM and return list of (text, log_conf) tuples."""
        from openai import OpenAI

        options = dict(
            temperature=temperature,
            n=n_sample,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        # Qwen3.x models default thinking=ON; disabling it forces reasoning
        # into content, breaking answer extraction.  Keep it for other models.
        if not self.model_name.startswith(("gpt-", "qwen3.6-plus")):
            options["extra_body"] = {"enable_thinking": False}

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        client = OpenAI(api_key=self.key, base_url=self.base_url, timeout=600.0)
        retry_num = 0
        retry_limit = 2
        error = None

        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model_name, messages=messages, **options
                )
                error = None
                break
            except Exception as e:
                err_str = str(e)
                if "data_inspection_failed" in err_str or "inappropriate content" in err_str:
                    # content moderation — return placeholder
                    return [("", -999.0)] * n_sample
                elif "maximum context length" in err_str:
                    return [("", -999.0)] * n_sample
                elif retry_num > retry_limit:
                    error = err_str
                    break
                else:
                    print(f"[RETRY] {err_str}", flush=True)
                    time.sleep(60)
                    retry_num += 1

        if error:
            raise Exception(error)

        # Log token usage
        if hasattr(response, 'usage') and response.usage:
            self._log_token_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

        results = []
        for i, choice in enumerate(response.choices):
            text = choice.message.content
            # Some API backends return content as a list (especially with n>1);
            # coerce to plain string so downstream extractors never see a list.
            if isinstance(text, list):
                text = "\n".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in text
                )
            fake_conf = np.log((len(response.choices) - i) / len(response.choices))
            results.append((text, fake_conf))

        return results


# ===========================================================================
# Table formatting helpers
# ===========================================================================

def format_table(sample, dataset):
    """Convert sample's table_text to linearized string."""
    if dataset == "wikitq":
        return wikitq_table2string(sample["table_text"])
    elif dataset == "tabfact":
        caption = sample.get("table_caption", None)
        return tabfact_table2string(sample["table_text"], caption=caption)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ===========================================================================
# Answer extraction
# ===========================================================================

def extract_answer_e2e(response, dataset):
    """Extract answer from E2E / Few-Shot direct response."""
    text = response.strip()
    # Remove trailing period
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def extract_answer_cot(response, dataset):
    """Extract answer from CoT response (after 'therefore, the answer is:')."""
    text = response.strip()

    # Try multiple patterns
    patterns = [
        r"[Tt]herefore,\s*the answer is:\s*(.+?)(?:\.|$)",
        r"[Tt]herefore, the answer is:\s*(.+?)(?:\.|$)",
        r"[Tt]he answer is:\s*(.+?)(?:\.|$)",
        r"[Aa]nswer:\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer.endswith("."):
                answer = answer[:-1].strip()
            return answer

    # Fallback: return last line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[-1] if lines else text


def extract_answer(response, method, dataset):
    """Dispatch to appropriate extractor."""
    if method in ("e2e", "few_shot"):
        return extract_answer_e2e(response, dataset)
    elif method in ("cot", "cot_consist"):
        return extract_answer_cot(response, dataset)
    else:
        raise ValueError(f"Unknown method: {method}")


# ===========================================================================
# Per-sample execution (for multiprocessing)
# ===========================================================================

def _exec_one_sample(args):
    """Run one sample through the baseline method."""
    sample, dataset, method, model_name, api_key, base_url, n_sample, token_log_dir, run_tag = args

    llm = LLM(model_name, api_key, base_url, token_log_dir=token_log_dir, run_tag=run_tag)
    table_str = format_table(sample, dataset)
    query = sample["statement"]

    # Build prompt (CoT-Consist uses the same CoT prompt)
    prompt_method = "cot" if method == "cot_consist" else method
    prompt = build_prompt(dataset, prompt_method, table_str, query)

    if method == "cot_consist":
        # Sample N times with temperature > 0
        temperature = 0.7
        results = llm.generate(prompt, temperature=temperature, max_tokens=512, n_sample=n_sample)
        answers = [extract_answer(text, "cot", dataset) for text, _ in results]
        # Majority vote
        counter = Counter(answers)
        best_answer, _ = counter.most_common(1)[0]
        final_answer = best_answer
    else:
        # Single generation
        results = llm.generate(prompt, temperature=0, max_tokens=512)
        final_answer = extract_answer(results[0][0], method, dataset)

    # Populate chain in the format expected by evaluators
    sample["chain"] = [
        {"name": "final_query", "parameter_and_conf": [(final_answer, 0.0)]}
    ]
    return sample


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baseline methods")
    parser.add_argument("--dataset", type=str, required=True, choices=["wikitq", "tabfact"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["e2e", "few_shot", "cot", "cot_consist"])
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--base_url", type=str, default="https://yunwu.ai/v1")
    parser.add_argument("--api_key", type=str, default=os.environ.get("YUNWU_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    parser.add_argument("--first_n", type=int, default=-1, help="Number of samples (-1 for all)")
    parser.add_argument("--n_proc", type=int, default=8, help="Number of parallel processes")
    parser.add_argument("--n_sample", type=int, default=5, help="CoT-Consist: number of samples")
    parser.add_argument("--output_dir", type=str, default="results/other_method")
    parser.add_argument("--chunk_size", type=int, default=4, help="Batch size for multiprocessing")
    args = parser.parse_args()

    # ---- Load dataset ----
    print(f"Loading {args.dataset} dataset...", flush=True)
    if args.dataset == "wikitq":
        dataset_path = os.path.join(PROJECT_ROOT, "thought/TableQA/data/wikitq/test_lower.jsonl")
        dataset = load_wikitq_dataset(dataset_path, tag="test", first_n=args.first_n)
    else:
        dataset_path = os.path.join(PROJECT_ROOT, "thought/TableFV/data/tabfact/test.jsonl")
        raw2clean_path = os.path.join(PROJECT_ROOT, "thought/TableFV/data/tabfact/raw2clean.jsonl")
        dataset = load_tabfact_dataset(dataset_path, raw2clean_path, tag="test", first_n=args.first_n)

    print(f"Loaded {len(dataset)} samples", flush=True)

    # ---- Run method ----
    print(f"Running {args.method} on {args.dataset} with {args.model}...", flush=True)
    start_time = time.time()

    # Token log directory for this run
    token_log_dir = os.path.join(
        PROJECT_ROOT, args.output_dir, args.dataset, args.method, args.model, "token_logs"
    )
    os.makedirs(token_log_dir, exist_ok=True)

    run_tag = f"{args.dataset}_{args.method}"

    task_args = [
        (sample, args.dataset, args.method, args.model, args.api_key, args.base_url, args.n_sample, token_log_dir, run_tag)
        for sample in dataset
    ]

    if args.n_proc > 1:
        with Pool(args.n_proc) as pool:
            results = list(tqdm(
                pool.imap(_exec_one_sample, task_args, chunksize=args.chunk_size),
                total=len(task_args),
                desc=args.method,
            ))
    else:
        results = []
        for arg in tqdm(task_args, desc=args.method):
            results.append(_exec_one_sample(arg))

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.1f}s", flush=True)

    # ---- Evaluate ----
    print("Evaluating...", flush=True)
    if args.dataset == "wikitq":
        tagged_path = os.path.join(PROJECT_ROOT, "thought/TableQA/data/wikitq/tagged_data")
        acc = wikitq_match_func_for_samples(results, strategy="top", tagged_dataset_path=tagged_path)
    else:
        acc = tabfact_match_func_for_samples(results, strategy="top")

    # ---- Collect token usage ----
    token_stats = _collect_token_usage(token_log_dir)

    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset} | Method: {args.method} | Model: {args.model}")
    print(f"Accuracy: {acc:.4f} ({sum(1 for s in results if _is_correct(s, args.dataset))}/{len(results)})")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tokens: input={token_stats['input_tokens']:,} | output={token_stats['output_tokens']:,} | total={token_stats['total_tokens']:,} | calls={token_stats['api_calls']}")
    print(f"{'='*60}", flush=True)

    # ---- Save results ----
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir, args.dataset, args.method, args.model)
    os.makedirs(output_dir, exist_ok=True)

    result_path = os.path.join(output_dir, "final_result.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(results, f)

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "model": args.model,
        "accuracy": acc,
        "total_samples": len(results),
        "elapsed_seconds": elapsed,
        "n_sample": args.n_sample if args.method == "cot_consist" else 1,
        "token_usage": token_stats,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to {output_dir}", flush=True)


def _collect_token_usage(log_dir):
    """Aggregate per-process token logs into total statistics."""
    import glob
    total_input = 0
    total_output = 0
    total_calls = 0
    log_files = glob.glob(os.path.join(log_dir, "token_*.jsonl"))
    for lf in log_files:
        with open(lf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                total_input += entry.get("prompt_tokens", 0)
                total_output += entry.get("completion_tokens", 0)
                total_calls += 1
    return {
        "api_calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
    }


def _is_correct(sample, dataset):
    """Quick correctness check for summary stats."""
    try:
        if dataset == "wikitq":
            tagged_path = os.path.join(PROJECT_ROOT, "thought/TableQA/data/wikitq/tagged_data")
            from thought.TableQA.utils.evaluate import (
                to_value_list, check_denotation, tsv_unescape_list
            )
            target_values_map = {}
            for filename in os.listdir(tagged_path):
                if filename[0] == '.':
                    continue
                filename = os.path.join(tagged_path, filename)
                with open(filename, 'r', 'utf8') as fin:
                    header = fin.readline().rstrip('\n').split('\t')
                    for line in fin:
                        stuff = dict(zip(header, line.rstrip('\n').split('\t')))
                        ex_id = stuff['id']
                        original_strings = tsv_unescape_list(stuff['targetValue'])
                        canon_strings = tsv_unescape_list(stuff['targetCanon'])
                        target_values_map[ex_id] = to_value_list(original_strings, canon_strings)
            res = sample["chain"][-1]["parameter_and_conf"][0][0]
            pred = [res.lower()] if '|' not in res else [r for r in res.lower().split('|')]
            pred = to_value_list(pred)
            return check_denotation(target_values_map[sample['ids']], pred)
        else:
            from thought.TableFV.utils.evaluate import tabfact_match_func
            return tabfact_match_func(sample)
    except:
        return False


if __name__ == "__main__":
    main()
