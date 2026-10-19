import multiprocessing as mp
import pickle
import subprocess
import fire
import os
import sys
from tqdm import tqdm
sys.path.insert(0, 'critic/TableQA')
sys.path.insert(0, 'refine/TableQA')
sys.path.insert(0, '.')
from utils.read_pkl import read_pkl
from utils.extract_step import return_incorrect_max_step
from utils.llm import LLM
from utils.helper import *
from utils.evaluate import *
from utils.chain import *
from operations import *
from tools import read_pkl
from utils.controller import controller_main_loop


def _resolve_openai_api_key(openai_api_key):
    if openai_api_key == "<env:OPENAI_API_KEY>":
        return os.environ.get("OPENAI_API_KEY", "EMPTY")
    return openai_api_key


def _refine_one_sample_mp_core(arg):
    """Worker function for multiprocessing refinement of a single sample."""
    (llm, sample, sample_idx, llm_options, cache_dir,
     use_clarifier, thought_results_dir, max_iterations, frozen_memory,
     use_verifier, use_diff_critic, diff_critic_mode, router_variant) = arg
    try:
        refined = controller_main_loop(
            sample, llm=llm, llm_options=llm_options,
            max_iterations=max_iterations, cache_dir=cache_dir,
            sample_idx=sample_idx, use_clarifier=use_clarifier,
            thought_results_dir=thought_results_dir,
            frozen_memory=frozen_memory,
            use_verifier=use_verifier,
            use_diff_critic=use_diff_critic,
            diff_critic_mode=diff_critic_mode,
            router_variant=router_variant,
        )
        return sample_idx, refined
    except Exception as e:
        print(f"Error refining sample {sample_idx}: {e}")
        return sample_idx, sample


def main(
    thought_results_dir: str = "",
    refine_results_dir: str = "",
    base_url="http://localhost:11434/v1",
    openai_api_key="EMPTY",
    model_name="qwen3:14b",
    first_n=-1,
    n_proc=1,
    chunk_size=1,
    use_clarifier: bool = True,
    frozen_memory: bool = False,
    use_verifier: bool = False,
    use_diff_critic: bool = False,
    diff_critic_mode: str = "diagnose_only",
    router_variant: str = "",
):
    openai_api_key = _resolve_openai_api_key(openai_api_key)

    # Auto-switch results directory
    mode_dir = "new"
    model_dir = f"wikitq/{model_name}"
    if not thought_results_dir:
        thought_results_dir = f"results/{mode_dir}/thought/{model_dir}"
    if not refine_results_dir:
        refine_results_dir = f"results/{mode_dir}/refine/{model_dir}"

    result_pkl = os.path.join(thought_results_dir, "final_result.pkl")

    if first_n != -1:
        all_samples = read_pkl(result_pkl)[:first_n]
    else:
        all_samples = read_pkl(result_pkl)

    gpt_llm = LLM(
        model_name=model_name,
        key=openai_api_key,
        base=base_url
    )

    # Create results directory structure
    os.makedirs(refine_results_dir, exist_ok=True)
    cache_dir = os.path.join(refine_results_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 启用token日志
    gpt_llm.set_token_log_dir(refine_results_dir, run_tag="wikitq_refine")

    # Use controller-based refinement
    print("Using controller-based refinement...")
    print(f"Clarifier enabled: {use_clarifier}")
    print(f"Frozen memory enabled: {frozen_memory}")
    print(f"Verifier enabled: {use_verifier}")
    print(f"Diff-Critic enabled: {use_diff_critic} (mode: {diff_critic_mode})")
    print(f"Router variant: {router_variant or 'none (standard FULL)'}")
    print(f"Concurrency: n_proc={n_proc}, chunk_size={chunk_size}")

    # Filter out None samples (failed in thought stage)
    valid_entries = []
    for idx, sample in enumerate(all_samples):
        if sample is None:
            print(f"Warning: Sample {idx} is None (failed in thought stage), skipping...")
            valid_entries.append((idx, None))
        else:
            valid_entries.append((idx, sample))

    llm_options = gpt_llm.get_model_options(
        temperature=0,
        per_example_max_decode_steps=2048,
        per_example_top_p=1
    )

    if n_proc > 1:
        # Multiprocessing mode: mp.Pool + imap_unordered
        args_list = [
            (gpt_llm, sample, idx, llm_options, cache_dir,
             use_clarifier, thought_results_dir, 2, frozen_memory,
             use_verifier, use_diff_critic, diff_critic_mode,
             router_variant or None)
            for idx, sample in valid_entries if sample is not None
        ]

        refined_samples = [None] * len(all_samples)
        with mp.Pool(n_proc) as pool:
            for ret_idx, refined in tqdm(
                pool.imap_unordered(_refine_one_sample_mp_core, args_list, chunksize=chunk_size),
                total=len(args_list),
                desc="Controller-based refinement (mp)",
            ):
                refined_samples[ret_idx] = refined
    else:
        # Single-threaded mode (original behavior)
        refined_samples = [None] * len(all_samples)
        for idx, sample in tqdm(enumerate(all_samples), total=len(all_samples), desc="Controller-based refinement"):
            if sample is None:
                continue

            refined_sample = controller_main_loop(
                sample,
                llm=gpt_llm,
                llm_options=llm_options,
                max_iterations=2,
                cache_dir=cache_dir,
                sample_idx=idx,
                use_clarifier=use_clarifier,
                thought_results_dir=thought_results_dir,
                frozen_memory=frozen_memory,
                use_verifier=use_verifier,
                router_variant=router_variant or None,
                use_diff_critic=use_diff_critic,
                diff_critic_mode=diff_critic_mode,
            )
            refined_samples[idx] = refined_sample

    refine_list = refined_samples

    # Print route distribution
    if router_variant:
        route_stats = {}
        for s in refine_list:
            if s is not None:
                route = s.get("_route", "UNKNOWN")
                route_stats[route] = route_stats.get(route, 0) + 1
        total = sum(route_stats.values())
        if total > 0:
            print(f"\n[ROUTER] Route distribution ({router_variant}):")
            for route, count in route_stats.items():
                pct = count / total * 100
                print(f"  {route}: {count} ({pct:.1f}%)")

    # Calculate accuracy
    acc = wikitq_match_func_for_samples(refine_list)
    print("Accuracy:", acc)

    # Save results
    with open(os.path.join(refine_results_dir, "result.txt"), "w") as f:
        f.write(f'Accuracy: {acc}\n')

    pickle.dump(
        refine_list, open(os.path.join(refine_results_dir, "final_result.pkl"), "wb")
    )

    # Save accuracy
    with open(os.path.join(refine_results_dir, "acc.txt"), "w") as f:
        f.write(f"Refine Stage Accuracy: {acc}\n")

    # 汇总并保存token统计
    token_usage = LLM.collect_token_usage(
        refine_results_dir,
        output_path=os.path.join(refine_results_dir, "token_usage.json")
    )
    if token_usage["total_tokens"] > 0:
        print(f"\nRefine Stage Token Usage:")
        print(f"  Input Tokens:  {token_usage['input_tokens']:,}")
        print(f"  Output Tokens: {token_usage['output_tokens']:,}")
        print(f"  Total Tokens: {token_usage['total_tokens']:,} ({token_usage['api_calls']} calls)")
    else:
        print(f"\nRefine Stage: no new API calls (all cached).")


if __name__ == "__main__":
    fire.Fire(main)
