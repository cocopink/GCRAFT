# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import subprocess
import fire
import os
import sys
import pickle
sys.path.insert(0, 'thought/TableQA')
sys.path.insert(0, '.')
sys.path.insert(0, 'critic/TableQA')
from utils.load_data import load_wikitq_dataset
from utils.llm import LLM
from utils.helper import *
from utils.evaluate import *
from utils.chain import *
from operations import *
from tools.read_pkl import read_pkl


def _resolve_openai_api_key(openai_api_key):
    if openai_api_key == "<env:OPENAI_API_KEY>":
        return os.environ.get("OPENAI_API_KEY", "EMPTY")
    return openai_api_key


def main(
    dataset_path: str = "thought/TableQA/data/wikitq/test_lower.jsonl",
    thought_results_dir: str = "",
    base_url="",
    openai_api_key="EMPTY",
    model_name="qwen2.5-72b-instruct",
    first_n=-1,
    n_proc=8,
    chunk_size=4,
    use_clarifier: bool = True,
):
    openai_api_key = _resolve_openai_api_key(openai_api_key)

    # Auto-switch results directory based on mode
    mode_dir = "new" if use_clarifier else "orig"
    if not thought_results_dir:
        thought_results_dir = f"results/{mode_dir}/thought/wikitq"

    dataset = load_wikitq_dataset(dataset_path, first_n=first_n)

    gpt_llm = LLM(
        model_name=model_name,
        key=openai_api_key,
        base=base_url
    )

    os.makedirs(thought_results_dir, exist_ok=True)

    # 启用token日志
    gpt_llm.set_token_log_dir(thought_results_dir, run_tag="wikitq_thought")

    if use_clarifier:
        from agents.clarifier_agent import ClarifierAgent, create_clarifier_result_path

        # Initialize ClarifierAgent and extract schema anchors
        print("Initializing ClarifierAgent for schema anchoring...")
        clarifier = ClarifierAgent(llm=gpt_llm)
        dataset = clarifier.clarify_batch(dataset)
        print(f"Clarified {len(dataset)} samples")

        # Save clarifier results
        clarifier_dir = os.path.join(thought_results_dir, "clarifier")
        os.makedirs(clarifier_dir, exist_ok=True)

        for sample in dataset:
            sample_id = sample.get('id', 'unknown')
            clarifier_path = os.path.join(clarifier_dir, f'case_dict_{sample_id}.pkl')
            pickle.dump(
                sample['clarifier'],
                open(clarifier_path, "wb")
            )
        print(f"Saved clarifier results to {clarifier_dir}")

    proc_samples, _ = dynamic_chain_exec_with_cache_mp(
        dataset,
        llm=gpt_llm,
        llm_options=gpt_llm.get_model_options(
            temperature=0.0, per_example_max_decode_steps=2048, per_example_top_p=1.0
        ),
        strategy="top",
        cache_dir=os.path.join(thought_results_dir, "cache"),
        n_proc=n_proc,
        chunk_size=chunk_size,
    )
    fixed_chain = [
        (
            "Simple query",
            simple_query,
            dict(use_demo=True),
            dict(
                temperature=0, per_example_max_decode_steps=2048, per_example_top_p=1.0
            ),
        ),
    ]
    
    final_path = os.path.join(thought_results_dir, "final_result.pkl")
    if os.path.exists(final_path):
        final_result = read_pkl(final_path)
    else:
        final_result, _ = fixed_chain_exec_mp(gpt_llm, proc_samples, fixed_chain, n_proc=4, chunk_size=2)

        pickle.dump(
            final_result, open(os.path.join(thought_results_dir, "final_result.pkl"), "wb")
        )


    # Calculate and save accuracy
    from utils.evaluate import wikitq_match_func_for_samples
    acc = wikitq_match_func_for_samples(final_result)
    print(f"Thought Stage Accuracy: {acc}")
    with open(os.path.join(thought_results_dir, "acc.txt"), "w") as f:
        f.write(f"Thought Stage Accuracy: {acc}\n")

    # 汇总并保存token统计
    token_usage = LLM.collect_token_usage(
        thought_results_dir,
        output_path=os.path.join(thought_results_dir, "token_usage.json")
    )
    if token_usage["total_tokens"] > 0:
        print(f"\nThought Stage Token Usage:")
        print(f"  Input Tokens:  {token_usage['input_tokens']:,}")
        print(f"  Output Tokens: {token_usage['output_tokens']:,}")
        print(f"  Total Tokens: {token_usage['total_tokens']:,} ({token_usage['api_calls']} calls)")
    else:
        print(f"\nThought Stage: no new API calls (all cached).")


if __name__ == "__main__":
    fire.Fire(main)
64000
