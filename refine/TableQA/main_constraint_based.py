"""
Constraint-Based Table Reasoning Main Entry
基于约束诱导式剪枝推理的主入口
"""

import pickle
import subprocess
import fire
import os
from tqdm import tqdm

from utils.constraint_aware_chain import constraint_aware_chain_exec_with_cache_mp
from utils.constraint_state import ConstraintState
from utils.constraint_induction import ConstraintInducer
from utils.llm import LLM
from utils.read_pkl import read_pkl


def main(
    thought_results_dir: str = "results/thought/wikitq",
    constraint_results_dir: str = "results/constraint_based/wikitq",
    base_url: str = "",
    openai_api_key: str = "EMPTY",
    model_name: str = "qwen2.5-72b-instruct",
    first_n: int = -1,
    n_proc: int = 1,
    chunk_size: int = 1,
    max_rounds: int = 5,
    strategy: str = "voting",
    temperature: float = 0.5,
    use_cache: bool = True,
    clear_cache: bool = False,
):
    """
    执行约束诱导式剪枝推理
    
    Args:
        thought_results_dir: 初始推理结果目录
        constraint_results_dir: 约束推理结果目录
        base_url: LLM API base URL
        openai_api_key: LLM API密钥
        model_name: 模型名称
        first_n: 处理前N个样本（-1表示全部）
        n_proc: 进程数
        chunk_size: 块大小
        max_rounds: 最大推理轮次
        strategy: 推理策略（"top"或"voting"）
        temperature: 温度参数
        use_cache: 是否使用缓存
        clear_cache: 是否清除缓存
    """
    
    os.makedirs(constraint_results_dir, exist_ok=True)
    
    # 读取初始推理结果
    result_pkl = os.path.join(thought_results_dir, "final_result.pkl")
    
    if not os.path.exists(result_pkl):
        print(f"Error: {result_pkl} not found!")
        print("Please run the thought phase first.")
        return
    
    if first_n != -1:
        all_samples = read_pkl(result_pkl)[:first_n]
    else:
        all_samples = read_pkl(result_pkl)
    
    print(f"Loaded {len(all_samples)} samples from {result_pkl}")
    
    # 初始化LLM
    gpt_llm = LLM(
        model_name=model_name,
        key=openai_api_key,
        base=base_url
    )
    
    # 设置LLM选项
    llm_options = gpt_llm.get_model_options(
        temperature=temperature,
        per_example_max_decode_steps=2048,
        per_example_top_p=1.0
    )
    
    # 缓存目录
    cache_dir = os.path.join(constraint_results_dir, "cache")
    
    # 清除缓存
    if clear_cache and os.path.exists(cache_dir):
        subprocess.run(["rm", "-rf", cache_dir], check=True)
        print(f"Cleared cache: {cache_dir}")
    
    # 执行约束感知的推理
    print(f"\nStarting constraint-aware reasoning...")
    print(f"Strategy: {strategy}")
    print(f"Max rounds: {max_rounds}")
    print(f"Temperature: {temperature}")
    print(f"Processes: {n_proc}")
    print(f"Cache: {'enabled' if use_cache else 'disabled'}")
    print()
    
    result_samples, constraint_states, reasoning_logs = constraint_aware_chain_exec_with_cache_mp(
        all_samples=all_samples,
        llm=gpt_llm,
        llm_options=llm_options,
        strategy=strategy,
        cache_dir=cache_dir,
        n_proc=n_proc,
        chunk_size=chunk_size,
        max_rounds=max_rounds
    )
    
    # 保存结果
    print("\nSaving results...")
    
    # 保存最终结果
    final_result_path = os.path.join(constraint_results_dir, "final_result.pkl")
    pickle.dump(result_samples, open(final_result_path, "wb"))
    print(f"Saved final results to {final_result_path}")
    
    # 保存约束状态
    constraint_state_path = os.path.join(constraint_results_dir, "constraint_states.pkl")
    pickle.dump(constraint_states, open(constraint_state_path, "wb"))
    print(f"Saved constraint states to {constraint_state_path}")
    
    # 保存推理日志
    reasoning_log_path = os.path.join(constraint_results_dir, "reasoning_logs.pkl")
    pickle.dump(reasoning_logs, open(reasoning_log_path, "wb"))
    print(f"Saved reasoning logs to {reasoning_log_path}")
    
    # 分析和统计
    print("\n" + "="*60)
    print("Constraint-Based Reasoning Statistics")
    print("="*60)
    
    analyze_results(result_samples, constraint_states, reasoning_logs)
    
    # 清理缓存
    if use_cache and os.path.exists(cache_dir):
        try:
            subprocess.run(["rm", "-rf", cache_dir], check=True)
            print(f"\nCleaned up cache: {cache_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clean cache: {e}")


def analyze_results(
    result_samples: list,
    constraint_states: list,
    reasoning_logs: list
) -> None:
    """
    分析结果并打印统计信息
    
    Args:
        result_samples: 处理后的样本列表
        constraint_states: 约束状态列表
        reasoning_logs: 推理日志列表
    """
    total_samples = len(result_samples)
    correct_samples = sum(
        1 for s in result_samples 
        if s and s.get('conclusion') == '[Correct]'
    )
    
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # 约束统计
    total_constraints = 0
    total_rounds = 0
    constraint_types = {
        "row_constraints": 0,
        "column_constraints": 0,
        "operation_constraints": 0,
        "aggregation_constraints": 0
    }
    
    for cs in constraint_states:
        if cs:
            summary = cs.get_constraint_summary()
            total_constraints += summary['total_constraints']
            total_rounds += summary['current_round']
            
            for ctype in constraint_types:
                constraint_types[ctype] += summary['constraint_breakdown'].get(ctype, 0)
    
    print(f"\nConstraint Statistics:")
    print(f"Total constraints added: {total_constraints}")
    print(f"Average constraints per sample: {total_constraints/total_samples:.2f}" if total_samples > 0 else "N/A")
    print(f"Total reasoning rounds: {total_rounds}")
    print(f"Average rounds per sample: {total_rounds/total_samples:.2f}" if total_samples > 0 else "N/A")
    
    print(f"\nConstraint Types:")
    for ctype, count in constraint_types.items():
        print(f"  {ctype}: {count}")
    
    # 推理轮次分布
    round_distribution = {}
    for log in reasoning_logs:
        if log:
            rounds = len(log)
            round_distribution[rounds] = round_distribution.get(rounds, 0) + 1
    
    print(f"\nReasoning Round Distribution:")
    for rounds in sorted(round_distribution.keys()):
        count = round_distribution[rounds]
        percentage = count / total_samples * 100 if total_samples > 0 else 0
        print(f"  {rounds} round(s): {count} ({percentage:.1f}%)")


def evaluate_constraints(
    constraint_results_dir: str,
    output_file: str = None
) -> None:
    """
    评估约束效果
    
    Args:
        constraint_results_dir: 约束推理结果目录
        output_file: 输出文件路径（可选）
    """
    import json
    
    # 读取结果
    result_pkl = os.path.join(constraint_results_dir, "final_result.pkl")
    constraint_state_pkl = os.path.join(constraint_results_dir, "constraint_states.pkl")
    reasoning_log_pkl = os.path.join(constraint_results_dir, "reasoning_logs.pkl")
    
    if not all(os.path.exists(p) for p in [result_pkl, constraint_state_pkl, reasoning_log_pkl]):
        print("Error: Missing result files!")
        return
    
    result_samples = pickle.load(open(result_pkl, "rb"))
    constraint_states = pickle.load(open(constraint_state_pkl, "rb"))
    reasoning_logs = pickle.load(open(reasoning_log_pkl, "rb"))
    
    # 详细分析
    analysis = {
        "total_samples": len(result_samples),
        "correct_samples": sum(1 for s in result_samples if s and s.get('conclusion') == '[Correct]'),
        "constraint_statistics": {},
        "round_distribution": {},
        "sample_details": []
    }
    
    # 约束统计
    for idx, cs in enumerate(constraint_states):
        if cs:
            summary = cs.get_constraint_summary()
            analysis["constraint_statistics"][idx] = summary
    
    # 轮次分布
    for idx, log in enumerate(reasoning_logs):
        if log:
            rounds = len(log)
            analysis["round_distribution"][idx] = rounds
            
            # 样本详情
            sample_detail = {
                "sample_id": result_samples[idx].get('id', idx) if result_samples[idx] else idx,
                "rounds": rounds,
                "final_conclusion": result_samples[idx].get('conclusion') if result_samples[idx] else None,
                "constraint_summary": constraint_states[idx].get_constraint_summary() if constraint_states[idx] else None
            }
            analysis["sample_details"].append(sample_detail)
    
    # 输出结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Saved analysis to {output_file}")
    else:
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
