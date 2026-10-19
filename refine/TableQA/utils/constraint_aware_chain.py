"""
Constraint-Aware Chain Execution Module
约束感知的推理链执行，支持约束诱导式剪枝推理
"""

import copy
import re
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .constraint_state import ConstraintState
from .constraint_induction import ConstraintInducer
from .helper import table2string
from .extract_step import return_incorrect_max_step


def constraint_aware_chain_exec_one_sample(
    sample: Dict,
    llm,
    llm_options: Optional[Dict] = None,
    strategy: str = "top",
    debug: bool = False,
    constraint_state: Optional[ConstraintState] = None,
    constraint_inducer: Optional[ConstraintInducer] = None,
    max_rounds: int = 5,
    operation_parameter_dict: Optional[Dict] = None,
) -> Tuple[Dict, ConstraintState, List[Dict]]:
    """
    约束感知的单样本推理执行
    
    Args:
        sample: 输入样本
        llm: LLM实例
        llm_options: LLM选项
        strategy: 推理策略（"top"或"voting"）
        debug: 是否启用调试模式
        constraint_state: 约束状态（如果为None则创建新的）
        constraint_inducer: 约束归纳器
        max_rounds: 最大推理轮次
        operation_parameter_dict: 操作参数字典
        
    Returns:
        Tuple[Dict, ConstraintState, List[Dict]]: 
            - 处理后的样本
            - 最终约束状态
            - 推理日志
    """
    # 初始化约束状态
    if constraint_state is None:
        constraint_state = ConstraintState()
    
    # 初始化约束归纳器
    if constraint_inducer is None:
        constraint_inducer = ConstraintInducer(llm=llm)
    
    # 初始化操作参数字典
    if operation_parameter_dict is None:
        from operations import (
            add_column_func, select_row_func, select_column_func,
            group_column_func, sort_column_func
        )
        operation_parameter_dict = {
            "add_column": (
                "addColumn",
                add_column_func,
                {},
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
            "select_row": (
                "selectRow",
                select_row_func,
                {},
                llm.get_model_options(
                    temperature=0.5,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                    n_sample=8,
                ),
            ),
            "select_column": (
                "selectColumn",
                select_column_func,
                {},
                llm.get_model_options(
                    temperature=0.5,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                    n_sample=8,
                ),
            ),
            "group_column": (
                "groupColumn",
                group_column_func,
                {"skip_op": []},
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
            "sort_column": (
                "sortColumn",
                sort_column_func,
                {"skip_op": []},
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
        }
    
    # 推理日志
    reasoning_log = []
    
    # 多轮推理
    current_sample = copy.deepcopy(sample)
    for round_num in range(max_rounds):
        round_log = {
            "round": round_num + 1,
            "constraints_before": constraint_state.get_constraint_summary(),
            "reasoning_steps": []
        }
        
        # 执行推理链
        try:
            current_sample = _execute_reasoning_chain(
                current_sample,
                llm,
                llm_options,
                strategy,
                debug,
                constraint_state,
                operation_parameter_dict,
                round_log
            )
        except Exception as e:
            round_log["error"] = str(e)
            reasoning_log.append(round_log)
            break
        
        # 检查推理是否正确
        if current_sample.get('conclusion') == '[Correct]':
            round_log["status"] = "success"
            reasoning_log.append(round_log)
            break
        
        # 如果推理失败，使用Critic进行评估
        # 注意：这里需要从critic/TableQA/tools导入
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../critic/TableQA'))
        from tools.multiprocess import critic_exec_one_sample
        
        critic_sample = critic_exec_one_sample(
            current_sample, 
            "random",  # error_route
            llm=llm, 
            llm_options=llm_options
        )
        
        round_log["critique"] = critic_sample.get('critique', '')
        round_log["conclusion"] = critic_sample.get('conclusion', '')
        
        # 如果Critic判定正确，结束推理
        if critic_sample.get('conclusion') == '[Correct]':
            round_log["status"] = "success_after_critic"
            reasoning_log.append(round_log)
            break
        
        # 诱导约束
        incorrect_step, max_step = return_incorrect_max_step(critic_sample)
        if incorrect_step:
            constraint_state, induction_result = constraint_inducer.induce_constraints(
                critic_sample.get('critique', ''),
                incorrect_step,
                critic_sample,
                constraint_state
            )
            round_log["induction_result"] = induction_result
            round_log["constraints_after"] = constraint_state.get_constraint_summary()
        
        round_log["status"] = "refining"
        reasoning_log.append(round_log)
        
        # 准备下一轮推理
        current_sample = critic_sample
    
    return current_sample, constraint_state, reasoning_log


def _execute_reasoning_chain(
    sample: Dict,
    llm,
    llm_options: Optional[Dict],
    strategy: str,
    debug: bool,
    constraint_state: ConstraintState,
    operation_parameter_dict: Dict,
    round_log: Dict
) -> Dict:
    """
    执行推理链（单轮）
    
    Args:
        sample: 输入样本
        llm: LLM实例
        llm_options: LLM选项
        strategy: 推理策略
        debug: 是否启用调试
        constraint_state: 约束状态
        operation_parameter_dict: 操作参数字典
        round_log: 轮次日志
        
    Returns:
        Dict: 处理后的样本
    """
    current_sample = copy.deepcopy(sample)
    
    # 获取表格信息
    from .chain import get_table_info
    table_info = get_table_info(current_sample)
    
    # 添加约束到prompt
    constraints_prompt = constraint_state.get_constraints_for_prompt()
    
    while True:
        # 生成下一个操作
        next_operation = _generate_next_operation_with_constraints(
            current_sample,
            table_info,
            constraints_prompt,
            llm,
            llm_options,
            strategy,
            debug,
            constraint_state
        )
        
        step_log = {
            "operation": next_operation,
            "constraints_applied": constraints_prompt
        }
        
        if debug:
            print(f"Next operation: {next_operation}")
        
        if next_operation == "<END>":
            step_log["status"] = "completed"
            round_log["reasoning_steps"].append(step_log)
            break
        
        # 检查操作是否被约束禁止
        # 提取操作参数（从LLM响应中）
        operation_params = []
        if next_operation != "<END>":
            # 从table_info中获取最近的操作参数
            if table_info.get("act_chain"):
                last_act = table_info["act_chain"][-1]
                # 尝试从操作字符串中提取参数
                if "select_row" in last_act:
                    import re
                    params_match = re.search(r'select_row\((.*?)\)', last_act)
                    if params_match:
                        params_str = params_match.group(1)
                        # 解析参数
                        import ast
                        try:
                            params = ast.literal_eval(params_str)
                            if isinstance(params, list):
                                operation_params = [str(p) for p in params]
                        except:
                            pass
                elif "select_column" in last_act:
                    import re
                    params_match = re.search(r'select_column\((.*?)\)', last_act)
                    if params_match:
                        params_str = params_match.group(1)
                        import ast
                        try:
                            params = ast.literal_eval(params_str)
                            if isinstance(params, list):
                                operation_params = [str(p).strip().strip('"\'') for p in params]
                        except:
                            pass
                elif "group_column" in last_act:
                    import re
                    params_match = re.search(r'group_column\((.*?)\)', last_act)
                    if params_match:
                        param = params_match.group(1).strip().strip('"\'')
                        operation_params = [param]
        
        if _is_operation_forbidden_by_constraints(
            next_operation, 
            operation_params,
            constraint_state
        ):
            step_log["status"] = "skipped_by_constraint"
            step_log["reason"] = f"Operation {next_operation}({', '.join(operation_params)}) is forbidden by constraints"
            round_log["reasoning_steps"].append(step_log)
            continue
        
        # 执行操作
        param = operation_parameter_dict[next_operation]
        op_name, solver_func, kargs, op_llm_options = param
        
        # 将约束传递给操作函数
        kargs = kargs.copy()
        kargs['constraint_state'] = constraint_state
        
        try:
            current_sample = solver_func(
                current_sample, 
                table_info, 
                llm=llm, 
                llm_options=op_llm_options, 
                **kargs
            )
            step_log["status"] = "executed"
        except Exception as e:
            step_log["status"] = "failed"
            step_log["error"] = str(e)
        
        round_log["reasoning_steps"].append(step_log)
        
        # 更新表格信息
        table_info = get_table_info(current_sample)
    
    return current_sample


def _generate_next_operation_with_constraints(
    sample: Dict,
    table_info: Dict,
    constraints_prompt: str,
    llm,
    llm_options: Optional[Dict],
    strategy: str,
    debug: bool,
    constraint_state: ConstraintState
) -> str:
    """
    生成下一个操作（考虑约束）
    
    Args:
        sample: 输入样本
        table_info: 表格信息
        constraints_prompt: 约束prompt
        llm: LLM实例
        llm_options: LLM选项
        strategy: 推理策略
        debug: 是否启用调试
        constraint_state: 约束状态
        
    Returns:
        str: 下一个操作名称
    """
    from .chain import possible_next_operation_dict, get_operation_name
    
    act_chain = table_info["act_chain"]
    
    if debug:
        print("Act Chain: ", act_chain, flush=True)
    
    kept_act_chain = [x for x in act_chain if not x.startswith("skip")]
    kept_act_chain_str = " -> ".join(kept_act_chain)
    if kept_act_chain_str:
        kept_act_chain_str += " ->"
    
    skip_act_chain = [x for x in act_chain if x.startswith("skip")]
    skip_act_chain_op_names = []
    for op in skip_act_chain:
        op = op[len("skip ") :]
        op_name = get_operation_name(op)
        skip_act_chain_op_names.append(op_name)
    
    last_operation = (
        "<init>" if not kept_act_chain else get_operation_name(kept_act_chain[-1])
    )
    
    # 获取可能的下一个操作
    possible_next_operations = possible_next_operation_dict[last_operation]
    
    # 过滤掉被跳过的操作
    possible_next_operations = [
        x for x in possible_next_operations if x not in skip_act_chain_op_names
    ]
    
    # 如果只有一个可能的操作，直接返回
    if len(possible_next_operations) == 1:
        return possible_next_operations[0]
    
    # 构建prompt（包含约束）
    prompt = constraints_prompt + "\n"
    
    # 添加操作示例
    from .chain import (
        plan_add_column_demo, plan_select_column_demo, 
        plan_select_row_demo, plan_group_column_demo, 
        plan_sort_column_demo, plan_full_demo_simple
    )
    
    for operation in possible_next_operations:
        if operation == "<END>":
            continue
        prompt += eval(f"plan_{operation}_demo") + "\n\n"
    
    prompt += plan_full_demo_simple + "\n\n"
    
    # 添加表格和问题
    prompt += "/*\n" + table2string(table_info["table_text"]) + "\n*/\n"
    prompt += "Question: " + sample["statement"] + "\n"
    
    _possible_next_operations_str = " or ".join(
        [f"f_{op}()" if op != "<END>" else op for op in possible_next_operations]
    )
    
    if len(possible_next_operations) > 1:
        prompt += (
            f"The next operation must be one of {_possible_next_operations_str}.\n"
        )
    else:
        prompt += f"The next operation must be {_possible_next_operations_str}.\n"
    
    prompt += "Function Chain: " + kept_act_chain_str
    
    # 生成响应
    responses = llm.generate_plus_with_score(
        prompt, options=llm_options, end_str="\n\n"
    )
    
    # 选择操作
    if strategy == "top":
        response = responses[0][0]
        generate_operations = _get_all_operation_names(response)
        if debug:
            print('Response:', response)
            print("Generated Operations: ", generate_operations)
        
        next_operation = "<END>"
        for operation in generate_operations:
            if operation in possible_next_operations:
                next_operation = operation
                break
    elif strategy == "voting":
        from collections import defaultdict
        next_operation_conf_dict = defaultdict(float)
        for response, score in responses:
            generate_operations = _get_all_operation_names(response)
            for operation in generate_operations:
                if operation in possible_next_operations:
                    next_operation_conf_dict[operation] += np.exp(score)
        
        if len(next_operation_conf_dict) != 0:
            next_operation_conf_pairs = sorted(
                next_operation_conf_dict.items(), key=lambda x: x[1], reverse=True
            )
            next_operation = next_operation_conf_pairs[0][0]
        else:
            next_operation = "<END>"
    
    return next_operation


def _is_operation_forbidden_by_constraints(
    operation_name: str,
    operation_params: List[str],
    table_info: Dict,
    constraint_state: ConstraintState
) -> bool:
    """
    检查操作是否被约束禁止
    
    Args:
        operation_name: 操作名称
        operation_params: 操作参数列表
        table_info: 表格信息
        constraint_state: 约束状态
        
    Returns:
        bool: 是否被禁止
    """
    # 检查操作约束
    if constraint_state.is_operation_forbidden(operation_name, operation_params):
        return True
    
    # 检查行约束（对于select_row操作）
    if operation_name == "select_row" and operation_params:
        for param in operation_params:
            # 提取行号
            import re
            row_match = re.search(r'row\s*(\d+)', param, re.IGNORECASE)
            if row_match:
                row_id = row_match.group(1)
                if constraint_state.is_row_forbidden(f"row {row_id}"):
                    return True
    
    # 检查列约束（对于select_column操作）
    if operation_name == "select_column" and operation_params:
        for param in operation_params:
            # 清理列名
            column_name = param.strip().strip('"\'')
            if constraint_state.is_column_forbidden(column_name):
                return True
    
    # 检查聚合约束（对于group_column操作）
    if operation_name == "group_column" and operation_params:
        for param in operation_params:
            column_name = param.strip().strip('"\'')
            # 检查是否有强制聚合约束
            suggested_column = constraint_state.get_aggregation_constraint("group_column")
            if suggested_column and column_name != suggested_column:
                # 如果有强制约束，且当前参数不匹配，则禁止
                return True
    
    return False


def _get_all_operation_names(string: str) -> List[str]:
    """
    从字符串中提取所有操作名称
    
    Args:
        string: 包含操作调用的字符串
        
    Returns:
        List[str]: 操作名称列表
    """
    operation_names = []
    parts = string.split("->")
    for part in parts:
        part = part.strip()
        if part == "<END>":
            operation_names.append("<END>")
        else:
            res = re.findall(r"f_(.*?)\(.*\)", part)
            if res:
                operation_names.append(res[0])
    return operation_names


def constraint_aware_chain_exec_with_cache_mp(
    all_samples: List[Dict],
    llm,
    llm_options: Optional[Dict] = None,
    strategy: str = "voting",
    cache_dir: str = "./results/constraint_aware",
    n_proc: int = 10,
    chunk_size: int = 50,
    max_rounds: int = 5,
) -> Tuple[List[Dict], List[ConstraintState], List[List[Dict]]]:
    """
    约束感知的多进程推理执行
    
    Args:
        all_samples: 所有样本
        llm: LLM实例
        llm_options: LLM选项
        strategy: 推理策略
        cache_dir: 缓存目录
        n_proc: 进程数
        chunk_size: 块大小
        max_rounds: 最大推理轮次
        
    Returns:
        Tuple[List[Dict], List[ConstraintState], List[List[Dict]]]:
            - 处理后的样本列表
            - 约束状态列表
            - 推理日志列表
    """
    import os
    import pickle
    import multiprocessing as mp
    
    os.makedirs(cache_dir, exist_ok=True)
    result_samples = [None for _ in range(len(all_samples))]
    constraint_states = [None for _ in range(len(all_samples))]
    reasoning_logs = [None for _ in range(len(all_samples))]
    
    args = [
        (idx, sample, llm, llm_options, strategy, cache_dir, max_rounds)
        for idx, sample in enumerate(all_samples)
    ]
    
    with mp.Pool(n_proc) as p:
        for idx, proc_sample, constraint_state, log in tqdm(
            p.imap_unordered(
                _constraint_aware_exec_with_cache_mp_core, 
                args, 
                chunksize=chunk_size
            ),
            total=len(all_samples),
            desc=f"Constraint-aware chain execution"
        ):
            result_samples[idx] = proc_sample
            constraint_states[idx] = constraint_state
            reasoning_logs[idx] = log
    
    return result_samples, constraint_states, reasoning_logs


def _constraint_aware_exec_with_cache_mp_core(arg: Tuple) -> Tuple:
    """
    约束感知的多进程执行核心函数
    
    Args:
        arg: 参数元组
        
    Returns:
        Tuple: (idx, proc_sample, constraint_state, log)
    """
    import os
    import pickle
    
    idx, sample, llm, llm_options, strategy, cache_dir, max_rounds = arg
    
    cache_filename = "case-{}.pkl"
    try:
        sample_id = sample["id"]
        cache_path = os.path.join(cache_dir, cache_filename.format(idx))
        
        if os.path.exists(cache_path):
            cached_data = pickle.load(open(cache_path, "rb"))
            return idx, cached_data["sample"], cached_data["constraint_state"], cached_data["log"]
        else:
            from .constraint_induction import ConstraintInducer
            
            constraint_inducer = ConstraintInducer(llm=llm)
            
            proc_sample, constraint_state, log = constraint_aware_chain_exec_one_sample(
                sample,
                llm=llm,
                llm_options=llm_options,
                strategy=strategy,
                max_rounds=max_rounds,
                constraint_inducer=constraint_inducer
            )
            
            # 缓存结果
            cache_data = {
                "sample": proc_sample,
                "constraint_state": constraint_state,
                "log": log
            }
            pickle.dump(cache_data, open(cache_path, "wb"))
            
            return idx, proc_sample, constraint_state, log
    except Exception as e:
        print(f"Error in {sample_id}: {e}", flush=True)
        return idx, None, None, None
