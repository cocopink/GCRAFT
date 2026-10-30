"""P6 图证据驱动验证 (一期: CONFLICT / UNKNOWN)

确定性核验器: 对 Thought 链中的筛选类操作做零 LLM 成本的图证据检查。
在 Judge 判定 [Correct] 后发现确定性冲突时, 改写 judge 文本使
"[Correct]" 子串判定失效, 让 Controller / legacy chain 的纠错通路接管
(一期唯一流程改变点; 判错最多多跑一次 Critic, 零抑制风险)。

三态语义 (docs/modules_p1_p6.md §7):
- CONFLICT:  核验发现确定性冲突证据 → 覆盖 Correct, 触发纠错
- UNKNOWN:   图信息不足以判断 → 保留原 Judge 行为 (UNKNOWN ≠ Correct)
- SUPPORTED: 二期 (证据支持现有执行 ≠ 答案正确), 一期不实现

一期核验范围 (确定性检查, 宁缺毋滥):
- select_row: 主候选行号越界 (行号口径 1-based 含表头, 与 select_row_act 一致)
- select_column / sort_column / group_column: 引用列与表头无匹配
  (精确 + difflib 最佳相似度双口径; 空选择 [] 记零删减不视为冲突)
- add_column / simple_query 无图核验 (UNKNOWN 理论下限来源)

启用开关: sample["graph_metadata"]["enable_p6"]
(main_tree_based.py 运行参数注入, 与 P3 bias_weight 同款传递机制)。
"""

import ast
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# 列名"确定性无匹配"的相似度下界: 低于该值连模糊容错也无法对应到任何真实表头
_MIN_COLUMN_SIMILARITY = 0.55


def _parse_step_params(operation: Dict) -> Optional[Any]:
    """解析主候选 (最高置信第一对) 的参数对象。

    真实格式 (2026-10-30 pkl 抽查):
    - select_row/select_column/group_column/simple_query: [(字符串化字面量, conf)]
    - sort_column: 扁平 (引用列, 序, 类型, index_order, max, min, conf), 引用列直接可读
    字面量解析失败时原样返回字符串 (sort 的列名即此形态), 空候选返回 None。
    """
    params = operation.get("parameter_and_conf") or []
    if not params:
        return None
    raw = params[0][0]
    try:
        return ast.literal_eval(str(raw))
    except (SyntaxError, ValueError, TypeError):
        return raw


def _trace_table_states(sample: Dict) -> List[Tuple[Dict, Optional[Dict], int]]:
    """重放执行链, 返回每步 (operation, 操作前 table_info, 步骤号)。

    - 表选择口径与 get_table_info 一致: atg_reranked_table > atgo_reranked_table > table_text
    - 步骤号口径与 get_critic_table_info 一致: 跳过 skip 占位步的 1-based 计数
    - 重放异常时该步标记不可核验 (table_info=None) 并终止重放
    """
    try:
        from utils.chain import get_act_func
    except ImportError:
        return []

    # 表选择口径与 get_table_info 一致: atg_reranked_table > atgo_reranked_table > table_text
    table_text = (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )
    if not table_text:
        return []

    table_info = {"table_text": table_text, "act_chain": []}
    traced: List[Tuple[Dict, Optional[Dict], int]] = []
    step = 0
    for operation in sample.get("chain", []):
        try:
            act_func = get_act_func(operation.get("operation_name", ""))
            new_info = act_func(table_info, operation, skip_op=[])
        except Exception:
            traced.append((operation, None, step))
            break
        # 步骤号口径与 get_critic_table_info 一致: 跳过 skip 占位步的 1-based 计数;
        # skip 步为执行失败占位, 不占号也不核验
        if new_info.get("act_chain") and "skip" not in new_info["act_chain"][-1]:
            step += 1
            traced.append((operation, table_info, step))
        table_info = new_info
    return traced


def _verify_select_row(operation: Dict, table_info: Dict) -> Optional[str]:
    """select_row 确定性核验: 主候选行号越界。空选择 [] 记零删减, 不视为冲突。"""
    table = table_info.get("table_text") or []
    if not table:
        return None
    params = _parse_step_params(operation)
    if params is None:
        return None
    if not isinstance(params, list):
        params = [params]
    row_ids = []
    for item in params:
        match = re.search(r"\d+", str(item))
        if match:
            row_ids.append(int(match.group(0)))
    if not row_ids:
        return None
    num_entries = len(table)  # 含表头; 行号口径与 select_row_act 的 enumerate 一致 (0=表头)
    out_of_range = sorted({r for r in row_ids if r >= num_entries})
    if out_of_range:
        return (
            f"select_row 行号越界: {out_of_range} >= 表条目数 {num_entries} (含表头)"
        )
    return None


def _extract_ref_column(params: Any) -> Optional[str]:
    """从解析后的参数对象提取引用列名 (兼容嵌套元组/扁平元组/裸字符串)。"""
    if not params:
        return None
    first = params[0] if isinstance(params, (list, tuple)) else params
    if isinstance(first, (list, tuple)):
        first = first[0] if first else None
    return str(first) if first is not None else None


def _verify_column_ref(operation: Dict, table_info: Dict, op_name: str) -> Optional[str]:
    """select_column / sort_column / group_column 确定性核验: 引用列无匹配。

    双口径: 精确 (strip+lower) 命中即通过; 未命中看 difflib 最佳相似度,
    低于 _MIN_COLUMN_SIMILARITY 视为确定性冲突 (LLM 输出跑偏/幻觉列名)。
    """
    table = table_info.get("table_text") or []
    if not table:
        return None
    headers = {str(h).strip().lower() for h in table[0]}
    params = _parse_step_params(operation)
    if params is None:
        return None
    if op_name in ("sort_column", "group_column"):
        col = _extract_ref_column(params)
        col_names = [col] if col else []
    elif isinstance(params, list):
        col_names = [str(c) for c in params]
    else:
        col_names = [str(params)]
    for col in col_names:
        normalized = col.strip().lower()
        if normalized in headers:
            continue
        best = max(
            (SequenceMatcher(None, normalized, h).ratio() for h in headers),
            default=0.0,
        )
        if best < _MIN_COLUMN_SIMILARITY:
            return (
                f"{op_name} 引用列不存在: '{col}' "
                f"(与全部表头最大相似度 {best:.2f} < {_MIN_COLUMN_SIMILARITY})"
            )
    return None


def graph_verdict(sample: Dict) -> Tuple[str, Optional[int]]:
    """三态图判定 (一期: CONFLICT / UNKNOWN)。

    Returns:
        (verdict, suspect_step): 首个确定性冲突的步骤号
        (口径同 get_critic_table_info 的 incorrect_step); 无冲突为 (UNKNOWN, None)。
    """
    sample.pop("_p6_evidence", None)  # 多轮判定不累积证据
    for operation, table_info, step_no in _trace_table_states(sample):
        if table_info is None:
            continue
        op_name = operation.get("operation_name", "")
        evidence = None
        if op_name == "select_row":
            evidence = _verify_select_row(operation, table_info)
        elif op_name in ("select_column", "sort_column", "group_column"):
            evidence = _verify_column_ref(operation, table_info, op_name)
        if evidence:
            sample.setdefault("_p6_evidence", []).append(
                {"step": step_no, "operation": op_name, "evidence": evidence}
            )
            return "CONFLICT", step_no
    return "UNKNOWN", None


def override_judge_if_conflict(sample: Dict) -> bool:
    """Judge 判定后的 P6 覆盖检查 (controller 与 legacy chain 共用入口)。

    - enable_p6 关闭: 不做任何事, 返回 False
    - judge 判错: 仅记录核验结论 (_p6_verdict, 供 p6_evaluate 对照分析),
      管线本就走向纠错, 不改判定
    - judge 判 [Correct] + 核验 CONFLICT: 改写 judge 文本使 "[Correct]"
      子串失效 (controller 的 in 判定与 legacy chain 的 ==/in 判定均不再命中),
      纠错通路接管; 原始判定保留在 _p6_original_judge

    Returns:
        True 表示发生了 CONFLICT 覆盖。
    """
    metadata = sample.get("graph_metadata") or {}
    if not metadata.get("enable_p6", False):
        return False
    judge_text = sample.get("judge", "")
    verdict, suspect_step = graph_verdict(sample)
    sample["_p6_verdict"] = verdict
    if suspect_step is not None:
        sample["_p6_suspect_step"] = suspect_step
    if verdict != "CONFLICT" or "[Correct]" not in judge_text:
        return False
    sample["_p6_original_judge"] = judge_text
    sample["judge"] = judge_text.replace("[Correct]", "[Graph-Conflict]")
    return True
