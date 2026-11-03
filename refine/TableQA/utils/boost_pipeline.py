"""R3a 确定性候选验证流水线 v2 (boost, 2026-11-03)

v1 教训: act_chain 的 "skip" 标记在 DSL 语义中多为合法跳过 (空参数 add_column /
空选择 select_row = 零删减), 95.5% 真实链被误判 invalid —— 判据不可用。

v2 核心信号 = **终态对比** (确定性, 语义干净):
  - terminal_state_changed: 候选链重放终态 (行数/表头/表内容哈希) vs 原错误链
    终态 —— 不同 → 修正确实改变了推理路径; 相同 → 修正无效 (白修)
  - parse_error: act 重放抛异常 (代码级错误, 罕见)
  - soft: 越界行号 / 列名模糊匹配 (信息性记录, 不参与判定)

择优 (boost_select):
  1. terminal_state_changed 的候选优先 (真修正)
  2. 并列 → greedy 优先序
  3. 全部未变化 → greedy 原样 (回退原行为, 避免白修浪费 judge)
"""

import copy
import re


def _parse_row_ids(param_str):
    """select_row 主候选参数字符串 → 0-based 行号列表 (对齐 p4_probing_lib 口径)。"""
    try:
        items = eval(param_str)
        if not isinstance(items, list):
            items = [items]
    except Exception:
        return [], True
    ids, bad = [], False
    for item in items:
        s = str(item).strip()
        if s == "*":
            return [], True
        m = re.search(r"\d+", s)
        if m:
            ids.append(int(m.group(0)) - 1)
        else:
            bad = True
    return ids, bad


def _table_fingerprint(table_text):
    """表终态指纹: (行数, 表头元组, 首列内容哈希)。"""
    if not table_text:
        return (0, (), "")
    header = tuple(str(c) for c in table_text[0]) if table_text[0] else ()
    body_hash = hash(str([row[:2] for row in table_text[1:4]]))
    return (len(table_text) - 1, header, body_hash)


def _replay_chain(sample, upto=None):
    """重放链的 DSL (复用 get_act_func), 返回终态 table_text。simple_query 步跳过。"""
    from refine.TableQA.utils.chain import get_act_func

    table_text = (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )
    ti = {"table_text": copy.deepcopy(table_text), "act_chain": []}
    chain = sample.get("chain", [])
    if upto is not None:
        chain = chain[:upto]
    for op in chain:
        if op.get("operation_name") == "simple_query":
            continue
        ti = get_act_func(op.get("operation_name", ""))(ti, op)
    return ti["table_text"]


def validate_candidate(sample, incorrect_step=None, original_sample=None):
    """候选链确定性验证 (v2 终态对比版)。

    Returns:
        {
          "terminal_state_changed": bool,  # 候选终态 ≠ 原错误链终态 (真修正)
          "parse_error": bool,             # act 重放异常
          "soft": [信息性记录],
        }
    """
    soft = []
    parse_error = False
    try:
        cand_fp = _table_fingerprint(_replay_chain(sample))
    except Exception as e:
        return {"terminal_state_changed": False, "parse_error": True,
                "soft": [f"act 重放异常: {type(e).__name__}: {e}"]}

    orig_fp = None
    if original_sample is not None:
        try:
            orig_fp = _table_fingerprint(_replay_chain(original_sample))
        except Exception:
            orig_fp = None

    # soft 信息性记录
    table_text = (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )
    n_data = len(table_text) - 1 if table_text else 0
    for op in sample.get("chain", []):
        if op.get("operation_name") == "select_row":
            params = op.get("parameter_and_conf", [])
            if params:
                ids, bad = _parse_row_ids(params[0][0])
                oob = [i for i in ids if i < 0 or i >= n_data]
                if oob:
                    soft.append(f"select_row 越界行号 {oob[:3]} (数据行数 {n_data})")

    return {"terminal_state_changed":
            (orig_fp is not None and cand_fp != orig_fp) or orig_fp is None,
            "parse_error": parse_error, "soft": soft}


def boost_select(candidates, incorrect_step=None, original_sample=None, greedy_fallback=None):
    """多候选择优 (R3a v2): 终态变化者优先, 并列取 greedy, 全未变化回退 greedy。

    Returns:
        (selected_sample, decision_info)
    """
    infos, selected = [], None
    for tag, s in candidates:
        v = validate_candidate(s, incorrect_step, original_sample)
        infos.append({"tag": tag, **v})
        if v["terminal_state_changed"] and not v["parse_error"] and selected is None:
            order = 0 if tag == "greedy" else 1
            if selected is None or order < selected[0]:
                selected = (order, s, tag)

    decision = {"n_candidates": len(candidates), "details": infos}
    if selected is not None:
        decision["selected_by"] = f"changed_{selected[2]}"
        return selected[1], decision
    decision["selected_by"] = "fallback_greedy_none_changed"
    return greedy_fallback if greedy_fallback is not None else (candidates[0][1] if candidates else None), decision
