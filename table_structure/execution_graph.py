"""P5 执行感知图推理——执行关系建模与步骤异常分 (docs/modules_p1_p6.md §6/§9.2)

首批只做 EXCLUDES + DEPENDS_ON (信号最强、实现最直接); READS/RETAINS/PRODUCES 二期扩展。

- EXCLUDES(j): 步骤 j 的 select_row 排除的数据行 (由 p4_probing_lib.trace_row_sets
  的增量行集合追踪推得, 本模块不重放表内容)
- DEPENDS_ON(k): 前序行筛选/列筛选步骤中, 处于步骤 k 数据上游者——
  公式化定义: j ∈ DEPENDS_ON(k) ⟺ j < k ∧ (rows_removed(j) ≠ ∅ ∨ cols 变更(j))
  ∧ (引用列(k) = ∅ ∨ 引用列(k) ∩ cols_after(j) ≠ ∅)
- step_anomaly_score(j) = EXCLUDES 权重占比(j) × 下游 DEPENDS_ON 覆盖度(j)
  (公式显式定义, 阈值由调用方按注入集校准; PRODUCES 新列无分数, 不参与异常信号)

中间节点分数方案: PRODUCES 产生的新列 (如 add_column) 取来源列分数均值传播——
一期 trace 中 add_column 不改变列集合, 无分数节点直接不参与 (与模块定义 §6 一致)。
"""

from typing import Dict, List, Optional, Tuple


def build_execution_relations(trace: Dict) -> Dict:
    """基于 p4_probing_lib.trace_row_sets 的追踪结果构建执行关系。

    Args:
        trace: trace_row_sets(sample) 输出 (含 steps / row_scores / col_scores_norm)

    Returns:
        {
          "relations": [ 每步 {"step", "op", "excludes": frozenset, "depends_on": set[int],
                               "excludes_weight_ratio": float, "refer_cols": set[str]} ],
          "col_scores": 归一化列分数,
        }
    """
    col_scores = trace["col_scores_norm"]
    row_scores = trace["row_scores"]
    steps = trace["steps"]

    relations = []
    for k, st in enumerate(steps):
        ref_cols = set(st.get("cols_selected") or [])
        if st["op"] == "add_column":
            ref_cols = set()  # 新列不在原始图分数中, 无追溯
        depends_on = set()
        for j in range(k):
            prev = steps[j]
            changed = bool(prev["rows_removed"]) or prev["cols_before"] != prev["cols_after"]
            if not changed:
                continue
            if not ref_cols or (ref_cols & set(prev["cols_after"])):
                depends_on.add(prev["step"])

        before_total = sum(row_scores.get(r, 0.0) for r in st["rows_before"])
        removed_total = sum(row_scores.get(r, 0.0) for r in st["rows_removed"])
        excludes_ratio = removed_total / before_total if before_total > 0 else 0.0

        relations.append({
            "step": st["step"],
            "op": st["op"],
            "excludes": st["rows_removed"],
            "depends_on": depends_on,
            "excludes_weight_ratio": excludes_ratio,
            "refer_cols": ref_cols,
        })

    # 下游覆盖度: 依赖 j 的后续步骤数 / j 之后的步骤总数
    step_to_idx = {rel["step"]: i for i, rel in enumerate(relations)}
    downstream_hit = [0] * len(relations)
    for rel in relations:
        for j_step in rel["depends_on"]:
            j_idx = step_to_idx.get(j_step)
            if j_idx is not None:
                downstream_hit[j_idx] += 1
    for j, rel in enumerate(relations):
        downstream_total = len(relations) - 1 - j
        rel["downstream_coverage"] = (
            downstream_hit[j] / downstream_total if downstream_total > 0 else 0.0
        )

    return {"relations": relations, "col_scores": col_scores}


def step_anomaly_score(exec_graph: Dict, threshold: float = 0.0) -> List[Tuple[int, float]]:
    """步骤异常分排序 (降序)。

    score(j) = EXCLUDES 权重占比(j) × 下游 DEPENDS_ON 覆盖度(j)

    Args:
        exec_graph: build_execution_relations 输出
        threshold: 显式阈值 (>= threshold 判可疑; 0 = 返回全部排序)

    Returns:
        [(step, score)] 降序; 仅含 score > 0 的步骤。
    """
    scored = []
    for rel in exec_graph["relations"]:
        score = rel["excludes_weight_ratio"] * rel["downstream_coverage"]
        if score > threshold:
            scored.append((rel["step"], float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def suspect_step_by_anomaly(sample: Dict, trace: Dict) -> Optional[int]:
    """便捷入口: 返回最高异常分步骤号 (无正分步骤返回 None)。"""
    exec_graph = build_execution_relations(trace)
    ranked = step_anomaly_score(exec_graph)
    return ranked[0][0] if ranked else None
