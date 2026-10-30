"""P4 图错误信号探测——共享解析/特征/指标库 (P5 复用解析与重放)

按 docs/modules_p1_p6.md §5/§9.1 预注册口径实现:
- 特征 F1/F2 为闸门判定主特征; F3–F6 为探索性 (不参与 go/no-go)
- 参数解析协议: 主候选 = 第一对 (最高置信); ast.literal_eval 解析;
  空选择 [] 记零删减 (F2=0), 不视为删除全部行
- 行号对齐: 链中行号为重排后 1-based 含表头展示号, -1 转数据行 0-based
  直接索引 row_relevance_scores
- F1 列名对齐: 精确 → normalize_value 归一化 → 仍不匹配跳过并计数
- 聚合: max over steps 为主方案, mean 为消融
- 指标: 纯 numpy (AUROC = Mann-Whitney 秩统计, AUPRC = PR 阶梯积分,
  bootstrap = 重采样百分位), 本环境 sklearn/scipy 不可用
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from table_structure.value_similarity import normalize_value


# ==================== 参数解析 ====================

def parse_op_params(operation: Dict) -> Optional[Any]:
    """解析主候选参数对象; 返回 None 表示空候选/无候选。

    格式 (2026-10-30 pkl 抽查):
    - select_row/select_column/group_column/simple_query: [(字符串化字面量, conf)]
    - sort_column: 扁平 (引用列, 序, 类型, index_order, max, min, conf)
    - add_column: 可能为空占位 []
    """
    params = operation.get("parameter_and_conf") or []
    if not params:
        return None
    raw = params[0][0]
    try:
        return ast.literal_eval(str(raw))
    except (SyntaxError, ValueError, TypeError):
        return raw


def parse_top_conf(operation: Dict) -> Optional[float]:
    """主候选置信度 (np.exp 聚合前的原始 conf 已是 exp 形式)。"""
    params = operation.get("parameter_and_conf") or []
    if not params:
        return None
    try:
        return float(params[0][1])
    except (TypeError, ValueError, IndexError):
        return None


def parse_row_ids(operation: Dict) -> List[int]:
    """select_row 主候选 → 数据行 0-based 索引列表 (候选为 1-based 含表头)。

    空选择 [] 返回空列表 (零删减协议); 无数字/解析失败返回空列表。
    """
    params = parse_op_params(operation)
    if params is None:
        return []
    if not isinstance(params, list):
        params = [params]
    ids = []
    for item in params:
        if str(item).strip() == "*":
            return []
        match = re.search(r"\d+", str(item))
        if match:
            ids.append(int(match.group(0)) - 1)
    return ids


def parse_col_names(operation: Dict) -> List[str]:
    """select_column 主候选 → 列名列表; group/sort 的引用列取 [col]。

    空候选/解析失败返回空列表。
    """
    name = operation.get("operation_name", "")
    params = parse_op_params(operation)
    if params is None:
        return []
    if name in ("sort_column", "group_column"):
        first = params[0] if isinstance(params, (list, tuple)) and params else params
        if isinstance(first, (list, tuple)):
            first = first[0] if first else None
        return [str(first)] if first is not None else []
    if isinstance(params, list):
        return [str(c) for c in params]
    return [str(params)]


# ==================== 链追踪 (不重放表内容) ====================

def trace_row_sets(sample: Dict) -> Optional[Dict]:
    """增量行/列集合追踪 (P4 特征 + P5 EXCLUDES/DEPENDS_ON 的共享基础)。

    只跟踪集合 memberships, 不重放表内容 (零 deepcopy 开销)。

    Returns:
        {
          "num_rows_init": 初始数据行数 (重排后口径),
          "col_scores_norm": {归一化列名: 分数},
          "steps": [ 每步 {
              "op": 操作名, "step": 1-based 有效步号,
              "rows_before": frozenset, "rows_after": frozenset,
              "rows_removed": frozenset, "cols_before": frozenset, "cols_after": frozenset,
              "row_conf"/"col_conf": 主候选 conf,
          } ],
          "rows_final", "cols_final": 终态集合,
          "unmatched_cols": F1 对齐失败列名列表,
        }
        样本缺图分数/表时返回 None。
    """
    row_scores = sample.get("row_relevance_scores") or {}
    col_scores = sample.get("col_relevance_scores") or {}
    table = (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )
    if not row_scores or not col_scores or not table:
        return None

    num_rows_init = len(table) - 1  # 数据行数 (不含表头)
    if num_rows_init <= 0:
        return None

    col_scores_norm = {normalize_value(str(k)): float(v) for k, v in col_scores.items()}
    header = table[0]
    cols_init = frozenset(normalize_value(str(h)) for h in header)
    rows_cur = frozenset(range(num_rows_init))
    cols_cur = cols_init

    steps = []
    unmatched_cols: List[str] = []
    step = 0
    for operation in sample.get("chain", []):
        op = operation.get("operation_name", "")
        rows_before, cols_before = rows_cur, cols_cur
        conf = parse_top_conf(operation)

        if op == "select_row":
            keep = {r for r in parse_row_ids(operation) if r in rows_cur}
            if keep:
                rows_cur = frozenset(keep)
            # 空选择 [] / 无有效行 → 零删减协议: 集合不变
            selected_cols = []
        elif op == "select_column":
            raw_cols = parse_col_names(operation)
            selected = [normalize_value(c) for c in raw_cols]
            for c, norm in zip(raw_cols, selected):
                if norm not in col_scores_norm and norm not in cols_cur:
                    unmatched_cols.append(c)
            known = {c for c in selected if c in cols_cur}
            if known:
                cols_cur = frozenset(known)
            selected_cols = selected
        elif op in ("sort_column", "group_column"):
            raw_cols = parse_col_names(operation)
            selected_cols = [normalize_value(c) for c in raw_cols]
            for c, norm in zip(raw_cols, selected_cols):
                if norm not in col_scores_norm:
                    unmatched_cols.append(c)
            # sort/group 不改变列集合 (group 生成 group_sub_table 属中间态, 一期不计)
        else:
            selected_cols = []
        # add_column 新列不在原始 col_relevance_scores 中 (中间节点无分数, 不参与异常信号)

        if op != "simple_query":
            step += 1
        rows_removed = rows_before - rows_cur
        steps.append({
            "op": op,
            "step": step,
            "rows_before": rows_before,
            "rows_after": rows_cur,
            "rows_removed": rows_removed,
            "cols_before": cols_before,
            "cols_after": cols_cur,
            "cols_selected": selected_cols,
            "conf": conf,
        })

    return {
        "num_rows_init": num_rows_init,
        "row_scores": {int(k): float(v) for k, v in row_scores.items()},
        "col_scores_norm": col_scores_norm,
        "steps": steps,
        "rows_final": rows_cur,
        "cols_final": cols_cur,
        "unmatched_cols": unmatched_cols,
    }


# ==================== 特征 (预注册 F1/F2 主特征; F4/F5/F6 探索) ====================

def feature_f1(trace: Dict) -> float:
    """F1 列相关性偏差 (主特征): 列选择类操作所选列的 PPR 分数
    与最高相关列分数之差 (gap), max over (ops × cols)。

    gap 大 = 选中了低相关列 (错列/错排序列/错分组键信号)。
    对齐失败列 gap 记 0 (不计入信号, 对齐失败率单独报告)。
    """
    col_scores = trace["col_scores_norm"]
    max_score = max(col_scores.values()) if col_scores else 0.0
    gaps = []
    for st in trace["steps"]:
        if st["op"] not in ("select_column", "sort_column", "group_column"):
            continue
        for norm_col in st["cols_selected"]:
            score = col_scores.get(norm_col)
            if score is not None:
                gaps.append(max_score - score)
    return float(max(gaps)) if gaps else 0.0


def feature_f2(trace: Dict) -> float:
    """F2 行证据损失 (主特征): 每个 select_row 删除行的 PPR 权重占比
    Σ(被删行)/Σ(操作前), max over steps。空选择 → 0。"""
    row_scores = trace["row_scores"]
    vals = []
    for st in trace["steps"]:
        if st["op"] != "select_row" or not st["rows_removed"]:
            continue
        before_total = sum(row_scores.get(r, 0.0) for r in st["rows_before"])
        removed_total = sum(row_scores.get(r, 0.0) for r in st["rows_removed"])
        if before_total > 0:
            vals.append(removed_total / before_total)
    return float(max(vals)) if vals else 0.0


def feature_f4(trace: Dict) -> float:
    """F4 删除极值信号 (探索): max(被删行分数)/max(操作前行分数), max over steps。
    与 F2 互补——F2 量占比, F4 抓"删了分数最高的行"。"""
    row_scores = trace["row_scores"]
    vals = []
    for st in trace["steps"]:
        if st["op"] != "select_row" or not st["rows_removed"]:
            continue
        before_max = max((row_scores.get(r, 0.0) for r in st["rows_before"]), default=0.0)
        removed_max = max((row_scores.get(r, 0.0) for r in st["rows_removed"]), default=0.0)
        if before_max > 0:
            vals.append(removed_max / before_max)
    return float(max(vals)) if vals else 0.0


def feature_f5(trace: Dict) -> Tuple[float, float]:
    """F5 终点证据质量 (探索): (行证据占比, 列证据占比)
    Σ(最终保留行分数)/Σ(初始行分数), 列同理。"""
    row_scores = trace["row_scores"]
    col_scores = trace["col_scores_norm"]
    n_init = trace["num_rows_init"]
    row_ratio = (
        sum(row_scores.get(r, 0.0) for r in trace["rows_final"])
        / sum(row_scores.get(r, 0.0) for r in range(n_init))
        if n_init and sum(row_scores.get(r, 0.0) for r in range(n_init)) > 0
        else 0.0
    )
    col_ratio = (
        sum(col_scores.get(c, 0.0) for c in trace["cols_final"])
        / sum(col_scores.values())
        if sum(col_scores.values()) > 0
        else 0.0
    )
    return float(row_ratio), float(col_ratio)


def feature_f6(sample: Dict, trace: Dict) -> Dict:
    """F6 答案-图追溯 (探索, 三态): 预测答案精确匹配最终保留单元格时,
    取该单元格的行/列分数及所在列分数排名。

    Returns:
        {"status": "in_final"|"only_removed"|"not_in_table",
         "row_score": float|None, "col_score": float|None, "col_rank": int|None}
        答案为聚合值/表外值时 status 报 not_in_table (可算率由统计层汇总)。
    """
    result = {"status": "not_in_table", "row_score": None, "col_score": None, "col_rank": None}
    try:
        pred = str(sample["chain"][-1]["parameter_and_conf"][0][0]).strip().lower()
    except (IndexError, KeyError, TypeError):
        return result
    if "|" in pred:  # 多答案取第一个
        pred = pred.split("|")[0].strip()
    if not pred:
        return result

    table = (
        sample.get("atg_reranked_table")
        or sample.get("atgo_reranked_table")
        or sample.get("table_text")
        or sample.get("table")
    )
    if not table:
        return result

    row_scores = trace["row_scores"]
    col_scores = trace["col_scores_norm"]
    header = [normalize_value(str(h)) for h in table[0]]

    def col_rank_of(norm_col):
        ranked = sorted(col_scores.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (c, _) in enumerate(ranked, 1):
            if c == norm_col:
                return rank
        return None

    # 先在最终保留行集中找
    for r in sorted(trace["rows_final"]):
        row = table[r + 1] if r + 1 < len(table) else None
        if not row:
            continue
        for ci, cell in enumerate(row):
            if ci >= len(header):
                break
            if str(cell).strip().lower() == pred:
                norm_col = header[ci]
                return {
                    "status": "in_final",
                    "row_score": row_scores.get(r),
                    "col_score": col_scores.get(norm_col),
                    "col_rank": col_rank_of(norm_col),
                }

    # 仅在被删集中
    removed = set()
    for st in trace["steps"]:
        removed |= set(st["rows_removed"])
    for r in sorted(removed - set(trace["rows_final"])):
        row = table[r + 1] if r + 1 < len(table) else None
        if not row:
            continue
        for ci, cell in enumerate(row):
            if ci >= len(header):
                break
            if str(cell).strip().lower() == pred:
                norm_col = header[ci]
                return {
                    "status": "only_removed",
                    "row_score": row_scores.get(r),
                    "col_score": col_scores.get(norm_col),
                    "col_rank": col_rank_of(norm_col),
                }
    return result


def b1_confidence(trace: Dict) -> float:
    """B1 基线: 链内各操作主候选 conf 的均值 (conf 已是 np.exp(score) 形式)。"""
    confs = [st["conf"] for st in trace["steps"] if st["conf"] is not None]
    return float(np.mean(confs)) if confs else 0.0


def aggregate(step_values: List[float], mode: str = "max") -> float:
    """逐操作特征聚合到样本级 (主方案 max, 消融 mean)。"""
    if not step_values:
        return 0.0
    arr = np.asarray(step_values, dtype=float)
    return float(arr.max() if mode == "max" else arr.mean())


# ==================== 指标 (纯 numpy, 无 sklearn/scipy) ====================

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC = Mann-Whitney U 秩统计 (并列取平均秩)。labels: 1=正类(错误)。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort") + 1
    # 并列值平均秩
    sorted_idx = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[sorted_idx]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        ranks[sorted_idx[i:j + 1]] = avg_rank
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUPRC = PR 阶梯积分 (按分数降序扫描, 精确率对召回率线性插值)。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int(labels.sum())
    if n_pos == 0 or len(labels) == n_pos:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    recall = tp / n_pos
    precision = tp / np.maximum(tp + fp, 1)
    # PR 阶梯: 召回率相等处取最大精确率
    ap = 0.0
    prev_recall = 0.0
    for r, p in zip(recall, precision):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float) -> float:
    """固定 FPR 下的 TPR (P6 CONFLICT 阈值选择依据)。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    threshold = np.quantile(neg, 1 - fpr_target)
    return float((pos >= threshold).mean())


def bootstrap_ci(scores, labels, metric_fn, n_boot: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """bootstrap 百分位 CI。metric_fn(scores, labels) -> float。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(scores[idx], labels[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def pearson_corr(matrix: np.ndarray) -> np.ndarray:
    """Pearson 相关矩阵 (纯 numpy; 特征相关矩阵用)。"""
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    normed = centered / std
    return (normed.T @ normed) / len(matrix)
