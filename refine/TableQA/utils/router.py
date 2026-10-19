"""
AdaRefine Router: zero-LLM-cost adaptive routing for Refine stage

Routes samples into SKIP / LITE / FULL tiers based on
Judge pre-check, chain complexity, and table features.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


class RouteDecision(Enum):
    SKIP = "SKIP"   # Judge says correct, skip all Refine
    LITE = "LITE"   # Simple chain, only re-query (1 LLM call)
    FULL = "FULL"   # Standard Controller pipeline (N LLM calls)


@dataclass
class RouteResult:
    decision: RouteDecision
    signals: Dict[str, Any]
    reason: str

    def __repr__(self):
        return f"RouteResult({self.decision.value}, reason={self.reason!r})"


def _get_chain_ops(sample: Dict[str, Any]) -> List[str]:
    chain = sample.get("chain", [])
    return [op.get("operation_name", "") for op in chain]


def _chain_length(chain: Dict[str, Any]) -> int:
    return len(chain.get("chain", []))


def _has_complex_ops(ops: List[str]) -> bool:
    complex_ops = {"group_column", "sort_column"}
    return bool(complex_ops.intersection(ops))


def _table_row_count(sample: Dict[str, Any]) -> int:
    table_text = sample.get("table_text", [])
    # table_text[0] is header row
    return max(0, len(table_text) - 1)


def _table_col_count(sample: Dict[str, Any]) -> int:
    table_text = sample.get("table_text", [])
    if not table_text:
        return 0
    # header row is the first row
    return len(table_text[0])


def estimate_complexity(sample: Dict[str, Any]) -> RouteResult:
    """
    Estimate sample complexity and decide routing tier.

    Signals (all zero-LLM-cost):
    1. Judge pre-check: judge == "[Correct]" -> SKIP
    2. Chain complexity: len <= 2, no group/sort -> LITE candidate
    3. Table features: small table (rows <= 5, cols <= 4) -> LITE candidate

    Returns:
        RouteResult with decision, signals, and reason
    """
    signals = {}

    # Signal 1: Judge pre-check (requires 1 LLM call, done in controller_main_loop)
    judge_result = sample.get("judge", "")
    signals["judge_correct"] = "[Correct]" in judge_result

    if signals["judge_correct"]:
        return RouteResult(
            decision=RouteDecision.SKIP,
            signals=signals,
            reason="Judge says correct"
        )

    # Signal 2: Chain complexity
    ops = _get_chain_ops(sample)
    chain_len = len(ops)
    has_complex = _has_complex_ops(ops)
    signals["chain_length"] = chain_len
    signals["has_complex_ops"] = has_complex

    # Signal 3: Table features
    n_rows = _table_row_count(sample)
    n_cols = _table_col_count(sample)
    signals["table_rows"] = n_rows
    signals["table_cols"] = n_cols

    # Routing logic
    # LITE: the problem is simple enough that a direct re-query
    # (without Critic diagnosis) has a good chance of fixing it.
    # Relaxed criteria: chain is not too long, table is not huge.
    n_complex_ops = sum(1 for op in ops if op in {"group_column", "sort_column"})

    if chain_len <= 3 and n_complex_ops <= 1 and n_rows <= 10:
        return RouteResult(
            decision=RouteDecision.LITE,
            signals=signals,
            reason=f"Moderate complexity (len={chain_len}, complex_ops={n_complex_ops}), table {n_rows}x{n_cols}"
        )

    return RouteResult(
        decision=RouteDecision.FULL,
        signals=signals,
        reason=f"High complexity (len={chain_len}, complex_ops={n_complex_ops}), table {n_rows}x{n_cols}"
    )


def estimate_complexity_no_judge(sample: Dict[str, Any]) -> RouteResult:
    """Variant that ignores judge signal (for ablation: No-Judge)."""
    result = estimate_complexity(sample)
    if result.decision == RouteDecision.SKIP:
        ops = _get_chain_ops(sample)
        chain_len = len(ops)
        n_complex_ops = sum(1 for op in ops if op in {"group_column", "sort_column"})
        n_rows = _table_row_count(sample)
        n_cols = _table_col_count(sample)

        signals = {
            "judge_correct": False,
            "chain_length": chain_len,
            "has_complex_ops": n_complex_ops > 0,
            "table_rows": n_rows,
            "table_cols": n_cols,
        }

        if chain_len <= 3 and n_complex_ops <= 1 and n_rows <= 10:
            return RouteResult(
                decision=RouteDecision.LITE,
                signals=signals,
                reason=f"No-Judge: moderate complexity (len={chain_len}, complex_ops={n_complex_ops})"
            )

        return RouteResult(
            decision=RouteDecision.FULL,
            signals=signals,
            reason=f"No-Judge: high complexity (len={chain_len}, complex_ops={n_complex_ops})"
        )

    return result


def estimate_complexity_no_chain(sample: Dict[str, Any]) -> RouteResult:
    """Variant that ignores chain signal (for ablation: No-chain)."""
    signals = {}

    judge_result = sample.get("judge", "")
    signals["judge_correct"] = "[Correct]" in judge_result

    if signals["judge_correct"]:
        return RouteResult(
            decision=RouteDecision.SKIP,
            signals=signals,
            reason="Judge says correct (no-chain ablation)"
        )

    n_rows = _table_row_count(sample)
    n_cols = _table_col_count(sample)
    signals["table_rows"] = n_rows
    signals["table_cols"] = n_cols

    if n_rows <= 8 and n_cols <= 5:
        return RouteResult(
            decision=RouteDecision.LITE,
            signals=signals,
            reason=f"No-chain: small table ({n_rows}x{n_cols})"
        )

    return RouteResult(
        decision=RouteDecision.FULL,
        signals=signals,
        reason=f"No-chain: large table ({n_rows}x{n_cols})"
    )


def estimate_complexity_no_table(sample: Dict[str, Any]) -> RouteResult:
    """Variant that ignores table signal (for ablation: No-table)."""
    signals = {}

    judge_result = sample.get("judge", "")
    signals["judge_correct"] = "[Correct]" in judge_result

    if signals["judge_correct"]:
        return RouteResult(
            decision=RouteDecision.SKIP,
            signals=signals,
            reason="Judge says correct (no-table ablation)"
        )

    ops = _get_chain_ops(sample)
    chain_len = len(ops)
    n_complex_ops = sum(1 for op in ops if op in {"group_column", "sort_column"})
    signals["chain_length"] = chain_len
    signals["has_complex_ops"] = n_complex_ops > 0

    if chain_len <= 3 and n_complex_ops <= 1:
        return RouteResult(
            decision=RouteDecision.LITE,
            signals=signals,
            reason=f"No-table: moderate chain (len={chain_len}, complex_ops={n_complex_ops})"
        )

    return RouteResult(
        decision=RouteDecision.FULL,
        signals=signals,
        reason=f"No-table: complex chain (len={chain_len}, complex_ops={n_complex_ops})"
    )


ROUTER_VARIANTS = {
    "full": estimate_complexity,
    "no_judge": estimate_complexity_no_judge,
    "no_chain": estimate_complexity_no_chain,
    "no_table": estimate_complexity_no_table,
}
