"""Evidence Pack Builder: extract row/column/sort/group/answer evidence per step."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .operation_trace import TraceStep, OperationTrace


@dataclass
class StepEvidence:
    """Structured evidence for a single operation step."""

    step_num: int
    operation_name: str
    kind: str
    summary: str
    row_diff: Dict[str, Any] = field(default_factory=dict)
    column_diff: Dict[str, Any] = field(default_factory=dict)
    sort_evidence: Dict[str, Any] = field(default_factory=dict)
    group_evidence: Dict[str, Any] = field(default_factory=dict)
    answer_evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Format evidence as a compact prompt block for Critic."""
        lines = [
            f"Step {self.step_num} ({self.operation_name}) evidence:",
            f"- Summary: {self.summary}",
        ]
        for warning in self.warnings:
            lines.append(f"- Warning: {warning}")
        for name, value in [
            ("row_diff", self.row_diff),
            ("column_diff", self.column_diff),
            ("sort_evidence", self.sort_evidence),
            ("group_evidence", self.group_evidence),
            ("answer_evidence", self.answer_evidence),
        ]:
            if value:
                lines.append(f"- {name}: {value}")
        block = "\n".join(lines)
        # Cap at 1000 characters
        if len(block) > 1000:
            block = block[:997] + "..."
        return block


def _table_to_tuples(table: List[List[Any]]) -> List[tuple]:
    """Convert table rows to hashable tuples for set operations."""
    return [tuple(row) for row in table]


def _extract_question_keywords(question: str) -> List[str]:
    """Extract meaningful keywords from a question, filtering stop words."""
    from refine.TableQA.utils.verifier import _extract_question_keywords as _ek
    return _ek(question)


# --- Per-operation evidence builders ---

def _build_row_diff(step: TraceStep) -> Dict[str, Any]:
    before_rows = set(_table_to_tuples(step.before_table[1:])) if len(step.before_table) > 1 else set()
    after_rows = set(_table_to_tuples(step.after_table[1:])) if len(step.after_table) > 1 else set()
    kept = before_rows & after_rows
    removed = before_rows - after_rows
    added = after_rows - before_rows
    return {
        "before_count": len(before_rows),
        "after_count": len(after_rows),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "added_count": len(added),
    }


def _build_column_diff(step: TraceStep, question: str) -> Dict[str, Any]:
    before_headers = [str(h).lower() for h in step.before_table[0]] if step.before_table else []
    after_headers = [str(h).lower() for h in step.after_table[0]] if step.after_table else []
    dropped = [h for h in before_headers if h not in after_headers]
    kept = [h for h in before_headers if h in after_headers]

    # Check if dropped headers are relevant to the question
    warnings = []
    if dropped:
        keywords = [kw.lower() for kw in _extract_question_keywords(question)]
        relevant_dropped = [h for h in dropped if any(kw in h for kw in keywords)]
        if relevant_dropped:
            warnings.append(f"Dropped potentially relevant column(s): {relevant_dropped}")

    return {
        "kept_headers": kept,
        "dropped_headers": dropped,
    }, warnings


def _build_sort_evidence(step: TraceStep) -> Dict[str, Any]:
    from refine.TableQA.utils.verifier import _parse_sort_column, _safe_float

    result: Dict[str, Any] = {}
    sort_col = _parse_sort_column(step.action)
    if sort_col:
        result["sort_column"] = sort_col
    return result


def _build_group_evidence(step: TraceStep) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if step.after_info and "group_sub_table" in step.after_info:
        group_col, group_info = step.after_info["group_sub_table"]
        result["group_column"] = group_col
        result["group_count"] = len(group_info)
    return result


def _build_answer_evidence(step: TraceStep) -> Dict[str, Any]:
    if not step.answer:
        return {}
    # Check if answer appears in the before_table cells
    cells = set()
    for row in step.before_table[1:]:
        for cell in row:
            cells.add(str(cell).strip().lower())
    answer_str = str(step.answer).strip().lower()
    parts = [p.strip() for p in answer_str.split("|")]
    found = all(p.lower() in cells for p in parts if p)
    return {
        "answer": step.answer,
        "answer_in_table": found,
    }


def build_step_evidence(step: TraceStep, question: str) -> StepEvidence:
    """Build evidence for a single step based on its operation type."""
    op_name = step.operation_name
    warnings: List[str] = []
    row_diff: Dict[str, Any] = {}
    column_diff: Dict[str, Any] = {}
    sort_evidence: Dict[str, Any] = {}
    group_evidence: Dict[str, Any] = {}
    answer_evidence: Dict[str, Any] = {}

    if "select_row" in op_name:
        row_diff = _build_row_diff(step)
        summary = f"Row selection: {row_diff.get('before_count', '?')} → {row_diff.get('after_count', '?')} rows"

    elif "select_column" in op_name:
        col_result = _build_column_diff(step, question)
        column_diff, warnings = col_result[0], col_result[1]
        summary = f"Column selection: kept {len(column_diff.get('kept_headers', []))}, dropped {len(column_diff.get('dropped_headers', []))}"

    elif "sort_column" in op_name:
        sort_evidence = _build_sort_evidence(step)
        summary = f"Sort on {sort_evidence.get('sort_column', 'unknown column')}"

    elif "group_column" in op_name:
        group_evidence = _build_group_evidence(step)
        summary = f"Group by {group_evidence.get('group_column', 'unknown column')}, {group_evidence.get('group_count', '?')} groups"

    elif "simple_query" in op_name:
        answer_evidence = _build_answer_evidence(step)
        if not answer_evidence.get("answer_in_table", True):
            warnings.append(f"Answer '{step.answer}' not found in table cells")
        summary = f"Query answer: {step.answer}"

    else:
        summary = f"Operation: {op_name}"

    return StepEvidence(
        step_num=step.step_num,
        operation_name=op_name,
        kind=op_name,
        summary=summary,
        row_diff=row_diff,
        column_diff=column_diff,
        sort_evidence=sort_evidence,
        group_evidence=group_evidence,
        answer_evidence=answer_evidence,
        warnings=warnings,
    )


def build_evidence_pack(trace: OperationTrace, question: Optional[str] = None) -> List[StepEvidence]:
    """Build evidence for all steps in a trace."""
    if question is None:
        question = trace.question
    return [build_step_evidence(step, question) for step in trace.steps]
