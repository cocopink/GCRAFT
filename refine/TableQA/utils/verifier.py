from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import re
from collections import Counter


class Verdict(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class StepVerification:
    step_num: int
    operation_name: str
    action_str: str
    verdict: Verdict
    checks: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


@dataclass
class VerificationReport:
    steps: List[StepVerification] = field(default_factory=list)
    total_steps: int = 0
    failed_steps: List[int] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return len(self.failed_steps) > 0

    @property
    def first_fail_step(self) -> Optional[int]:
        return self.failed_steps[0] if self.failed_steps else None

    def to_critic_prompt(self) -> str:
        if not self.steps:
            return ""
        lines = ["\n[Automated Step Verification Results]\n"]
        for sv in self.steps:
            icon = "OK" if sv.verdict == Verdict.PASS else ("!!" if sv.verdict == Verdict.FAIL else "--")
            lines.append(f"  Step {sv.step_num} ({sv.operation_name}): [{icon}] {sv.summary}")
            for check in sv.checks:
                status = "PASS" if check["passed"] else "FAIL"
                lines.append(f"    - {check['name']}: [{status}] {check['detail']}")
        if self.failed_steps:
            lines.append(f"\n  ** Verification detected potential issues at steps: {self.failed_steps}")
        else:
            lines.append(f"\n  ** All verifiable steps passed automated checks.")
        lines.append("[End Verification Results]\n")
        return "\n".join(lines)

    def to_critic_step_prompt(self, step_num: int) -> str:
        for sv in self.steps:
            if sv.step_num == step_num:
                lines = [f"\n[Automated Verification for Step {step_num} ({sv.operation_name})]\n"]
                for check in sv.checks:
                    status = "PASS" if check["passed"] else "FAIL"
                    lines.append(f"  - {check['name']}: [{status}] {check['detail']}")
                if sv.verdict == Verdict.FAIL:
                    lines.append(f"\n  ** This step FAILED automated checks — likely incorrect.")
                elif sv.verdict == Verdict.PASS:
                    lines.append(f"\n  ** This step PASSED automated checks — but may still have semantic issues.")
                lines.append("[End Verification]\n")
                return "\n".join(lines)
        return ""


# --- Helper Functions ---

def _parse_selected_rows(action_str: str) -> List[str]:
    match = re.findall(r'row\s+(\d+)', action_str)
    return match


def _parse_sort_column(action_str: str) -> Optional[str]:
    match = re.findall(r'f_sort_column\((.*?)\)', action_str)
    return match[0].strip() if match else None


def _parse_sort_order(action_str: str) -> Optional[str]:
    match = re.findall(r'the order is "(.*?)"', action_str)
    return match[0].strip() if match else None


def _parse_group_column(action_str: str) -> Optional[str]:
    match = re.findall(r'f_group_column\((.*?)\)', action_str)
    return match[0].strip() if match else None


_STOP_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to',
    'for', 'and', 'or', 'what', 'which', 'how', 'many', 'much', 'does',
    'did', 'do', 'that', 'this', 'it', 'its', 'on', 'at', 'by', 'with',
    'from', 'as', 'be', 'been', 'has', 'have', 'had', 'not', 'but', 'if',
    'than', 'so', 'no', 'who', 'whom', 'where', 'when', 'why', 'about',
    'up', 'out', 'also', 'just', 'only', 'very', 'most', 'more', 'some',
    'all', 'any', 'each', 'every', 'other', 'into', 'over', 'after',
    'before', 'between', 'under', 'again', 'then', 'once', 'here',
    'there', 'these', 'those', 'such', 'can', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall',
})


def _extract_question_keywords(question: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z]+\b', question.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _safe_float(s: str) -> Optional[float]:
    cleaned = re.sub(r'[^\d.\-]', '', s)
    if not cleaned or cleaned == '-' or cleaned == '.':
        return None
    try:
        return float(cleaned)
    except (ValueError, OverflowError):
        return None


def _build_summary(checks: List[Dict], step_num: int, op_name: str) -> str:
    failed = [c for c in checks if not c["passed"]]
    if not failed:
        return f"Step {step_num} ({op_name}): All checks passed"
    details = "; ".join(f"{c['name']} failed: {c['detail']}" for c in failed)
    return f"Step {step_num} ({op_name}): ISSUES DETECTED - {details}"


def _make_check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


# --- Per-Operation Verifiers ---

def _verify_select_row(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    pre_data_rows = len(pre_table) - 1
    post_data_rows = len(post_table) - 1

    selected_row_strs = _parse_selected_rows(action_str)
    selected_indices = []
    for rs in selected_row_strs:
        try:
            idx = int(rs) - 1
            if 0 <= idx < pre_data_rows:
                selected_indices.append(idx)
        except ValueError:
            pass

    # Check 1: Row indices in bounds
    all_in_bounds = len(selected_indices) == len(selected_row_strs) and all(
        0 <= idx < pre_data_rows for idx in selected_indices
    )
    checks.append(_make_check(
        "row_bounds",
        all_in_bounds,
        f"Selected {len(selected_indices)}/{len(selected_row_strs)} rows within bounds (total: {pre_data_rows})"
    ))

    # Check 2: Selected rows contain question keywords
    question_keywords = _extract_question_keywords(question)
    if question_keywords and selected_indices and pre_data_rows > 0:
        relevant_count = 0
        for idx in selected_indices:
            if idx < len(pre_table):
                row_text = " ".join(str(c).lower() for c in pre_table[idx + 1])
                if any(kw.lower() in row_text for kw in question_keywords):
                    relevant_count += 1
        relevance_ratio = relevant_count / len(selected_indices)
        checks.append(_make_check(
            "keyword_relevance",
            relevance_ratio >= 0.3,
            f"{relevant_count}/{len(selected_indices)} selected rows contain question keywords"
        ))

    # Check 3: Meaningful selection (not selecting all rows)
    if pre_data_rows > 3:
        selection_ratio = post_data_rows / max(pre_data_rows, 1)
        checks.append(_make_check(
            "meaningful_selection",
            selection_ratio < 0.95,
            f"Selected {post_data_rows}/{pre_data_rows} rows ({selection_ratio:.0%})"
        ))

    verdict = _compute_verdict(checks)
    return StepVerification(step_num, "select_row", action_str, verdict, checks, _build_summary(checks, step_num, "select_row"))


def _verify_select_column(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    pre_headers = [h.lower() for h in pre_table[0]] if pre_table else []
    post_headers = [h.lower() for h in post_table[0]] if post_table and len(post_table) > 0 else []

    # Check 1: Remaining columns exist in original headers
    all_exist = all(h in pre_headers for h in post_headers)
    checks.append(_make_check(
        "column_existence",
        all_exist,
        f"Retained {len(post_headers)}/{len(pre_headers)} columns"
    ))

    # Check 2: Question-relevant columns not dropped
    question_keywords = _extract_question_keywords(question)
    if question_keywords and pre_headers:
        dropped = set(pre_headers) - set(post_headers)
        important_dropped = [h for h in dropped if any(kw.lower() in h for kw in question_keywords)]
        checks.append(_make_check(
            "relevant_columns_retained",
            len(important_dropped) == 0,
            f"Dropped: {dropped or 'none'}" + (f"; potentially relevant: {important_dropped}" if important_dropped else "")
        ))

    # Check 3: Meaningful reduction
    checks.append(_make_check(
        "meaningful_reduction",
        len(post_headers) < len(pre_headers) or len(pre_headers) <= 2,
        f"Retained {len(post_headers)}/{len(pre_headers)} columns"
    ))

    verdict = _compute_verdict(checks)
    return StepVerification(step_num, "select_column", action_str, verdict, checks, _build_summary(checks, step_num, "select_column"))


def _verify_sort_column(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    if not pre_table or not post_table:
        return StepVerification(step_num, "sort_column", action_str, Verdict.SKIP, [], "Cannot verify: empty table")

    pre_headers = pre_table[0]
    post_headers = post_table[0]

    # Check 1: Headers preserved
    checks.append(_make_check(
        "headers_preserved",
        pre_headers == post_headers,
        "Headers unchanged after sort" if pre_headers == post_headers else "Headers changed!"
    ))

    # Check 2: Sort order is actually valid
    sort_col_name = _parse_sort_column(action_str)
    if sort_col_name:
        col_idx = next((i for i, h in enumerate(pre_headers) if h.lower() == sort_col_name.lower()), None)
        if col_idx is not None:
            pre_values = [str(pre_table[r][col_idx]).strip() for r in range(1, len(pre_table))]
            post_values = [str(post_table[r][col_idx]).strip() for r in range(1, len(post_table))]

            # Try numeric comparison
            pre_numeric = [_safe_float(v) for v in pre_values]
            if all(v is not None for v in pre_numeric):
                post_numeric = [_safe_float(v) for v in post_values]
                if all(v is not None for v in post_numeric):
                    is_asc = post_numeric == sorted(pre_numeric)
                    is_desc = post_numeric == sorted(pre_numeric, reverse=True)
                    checks.append(_make_check(
                        "sort_order_valid",
                        is_asc or is_desc,
                        f"Numeric sort: ascending={is_asc}, descending={is_desc}"
                    ))
                else:
                    checks.append(_make_check("sort_order_valid", True, "Cannot verify post-values (mixed types)"))
            else:
                # String comparison
                is_asc = post_values == sorted(pre_values, key=lambda x: x.lower())
                is_desc = post_values == sorted(pre_values, key=lambda x: x.lower(), reverse=True)
                checks.append(_make_check(
                    "sort_order_valid",
                    is_asc or is_desc,
                    f"String sort: ascending={is_asc}, descending={is_desc}"
                ))
        else:
            checks.append(_make_check("sort_column_exists", False, f"Sort column '{sort_col_name}' not found in headers"))
    else:
        checks.append(_make_check("sort_column_parsed", False, "Could not parse sort column from action"))

    # Check 3: Row count preserved
    checks.append(_make_check(
        "row_count_preserved",
        len(pre_table) == len(post_table),
        f"{len(pre_table)} rows before, {len(post_table)} rows after"
    ))

    verdict = _compute_verdict(checks)
    return StepVerification(step_num, "sort_column", action_str, verdict, checks, _build_summary(checks, step_num, "sort_column"))


def _verify_group_column(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    if not pre_table:
        return StepVerification(step_num, "group_column", action_str, Verdict.SKIP, [], "Cannot verify: empty table")

    group_col = _parse_group_column(action_str)
    pre_headers_lower = [h.lower() for h in pre_table[0]]

    # Check 1: Group column exists
    col_exists = group_col and group_col.lower() in pre_headers_lower
    checks.append(_make_check(
        "column_exists",
        col_exists,
        f"Group column '{group_col}' {'found' if col_exists else 'NOT found'} in headers"
    ))

    # Check 2: Value counts are accurate
    if col_exists and "group_sub_table" in post_info:
        gcol, ginfo = post_info["group_sub_table"]
        col_idx = pre_headers_lower.index(group_col.lower())
        actual_values = [str(pre_table[r][col_idx]).strip() for r in range(1, len(pre_table))]
        actual_counts = Counter(actual_values)
        verified = all(actual_counts.get(v, 0) == c for v, c in ginfo)
        checks.append(_make_check(
            "group_counts_accurate",
            verified,
            f"Group info: {len(ginfo)} groups, counts match={verified}"
        ))
    elif col_exists:
        checks.append(_make_check("group_metadata_set", False, "group_sub_table metadata not set"))

    # Check 3: Table unchanged (metadata-only operation)
    table_unchanged = (pre_table == post_table)
    checks.append(_make_check(
        "table_unchanged",
        table_unchanged,
        "Table unchanged (metadata-only)" if table_unchanged else "Table was modified!"
    ))

    verdict = _compute_verdict(checks)
    return StepVerification(step_num, "group_column", action_str, verdict, checks, _build_summary(checks, step_num, "group_column"))


def _verify_add_column(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    if not pre_table or not post_table:
        return StepVerification(step_num, "add_column", action_str, Verdict.SKIP, [], "Cannot verify: empty table")

    pre_headers_lower = [h.lower() for h in pre_table[0]]
    post_headers_lower = [h.lower() for h in post_table[0]]
    new_headers = set(post_headers_lower) - set(pre_headers_lower)

    # Check 1: Exactly one new column added
    checks.append(_make_check(
        "unique_column_name",
        len(new_headers) == 1,
        f"New columns: {new_headers}" if new_headers else "No new column added"
    ))

    # Check 2: Values are substrings of row cells
    if new_headers and len(post_table) > 1:
        new_col_name = list(new_headers)[0]
        new_col_idx = post_headers_lower.index(new_col_name)
        new_values = [str(post_table[r][new_col_idx]).strip() for r in range(1, len(post_table))]

        all_substrings = True
        for row_idx, val in enumerate(new_values):
            if not val:
                all_substrings = False
                break
            row_cells = [str(c).lower() for c in pre_table[row_idx + 1]] if row_idx + 1 < len(pre_table) else []
            if not any(val.lower() in c for c in row_cells):
                all_substrings = False
                break
        checks.append(_make_check(
            "values_are_substrings",
            all_substrings,
            f"All {len(new_values)} values are substrings of row cells" if all_substrings else "Some values not found in row cells"
        ))

    # Check 3: Not all values identical
    if new_headers:
        unique_vals = len(set(new_values))
        checks.append(_make_check(
            "values_not_identical",
            unique_vals > 1,
            f"{unique_vals} unique values out of {len(new_values)} rows"
        ))

    # Check 4: Row count preserved
    checks.append(_make_check(
        "row_count_preserved",
        len(pre_table) == len(post_table),
        f"{len(pre_table)} rows"
    ))

    verdict = _compute_verdict(checks)
    return StepVerification(step_num, "add_column", action_str, verdict, checks, _build_summary(checks, step_num, "add_column"))


def _verify_simple_query(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    checks = []
    answer = ""
    if operation and operation.get("parameter_and_conf"):
        answer = operation["parameter_and_conf"][0][0] if operation["parameter_and_conf"] else ""

    # Check 1: Answer non-empty
    non_empty = bool(answer and str(answer).strip())
    checks.append(_make_check(
        "answer_non_empty",
        non_empty,
        f"Answer: {str(answer)[:50]}"
    ))

    # Check 2: Answer found in table (single-value answers)
    if non_empty and pre_table:
        answer_str = str(answer).strip()
        # Handle multi-value answers (separated by |)
        parts = [p.strip() for p in answer_str.split("|")]
        table_cells = set()
        for row in pre_table[1:]:
            for cell in row:
                table_cells.add(str(cell).strip())
        found = all(p.lower() in table_cells for p in parts if p)
        checks.append(_make_check(
            "answer_in_table",
            found,
            f"Answer {'found' if found else 'NOT fully found'} in table cells"
        ))

    return StepVerification(step_num, "simple_query", action_str, Verdict.SKIP, checks, "Query result (cannot verify answer correctness deterministically)")


def _verify_unknown(step_num, action_str, operation, pre_table, post_table, post_info, question, original_table):
    return StepVerification(step_num, "unknown", action_str, Verdict.SKIP, [], f"Unknown operation, skipped")


def _compute_verdict(checks: List[Dict]) -> Verdict:
    failed = [c for c in checks if not c["passed"]]
    if not failed:
        return Verdict.PASS
    # If any critical check failed, overall FAIL
    critical_names = {"row_bounds", "column_existence", "headers_preserved", "sort_order_valid",
                       "column_exists", "group_counts_accurate", "table_unchanged", "unique_column_name"}
    critical_failed = [c["name"] for c in failed if c["name"] in critical_names]
    if critical_failed:
        return Verdict.FAIL
    return Verdict.WARN


# --- Dispatcher ---

_DISPATCH = {
    "select_row": _verify_select_row,
    "select_column": _verify_select_column,
    "sort_column": _verify_sort_column,
    "group_column": _verify_group_column,
    "add_column": _verify_add_column,
    "simple_query": _verify_simple_query,
}


def verify_chain(sample: Dict[str, Any]) -> VerificationReport:
    from critic.TableQA.tools.get_info import get_table_log

    if not sample.get("chain") or not sample["chain"]:
        return VerificationReport()

    table_log, thought_log = get_table_log(sample)
    report = VerificationReport()

    step_num = 0
    for idx in range(len(table_log) - 1):
        pre_info = table_log[idx]
        post_info = table_log[idx + 1]

        if not post_info.get("act_chain"):
            continue
        action = post_info["act_chain"][-1]
        if "skip" in str(action):
            continue

        step_num += 1
        if idx >= len(sample["chain"]):
            break
        operation = sample["chain"][idx]
        op_name = operation.get("operation_name", "")

        verifier = _DISPATCH.get(op_name, _verify_unknown)
        sv = verifier(
            step_num=step_num,
            action_str=str(action),
            operation=operation,
            pre_table=pre_info["table_text"],
            post_table=post_info["table_text"],
            post_info=post_info,
            question=sample.get("statement", ""),
            original_table=sample.get("table_text", pre_info["table_text"]),
        )
        report.steps.append(sv)

    report.total_steps = step_num
    report.failed_steps = [s.step_num for s in report.steps if s.verdict == Verdict.FAIL]
    return report
