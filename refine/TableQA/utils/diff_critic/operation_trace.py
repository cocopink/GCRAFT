"""Operation Trace Builder: convert chain log into structured before/after operation snapshots."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Table = List[List[Any]]


@dataclass
class TraceStep:
    """A single visible operation step in the reasoning chain."""

    step_num: int
    chain_idx: int
    operation_name: str
    operation: Dict[str, Any]
    action: str
    thought: str
    before_table: Table
    after_table: Table
    before_info: Dict[str, Any] = field(default_factory=dict)
    after_info: Dict[str, Any] = field(default_factory=dict)
    answer: Optional[str] = None


@dataclass
class OperationTrace:
    """Full structured trace of a sample's reasoning chain."""

    sample_id: Any
    question: str
    original_table: Table
    steps: List[TraceStep]


def _load_table_log(sample: Dict[str, Any]) -> Tuple[List[Dict], List[str]]:
    """Load table replay log and thought log from a sample."""
    from critic.TableQA.tools.get_info import get_table_log
    return get_table_log(sample)


def _last_visible_action(table_info: Dict[str, Any]) -> Optional[str]:
    """Extract the last action from act_chain, or None if empty."""
    act_chain = table_info.get("act_chain", [])
    return act_chain[-1] if act_chain else None


def build_operation_trace(sample: Dict[str, Any]) -> OperationTrace:
    """Build a structured operation trace from a sample.

    Iterates through the chain and matches each operation against
    the table_log before/after states. Skips 'skip' actions.
    """
    table_log, thought_log = _load_table_log(sample)
    steps: List[TraceStep] = []
    visible_step_num = 0

    for chain_idx, operation in enumerate(sample.get("chain", [])):
        # Need at least the next table_log entry for post-state
        if chain_idx + 1 >= len(table_log):
            break

        post_info = table_log[chain_idx + 1]
        action = _last_visible_action(post_info)

        if not action or "skip" in str(action):
            continue

        visible_step_num += 1

        # Extract answer for query operations
        answer = None
        op_name = operation.get("operation_name", "")
        if "query" in op_name and "cotable_result" in post_info:
            answer = str(post_info["cotable_result"])

        steps.append(TraceStep(
            step_num=visible_step_num,
            chain_idx=chain_idx,
            operation_name=op_name,
            operation=operation,
            action=str(action),
            thought=(
                thought_log[chain_idx + 1]
                if chain_idx + 1 < len(thought_log)
                else operation.get("thought", "")
            ),
            before_table=table_log[chain_idx].get("table_text", []),
            after_table=post_info.get("table_text", []),
            before_info=table_log[chain_idx],
            after_info=post_info,
            answer=answer,
        ))

    return OperationTrace(
        sample_id=sample.get("id"),
        question=sample.get("statement", ""),
        original_table=sample.get("table_text", []),
        steps=steps,
    )
