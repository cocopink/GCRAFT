"""TableAnalyzer: zero-LLM-cost table structure analysis.

Orchestrates HeaderTree, ColumnNormalization, and format normalization
to produce a ``table_analysis`` dict stored on each sample.  Downstream
consumer points (table2string, final_query, select_row, sort_by) read
from this dict to inject analysis hints into LLM prompts.

Error Pattern Coverage:
- Pattern 1: Answer format mismatch (exact-value reminder)
- Pattern 2: Question-column value mismatch (fuzzy normalization)
- Pattern 3: Mixed numeric/text formats (numeric extraction)
- Pattern 4: Compound cell encoding (HeaderTree)
"""

import re
from typing import Any, Dict, List

from preprocess_utils.header_tree import HeaderTree
from preprocess_utils.column_norm import (
    ColumnNormalization,
    detect_mixed_format_column,
    normalize_column,
    normalize_column_format,
)

_STOP_WORDS = frozenset(
    {
        "what", "which", "who", "where", "when", "how", "many", "much",
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "for", "and", "or", "that", "this", "with", "on", "at", "by",
        "from", "do", "does", "did", "has", "have", "had", "be", "been",
        "being", "not", "but", "if", "then", "than", "so", "as", "it",
        "its", "no", "yes", "all", "each", "every", "any", "other",
    }
)


def _extract_keywords(question: str) -> List[str]:
    """Extract meaningful words from a question for column matching."""
    words = re.findall(r"\b[a-zA-Z]+\b", question.lower())
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


class TableAnalyzer:
    """Analyze table structure and produce hints for downstream LLM prompts."""

    def analyze(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run all analysis passes on a single sample.

        Args:
            sample: Dataset sample with ``table`` (List[List]) and
                    ``question`` / ``statement`` keys.

        Returns:
            Analysis dict suitable for storing as ``sample["table_analysis"]``.
        """
        table_text = sample.get("table", sample.get("table_text", []))
        question = sample.get("question", sample.get("statement", ""))

        analysis: Dict[str, Any] = {}

        # Pattern 4 — HeaderTree
        tree = HeaderTree.build(table_text)
        analysis["header_tree_text"] = tree.serialize_for_prompt()

        # Pattern 1 — Answer format hint (always present)
        analysis["answer_format_hint"] = (
            "RULE: Your answer must use the EXACT values as they appear in the table."
        )

        # Pattern 2 — Column normalization
        keywords = _extract_keywords(question)
        column_norms: List[Dict[str, Any]] = []
        n_cols = len(table_text[0]) if table_text else 0
        for col_idx in range(n_cols):
            norm = normalize_column(keywords, table_text, col_idx)
            if norm and norm.matched_values:
                column_norms.append(
                    {
                        "column_name": norm.column_name,
                        "matched_values": norm.matched_values,
                        "hint": (
                            f'Column "{norm.column_name}" has similar values: '
                            f'{", ".join(norm.matched_values[:5])} '
                            f"— treat them as equivalent."
                        ),
                    }
                )
        analysis["column_normalizations"] = column_norms

        # Pattern 3 — Format normalization
        format_norms: Dict[str, Dict[str, Any]] = {}
        for col_idx in range(n_cols):
            col_name = str(table_text[0][col_idx]) if table_text else ""
            values = [
                str(row[col_idx]) for row in table_text[1:] if col_idx < len(row)
            ]
            if detect_mixed_format_column(values):
                mapping = normalize_column_format(values)
                non_none = {k: v for k, v in mapping.items() if v is not None}
                if non_none:
                    format_norms[col_name] = {
                        "mapping": mapping,
                        "hint": (
                            f'Column "{col_name}" has mixed formats. '
                            f"Normalized numeric values: "
                            f"{dict(list(non_none.items())[:5])}"
                        ),
                    }
        analysis["format_normalizations"] = format_norms

        return analysis


def format_analysis_for_prompt(analysis: Dict[str, Any]) -> str:
    """Serialize analysis dict into a single LLM-readable block.

    Used by ``table2string`` to inject context between header and data rows.

    Args:
        analysis: Dict produced by ``TableAnalyzer.analyze()``.

    Returns:
        Multi-line string (empty if nothing to report).
    """
    parts: List[str] = []

    if analysis.get("header_tree_text"):
        parts.append(analysis["header_tree_text"])

    for norm in analysis.get("column_normalizations", []):
        parts.append(norm["hint"])

    for _col, info in analysis.get("format_normalizations", {}).items():
        parts.append(info["hint"])

    return "\n".join(parts)
