"""Column normalization utilities for table analysis.

Provides fuzzy matching between question keywords and column values,
basic stemming, mixed-format detection, and numeric extraction.

Error Pattern Coverage:
- Pattern 2: Question-column value mismatch (e.g., "points" vs "pts")
- Pattern 3: Mixed numeric/text formats in columns (e.g., "heat 3" vs "3")
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pylcs


@dataclass
class ColumnNormalization:
    """Result of normalizing a column against question keywords."""

    column_name: str
    question_form: str
    matched_values: List[str] = field(default_factory=list)
    expanded_forms: Dict[str, str] = field(default_factory=dict)


def simple_stem(word: str) -> str:
    """Remove common English suffixes for fuzzy matching.

    Args:
        word: Input word.

    Returns:
        Stemmed word.
    """
    word = word.lower().strip()
    if len(word) <= 3:
        return word
    # Handle doubled consonant before -ing/-ed (running→run, stopped→stop)
    for suffix in ["ing", "ed"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]
            return stem
    for suffix in ["tion", "ness", "ment", "ous", "ive", "ful", "less", "ly", "es", "er", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def fuzzy_match_values(
    question_keyword: str,
    column_values: List[str],
    threshold: float = 0.8,
) -> List[str]:
    """Find column values that fuzzy-match a question keyword via LCS.

    Args:
        question_keyword: Keyword extracted from the question.
        column_values: Non-empty cell values from the target column.
        threshold: Minimum similarity ratio (0-1).

    Returns:
        List of matching values (original casing preserved).
    """
    kw = question_keyword.lower().strip()
    if not kw:
        return []

    matches = []
    for val in column_values:
        v = val.strip()
        if not v:
            continue
        lcs_len = pylcs.lcs(kw, v.lower())
        # Use min-length denominator: short strings like "pts" vs "points"
        # get similarity 3/3=1.0 instead of 3/6=0.5
        denom = min(len(kw), len(v))
        if denom == 0:
            continue
        if lcs_len / denom >= threshold:
            matches.append(v)
    return matches


def normalize_column(
    question_keywords: List[str],
    table_text: List[List],
    column_idx: int,
) -> Optional[ColumnNormalization]:
    """Fuzzy-match question keywords against a single column.

    Args:
        question_keywords: Keywords extracted from the question.
        table_text: Table as list of lists (first row is header).
        column_idx: Target column index.

    Returns:
        ColumnNormalization if matches found, else None.
    """
    if not table_text or column_idx >= len(table_text[0]):
        return None

    column_name = str(table_text[0][column_idx])
    column_values = [
        str(row[column_idx]) for row in table_text[1:] if column_idx < len(row)
    ]
    column_values = [v for v in column_values if v.strip()]
    if not column_values:
        return None

    all_matches: Dict[str, str] = {}  # original_value -> matched_keyword
    matched_keywords: List[str] = []

    # Match keywords against column NAME first (e.g., "points" matches "pts")
    name_matched = False
    for keyword in question_keywords:
        hits = fuzzy_match_values(keyword, [column_name])
        if hits:
            name_matched = True
            if keyword not in matched_keywords:
                matched_keywords.append(keyword)
            break

    # Also match against column VALUES
    for keyword in question_keywords:
        hits = fuzzy_match_values(keyword, column_values)
        for h in hits:
            if h not in all_matches:
                all_matches[h] = keyword
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)

    # If keyword matched the column name but no values, still report it
    if name_matched and not all_matches:
        return ColumnNormalization(
            column_name=column_name,
            question_form=", ".join(matched_keywords),
            matched_values=[column_name],
            expanded_forms={column_name: simple_stem(matched_keywords[0])},
        )

    if not all_matches and not name_matched:
        return None

    expanded: Dict[str, str] = {val: simple_stem(kw) for val, kw in all_matches.items()}

    return ColumnNormalization(
        column_name=column_name,
        question_form=", ".join(matched_keywords),
        matched_values=list(all_matches.keys()),
        expanded_forms=expanded,
    )


# --- Mixed-format detection & normalization (Pattern 3) ---

_EXCLUDED_VALUES = frozenset(
    {"—", "–", "DNS", "DNF", "did not start", "did not finish", "N/A", "n/a", ""}
)


def extract_numeric_from_mixed(value: str) -> Optional[float]:
    """Extract a numeric value from a mixed-format string.

    Patterns tried (in order):
      1. Parenthesized number: ``"(5th)"`` → 5
      2. Ordinal phrase: ``"round of 16"`` → 16, ``"heat 3"`` → 3
      3. Trailing standalone number

    Args:
        value: Cell value string.

    Returns:
        Extracted float, or None for non-numeric values.
    """
    v = value.strip()
    if v in _EXCLUDED_VALUES:
        return None

    # Parenthesized number
    m = re.search(r"\((\d+)", v)
    if m:
        return float(m.group(1))

    # Ordinal phrase
    m = re.search(
        r"(?:round|heat|race|game|set|match|stage|leg|quarter|semi|final)\s+(?:of\s+)?(\d+)",
        v,
        re.IGNORECASE,
    )
    if m:
        return float(m.group(1))

    # Standalone number (prefer the last one for strings like "round 1 (2)")
    nums = re.findall(r"(\d+(?:\.\d+)?)", v)
    if nums:
        return float(nums[-1])

    return None


def detect_mixed_format_column(values: List[str]) -> bool:
    """Check whether a column mixes numeric and non-numeric representations.

    Args:
        values: Non-empty cell values from a column.

    Returns:
        True if the column contains both extractable numbers and pure text.
    """
    has_num = False
    has_text = False
    for v in values:
        v = v.strip()
        if not v:
            continue
        if extract_numeric_from_mixed(v) is not None:
            has_num = True
        else:
            has_text = True
        if has_num and has_text:
            return True
    return False


def normalize_column_format(values: List[str]) -> Dict[str, Optional[float]]:
    """Batch-normalize a column that has mixed numeric/text formats.

    Args:
        values: Cell values from the column.

    Returns:
        Dict mapping original value string → extracted float (or None).
    """
    return {v.strip(): extract_numeric_from_mixed(v) for v in values if v.strip()}
