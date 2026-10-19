"""Detect compound headers in tables."""

from typing import List, Tuple, Dict

# Separator priority order (most common first)
# Note: \\n is for JSON-escaped newlines (two characters: backslash + n)
SEPARATOR_PRIORITY = ['/', '\\n', '\n', ' - ', '-', '|', ',']

def has_compound_headers(table: List[List]) -> bool:
    """Check if table has compound column headers.

    Args:
        table: Table as list of lists (first row is header)

    Returns:
        True if compound headers detected, False otherwise
    """
    if not table or len(table) == 0:
        return False

    header = table[0]

    for cell in header:
        cell_str = str(cell)
        for sep in SEPARATOR_PRIORITY:
            if sep in cell_str:
                return True

    return False

def find_compound_columns(table: List[List]) -> List[int]:
    """Find indices of compound columns.

    Args:
        table: Table as list of lists

    Returns:
        List of column indices that have compound headers
    """
    if not table or len(table) == 0:
        return []

    header = table[0]
    compound_indices = []

    for idx, cell in enumerate(header):
        cell_str = str(cell)
        for sep in SEPARATOR_PRIORITY:
            if sep in cell_str:
                compound_indices.append(idx)
                break

    return compound_indices

def get_separator(header_cell: str) -> str:
    """Get the separator used in a header cell.

    Args:
        header_cell: Header cell value

    Returns:
        Separator string found, or empty string if none
    """
    for sep in SEPARATOR_PRIORITY:
        if sep in str(header_cell):
            return sep
    return ""
