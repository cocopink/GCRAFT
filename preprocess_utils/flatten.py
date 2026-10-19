"""Core flattening logic for tables."""

from typing import List, Tuple, Dict, Any
from .detect import has_compound_headers, find_compound_columns, get_separator, SEPARATOR_PRIORITY


def split_cell_value(value: str, primary_separator: str, expected_count: int) -> List[str]:
    """Split a cell value intelligently with fallback strategies.

    Strategy 1: Try primary separator from header
    Strategy 2: Fallback to space separator
    Strategy 3: Copy value to all columns

    Args:
        value: Cell value to split
        primary_separator: Separator used in the header (e.g., '/', '\n')
        expected_count: Number of expected parts after splitting

    Returns:
        List of split values
    """
    value_str = str(value)

    # Strategy 1: Try primary separator
    if primary_separator in value_str:
        parts = value_str.split(primary_separator)
        if len(parts) == expected_count:
            return [p.strip() for p in parts]
        # If we got more parts than expected, take first N-1 and join the rest
        if len(parts) > expected_count:
            result = [p.strip() for p in parts[:expected_count-1]]
            result.append(' '.join(p.strip() for p in parts[expected_count-1:]))
            return result

    # Strategy 2: Fallback to space separator
    # Only if primary separator is not space
    if primary_separator != ' ' and ' ' in value_str:
        parts = value_str.split()
        # Try to create expected_count parts
        if len(parts) >= expected_count:
            return parts[:expected_count]
        else:
            # Pad with empty strings
            return parts + [''] * (expected_count - len(parts))

    # Strategy 3: Copy value to all columns (fallback)
    return [value_str] * expected_count


def flatten_table(table: List[List]) -> Tuple[List[List], Dict[str, Any]]:
    """Flatten a table with compound headers.

    Detects compound column headers and splits them into multiple columns.
    Cell values are intelligently split based on the separator used in the header.

    Args:
        table: Table as list of lists (first row is header)

    Returns:
        Tuple of (flattened_table, metadata)
        - flattened_table: Table with compound headers split into multiple columns
        - metadata: Dict with keys:
            * 'flattened': bool - Whether flattening was performed
            * 'split_columns': int - Number of columns that were split
            * 'original_shape': tuple - Original (rows, cols) shape
            * 'new_shape': tuple - New (rows, cols) shape
    """
    metadata = {
        'flattened': False,
        'split_columns': 0,
        'original_shape': (len(table), len(table[0]) if table else 0),
        'new_shape': None,
    }

    # Handle empty table
    if not table or len(table) == 0:
        return table, metadata

    # Check if table has compound headers
    if not has_compound_headers(table):
        metadata['new_shape'] = metadata['original_shape']
        return table, metadata

    # Find compound columns
    compound_indices = find_compound_columns(table)
    if not compound_indices:
        metadata['new_shape'] = metadata['original_shape']
        return table, metadata

    # Build new table
    new_table = []
    split_count = 0

    for row_idx, row in enumerate(table):
        new_row = []

        for col_idx, cell in enumerate(row):
            if col_idx in compound_indices:
                # Get separator from header
                separator = get_separator(table[0][col_idx])

                # Count expected parts from header
                header_parts = table[0][col_idx].split(separator)
                expected_count = len(header_parts)

                # Split the cell value
                if row_idx == 0:
                    # Header row: split by separator
                    split_values = [part.strip() for part in header_parts]
                else:
                    # Data row: use intelligent splitting
                    split_values = split_cell_value(cell, separator, expected_count)

                new_row.extend(split_values)

                if row_idx == 0:
                    split_count += 1
            else:
                # Non-compound column: keep as is
                new_row.append(cell)

        new_table.append(new_row)

    # Update metadata
    metadata['flattened'] = True
    metadata['split_columns'] = split_count
    metadata['new_shape'] = (len(new_table), len(new_table[0]) if new_table else 0)

    return new_table, metadata


def flatten_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten all tables in a dataset.

    Args:
        dataset: List of sample dicts, each with a 'table_text' key (TableQA) or 'table' key (TableFV)

    Returns:
        Updated dataset with flattened tables and metadata
    """
    flattened_count = 0

    for sample in dataset:
        # Support both 'table_text' (TableQA) and 'table' (TableFV) keys
        table_key = 'table_text' if 'table_text' in sample else 'table'
        if table_key not in sample:
            continue

        table = sample[table_key]
        flattened_table, metadata = flatten_table(table)

        sample[table_key] = flattened_table
        sample['flatten_metadata'] = metadata

        if metadata['flattened']:
            flattened_count += 1

    print(f"Flattened {flattened_count}/{len(dataset)} samples")
    return dataset
