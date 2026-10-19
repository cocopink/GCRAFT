"""HeaderTree module for parsing compound headers and compound cells.

Parses compound table headers into a tree structure, detects and splits
compound cells in corresponding columns. Zero LLM cost — pure computation.

Error Pattern Coverage:
- Pattern 4: Compound cell encoding (e.g., "11 may 1917 @ 1950 hours")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

from .detect import find_compound_columns, get_separator, SEPARATOR_PRIORITY


@dataclass
class HeaderNode:
    """A node in the header tree representing a column or sub-column."""

    label: str
    column_indices: List[int] = field(default_factory=list)
    children: List['HeaderNode'] = field(default_factory=list)
    parent: Optional['HeaderNode'] = None
    is_compound: bool = False
    separator: str = ""
    cell_separators: List[str] = field(default_factory=list)
    depth: int = 0

    def add_child(self, child: 'HeaderNode') -> None:
        """Add a child node and set parent reference."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)


@dataclass
class CellParsingRule:
    """Rule for parsing compound cells in a specific column."""

    column_idx: int
    header_node: HeaderNode
    detected_separator: str
    confidence: float
    sub_field_names: List[str] = field(default_factory=list)


# Separators commonly found in compound cells (distinct from header separators)
CELL_SEPARATOR_CANDIDATES = ['@', '|', '(', ';', ':', '/', '-']


@dataclass
class HeaderTree:
    """Tree representation of table headers with compound cell parsing."""

    root: HeaderNode
    flat_columns: List[str]
    compound_count: int
    max_depth: int
    cell_parsing_rules: Dict[int, CellParsingRule] = field(default_factory=dict)

    @classmethod
    def build(cls, table_text: List[List]) -> 'HeaderTree':
        """Build a HeaderTree from table text.

        Args:
            table_text: Table as list of lists (first row is header).

        Returns:
            HeaderTree instance.
        """
        if not table_text or len(table_text) == 0:
            return cls(
                root=HeaderNode(label="(empty)", column_indices=[], depth=0),
                flat_columns=[],
                compound_count=0,
                max_depth=0,
            )

        header = table_text[0]
        compound_indices = find_compound_columns(table_text)

        root = HeaderNode(label="(root)", depth=0)
        flat_columns = []
        compound_count = 0
        max_depth = 0

        for col_idx, cell in enumerate(header):
            cell_str = str(cell)
            if col_idx in compound_indices:
                # Compound header — split into sub-nodes
                separator = get_separator(cell_str)
                parts = [p.strip() for p in cell_str.split(separator) if p.strip()]
                parent_node = HeaderNode(
                    label=cell_str,
                    column_indices=[col_idx],
                    is_compound=True,
                    separator=separator,
                    depth=1,
                )
                root.add_child(parent_node)
                compound_count += 1

                for part in parts:
                    child = HeaderNode(
                        label=part,
                        column_indices=[col_idx],
                        depth=2,
                    )
                    parent_node.add_child(child)
                    flat_columns.append(part)

                max_depth = max(max_depth, 2)
            else:
                # Simple header
                node = HeaderNode(
                    label=cell_str,
                    column_indices=[col_idx],
                    depth=1,
                )
                root.add_child(node)
                flat_columns.append(cell_str)
                max_depth = max(max_depth, 1)

        tree = cls(
            root=root,
            flat_columns=flat_columns,
            compound_count=compound_count,
            max_depth=max_depth,
        )
        tree._table_text = table_text
        tree._detect_compound_cells(table_text)

        return tree

    def _detect_compound_cells(self, table_text: List[List]) -> None:
        """Detect and record compound cell parsing rules for compound columns.

        Args:
            table_text: Table as list of lists (first row is header).
        """
        if len(table_text) < 2:
            return

        # Find compound header nodes
        compound_nodes = []
        for child in self.root.children:
            if child.is_compound:
                compound_nodes.append(child)

        if not compound_nodes:
            return

        for node in compound_nodes:
            col_idx = node.column_indices[0]
            header_sep = node.separator

            # Sample first 20 non-empty data rows
            values = []
            for row in table_text[1:21]:
                if col_idx < len(row):
                    val = str(row[col_idx]).strip()
                    if val:
                        values.append(val)

            if not values:
                continue

            # Detect extra separators not used in header
            extra_sep_counts: Counter = Counter()
            for val in values:
                for sep in CELL_SEPARATOR_CANDIDATES:
                    if sep != header_sep and sep in val:
                        extra_sep_counts[sep] += 1

            if not extra_sep_counts:
                continue

            # Use the most common extra separator
            best_sep, best_count = extra_sep_counts.most_common(1)[0]
            consistency = best_count / len(values)

            # Only adopt if separator appears in >= 40% of non-empty values
            if consistency >= 0.4:
                # Infer sub-field names from header children
                sub_field_names = [c.label for c in node.children]

                self.cell_parsing_rules[col_idx] = CellParsingRule(
                    column_idx=col_idx,
                    header_node=node,
                    detected_separator=best_sep,
                    confidence=round(consistency, 3),
                    sub_field_names=sub_field_names,
                )

    def parse_compound_cells(
        self, table_text: List[List]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Parse compound cells according to detected rules.

        Args:
            table_text: Table as list of lists (first row is header).

        Returns:
            Dict mapping column index to list of parsed cell info.
        """
        result: Dict[int, List[Dict[str, Any]]] = {}

        for col_idx, rule in self.cell_parsing_rules.items():
            sep = rule.detected_separator
            parsed_cells = []

            for row_idx, row in enumerate(table_text[1:], start=1):
                if col_idx >= len(row):
                    continue
                cell = str(row[col_idx])
                if not cell.strip() or sep not in cell:
                    continue

                parts = cell.split(sep, 1)
                parsed = {
                    "row_idx": row_idx,
                    "original": cell,
                    "parts": {},
                }
                for i, name in enumerate(rule.sub_field_names):
                    if i < len(parts):
                        parsed["parts"][name] = parts[i].strip()
                    else:
                        parsed["parts"][name] = ""

                parsed_cells.append(parsed)

            if parsed_cells:
                result[col_idx] = parsed_cells

        return result

    def serialize_for_prompt(self, include_cells: bool = True) -> str:
        """Serialize header tree into LLM-readable text.

        Args:
            include_cells: Whether to include compound cell examples.

        Returns:
            Formatted string for LLM prompt.
        """
        if self.compound_count == 0:
            return ""

        lines = ["Column Hierarchy:"]

        for child in self.root.children:
            if child.is_compound:
                lines.append(f"  {child.label} (compound, separator=\"{child.separator}\")")
                for grandchild in child.children:
                    lines.append(f"    - {grandchild.label}")

                # Add compound cell info if available
                if include_cells and child.column_indices[0] in self.cell_parsing_rules:
                    rule = self.cell_parsing_rules[child.column_indices[0]]
                    lines.append(
                        f"    [NOTE: cells contain \"{rule.detected_separator}\" "
                        f"separator, confidence={rule.confidence:.0%}]"
                    )
                    # Show up to 2 examples
                    col_idx = child.column_indices[0]
                    example_count = 0
                    for row_idx, row in enumerate(
                        self._table_text if hasattr(self, '_table_text') else []
                    ):
                        if example_count >= 2:
                            break
                        if col_idx < len(row):
                            cell = str(row[col_idx])
                            if rule.detected_separator in cell:
                                parts = cell.split(rule.detected_separator, 1)
                                lines.append(
                                    f'      Example: "{cell}"'
                                )
                                for i, name in enumerate(rule.sub_field_names):
                                    if i < len(parts):
                                        lines.append(
                                            f'        -> {name}: "{parts[i].strip()}"'
                                        )
                                example_count += 1
            else:
                lines.append(f"  {child.label} (simple)")

        return "\n".join(lines)

    def set_table_text(self, table_text: List[List]) -> None:
        """Store table_text reference for example generation in serialize_for_prompt.

        Args:
            table_text: Table as list of lists.
        """
        self._table_text = table_text
