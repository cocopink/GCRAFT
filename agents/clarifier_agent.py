# Copyright 2024 Table-Critic contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ClarifierAgent: Schema Anchor Extractor for Table Reasoning

This agent extracts key information from tables to create lightweight schema anchors:
- Headers: Column names from the table
- Entities: Key entities mentioned in the table
- Units: Numerical units and measurement types
- Keywords: Important keywords for question answering
"""

import re
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from .multi_agent_framework import BaseAgent, AgentType


class ClarifierAgent(BaseAgent):
    """
    Clarifier Agent for extracting schema anchors from tables.

    This agent analyzes the table structure and content to generate
    a lightweight keyword dictionary that serves as a schema anchor
    for subsequent reasoning stages.
    """

    def __init__(self, llm=None):
        """
        Initialize the ClarifierAgent.

        Args:
            llm: Language model instance for advanced extraction (optional)
        """
        super().__init__(agent_type=AgentType.CLARIFIER, llm=llm)

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample using the clarify_sample method.

        Args:
            sample: Input sample to process
            context: Additional context (not used)

        Returns:
            Processed sample with clarification information
        """
        return self.clarify_sample(sample)

    def extract_headers(self, table_text: List[List[str]]) -> List[str]:
        """
        Extract column headers from the table.

        Args:
            table_text: Table data as list of lists (first row is headers)

        Returns:
            List of column headers
        """
        if not table_text or len(table_text) == 0:
            return []

        headers = table_text[0]
        return [str(h).strip() for h in headers if h]

    def extract_entities(self, table_text: List[List[str]], question: str = "") -> Dict[str, List[str]]:
        """
        Extract key entities from the table.

        Args:
            table_text: Table data as list of lists
            question: Optional question to guide entity extraction

        Returns:
            Dictionary mapping entity types to lists of entities
        """
        if not table_text or len(table_text) < 2:
            return {}

        headers = table_text[0]
        rows = table_text[1:]

        entities = defaultdict(list)

        # Extract entities from each column
        for col_idx, header in enumerate(headers):
            col_values = []
            for row in rows:
                if col_idx < len(row) and row[col_idx]:
                    value = str(row[col_idx]).strip()
                    if value and value != "" and value != "n/a":
                        col_values.append(value)

            # Categorize by column type
            header_lower = str(header).lower() if header else ""

            # Date columns
            if any(keyword in header_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
                entities['dates'].extend(col_values[:5])  # Limit to first 5

            # Name/entity columns
            elif any(keyword in header_lower for keyword in ['name', 'player', 'team', 'club', 'athlete', 'person']):
                entities['names'].extend(col_values[:10])

            # Location columns
            elif any(keyword in header_lower for keyword in ['location', 'place', 'city', 'country', 'venue']):
                entities['locations'].extend(col_values[:5])

            # Numeric value columns (for units)
            elif any(keyword in header_lower for keyword in ['score', 'points', 'value', 'amount', 'count', 'number']):
                entities['numeric_fields'].append(header)

        return dict(entities)

    def extract_units(self, table_text: List[List[str]]) -> Dict[str, str]:
        """
        Extract numerical units from the table.

        Args:
            table_text: Table data as list of lists

        Returns:
            Dictionary mapping column names to their units
        """
        if not table_text or len(table_text) < 2:
            return {}

        headers = table_text[0]
        rows = table_text[1:]
        units = {}

        # Common unit patterns
        unit_patterns = [
            (r'\d+\s*([kKmMgG]?[bB])', 'bytes'),  # KB, MB, GB, etc.
            (r'\d+\s*([kKmMtTgG]?[hH]z)', 'hz'),    # KHz, MHz, GHz, etc.
            (r'\d+\s*(%|percent)', 'percentage'),
            (r'\d+\s*\$', 'currency'),
            (r'\d+:\d+', 'time'),
            (r'\d+\.\d+', 'decimal'),
            (r'\d+\/\d+', 'fraction'),
        ]

        for col_idx, header in enumerate(headers):
            if not header:
                continue

            # Check rows for unit patterns
            for row in rows[:10]:  # Sample first 10 rows
                if col_idx < len(row) and row[col_idx]:
                    value = str(row[col_idx])

                    for pattern, unit_type in unit_patterns:
                        if re.search(pattern, value):
                            units[str(header)] = unit_type
                            break
                if str(header) in units:
                    break

        return units

    def extract_keywords(self, table_text: List[List[str]], question: str) -> Dict[str, Any]:
        """
        Extract keywords from question and table for schema anchoring.

        Args:
            table_text: Table data as list of lists
            question: User question about the table

        Returns:
            Dictionary containing schema anchor information
        """
        headers = self.extract_headers(table_text)
        entities = self.extract_entities(table_text, question)
        units = self.extract_units(table_text)

        # Extract keywords from question
        question_keywords = self._extract_question_keywords(question, headers)

        # Create keyword dictionary
        keyword_dict = {
            'headers': headers,
            'entities': entities,
            'units': units,
            'question_keywords': question_keywords,
            'column_mapping': self._create_column_mapping(table_text, question)
        }

        return keyword_dict

    def _extract_question_keywords(self, question: str, headers: List[str]) -> Dict[str, Any]:
        """
        Extract relevant keywords from the question.

        Args:
            question: User question
            headers: Table column headers

        Returns:
            Dictionary of question keywords
        """
        question_lower = question.lower()

        keywords = {
            'numbers': [],
            'comparisons': [],
            'operations': [],
            'matched_columns': []
        }

        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        keywords['numbers'] = numbers

        # Extract comparison words
        comparison_words = ['highest', 'lowest', 'most', 'least', 'greater', 'less', 'maximum', 'minimum', 'average', 'total']
        for word in comparison_words:
            if word in question_lower:
                keywords['comparisons'].append(word)

        # Extract operation words
        operation_words = ['how many', 'count', 'sum', 'average', 'mean', 'median', 'total', 'difference']
        for op in operation_words:
            if op in question_lower:
                keywords['operations'].append(op)

        # Match question terms to headers
        for header in headers:
            header_lower = str(header).lower()
            # Check if header or parts of it appear in question
            if header_lower in question_lower or any(word in question_lower for word in header_lower.split()):
                keywords['matched_columns'].append(header)

        return keywords

    def _create_column_mapping(self, table_text: List[List[str]], question: str) -> Dict[str, str]:
        """
        Create a mapping from question terms to table columns.

        Args:
            table_text: Table data
            question: User question

        Returns:
            Dictionary mapping question terms to column names
        """
        if not table_text or len(table_text) == 0:
            return {}

        headers = table_text[0]
        mapping = {}
        question_lower = question.lower()

        for header in headers:
            header_lower = str(header).lower()
            header_words = header_lower.split()

            # Direct match
            if header_lower in question_lower:
                mapping[header_lower] = str(header)

            # Partial word match
            for word in header_words:
                if len(word) > 3 and word in question_lower:
                    mapping[word] = str(header)
                    break

        return mapping

    def clarify_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clarify a single sample by extracting schema anchors.

        Args:
            sample: Sample dictionary containing 'table_text' and 'statement'

        Returns:
            Modified sample with added 'clarifier' field
        """
        table_text = sample.get('table_text', [])
        question = sample.get('statement', '')

        # Extract keywords
        keyword_dict = self.extract_keywords(table_text, question)

        # Create clarifier result
        clarifier_result = {
            'keyword_dict': keyword_dict,
            'headers': keyword_dict['headers'],
            'entities': keyword_dict['entities'],
            'units': keyword_dict['units'],
            'question_keywords': keyword_dict['question_keywords'],
            'column_mapping': keyword_dict['column_mapping']
        }

        # Add to sample
        sample['clarifier'] = clarifier_result

        return sample

    def clarify_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clarify a batch of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of samples with added clarifier information
        """
        clarified_samples = []
        for sample in samples:
            clarified_sample = self.clarify_sample(sample)
            clarified_samples.append(clarified_sample)

        return clarified_samples


def create_clarifier_result_path(base_dir: str, model_name: str, sample_id: str) -> str:
    """
    Create the file path for saving clarifier results.

    Args:
        base_dir: Base directory for results
        model_name: Name of the model
        sample_id: ID of the sample

    Returns:
        File path for saving clarifier results
    """
    import os
    clarifier_dir = os.path.join(base_dir, model_name, 'thought', 'clarifier')
    os.makedirs(clarifier_dir, exist_ok=True)
    return os.path.join(clarifier_dir, f'case_dict_{sample_id}.pkl')
