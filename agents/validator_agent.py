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
ValidatorAgent: Auditor agent for strict fact-checking

This agent performs hard audits based on table facts provided by the
Clarifier, including numerical verification and cell location validation.
"""

import re
import copy
from typing import Dict, List, Any, Optional, Tuple
from thought.TableQA.utils.helper import table2string
from .multi_agent_framework import BaseAgent, AgentType, AgentMessage, MessageType


class ValidatorAgent(BaseAgent):
    """
    Validator Agent that performs strict fact-checking audits.

    This agent acts as an auditor, verifying answers against table facts
    with rigorous logic validation.
    """

    def __init__(self, llm):
        """
        Initialize the ValidatorAgent.

        Args:
            llm: Language model instance
        """
        super().__init__(AgentType.VALIDATOR, llm)
        self.validation_templates = self._initialize_validation_templates()

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample and perform validation audit.

        Args:
            sample: Input sample with answer to validate
            context: Additional context

        Returns:
            Sample with validation results
        """
        # Perform different types of validation
        numerical_result = self._validate_numerical_answers(sample)
        cell_result = self._validate_cell_locations(sample)
        logic_result = self._validate_logic(sample)

        # Combine results
        validation_summary = {
            'numerical_check': numerical_result,
            'cell_check': cell_result,
            'logic_check': logic_result,
            'overall_confidence': self._calculate_overall_confidence([
                numerical_result,
                cell_result,
                logic_result
            ])
        }

        # Determine final conclusion
        if validation_summary['overall_confidence'] >= 0.8:
            validator_conclusion = '[Correct]'
        elif validation_summary['overall_confidence'] >= 0.5:
            validator_conclusion = '[Partially Correct]'
        else:
            validator_conclusion = '[Incorrect]'

        sample['validator_feedback'] = validation_summary
        sample['validator_conclusion'] = validator_conclusion
        sample['validation_summary'] = validation_summary

        # Send message to judge
        if self.state:
            message = AgentMessage(
                sender=AgentType.VALIDATOR,
                receiver=AgentType.JUDGE,
                message_type=MessageType.FEEDBACK,
                content=validation_summary
            )
            self.send_message(message)

        return sample

    def _initialize_validation_templates(self) -> Dict[str, str]:
        """Initialize validation prompt templates."""
        return {
            'numerical': """You are a numerical auditor. Verify the following calculation:

Table Data:
{table}

Question: {question}
Answer: {answer}

Task:
1. Extract all numbers from the table relevant to the question
2. Perform the calculation step-by-step
3. Compare with the provided answer
4. Return format: "VALID: [correct/incorrect]" followed by explanation
""",

            'cell_location': """You are a cell location auditor. Verify the following answer:

Table Data:
{table}

Question: {question}
Answer: {answer}

Task:
1. Identify which cells contain the answer
2. Verify the answer matches the cell content
3. Return format: "VALID: [correct/incorrect]" followed by explanation
""",

            'logic': """You are a logic auditor. Verify the following reasoning:

Table Data:
{table}

Question: {question}
Answer: {answer}

Task:
1. Analyze the logical structure of the question
2. Verify the answer addresses the question logically
3. Check for common logical fallacies
4. Return format: "VALID: [correct/incorrect]" followed by explanation
"""
        }

    def _validate_numerical_answers(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform numerical validation on the answer.

        Args:
            sample: Sample to validate

        Returns:
            Dictionary with numerical validation results
        """
        # Extract answer
        answer = self._extract_answer(sample)
        question = sample.get('statement', '')

        # Check if question requires numerical answer
        if not self._is_numerical_question(question, answer):
            return {'check_performed': False, 'reason': 'Non-numerical question'}

        # Extract numbers from answer
        answer_numbers = self._extract_numbers(answer)
        if not answer_numbers:
            return {
                'check_performed': True,
                'valid': False,
                'reason': 'No numbers found in answer'
            }

        # Perform table-based verification
        table_verification = self._verify_numbers_in_table(sample, answer_numbers)

        return {
            'check_performed': True,
            'valid': table_verification['valid'],
            'answer_numbers': answer_numbers,
            'table_numbers': table_verification['table_numbers'],
            'reason': table_verification['reason']
        }

    def _validate_cell_locations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that answer corresponds to correct table cells.

        Args:
            sample: Sample to validate

        Returns:
            Dictionary with cell validation results
        """
        answer = self._extract_answer(sample)
        question = sample.get('statement', '')
        table_text = sample.get('table_text', [])

        if not table_text or len(table_text) < 2:
            return {
                'check_performed': False,
                'reason': 'Invalid table structure'
            }

        # Get clarifier info if available
        clarifier = sample.get('clarifier', {})
        headers = clarifier.get('headers', [])

        # Try to locate answer in table
        cell_locations = self._find_answer_in_table(answer, table_text, headers)

        return {
            'check_performed': True,
            'valid': len(cell_locations) > 0,
            'cell_locations': cell_locations,
            'answer_found_in_table': len(cell_locations) > 0
        }

    def _validate_logic(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform logical validation of the answer.

        Args:
            sample: Sample to validate

        Returns:
            Dictionary with logic validation results
        """
        answer = self._extract_answer(sample)
        question = sample.get('statement', '')

        # Use LLM for logic validation
        prompt = self.validation_templates['logic'].format(
            table=table2string(sample['table_text']),
            question=question,
            answer=answer
        )

        try:
            response = self.llm.generate(
                prompt,
                options=self.llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=300,
                    per_example_top_p=1.0
                )
            )

            # Parse response
            is_valid = 'VALID: correct' in response.upper()

            return {
                'check_performed': True,
                'valid': is_valid,
                'explanation': response
            }

        except Exception as e:
            return {
                'check_performed': False,
                'reason': f'LLM validation failed: {e}'
            }

    def _extract_answer(self, sample: Dict[str, Any]) -> str:
        """Extract the answer from a sample."""
        # Try different sources
        if 'final_answer' in sample:
            return str(sample['final_answer'])
        elif 'chain' in sample and len(sample['chain']) > 0:
            last_op = sample['chain'][-1]
            if 'parameter_and_conf' in last_op and len(last_op['parameter_and_conf']) > 0:
                return str(last_op['parameter_and_conf'][0][0])
        elif 'answer' in sample:
            return str(sample['answer'])
        return ""

    def _is_numerical_question(self, question: str, answer: str) -> bool:
        """Check if the question requires a numerical answer."""
        # Check for numerical question patterns
        numerical_patterns = [
            r'how many',
            r'how much',
            r'what is the (average|mean|sum|total|maximum|minimum)',
            r'count',
            r'calculate',
            r'\d+.*\+.*\d+',  # Math operations
        ]

        question_lower = question.lower()
        for pattern in numerical_patterns:
            if re.search(pattern, question_lower):
                return True

        # Check if answer contains numbers
        answer_numbers = self._extract_numbers(answer)
        return len(answer_numbers) > 0

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        # Match integers, decimals, fractions
        patterns = [
            r'\d+\.?\d*',  # Decimals
            r'\d+/\d+',    # Fractions
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    if '/' in match:
                        # Handle fraction
                        num, den = match.split('/')
                        numbers.append(float(num) / float(den))
                    else:
                        numbers.append(float(match))
                except:
                    pass

        return numbers

    def _verify_numbers_in_table(
        self,
        sample: Dict[str, Any],
        answer_numbers: List[float]
    ) -> Dict[str, Any]:
        """Verify numbers against table data."""
        table_text = sample.get('table_text', [])

        if not table_text or len(table_text) < 2:
            return {
                'valid': False,
                'table_numbers': [],
                'reason': 'No table data'
            }

        # Extract all numbers from table
        table_numbers = []
        for row in table_text[1:]:  # Skip header
            for cell in row:
                if cell:
                    cell_numbers = self._extract_numbers(str(cell))
                    table_numbers.extend(cell_numbers)

        # Check if answer numbers are present or derivable from table
        # For simple cases, check if answer numbers are in table
        found_all = all(
            any(abs(an - tn) < 0.01 for tn in table_numbers)
            for an in answer_numbers
        )

        return {
            'valid': found_all,
            'table_numbers': table_numbers[:10],  # Limit to first 10
            'reason': 'All numbers found in table' if found_all else 'Some numbers not in table'
        }

    def _find_answer_in_table(
        self,
        answer: str,
        table_text: List[List[str]],
        headers: List[str]
    ) -> List[Dict[str, Any]]:
        """Find the answer in table cells."""
        locations = []
        answer_lower = answer.lower().strip()

        for row_idx, row in enumerate(table_text[1:], start=1):  # Skip header row
            for col_idx, cell in enumerate(row):
                if cell:
                    cell_lower = str(cell).lower().strip()
                    # Check for exact match or partial match
                    if answer_lower in cell_lower or cell_lower in answer_lower:
                        header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx}"
                        locations.append({
                            'row': row_idx,
                            'column': col_idx,
                            'header': header,
                            'cell_value': cell
                        })

        return locations

    def _calculate_overall_confidence(
        self,
        validation_results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence from multiple validation results.

        Args:
            validation_results: List of validation result dictionaries

        Returns:
            Overall confidence score (0-1)
        """
        if not validation_results:
            return 0.5  # Neutral if no results

        scores = []
        for result in validation_results:
            if not result.get('check_performed', False):
                continue  # Skip unperformed checks

            if result.get('valid', False):
                scores.append(1.0)
            else:
                scores.append(0.0)

        if not scores:
            return 0.5  # Neutral if no valid scores

        return sum(scores) / len(scores)

    def perform_strict_audit(
        self,
        sample: Dict[str, Any],
        required_checks: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform a strict audit with required validation checks.

        Args:
            sample: Sample to audit
            required_checks: List of required check types (default: all)

        Returns:
            Dictionary with strict audit results
        """
        if required_checks is None:
            required_checks = ['numerical', 'cell_location', 'logic']

        audit_results = {
            'required_checks': required_checks,
            'performed_checks': [],
            'failed_checks': [],
            'overall_valid': True
        }

        for check_type in required_checks:
            if check_type == 'numerical':
                result = self._validate_numerical_answers(sample)
            elif check_type == 'cell_location':
                result = self._validate_cell_locations(sample)
            elif check_type == 'logic':
                result = self._validate_logic(sample)
            else:
                continue

            audit_results['performed_checks'].append(check_type)

            if result.get('check_performed', False) and not result.get('valid', True):
                audit_results['failed_checks'].append(check_type)
                audit_results['overall_valid'] = False

        return audit_results

    def get_validation_report(self, sample: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation report.

        Args:
            sample: Sample with validation results

        Returns:
            Formatted validation report
        """
        validation = sample.get('validation_summary', {})

        report = "=== Validation Report ===\n\n"

        # Numerical check
        numerical = validation.get('numerical_check', {})
        if numerical.get('check_performed'):
            status = "PASS" if numerical.get('valid') else "FAIL"
            report += f"Numerical Check: {status}\n"
            if not numerical.get('valid'):
                report += f"  Reason: {numerical.get('reason', 'N/A')}\n"

        # Cell location check
        cell = validation.get('cell_check', {})
        if cell.get('check_performed'):
            status = "PASS" if cell.get('valid') else "FAIL"
            report += f"Cell Location Check: {status}\n"
            if cell.get('cell_locations'):
                report += f"  Found in {len(cell['cell_locations'])} location(s)\n"

        # Logic check
        logic = validation.get('logic_check', {})
        if logic.get('check_performed'):
            status = "PASS" if logic.get('valid') else "FAIL"
            report += f"Logic Check: {status}\n"

        # Overall
        confidence = validation.get('overall_confidence', 0)
        report += f"\nOverall Confidence: {confidence:.2f}\n"
        report += f"Validator Conclusion: {sample.get('validator_conclusion', 'N/A')}\n"

        return report
