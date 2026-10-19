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
Validator Agent for Table Reasoning

This module provides validation functionality for table reasoning results,
performing strict logical and numerical checks.
"""

from thought.TableQA.utils.helper import table2string


def validate_sample(sample, clarifier_info, llm, llm_options):
    """
    Validate a sample's reasoning result.
    
    Args:
        sample: Sample to validate
        clarifier_info: Clarifier information with schema anchors
        llm: Language model instance
        llm_options: LLM generation options
    
    Returns:
        Dictionary with validation result
    """
    prompt = _generate_validator_prompt(sample, clarifier_info)
    response = llm.generate_plus_with_score(prompt, options=llm_options)
    return _parse_validator_response(response[0][0])


def _generate_validator_prompt(sample, clarifier_info):
    """
    Generate validator prompt.
    
    Args:
        sample: Sample to validate
        clarifier_info: Clarifier information with schema anchors
    
    Returns:
        Validator prompt string
    """
    prompt = f"""You are a table reasoning auditor. Please strictly audit the following reasoning:

Table:
{table2string(sample['table_text'])}

Key Information (from Clarifier):
- Headers: {clarifier_info.get('headers', [])}
- Key Entities: {clarifier_info.get('entities', [])}
- Numeric Units: {clarifier_info.get('units', {})}

Question:
{sample['statement']}

Reasoning Steps:
{_format_chain(sample.get('chain', []))}

Final Answer:
{sample.get('answer', '')}

Please check:
1. Are numerical calculations correct?
2. Are cell references accurate?
3. Is the logical reasoning rigorous?

Return audit result: [Valid] or [Invalid], and explain the reason."""
    
    return prompt


def _parse_validator_response(response):
    """
    Parse validator response.
    
    Args:
        response: LLM response string
    
    Returns:
        Dictionary with validation result
    """
    if '[Valid]' in response:
        return {'valid': True, 'reason': response}
    else:
        return {'valid': False, 'reason': response}


def _format_chain(chain):
    """
    Format reasoning chain for display.
    
    Args:
        chain: Reasoning chain
    
    Returns:
        Formatted chain string
    """
    result = []
    for idx, step in enumerate(chain):
        result.append(f"Step {idx+1}: {step.get('operation_name', '')}")
    return '\n'.join(result)
