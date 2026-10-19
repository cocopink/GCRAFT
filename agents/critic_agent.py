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
CriticAgent: Mentor agent that provides corrective guidance

This agent retrieves blueprints from memory and provides corrective
instructions to the refiner. It implements a staged prompt strategy:
- Stage 1: Only introduce Blueprint
- Stage 2: Introduce original text as few-shot learning
"""

import json
import re
import random
from typing import Dict, List, Any, Optional
from thought.TableQA.utils.helper import table2string
from .multi_agent_framework import BaseAgent, AgentType, AgentMessage, MessageType


class CriticAgent(BaseAgent):
    """
    Critic Agent that provides corrective guidance based on memory blueprints.

    This agent acts as a mentor, analyzing errors and providing targeted
    feedback for improvement.
    """

    def __init__(self, llm, memory_path: str = None):
        """
        Initialize the CriticAgent.

        Args:
            llm: Language model instance
            memory_path: Path to the blueprint/error tree memory
        """
        super().__init__(AgentType.CRITIC, llm)
        self.memory_path = memory_path
        self.blueprint_cache = {}
        self.prompt_stage = 1  # Start with stage 1 prompts

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample and provide critique.

        Args:
            sample: Input sample
            context: Additional context (may contain error_route)

        Returns:
            Sample with critique added
        """
        # Determine error route if not provided
        error_route = context.get('error_route', 'random') if context else 'random'

        # Build and execute critique prompt
        prompt = self._build_critique_prompt(sample, error_route)

        llm_options = self.llm.get_model_options(
            temperature=0.0,
            per_example_max_decode_steps=500,
            per_example_top_p=1.0
        )

        response = self.llm.generate(prompt, options=llm_options)

        # Parse response
        critique, conclusion, max_step = self._parse_critique_response(response, sample)

        # Add to sample
        sample['critique'] = critique
        sample['critic_conclusion'] = conclusion
        sample['max_step'] = max_step

        # Store in history if state exists
        if self.state:
            message = AgentMessage(
                sender=AgentType.CRITIC,
                receiver=AgentType.REFINER,
                message_type=MessageType.CRITIQUE,
                content={
                    'critique': critique,
                    'conclusion': conclusion,
                    'max_step': max_step,
                    'error_route': error_route
                }
            )
            self.send_message(message)

        return sample

    def _build_critique_prompt(
        self,
        sample: Dict[str, Any],
        error_route: str = 'random',
        use_few_shot: bool = False
    ) -> str:
        """
        Build the critique prompt based on the current stage.

        Args:
            sample: Sample to critique
            error_route: Error route from tree classification
            use_few_shot: Whether to use few-shot examples (stage 2)

        Returns:
            Complete prompt string
        """
        prompt = self._get_base_instruction()

        # Stage-based content
        if self.prompt_stage == 1 or not use_few_shot:
            # Stage 1: Blueprint only
            blueprint = self._retrieve_blueprint(error_route)
            prompt += blueprint
        else:
            # Stage 2: Blueprint + few-shot examples
            blueprint = self._retrieve_blueprint(error_route)
            few_shot = self._retrieve_few_shot_examples(error_route)
            prompt += blueprint + "\n\n"
            prompt += few_shot

        # Add current sample context
        prompt += self._get_sample_context(sample)

        return prompt

    def _get_base_instruction(self) -> str:
        """Get the base critic instruction."""
        return """You are an intelligent critic tasked with determining which step of the table reasoning is incorrect based on the following information:

1. Original Table: The raw table data.
2. Question: The question pertaining to the table data.
3. Reasoning Steps: A step-by-step process of sub-table transformations and extractions based on the following functions.
    - f_add_column(): Adds a new column to the table.
    - f_select_row(): Selects specific rows based on the question.
    - f_select_column(): Removes irrelevant columns from the table.
    - f_group_column(): Groups rows based on the values in a specific column.
    - f_sort_column(): Sorts rows based on the values in a specified column.
4. Prediction Answer: Final derived answer following the reasoning chain.

Instruction:
1. **Step-wise Analysis**: Conduct an evaluation of each reasoning step's validity. The step that is unnecessary but does not affect the answer is considered correct.
2. **Analysis Categories:**
    - For correct steps: Provide validation reasoning and mark as `Step <NUM> is correct.`
    - For incorrect steps: Detail the logical flaws and mark as `Step <NUM> is incorrect.`
    - You should stop at the first incorrect step.
3. **Conclude this critique**: Summarize this critique with an explicit conclusion.
4. **Conclusion Categories**:
    - Conclude with 'Conclusion: [Incorrect] Step <NUM>'.

"""

    def _retrieve_blueprint(self, error_route: str) -> str:
        """
        Retrieve blueprint from memory based on error route.

        Args:
            error_route: Error classification route

        Returns:
            Blueprint prompt text
        """
        if error_route in self.blueprint_cache:
            return self.blueprint_cache[error_route]

        try:
            with open(self.memory_path, 'r') as f:
                error_tree = json.load(f)

            # Navigate the error tree to find relevant blueprint
            if error_route != 'random':
                route_parts = error_route.split('->')
                current_level = error_tree

                for part in route_parts:
                    part = part.strip()
                    if part in current_level:
                        current_level = current_level[part]
                        if isinstance(current_level, dict):
                            # Extract blueprint summary if available
                            if 'blueprint' in current_level:
                                blueprint = f"\n--- Error Pattern Blueprint ---\n"
                                blueprint += f"Error Route: {error_route}\n"
                                blueprint += f"Pattern Summary: {current_level['blueprint']}\n"
                                blueprint += "--- End Blueprint ---\n\n"
                                self.blueprint_cache[error_route] = blueprint
                                return blueprint

            # Default blueprint if not found
            blueprint = f"\n--- General Error Pattern ---\n"
            blueprint += f"Focus on identifying the first incorrect step in the reasoning chain.\n"
            blueprint += "--- End Pattern ---\n\n"
            self.blueprint_cache[error_route] = blueprint
            return blueprint

        except Exception as e:
            print(f"Warning: Could not load blueprint from {self.memory_path}: {e}")
            return ""

    def _retrieve_few_shot_examples(self, error_route: str, num_examples: int = 3) -> str:
        """
        Retrieve few-shot examples from memory.

        Args:
            error_route: Error classification route
            num_examples: Number of examples to retrieve

        Returns:
            Few-shot examples text
        """
        try:
            with open(self.memory_path, 'r') as f:
                error_tree = json.load(f)

            examples = []

            # Navigate tree to find relevant examples
            if error_route != 'random':
                route_parts = error_route.split('->')
                current_level = error_tree

                for part in route_parts:
                    part = part.strip()
                    if part in current_level:
                        current_level = current_level[part]
                        if isinstance(current_level, list):
                            # Found examples
                            examples = current_level[:num_examples]
                            break
                        elif isinstance(current_level, dict):
                            continue
            else:
                # Random selection from terminal nodes
                examples = self._get_random_terminal_examples(error_tree, num_examples)

            if examples:
                few_shot_text = "\nHere are some examples.\n\n"
                for idx, example in enumerate(examples):
                    # Handle both dict format (with blueprint/content) and string format
                    if isinstance(example, dict) and 'content' in example:
                        few_shot_text += f"Example {idx+1}:\n{example['content']}\n\n\n"
                    else:
                        few_shot_text += f"Example {idx+1}:\n{example}\n\n\n"
                return few_shot_text

            return ""

        except Exception as e:
            print(f"Warning: Could not load few-shot examples: {e}")
            return ""

    def _get_random_terminal_examples(self, tree: Dict, num_examples: int) -> List[str]:
        """Get random examples from terminal nodes."""
        examples = []

        def traverse(node):
            if isinstance(node, list):
                examples.extend(node)
            elif isinstance(node, dict):
                for value in node.values():
                    traverse(value)

        traverse(tree)

        if examples:
            return random.sample(examples, min(num_examples, len(examples)))
        return []

    def _get_sample_context(self, sample: Dict[str, Any]) -> str:
        """Get the context (table, question, reasoning) for a sample."""
        from critic.TableQA.tools.get_info import get_cot_for_critic
        cot, max_step = get_cot_for_critic(sample)
        return cot

    def _parse_critique_response(
        self,
        response: str,
        sample: Dict[str, Any]
    ) -> tuple[str, str, int]:
        """
        Parse the critique response from LLM.

        Args:
            response: Full LLM response
            sample: Original sample (for max_step fallback)

        Returns:
            Tuple of (critique, conclusion, max_step)
        """
        # Extract conclusion
        if "Conclusion:" in response:
            parts = response.split("Conclusion:")
            critique = parts[0].strip()
            conclusion = parts[1].strip() if len(parts) > 1 else ""
        else:
            critique = response.strip()
            conclusion = "[Incorrect] Unknown"

        # Extract step number from conclusion
        max_step = 1
        step_match = re.search(r'Step\s*(\d+)', conclusion)
        if step_match:
            max_step = int(step_match.group(1))
        elif 'max_step' in sample:
            max_step = sample['max_step']

        return critique, conclusion, max_step

    def switch_to_stage_2(self) -> None:
        """Switch to stage 2 prompting (with few-shot examples)."""
        self.prompt_stage = 2
        print("Critic Agent switched to Stage 2 (few-shot learning mode)")

    def switch_to_stage_1(self) -> None:
        """Switch to stage 1 prompting (blueprint only)."""
        self.prompt_stage = 1
        print("Critic Agent switched to Stage 1 (blueprint only mode)")

    def critique_with_blueprint(
        self,
        sample: Dict[str, Any],
        error_route: str
    ) -> Dict[str, Any]:
        """
        Provide critique using blueprint only (Stage 1).

        Args:
            sample: Sample to critique
            error_route: Error classification route

        Returns:
            Sample with critique
        """
        original_stage = self.prompt_stage
        self.switch_to_stage_1()

        result = self.process(sample, context={'error_route': error_route})

        self.prompt_stage = original_stage
        return result

    def critique_with_few_shot(
        self,
        sample: Dict[str, Any],
        error_route: str
    ) -> Dict[str, Any]:
        """
        Provide critique using blueprint and few-shot examples (Stage 2).

        Args:
            sample: Sample to critique
            error_route: Error classification route

        Returns:
            Sample with critique
        """
        original_stage = self.prompt_stage
        self.switch_to_stage_2()

        result = self.process(sample, context={'error_route': error_route})

        self.prompt_stage = original_stage
        return result

    def get_critique_summary(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of the critique for logging/analysis.

        Args:
            sample: Sample with critique

        Returns:
            Dictionary with critique summary
        """
        return {
            'conclusion': sample.get('critic_conclusion', 'N/A'),
            'incorrect_step': sample.get('max_step', 0),
            'has_critique': 'critique' in sample,
            'critique_length': len(sample.get('critique', ''))
        }
