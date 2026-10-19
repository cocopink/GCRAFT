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
InitialReasoner: Initial reasoning and judgment for table QA answers

This agent evaluates whether the predicted answer is correct or incorrect
based on the original table and question. It serves as the first stage
of the diagnosis and routing phase.
"""

import re
import json
import random
from typing import Dict, List, Any, Tuple
from thought.TableQA.utils.helper import table2string
from .multi_agent_framework import BaseAgent, AgentType


class InitialReasoner(BaseAgent):
    """
    Initial Reasoner for evaluating table QA answers.

    This agent determines whether a given prediction answer is correct
    or incorrect based on the table and question.
    """

    def __init__(self, llm):
        """
        Initialize the InitialReasoner.

        Args:
            llm: Language model instance for evaluation
        """
        super().__init__(agent_type=AgentType.REASONER, llm=llm)

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample using the judge_sample method.

        Args:
            sample: Input sample to process
            context: Additional context (not used)

        Returns:
            Processed sample with judge information
        """
        return self.judge_sample(sample)

    def build_judge_prompt(self, sample: Dict[str, Any], few_shot_examples: List[str] = None) -> str:
        """
        Build the prompt for judging an answer.

        Args:
            sample: Sample dictionary containing table, question, and answer
            few_shot_examples: Optional few-shot examples for better judgment

        Returns:
            Complete prompt for judgment
        """
        prompt = self._get_judge_instruction()

        if few_shot_examples:
            prompt += "\nHere are some examples.\n\n"
            for idx, example in enumerate(few_shot_examples):
                prompt += f"Example {idx+1}:\n{example}\n\n\n"

        prompt += self._get_sample_context(sample)
        prompt += "Explanation:"

        return prompt

    def _get_judge_instruction(self) -> str:
        """Get the judge instruction prompt."""
        return """You are an intelligent judge tasked with determining whether the given Prediction Answer is correct or incorrect based on the following information:

1. Original Table: The raw table data.
2. Question: The question pertaining to the table data.
3. Prediction Answer: The answer to the above question, which needs validation.

Instruction:
1. **Explanation**: Conduct an explanation of why the Prediction Answer is correct or incorrect.
2. **Conclusion**:
    - If the Prediction Answer is correct, conclude with 'Conclusion: [Correct]'.
    - If the Prediction Answer is incorrect, conclude with 'Conclusion: [Incorrect]'.

"""

    def _get_sample_context(self, sample: Dict[str, Any]) -> str:
        """Get the context (table, question, answer) for a sample."""
        cot = "Original Table:\n/*\n"
        cot += table2string(sample['table_text']) + "\n*/\n\n"
        cot += "Question: \n" + sample['statement'] + "\n\n"

        # Get the prediction answer from the chain
        table_log, _ = self._get_table_log(sample)
        cot += "Prediction Answer: \n" + table_log[-1]["cotable_result"].lower() + "\n\n"

        return cot

    def _get_table_log(self, sample: Dict[str, Any]) -> Tuple[List, List]:
        """Get table log and thought log from sample."""
        from critic.TableQA.tools.get_info import get_table_log
        return get_table_log(sample)

    def judge_sample(self, sample: Dict[str, Any], llm_options: Dict[str, Any] = None,
                     few_shot_examples: List[str] = None) -> Dict[str, Any]:
        """
        Judge a single sample.

        Args:
            sample: Sample dictionary
            llm_options: Options for LLM generation
            few_shot_examples: Optional few-shot examples

        Returns:
            Sample with added judge field containing conclusion
        """
        if llm_options is None:
            llm_options = self.llm.get_model_options(
                temperature=0.0,
                per_example_max_decode_steps=500,
                per_example_top_p=1.0
            )

        prompt = self.build_judge_prompt(sample, few_shot_examples)

        response = self.llm.generate(prompt, options=llm_options)

        # Extract conclusion from response
        conclusion = self._extract_conclusion(response)

        # Add judge information to sample
        sample['judge'] = conclusion
        sample['judge_response'] = response

        return sample

    def _extract_conclusion(self, response: str) -> str:
        """
        Extract the conclusion from the judge response.

        Args:
            response: Full response from LLM

        Returns:
            Extracted conclusion ([Correct] or [Incorrect])
        """
        # Look for Conclusion patterns
        patterns = [
            r'Conclusion:\s*\[([^\]]+)\]',
            r'conclusion:\s*\[([^\]]+)\]',
            r'\[([Cc]orrect)\]',
            r'\[([Ii]ncorrect)\]'
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                conclusion = match.group(1)
                # Normalize conclusion format
                if 'correct' in conclusion.lower():
                    return '[Correct]'
                elif 'incorrect' in conclusion.lower():
                    return '[Incorrect]'

        # Default to incorrect if no clear conclusion found
        return '[Incorrect]'

    def judge_batch(self, samples: List[Dict[str, Any]], llm_options: Dict[str, Any] = None,
                    few_shot_json: str = "critic/TableQA/tools/few_shot_judge.json") -> List[Dict[str, Any]]:
        """
        Judge a batch of samples.

        Args:
            samples: List of sample dictionaries
            llm_options: Options for LLM generation
            few_shot_json: Path to few-shot examples JSON file

        Returns:
            List of samples with judge information added
        """
        # Load few-shot examples
        few_shot_examples = self._load_few_shot_examples(few_shot_json)

        judged_samples = []
        for sample in samples:
            judged_sample = self.judge_sample(sample, llm_options, few_shot_examples)
            judged_samples.append(judged_sample)

        return judged_samples

    def _load_few_shot_examples(self, json_path: str) -> List[str]:
        """Load few-shot examples from JSON file."""
        try:
            with open(json_path, 'r') as f:
                few_shot_dict = json.load(f)

            selected_examples = []
            for value in few_shot_dict.values():
                if isinstance(value, list) and len(value) > 0:
                    random_index = random.randint(0, len(value) - 1)
                    selected_examples.append(value[random_index])

            random.shuffle(selected_examples)
            return selected_examples
        except Exception as e:
            print(f"Warning: Could not load few-shot examples from {json_path}: {e}")
            return []

    def filter_correct_incorrect(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples into correct and incorrect lists.

        Args:
            samples: List of judged samples

        Returns:
            Tuple of (correct_samples, incorrect_samples)
        """
        correct = []
        incorrect = []

        for sample in samples:
            judge = sample.get('judge', '').strip()
            if judge == '[Correct]':
                correct.append(sample)
            else:
                incorrect.append(sample)

        return correct, incorrect

    def calculate_accuracy(self, samples: List[Dict[str, Any]]) -> float:
        """
        Calculate accuracy of judged samples.

        Args:
            samples: List of judged samples

        Returns:
            Accuracy as a float between 0 and 1
        """
        if not samples:
            return 0.0

        correct_count = sum(1 for s in samples if s.get('judge', '').strip() == '[Correct]')
        return correct_count / len(samples)
