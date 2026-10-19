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
RefinerAgent: Executor agent that reconstructs reasoning paths

This agent takes corrective instructions from the Critic and
reconstructs the reasoning path to fix errors.
"""

import copy
import re
from typing import Dict, List, Any, Optional, Tuple
from thought.TableQA.utils.helper import table2string
from .multi_agent_framework import BaseAgent, AgentType, AgentMessage, MessageType


class RefinerAgent(BaseAgent):
    """
    Refiner Agent that reconstructs reasoning paths based on critic feedback.

    This agent acts as the executor, taking guidance and implementing
    corrected reasoning chains.
    """

    def __init__(self, llm):
        """
        Initialize the RefinerAgent.

        Args:
            llm: Language model instance
        """
        super().__init__(AgentType.REFINER, llm)
        self.refinement_history = []

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample and refine the reasoning path.

        Args:
            sample: Input sample with critic feedback
            context: Additional context

        Returns:
            Sample with refined reasoning
        """
        # Check if there's critic feedback to address
        critique = sample.get('critique', '')
        incorrect_step = sample.get('max_step', 0)

        if not critique or incorrect_step == 0:
            # No refinement needed
            sample['refiner_conclusion'] = sample.get('conclusion', '[Incorrect]')
            return sample

        # Determine the type of error
        error_type = self._classify_error(sample, incorrect_step)

        # Refine based on error type
        if error_type == 'chain_error':
            # Error in reasoning chain - reconstruct
            refined_sample = self._refine_reasoning_chain(sample, incorrect_step)
        elif error_type == 'query_error':
            # Error in final query - regenerate answer
            refined_sample = self._refine_final_query(sample)
        else:
            # Unknown error type - attempt general refinement
            refined_sample = self._general_refinement(sample)

        # Store refinement history
        self.refinement_history.append({
            'sample_id': sample.get('id', 'unknown'),
            'original_conclusion': sample.get('conclusion', 'N/A'),
            'refined_conclusion': refined_sample.get('refiner_conclusion', 'N/A'),
            'error_step': incorrect_step
        })

        return refined_sample

    def _classify_error(self, sample: Dict[str, Any], incorrect_step: int) -> str:
        """
        Classify the type of error.

        Args:
            sample: Sample to analyze
            incorrect_step: Step where error occurred

        Returns:
            Error type: 'chain_error', 'query_error', or 'unknown'
        """
        max_step = sample.get('max_step', 0)
        chain_length = len(sample.get('chain', []))

        if incorrect_step == max_step and incorrect_step == chain_length:
            return 'query_error'
        elif incorrect_step < max_step:
            return 'chain_error'
        else:
            return 'unknown'

    def _refine_reasoning_chain(
        self,
        sample: Dict[str, Any],
        incorrect_step: int
    ) -> Dict[str, Any]:
        """
        Refine the reasoning chain from the error step onwards.

        Args:
            sample: Sample with error
            incorrect_step: Step where error occurred

        Returns:
            Sample with refined chain
        """
        from refine.TableQA.utils.chain import (
            get_table_info,
            get_critic_table_info,
            dynamic_chain_exec_one_sample
        )

        # Get table state before the error
        try:
            pre_table_info, current_table_info = get_critic_table_info(
                sample, incorrect_step
            )
        except:
            # Fallback if get_critic_table_info fails
            pre_table_info = get_table_info(sample, skip_op=[], first_n_op=incorrect_step-1)
            current_table_info = pre_table_info

        # Build prompt for refinement
        prompt = self._build_refinement_prompt(sample, incorrect_step, pre_table_info)

        llm_options = self.llm.get_model_options(
            temperature=0.0,
            per_example_max_decode_steps=2048,
            per_example_top_p=1.0
        )

        # Generate refined chain
        try:
            refined_sample = dynamic_chain_exec_one_sample(
                sample,
                incorrect_step=incorrect_step,
                max_step=sample.get('max_step', 0),
                llm=self.llm,
                llm_options=llm_options,
                strategy="top"
            )

            # Add refiner conclusion
            refined_sample['refiner_conclusion'] = self._extract_conclusion_from_chain(refined_sample)

            # Send message to validator
            if self.state:
                message = AgentMessage(
                    sender=AgentType.REFINER,
                    receiver=AgentType.VALIDATOR,
                    message_type=MessageType.RESPONSE,
                    content={
                        'refined_chain': refined_sample.get('chain', []),
                        'error_step': incorrect_step,
                        'original_critique': sample.get('critique', '')
                    }
                )
                self.send_message(message)

            return refined_sample

        except Exception as e:
            print(f"agents-refiner_agent.py-_refine_reasoning_chain Error in refinement: {e}")
            sample['refiner_conclusion'] = '[Incorrect] Refinement failed'
            return sample

    def _refine_final_query(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine only the final query (answer generation).

        Args:
            sample: Sample with query error

        Returns:
            Sample with refined query
        """
        from refine.TableQA.utils.chain import get_table_info, simple_query_with_critic

        # Remove the final query operation
        wo_query_sample = copy.deepcopy(sample)
        wo_query_sample['chain'] = wo_query_sample['chain'][:-1]

        # Get table info
        table_info = get_table_info(wo_query_sample, skip_op=[], first_n_op=None)

        # Generate new query
        llm_options = self.llm.get_model_options(
            temperature=0,
            per_example_max_decode_steps=200,
            per_example_top_p=1.0
        )

        try:
            refined_sample = simple_query_with_critic(
                wo_query_sample,
                table_info,
                self.llm,
                llm_options=llm_options
            )

            refined_sample['refiner_conclusion'] = self._extract_conclusion_from_chain(refined_sample)

            return refined_sample

        except Exception as e:
            print(f"agents-refiner_agent.py-_refine_final_query Error in query refinement: {e}")
            sample['refiner_conclusion'] = '[Incorrect] Query refinement failed'
            return sample

    def _general_refinement(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt general refinement when error type is unclear.

        Args:
            sample: Sample to refine

        Returns:
            Sample with attempted refinement
        """
        # Use the critique to guide a general correction attempt
        critique = sample.get('critique', '')

        # Build a simple correction prompt
        prompt = f"""Based on the following critique, please correct the reasoning:

{critique}

Original Question: {sample.get('statement', '')}

Please provide a corrected answer."""

        try:
            response = self.llm.generate(
                prompt,
                options=self.llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=500,
                    per_example_top_p=1.0
                )
            )

            sample['refiner_conclusion'] = self._assess_refinement_quality(response)
            sample['refinement_response'] = response

            return sample

        except Exception as e:
            print(f"agents-refiner_agent.py-_general_refinement Error in general refinement: {e}")
            sample['refiner_conclusion'] = '[Incorrect] General refinement failed'
            return sample

    def _build_refinement_prompt(
        self,
        sample: Dict[str, Any],
        incorrect_step: int,
        table_info: Dict[str, Any]
    ) -> str:
        """
        Build prompt for chain refinement.

        Args:
            sample: Sample with error
            incorrect_step: Error step number
            table_info: Current table state

        Returns:
            Refinement prompt
        """
        prompt = "Now, we have produced part of the Function Chain, but gained a critique.\n\n"

        # Add chain before error
        prompt += f"Function Chain before error: "
        chain = sample.get('chain', [])

        # Get the chain up to the error
        for i, op in enumerate(chain[:incorrect_step-1]):
            op_name = op.get('operation_name', '')
            if op_name:
                prompt += f"f_{op_name}() -> "

        prompt += "\n\n"

        # Add current table state
        prompt += "Current sub-table:\n/*\n"
        prompt += table2string(table_info['table_text']) + "\n*/\n\n"

        # Add critique
        prompt += "Critique:\n"
        prompt += sample.get('critique', '') + "\n\n"

        # Add question
        prompt += f"Question: {sample.get('statement', '')}\n\n"

        prompt += "Based on the critique, please continue the Function Chain to correct the error."

        return prompt

    def _extract_conclusion_from_chain(self, sample: Dict[str, Any]) -> str:
        """
        Extract conclusion from refined chain.

        Args:
            sample: Sample with refined chain

        Returns:
            Conclusion string
        """
        # Try to get from various sources
        if 'final_answer' in sample:
            answer = sample['final_answer']
        elif 'chain' in sample and len(sample['chain']) > 0:
            last_op = sample['chain'][-1]
            if 'parameter_and_conf' in last_op and len(last_op['parameter_and_conf']) > 0:
                answer = last_op['parameter_and_conf'][0][0]
            else:
                answer = ""
        else:
            answer = ""

        # Simple heuristic: if answer exists and is not empty, mark as potentially correct
        if answer and answer.strip():
            return '[Correct]'  # Temporary, will be verified by validator
        else:
            return '[Incorrect]'

    def _assess_refinement_quality(self, response: str) -> str:
        """
        Assess the quality of a refinement response.

        Args:
            response: LLM response

        Returns:
            Assessment conclusion
        """
        # Simple heuristic based on response characteristics
        if len(response) > 50 and 'answer' in response.lower():
            return '[Correct]'
        else:
            return '[Incorrect]'

    def apply_corrections(
        self,
        sample: Dict[str, Any],
        corrections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply a list of corrections to the sample.

        Args:
            sample: Sample to correct
            corrections: List of correction instructions

        Returns:
            Corrected sample
        """
        for correction in corrections:
            correction_type = correction.get('type', 'unknown')

            if correction_type == 'reconstruct_chain':
                sample = self._refine_reasoning_chain(
                    sample,
                    correction.get('step', 1)
                )
            elif correction_type == 'regenerate_query':
                sample = self._refine_final_query(sample)
            elif correction_type == 'general':
                sample = self._general_refinement(sample)

        return sample

    def get_refinement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about refinements performed.

        Returns:
            Dictionary with refinement statistics
        """
        if not self.refinement_history:
            return {'total': 0}

        total = len(self.refinement_history)
        successful = sum(
            1 for r in self.refinement_history
            if '[Correct]' in r.get('refined_conclusion', '')
        )

        return {
            'total': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0
        }

    def reset_history(self) -> None:
        """Reset refinement history."""
        self.refinement_history = []
