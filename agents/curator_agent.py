4# Copyright 2024 Table-Critic contributors
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
CuratorAgent: Archive manager for memory evolution

This agent implements summarization and blueprint generation for
error patterns, facilitating memory evolution in the system.
"""

import json
import copy
from typing import Dict, List, Any, Optional
from thought.TableQA.utils.helper import table2string
from .multi_agent_framework import BaseAgent, AgentType


class CuratorAgent(BaseAgent):
    """
    Curator Agent that manages memory archives and generates blueprints.

    This agent extracts error patterns, generates summaries, and maintains
    the evolutionary memory structure.
    """

    def __init__(self, llm, memory_path: str = None):
        """
        Initialize the CuratorAgent.

        Args:
            llm: Language model instance
            memory_path: Path to the error tree memory file
        """
        super().__init__(AgentType.CURATOR, llm)
        self.memory_path = memory_path
        self.error_tree = self._load_error_tree()

    def _load_error_tree(self) -> Dict[str, Any]:
        """Load the error tree from memory."""
        try:
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load error tree: {e}")
            return {}

    def _save_error_tree(self) -> None:
        """Save the error tree to memory."""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.error_tree, f, indent=4)
        except Exception as e:
            print(f"Error saving error tree: {e}")

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample and update memory.

        Args:
            sample: Sample to process
            context: Additional context (may contain error_route)

        Returns:
            Sample with curation metadata
        """
        error_route = context.get('error_route', 'random') if context else 'random'

        # Generate blueprint from error
        blueprint = self._generate_blueprint(sample)

        # Update memory with new case
        self._add_case_to_memory(sample, error_route, blueprint)

        # Add curation metadata to sample
        sample['blueprint'] = blueprint
        sample['curator_metadata'] = {
            'error_route': error_route,
            'confidence_score': 1.0,  # Initial confidence
            'added_to_tree': True
        }

        return sample

    def _generate_blueprint(self, sample: Dict[str, Any]) -> str:
        """
        Generate a blueprint (error pattern summary) from a sample.

        Args:
            sample: Sample with error

        Returns:
            Blueprint string describing the error pattern
        """
        # Extract key information
        critique = sample.get('critique', '')
        conclusion = sample.get('conclusion', '')
        question = sample.get('statement', '')

        # Build prompt for blueprint generation
        prompt = f"""Based on the following error case, generate a concise blueprint (error pattern summary) in one sentence:

Question: {question}

Critique: {critique}

Conclusion: {conclusion}

The blueprint should describe the general error pattern that would help identify similar errors in the future.
Blueprint:"""

        try:
            response = self.llm.generate(
                prompt,
                options=self.llm.get_model_options(
                    temperature=0.3,
                    per_example_max_decode_steps=100,
                    per_example_top_p=1.0
                )
            )

            # Clean up the response
            blueprint = response.strip().strip('."')
            if not blueprint:
                blueprint = self._fallback_blueprint(sample)

            return blueprint

        except Exception as e:
            print(f"Error generating blueprint: {e}")
            return self._fallback_blueprint(sample)

    def _fallback_blueprint(self, sample: Dict[str, Any]) -> str:
        """Generate a fallback blueprint when LLM generation fails."""
        conclusion = sample.get('conclusion', '')
        if 'Step 1' in conclusion:
            return "agents-curator_agent.py-_fallback_blueprint Error in initial row or column selection"
        elif 'Step 2' in conclusion:
            return "agents-curator_agent.py-_fallback_blueprint Error in filtering or data processing"
        elif 'Step 3' in conclusion:
            return "agents-curator_agent.py-_fallback_blueprint Error in final calculation or query"
        else:
            return "General reasoning error"

    def _add_case_to_memory(
        self,
        sample: Dict[str, Any],
        error_route: str,
        blueprint: str
    ) -> None:
        """
        Add a case to the error tree memory.

        Args:
            sample: Sample to add
            error_route: Error classification route
            blueprint: Error pattern blueprint
        """
        # Create the case template (similar to original update_tree logic)
        case_template = self._create_case_template(sample)

        # Add to error tree
        if error_route != 'random':
            route_parts = error_route.split('->')
            current_level = self.error_tree

            for i, part in enumerate(route_parts):
                part = part.strip()
                if part in current_level:
                    if i == len(route_parts) - 1:
                        # Last part - this is where we add the case
                        if isinstance(current_level[part], list):
                            current_level[part].append(case_template)
                        elif isinstance(current_level[part], dict):
                            # Convert to list if needed
                            if 'cases' not in current_level[part]:
                                current_level[part]['cases'] = []
                            current_level[part]['cases'].append(case_template)
                        # Add/update blueprint
                        if isinstance(current_level[part], dict):
                            current_level[part]['blueprint'] = blueprint
                        break
                    else:
                        current_level = current_level[part]
                else:
                    # Create new node
                    current_level[part] = {'cases': [case_template], 'blueprint': blueprint}
                    break
        else:
            # Random route - add to root level or create new category
            self._handle_random_case(case_template, blueprint)

        # Save updated tree
        self._save_error_tree()

    def _create_case_template(self, sample: Dict[str, Any]) -> str:
        """Create a case template string for storing in memory."""
        from critic.TableQA.tools.get_info import get_table_log

        template = "Original Table:\n/*\n"
        template += table2string(sample['table_text']) + "\n*/\n\n"
        template += "Question: \n" + sample['statement'] + "\n\n"

        template += "Reasoning Steps:\n"

        table_log, thought_log = get_table_log(sample)

        step = 0
        action_list = []
        table_text = sample['table_text']
        for idx, table_info in enumerate(table_log[:-1]):
            if table_info["act_chain"]:
                table_action = table_info["act_chain"][-1]
                if "skip" not in table_action:
                    table_text = table_info["table_text"]
                    action_list.append(table_action)
                    template += f"Step{step+1}: {thought_log[idx]}\n"
                    template += f"So we use {table_action}.\n\n"
                    step += 1

        if len(action_list):
            template += f"Step{step+1}: After using "
            max_idx = len(action_list) - 1
            for idx, act in enumerate(action_list):
                template += act
                if idx < max_idx - 1:
                    template += ", "
                elif idx == max_idx - 1:
                    template += " and "
            template += ", we obtain the sub-table:\n/*\n"
            template += f"{table2string(table_text)}\n*/\n"

            if "group_sub_table" in table_info:
                group_column, group_info = table_info["group_sub_table"]
                template += "/*\n"
                template += f"Group the rows according to column: {group_column}.\n"
                group_headers = ["Group ID", group_column, "Count"]
                group_rows = []
                for i, (v, count) in enumerate(group_info):
                    if v.strip() == "":
                        v = "[Empty Cell]"
                    group_rows.append([f"Group {i+1}", v, str(count)])
                template += " | ".join(group_headers) + "\n"
                for row in group_rows:
                    template += " | ".join(row) + "\n"
                template += "*/\n"

        template += f"{thought_log[-1]}\n\n"
        template += "Prediction Answer: \n" + table_log[-1]["cotable_result"].lower() + "\n\n"
        template += "Critique:\n" + sample.get("critique", "") + "\n\n"
        template += "Conclusion:\n" + sample.get("conclusion", "")

        return template

    def _handle_random_case(self, case_template: str, blueprint: str) -> None:
        """Handle cases with random/unclassified error routes."""
        # Try to find the best existing category
        best_category = self._find_best_category(blueprint)

        if best_category:
            if isinstance(self.error_tree[best_category], list):
                self.error_tree[best_category].append(case_template)
            elif isinstance(self.error_tree[best_category], dict):
                if 'cases' not in self.error_tree[best_category]:
                    self.error_tree[best_category]['cases'] = []
                self.error_tree[best_category]['cases'].append(case_template)
        else:
            # Create a new category based on blueprint
            category_name = self._derive_category_name(blueprint)
            self.error_tree[category_name] = {
                'cases': [case_template],
                'blueprint': blueprint
            }

    def _find_best_category(self, blueprint: str) -> Optional[str]:
        """Find the best existing category for a blueprint."""
        # Simple keyword matching
        blueprint_lower = blueprint.lower()

        for category, value in self.error_tree.items():
            if isinstance(value, dict) and 'blueprint' in value:
                existing_blueprint = value['blueprint'].lower()
                # Check for keyword overlap
                blueprint_words = set(blueprint_lower.split())
                existing_words = set(existing_blueprint.split())
                overlap = blueprint_words & existing_words

                if len(overlap) >= 2:  # At least 2 words in common
                    return category

        return None

    def _derive_category_name(self, blueprint: str) -> str:
        """Derive a category name from a blueprint."""
        # Simple heuristic: take first few meaningful words
        words = blueprint.lower().split()
        meaningful_words = [w for w in words if len(w) > 3][:4]
        return "_".join(meaningful_words)

    def extract_error_patterns(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract error patterns from a batch of samples.

        Args:
            samples: List of samples

        Returns:
            List of error pattern dictionaries
        """
        patterns = []

        for sample in samples:
            if sample.get('conclusion', '').startswith('[Incorrect]'):
                blueprint = self._generate_blueprint(sample)
                critique = sample.get('critique', '')
                question = sample.get('statement', '')

                patterns.append({
                    'blueprint': blueprint,
                    'critique': critique,
                    'question': question,
                    'sample_id': sample.get('id', 'unknown')
                })

        return patterns

    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory state.

        Returns:
            Dictionary with memory statistics
        """
        def count_nodes(node):
            if isinstance(node, list):
                return len(node), 0
            elif isinstance(node, dict):
                total_cases = 0
                total_categories = 0
                for value in node.values():
                    if isinstance(value, list):
                        total_cases += len(value)
                    elif isinstance(value, dict):
                        cases, cats = count_nodes(value)
                        total_cases += cases
                        total_categories += cats + 1
                return total_cases, total_categories
            return 0, 0

        total_cases, total_categories = count_nodes(self.error_tree)

        return {
            'total_categories': total_categories,
            'total_cases': total_cases,
            'memory_path': self.memory_path
        }

    def prune_low_confidence_cases(self, threshold: float = 0.2) -> int:
        """
        Prune cases with low confidence scores (for active forgetting).

        Args:
            threshold: Confidence threshold below which to prune

        Returns:
            Number of cases pruned
        """
        pruned_count = 0

        def prune_recursive(node):
            nonlocal pruned_count
            if isinstance(node, dict):
                keys_to_delete = []
                for key, value in node.items():
                    if isinstance(value, dict):
                        if 'confidence_score' in value and value['confidence_score'] < threshold:
                            keys_to_delete.append(key)
                        else:
                            prune_recursive(value)
                    elif isinstance(value, list):
                        # Filter list based on confidence
                        filtered_list = [
                            item for item in value
                            if not isinstance(item, dict) or
                            item.get('confidence_score', 1.0) >= threshold
                        ]
                        pruned_count += len(value) - len(filtered_list)
                        node[key] = filtered_list

                for key in keys_to_delete:
                    del node[key]

        prune_recursive(self.error_tree)
        self._save_error_tree()

        return pruned_count
