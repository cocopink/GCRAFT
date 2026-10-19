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
Memory Integration Module

This module integrates the CuratorAgent with the ActiveForgettingManager
to provide a complete memory evolution system.
"""

import json
from typing import Dict, Any
from .curator_agent import CuratorAgent
from .active_forgetting import ActiveForgettingManager, CaseRecord


class MemoryEvolutionManager:
    """
    Manager for memory evolution combining blueprint generation
    and active forgetting mechanisms.
    """

    def __init__(
        self,
        llm,
        memory_path: str = None,
        confidence_threshold: float = 0.3,
        max_cases_per_category: int = 50
    ):
        """
        Initialize the MemoryEvolutionManager.

        Args:
            llm: Language model instance
            memory_path: Path to the error tree memory
            confidence_threshold: Threshold for pruning cases
            max_cases_per_category: Maximum cases per category
        """
        self.llm = llm
        self.curator = CuratorAgent(llm=llm, memory_path=memory_path)
        self.forgetting_manager = ActiveForgettingManager(
            memory_path=memory_path,
            confidence_threshold=confidence_threshold,
            max_cases_per_category=max_cases_per_category
        )
        self.memory_path = memory_path

    def update_memory_with_case(
        self,
        sample: Dict[str, Any],
        error_route: str,
        correction_successful: bool = None
    ) -> Dict[str, Any]:
        """
        Update memory with a new case using CuratorAgent.

        Args:
            sample: Sample to add to memory
            error_route: Error classification route
            correction_successful: Whether the case guided a successful correction

        Returns:
            Dictionary with update results
        """
        # Process sample with curator to generate blueprint
        curated_sample = self.curator.process(sample, context={'error_route': error_route})

        blueprint = curated_sample.get('blueprint', '')
        case_id = f"{sample.get('id', 'unknown')}_{error_route}"

        # Register case with forgetting manager
        self.forgetting_manager.register_case(
            case_id=case_id,
            blueprint=blueprint,
            template="",  # Template is already added by curator
            initial_confidence=1.0
        )

        # Update confidence based on correction result
        if correction_successful is not None:
            if correction_successful:
                self.forgetting_manager.update_on_correction_success(case_id)
            else:
                self.forgetting_manager.update_on_correction_failure(case_id)

        return {
            'case_id': case_id,
            'blueprint': blueprint,
            'confidence': self.forgetting_manager.case_records[case_id].confidence_score
        }

    def update_memory_from_dispute(
        self,
        sample: Dict[str, Any],
        dispute_history: Any
    ) -> Dict[str, Any]:
        """
        Update memory based on dispute resolution history.

        Args:
            sample: Sample that went through dispute resolution
            dispute_history: Dispute resolution history

        Returns:
            Dictionary with update results
        """
        # Determine if the dispute was resolved successfully
        final_status = dispute_history.final_status if hasattr(dispute_history, 'final_status') else dispute_history.get('final_status', 'unresolved')
        correction_successful = (final_status.value == 'resolved')

        # Get error route from sample or history
        error_route = sample.get('error_route', 'random')

        return self.update_memory_with_case(
            sample,
            error_route,
            correction_successful
        )

    def update_confidence_scores(
        self,
        case_outcomes: Dict[str, bool]
    ) -> None:
        """
        Update confidence scores based on case outcomes.

        Args:
            case_outcomes: Dictionary mapping case_ids to success (True/False)
        """
        for case_id, success in case_outcomes.items():
            if success:
                self.forgetting_manager.update_on_correction_success(case_id)
            else:
                self.forgetting_manager.update_on_correction_failure(case_id)

    def run_maintenance(
        self,
        apply_decay: bool = True,
        prune_low: bool = True,
        enforce_limits: bool = True
    ) -> Dict[str, Any]:
        """
        Run maintenance cycle on memory.

        Args:
            apply_decay: Apply temporal decay
            prune_low: Prune low confidence cases
            enforce_limits: Enforce capacity limits

        Returns:
            Dictionary with maintenance results
        """
        # Load current memory
        memory = self.forgetting_manager.load_memory()

        # Run maintenance
        results = self.forgetting_manager.run_maintenance_cycle(
            memory=memory,
            apply_decay=apply_decay,
            prune_low=prune_low,
            enforce_limits=enforce_limits
        )

        # Add statistics
        results['statistics'] = self.forgetting_manager.get_case_statistics()

        return results

    def get_memory_report(self) -> str:
        """
        Generate a comprehensive memory report.

        Returns:
            Formatted memory report
        """
        stats = self.forgetting_manager.get_case_statistics()
        curator_stats = self.curator.get_memory_statistics()

        report = "=== Memory Evolution Report ===\n\n"

        report += "Case Records:\n"
        report += f"  Total Cases: {stats['total_cases']}\n"
        report += f"  Average Confidence: {stats['average_confidence']:.2f}\n"
        report += f"  High Confidence (>0.8): {stats['high_confidence_cases']}\n"
        report += f"  Medium Confidence (0.5-0.8): {stats['medium_confidence_cases']}\n"
        report += f"  Low Confidence (<0.5): {stats['low_confidence_cases']}\n\n"

        report += "Performance:\n"
        report += f"  Successful Corrections: {stats['total_successful_corrections']}\n"
        report += f"  Failed Corrections: {stats['total_failed_corrections']}\n"
        report += f"  Success Rate: {stats['success_rate']:.2%}\n\n"

        report += "Error Tree Structure:\n"
        report += f"  Total Categories: {curator_stats['total_categories']}\n"
        report += f"  Total Cases in Tree: {curator_stats['total_cases']}\n\n"

        # Recommend cases for pruning
        to_prune = self.forgetting_manager.recommend_for_pruning(5)
        if to_prune:
            report += "Cases Recommended for Pruning:\n"
            for case_id in to_prune:
                record = self.forgetting_manager.case_records.get(case_id)
                if record:
                    report += f"  - {case_id}: {record.confidence_score:.2f} confidence\n"
            report += "\n"

        return report

    def export_memory_state(self, export_path: str) -> None:
        """
        Export current memory state to file.

        Args:
            export_path: Path to export memory state
        """
        state = {
            'case_records': {
                case_id: {
                    'blueprint': record.blueprint,
                    'confidence_score': record.confidence_score,
                    'created_at': record.created_at.isoformat(),
                    'last_accessed': record.last_accessed.isoformat(),
                    'access_count': record.access_count,
                    'successful_corrections': record.successful_corrections,
                    'failed_corrections': record.failed_corrections
                }
                for case_id, record in self.forgetting_manager.case_records.items()
            },
            'error_tree': self.forgetting_manager.load_memory()
        }

        with open(export_path, 'w') as f:
            json.dump(state, f, indent=2)

    def import_memory_state(self, import_path: str) -> None:
        """
        Import memory state from file.

        Args:
            import_path: Path to import memory state from
        """
        with open(import_path, 'r') as f:
            state = json.load(f)

        # Restore case records
        for case_id, record_data in state.get('case_records', {}).items():
            record = CaseRecord(
                case_id=case_id,
                blueprint=record_data['blueprint'],
                template="",  # Not stored in export
                confidence_score=record_data['confidence_score']
            )
            record.created_at = datetime.fromisoformat(record_data['created_at'])
            record.last_accessed = datetime.fromisoformat(record_data['last_accessed'])
            record.access_count = record_data['access_count']
            record.successful_corrections = record_data['successful_corrections']
            record.failed_corrections = record_data['failed_corrections']

            self.forgetting_manager.case_records[case_id] = record

        # Restore error tree
        if 'error_tree' in state:
            self.forgetting_manager.save_memory(state['error_tree'])
