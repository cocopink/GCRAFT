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
Active Forgetting Mechanism for Memory Evolution

This module implements the active forgetting mechanism that maintains
and prunes memory based on confidence scores.
"""

import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CaseRecord:
    """Record for a single case in memory."""
    case_id: str
    blueprint: str
    template: str
    confidence_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    successful_corrections: int = 0
    failed_corrections: int = 0

    def update_on_success(self) -> None:
        """Update record when the case successfully guides a correction."""
        self.confidence_score = min(1.0, self.confidence_score + 0.1)
        self.successful_corrections += 1
        self.last_accessed = datetime.now()
        self.access_count += 1

    def update_on_failure(self) -> None:
        """Update record when the case fails to guide a correction."""
        self.confidence_score = max(0.0, self.confidence_score - 0.2)
        self.failed_corrections += 1
        self.last_accessed = datetime.now()
        self.access_count += 1


class ActiveForgettingManager:
    """
    Manager for active forgetting mechanism.

    This class maintains confidence scores for cases and implements
    pruning strategies to make room for new patterns.
    """

    def __init__(
        self,
        memory_path: str = None,
        confidence_threshold: float = 0.3,
        decay_rate: float = 0.05,
        max_cases_per_category: int = 50
    ):
        """
        Initialize the ActiveForgettingManager.

        Args:
            memory_path: Path to the error tree memory file
            confidence_threshold: Threshold below which cases are pruned
            decay_rate: Rate at which confidence scores decay over time
            max_cases_per_category: Maximum cases to keep per category
        """
        self.memory_path = memory_path
        self.confidence_threshold = confidence_threshold
        self.decay_rate = decay_rate
        self.max_cases_per_category = max_cases_per_category
        self.case_records: Dict[str, CaseRecord] = {}

    def load_memory(self) -> Dict[str, Any]:
        """Load the error tree from memory."""
        try:
            with open(self.memory_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
            return {}

    def save_memory(self, memory: Dict[str, Any]) -> None:
        """Save the error tree to memory."""
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(memory, f, indent=4)
        except Exception as e:
            print(f"Error saving memory: {e}")

    def register_case(
        self,
        case_id: str,
        blueprint: str,
        template: str,
        initial_confidence: float = 1.0
    ) -> None:
        """
        Register a new case in memory.

        Args:
            case_id: Unique identifier for the case
            blueprint: Error pattern blueprint
            template: Full case template
            initial_confidence: Initial confidence score
        """
        self.case_records[case_id] = CaseRecord(
            case_id=case_id,
            blueprint=blueprint,
            template=template,
            confidence_score=initial_confidence
        )

    def update_on_correction_success(self, case_id: str) -> None:
        """
        Update confidence when a case successfully guides correction.

        Args:
            case_id: ID of the case that guided the correction
        """
        if case_id in self.case_records:
            self.case_records[case_id].update_on_success()
            print(f"Updated confidence for case {case_id}: {self.case_records[case_id].confidence_score}")

    def update_on_correction_failure(self, case_id: str) -> None:
        """
        Update confidence when a case fails to guide correction.

        Args:
            case_id: ID of the case that failed to guide
        """
        if case_id in self.case_records:
            self.case_records[case_id].update_on_failure()
            print(f"Decreased confidence for case {case_id}: {self.case_records[case_id].confidence_score}")

    def apply_temporal_decay(self, days_threshold: int = 30) -> None:
        """
        Apply temporal decay to confidence scores.

        Args:
            days_threshold: Number of days before applying decay
        """
        threshold_date = datetime.now() - timedelta(days=days_threshold)

        for case_id, record in self.case_records.items():
            if record.last_accessed < threshold_date:
                record.confidence_score = max(
                    0.0,
                    record.confidence_score * (1 - self.decay_rate)
                )

    def prune_low_confidence_cases(self, memory: Dict[str, Any] = None) -> int:
        """
        Prune cases with low confidence scores.

        Args:
            memory: Error tree memory (will load if not provided)

        Returns:
            Number of cases pruned
        """
        if memory is None:
            memory = self.load_memory()

        pruned_count = 0

        def prune_recursive(node):
            nonlocal pruned_count
            if isinstance(node, dict):
                keys_to_delete = []

                for key, value in node.items():
                    if key == 'cases' and isinstance(value, list):
                        # Filter cases based on confidence
                        filtered_cases = []
                        for i, case in enumerate(value):
                            case_id = f"{key}_{i}"

                            if case_id in self.case_records:
                                record = self.case_records[case_id]
                                if record.confidence_score >= self.confidence_threshold:
                                    filtered_cases.append(case)
                                else:
                                    pruned_count += 1
                                    del self.case_records[case_id]
                            else:
                                # No record, keep the case
                                filtered_cases.append(case)

                        node[key] = filtered_cases

                    elif isinstance(value, dict):
                        # Check for blueprint confidence
                        if 'confidence_score' in value:
                            if value['confidence_score'] < self.confidence_threshold:
                                keys_to_delete.append(key)
                            else:
                                prune_recursive(value)
                        else:
                            prune_recursive(value)
                    else:
                        prune_recursive(value)

                for key in keys_to_delete:
                    del node[key]

        prune_recursive(memory)
        self.save_memory(memory)

        return pruned_count

    def enforce_capacity_limits(self, memory: Dict[str, Any] = None) -> int:
        """
        Enforce maximum case limits per category.

        Args:
            memory: Error tree memory (will load if not provided)

        Returns:
            Number of cases removed
        """
        if memory is None:
            memory = self.load_memory()

        removed_count = 0

        def enforce_limit_recursive(node):
            nonlocal removed_count
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == 'cases' and isinstance(value, list):
                        if len(value) > self.max_cases_per_category:
                            # Sort by last accessed (recent = more important)
                            cases_with_meta = []
                            for i, case in enumerate(value):
                                case_id = f"{key}_{i}"
                                if case_id in self.case_records:
                                    record = self.case_records[case_id]
                                    cases_with_meta.append((case, record.last_accessed, record.confidence_score))
                                else:
                                    cases_with_meta.append((case, datetime.now(), 1.0))

                            # Sort by confidence (descending) then by last accessed
                            cases_with_meta.sort(key=lambda x: (x[2], x[1]), reverse=True)

                            # Keep only the top cases
                            node[key] = [case for case, _, _ in cases_with_meta[:self.max_cases_per_category]]
                            removed_count += len(value) - len(node[key])

                    elif isinstance(value, dict):
                        enforce_limit_recursive(value)

        enforce_limit_recursive(memory)
        self.save_memory(memory)

        return removed_count

    def get_case_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about case records.

        Returns:
            Dictionary with case statistics
        """
        if not self.case_records:
            return {'total': 0}

        total_cases = len(self.case_records)
        avg_confidence = sum(r.confidence_score for r in self.case_records.values()) / total_cases

        high_confidence = sum(1 for r in self.case_records.values() if r.confidence_score >= 0.8)
        medium_confidence = sum(1 for r in self.case_records.values() if 0.5 <= r.confidence_score < 0.8)
        low_confidence = sum(1 for r in self.case_records.values() if r.confidence_score < 0.5)

        total_corrections = sum(r.successful_corrections for r in self.case_records.values())
        total_failures = sum(r.failed_corrections for r in self.case_records.values())

        return {
            'total_cases': total_cases,
            'average_confidence': avg_confidence,
            'high_confidence_cases': high_confidence,
            'medium_confidence_cases': medium_confidence,
            'low_confidence_cases': low_confidence,
            'total_successful_corrections': total_corrections,
            'total_failed_corrections': total_failures,
            'success_rate': total_corrections / (total_corrections + total_failures) if (total_corrections + total_failures) > 0 else 0
        }

    def recommend_for_pruning(self, top_n: int = 10) -> List[str]:
        """
        Recommend cases for pruning based on low confidence.

        Args:
            top_n: Number of top cases to recommend

        Returns:
            List of case IDs recommended for pruning
        """
        # Sort by confidence score (ascending)
        sorted_cases = sorted(
            self.case_records.items(),
            key=lambda x: x[1].confidence_score
        )

        return [case_id for case_id, _ in sorted_cases[:top_n]]

    def boost_confidence_for_successful_patterns(self, min_corrections: int = 3) -> int:
        """
        Boost confidence for cases with proven success track record.

        Args:
            min_corrections: Minimum successful corrections to qualify

        Returns:
            Number of cases boosted
        """
        boosted_count = 0

        for case_id, record in self.case_records.items():
            if record.successful_corrections >= min_corrections:
                old_confidence = record.confidence_score
                record.confidence_score = min(1.0, record.confidence_score + 0.2)
                if record.confidence_score > old_confidence:
                    boosted_count += 1

        return boosted_count

    def run_maintenance_cycle(
        self,
        memory: Dict[str, Any] = None,
        apply_decay: bool = True,
        prune_low: bool = True,
        enforce_limits: bool = True
    ) -> Dict[str, int]:
        """
        Run a complete maintenance cycle on memory.

        Args:
            memory: Error tree memory
            apply_decay: Whether to apply temporal decay
            prune_low: Whether to prune low confidence cases
            enforce_limits: Whether to enforce capacity limits

        Returns:
            Dictionary with maintenance results
        """
        results = {}

        if apply_decay:
            self.apply_temporal_decay()
            results['decay_applied'] = 1

        if memory is None:
            memory = self.load_memory()

        if prune_low:
            pruned = self.prune_low_confidence_cases(memory)
            results['pruned_low_confidence'] = pruned

        if enforce_limits:
            removed = self.enforce_capacity_limits(memory)
            results['enforced_capacity_limits'] = removed

        return results
