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
Active Forgetting Mechanism for Error Tree

This module provides active forgetting functionality to maintain
the error tree by pruning low-confidence cases.
"""

import json
import os


class ActiveForgetting:
    """
    Active forgetting mechanism: maintains and updates confidence scores for cases.
    
    This class implements a mechanism to:
    - Track confidence scores for each case
    - Increase weight when a case successfully guides a fix
    - Prune low-weight cases to make room for new errors
    """
    
    def __init__(self, min_confidence=0.0000001, max_cases=1000000):
        """
        Initialize active forgetting mechanism.
        
        Args:
            min_confidence: Minimum confidence threshold (default: 0.0000001)
                           Set extremely low to almost never execute forgetting
            max_cases: Maximum number of cases to keep (default: 1000000)
        """
        self.min_confidence = min_confidence
        self.max_cases = max_cases
        self.confidence_scores = {}
    
    def update_confidence(self, case_id, success):
        """
        Update confidence score for a case.
        
        Args:
            case_id: Unique identifier for the case
            success: Whether the case successfully guided a fix
        """
        if case_id not in self.confidence_scores:
            self.confidence_scores[case_id] = 1.0
        
        if success:
            # Successfully guided a fix, increase weight
            self.confidence_scores[case_id] += 1
        else:
            # Failed to guide a fix, decrease weight
            self.confidence_scores[case_id] *= 0.9
        
        # Ensure weight is within reasonable range
        self.confidence_scores[case_id] = max(
            self.min_confidence,
            min(10.0, self.confidence_scores[case_id])
        )
    
    def prune_low_confidence_cases(self, error_tree):
        """
        Prune low-confidence cases from error tree.
        
        Args:
            error_tree: Error tree dictionary to prune
        """
        def _prune_recursive(node):
            if isinstance(node, dict):
                for key, value in list(node.items()):
                    if isinstance(value, list):
                        # Filter low-confidence cases
                        node[key] = [
                            case for case in value
                            if case.get('confidence_score', 1.0) >= self.min_confidence
                        ]
                    else:
                        _prune_recursive(value)
            elif isinstance(node, list):
                # Filter low-confidence cases
                return [
                    case for case in node
                    if case.get('confidence_score', 1.0) >= self.min_confidence
                ]
        
        _prune_recursive(error_tree)
        
        # If case count exceeds maximum, prune lowest-weight cases
        total_cases = self._count_cases(error_tree)
        if total_cases > self.max_cases:
            self._prune_excess_cases(error_tree)
    
    def _count_cases(self, node):
        """
        Count total number of cases in the tree.
        
        Args:
            node: Tree node to count
        
        Returns:
            Total number of cases
        """
        if isinstance(node, dict):
            return sum(self._count_cases(v) for v in node.values())
        elif isinstance(node, list):
            return len(node)
        return 0
    
    def _prune_excess_cases(self, error_tree):
        """
        Prune excess cases when total exceeds maximum.
        
        Args:
            error_tree: Error tree dictionary to prune
        """
        # Collect all cases with their paths
        cases_with_paths = []
        
        def _collect_cases(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    _collect_cases(value, path + [key])
            elif isinstance(node, list):
                for idx, case in enumerate(node):
                    cases_with_paths.append({
                        'case': case,
                        'path': path + [idx],
                        'confidence': case.get('confidence_score', 1.0)
                    })
        
        _collect_cases(error_tree, [])
        
        # Sort by confidence
        cases_with_paths.sort(key=lambda x: x['confidence'])
        
        # Prune lowest-confidence cases
        num_to_remove = len(cases_with_paths) - self.max_cases
        for i in range(num_to_remove):
            case_info = cases_with_paths[i]
            path = case_info['path']
            
            # Delete this case from tree
            current = error_tree
            for key in path[:-1]:
                current = current[key]
            del current[path[-1]]
    
    def get_statistics(self):
        """
        Get statistics about confidence scores.
        
        Returns:
            Dictionary with statistics
        """
        if not self.confidence_scores:
            return {
                'total_cases': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0
            }
        
        scores = list(self.confidence_scores.values())
        return {
            'total_cases': len(self.confidence_scores),
            'avg_confidence': sum(scores) / len(scores),
            'min_confidence': min(scores),
            'max_confidence': max(scores)
        }
