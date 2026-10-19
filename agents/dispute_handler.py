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
DisputeHandler: Three-round dispute resolution mechanism

This module implements the three-stage dispute resolution process:
- Round 1: Critic with Validator suggestions
- Round 2: Add few-shot learning examples
- Round 3: Trigger questioning, submit to Judge for final ruling
- Includes "uncorrectable" marking mechanism
"""

import copy
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from critic.TableQA.tools.update_tree import update_error_tree


class DisputeRound(Enum):
    """Enumeration of dispute resolution rounds."""
    FIRST = 1
    SECOND = 2
    THIRD = 3


class DisputeStatus(Enum):
    """Status of dispute resolution."""
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    UNCORRECTABLE = "uncorrectable"
    PENDING = "pending"


@dataclass
class DisputeRecord:
    """Record of a single dispute round."""
    round_number: DisputeRound
    critic_opinion: str
    refiner_response: str
    validator_suggestion: str = ""
    outcome: DisputeStatus = DisputeStatus.PENDING
    used_few_shot: bool = False
    notes: str = ""


@dataclass
class DisputeHistory:
    """Complete history of dispute resolution for a sample."""
    sample_id: str
    records: List[DisputeRecord] = field(default_factory=list)
    final_status: DisputeStatus = DisputeStatus.PENDING
    total_rounds: int = 0
    requires_arbitration: bool = False


class DisputeHandler:
    """
    Handler for the three-round dispute resolution process.

    This class manages the iterative dispute resolution between
    Critic, Refiner, and Validator agents.
    """

    def __init__(self, critic_agent, refiner_agent, validator_agent, judge_agent):
        """
        Initialize the DisputeHandler.

        Args:
            critic_agent: CriticAgent instance
            refiner_agent: RefinerAgent instance
            validator_agent: ValidatorAgent instance
            judge_agent: JudgeAgent instance
        """
        self.critic = critic_agent
        self.refiner = refiner_agent
        self.validator = validator_agent
        self.judge = judge_agent
        self.max_rounds = 3
        self.dispute_histories: Dict[str, DisputeHistory] = {}

    def resolve_dispute(
        self,
        sample: Dict[str, Any],
        error_route: str = 'random'
    ) -> Tuple[Dict[str, Any], DisputeHistory]:
        """
        Run the complete three-round dispute resolution process.

        Args:
            sample: Sample with initial error
            error_route: Error classification route

        Returns:
            Tuple of (resolved_sample, dispute_history)
        """
        sample_id = sample.get('id', 'unknown')
        history = DisputeHistory(sample_id=sample_id)
        current_sample = copy.deepcopy(sample)

        # Round 1: Critic with Validator suggestions
        print(f"Starting Round 1 dispute resolution for {sample_id}")
        current_sample, round1_record = self._round_1_dispute(current_sample, error_route)
        history.records.append(round1_record)

        if self._check_resolution(current_sample):
            history.final_status = DisputeStatus.RESOLVED
            history.total_rounds = 1
            self.dispute_histories[sample_id] = history
            return current_sample, history

        # Round 2: Add few-shot learning
        print(f"Starting Round 2 dispute resolution for {sample_id}")
        current_sample, round2_record = self._round_2_dispute(current_sample, error_route)
        history.records.append(round2_record)

        if self._check_resolution(current_sample):
            history.final_status = DisputeStatus.RESOLVED
            history.total_rounds = 2
            self.dispute_histories[sample_id] = history
            return current_sample, history

        # Round 3: Trigger questioning, submit to Judge
        print(f"Starting Round 3 dispute resolution for {sample_id}")
        current_sample, round3_record = self._round_3_dispute(current_sample)
        history.records.append(round3_record)

        # Final status
        if self._check_resolution(current_sample):
            history.final_status = DisputeStatus.RESOLVED
        elif self._is_uncorrectable(current_sample):
            history.final_status = DisputeStatus.UNCORRECTABLE
        else:
            history.final_status = DisputeStatus.UNRESOLVED
            history.requires_arbitration = True

        history.total_rounds = 3
        self.dispute_histories[sample_id] = history

        return current_sample, history

    def _round_1_dispute(
        self,
        sample: Dict[str, Any],
        error_route: str
    ) -> Tuple[Dict[str, Any], DisputeRecord]:
        """
        Round 1: Critic provides critique with Validator suggestions.

        Args:
            sample: Sample with error
            error_route: Error classification route

        Returns:
            Tuple of (updated_sample, dispute_record)
        """
        # Critic provides critique (Stage 1: Blueprint only)
        sample = self.critic.critique_with_blueprint(sample, error_route)
        critic_opinion = sample.get('critic_conclusion', '[Incorrect]')

        # Validator provides suggestions
        sample = self.validator.process(sample)
        validator_suggestion = sample.get('validator_conclusion', '[Incorrect]')

        # Refiner attempts correction
        sample = self.refiner.process(sample)
        refiner_response = sample.get('refiner_conclusion', '[Incorrect]')

        # Create record
        record = DisputeRecord(
            round_number=DisputeRound.FIRST,
            critic_opinion=critic_opinion,
            refiner_response=refiner_response,
            validator_suggestion=validator_suggestion,
            used_few_shot=False,
            notes="First round: Blueprint-based critique with validator suggestions"
        )

        # Check outcome
        if self._check_convergence(sample):
            record.outcome = DisputeStatus.RESOLVED
        else:
            record.outcome = DisputeStatus.UNRESOLVED

        return sample, record

    def _round_2_dispute(
        self,
        sample: Dict[str, Any],
        error_route: str
    ) -> Tuple[Dict[str, Any], DisputeRecord]:
        """
        Round 2: Add few-shot learning examples.

        Args:
            sample: Sample from Round 1
            error_route: Error classification route

        Returns:
            Tuple of (updated_sample, dispute_record)
        """
        # Critic provides critique with few-shot examples (Stage 2)
        sample = self.critic.critique_with_few_shot(sample, error_route)
        critic_opinion = sample.get('critic_conclusion', '[Incorrect]')

        # Validator re-checks
        sample = self.validator.process(sample)
        validator_suggestion = sample.get('validator_conclusion', '[Incorrect]')

        # Refiner attempts correction again
        sample = self.refiner.process(sample)
        refiner_response = sample.get('refiner_conclusion', '[Incorrect]')

        # Create record
        record = DisputeRecord(
            round_number=DisputeRound.SECOND,
            critic_opinion=critic_opinion,
            refiner_response=refiner_response,
            validator_suggestion=validator_suggestion,
            used_few_shot=True,
            notes="Second round: Critique with few-shot examples"
        )

        # Check outcome
        if self._check_convergence(sample):
            record.outcome = DisputeStatus.RESOLVED
        else:
            record.outcome = DisputeStatus.UNRESOLVED

        return sample, record

    def _round_3_dispute(
        self,
        sample: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], DisputeRecord]:
        """
        Round 3: Trigger questioning, submit to Judge for final ruling.

        Args:
            sample: Sample from Round 2

        Returns:
            Tuple of (updated_sample, dispute_record)
        """
        # Collect final opinions from all agents
        critic_opinion = sample.get('critic_conclusion', '[Incorrect]')
        refiner_response = sample.get('refiner_conclusion', '[Incorrect]')
        validator_suggestion = sample.get('validator_conclusion', '[Incorrect]')

        # Submit to Judge for final arbitration
        sample = self.judge.final_arbitration(
            sample,
            critic_conclusion=critic_opinion,
            validator_conclusion=validator_suggestion,
            refiner_conclusion=refiner_response
        )

        final_judgment = sample.get('judge', '[Incorrect]')

        # Create record
        record = DisputeRecord(
            round_number=DisputeRound.THIRD,
            critic_opinion=critic_opinion,
            refiner_response=refiner_response,
            validator_suggestion=validator_suggestion,
            used_few_shot=True,
            notes=f"Third round: Final arbitration by Judge. Final judgment: {final_judgment}"
        )

        # Check outcome
        if final_judgment == '[Correct]':
            record.outcome = DisputeStatus.RESOLVED
        else:
            # Determine if uncorrectable
            if self._is_uncorrectable(sample):
                record.outcome = DisputeStatus.UNCORRECTABLE
            else:
                record.outcome = DisputeStatus.UNRESOLVED

        return sample, record

    def _check_resolution(self, sample: Dict[str, Any]) -> bool:
        """
        Check if the dispute has been resolved (agents agree).

        Args:
            sample: Current sample state

        Returns:
            True if resolved, False otherwise
        """
        refiner_conclusion = sample.get('refiner_conclusion', '')
        validator_conclusion = sample.get('validator_conclusion', '')

        # Check if refiner and validator agree on correctness
        if refiner_conclusion == validator_conclusion == '[Correct]':
            return True

        # Also check judge if available
        judge_conclusion = sample.get('judge', '')
        if judge_conclusion == '[Correct]':
            return True

        return False

    def _check_convergence(self, sample: Dict[str, Any]) -> bool:
        """Check if all agents have converged to the same conclusion."""
        conclusions = []

        if 'refiner_conclusion' in sample:
            conclusions.append(sample['refiner_conclusion'])
        if 'validator_conclusion' in sample:
            conclusions.append(sample['validator_conclusion'])
        if 'critic_conclusion' in sample:
            conclusions.append(sample['critic_conclusion'])

        # All agree and it's correct
        return len(set(conclusions)) == 1 and '[Correct]' in conclusions

    def _is_uncorrectable(self, sample: Dict[str, Any]) -> bool:
        """
        Determine if a sample is uncorrectable.

        Args:
            sample: Sample to evaluate

        Returns:
            True if uncorrectable, False otherwise
        """
        # Check if all three rounds failed
        if 'dispute_history' not in sample:
            return False

        # Check validator confidence
        validation = sample.get('validation_summary', {})
        confidence = validation.get('overall_confidence', 1.0)

        # If confidence is very low after all rounds, mark as uncorrectable
        if confidence < 0.3:
            return True

        # Check if multiple validation checks failed
        failed_checks = 0
        if validation.get('numerical_check', {}).get('valid') is False:
            failed_checks += 1
        if validation.get('cell_check', {}).get('valid') is False:
            failed_checks += 1
        if validation.get('logic_check', {}).get('valid') is False:
            failed_checks += 1

        # If 2 or more checks failed, consider uncorrectable
        return failed_checks >= 2

    def mark_uncorrectable(
        self,
        sample: Dict[str, Any],
        reason: str = ""
    ) -> Dict[str, Any]:
        """
        Mark a sample as uncorrectable.

        Args:
            sample: Sample to mark
            reason: Reason for marking

        Returns:
            Sample with uncorrectable marker
        """
        sample['uncorrectable'] = True
        sample['uncorrectable_reason'] = reason
        sample['final_status'] = 'UNCORRECTABLE'

        return sample

    def get_dispute_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about dispute resolutions.

        Returns:
            Dictionary with dispute statistics
        """
        if not self.dispute_histories:
            return {'total': 0}

        total = len(self.dispute_histories)
        resolved = sum(1 for h in self.dispute_histories.values() if h.final_status == DisputeStatus.RESOLVED)
        uncorrectable = sum(1 for h in self.dispute_histories.values() if h.final_status == DisputeStatus.UNCORRECTABLE)
        unresolved = sum(1 for h in self.dispute_histories.values() if h.final_status == DisputeStatus.UNRESOLVED)

        # Round statistics
        round1_resolved = sum(1 for h in self.dispute_histories.values() if h.total_rounds == 1)
        round2_resolved = sum(1 for h in self.dispute_histories.values() if h.total_rounds == 2)
        round3_resolved = sum(1 for h in self.dispute_histories.values() if h.total_rounds == 3)

        return {
            'total_disputes': total,
            'resolved': resolved,
            'uncorrectable': uncorrectable,
            'unresolved': unresolved,
            'resolution_rate': resolved / total if total > 0 else 0,
            'round_1_resolved': round1_resolved,
            'round_2_resolved': round2_resolved,
            'round_3_resolved': round3_resolved,
            'avg_rounds': sum(h.total_rounds for h in self.dispute_histories.values()) / total if total > 0 else 0
        }

    def generate_dispute_report(self, sample_id: str) -> str:
        """
        Generate a human-readable dispute resolution report.

        Args:
            sample_id: ID of the sample

        Returns:
            Formatted dispute report
        """
        if sample_id not in self.dispute_histories:
            return f"No dispute history found for sample {sample_id}"

        history = self.dispute_histories[sample_id]

        report = f"=== Dispute Resolution Report: {sample_id} ===\n\n"
        report += f"Final Status: {history.final_status.value}\n"
        report += f"Total Rounds: {history.total_rounds}\n"
        report += f"Requires Arbitration: {history.requires_arbitration}\n\n"

        for i, record in enumerate(history.records, 1):
            report += f"--- Round {i} ({record.round_number.name}) ---\n"
            report += f"Critic Opinion: {record.critic_opinion}\n"
            report += f"Refiner Response: {record.refiner_response}\n"
            if record.validator_suggestion:
                report += f"Validator Suggestion: {record.validator_suggestion}\n"
            report += f"Used Few-Shot: {record.used_few_shot}\n"
            report += f"Outcome: {record.outcome.value}\n"
            report += f"Notes: {record.notes}\n\n"

        return report
