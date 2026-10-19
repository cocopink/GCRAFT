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
Multi-Agent Framework for Game-Driven Table Reasoning

This module provides a framework for multiple agents to collaborate
on table reasoning tasks through structured communication protocols.
"""

import asyncio
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import copy


class AgentType(Enum):
    """Types of agents in the framework."""
    CLARIFIER = "clarifier"
    REASONER = "reasoner"
    JUDGE = "judge"
    CRITIC = "critic"
    REFINER = "refiner"
    VALIDATOR = "validator"
    CURATOR = "curator"
    RETRIEVER = "retriever"


class MessageType(Enum):
    """Types of messages between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    CRITIQUE = "critique"
    FEEDBACK = "feedback"
    DISPUTE = "dispute"
    RESOLUTION = "resolution"
    FINAL_JUDGMENT = "final_judgment"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    sender: AgentType
    receiver: AgentType
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float = 0
    message_id: str = ""

    def __post_init__(self):
        """Generate message ID and timestamp if not provided."""
        if not self.message_id:
            import time
            self.timestamp = time.time()
            self.message_id = f"{self.sender.value}_{self.receiver.value}_{int(self.timestamp)}"


@dataclass
class AgentState:
    """State of an agent during execution."""
    agent_type: AgentType
    current_sample: Dict[str, Any]
    iteration: int = 0
    history: List[AgentMessage] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)

    def add_to_history(self, message: AgentMessage):
        """Add a message to the agent's history."""
        self.history.append(message)

    def update_working_memory(self, key: str, value: Any):
        """Update the agent's working memory."""
        self.working_memory[key] = value


class BaseAgent:
    """
    Base class for all agents in the multi-agent framework.
    """

    def __init__(self, agent_type: AgentType, llm=None):
        """
        Initialize the base agent.

        Args:
            agent_type: Type of the agent
            llm: Language model instance (optional)
        """
        self.agent_type = agent_type
        self.llm = llm
        self.state = None

    def process(self, sample: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample. Must be implemented by subclasses.

        Args:
            sample: Input sample
            context: Additional context

        Returns:
            Processed sample
        """
        raise NotImplementedError("Subclasses must implement process()")

    def send_message(self, message: AgentMessage) -> AgentMessage:
        """
        Send a message to another agent.

        Args:
            message: Message to send

        Returns:
            Response message (if any)
        """
        if self.state:
            self.state.add_to_history(message)
        return message

    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent.

        Args:
            message: Received message
        """
        if self.state:
            self.state.add_to_history(message)
            # Update working memory based on message content
            if message.message_type == MessageType.CRITIQUE:
                self.state.update_working_memory('latest_critique', message.content)
            elif message.message_type == MessageType.FEEDBACK:
                self.state.update_working_memory('validator_feedback', message.content)


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents to collaborate on table reasoning tasks.

    This class manages the communication and coordination between agents,
    implementing the game-theoretic approach to iterative refinement.
    """

    def __init__(self, llm=None):
        """
        Initialize the orchestrator.

        Args:
            llm: Language model instance to share among agents
        """
        self.llm = llm
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.message_log: List[AgentMessage] = []
        self.max_iterations = 3  # Maximum rounds of dispute resolution

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
        """
        self.agents[agent.agent_type] = agent

    def process_sample(
        self,
        sample: Dict[str, Any],
        agent_sequence: List[AgentType] = None
    ) -> Tuple[Dict[str, Any], List[AgentMessage]]:
        """
        Process a sample through the multi-agent pipeline.

        Args:
            sample: Input sample
            agent_sequence: Ordered list of agents to process the sample

        Returns:
            Tuple of (processed_sample, message_log)
        """
        if agent_sequence is None:
            agent_sequence = [
                AgentType.CLARIFIER,
                AgentType.REASONER,
                AgentType.CRITIC,
                AgentType.REFINER,
                AgentType.VALIDATOR,
                AgentType.JUDGE
            ]

        current_sample = copy.deepcopy(sample)
        session_messages = []

        for agent_type in agent_sequence:
            if agent_type not in self.agents:
                continue

            agent = self.agents[agent_type]
            agent.state = AgentState(
                agent_type=agent_type,
                current_sample=current_sample
            )

            # Process the sample
            current_sample = agent.process(current_sample)

            # Collect messages from this agent's state
            if agent.state.history:
                session_messages.extend(agent.state.history)

        self.message_log.extend(session_messages)
        return current_sample, session_messages

    def run_dispute_resolution(
        self,
        sample: Dict[str, Any],
        max_rounds: int = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run the dispute resolution process (third-stage gameplay).

        Args:
            sample: Sample with initial disagreement
            max_rounds: Maximum rounds of dispute (default: self.max_iterations)

        Returns:
            Tuple of (final_sample, resolution_summary)
        """
        if max_rounds is None:
            max_rounds = self.max_iterations

        current_sample = copy.deepcopy(sample)
        resolution_summary = {
            'rounds': 0,
            'disputes': [],
            'final_resolutions': [],
            'converged': False
        }

        for round_num in range(1, max_rounds + 1):
            resolution_summary['rounds'] = round_num

            # Get critic assessment
            if AgentType.CRITIC in self.agents:
                critic = self.agents[AgentType.CRITIC]
                critic.state = AgentState(
                    agent_type=AgentType.CRITIC,
                    current_sample=current_sample,
                    iteration=round_num
                )
                current_sample = critic.process(current_sample)

            # Get refiner response
            if AgentType.REFINER in self.agents:
                refiner = self.agents[AgentType.REFINER]
                refiner.state = AgentState(
                    agent_type=AgentType.REFINER,
                    current_sample=current_sample,
                    iteration=round_num
                )
                current_sample = refiner.process(current_sample)

            # Get validator audit
            if AgentType.VALIDATOR in self.agents:
                validator = self.agents[AgentType.VALIDATOR]
                validator.state = AgentState(
                    agent_type=AgentType.VALIDATOR,
                    current_sample=current_sample,
                    iteration=round_num
                )
                current_sample = validator.process(current_sample)

            # Check for convergence (all agents agree)
            if self._check_convergence(current_sample):
                resolution_summary['converged'] = True
                resolution_summary['final_resolutions'].append({
                    'round': round_num,
                    'outcome': 'converged',
                    'conclusion': current_sample.get('refiner_conclusion', 'N/A')
                })
                break

            # Record the dispute
            dispute = self._record_dispute(current_sample, round_num)
            resolution_summary['disputes'].append(dispute)

            # If this is the last round, mark as unconverged
            if round_num == max_rounds:
                resolution_summary['final_resolutions'].append({
                    'round': round_num,
                    'outcome': 'unconverged',
                    'requires_arbitration': True
                })

        return current_sample, resolution_summary

    def _check_convergence(self, sample: Dict[str, Any]) -> bool:
        """
        Check if all agents have reached agreement.

        Args:
            sample: Sample to check

        Returns:
            True if converged, False otherwise
        """
        conclusions = []

        if 'critic_conclusion' in sample:
            conclusions.append(sample['critic_conclusion'])
        if 'refiner_conclusion' in sample:
            conclusions.append(sample['refiner_conclusion'])
        if 'validator_conclusion' in sample:
            conclusions.append(sample['validator_conclusion'])

        # All conclusions are the same
        return len(set(conclusions)) <= 1

    def _record_dispute(self, sample: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """
        Record the details of a dispute.

        Args:
            sample: Current sample state
            round_num: Current round number

        Returns:
            Dispute record
        """
        return {
            'round': round_num,
            'critic_opinion': sample.get('critic_conclusion', 'N/A'),
            'refiner_opinion': sample.get('refiner_conclusion', 'N/A'),
            'validator_opinion': sample.get('validator_conclusion', 'N/A'),
            'disagreement_type': self._identify_disagreement_type(sample)
        }

    def _identify_disagreement_type(self, sample: Dict[str, Any]) -> str:
        """
        Identify the type of disagreement between agents.

        Args:
            sample: Sample to analyze

        Returns:
            String describing the disagreement type
        """
        critic = sample.get('critic_conclusion', '')
        refiner = sample.get('refiner_conclusion', '')
        validator = sample.get('validator_conclusion', '')

        if critic != refiner and validator and validator != refiner:
            return "full_disagreement"
        elif critic != refiner:
            return "critic_refiner_disagreement"
        elif validator and validator != refiner:
            return "validator_refiner_disagreement"
        else:
            return "unknown"

    def final_arbitration(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform final arbitration when agents cannot agree.

        Args:
            sample: Sample with unresolved disputes

        Returns:
            Sample with final judgment
        """
        if AgentType.JUDGE in self.agents:
            judge = self.agents[AgentType.JUDGE]
            return judge.final_arbitration(
                sample,
                critic_conclusion=sample.get('critic_conclusion'),
                validator_conclusion=sample.get('validator_conclusion'),
                refiner_conclusion=sample.get('refiner_conclusion')
            )
        return sample

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the multi-agent system.

        Returns:
            Dictionary with system statistics
        """
        total_messages = len(self.message_log)
        message_types = {}
        for msg in self.message_log:
            msg_type = msg.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        return {
            'total_messages': total_messages,
            'registered_agents': list(self.agents.keys()),
            'message_types': message_types,
            'max_iterations': self.max_iterations
        }


def create_communication_protocol(
    sender: AgentType,
    receiver: AgentType,
    message_type: MessageType,
    content: Dict[str, Any]
) -> AgentMessage:
    """
    Create a standardized communication protocol message.

    Args:
        sender: Sending agent type
        receiver: Receiving agent type
        message_type: Type of message
        content: Message content

    Returns:
        Formatted agent message
    """
    return AgentMessage(
        sender=sender,
        receiver=receiver,
        message_type=message_type,
        content=content
    )


class GameTheoryController:
    """
    Controller for implementing game-theoretic strategies
    in the multi-agent interactions.
    """

    def __init__(self, orchestrator: MultiAgentOrchestrator):
        """
        Initialize the game theory controller.

        Args:
            orchestrator: Multi-agent orchestrator instance
        """
        self.orchestrator = orchestrator
        self.payoff_matrix = self._initialize_payoff_matrix()

    def _initialize_payoff_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize the payoff matrix for agent interactions.

        Returns:
            Dictionary representing payoff structure
        """
        return {
            'cooperate_correct': 1.0,
            'cooperate_incorrect': -0.5,
            'disagree_correct': 0.5,
            'disagree_incorrect': -1.0,
            'converge_bonus': 0.3
        }

    def calculate_payoff(self, outcome: Dict[str, Any]) -> float:
        """
        Calculate the payoff for a given outcome.

        Args:
            outcome: Outcome dictionary

        Returns:
            Calculated payoff value
        """
        if outcome.get('converged', False):
            base_payoff = self.payoff_matrix['cooperate_correct']
            bonus = self.payoff_matrix['converge_bonus']
            return base_payoff + bonus

        # Calculate based on final correctness
        conclusion = outcome.get('final_conclusion', '')
        if '[Correct]' in conclusion:
            return self.payoff_matrix['cooperate_correct']
        else:
            return self.payoff_matrix['cooperate_incorrect']
