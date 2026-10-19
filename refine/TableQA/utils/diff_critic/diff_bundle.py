"""Diff Bundle: package trace, evidence, verification, and localization for Critic consumption."""

from dataclasses import dataclass
from typing import Dict, List, Optional

from .operation_trace import build_operation_trace, OperationTrace
from .evidence_pack import build_evidence_pack, StepEvidence
from .step_localizer import localize_suspicious_step, LocalizationResult
from refine.TableQA.utils.verifier import verify_chain, VerificationReport


@dataclass
class DiffBundle:
    """Packaged diagnosis context for a single sample."""

    trace: OperationTrace
    evidence_pack: List[StepEvidence]
    verification_report: VerificationReport
    localization: LocalizationResult

    def suspected_evidence(self) -> Optional[StepEvidence]:
        """Return evidence for the localized suspicious step, if any."""
        if self.localization.step_num is None:
            return None
        return next(
            (e for e in self.evidence_pack if e.step_num == self.localization.step_num),
            None,
        )

    def to_prompt_block(self) -> str:
        """Format the diff evidence as a Critic-consumable prompt block.

        Only outputs the suspected step evidence, not the full trace.
        """
        evidence = self.suspected_evidence()
        loc = self.localization

        if evidence is None:
            return (
                "[Operation Diff Evidence]\n"
                "No high-confidence suspicious step was found.\n"
                "[End Operation Diff Evidence]"
            )

        lines = [
            "[Operation Diff Evidence]",
            f"Suspicious step: {loc.step_num}",
            f"Localization confidence: {loc.confidence:.2f}",
            f"Localization reason: {loc.reason}",
            evidence.to_prompt_block(),
            "[End Operation Diff Evidence]",
        ]
        return "\n".join(lines)


class DiffBundleError(Exception):
    """Raised when diff bundle construction fails at any stage."""

    def __init__(self, stage: str, cause: Exception):
        self.stage = stage
        self.cause = cause
        super().__init__(f"DiffBundle failed at '{stage}': {cause}")


def build_diff_bundle(sample: Dict) -> DiffBundle:
    """One-stop function: build a complete DiffBundle for a sample.

    Internally chains: trace → verify → evidence → localize → bundle.
    Any failure raises DiffBundleError with a descriptive stage name.
    """
    # Stage 1: Operation trace
    try:
        trace = build_operation_trace(sample)
    except Exception as e:
        raise DiffBundleError("trace", e)

    # Stage 2: Verification
    try:
        verification_report = verify_chain(sample)
    except Exception as e:
        raise DiffBundleError("verification", e)

    # Stage 3: Evidence pack
    try:
        evidence_pack = build_evidence_pack(trace)
    except Exception as e:
        raise DiffBundleError("evidence", e)

    # Stage 4: Localization
    try:
        localization = localize_suspicious_step(evidence_pack, verification_report)
    except Exception as e:
        raise DiffBundleError("localization", e)

    return DiffBundle(
        trace=trace,
        evidence_pack=evidence_pack,
        verification_report=verification_report,
        localization=localization,
    )
