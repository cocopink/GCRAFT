"""Step Localizer: fuse verifier results and evidence signals to find the most suspicious step."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .evidence_pack import StepEvidence
from refine.TableQA.utils.verifier import VerificationReport, Verdict


@dataclass
class LocalizationResult:
    """Result of step localization."""

    step_num: Optional[int]
    operation_name: str
    confidence: float
    reason: str
    source: str  # "verifier" | "evidence" | "combined"


# --- Scoring rules ---

_SCORE_VERIFIER_FAIL = 10
_SCORE_VERIFIER_WARN = 3
_SCORE_EVIDENCE_DROPPED_RELEVANT = 4
_SCORE_EVIDENCE_ANSWER_NOT_FOUND = 4
_SCORE_EVIDENCE_SORT = 3

# --- Confidence thresholds ---

_CONFIDENCE_HIGH = 0.9   # score >= 10
_CONFIDENCE_MED = 0.7    # score >= 5
_CONFIDENCE_LOW = 0.55   # score >= 3
_CONFIDENCE_NONE = 0.0   # score < 3


def _score_step(
    step_num: int,
    operation_name: str,
    evidence: Optional[StepEvidence],
    verification_report: Optional[VerificationReport],
) -> int:
    """Compute suspicion score for a single step."""
    score = 0

    # Verifier signals
    if verification_report and verification_report.steps:
        for sv in verification_report.steps:
            if sv.step_num == step_num:
                if sv.verdict == Verdict.FAIL:
                    score += _SCORE_VERIFIER_FAIL
                elif sv.verdict == Verdict.WARN:
                    score += _SCORE_VERIFIER_WARN
                break

    # Evidence signals
    if evidence:
        for warning in evidence.warnings:
            wl = warning.lower()
            if "dropped" in wl and "relevant" in wl:
                score += _SCORE_EVIDENCE_DROPPED_RELEVANT
            if "answer" in wl and "not found" in wl:
                score += _SCORE_EVIDENCE_ANSWER_NOT_FOUND
            if "sort" in wl:
                score += _SCORE_EVIDENCE_SORT

    return score


def _score_to_confidence(score: int) -> float:
    """Map raw score to confidence value."""
    if score >= 10:
        return _CONFIDENCE_HIGH
    if score >= 5:
        return _CONFIDENCE_MED
    if score >= 3:
        return _CONFIDENCE_LOW
    return _CONFIDENCE_NONE


def localize_suspicious_step(
    evidence_pack: List[StepEvidence],
    verification_report: Optional[VerificationReport] = None,
) -> LocalizationResult:
    """Find the most suspicious step by fusing verifier and evidence signals.

    Priority: earlier high-scoring steps are preferred since later errors
    may be propagation results from earlier mistakes.
    """
    if not evidence_pack:
        return LocalizationResult(None, "", 0.0, "No evidence available", "evidence")

    scored_steps = []
    for evidence in evidence_pack:
        score = _score_step(
            step_num=evidence.step_num,
            operation_name=evidence.operation_name,
            evidence=evidence,
            verification_report=verification_report,
        )
        scored_steps.append((score, evidence.step_num, evidence.operation_name))

    # Sort by score descending, then by step_num ascending (earlier first)
    scored_steps.sort(key=lambda x: (-x[0], x[1]))

    best_score, best_step, best_op = scored_steps[0]
    confidence = _score_to_confidence(best_score)

    if confidence == _CONFIDENCE_NONE:
        return LocalizationResult(
            step_num=None,
            operation_name=best_op,
            confidence=0.0,
            reason=f"No strong signal (best score: {best_score})",
            source="combined",
        )

    # Determine source
    source = "combined"
    if verification_report and verification_report.failed_steps:
        if best_step in verification_report.failed_steps:
            source = "verifier"
    elif best_score < _SCORE_VERIFIER_FAIL:
        source = "evidence"

    return LocalizationResult(
        step_num=best_step,
        operation_name=best_op,
        confidence=confidence,
        reason=f"Suspicious step {best_step} (score: {best_score}, op: {best_op})",
        source=source,
    )
