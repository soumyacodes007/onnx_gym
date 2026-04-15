from __future__ import annotations

from dataclasses import dataclass


PHASE_ORDER = ("triage", "investigate", "repair", "verify", "finalize")
PHASE_INDEX = {name: idx for idx, name in enumerate(PHASE_ORDER)}


@dataclass(frozen=True)
class JudgeResult:
    score_delta: float
    feedback: str
    phase: str
    next_phase_index: int


def action_phase(action_type: str) -> str:
    if action_type in {"inspect_task", "inspect_bundle"}:
        return "triage"
    if action_type in {"inspect_patches", "inspect_report"}:
        return "investigate"
    if action_type == "apply_patch":
        return "repair"
    if action_type == "validate_bundle":
        return "verify"
    return "finalize"


def evaluate_step(action_type: str, phase_cursor: int, checks_run: int, history: list[str], persona: str) -> JudgeResult:
    phase = action_phase(action_type)
    phase_idx = PHASE_INDEX[phase]
    expected_idx = min(phase_cursor, len(PHASE_ORDER) - 1)

    strictness = {"junior": 0.8, "senior": 1.0, "principal": 1.2}.get(persona, 1.0)
    score_delta = 0.0
    feedback_parts: list[str] = []

    if phase_idx == expected_idx:
        score_delta += 0.03 * strictness
        feedback_parts.append(f"workflow phase '{phase}' executed in expected order")
        next_phase_index = min(expected_idx + 1, len(PHASE_ORDER) - 1)
    elif phase_idx < expected_idx:
        score_delta -= 0.01 * strictness
        feedback_parts.append("returned to an earlier phase")
        next_phase_index = expected_idx
    else:
        skipped = PHASE_ORDER[expected_idx:phase_idx]
        score_delta -= (0.02 + 0.01 * len(skipped)) * strictness
        feedback_parts.append("skipped phase(s): " + ", ".join(skipped))
        next_phase_index = expected_idx

    repeats = sum(1 for item in history if item == action_type)
    if repeats >= 2:
        score_delta -= 0.02 * strictness
        feedback_parts.append("repeated action pattern")

    if action_type == "submit_final" and checks_run == 0:
        score_delta -= 0.03 * strictness
        feedback_parts.append("submitted without validation")
    if action_type == "validate_bundle" and not any(a == "apply_patch" for a in history):
        score_delta -= 0.01 * strictness
        feedback_parts.append("validated before attempting repairs")

    feedback = "; ".join(feedback_parts) if feedback_parts else "neutral step"
    return JudgeResult(score_delta=round(score_delta, 4), feedback=feedback, phase=phase, next_phase_index=next_phase_index)
