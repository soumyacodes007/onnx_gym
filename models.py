from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


ActionType = Literal[
    "inspect_task",
    "inspect_bundle",
    "inspect_patches",
    "inspect_report",
    "apply_patch",
    "validate_bundle",
    "submit_final",
]


MIN_SCORE = 0.0


class OnnxAction(Action):
    action_type: ActionType = Field(..., description="Structured action type")
    slot_name: str = Field(default="", description="Optional bundle section to target")
    patch_id: str = Field(default="", description="Patch identifier for apply_patch")
    rationale: str = Field(default="", description="Brief optional reasoning note")


class OnnxObservation(Observation):
    task_id: str = Field(default="")
    task_title: str = Field(default="")
    difficulty: str = Field(default="easy")
    task_description: str = Field(default="")
    product_brief: str = Field(default="")
    deployment_profile: str = Field(default="")
    variant_label: str = Field(default="")
    current_bundle: dict[str, Any] = Field(default_factory=dict)
    bundle_preview: str = Field(default="")
    graph_preview: str = Field(default="")
    profile_summary: dict[str, Any] = Field(default_factory=dict)
    available_slots: list[str] = Field(default_factory=list)
    slot_status: dict[str, Any] = Field(default_factory=dict)
    patch_catalog: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    patch_dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    requirement_status: dict[str, Any] = Field(default_factory=dict)
    validation_report: dict[str, Any] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    visible_issues: list[dict[str, Any]] = Field(default_factory=list)
    hidden_issues: int = Field(default=0)
    cascade_depth: int = Field(default=0)
    cascade_depth_remaining: int = Field(default=0)
    flag_conflict_map: dict[str, Any] = Field(default_factory=dict)
    target_ep: str = Field(default="CPUExecutionProvider")
    memory_budget_mb: float = Field(default=0.0)
    estimated_model_mb: float = Field(default=0.0)
    estimated_activation_mb: float = Field(default=0.0)
    endpoint_coverage: float = Field(default=0.0)
    severity_weighted_score: float = Field(default=0.0)
    checker_passed: bool = Field(default=False)
    shape_inference_passed: bool = Field(default=False)
    ort_session_passed: bool = Field(default=False)
    inferred_value_info_count: int = Field(default=0)
    dynamic_dim_count: int = Field(default=0)
    node_count: int = Field(default=0)
    initializer_count: int = Field(default=0)
    unsupported_ops: list[str] = Field(default_factory=list)
    error_log: list[str] = Field(default_factory=list)
    top_blockers: list[str] = Field(default_factory=list)
    why_not_perfect: str = Field(default="")
    fix_history: list[dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    repair_summary: str = Field(default="")
    recommended_next_action: str = Field(default="")
    checks_run: int = Field(default=0)
    steps_taken: int = Field(default=0)
    max_steps: int = Field(default=0)
    current_score: float = Field(default=MIN_SCORE)
    best_score: float = Field(default=MIN_SCORE)
    success_threshold: float = Field(default=0.95)
    is_success: bool = Field(default=False)
    message: str = Field(default="")
    last_action_error: str = Field(default="")
    final_score: float = Field(default=MIN_SCORE)
    possible_actions: list[str] = Field(default_factory=list)
    last_report: dict[str, Any] = Field(default_factory=dict)
    curriculum_stats: dict[str, Any] = Field(default_factory=dict)
    workflow_phase: str = Field(default="triage")
    workflow_feedback: str = Field(default="")
    judge_persona: str = Field(default="junior")
    episode_mode: str = Field(default="standard")
    incident_brief: str = Field(default="")
    incident_id: str = Field(default="")
    adversarial_seed: int = Field(default=0)
    fault_bundle: list[str] = Field(default_factory=list)


class OnnxState(State):
    task_id: str = Field(default="")
    task_title: str = Field(default="")
    difficulty: str = Field(default="easy")
    deployment_profile: str = Field(default="")
    variant_label: str = Field(default="")
    current_bundle: dict[str, Any] = Field(default_factory=dict)
    selected_patches: dict[str, str] = Field(default_factory=dict)
    patch_history: list[dict[str, Any]] = Field(default_factory=list)
    checks_run: int = Field(default=0)
    best_score: float = Field(default=MIN_SCORE)
    submitted: bool = Field(default=False)
    last_report: dict[str, Any] = Field(default_factory=dict)
    seen_inspections: list[str] = Field(default_factory=list)
    cumulative_reward: float = Field(default=0.0)
    curriculum_stats: dict[str, Any] = Field(default_factory=dict)
    action_history: list[str] = Field(default_factory=list)
    phase_cursor: int = Field(default=0)
    workflow_feedback: str = Field(default="")
    workflow_phase: str = Field(default="triage")
    judge_persona: str = Field(default="junior")
    episode_mode: str = Field(default="standard")
    incident_brief: str = Field(default="")
    incident_id: str = Field(default="")
    adversarial_seed: int = Field(default=0)
    fault_bundle: list[str] = Field(default_factory=list)
