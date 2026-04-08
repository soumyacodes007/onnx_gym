from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import OnnxAction, OnnxObservation, OnnxState
except ImportError:
    from models import OnnxAction, OnnxObservation, OnnxState


class OnnxEnv(EnvClient[OnnxAction, OnnxObservation, OnnxState]):
    def _step_payload(self, action: OnnxAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "slot_name": action.slot_name,
            "patch_id": action.patch_id,
            "rationale": action.rationale,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[OnnxObservation]:
        obs_data = payload.get("observation", {})
        observation = OnnxObservation(
            task_id=obs_data.get("task_id", ""),
            task_title=obs_data.get("task_title", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            task_description=obs_data.get("task_description", ""),
            product_brief=obs_data.get("product_brief", ""),
            deployment_profile=obs_data.get("deployment_profile", ""),
            variant_label=obs_data.get("variant_label", ""),
            current_bundle=obs_data.get("current_bundle", {}),
            bundle_preview=obs_data.get("bundle_preview", ""),
            graph_preview=obs_data.get("graph_preview", ""),
            profile_summary=obs_data.get("profile_summary", {}),
            available_slots=obs_data.get("available_slots", []),
            slot_status=obs_data.get("slot_status", {}),
            patch_catalog=obs_data.get("patch_catalog", {}),
            patch_dependency_graph=obs_data.get("patch_dependency_graph", {}),
            requirement_status=obs_data.get("requirement_status", {}),
            validation_report=obs_data.get("validation_report", {}),
            missing_requirements=obs_data.get("missing_requirements", []),
            visible_issues=obs_data.get("visible_issues", []),
            hidden_issues=obs_data.get("hidden_issues", 0),
            cascade_depth=obs_data.get("cascade_depth", 0),
            cascade_depth_remaining=obs_data.get("cascade_depth_remaining", 0),
            flag_conflict_map=obs_data.get("flag_conflict_map", {}),
            target_ep=obs_data.get("target_ep", "CPUExecutionProvider"),
            memory_budget_mb=obs_data.get("memory_budget_mb", 0.0),
            estimated_model_mb=obs_data.get("estimated_model_mb", 0.0),
            estimated_activation_mb=obs_data.get("estimated_activation_mb", 0.0),
            endpoint_coverage=obs_data.get("endpoint_coverage", 0.0),
            severity_weighted_score=obs_data.get("severity_weighted_score", 0.0),
            checker_passed=obs_data.get("checker_passed", False),
            shape_inference_passed=obs_data.get("shape_inference_passed", False),
            ort_session_passed=obs_data.get("ort_session_passed", False),
            inferred_value_info_count=obs_data.get("inferred_value_info_count", 0),
            dynamic_dim_count=obs_data.get("dynamic_dim_count", 0),
            node_count=obs_data.get("node_count", 0),
            initializer_count=obs_data.get("initializer_count", 0),
            unsupported_ops=obs_data.get("unsupported_ops", []),
            error_log=obs_data.get("error_log", []),
            top_blockers=obs_data.get("top_blockers", []),
            why_not_perfect=obs_data.get("why_not_perfect", ""),
            fix_history=obs_data.get("fix_history", []),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            repair_summary=obs_data.get("repair_summary", ""),
            recommended_next_action=obs_data.get("recommended_next_action", ""),
            checks_run=obs_data.get("checks_run", 0),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 0),
            current_score=obs_data.get("current_score", 0.0),
            best_score=obs_data.get("best_score", 0.0),
            success_threshold=obs_data.get("success_threshold", 0.95),
            is_success=obs_data.get("is_success", False),
            message=obs_data.get("message", ""),
            last_action_error=obs_data.get("last_action_error", ""),
            final_score=obs_data.get("final_score", 0.0),
            possible_actions=obs_data.get("possible_actions", []),
            last_report=obs_data.get("last_report", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict[str, Any]) -> OnnxState:
        return OnnxState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_title=payload.get("task_title", ""),
            difficulty=payload.get("difficulty", "easy"),
            deployment_profile=payload.get("deployment_profile", ""),
            variant_label=payload.get("variant_label", ""),
            current_bundle=payload.get("current_bundle", {}),
            selected_patches=payload.get("selected_patches", {}),
            patch_history=payload.get("patch_history", []),
            checks_run=payload.get("checks_run", 0),
            best_score=payload.get("best_score", 0.0),
            submitted=payload.get("submitted", False),
            last_report=payload.get("last_report", {}),
            seen_inspections=payload.get("seen_inspections", []),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
        )
