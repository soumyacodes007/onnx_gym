from __future__ import annotations

from typing import Any


ISSUE_PATCH_ORDER = {
    "label_dtype_mismatch": ["set_label_output_int64"],
    "static_batch_only": ["set_dynamic_batch"],
    "low_opset": ["set_opset_17"],
    "needs_extended_optim": ["set_extended_optim", "set_all_optim"],
    "token_ids_not_int64": ["set_input_ids_int64"],
    "needs_dynamic_sequence": ["set_dynamic_sequence", "set_attention_mask_dynamic"],
    "memory_budget_exceeded": ["prune_debug_initializer", "prune_browser_cache", "prune_quant_debug_tensors", "externalize_weights", "externalize_release_weights"],
    "resize_has_scales_and_sizes": ["resize_sizes_only"],
    "shape_pipe_not_ready": ["set_dynamic_batch"],
    "rank3_input": ["set_rank4_nchw"],
    "nms_not_polyfilled": ["polyfill_nms"],
    "wrong_provider": ["switch_nnapi_provider", "switch_webnn_provider", "switch_coreml_provider"],
    "needs_npu_optim": ["enable_npu_optim"],
    "mixed_webnn_dims": ["set_all_dynamic_dims", "set_all_static_dims"],
    "webnn_budget_exceeded": ["prune_browser_cache"],
    "needs_browser_optim": ["enable_browser_optim"],
    "insufficient_dynamic_dims": ["set_all_dynamic_dims", "set_attention_mask_dynamic", "set_dynamic_batch"],
    "inline_weights": ["externalize_weights"],
    "static_attention_mask": ["set_attention_mask_dynamic"],
    "needs_packaging_optim": ["set_extended_optim"],
    "packaging_budget_exceeded": ["externalize_weights"],
    "quant_scale_mismatch": ["align_qdq_scales"],
    "low_quant_opset": ["raise_quant_opset_21"],
    "debug_tensor_bloat": ["prune_quant_debug_tensors"],
    "needs_quant_optim": ["enable_quant_fusion"],
    "fixed_quant_batch": ["enable_quant_fusion"],
    "layout_bridge_missing": ["insert_nchw_bridge"],
    "stage_contract_mismatch": ["align_stage_contract"],
    "bridge_batch_frozen": ["set_dynamic_batch"],
    "needs_bridge_optim": ["set_extended_optim"],
    "release_low_opset": ["raise_release_opset_18"],
    "release_resize_invalid": ["fix_release_resize"],
    "release_precision_unsafe": ["set_safe_mixed_precision"],
    "release_batch_static": ["set_dynamic_batch"],
    "release_provider_wrong": ["switch_coreml_provider"],
    "release_packaging_inline": ["externalize_release_weights"],
    "release_needs_optim": ["set_all_optim"],
}


def find_patch_slot(observation: Any, patch_id: str) -> str:
    for slot_name, patches in observation.patch_catalog.items():
        for patch in patches:
            if patch.get("patch_id") == patch_id and observation.slot_status.get(slot_name, {}).get("selected_patch") != patch_id:
                return slot_name
    return ""


def deterministic_action(observation: Any, seen_patch_ids: set[str] | None = None) -> dict[str, Any]:
    seen_patch_ids = seen_patch_ids or set()

    if observation.checks_run > 0 and observation.current_score >= observation.success_threshold:
        return {"action_type": "submit_final", "slot_name": "", "patch_id": "", "rationale": "validation already passed threshold"}

    if observation.steps_taken == 0:
        return {"action_type": "inspect_task", "slot_name": "", "patch_id": "", "rationale": "read the deployment brief first"}
    if observation.steps_taken == 1:
        return {"action_type": "inspect_bundle", "slot_name": "", "patch_id": "", "rationale": "inspect the broken bundle before surgery"}

    for issue in observation.visible_issues:
        for patch_id in ISSUE_PATCH_ORDER.get(issue.get("issue_id", ""), []):
            if patch_id in seen_patch_ids:
                continue
            slot = find_patch_slot(observation, patch_id)
            if slot:
                return {"action_type": "apply_patch", "slot_name": slot, "patch_id": patch_id, "rationale": f"address issue {issue.get('issue_id', '')}"}

    if observation.checks_run == 0 or observation.recommended_next_action == "validate_bundle":
        return {"action_type": "validate_bundle", "slot_name": "", "patch_id": "", "rationale": "refresh diagnostics"}

    if observation.current_score >= observation.success_threshold:
        return {"action_type": "submit_final", "slot_name": "", "patch_id": "", "rationale": "current score already meets success threshold"}

    return {"action_type": "inspect_report", "slot_name": "", "patch_id": "", "rationale": "review blockers again"}
