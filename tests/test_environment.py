import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onnx_env.server.onnx_env_environment import OnnxEnvironment
from onnx_env import OnnxAction


def test_label_head_dtype_repair_success():
    env = OnnxEnvironment()
    obs = env.reset(task_id='label_head_dtype_repair')
    assert 0.0 < obs.current_score < 0.95
    assert 0.0 <= obs.best_score <= 1.0
    assert 0.0 <= obs.final_score <= 1.0
    env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_label_output_int64'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_opset_17'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_extended_optim'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True
    assert 0.0 <= obs.current_score <= 1.0
    assert 0.0 <= obs.best_score <= 1.0
    assert 0.0 <= obs.final_score <= 1.0


def test_embedding_ranker_partial_then_full():
    env = OnnxEnvironment()
    env.reset(task_id='embedding_ranker_contract')
    env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_input_ids_int64'))
    obs = env.step(OnnxAction(action_type='validate_bundle'))
    assert 0.0 < obs.current_score < 0.96
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_sequence'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_extended_optim'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True
    assert 0.0 <= obs.current_score <= 1.0
    assert 0.0 <= obs.best_score <= 1.0
    assert 0.0 <= obs.final_score <= 1.0


def test_vision_resize_mobile_requires_resize_fix():
    env = OnnxEnvironment()
    env.reset(task_id='vision_resize_mobile')
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
    obs = env.step(OnnxAction(action_type='validate_bundle'))
    assert obs.current_score < 0.97
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='resize_sizes_only'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_all_optim'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='prune_debug_initializer'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True
    assert 0.0 <= obs.current_score <= 1.0
    assert 0.0 <= obs.best_score <= 1.0
    assert 0.0 <= obs.final_score <= 1.0


def test_reset_cycles_variant_labels():
    env = OnnxEnvironment()
    obs1 = env.reset(task_id='label_head_dtype_repair')
    obs2 = env.reset(task_id='label_head_dtype_repair')
    assert obs1.variant_label != ""
    assert obs2.variant_label != ""
    assert obs1.variant_label != obs2.variant_label


def test_npu_gateway_requires_rank_then_provider_chain():
    env = OnnxEnvironment()
    env.reset(task_id='npu_gateway_surgery')
    obs = env.step(OnnxAction(action_type='validate_bundle'))
    assert obs.current_score < 0.97
    env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_rank4_nchw'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='polyfill_nms'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='deployment', patch_id='switch_nnapi_provider'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='enable_npu_optim'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True
    assert obs.curriculum_stats["tier"] in {"warmup", "runtime", "compound"}


def test_release_candidate_gate_full_chain():
    env = OnnxEnvironment()
    env.reset(task_id='release_candidate_gate')
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='raise_release_opset_18'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='fix_release_resize'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_safe_mixed_precision'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='deployment', patch_id='switch_coreml_provider'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='externalize_release_weights'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_all_optim'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True


def test_phase_skip_penalty_applied():
    env = OnnxEnvironment()
    env.reset(task_id='label_head_dtype_repair')
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert "skipped phase" in obs.workflow_feedback


def test_repeat_penalty_applied():
    env = OnnxEnvironment()
    env.reset(task_id='embedding_ranker_contract')
    env.step(OnnxAction(action_type='inspect_task'))
    env.step(OnnxAction(action_type='inspect_task'))
    obs = env.step(OnnxAction(action_type='inspect_task'))
    assert "repeated action pattern" in obs.workflow_feedback
    assert obs.reward < 0.0


def test_negative_reward_preserved():
    env = OnnxEnvironment()
    env.reset(task_id='label_head_dtype_repair')
    env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_label_output_int64'))
    obs = env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_label_output_int64'))
    assert obs.reward < 0.0


def test_curriculum_tier_transition():
    env = OnnxEnvironment()
    task_id = "label_head_dtype_repair"
    for _ in range(4):
        env.reset(task_id=task_id)
        env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_label_output_int64'))
        env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
        env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_opset_17'))
        env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_extended_optim'))
        env.step(OnnxAction(action_type='submit_final'))
    stats = env._curriculum.stats()
    assert stats["episode_count"] >= 4
    assert stats["tier"] in {"warmup", "runtime", "compound"}
