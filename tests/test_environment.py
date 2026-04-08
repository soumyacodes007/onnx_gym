import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from onnx_env.server.onnx_env_environment import OnnxEnvironment
from onnx_env import OnnxAction


def test_label_head_dtype_repair_success():
    env = OnnxEnvironment()
    obs = env.reset(task_id='label_head_dtype_repair')
    assert obs.current_score < 0.95
    env.step(OnnxAction(action_type='apply_patch', slot_name='io_contract', patch_id='set_label_output_int64'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_dynamic_batch'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_opset_17'))
    env.step(OnnxAction(action_type='apply_patch', slot_name='graph_config', patch_id='set_extended_optim'))
    obs = env.step(OnnxAction(action_type='submit_final'))
    assert obs.is_success is True


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


def test_reset_cycles_variant_labels():
    env = OnnxEnvironment()
    obs1 = env.reset(task_id='label_head_dtype_repair')
    obs2 = env.reset(task_id='label_head_dtype_repair')
    assert obs1.variant_label != ""
    assert obs2.variant_label != ""
    assert obs1.variant_label != obs2.variant_label
