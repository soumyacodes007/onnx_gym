from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, shape_inference

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import OnnxAction, OnnxObservation, OnnxState
    from ..tasks import PATCH_CATALOG, PATCH_DEPENDENCY_GRAPHS, PATCH_LOOKUP, PROFILE_SPECS, TASKS, OnnxTask, RequirementSpec
    from .curriculum import CurriculumController
    from .generator import EpisodeGenerator
    from .judge import evaluate_step
except ImportError:
    from models import OnnxAction, OnnxObservation, OnnxState
    from tasks import PATCH_CATALOG, PATCH_DEPENDENCY_GRAPHS, PATCH_LOOKUP, PROFILE_SPECS, TASKS, OnnxTask, RequirementSpec
    from server.curriculum import CurriculumController
    from server.generator import EpisodeGenerator
    from server.judge import evaluate_step


MIN_FINAL_SCORE = 0.0
MAX_FINAL_SCORE = 1.0
MIN_STEP_REWARD = -1.0
MAX_STEP_REWARD = 1.0


def _clamp_final_score(value: float) -> float:
    return round(max(MIN_FINAL_SCORE, min(MAX_FINAL_SCORE, float(value))), 2)


def _clamp_step_reward(value: float) -> float:
    return round(max(MIN_STEP_REWARD, min(MAX_STEP_REWARD, float(value))), 2)


class OnnxEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_order = list(TASKS.keys())
        self._generator = EpisodeGenerator()
        self._curriculum = CurriculumController()
        self._mode = os.environ.get("GYM_MODE", "standard").lower()
        self._state = self._fresh_state()
        self._task = TASKS[self._task_order[0]]
        self._bundle = self._task.fresh_bundle()
        self._patches_since_validation = 0
        self._inspection_before_patch = False
        self._episode_log_path = Path(os.environ.get('EPISODE_LOG', 'outputs/onnx_episode_transcripts.jsonl'))

    def _fresh_state(self) -> OnnxState:
        return OnnxState(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: str | None = None) -> OnnxObservation:
        split = "eval" if self._mode == "eval" else "train"
        plan = self._generator.next_plan(curriculum=self._curriculum, split=split, task_id=task_id)
        self._task = TASKS[plan.task_id]
        self._bundle = self._task.fresh_bundle(variant_index=plan.variant_index)
        self._state = self._fresh_state()
        self._state.task_id = self._task.task_id
        self._state.task_title = self._task.title
        self._state.difficulty = self._task.difficulty
        self._state.deployment_profile = self._task.deployment_profile
        self._state.variant_label = self._bundle.get('variant_label', 'default')
        self._state.current_bundle = deepcopy(self._bundle)
        self._state.selected_patches = {}
        self._state.patch_history = []
        self._state.checks_run = 0
        self._state.best_score = 0.0
        self._state.submitted = False
        self._state.last_report = {}
        self._state.seen_inspections = []
        self._state.cumulative_reward = 0.0
        self._state.action_history = []
        self._state.phase_cursor = 0
        self._state.workflow_feedback = "Episode started."
        self._state.workflow_phase = "triage"
        self._state.judge_persona = plan.judge_persona
        self._state.episode_mode = plan.mode
        self._state.incident_brief = plan.incident_brief
        self._state.incident_id = plan.incident_id
        self._state.adversarial_seed = plan.adversarial_seed
        self._state.fault_bundle = list(plan.fault_bundle)
        self._state.curriculum_stats = self._curriculum.stats()
        self._patches_since_validation = 0
        self._inspection_before_patch = False
        if plan.fault_bundle:
            self._inject_fault_bundle(list(plan.fault_bundle))
        return self._make_observation(
            message='Broken ONNX deployment bundle loaded. Inspect and repair it.',
            reward=0.0,
            done=False,
        )

    def step(self, action: OnnxAction) -> OnnxObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        done = False
        error = ''
        message = ''

        if action.action_type == 'inspect_task':
            reward = self._inspection_reward('inspect_task')
            self._inspection_before_patch = True
            message = 'Reviewed the task brief, profile, and release goal.'
        elif action.action_type == 'inspect_bundle':
            reward = self._inspection_reward('inspect_bundle')
            self._inspection_before_patch = True
            message = 'Inspected the graph config, IO contract, and deployment bundle.'
        elif action.action_type == 'inspect_patches':
            reward = self._inspection_reward('inspect_patches')
            self._inspection_before_patch = True
            message = 'Reviewed patch catalog and dependency graph.'
        elif action.action_type == 'inspect_report':
            reward = self._inspection_reward('inspect_report')
            self._inspection_before_patch = True
            message = 'Reviewed runtime diagnostics and validation blockers.'
        elif action.action_type == 'apply_patch':
            reward, error, message = self._apply_patch(action.slot_name, action.patch_id)
            if not self._inspection_before_patch:
                reward -= 0.02
                message += ' Applied before any inspection; workflow penalty applied.'
        elif action.action_type == 'validate_bundle':
            reward, message = self._validate('Validation complete.')
            if self._patches_since_validation > 0:
                reward = reward + 0.03
                message += ' Validation loop bonus applied.'
            self._patches_since_validation = 0
        elif action.action_type == 'submit_final':
            reward, message = self._validate('Final submission scored.')
            if self._state.checks_run == 0:
                reward = reward - 0.05
                message += ' Submit-before-validate penalty applied.'
            self._patches_since_validation = 0
            self._state.submitted = True
            done = True
        else:
            error = f'Unsupported action_type: {action.action_type}'
            message = error

        if self._state.step_count >= self._task.max_steps and not done:
            reward, message = self._validate('Step budget exhausted. Auto-submitted current bundle.')
            self._state.submitted = True
            done = True

        judge_result = evaluate_step(
            action_type=action.action_type,
            phase_cursor=self._state.phase_cursor,
            checks_run=self._state.checks_run,
            history=self._state.action_history,
            persona=self._state.judge_persona,
        )
        reward = _clamp_step_reward(reward + judge_result.score_delta)
        message = message + f" Workflow: {judge_result.feedback}."
        self._state.phase_cursor = judge_result.next_phase_index
        self._state.workflow_phase = judge_result.phase
        self._state.workflow_feedback = judge_result.feedback
        self._state.action_history.append(action.action_type)

        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        observation = self._make_observation(message=message, reward=reward, done=done, last_action_error=error)
        if done:
            self._finish_episode(observation)
        return observation

    def _inspection_reward(self, name: str) -> float:
        if name in self._state.seen_inspections:
            return 0.0
        self._state.seen_inspections.append(name)
        return 0.02

    def _inject_fault_bundle(self, faults: list[str]) -> None:
        gc = self._bundle.get("graph_config", {})
        io = self._bundle.get("io_contract", {})
        dep = self._bundle.get("deployment", {})
        for fault in faults:
            if fault == "force_low_opset":
                gc["opset"] = min(int(gc.get("opset", 18)), 13)
            elif fault == "force_release_low_opset":
                gc["opset"] = min(int(gc.get("opset", 18)), 11)
            elif fault == "force_static_batch":
                gc["dynamic_batch"] = False
                if isinstance(io.get("input_shape"), list) and io["input_shape"]:
                    io["input_shape"][0] = 1
                if isinstance(io.get("output_shape"), list) and io["output_shape"]:
                    io["output_shape"][0] = 1
            elif fault == "force_static_sequence":
                gc["dynamic_sequence"] = False
                if isinstance(io.get("input_shape"), list) and len(io["input_shape"]) > 1:
                    io["input_shape"][1] = 16
            elif fault == "force_label_dtype_float":
                gc["label_output_dtype"] = "float"
                io["output_dtype"] = "float"
            elif fault == "force_input_ids_int32":
                gc["input_ids_dtype"] = "int32"
                io["input_dtype"] = "int32"
            elif fault == "force_basic_optim":
                gc["optimization_level"] = "ORT_ENABLE_BASIC"
            elif fault == "inflate_model_size":
                gc["estimated_model_mb"] = float(gc.get("estimated_model_mb", 0.0)) + 72.0
            elif fault == "force_resize_both":
                gc["resize_mode"] = "both"
            elif fault == "force_rank3_input":
                gc["input_rank"] = 3
                io["input_shape"] = [1, 224, 224]
            elif fault == "force_raw_nms":
                gc["nms_mode"] = "raw"
            elif fault == "force_wrong_provider_cpu":
                dep["target_ep"] = "CPUExecutionProvider"
            elif fault == "force_webnn_mixed_dims":
                gc["webnn_dim_strategy"] = "mixed"
                gc["dynamic_batch"] = True
                gc["dynamic_sequence"] = False
            elif fault == "force_inline_weights":
                gc["external_data"] = False
            elif fault == "force_static_attention_mask":
                gc["attention_mask_dynamic"] = False
                gc["dynamic_sequence"] = False
            elif fault == "force_quant_scale_mismatch":
                gc["quant_scale_mode"] = "mismatched"
            elif fault == "force_quant_low_opset":
                gc["opset"] = min(int(gc.get("opset", 21)), 13)
            elif fault == "force_quant_debug_inline":
                gc["debug_tensors"] = "inline"
                gc["estimated_model_mb"] = float(gc.get("estimated_model_mb", 0.0)) + 44.0
            elif fault == "force_layout_broken":
                gc["bridge_layout"] = "broken"
            elif fault == "force_stage_contract_mismatch":
                gc["stage_contract"] = "mismatch"
            elif fault == "force_unsafe_precision":
                gc["mixed_precision_mode"] = "unsafe"
                io["input_dtype"] = "float16"
                io["output_dtype"] = "float16"

    def _apply_patch(self, slot_name: str, patch_id: str) -> tuple[float, str, str]:
        patch = PATCH_LOOKUP[self._task.task_id].get(patch_id)
        if not patch:
            return 0.0, f'Unknown patch_id: {patch_id}', 'Patch failed.'
        if patch.slot_name != slot_name and slot_name:
            return 0.0, f'Patch {patch_id} belongs to slot {patch.slot_name}, not {slot_name}.', 'Patch failed.'
        if self._state.selected_patches.get(patch.slot_name) == patch.patch_id:
            return -0.03, '', f'Patch {patch.patch_id} already applied to {patch.slot_name}; duplicate patch penalty applied.'

        for section, values in patch.effect.items():
            self._bundle.setdefault(section, {})
            self._bundle[section].update(values)

        self._state.selected_patches[patch.slot_name] = patch.patch_id
        self._state.patch_history.append({
            'slot_name': patch.slot_name,
            'patch_id': patch.patch_id,
            'label': patch.label,
            'resolves': list(patch.resolves),
            'unlocks': list(patch.unlocks),
        })
        self._state.current_bundle = deepcopy(self._bundle)
        self._patches_since_validation += 1
        return 0.0, '', f'Applied patch {patch.patch_id} to {patch.slot_name}.'

    def _validate(self, prefix: str) -> tuple[float, str]:
        report = self._compute_report()
        self._state.checks_run += 1
        self._state.last_report = report
        self._state.best_score = max(self._state.best_score, report['score'])
        missing = ', '.join(report['missing_requirements']) if report['missing_requirements'] else 'none'
        return report['score'], f"{prefix} Score={report['score']:.2f}; missing={missing}."

    def _compute_report(self) -> dict:
        model, build_meta = _build_model(self._task, self._bundle)
        checker_passed, checker_error = _run_checker(self._task, self._bundle, model)
        shape_passed, shape_error, inferred_count = _run_shape_inference(self._task, self._bundle, model)
        ort_passed, ort_error = _run_ort_session(self._task, self._bundle, model)

        graph_cfg = self._bundle['graph_config']
        io = self._bundle['io_contract']
        profile = PROFILE_SPECS[self._task.deployment_profile]
        profile_summary = {
            'name': self._task.deployment_profile,
            'hardware': profile['hardware'],
            'priority': profile['priority'],
            'memory_budget_mb': float(profile['memory_budget_mb']),
            'required_dynamic_dims': int(profile['required_dynamic_dims']),
        }
        estimated_model_mb = float(graph_cfg.get('estimated_model_mb', 0.0))
        estimated_activation_mb = _estimated_activation_mb(io)
        dynamic_dim_count = _count_dynamic_dims(io)
        endpoint_coverage = _endpoint_coverage(checker_passed, shape_passed, ort_passed)
        conflicts = _conflict_map(self._task, self._bundle)
        checks = _build_checks(self._task, self._bundle, checker_passed, shape_passed, ort_passed, dynamic_dim_count)

        evaluated = []
        resolved_issue_ids = set()
        severity_total = 0.0
        severity_resolved = 0.0
        for spec in self._task.requirement_specs:
            passed = _eval_requirement(spec, checks)
            evaluated.append((spec, passed))
            severity_total += spec.severity
            if passed:
                resolved_issue_ids.add(spec.issue_id)
                severity_resolved += spec.severity

        requirement_status = []
        visible_issues = []
        missing = []
        for spec, passed in evaluated:
            visible = _is_visible(spec, resolved_issue_ids)
            if not passed and visible:
                missing.append(spec.description)
                visible_issues.append({'issue_id': spec.issue_id, 'description': spec.description, 'severity': spec.severity, 'error_hint': spec.error_hint})
            requirement_status.append({'key': spec.key, 'description': spec.description, 'passed': passed, 'weight': spec.weight, 'severity': spec.severity, 'issue_id': spec.issue_id, 'visible': visible})

        hidden_specs = [spec for spec, passed in evaluated if (not passed) and not _is_visible(spec, resolved_issue_ids)]
        severity_ratio = 1.0 if severity_total == 0 else severity_resolved / severity_total
        efficiency = max(0.0, 1.0 - (self._state.step_count / max(self._task.max_steps, 1)))
        score = (severity_ratio * 0.6) + (endpoint_coverage * 0.25) + (efficiency * 0.15)
        if conflicts['detected']:
            score -= 0.05 * min(len(conflicts['detected']), 2)
        if not missing and not conflicts['detected']:
            score = 1.0
        score = _clamp_final_score(score)

        return {
            'score': score,
            'severity_weighted_score': round(severity_ratio, 4),
            'requirements': requirement_status,
            'missing_requirements': missing,
            'visible_issues': visible_issues,
            'hidden_issues': len(hidden_specs),
            'cascade_depth': len(self._task.requirement_specs),
            'cascade_depth_remaining': len(hidden_specs),
            'flag_conflict_map': conflicts,
            'target_ep': profile['target_ep'],
            'memory_budget_mb': float(profile['memory_budget_mb']),
            'estimated_model_mb': estimated_model_mb,
            'estimated_activation_mb': estimated_activation_mb,
            'endpoint_coverage': round(endpoint_coverage, 4),
            'checker_passed': checker_passed,
            'shape_inference_passed': shape_passed,
            'ort_session_passed': ort_passed,
            'inferred_value_info_count': inferred_count,
            'dynamic_dim_count': dynamic_dim_count,
            'node_count': len(model.graph.node),
            'initializer_count': len(model.graph.initializer),
            'unsupported_ops': build_meta['unsupported_ops'],
            'error_log': _error_log(self._task, checker_error, shape_error, ort_error, visible_issues, conflicts),
            'graph_preview': build_meta['graph_preview'],
            'notes': _notes(self._task),
            'profile_summary': profile_summary,
            'variant_label': self._bundle.get('variant_label', 'default'),
            'top_blockers': _top_blockers(visible_issues, conflicts),
            'why_not_perfect': _why_not_perfect(missing, conflicts, estimated_model_mb, profile['memory_budget_mb']),
        }

    def _make_observation(self, *, message: str, reward: float, done: bool, last_action_error: str = '') -> OnnxObservation:
        report = self._state.last_report or self._compute_report()
        self._state.current_bundle = deepcopy(self._bundle)
        self._state.curriculum_stats = self._curriculum.stats()
        return OnnxObservation(
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            task_description=self._task.description,
            product_brief=self._task.product_brief,
            deployment_profile=self._task.deployment_profile,
            variant_label=report['variant_label'],
            current_bundle=deepcopy(self._bundle),
            bundle_preview=_bundle_preview(self._bundle),
            graph_preview=report['graph_preview'],
            profile_summary=report['profile_summary'],
            available_slots=list(PATCH_CATALOG[self._task.task_id].keys()),
            slot_status=_slot_status(self._task, self._state.selected_patches),
            patch_catalog=PATCH_CATALOG[self._task.task_id],
            patch_dependency_graph=PATCH_DEPENDENCY_GRAPHS[self._task.task_id],
            requirement_status={item['key']: item for item in report['requirements']},
            validation_report=report,
            missing_requirements=report['missing_requirements'],
            visible_issues=report['visible_issues'],
            hidden_issues=report['hidden_issues'],
            cascade_depth=report['cascade_depth'],
            cascade_depth_remaining=report['cascade_depth_remaining'],
            flag_conflict_map=report['flag_conflict_map'],
            target_ep=report['target_ep'],
            memory_budget_mb=report['memory_budget_mb'],
            estimated_model_mb=report['estimated_model_mb'],
            estimated_activation_mb=report['estimated_activation_mb'],
            endpoint_coverage=report['endpoint_coverage'],
            severity_weighted_score=report['severity_weighted_score'],
            checker_passed=report['checker_passed'],
            shape_inference_passed=report['shape_inference_passed'],
            ort_session_passed=report['ort_session_passed'],
            inferred_value_info_count=report['inferred_value_info_count'],
            dynamic_dim_count=report['dynamic_dim_count'],
            node_count=report['node_count'],
            initializer_count=report['initializer_count'],
            unsupported_ops=report['unsupported_ops'],
            error_log=report['error_log'],
            top_blockers=report['top_blockers'],
            why_not_perfect=report['why_not_perfect'],
            fix_history=self._state.patch_history,
            cumulative_reward=round(self._state.cumulative_reward, 4),
            repair_summary=_repair_summary(report),
            recommended_next_action=_recommended_next_action(report, done),
            checks_run=self._state.checks_run,
            steps_taken=self._state.step_count,
            max_steps=self._task.max_steps,
            current_score=report['score'],
            best_score=max(self._state.best_score, report['score']),
            success_threshold=self._task.success_threshold,
            is_success=report['score'] >= self._task.success_threshold,
            message=message,
            last_action_error=last_action_error,
            final_score=report['score'] if done else 0.0,
            possible_actions=_ranked_actions(self._task, report, self._state.selected_patches),
            last_report=report,
            curriculum_stats=self._state.curriculum_stats,
            workflow_phase=self._state.workflow_phase,
            workflow_feedback=self._state.workflow_feedback,
            judge_persona=self._state.judge_persona,
            episode_mode=self._state.episode_mode,
            incident_brief=self._state.incident_brief,
            incident_id=self._state.incident_id,
            adversarial_seed=self._state.adversarial_seed,
            fault_bundle=list(self._state.fault_bundle),
            done=done,
            reward=reward,
            metadata={'repair_domain': 'onnx deployment', 'task_requirements': report['missing_requirements']},
        )

    def _finish_episode(self, observation: OnnxObservation) -> None:
        success = bool(observation.is_success)
        self._curriculum.record(self._task.task_id, success, observation.current_score, self._state.step_count)
        self._state.curriculum_stats = self._curriculum.stats()
        self._append_transcript(observation)

    def _append_transcript(self, observation: OnnxObservation) -> None:
        try:
            self._episode_log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'episode_id': self._state.episode_id,
                'task_id': self._task.task_id,
                'variant_label': observation.variant_label,
                'deployment_profile': observation.deployment_profile,
                'steps': self._state.step_count,
                'score': observation.current_score,
                'success': observation.is_success,
                'checks_run': observation.checks_run,
                'selected_patches': deepcopy(self._state.selected_patches),
                'patch_history': deepcopy(self._state.patch_history),
                'missing_requirements': list(observation.missing_requirements),
                'top_blockers': list(observation.top_blockers),
                'curriculum_stats': deepcopy(self._state.curriculum_stats),
                'episode_mode': self._state.episode_mode,
                'judge_persona': self._state.judge_persona,
                'incident_brief': self._state.incident_brief,
                'workflow_phase': self._state.workflow_phase,
                'workflow_feedback': self._state.workflow_feedback,
                'incident_id': self._state.incident_id,
                'adversarial_seed': self._state.adversarial_seed,
                'fault_bundle': list(self._state.fault_bundle),
            }
            with self._episode_log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(payload) + '\n')
        except Exception:
            pass

    @property
    def state(self) -> OnnxState:
        return self._state

def _build_model(task: OnnxTask, bundle: dict) -> tuple[onnx.ModelProto, dict]:
    gc = bundle['graph_config']
    io = bundle['io_contract']
    kind = gc.get('graph_kind', '')
    input_info = helper.make_tensor_value_info(io['input_name'], _tensor_dtype(io.get('input_dtype', 'float')), io['input_shape'])
    output_info = helper.make_tensor_value_info(io['output_name'], _tensor_dtype(io.get('output_dtype', 'float')), io['output_shape'])
    graph_preview = 'Identity'
    unsupported_ops: list[str] = []

    if kind == 'classifier_head':
        weight = helper.make_tensor('W', TensorProto.FLOAT, [4, 3], [0.1] * 12)
        nodes = [helper.make_node('MatMul', [io['input_name'], 'W'], ['logits']), helper.make_node('ArgMax', ['logits'], [io['output_name']], axis=1, keepdims=0)]
        initializers = [weight]
        graph_preview = 'MatMul -> ArgMax'
    elif kind == 'embedding_ranker':
        emb = helper.make_tensor('E', TensorProto.FLOAT, [32, 16], [0.01] * (32 * 16))
        nodes = [helper.make_node('Gather', ['E', io['input_name']], ['embedded'], axis=0), helper.make_node('ReduceMean', ['embedded'], [io['output_name']], axes=[1], keepdims=0)]
        initializers = [emb]
        graph_preview = 'Gather -> ReduceMean'
    elif kind in {'vision_resize', 'release_candidate'}:
        roi = helper.make_tensor('roi', TensorProto.FLOAT, [0], [])
        scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5])
        sizes = helper.make_tensor('sizes', TensorProto.INT64, [4], [1, 3, 112, 112])
        resize_inputs = [io['input_name'], 'roi']
        resize_mode = gc.get('resize_mode', 'sizes_only')
        if resize_mode == 'both':
            resize_inputs.extend(['scales', 'sizes'])
        elif resize_mode == 'sizes_only':
            resize_inputs.extend(['', 'sizes'])
        else:
            resize_inputs.extend(['scales', ''])
        nodes = [helper.make_node('Resize', resize_inputs, [io['output_name']], mode='linear')]
        initializers = [roi, scales, sizes]
        graph_preview = f"Resize(mode={resize_mode})"
    else:
        nodes = [helper.make_node('Identity', [io['input_name']], [io['output_name']])]
        initializers = []
        if kind == 'npu_gateway' and gc.get('nms_mode') == 'raw':
            unsupported_ops.append('NonMaxSuppression')
            graph_preview = 'Conv -> NonMaxSuppression'
        elif kind == 'npu_gateway':
            graph_preview = 'Conv -> DecodePolyfill -> TopK'
        elif kind == 'webnn_transform':
            graph_preview = f"Attention(dim_strategy={gc.get('webnn_dim_strategy')})"
        elif kind == 'external_data':
            graph_preview = f"MatMul(packaging={'external' if gc.get('external_data') else 'inline'})"
        elif kind == 'quantized_cascade':
            graph_preview = f"QDQ(scale_mode={gc.get('quant_scale_mode')})"
        elif kind == 'detection_bridge':
            graph_preview = f"Preprocess -> Bridge(layout={gc.get('bridge_layout')}) -> Extractor"

    graph = helper.make_graph(nodes, task.title, [input_info], [output_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', int(gc.get('opset', 18)))], producer_name='onnx-surgeon-gym')
    return model, {'unsupported_ops': unsupported_ops, 'graph_preview': graph_preview}


def _run_checker(task: OnnxTask, bundle: dict, model: onnx.ModelProto) -> tuple[bool, str]:
    gc = bundle['graph_config']
    if gc.get('resize_mode') == 'both':
        return False, 'Resize cannot define both scales and sizes in the same node.'
    if task.task_id == 'label_head_dtype_repair' and bundle['io_contract'].get('output_dtype') != 'int64':
        return False, 'Graph output type does not match ArgMax(int64) output.'
    if task.task_id == 'webnn_static_dynamic_pivot' and gc.get('webnn_dim_strategy') == 'mixed':
        return False, 'Inconsistent dimension annotations across the browser attention contract.'
    try:
        checker.check_model(model)
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _run_shape_inference(task: OnnxTask, bundle: dict, model: onnx.ModelProto) -> tuple[bool, str, int]:
    gc = bundle['graph_config']
    if task.task_id == 'multi_stage_detection_bridge' and gc.get('stage_contract') != 'aligned':
        return False, 'Preprocessor output rank does not match extractor input contract.', 0
    if task.task_id == 'external_data_packaging_failure' and not gc.get('attention_mask_dynamic'):
        return False, 'Attention mask remains frozen to export-time length.', 0
    if task.task_id == 'webnn_static_dynamic_pivot' and gc.get('webnn_dim_strategy') == 'mixed':
        return False, 'Mixed static and dynamic dimensions in the browser graph topology.', 0
    try:
        inferred = shape_inference.infer_shapes(model)
        return True, '', len(getattr(inferred.graph, 'value_info', []))
    except Exception as exc:
        return False, str(exc), 0


def _run_ort_session(task: OnnxTask, bundle: dict, model: onnx.ModelProto) -> tuple[bool, str]:
    gc = bundle['graph_config']
    dep = bundle['deployment']
    target_ep = dep.get('target_ep', 'CPUExecutionProvider')
    kind = gc.get('graph_kind', '')
    if kind == 'npu_gateway' and gc.get('nms_mode') != 'polyfill':
        return False, 'Hardware incompatibility: NonMaxSuppression must be polyfilled before NNAPI admission.'
    if kind == 'webnn_transform' and gc.get('optimization_level') not in ('ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL'):
        return False, 'WebNN runtime requires extended graph optimization.'
    if kind == 'external_data' and not gc.get('external_data'):
        return False, 'Inline weights exceed the sidecar artifact packaging limit.'
    if kind == 'quantized_cascade' and (gc.get('quant_scale_mode') != 'aligned' or int(gc.get('opset', 0)) < 21):
        return False, 'Quantized runtime contract is still invalid for mobile execution.'
    if kind == 'detection_bridge' and gc.get('optimization_level') not in ('ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL'):
        return False, 'Bridge runtime path requires extended optimization after stage alignment.'
    if kind == 'release_candidate' and gc.get('optimization_level') != 'ORT_ENABLE_ALL':
        return False, 'Release candidate requires the final optimization pass before validation.'
    if kind in {'external_data', 'quantized_cascade', 'detection_bridge'}:
        return True, ''
    if target_ep != 'CPUExecutionProvider':
        return True, ''
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, gc.get('optimization_level', 'ORT_ENABLE_BASIC'))
        ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=['CPUExecutionProvider'])
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _build_checks(task: OnnxTask, bundle: dict, checker_passed: bool, shape_passed: bool, ort_passed: bool, dynamic_dim_count: int) -> dict:
    gc = bundle['graph_config']
    io = bundle['io_contract']
    profile = PROFILE_SPECS[task.deployment_profile]
    estimated_model_mb = float(gc.get('estimated_model_mb', 0.0))
    return {
        'checker_passed': checker_passed,
        'shape_inference_passed': shape_passed,
        'ort_session_passed': ort_passed,
        'dynamic_batch': bool(gc.get('dynamic_batch')),
        'dynamic_sequence': bool(gc.get('dynamic_sequence')),
        'opset_valid': int(gc.get('opset', 0)) >= 17,
        'label_dtype_match': gc.get('label_output_dtype') == 'int64' and io.get('output_dtype') == 'int64',
        'token_ids_int64': gc.get('input_ids_dtype') == 'int64' and io.get('input_dtype') == 'int64',
        'memory_budget_ok': estimated_model_mb <= float(profile['memory_budget_mb']),
        'target_provider': bundle['deployment'].get('target_ep') == profile['target_ep'],
        'rank4_input': len(io.get('input_shape', [])) == 4 and gc.get('input_rank') == 4,
        'nms_polyfilled': gc.get('nms_mode') == 'polyfill',
        'webnn_dims_consistent': gc.get('webnn_dim_strategy') in ('all_dynamic', 'all_static'),
        'dynamic_dim_count_match': dynamic_dim_count >= int(profile['required_dynamic_dims']),
        'external_data': bool(gc.get('external_data')),
        'attention_mask_dynamic': bool(gc.get('attention_mask_dynamic')),
        'quant_scale_aligned': gc.get('quant_scale_mode') == 'aligned',
        'quant_opset_valid': int(gc.get('opset', 0)) >= 21,
        'layout_bridge_ok': gc.get('bridge_layout') == 'nchw_bridge',
        'stage_contract_ok': gc.get('stage_contract') == 'aligned',
        'mixed_precision_ok': gc.get('mixed_precision_mode') == 'safe',
        'resize_signature_ok': gc.get('resize_mode') == 'sizes_only',
    }


def _eval_requirement(spec: RequirementSpec, checks: dict) -> bool:
    return bool(checks.get(spec.key, False))


def _is_visible(spec: RequirementSpec, resolved_issue_ids: set[str]) -> bool:
    if spec.initial_visible:
        return True
    return all(dep in resolved_issue_ids for dep in spec.depends_on)


def _bundle_preview(bundle: dict) -> str:
    return 'graph_config=' + str(bundle['graph_config']) + '\n' + 'io_contract=' + str(bundle['io_contract']) + '\n' + 'deployment=' + str(bundle['deployment'])


def _slot_status(task: OnnxTask, selected: dict[str, str]) -> dict[str, dict[str, str]]:
    return {slot_name: {'selected_patch': selected.get(slot_name, ''), 'slot_state': 'repaired' if slot_name in selected else 'open'} for slot_name in PATCH_CATALOG[task.task_id].keys()}


def _count_dynamic_dims(io_contract: dict) -> int:
    count = 0
    for shape in (io_contract.get('input_shape', []), io_contract.get('output_shape', [])):
        count += sum(1 for dim in shape if isinstance(dim, str))
    return count


def _estimated_activation_mb(io_contract: dict) -> float:
    shape = io_contract.get('output_shape', [])
    numeric = 1
    for dim in shape:
        numeric *= 4 if isinstance(dim, str) else max(int(dim), 1)
    return round((numeric * 4) / (1024 * 1024), 4)


def _endpoint_coverage(checker_passed: bool, shape_passed: bool, ort_passed: bool) -> float:
    return round((0.34 if checker_passed else 0.0) + (0.33 if shape_passed else 0.0) + (0.33 if ort_passed else 0.0), 4)


def _conflict_map(task: OnnxTask, bundle: dict) -> dict:
    detected, severities, cascade_risk = [], [], []
    gc = bundle['graph_config']
    dep = bundle['deployment']
    profile = PROFILE_SPECS[task.deployment_profile]
    if dep.get('target_ep') != profile['target_ep']:
        detected.append('execution provider conflicts with deployment profile')
        severities.append(0.8)
        cascade_risk.append(True)
    if gc.get('optimization_level') == 'ORT_ENABLE_ALL' and gc.get('estimated_model_mb', 0.0) > profile['memory_budget_mb']:
        detected.append('aggressive optimization with oversized model footprint')
        severities.append(0.6)
        cascade_risk.append(False)
    if gc.get('graph_kind') == 'webnn_transform' and gc.get('webnn_dim_strategy') == 'all_static' and gc.get('estimated_model_mb', 0.0) > profile['memory_budget_mb']:
        detected.append('static WebNN path overflows browser memory budget')
        severities.append(0.7)
        cascade_risk.append(True)
    return {'detected': detected, 'severity': severities, 'cascade_risk': cascade_risk}

def _error_log(task: OnnxTask, checker_error: str, shape_error: str, ort_error: str, visible_issues: list[dict], conflicts: dict) -> list[str]:
    logs = []
    if checker_error:
        logs.append(f'onnx.checker: {checker_error}')
    if shape_error:
        logs.append(f'onnx.shape_inference: {shape_error}')
    if ort_error:
        logs.append(f'onnxruntime: {ort_error}')
    for issue in visible_issues[:3]:
        logs.append(issue['error_hint'])
    if conflicts['detected']:
        logs.append('deployment-policy: ' + ', '.join(conflicts['detected']))
    if not logs:
        logs.append(f'onnxruntime: {task.title} bundle validated successfully')
    return logs


def _repair_summary(report: dict) -> str:
    return f"Score {report['score']:.2f}; checker={report['checker_passed']}; shape_inference={report['shape_inference_passed']}; ort_session={report['ort_session_passed']}; blockers={len(report['top_blockers'])}; variant={report['variant_label']}."


def _recommended_next_action(report: dict, done: bool) -> str:
    if done or report['score'] >= 0.97:
        return 'submit_final'
    if report['visible_issues']:
        return 'apply_patch'
    return 'validate_bundle'


def _notes(task: OnnxTask) -> list[str]:
    notes = {
        'vision_resize_mobile': ['ONNX Resize must not keep both scales and sizes at the same time.'],
        'embedding_ranker_contract': ['Ranker/tokenizer deployments usually standardize on int64 token IDs and symbolic sequence dims.'],
        'label_head_dtype_repair': ['ArgMax labels are int64 in ONNX; output contracts must match that type.'],
        'npu_gateway_surgery': ['Mobile NPUs usually require rank-4 image admission and provider-safe post-processing.'],
        'webnn_static_dynamic_pivot': ['WebNN is strict about mixed static/dynamic dimensions in attention graphs.'],
        'external_data_packaging_failure': ['External data packaging is often required before larger transformer exports can ship.'],
        'broken_quantized_cascade': ['Quantized exports usually fail in coupled ways: scale alignment, opset age, and artifact bloat.'],
        'multi_stage_detection_bridge': ['Two-stage pipelines often break at the stage handoff, not in either stage individually.'],
        'release_candidate_gate': ['Release gates are often blocked by multiple linked export policy failures, not one isolated bug.'],
    }
    return notes.get(task.task_id, [])


def _top_blockers(visible_issues: list[dict], conflicts: dict) -> list[str]:
    blockers = [issue['description'] for issue in sorted(visible_issues, key=lambda item: item['severity'], reverse=True)[:3]]
    blockers.extend(conflicts.get('detected', [])[:1])
    return blockers


def _why_not_perfect(missing: list[str], conflicts: dict, estimated_model_mb: float, budget_mb: float) -> str:
    if not missing and not conflicts.get('detected'):
        return 'Bundle is deployment-ready.'
    reasons = []
    if missing:
        reasons.append(f'missing {len(missing)} required deployment checks')
    if conflicts.get('detected'):
        reasons.append('flag/profile conflicts still present')
    if estimated_model_mb > budget_mb:
        reasons.append(f'model footprint {estimated_model_mb:.1f}MB exceeds {budget_mb:.1f}MB budget')
    return '; '.join(reasons)


def _ranked_actions(task: OnnxTask, report: dict, selected_patches: dict[str, str]) -> list[str]:
    actions = ['inspect_task', 'inspect_bundle', 'inspect_patches', 'inspect_report']
    issue_ids = {issue['issue_id'] for issue in report['visible_issues']}
    for slot_name, patches in PATCH_CATALOG[task.task_id].items():
        for patch in patches:
            if selected_patches.get(slot_name) == patch['patch_id']:
                continue
            if set(patch.get('resolves', [])) & issue_ids:
                actions.append(f"apply_patch:{slot_name}:{patch['patch_id']}")
    actions.extend(['validate_bundle', 'submit_final'])
    deduped, seen = [], set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            deduped.append(action)
    return deduped


def _tensor_dtype(name: str) -> int:
    mapping = {
        'float': TensorProto.FLOAT,
        'float16': TensorProto.FLOAT16,
        'int64': TensorProto.INT64,
        'int32': TensorProto.INT32,
        'uint8': TensorProto.UINT8,
    }
    return mapping.get(name, TensorProto.FLOAT)
