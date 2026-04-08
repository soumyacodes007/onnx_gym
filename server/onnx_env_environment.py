from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import onnx
import onnxruntime as ort
from onnx import TensorProto, checker, helper, shape_inference

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import OnnxAction, OnnxObservation, OnnxState
    from ..tasks import PATCH_CATALOG, PATCH_DEPENDENCY_GRAPHS, PATCH_LOOKUP, PROFILE_SPECS, TASKS, OnnxTask, RequirementSpec
except ImportError:
    from models import OnnxAction, OnnxObservation, OnnxState
    from tasks import PATCH_CATALOG, PATCH_DEPENDENCY_GRAPHS, PATCH_LOOKUP, PROFILE_SPECS, TASKS, OnnxTask, RequirementSpec


class OnnxEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_order = list(TASKS.keys())
        self._task_index = 0
        self._variant_counters = {task_id: 0 for task_id in self._task_order}
        self._state = self._fresh_state()
        self._task = TASKS[self._task_order[0]]
        self._bundle = self._task.fresh_bundle()

    def _fresh_state(self) -> OnnxState:
        return OnnxState(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: str | None = None) -> OnnxObservation:
        if task_id and task_id in TASKS:
            self._task = TASKS[task_id]
            self._task_index = self._task_order.index(task_id)
        else:
            self._task = TASKS[self._task_order[self._task_index % len(self._task_order)]]
            self._task_index += 1
        variant_index = self._variant_counters[self._task.task_id]
        self._variant_counters[self._task.task_id] = variant_index + 1
        self._bundle = self._task.fresh_bundle(variant_index=variant_index)
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
        return self._make_observation(message='Broken ONNX deployment bundle loaded. Inspect and repair it.', reward=0.0, done=False)

    def step(self, action: OnnxAction) -> OnnxObservation:  # type: ignore[override]
        self._state.step_count += 1
        reward = 0.0
        done = False
        error = ''
        message = ''

        if action.action_type == 'inspect_task':
            reward = self._inspection_reward('inspect_task')
            message = 'Reviewed the ONNX deployment brief and hardware profile.'
        elif action.action_type == 'inspect_bundle':
            reward = self._inspection_reward('inspect_bundle')
            message = 'Inspected the current graph config, IO contract, and deployment target.'
        elif action.action_type == 'inspect_patches':
            reward = self._inspection_reward('inspect_patches')
            message = 'Reviewed the available graph surgery patches and dependency graph.'
        elif action.action_type == 'inspect_report':
            reward = self._inspection_reward('inspect_report')
            message = 'Reviewed checker output, shape inference logs, and ORT diagnostics.'
        elif action.action_type == 'apply_patch':
            reward, error, message = self._apply_patch(action.slot_name, action.patch_id)
        elif action.action_type == 'validate_bundle':
            reward, message = self._validate('Validation complete.')
        elif action.action_type == 'submit_final':
            reward, message = self._validate('Final submission scored.')
            self._state.submitted = True
            done = True
        else:
            error = f'Unsupported action_type: {action.action_type}'
            message = error

        if self._state.step_count >= self._task.max_steps and not done:
            reward, message = self._validate('Step budget exhausted. Auto-submitted current bundle.')
            self._state.submitted = True
            done = True

        self._state.cumulative_reward += reward
        return self._make_observation(message=message, reward=reward, done=done, last_action_error=error)

    def _inspection_reward(self, name: str) -> float:
        if name in self._state.seen_inspections:
            return 0.0
        self._state.seen_inspections.append(name)
        return 0.02

    def _apply_patch(self, slot_name: str, patch_id: str) -> tuple[float, str, str]:
        patch = PATCH_LOOKUP[self._task.task_id].get(patch_id)
        if not patch:
            return 0.0, f'Unknown patch_id: {patch_id}', 'Patch failed.'
        if patch.slot_name != slot_name and slot_name:
            return 0.0, f'Patch {patch_id} belongs to slot {patch.slot_name}, not {slot_name}.', 'Patch failed.'
        if self._state.selected_patches.get(patch.slot_name) == patch.patch_id:
            return 0.0, '', f'Patch {patch.patch_id} already applied to {patch.slot_name}; no change made.'

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
        checker_passed, checker_error = _run_checker(model)
        shape_passed, shape_error, inferred_count = _run_shape_inference(model)
        ort_passed, ort_error = _run_ort_session(model, self._bundle)

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
        endpoint_coverage = _endpoint_coverage(self._task, checker_passed, shape_passed, ort_passed)
        conflicts = _conflict_map(self._task, self._bundle)

        checks = {
            'checker_passed': checker_passed,
            'shape_inference_passed': shape_passed,
            'ort_session_passed': ort_passed,
            'dynamic_batch': bool(graph_cfg.get('dynamic_batch')),
            'dynamic_sequence': bool(graph_cfg.get('dynamic_sequence')),
            'opset_valid': int(graph_cfg.get('opset', 0)) >= 17,
            'label_dtype_match': graph_cfg.get('label_output_dtype') == 'int64' and io.get('output_dtype') == 'int64',
            'token_ids_int64': graph_cfg.get('input_ids_dtype') == 'int64' and io.get('input_dtype') == 'int64',
            'memory_budget_ok': estimated_model_mb <= profile['memory_budget_mb'],
            'target_provider': self._bundle['deployment'].get('target_ep') == profile['target_ep'],
        }
        issue_resolution = {
            'label_dtype_mismatch': checks['label_dtype_match'],
            'static_batch_only': checks['dynamic_batch'],
            'low_opset': checks['opset_valid'],
            'needs_extended_optim': graph_cfg.get('optimization_level') in ('ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL') and ort_passed,
            'wrong_provider': checks['target_provider'],
            'token_ids_not_int64': checks['token_ids_int64'],
            'needs_dynamic_sequence': checks['dynamic_sequence'],
            'memory_budget_exceeded': checks['memory_budget_ok'],
            'resize_has_scales_and_sizes': graph_cfg.get('resize_mode') == 'sizes_only' and checker_passed,
        }

        resolved_issue_ids = {issue_id for issue_id, resolved in issue_resolution.items() if resolved}
        requirement_status = []
        visible_issues = []
        missing = []
        severity_resolved = 0.0
        severity_total = 0.0
        for spec in self._task.requirement_specs:
            passed = _eval_requirement(spec, checks, graph_cfg, io)
            severity_total += spec.severity
            if passed:
                severity_resolved += spec.severity
            elif _is_visible(spec, resolved_issue_ids):
                missing.append(spec.description)
                visible_issues.append({
                    'issue_id': spec.issue_id,
                    'description': spec.description,
                    'severity': spec.severity,
                    'error_hint': spec.error_hint,
                })
            requirement_status.append({
                'key': spec.key,
                'description': spec.description,
                'passed': passed,
                'weight': spec.weight,
                'severity': spec.severity,
                'issue_id': spec.issue_id,
                'visible': _is_visible(spec, resolved_issue_ids),
            })

        hidden_specs = [spec for spec in self._task.requirement_specs if (not _is_visible(spec, resolved_issue_ids)) and not _eval_requirement(spec, checks, graph_cfg, io)]
        severity_ratio = 1.0 if severity_total == 0 else severity_resolved / severity_total
        efficiency = max(0.0, 1.0 - (self._state.step_count / max(self._task.max_steps, 1)))
        score = (severity_ratio * 0.6) + (endpoint_coverage * 0.25) + (efficiency * 0.15)
        if conflicts['detected']:
            score -= 0.05 * min(len(conflicts['detected']), 2)
        if not missing and not conflicts['detected']:
            score = 1.0
        score = round(max(0.0, min(score, 1.0)), 4)

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
            'notes': _notes(self._task, build_meta),
            'profile_summary': profile_summary,
            'variant_label': self._bundle.get('variant_label', 'default'),
            'top_blockers': _top_blockers(visible_issues, conflicts),
            'why_not_perfect': _why_not_perfect(missing, conflicts, estimated_model_mb, profile['memory_budget_mb']),
        }

    def _make_observation(self, *, message: str, reward: float, done: bool, last_action_error: str = '') -> OnnxObservation:
        report = self._state.last_report or self._compute_report()
        self._state.current_bundle = deepcopy(self._bundle)
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
            best_score=self._state.best_score,
            success_threshold=self._task.success_threshold,
            is_success=report['score'] >= self._task.success_threshold,
            message=message,
            last_action_error=last_action_error,
            final_score=report['score'] if done else 0.0,
            possible_actions=_ranked_actions(self._task, report, self._state.selected_patches),
            last_report=report,
            done=done,
            reward=reward,
            metadata={'repair_domain': 'onnx deployment', 'task_requirements': report['missing_requirements']},
        )

    @property
    def state(self) -> OnnxState:
        return self._state


def _build_model(task: OnnxTask, bundle: dict) -> tuple[onnx.ModelProto, dict]:
    gc = bundle['graph_config']
    io = bundle['io_contract']
    if task.task_id == 'label_head_dtype_repair':
        input_shape = io['input_shape']
        output_shape = io['output_shape']
        input_info = helper.make_tensor_value_info(io['input_name'], TensorProto.FLOAT, input_shape)
        weight = helper.make_tensor('W', TensorProto.FLOAT, [4, 3], [0.1] * 12)
        matmul_out = 'logits'
        argmax_out = io['output_name']
        output_dtype = TensorProto.INT64 if io['output_dtype'] == 'int64' else TensorProto.FLOAT
        output_info = helper.make_tensor_value_info(io['output_name'], output_dtype, output_shape)
        nodes = [
            helper.make_node('MatMul', [io['input_name'], 'W'], [matmul_out]),
            helper.make_node('ArgMax', [matmul_out], [argmax_out], axis=1, keepdims=0),
        ]
        graph = helper.make_graph(nodes, task.title, [input_info], [output_info], [weight])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', gc['opset'])], producer_name='onnx-surgeon-gym')
        return model, {'unsupported_ops': [], 'graph_preview': 'MatMul -> ArgMax'}

    if task.task_id == 'embedding_ranker_contract':
        input_dtype = TensorProto.INT64 if io['input_dtype'] == 'int64' else TensorProto.INT32
        input_info = helper.make_tensor_value_info(io['input_name'], input_dtype, io['input_shape'])
        output_info = helper.make_tensor_value_info(io['output_name'], TensorProto.FLOAT, io['output_shape'])
        emb = helper.make_tensor('E', TensorProto.FLOAT, [32, 16], [0.01] * (32 * 16))
        nodes = [
            helper.make_node('Gather', ['E', io['input_name']], ['embedded'], axis=0),
            helper.make_node('ReduceMean', ['embedded'], [io['output_name']], axes=[1], keepdims=0),
        ]
        graph = helper.make_graph(nodes, task.title, [input_info], [output_info], [emb])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', gc['opset'])], producer_name='onnx-surgeon-gym')
        return model, {'unsupported_ops': [], 'graph_preview': 'Gather -> ReduceMean'}

    # vision task
    input_info = helper.make_tensor_value_info(io['input_name'], TensorProto.FLOAT, io['input_shape'])
    output_info = helper.make_tensor_value_info(io['output_name'], TensorProto.FLOAT, io['output_shape'])
    roi = helper.make_tensor('roi', TensorProto.FLOAT, [0], [])
    scales = helper.make_tensor('scales', TensorProto.FLOAT, [4], [1.0, 1.0, 0.5, 0.5])
    sizes = helper.make_tensor('sizes', TensorProto.INT64, [4], [1, 3, 112, 112])
    inputs = [io['input_name'], 'roi']
    if gc['resize_mode'] == 'both':
        inputs.extend(['scales', 'sizes'])
    elif gc['resize_mode'] == 'sizes_only':
        inputs.extend(['', 'sizes'])
    else:
        inputs.extend(['scales', ''])
    nodes = [helper.make_node('Resize', inputs, [io['output_name']], mode='linear')]
    initializers = [roi, scales, sizes]
    graph = helper.make_graph(nodes, task.title, [input_info], [output_info], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', gc['opset'])], producer_name='onnx-surgeon-gym')
    return model, {'unsupported_ops': [], 'graph_preview': f"Resize(mode={gc['resize_mode']})"}


def _run_checker(model: onnx.ModelProto) -> tuple[bool, str]:
    try:
        checker.check_model(model)
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _run_shape_inference(model: onnx.ModelProto) -> tuple[bool, str, int]:
    try:
        inferred = shape_inference.infer_shapes(model)
        return True, '', len(getattr(inferred.graph, 'value_info', []))
    except Exception as exc:
        return False, str(exc), 0


def _run_ort_session(model: onnx.ModelProto, bundle: dict) -> tuple[bool, str]:
    try:
        provider = bundle['deployment']['target_ep']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, bundle['graph_config']['optimization_level'])
        ort.InferenceSession(model.SerializeToString(), sess_options=sess_options, providers=[provider])
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _eval_requirement(spec: RequirementSpec, checks: dict, gc: dict, io: dict) -> bool:
    if spec.key == 'checker_passed':
        if spec.issue_id == 'label_dtype_mismatch':
            return checks['checker_passed'] and checks['label_dtype_match']
        if spec.issue_id == 'resize_has_scales_and_sizes':
            return checks['checker_passed'] and gc.get('resize_mode') == 'sizes_only'
        return checks['checker_passed']
    if spec.key == 'shape_inference_passed':
        return checks['shape_inference_passed'] and gc.get('resize_mode') == 'sizes_only'
    if spec.key == 'ort_session_passed':
        return checks['ort_session_passed'] and gc.get('optimization_level') in ('ORT_ENABLE_EXTENDED', 'ORT_ENABLE_ALL')
    mapping = {
        'dynamic_batch': checks['dynamic_batch'],
        'dynamic_sequence': checks['dynamic_sequence'],
        'opset_valid': checks['opset_valid'],
        'label_dtype_match': checks['label_dtype_match'],
        'token_ids_int64': checks['token_ids_int64'],
        'memory_budget_ok': checks['memory_budget_ok'],
        'target_provider': checks['target_provider'],
    }
    return bool(mapping.get(spec.key, False))


def _is_visible(spec: RequirementSpec, resolved_issue_ids: set[str]) -> bool:
    if spec.initial_visible:
        return True
    return all(dep in resolved_issue_ids for dep in spec.depends_on)


def _bundle_preview(bundle: dict) -> str:
    return (
        'graph_config=' + str(bundle['graph_config']) + '\n'
        'io_contract=' + str(bundle['io_contract']) + '\n'
        'deployment=' + str(bundle['deployment'])
    )


def _slot_status(task: OnnxTask, selected: dict[str, str]) -> dict[str, dict[str, str]]:
    status = {}
    for slot_name in PATCH_CATALOG[task.task_id].keys():
        status[slot_name] = {
            'selected_patch': selected.get(slot_name, ''),
            'slot_state': 'repaired' if slot_name in selected else 'open',
        }
    return status


def _count_dynamic_dims(io_contract: dict) -> int:
    count = 0
    for shape in (io_contract.get('input_shape', []), io_contract.get('output_shape', [])):
        count += sum(1 for dim in shape if isinstance(dim, str))
    return count


def _estimated_activation_mb(io_contract: dict) -> float:
    shape = io_contract.get('output_shape', [])
    numeric = 1
    for dim in shape:
        if isinstance(dim, str):
            numeric *= 4
        else:
            numeric *= max(int(dim), 1)
    return round((numeric * 4) / (1024 * 1024), 4)


def _endpoint_coverage(task: OnnxTask, checker_passed: bool, shape_passed: bool, ort_passed: bool) -> float:
    base = 0.0
    base += 0.34 if checker_passed else 0.0
    base += 0.33 if shape_passed else 0.0
    base += 0.33 if ort_passed else 0.0
    return round(base, 4)


def _conflict_map(task: OnnxTask, bundle: dict) -> dict:
    detected = []
    severities = []
    cascade_risk = []
    gc = bundle['graph_config']
    dep = bundle['deployment']
    if dep.get('target_ep') != PROFILE_SPECS[task.deployment_profile]['target_ep']:
        detected.append('execution provider conflicts with deployment profile')
        severities.append(0.8)
        cascade_risk.append(True)
    if gc.get('optimization_level') == 'ORT_ENABLE_ALL' and gc.get('estimated_model_mb', 0.0) > PROFILE_SPECS[task.deployment_profile]['memory_budget_mb']:
        detected.append('aggressive optimization with oversized model footprint')
        severities.append(0.6)
        cascade_risk.append(False)
    return {'detected': detected, 'severity': severities, 'cascade_risk': cascade_risk}


def _error_log(task: OnnxTask, checker_error: str, shape_error: str, ort_error: str, visible_issues: list[dict], conflicts: dict) -> list[str]:
    logs = []
    if checker_error:
        logs.append(f'onnx.checker: {checker_error}')
    if shape_error:
        logs.append(f'onnx.shape_inference: {shape_error}')
    if ort_error:
        logs.append(f'onnxruntime: {ort_error}')
    for issue in visible_issues[:2]:
        logs.append(issue['error_hint'])
    if conflicts['detected']:
        logs.append('onnxruntime: deployment conflict detected: ' + ', '.join(conflicts['detected']))
    if not logs:
        logs.append(f'onnxruntime: {task.title} bundle validated successfully')
    return logs


def _repair_summary(report: dict) -> str:
    return (
        f"Score {report['score']:.2f}; checker={report['checker_passed']}; shape_inference={report['shape_inference_passed']}; "
        f"ort_session={report['ort_session_passed']}; blockers={len(report['top_blockers'])}; variant={report['variant_label']}."
    )


def _recommended_next_action(report: dict, done: bool) -> str:
    if done or report['score'] >= 0.97:
        return 'submit_final'
    if report['visible_issues']:
        return 'apply_patch'
    return 'validate_bundle'


def _notes(task: OnnxTask, build_meta: dict) -> list[str]:
    notes = []
    if task.task_id == 'vision_resize_mobile':
        notes.append('ONNX Resize must not keep both scales and sizes at the same time.')
    if task.task_id == 'embedding_ranker_contract':
        notes.append('Ranker/tokenizer deployments usually standardize on int64 token IDs and symbolic sequence dims.')
    if task.task_id == 'label_head_dtype_repair':
        notes.append('ArgMax labels are int64 in ONNX; output contracts must match that type.')
    return notes


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
            resolves = set(patch.get('resolves', []))
            if resolves & issue_ids:
                actions.append(f"apply_patch:{slot_name}:{patch['patch_id']}")
    actions.extend(['validate_bundle', 'submit_final'])
    deduped = []
    seen = set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            deduped.append(action)
    return deduped
