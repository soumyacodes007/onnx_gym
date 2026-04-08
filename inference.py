import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from models import OnnxAction
    from client import OnnxEnv
except ModuleNotFoundError:
    try:
        from onnx_env import OnnxAction, OnnxEnv
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from models import OnnxAction
        from client import OnnxEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = 'onnx_deployment_surgeon_gym'
TASK_IDS = ['label_head_dtype_repair', 'embedding_ranker_contract', 'vision_resize_mobile']
SUCCESS_THRESHOLD = 0.95

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an ONNX deployment engineer repairing broken export bundles.

    Your goal is to make each deployment bundle satisfy the product brief using the smallest number of high-value repairs.

    Important rules:
    - Prefer one inspection early, then patch, then validate.
    - Do not repeat the same patch.
    - After applying a patch, validate_bundle on the next turn.
    - If the current_score is already above the success threshold after validation, submit_final.
    - Return exactly one JSON object and nothing else.

    Valid actions:
    {
      "action_type": "inspect_task|inspect_bundle|inspect_patches|inspect_report|apply_patch|validate_bundle|submit_final",
      "slot_name": "optional slot name",
      "patch_id": "required only for apply_patch",
      "rationale": "brief optional note"
    }

    No markdown. No code fences. No extra text.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    print(f'[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else "null"}', flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(f'[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={",".join(f"{r:.2f}" for r in rewards)}', flush=True)


async def _connect_env() -> OnnxEnv:
    if LOCAL_IMAGE_NAME:
        return await OnnxEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return OnnxEnv(base_url=ENV_BASE_URL or 'http://localhost:7860')


def _build_user_prompt(observation: Any) -> str:
    return textwrap.dedent(
        f"""
        Task: {observation.task_title} ({observation.difficulty})
        Description: {observation.task_description}
        Product brief: {observation.product_brief}
        Deployment profile: {observation.deployment_profile}
        Variant: {observation.variant_label}
        Profile summary:
        {json.dumps(observation.profile_summary, indent=2)}

        Bundle:
        {json.dumps(observation.current_bundle, indent=2)}

        Validation report:
        {json.dumps(observation.validation_report, indent=2)}

        Visible issues:
        {json.dumps(observation.visible_issues, indent=2)}

        Error log:
        {json.dumps(observation.error_log, indent=2)}

        Top blockers:
        {json.dumps(observation.top_blockers, indent=2)}

        Why not perfect:
        {observation.why_not_perfect}

        Possible actions:
        {json.dumps(observation.possible_actions, indent=2)}
        """
    ).strip()


def _find_patch(observation: Any, patch_id: str) -> str:
    for slot_name, patches in observation.patch_catalog.items():
        for patch in patches:
            if patch.get('patch_id') == patch_id and observation.slot_status.get(slot_name, {}).get('selected_patch') != patch_id:
                return slot_name
    return ''


def _deterministic_policy(observation: Any) -> dict[str, Any] | None:
    if observation.checks_run > 0 and observation.current_score >= observation.success_threshold:
        return {'action_type': 'submit_final', 'slot_name': '', 'patch_id': '', 'rationale': 'validation already passed target'}
    if observation.steps_taken == 0:
        return {'action_type': 'inspect_task', 'slot_name': '', 'patch_id': '', 'rationale': 'read deployment brief first'}

    issue_to_patches = {
        'label_dtype_mismatch': ['set_label_output_int64'],
        'static_batch_only': ['set_dynamic_batch'],
        'low_opset': ['set_opset_17'],
        'needs_extended_optim': ['set_extended_optim', 'set_all_optim'],
        'token_ids_not_int64': ['set_input_ids_int64'],
        'needs_dynamic_sequence': ['set_dynamic_sequence'],
        'memory_budget_exceeded': ['prune_debug_initializer'],
        'resize_has_scales_and_sizes': ['resize_sizes_only'],
    }
    for issue in observation.visible_issues:
        for patch_id in issue_to_patches.get(issue.get('issue_id', ''), []):
            slot = _find_patch(observation, patch_id)
            if slot:
                return {'action_type': 'apply_patch', 'slot_name': slot, 'patch_id': patch_id, 'rationale': f"address issue: {issue.get('issue_id', '')}"}

    missing = ' | '.join(observation.missing_requirements).lower()
    wanted = [
        ('int64', 'set_label_output_int64'),
        ('dynamic batch', 'set_dynamic_batch'),
        ('opset 17', 'set_opset_17'),
        ('extended graph optimization', 'set_extended_optim'),
        ('token ids', 'set_input_ids_int64'),
        ('sequence dim', 'set_dynamic_sequence'),
        ('memory budget', 'prune_debug_initializer'),
        ('resize', 'resize_sizes_only'),
        ('all graph optimizations', 'set_all_optim'),
    ]
    for needle, patch_id in wanted:
        if needle in missing:
            slot = _find_patch(observation, patch_id)
            if slot:
                return {'action_type': 'apply_patch', 'slot_name': slot, 'patch_id': patch_id, 'rationale': f'address requirement: {needle}'}
    return None


def _call_llm(client: OpenAI | None, observation: Any) -> dict[str, Any]:
    deterministic = _deterministic_policy(observation)
    if deterministic is not None:
        return deterministic
    if client is None:
        return {'action_type': 'validate_bundle', 'slot_name': '', 'patch_id': '', 'rationale': 'llm unavailable'}
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': _build_user_prompt(observation)}],
            temperature=0.0,
            max_tokens=180,
        )
        text = (response.choices[0].message.content or '').strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        payload = json.loads(text[start:end])
    except Exception:
        return {'action_type': 'validate_bundle', 'slot_name': '', 'patch_id': '', 'rationale': 'fallback'}
    allowed = {'inspect_task', 'inspect_bundle', 'inspect_patches', 'inspect_report', 'apply_patch', 'validate_bundle', 'submit_final'}
    if payload.get('action_type') not in allowed:
        payload['action_type'] = 'validate_bundle'
    payload.setdefault('slot_name', '')
    payload.setdefault('patch_id', '')
    payload.setdefault('rationale', '')
    return payload


def _action_to_string(payload: dict[str, Any]) -> str:
    parts = [payload.get('action_type', 'validate_bundle')]
    if payload.get('slot_name'):
        parts.append(f"slot={payload['slot_name']}")
    if payload.get('patch_id'):
        parts.append(f"patch={payload['patch_id']}")
    return '|'.join(parts)


async def run_task(client: OpenAI | None, task_id: str) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_was_patch = False
    seen_patch_ids: set[str] = set()
    env: OnnxEnv | None = None
    try:
        env = await _connect_env()
        result = await env.reset(task_id=task_id)
        observation = result.observation
        for step in range(1, (observation.max_steps or 10) + 1):
            if last_was_patch:
                payload = {'action_type': 'validate_bundle', 'slot_name': '', 'patch_id': '', 'rationale': 'forced validation after patch'}
            elif observation.current_score >= observation.success_threshold and observation.checks_run > 0:
                payload = {'action_type': 'submit_final', 'slot_name': '', 'patch_id': '', 'rationale': 'score already good'}
            else:
                payload = _call_llm(client, observation)
                if payload.get('action_type') == 'apply_patch' and payload.get('patch_id') in seen_patch_ids:
                    payload = {'action_type': 'validate_bundle', 'slot_name': '', 'patch_id': '', 'rationale': 'avoid repeating same patch'}
            action = OnnxAction(action_type=payload.get('action_type', 'validate_bundle'), slot_name=payload.get('slot_name', ''), patch_id=payload.get('patch_id', ''), rationale=payload.get('rationale', ''))
            result = await env.step(action)
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            log_step(step, _action_to_string(payload), reward, result.done, observation.last_action_error or None)
            if action.action_type == 'apply_patch' and action.patch_id:
                seen_patch_ids.add(action.patch_id)
            if result.done:
                break
            last_was_patch = action.action_type == 'apply_patch'
        if not result.done:
            result = await env.step(OnnxAction(action_type='submit_final'))
            observation = result.observation
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken += 1
            log_step(steps_taken, 'submit_final', reward, result.done, observation.last_action_error or None)
        score = float(observation.final_score or observation.current_score or observation.best_score)
        score = max(0.0, min(1.0, score))
        success = bool(observation.is_success) or score >= SUCCESS_THRESHOLD
    except Exception:
        score = 0.0
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success, steps_taken, score, rewards)
    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    scores = []
    for task_id in TASK_IDS:
        scores.append(await run_task(client, task_id))
    _ = scores


if __name__ == '__main__':
    asyncio.run(main())
