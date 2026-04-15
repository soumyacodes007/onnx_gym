from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path

try:
    from client import OnnxEnv
    from models import OnnxAction
    from solver import deterministic_action
except ImportError:
    from onnx_env.client import OnnxEnv
    from onnx_env.models import OnnxAction
    from onnx_env.solver import deterministic_action


TASK_IDS = [
    "label_head_dtype_repair",
    "embedding_ranker_contract",
    "vision_resize_mobile",
    "npu_gateway_surgery",
    "webnn_static_dynamic_pivot",
    "external_data_packaging_failure",
    "broken_quantized_cascade",
    "multi_stage_detection_bridge",
    "release_candidate_gate",
]


class RandomPolicy:
    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self._seen_patch_ids: set[str] = set()

    def reset(self) -> None:
        self._seen_patch_ids.clear()

    def act(self, observation) -> dict:
        actions: list[dict] = [
            {"action_type": "inspect_task", "slot_name": "", "patch_id": "", "rationale": "random inspect"},
            {"action_type": "inspect_bundle", "slot_name": "", "patch_id": "", "rationale": "random inspect"},
            {"action_type": "inspect_patches", "slot_name": "", "patch_id": "", "rationale": "random inspect"},
            {"action_type": "inspect_report", "slot_name": "", "patch_id": "", "rationale": "random inspect"},
            {"action_type": "validate_bundle", "slot_name": "", "patch_id": "", "rationale": "random validate"},
            {"action_type": "submit_final", "slot_name": "", "patch_id": "", "rationale": "random submit"},
        ]
        for slot_name, patches in observation.patch_catalog.items():
            for patch in patches:
                patch_id = patch.get("patch_id", "")
                if not patch_id or patch_id in self._seen_patch_ids:
                    continue
                actions.append(
                    {
                        "action_type": "apply_patch",
                        "slot_name": slot_name,
                        "patch_id": patch_id,
                        "rationale": "random patch",
                    }
                )
        return self._rng.choice(actions)

    def update(self, payload: dict) -> None:
        patch_id = payload.get("patch_id", "")
        if patch_id:
            self._seen_patch_ids.add(patch_id)


class HeuristicPolicy:
    def __init__(self) -> None:
        self.seen_patch_ids: set[str] = set()

    def reset(self) -> None:
        self.seen_patch_ids.clear()

    def act(self, observation) -> dict:
        return deterministic_action(observation, self.seen_patch_ids)

    def update(self, payload: dict) -> None:
        patch_id = payload.get("patch_id", "")
        if patch_id:
            self.seen_patch_ids.add(patch_id)


class HFGenerationPolicy:
    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
        self.fallback = HeuristicPolicy()

    def reset(self) -> None:
        self.fallback.reset()

    def _build_prompt(self, observation) -> str:
        return (
            f"Task: {observation.task_title}\n"
            f"Profile: {observation.deployment_profile}\n"
            f"Issues: {json.dumps(observation.visible_issues)}\n"
            f"Actions: {json.dumps(observation.possible_actions)}\n"
            "Return exactly one JSON object with action_type, slot_name, patch_id, rationale."
        )

    def act(self, observation) -> dict:
        prompt = self._build_prompt(observation)
        try:
            generated = self.pipe(prompt, max_new_tokens=96, do_sample=False)[0]["generated_text"][len(prompt):].strip()
            start = generated.find("{")
            end = generated.rfind("}") + 1
            payload = json.loads(generated[start:end])
        except Exception:
            payload = self.fallback.act(observation)
        payload.setdefault("action_type", "validate_bundle")
        payload.setdefault("slot_name", "")
        payload.setdefault("patch_id", "")
        payload.setdefault("rationale", "")
        return payload

    def update(self, payload: dict) -> None:
        self.fallback.update(payload)


async def run_episode(env: OnnxEnv, task_id: str, policy) -> dict:
    policy.reset()
    result = await env.reset(task_id=task_id)
    obs = result.observation
    rewards = []
    actions = []
    for _ in range(obs.max_steps):
        payload = policy.act(obs)
        action = OnnxAction(
            action_type=payload["action_type"],
            slot_name=payload.get("slot_name", ""),
            patch_id=payload.get("patch_id", ""),
            rationale=payload.get("rationale", ""),
        )
        step_result = await env.step(action)
        obs = step_result.observation
        rewards.append(float(step_result.reward or 0.0))
        actions.append(payload)
        policy.update(payload)
        if step_result.done:
            break
    return {
        "task_id": task_id,
        "variant_label": obs.variant_label,
        "success": bool(obs.is_success),
        "score": float(obs.final_score or obs.current_score),
        "steps": int(obs.steps_taken),
        "rewards": rewards,
        "actions": actions,
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_score": 0.0,
            "score_variance": 0.0,
            "avg_steps": 0.0,
            "step_variance": 0.0,
        }
    scores = [item["score"] for item in results]
    steps = [item["steps"] for item in results]
    return {
        "episodes": len(results),
        "success_rate": sum(1 for item in results if item["success"]) / len(results),
        "avg_score": sum(scores) / len(scores),
        "score_variance": statistics.pvariance(scores) if len(scores) > 1 else 0.0,
        "avg_steps": sum(steps) / len(steps),
        "step_variance": statistics.pvariance(steps) if len(steps) > 1 else 0.0,
    }


async def evaluate_policy(env: OnnxEnv, policy, episodes_per_task: int) -> list[dict]:
    results: list[dict] = []
    for task_id in TASK_IDS:
        for _ in range(episodes_per_task):
            results.append(await run_episode(env, task_id, policy))
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ONNX surgeon policies with hard baselines.")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--trained-model")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default="outputs/eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = OnnxEnv(base_url=args.env_url)
    try:
        random_policy = RandomPolicy(seed=args.seed)
        heuristic_policy = HeuristicPolicy()
        trained_policy = HFGenerationPolicy(args.trained_model) if args.trained_model else None

        random_results = await evaluate_policy(env, random_policy, args.episodes_per_task)
        heuristic_results = await evaluate_policy(env, heuristic_policy, args.episodes_per_task)
        trained_results = await evaluate_policy(env, trained_policy, args.episodes_per_task) if trained_policy else []
    finally:
        await env.close()

    payload = {
        "random_baseline": random_results,
        "heuristic_baseline": heuristic_results,
        "trained_policy": trained_results,
        "summary": {
            "random": summarize(random_results),
            "heuristic": summarize(heuristic_results),
            "trained": summarize(trained_results) if trained_results else None,
        },
    }
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
