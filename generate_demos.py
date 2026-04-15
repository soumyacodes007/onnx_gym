from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

try:
    from client import OnnxEnv
    from models import OnnxAction
    from solver import deterministic_action
except ImportError:
    from onnx_env.client import OnnxEnv
    from onnx_env.models import OnnxAction
    from onnx_env.solver import deterministic_action


DEFAULT_TASK_IDS = [
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


def build_prompt(observation) -> str:
    return (
        f"Task: {observation.task_title}\n"
        f"Description: {observation.task_description}\n"
        f"Product brief: {observation.product_brief}\n"
        f"Profile: {observation.deployment_profile}\n"
        f"Variant: {observation.variant_label}\n"
        f"Visible issues: {json.dumps(observation.visible_issues)}\n"
        f"Top blockers: {json.dumps(observation.top_blockers)}\n"
        f"Possible actions: {json.dumps(observation.possible_actions)}"
    )


async def run_episode(env: OnnxEnv, task_id: str) -> dict:
    result = await env.reset(task_id=task_id)
    obs = result.observation
    seen_patch_ids: set[str] = set()
    transcript = []
    for _ in range(obs.max_steps):
        action_payload = deterministic_action(obs, seen_patch_ids)
        action = OnnxAction(
            action_type=action_payload["action_type"],
            slot_name=action_payload.get("slot_name", ""),
            patch_id=action_payload.get("patch_id", ""),
            rationale=action_payload.get("rationale", ""),
        )
        step_result = await env.step(action)
        next_obs = step_result.observation
        transcript.append(
            {
                "prompt": build_prompt(obs),
                "response": json.dumps(action_payload, separators=(",", ":")),
                "task_id": obs.task_id,
                "variant_label": obs.variant_label,
                "reward": float(step_result.reward or 0.0),
                "done": bool(step_result.done),
                "score_after": float(next_obs.current_score),
            }
        )
        if action.patch_id:
            seen_patch_ids.add(action.patch_id)
        obs = next_obs
        if step_result.done:
            break
    return {
        "task_id": obs.task_id,
        "variant_label": obs.variant_label,
        "success": obs.is_success,
        "final_score": obs.final_score or obs.current_score,
        "steps": obs.steps_taken,
        "transcript": transcript,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect deterministic ONNX repair demonstrations.")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--out", default="outputs/demo_train.jsonl")
    parser.add_argument("--task-ids", nargs="*", default=DEFAULT_TASK_IDS)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = OnnxEnv(base_url=args.env_url)
    records = []
    try:
        for index in range(args.episodes):
            task_id = args.task_ids[index % len(args.task_ids)]
            episode = await run_episode(env, task_id)
            records.append(episode)
    finally:
        await env.close()

    with out_path.open("w", encoding="utf-8") as handle:
        for episode_index, episode in enumerate(records, start=1):
            for turn in episode["transcript"]:
                turn["episode_id"] = episode_index
                handle.write(json.dumps(turn) + "\n")

    reward_csv = out_path.with_suffix(".rewards.csv")
    with reward_csv.open("w", encoding="utf-8") as handle:
        handle.write("episode,task_id,reward,success,steps\n")
        for index, episode in enumerate(records, start=1):
            reward_total = sum(float(turn["reward"]) for turn in episode["transcript"])
            handle.write(f"{index},{episode['task_id']},{reward_total:.4f},{str(bool(episode['success'])).lower()},{episode['steps']}\n")

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "episodes": len(records),
        "avg_score": sum(float(item["final_score"]) for item in records) / max(len(records), 1),
        "success_rate": sum(1 for item in records if item["success"]) / max(len(records), 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
