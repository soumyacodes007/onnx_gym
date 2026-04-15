from __future__ import annotations

import argparse
import asyncio
import json

try:
    from client import OnnxEnv
    from models import OnnxAction
    from solver import deterministic_action
except ImportError:
    from onnx_env.client import OnnxEnv
    from onnx_env.models import OnnxAction
    from onnx_env.solver import deterministic_action


async def run_cli(env_url: str, task_id: str) -> None:
    env = OnnxEnv(base_url=env_url)
    seen_patch_ids: set[str] = set()
    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        print(json.dumps({"event": "start", "task_id": task_id, "max_steps": obs.max_steps}, indent=2))
        for step in range(1, obs.max_steps + 1):
            payload = deterministic_action(obs, seen_patch_ids)
            action = OnnxAction(
                action_type=payload["action_type"],
                slot_name=payload.get("slot_name", ""),
                patch_id=payload.get("patch_id", ""),
                rationale=payload.get("rationale", ""),
            )
            result = await env.step(action)
            obs = result.observation
            if action.patch_id:
                seen_patch_ids.add(action.patch_id)
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": step,
                        "action": payload,
                        "reward": float(result.reward or 0.0),
                        "score": float(obs.current_score),
                        "done": bool(result.done),
                    }
                )
            )
            if result.done:
                break
        print(json.dumps({"event": "end", "success": bool(obs.is_success), "final_score": float(obs.final_score or obs.current_score)}, indent=2))
    finally:
        await env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI runner for ONNX deployment repair episodes.")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--task-id", default="label_head_dtype_repair")
    args = parser.parse_args()
    asyncio.run(run_cli(args.env_url, args.task_id))


if __name__ == "__main__":
    main()
