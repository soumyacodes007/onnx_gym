from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

try:
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from trl.experimental.openenv import generate_rollout_completions
except ImportError as exc:
    raise SystemExit("Missing GRPO dependencies. Install with: pip install -e '.[train]'") from exc

try:
    from client import OnnxEnv
    from models import OnnxAction
    from solver import deterministic_action
except ImportError:
    from onnx_env.client import OnnxEnv
    from onnx_env.models import OnnxAction
    from onnx_env.solver import deterministic_action


SYSTEM_PROMPT = (
    "You are an ONNX deployment surgeon. "
    "Follow workflow: inspect -> patch -> validate -> submit. "
    "Return exactly one JSON action object per turn."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for ONNX Deployment Surgeon Gym.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--dataset-size", type=int, default=60)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--reward-log", default="grpo_rewards.csv")
    return parser.parse_args()


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_prompt(observation) -> str:
    return (
        f"Task: {observation.task_id}\n"
        f"Title: {observation.task_title}\n"
        f"Profile: {observation.deployment_profile}\n"
        f"Mode: {observation.episode_mode}\n"
        f"Judge persona: {observation.judge_persona}\n"
        f"Incident brief: {observation.incident_brief}\n"
        f"Visible issues: {json.dumps(observation.visible_issues)}\n"
        f"Top blockers: {json.dumps(observation.top_blockers)}\n"
        f"Possible actions: {json.dumps(observation.possible_actions)}\n"
        f"Current score: {observation.current_score:.2f}\n"
        "Return exactly one JSON object with keys action_type, slot_name, patch_id, rationale."
    )


def parse_action_payload(text: str, observation, seen_patch_ids: set[str]) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        payload = json.loads(text[start:end])
    except Exception:
        payload = deterministic_action(observation, seen_patch_ids)
    payload.setdefault("action_type", "validate_bundle")
    payload.setdefault("slot_name", "")
    payload.setdefault("patch_id", "")
    payload.setdefault("rationale", "")
    return payload


def rollout_once(trainer: GRPOTrainer, env: OnnxEnv, tokenizer: AutoTokenizer, max_turns: int) -> dict[str, object]:
    result = asyncio.run(env.reset())
    observation = result.observation
    seen_patch_ids: set[str] = set()

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    action_trace: list[dict] = []

    for _ in range(max_turns):
        if result.done:
            break
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation)},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)
        rollout_output = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_output["prompt_ids"])
        completion_ids.extend(rollout_output["completion_ids"])
        logprobs.extend(rollout_output["logprobs"])

        completion_text = rollout_output.get("text") or tokenizer.decode(rollout_output["completion_ids"], skip_special_tokens=True)
        payload = parse_action_payload(completion_text, observation, seen_patch_ids)
        action = OnnxAction(
            action_type=payload["action_type"],
            slot_name=payload.get("slot_name", ""),
            patch_id=payload.get("patch_id", ""),
            rationale=payload.get("rationale", ""),
        )
        result = asyncio.run(env.step(action))
        observation = result.observation
        reward = float(result.reward or 0.0)
        step_rewards.append(reward)
        if action.patch_id:
            seen_patch_ids.add(action.patch_id)
        action_trace.append(
            {
                "action_type": action.action_type,
                "slot_name": action.slot_name,
                "patch_id": action.patch_id,
                "reward": reward,
                "score_after": float(observation.current_score),
            }
        )
        if result.done:
            break

    total_reward = sum(step_rewards)
    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "final_score": float(observation.final_score or observation.current_score),
        "action_trace": action_trace,
        "success": bool(observation.is_success),
    }


def reward_total(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("total_reward")
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def reward_score(completions: list[str], **kwargs) -> list[float]:
    rewards = kwargs.get("final_score")
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"prompt": ["Repair this ONNX deployment incident."] * args.dataset_size})

    out_dir = Path(args.output_dir or Path("outputs") / f"onnx-grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_csv = out_dir / args.reward_log
    with reward_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "total_reward", "final_score", "success", "timestamp"])

    env = OnnxEnv(base_url=args.env_url)
    episode_counter = [0]

    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode="colocate",
        output_dir=str(out_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        total_rewards: list[float] = []
        final_scores: list[float] = []

        for _ in prompts:
            episode = rollout_once(trainer, env, tokenizer, args.max_turns)
            episode_counter[0] += 1
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            total_rewards.append(float(episode["total_reward"]))
            final_scores.append(float(episode["final_score"]))
            with reward_csv.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        episode_counter[0],
                        float(episode["total_reward"]),
                        float(episode["final_score"]),
                        str(bool(episode["success"])).lower(),
                        datetime.now().isoformat(),
                    ]
                )
        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "total_reward": total_rewards,
            "final_score": final_scores,
        }

    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_total, reward_score],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    try:
        trainer.train()
    finally:
        asyncio.run(env.close())

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


if __name__ == "__main__":
    main()
