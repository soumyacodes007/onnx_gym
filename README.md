---
title: ONNX Deployment Surgeon Gym
emoji: "tool"
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - onnx
  - onnxruntime
  - edge-ai
  - reinforcement-learning
---

# ONNX Deployment Surgeon Gym

An OpenEnv environment where an agent repairs broken ONNX deployment bundles using real checker, shape inference, and ONNX Runtime validation under realistic edge deployment profiles.

## Why this environment is strong

- Real-world utility: this mirrors real export-debug and deployment triage workflows.
- Deterministic and safe: tiny synthetic graphs, CPU-only, no large model downloads.
- Rich RL signal: cascading failures, profile constraints, dependency-aware patches, workflow shaping rewards.
- Phase-aware judge loop: rewards proper engineering flow (`triage -> investigate -> repair -> verify -> submit`).
- Adversarial curriculum mode: weak-spot targeting with dynamic task/persona scheduling for harder training episodes.
- Spec-focused: typed models, clean endpoints, web interface, Docker-first deployment.

## 9-task curriculum

1. `label_head_dtype_repair`
2. `embedding_ranker_contract`
3. `vision_resize_mobile`
4. `npu_gateway_surgery`
5. `webnn_static_dynamic_pivot`
6. `external_data_packaging_failure`
7. `broken_quantized_cascade`
8. `multi_stage_detection_bridge`
9. `release_candidate_gate`

The tasks are grouped into warmup, runtime, and compound tiers in `server/curriculum.py`.

## Finals features (kube-sre inspired)

- Curriculum controller with tier progression, weak-spot targeting, and dynamic difficulty scalar.
- Adversarial designer (`server/adversarial_designer.py`) that selects hardest weak tasks and strict judge personas.
- Step judge (`server/judge.py`) with workflow-order rewards/penalties and repeat-action penalties.
- Episode transcript logging with mode/persona/workflow metadata to `outputs/onnx_episode_transcripts.jsonl`.
- Procedural adversarial composition: each adversarial episode composes a deterministic 2-4 fault bundle with `incident_id` and `adversarial_seed`.

## Reward semantics

- `final_score` is clamped to `[0,1]`.
- per-step `reward` is signed (can be negative) and clamped to `[-1,1]`.
- this keeps evaluator-compliant final scoring while preserving dense RL gradients.

## Action space

- `inspect_task`
- `inspect_bundle`
- `inspect_patches`
- `inspect_report`
- `apply_patch`
- `validate_bundle`
- `submit_final`

## Environment variables for inference

Mandatory variables required by the benchmark setup:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `API_KEY` (preferred if available; script falls back to `HF_TOKEN`)
- `ENV_BASE_URL` (default `http://localhost:7860`)
- `LOCAL_IMAGE_NAME` (if running environment via `from_docker_image()`)

`inference.py` is in the project root and uses OpenAI Client for LLM calls.

## Local quickstart

```bash
uv sync --frozen --no-dev
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open web UI at `http://localhost:7860/web`.

Adversarial mode:

```bash
set GYM_MODE=adversarial
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t onnx-surgeon .
docker run -p 7860:7860 onnx-surgeon
```

The Docker image includes:
- Space runtime (`server/*`, `inference.py`)
- Trainer utilities (`train.py`, `grpo_train.py`, `eval.py`, `generate_demos.py`, `split_demos.py`, `train_pipeline.py`)

Note: runtime image installs non-dev dependencies by default. For GRPO training dependencies (`trl`, `peft`), use trainer runtime setup below.

## Validation

```bash
openenv validate
python -m pytest
python inference.py
```

## Training workflow

### Step 1: Generate demonstrations

```bash
python generate_demos.py --env-url http://localhost:7860 --episodes 180 --out outputs/demo_train.jsonl
```

### Step 2: Train SFT policy

```bash
python split_demos.py --input outputs/demo_train.jsonl --train-out outputs/train.jsonl --eval-out outputs/eval.jsonl

python train.py \
  --train-file outputs/train.jsonl \
  --eval-file outputs/eval.jsonl \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/onnx-sft
```

### Step 3: Evaluate baseline vs trained model

```bash
python eval.py --env-url http://localhost:7860 --trained-model outputs/onnx-sft --out-dir outputs/eval
```

### One-command pipeline (recommended)

`train_pipeline.py` runs generation, split, train, and eval end to end.

```bash
python train_pipeline.py \
  --env-url http://localhost:7860 \
  --episodes 180 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 1
```

## GRPO training (round-2 finals)

`grpo_train.py` performs online rollouts against the live environment and trains with TRL GRPO.

```bash
pip install -e ".[train]"
set GYM_MODE=adversarial

# terminal 1
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# terminal 2
python grpo_train.py \
  --env-url http://localhost:7860 \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-size 60 \
  --num-generations 4 \
  --max-turns 12
```

## Colab run order

1. Start the env server in one terminal/session.
2. Run:

```bash
pip install -U datasets transformers accelerate matplotlib trl peft
python train_pipeline.py --env-url http://localhost:7860 --episodes 180 --epochs 1
```

Artifacts:

- `outputs/training/demos.jsonl`
- `outputs/training/train.jsonl`
- `outputs/training/eval.jsonl`
- `outputs/training/sft-model/`
- `outputs/training/eval/eval_results.json`
- `outputs/training/pipeline_summary.json`
