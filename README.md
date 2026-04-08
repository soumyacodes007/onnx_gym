---
title: ONNX Deployment Surgeon Gym
emoji: "🩺"
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
  - model-export
  - reinforcement-learning
---

# ONNX Deployment Surgeon Gym

An OpenEnv environment where an agent acts as an ONNX deployment engineer and repairs broken ONNX export bundles so they pass model checking, shape inference, and ONNX Runtime loading under real hardware profiles.

This is a repair workflow, not a toy benchmark: the agent sees broken deployment artifacts, runtime-style diagnostics, memory budgets, profile constraints, and cascading export issues that appear only after earlier fixes land.

## Why this is strong

- **Real-world utility**: teams constantly debug ONNX export bundles, shape contracts, provider mismatches, and mobile memory budgets.
- **Grounded in official tooling**: validation uses real `onnx.checker`, real `onnx.shape_inference`, and real ONNX Runtime CPU session creation.
- **Safe runtime**: tiny synthetic graphs only; no heavy models, no remote downloads, no GPU assumptions.
- **Novelity**: this is graph and deployment surgery, not another generic coding or SQL environment.
- **Learnable but deep**: severity-weighted grading, visible vs hidden issues, ranked repair actions, and deterministic profile variants create genuine multi-step progress.

Official references:
- [ONNX checker and shape inference docs](https://onnx.ai/onnx/repo-docs/ShapeInference.html)
- [ONNX Runtime graph optimization levels](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

## Task curriculum

### 1. Label Head DType Repair
Fix a classifier export where ArgMax labels are declared with the wrong output type, the batch dimension is frozen, and the deployment variant changes between Android/iOS review contexts.

### 2. Embedding Ranker Contract
Repair a retrieval export so token IDs use int64, sequence dimensions are symbolic, and ORT runtime settings match the serving profile for reranker and search-sidecar variants.

### 3. Vision Resize Mobile
Repair a mobile vision export with an invalid Resize signature, static batch dimensions, and a tight memory budget under different mobile delivery variants.

Each task family cycles through deterministic variants, so the same task id can surface slightly different hardware labels, budgets, and deployment contexts while staying validator-safe.

## Action space

- `inspect_task`
- `inspect_bundle`
- `inspect_patches`
- `inspect_report`
- `apply_patch`
- `validate_bundle`
- `submit_final`

`apply_patch` uses a structured graph-surgery patch catalog so the environment stays deterministic and validator-safe while still exposing meaningful multi-step reasoning.

## Observation depth

The agent sees:

- graph config and IO contract
- deployment profile and target execution provider
- profile summary and deterministic variant label
- checker / shape inference / ORT session results
- visible vs hidden issues
- conflict map and patch dependency graph
- dynamic-dimension counts
- node count, initializer count, and estimated activation footprint
- estimated model size vs memory budget
- raw ONNX / ORT style error logs
- top blockers and "why not perfect yet" summary
- ranked possible actions

This makes the environment easy to debug in `/web` while still being rich enough for RL-style reward shaping.

## Reward design

Final and intermediate scores stay in `[0, 1]` and combine:

- severity-weighted requirement resolution
- endpoint coverage across checker, shape inference, and ORT loading
- efficiency bonus for solving with fewer steps
- small penalties for profile conflicts that would still break deployment

That gives dense signal without making the task gameable.

## Why judges should care

This environment targets a real deployment engineering workflow:

- exported graph is syntactically valid
- inferred shapes match serving expectations
- ORT can actually create a session on the target provider
- memory/profile constraints are respected

That is exactly the kind of task an edge-agent or export-repair assistant should be trained on.

## Local run

```bash
uv sync --frozen --no-dev
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open:
- `http://localhost:7860/web`

## Inference script

The root `inference.py`:
- uses the OpenAI client,
- reads `HF_TOKEN`, `API_BASE_URL`, `MODEL_NAME`, and optional `ENV_BASE_URL` / `LOCAL_IMAGE_NAME`,
- emits strict `[START]`, `[STEP]`, and `[END]` logs.

Expected environment variables:

```bash
HF_TOKEN=...
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

## Docker

```bash
docker build -t onnx-surgeon .
docker run -p 7860:7860 onnx-surgeon
```

## Hugging Face Spaces

Recommended secrets:
- `HF_TOKEN`
- `API_BASE_URL`
- `MODEL_NAME`

## Validation checklist

- `openenv validate`
- `python -m pytest`
- local `/web` smoke test
- local `python inference.py`
