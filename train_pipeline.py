from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from split_demos import split_rows, write_jsonl


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end ONNX Surgeon training pipeline.")
    parser.add_argument("--env-url", default="http://localhost:7860")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--work-dir", default="outputs/training")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    demos_path = work_dir / "demos.jsonl"

    _run(
        [
            sys.executable,
            "generate_demos.py",
            "--env-url",
            args.env_url,
            "--episodes",
            str(args.episodes),
            "--out",
            str(demos_path),
        ]
    )

    rows = _load_jsonl(demos_path)
    if len(rows) < 10:
        raise SystemExit("Not enough demonstration rows. Increase --episodes.")
    train_rows, eval_rows = split_rows(rows, args.train_ratio)
    if not eval_rows:
        raise SystemExit("Eval split is empty. Reduce --train-ratio or increase --episodes.")

    train_path = work_dir / "train.jsonl"
    eval_path = work_dir / "eval.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(eval_path, eval_rows)

    model_out_dir = work_dir / "sft-model"
    if not args.skip_train:
        _run(
            [
                sys.executable,
                "train.py",
                "--train-file",
                str(train_path),
                "--eval-file",
                str(eval_path),
                "--model-id",
                args.model_id,
                "--output-dir",
                str(model_out_dir),
                "--epochs",
                str(args.epochs),
                "--learning-rate",
                str(args.learning_rate),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--max-length",
                str(args.max_length),
            ]
        )

    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            "eval.py",
            "--env-url",
            args.env_url,
            "--out-dir",
            str(work_dir / "eval"),
        ]
        if model_out_dir.exists():
            eval_cmd.extend(["--trained-model", str(model_out_dir)])
        _run(eval_cmd)

    summary = {
        "demos_path": str(demos_path),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "model_dir": str(model_out_dir),
        "eval_dir": str(work_dir / "eval"),
    }
    (work_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
