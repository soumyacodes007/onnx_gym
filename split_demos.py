from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict], train_ratio: float) -> tuple[list[dict], list[dict]]:
    episodes_by_task: dict[str, list[int]] = defaultdict(list)
    rows_by_episode: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        episode_id = int(row.get("episode_id", index))
        task_id = str(row.get("task_id", "unknown"))
        rows_by_episode[episode_id].append(row)
        if episode_id not in episodes_by_task[task_id]:
            episodes_by_task[task_id].append(episode_id)

    train_episode_ids: set[int] = set()
    eval_episode_ids: set[int] = set()
    for episode_ids in episodes_by_task.values():
        episode_ids = sorted(episode_ids)
        split_idx = max(1, int(len(episode_ids) * train_ratio))
        if split_idx >= len(episode_ids):
            split_idx = max(1, len(episode_ids) - 1)
        train_episode_ids.update(episode_ids[:split_idx])
        eval_episode_ids.update(episode_ids[split_idx:])
        if not eval_episode_ids and episode_ids:
            eval_episode_ids.add(episode_ids[-1])

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for episode_id, episode_rows in rows_by_episode.items():
        if episode_id in train_episode_ids:
            train_rows.extend(episode_rows)
        else:
            eval_rows.extend(episode_rows)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split ONNX demo JSONL into train and eval sets.")
    parser.add_argument("--input", required=True, help="Path to demo JSONL file")
    parser.add_argument("--train-out", default="outputs/train.jsonl")
    parser.add_argument("--eval-out", default="outputs/eval.jsonl")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    if len(rows) < 10:
        raise SystemExit("Not enough rows to split. Generate more demos.")
    train_rows, eval_rows = split_rows(rows, args.train_ratio)
    if not eval_rows:
        raise SystemExit("Eval split is empty. Reduce train-ratio or generate more demos.")
    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)


if __name__ == "__main__":
    main()
