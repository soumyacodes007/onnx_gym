from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_csv(csv_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, rewards = [], []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    if not episodes:
        raise ValueError("No reward rows found in CSV.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(episodes, rewards, marker="o", alpha=0.45, label="episode reward")
    window = min(10, len(rewards))
    rolling = []
    for index in range(len(rewards)):
        values = rewards[max(0, index - window + 1) : index + 1]
        rolling.append(sum(values) / len(values))
    plt.plot(episodes, rolling, linewidth=2.2, label=f"rolling avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("ONNX Surgeon Reward Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ONNX reward curves.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot_csv(Path(args.csv), Path(args.out))


if __name__ == "__main__":
    main()
