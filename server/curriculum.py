from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class CurriculumTier:
    name: str
    task_ids: tuple[str, ...]
    min_episodes: int
    advance_rate: float
    max_difficulty: float


TIERS: tuple[CurriculumTier, ...] = (
    CurriculumTier(
        name="warmup",
        task_ids=(
            "label_head_dtype_repair",
            "embedding_ranker_contract",
            "vision_resize_mobile",
        ),
        min_episodes=6,
        advance_rate=0.70,
        max_difficulty=0.35,
    ),
    CurriculumTier(
        name="runtime",
        task_ids=(
            "npu_gateway_surgery",
            "webnn_static_dynamic_pivot",
            "external_data_packaging_failure",
        ),
        min_episodes=8,
        advance_rate=0.72,
        max_difficulty=0.72,
    ),
    CurriculumTier(
        name="compound",
        task_ids=(
            "broken_quantized_cascade",
            "multi_stage_detection_bridge",
            "release_candidate_gate",
        ),
        min_episodes=10,
        advance_rate=0.78,
        max_difficulty=0.95,
    ),
)

MASTERY_THRESHOLD = 0.82
MASTERY_WINDOW = 6


class CurriculumController:
    def __init__(self) -> None:
        self.history: dict[str, list[bool]] = defaultdict(list)
        self.rewards: dict[str, list[float]] = defaultdict(list)
        self.steps: dict[str, list[int]] = defaultdict(list)
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0
        self._force_adversarial = os.environ.get("GYM_MODE", "standard").lower() == "adversarial"

    def record(self, task_id: str, success: bool, reward: float, steps: int) -> None:
        self.history[task_id].append(success)
        self.rewards[task_id].append(reward)
        self.steps[task_id].append(steps)
        self.episode_count += 1
        self._tier_episodes += 1
        self._maybe_advance()

    def current_tier(self) -> CurriculumTier:
        return TIERS[self._tier_index]

    def difficulty(self) -> float:
        tier = self.current_tier()
        floor = 0.12 if self._tier_index == 0 else TIERS[self._tier_index - 1].max_difficulty
        return round(min(tier.max_difficulty, floor + self.recent_success_rate() * (tier.max_difficulty - floor)), 3)

    def judge_persona(self) -> str:
        d = self.difficulty()
        if d < 0.4:
            return "junior"
        if d < 0.75:
            return "senior"
        return "principal"

    def weak_spots(self) -> list[str]:
        tier = self.current_tier()
        weak = [task_id for task_id in tier.task_ids if self.success_rate(task_id) < MASTERY_THRESHOLD]
        return sorted(weak, key=lambda task_id: (self.success_rate(task_id), len(self.history.get(task_id, []))))

    def should_use_adversarial(self) -> bool:
        if self._force_adversarial:
            return True
        return self.difficulty() >= 0.65 and len(self.weak_spots()) > 0

    def next_task_id(self) -> str:
        tier = self.current_tier()
        unlocked = list(tier.task_ids)
        untried = [task_id for task_id in unlocked if not self.history.get(task_id)]
        if untried:
            return untried[0]

        weak = self.weak_spots()
        if weak:
            return weak[0]

        unlocked.sort(key=lambda task_id: (len(self.history.get(task_id, [])), self.average_steps(task_id)))
        return unlocked[0]

    def success_rate(self, task_id: str) -> float:
        history = self.history.get(task_id, [])
        if not history:
            return 0.0
        window = history[-MASTERY_WINDOW:]
        return sum(window) / len(window)

    def average_reward(self, task_id: str) -> float:
        rewards = self.rewards.get(task_id, [])
        if not rewards:
            return 0.0
        window = rewards[-MASTERY_WINDOW:]
        return sum(window) / len(window)

    def average_steps(self, task_id: str) -> float:
        steps = self.steps.get(task_id, [])
        if not steps:
            return 0.0
        window = steps[-MASTERY_WINDOW:]
        return sum(window) / len(window)

    def recent_success_rate(self, window: int = 12) -> float:
        merged: list[bool] = []
        for task_id in self.current_tier().task_ids:
            merged.extend(self.history.get(task_id, [])[-window:])
        if not merged:
            return 0.0
        return sum(merged) / len(merged)

    def stats(self) -> dict[str, object]:
        tier = self.current_tier()
        return {
            "episode_count": self.episode_count,
            "tier": tier.name,
            "tier_episodes": self._tier_episodes,
            "difficulty": self.difficulty(),
            "judge_persona": self.judge_persona(),
            "use_adversarial": self.should_use_adversarial(),
            "unlocked_tasks": list(tier.task_ids),
            "weak_spots": self.weak_spots(),
            "success_rates": {
                task_id: round(self.success_rate(task_id), 2) for task_id in tier.task_ids if self.history.get(task_id)
            },
            "avg_rewards": {
                task_id: round(self.average_reward(task_id), 2) for task_id in tier.task_ids if self.rewards.get(task_id)
            },
            "avg_steps": {
                task_id: round(self.average_steps(task_id), 2) for task_id in tier.task_ids if self.steps.get(task_id)
            },
        }

    def _maybe_advance(self) -> None:
        if self._tier_index >= len(TIERS) - 1:
            return
        tier = self.current_tier()
        rate = self.recent_success_rate()
        fast_track = self._tier_episodes >= 3 and rate >= 0.9
        if not fast_track and self._tier_episodes < tier.min_episodes:
            return
        if rate >= tier.advance_rate:
            self._tier_index += 1
            self._tier_episodes = 0
