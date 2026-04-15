from __future__ import annotations

from dataclasses import dataclass

try:
    from ..tasks import TASKS
    from .adversarial_designer import AdversarialDesigner, AdversarialPlan
    from .curriculum import CurriculumController
except ImportError:
    from tasks import TASKS
    from server.adversarial_designer import AdversarialDesigner, AdversarialPlan
    from server.curriculum import CurriculumController


TRAIN_TASK_IDS = (
    "label_head_dtype_repair",
    "embedding_ranker_contract",
    "vision_resize_mobile",
    "npu_gateway_surgery",
    "webnn_static_dynamic_pivot",
    "external_data_packaging_failure",
    "broken_quantized_cascade",
    "multi_stage_detection_bridge",
    "release_candidate_gate",
)

EVAL_TASK_IDS = TRAIN_TASK_IDS


@dataclass(frozen=True)
class EpisodePlan:
    task_id: str
    variant_index: int
    split: str
    mode: str = "standard"
    incident_brief: str = ""
    judge_persona: str = "junior"
    incident_id: str = ""
    adversarial_seed: int = 0
    fault_bundle: tuple[str, ...] = ()


class EpisodeGenerator:
    def __init__(self) -> None:
        self._split_counters = {"train": 0, "eval": 0}
        self._variant_counters = {task_id: 0 for task_id in TASKS}
        self._adversarial = AdversarialDesigner()

    def next_plan(self, curriculum: CurriculumController, split: str = "train", task_id: str | None = None) -> EpisodePlan:
        task_pool = TRAIN_TASK_IDS if split == "train" else EVAL_TASK_IDS
        if task_id is not None:
            variant_index = self._variant_counters[task_id]
            self._variant_counters[task_id] = variant_index + 1
            return EpisodePlan(
                task_id=task_id,
                variant_index=variant_index,
                split=split,
                mode="manual",
                incident_brief="Manually selected task episode.",
                judge_persona=curriculum.judge_persona(),
                incident_id=f"{task_id}-manual-{variant_index:02d}",
            )

        if split == "train" and curriculum.should_use_adversarial():
            adv: AdversarialPlan = self._adversarial.plan(curriculum)
            self._variant_counters[adv.task_id] = adv.variant_index + 1
            return EpisodePlan(
                task_id=adv.task_id,
                variant_index=adv.variant_index,
                split=split,
                mode="adversarial",
                incident_brief=adv.incident_brief,
                judge_persona=adv.judge_persona,
                incident_id=adv.incident_id,
                adversarial_seed=adv.adversarial_seed,
                fault_bundle=adv.fault_bundle,
            )

        if split == "train":
            chosen = curriculum.next_task_id()
        else:
            idx = self._split_counters[split] % len(task_pool)
            chosen = task_pool[idx]
            self._split_counters[split] += 1

        variant_index = self._variant_counters[chosen]
        self._variant_counters[chosen] = variant_index + 1
        return EpisodePlan(
            task_id=chosen,
            variant_index=variant_index,
            split=split,
            mode="standard",
            incident_brief="Curriculum-selected standard episode.",
            judge_persona=curriculum.judge_persona(),
            incident_id=f"{chosen}-std-{variant_index:02d}",
        )
