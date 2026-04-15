from __future__ import annotations

from dataclasses import dataclass
from random import Random

try:
    from ..tasks import TASKS
    from .curriculum import CurriculumController
except ImportError:
    from tasks import TASKS
    from server.curriculum import CurriculumController


FAULT_CATALOG: dict[str, tuple[str, ...]] = {
    "label_head_dtype_repair": (
        "force_low_opset",
        "force_static_batch",
        "force_label_dtype_float",
        "force_basic_optim",
    ),
    "embedding_ranker_contract": (
        "force_static_batch",
        "force_static_sequence",
        "force_input_ids_int32",
        "force_basic_optim",
        "inflate_model_size",
    ),
    "vision_resize_mobile": (
        "force_resize_both",
        "force_static_batch",
        "force_basic_optim",
        "inflate_model_size",
    ),
    "npu_gateway_surgery": (
        "force_rank3_input",
        "force_raw_nms",
        "force_wrong_provider_cpu",
        "force_basic_optim",
    ),
    "webnn_static_dynamic_pivot": (
        "force_webnn_mixed_dims",
        "force_wrong_provider_cpu",
        "force_basic_optim",
        "inflate_model_size",
    ),
    "external_data_packaging_failure": (
        "force_inline_weights",
        "force_static_attention_mask",
        "force_basic_optim",
        "inflate_model_size",
    ),
    "broken_quantized_cascade": (
        "force_quant_scale_mismatch",
        "force_quant_low_opset",
        "force_quant_debug_inline",
        "force_basic_optim",
    ),
    "multi_stage_detection_bridge": (
        "force_layout_broken",
        "force_stage_contract_mismatch",
        "force_static_batch",
        "force_basic_optim",
    ),
    "release_candidate_gate": (
        "force_release_low_opset",
        "force_resize_both",
        "force_unsafe_precision",
        "force_static_batch",
        "force_wrong_provider_cpu",
        "force_inline_weights",
        "force_basic_optim",
    ),
}


@dataclass(frozen=True)
class AdversarialPlan:
    task_id: str
    variant_index: int
    incident_brief: str
    judge_persona: str
    incident_id: str
    adversarial_seed: int
    fault_bundle: tuple[str, ...]


class AdversarialDesigner:
    """
    Deterministic adversarial planner with procedural fault bundles.

    Each adversarial episode is identified by (incident_id, seed) and composes
    2-4 faults from a task-specific catalog.
    """

    def __init__(self) -> None:
        self._variant_counter: dict[str, int] = {task_id: 0 for task_id in TASKS}

    def plan(self, curriculum: CurriculumController) -> AdversarialPlan:
        tier = curriculum.current_tier()
        unlocked = list(tier.task_ids)
        weak = curriculum.weak_spots()
        candidates = [task_id for task_id in unlocked if task_id in weak] or unlocked
        candidates.sort(key=lambda task_id: (curriculum.success_rate(task_id), len(curriculum.history.get(task_id, []))))
        task_id = candidates[0]

        variant_index = self._variant_counter[task_id]
        self._variant_counter[task_id] = variant_index + 1

        difficulty = curriculum.difficulty()
        if difficulty < 0.45:
            persona = "junior"
        elif difficulty < 0.75:
            persona = "senior"
        else:
            persona = "principal"

        seed = (curriculum.episode_count + 1) * 1009 + (variant_index + 1) * 917 + sum(ord(ch) for ch in task_id)
        rng = Random(seed)
        pool = list(FAULT_CATALOG.get(task_id, ("force_basic_optim",)))
        max_faults = min(4, len(pool))
        count = 2 if max_faults <= 2 else rng.randint(2, max_faults)
        rng.shuffle(pool)
        selected = tuple(pool[:count])

        incident_id = f"{task_id}-adv-{curriculum.episode_count + 1:04d}-{variant_index:02d}"
        brief = (
            f"Adversarial incident {incident_id}: composed {len(selected)} deterministic faults "
            f"for task '{task_id}' under tier '{tier.name}'. "
            f"Difficulty={difficulty:.2f}. Judge persona={persona}."
        )
        return AdversarialPlan(
            task_id=task_id,
            variant_index=variant_index,
            incident_brief=brief,
            judge_persona=persona,
            incident_id=incident_id,
            adversarial_seed=seed,
            fault_bundle=selected,
        )
