from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


PROFILE_SPECS = {
    "mobile_classifier": {
        "memory_budget_mb": 96.0,
        "target_ep": "CPUExecutionProvider",
        "priority": "latency_and_compatibility",
        "required_dynamic_dims": 1,
        "hardware": "mid-range ARM mobile CPU",
    },
    "retrieval_ranker": {
        "memory_budget_mb": 128.0,
        "target_ep": "CPUExecutionProvider",
        "priority": "shape_stability",
        "required_dynamic_dims": 2,
        "hardware": "2-vCPU retrieval worker",
    },
    "vision_mobile": {
        "memory_budget_mb": 160.0,
        "target_ep": "CPUExecutionProvider",
        "priority": "mobile_readiness",
        "required_dynamic_dims": 1,
        "hardware": "memory-constrained mobile vision target",
    },
}


@dataclass(frozen=True)
class PatchOption:
    patch_id: str
    slot_name: str
    label: str
    description: str
    effect: dict[str, Any]
    resolves: tuple[str, ...] = ()
    unlocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementSpec:
    key: str
    description: str
    weight: float
    severity: float
    issue_id: str
    initial_visible: bool = True
    depends_on: tuple[str, ...] = ()
    error_hint: str = ""


@dataclass(frozen=True)
class VariantSpec:
    label: str
    bundle_updates: dict[str, Any]


@dataclass(frozen=True)
class OnnxTask:
    task_id: str
    title: str
    difficulty: str
    description: str
    product_brief: str
    deployment_profile: str
    max_steps: int
    success_threshold: float
    bundle: dict[str, Any]
    patches: list[PatchOption]
    requirement_specs: list[RequirementSpec]
    variants: list[VariantSpec]

    def fresh_bundle(self, variant_index: int = 0) -> dict[str, Any]:
        bundle = deepcopy(self.bundle)
        variant = self.variants[variant_index % len(self.variants)] if self.variants else VariantSpec("default", {})
        bundle["variant_label"] = variant.label
        for section, values in variant.bundle_updates.items():
            if section == "variant_label":
                bundle["variant_label"] = values
                continue
            bundle.setdefault(section, {})
            bundle[section].update(deepcopy(values))
        return bundle


TASKS: dict[str, OnnxTask] = {}
PATCH_LOOKUP: dict[str, dict[str, PatchOption]] = {}
PATCH_CATALOG: dict[str, dict[str, list[dict[str, Any]]]] = {}
PATCH_DEPENDENCY_GRAPHS: dict[str, dict[str, list[str]]] = {}


def _patch_dict(patches: list[PatchOption]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for patch in patches:
        result.setdefault(patch.slot_name, []).append(
            {
                "patch_id": patch.patch_id,
                "label": patch.label,
                "description": patch.description,
                "resolves": list(patch.resolves),
                "unlocks": list(patch.unlocks),
            }
        )
    return result


def _dependency_graph(patches: list[PatchOption]) -> dict[str, list[str]]:
    return {patch.patch_id: list(patch.unlocks) for patch in patches if patch.unlocks}


def _register(task: OnnxTask) -> None:
    TASKS[task.task_id] = task
    PATCH_LOOKUP[task.task_id] = {patch.patch_id: patch for patch in task.patches}
    PATCH_CATALOG[task.task_id] = _patch_dict(task.patches)
    PATCH_DEPENDENCY_GRAPHS[task.task_id] = _dependency_graph(task.patches)


_register(
    OnnxTask(
        task_id="label_head_dtype_repair",
        title="Label Head DType Repair",
        difficulty="easy",
        description="Repair an ONNX classifier export so checker, shape inference, and ORT loading all succeed for a mobile classifier.",
        product_brief="The exported classifier must use opset 17, expose a dynamic batch dimension, and declare ArgMax labels as int64.",
        deployment_profile="mobile_classifier",
        max_steps=8,
        success_threshold=0.95,
        bundle={
            "graph_config": {
                "opset": 13,
                "dynamic_batch": False,
                "label_output_dtype": "float",
                "provider": "CPUExecutionProvider",
                "optimization_level": "ORT_ENABLE_BASIC",
                "estimated_model_mb": 54.0,
            },
            "io_contract": {
                "input_name": "features",
                "input_dtype": "float",
                "input_shape": [1, 4],
                "output_name": "labels",
                "output_dtype": "float",
                "output_shape": [1],
            },
            "deployment": {
                "target_ep": "CPUExecutionProvider",
                "target_device": "mobile-cpu",
            },
        },
        patches=[
            PatchOption(
                patch_id="set_opset_17",
                slot_name="graph_config",
                label="Raise opset to 17",
                description="Use a runtime-supported opset for the deployment contract.",
                effect={"graph_config": {"opset": 17}},
                resolves=("low_opset",),
            ),
            PatchOption(
                patch_id="set_dynamic_batch",
                slot_name="graph_config",
                label="Enable dynamic batch",
                description="Expose a symbolic batch dimension for serving.",
                effect={"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 4], "output_shape": ["batch"]}},
                resolves=("static_batch_only",),
            ),
            PatchOption(
                patch_id="set_label_output_int64",
                slot_name="io_contract",
                label="Declare labels as int64",
                description="Match ArgMax output type in ONNX.",
                effect={"graph_config": {"label_output_dtype": "int64"}, "io_contract": {"output_dtype": "int64"}},
                resolves=("label_dtype_mismatch",),
                unlocks=("needs_extended_optim",),
            ),
            PatchOption(
                patch_id="set_extended_optim",
                slot_name="graph_config",
                label="Use extended graph optimization",
                description="Move to ORT extended graph optimizations for production parity.",
                effect={"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}},
                resolves=("needs_extended_optim",),
            ),
            PatchOption(
                patch_id="switch_cuda_provider",
                slot_name="deployment",
                label="Switch to CUDA provider",
                description="Not valid in this CPU-only target profile.",
                effect={"deployment": {"target_ep": "CUDAExecutionProvider"}},
            ),
        ],
        requirement_specs=[
            RequirementSpec("checker_passed", "ONNX checker must pass", 0.30, 1.00, "label_dtype_mismatch", True, (), "onnx.checker.ValidationError: Graph output type does not match ArgMax(int64) output"),
            RequirementSpec("dynamic_batch", "Expose dynamic batch dimension", 0.20, 0.75, "static_batch_only", True, (), "shape inference warning: exported batch dimension is fixed to 1"),
            RequirementSpec("opset_valid", "Use opset 17 or newer", 0.15, 0.55, "low_opset", True, (), "ORT loader warning: opset below deployment policy target"),
            RequirementSpec("ort_session_passed", "ORT CPU session must load", 0.20, 0.85, "needs_extended_optim", False, ("label_dtype_mismatch",), "onnxruntime: graph optimization contract not met for deployment bundle"),
            RequirementSpec("target_provider", "Use CPU execution provider", 0.15, 0.45, "wrong_provider", True, (), "onnxruntime: requested provider is unavailable on target hardware"),
        ],
        variants=[
            VariantSpec("android-batch-audit", {"deployment": {"target_device": "android-midrange"}, "graph_config": {"estimated_model_mb": 54.0}}),
            VariantSpec("ios-store-check", {"deployment": {"target_device": "ios-coreml-gateway"}, "graph_config": {"estimated_model_mb": 58.0}}),
        ],
    )
)

_register(
    OnnxTask(
        task_id="embedding_ranker_contract",
        title="Embedding Ranker Contract",
        difficulty="medium",
        description="Repair an ONNX ranker export so token-id inputs, symbolic dimensions, and runtime loading all satisfy a retrieval profile.",
        product_brief="The ranker must accept int64 token IDs, use symbolic batch and sequence dims, and run with ORT extended optimizations.",
        deployment_profile="retrieval_ranker",
        max_steps=10,
        success_threshold=0.96,
        bundle={
            "graph_config": {
                "opset": 17,
                "dynamic_batch": False,
                "dynamic_sequence": False,
                "input_ids_dtype": "int32",
                "provider": "CPUExecutionProvider",
                "optimization_level": "ORT_ENABLE_BASIC",
                "estimated_model_mb": 84.0,
            },
            "io_contract": {
                "input_name": "input_ids",
                "input_dtype": "int32",
                "input_shape": [1, 16],
                "output_name": "pooled",
                "output_dtype": "float",
                "output_shape": [1, 16],
            },
            "deployment": {
                "target_ep": "CPUExecutionProvider",
                "target_device": "retrieval-cpu",
            },
        },
        patches=[
            PatchOption(
                patch_id="set_input_ids_int64",
                slot_name="io_contract",
                label="Use int64 token IDs",
                description="ONNX export should expose token IDs as int64 for serving compatibility.",
                effect={"graph_config": {"input_ids_dtype": "int64"}, "io_contract": {"input_dtype": "int64"}},
                resolves=("token_ids_not_int64",),
                unlocks=("needs_dynamic_sequence",),
            ),
            PatchOption(
                patch_id="set_dynamic_batch",
                slot_name="graph_config",
                label="Enable dynamic batch",
                description="Allow variable request batch size.",
                effect={"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 16], "output_shape": ["batch", 16]}},
                resolves=("static_batch_only",),
            ),
            PatchOption(
                patch_id="set_dynamic_sequence",
                slot_name="graph_config",
                label="Enable symbolic sequence length",
                description="Allow variable sequence length for tokenizer outputs.",
                effect={"graph_config": {"dynamic_sequence": True}, "io_contract": {"input_shape": ["batch", "seq"]}},
                resolves=("needs_dynamic_sequence",),
            ),
            PatchOption(
                patch_id="set_extended_optim",
                slot_name="graph_config",
                label="Use extended graph optimization",
                description="Raise optimization level for deployment.",
                effect={"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}},
                resolves=("needs_extended_optim",),
            ),
            PatchOption(
                patch_id="bloat_model_size",
                slot_name="graph_config",
                label="Inline oversized debug tensors",
                description="Makes the exported bundle exceed the retrieval memory profile.",
                effect={"graph_config": {"estimated_model_mb": 164.0}},
            ),
        ],
        requirement_specs=[
            RequirementSpec("checker_passed", "ONNX checker must pass", 0.20, 0.80, "token_ids_not_int64", True, (), "onnx.checker.ValidationError: Gather indices must align with serving contract"),
            RequirementSpec("token_ids_int64", "Expose int64 token IDs", 0.20, 0.95, "token_ids_not_int64", True, (), "runtime contract mismatch: tokenizer emits int64 but model expects int32"),
            RequirementSpec("dynamic_batch", "Expose symbolic batch dim", 0.15, 0.60, "static_batch_only", True, (), "shape inference warning: batch dimension is fixed"),
            RequirementSpec("dynamic_sequence", "Expose symbolic sequence dim", 0.15, 0.85, "needs_dynamic_sequence", False, ("token_ids_not_int64",), "shape inference warning: sequence length is frozen to export-time value"),
            RequirementSpec("ort_session_passed", "ORT CPU session must load", 0.15, 0.80, "needs_extended_optim", False, ("needs_dynamic_sequence",), "onnxruntime: deployment bundle requires extended optimization level"),
            RequirementSpec("memory_budget_ok", "Stay within retrieval memory budget", 0.15, 0.55, "memory_budget_exceeded", True, (), "bundle size estimate exceeds retrieval budget"),
        ],
        variants=[
            VariantSpec("rerank-api", {"deployment": {"target_device": "rerank-api-cpu"}, "graph_config": {"estimated_model_mb": 84.0}}),
            VariantSpec("search-sidecar", {"deployment": {"target_device": "search-sidecar-cpu"}, "graph_config": {"estimated_model_mb": 92.0}}),
        ],
    )
)

_register(
    OnnxTask(
        task_id="vision_resize_mobile",
        title="Vision Resize Mobile",
        difficulty="hard",
        description="Repair a mobile vision ONNX export so checker, shape inference, and ORT runtime all agree under a mobile memory budget.",
        product_brief="The vision export must keep only one Resize size spec, expose dynamic batch, stay under the memory budget, and run on ORT CPU for mobile.",
        deployment_profile="vision_mobile",
        max_steps=12,
        success_threshold=0.97,
        bundle={
            "graph_config": {
                "opset": 18,
                "dynamic_batch": False,
                "resize_mode": "both",  # invalid: scales and sizes together
                "provider": "CPUExecutionProvider",
                "optimization_level": "ORT_ENABLE_BASIC",
                "estimated_model_mb": 142.0,
            },
            "io_contract": {
                "input_name": "image",
                "input_dtype": "float",
                "input_shape": [1, 3, 224, 224],
                "output_name": "resized",
                "output_dtype": "float",
                "output_shape": [1, 3, 112, 112],
            },
            "deployment": {
                "target_ep": "CPUExecutionProvider",
                "target_device": "mobile-vision-cpu",
            },
        },
        patches=[
            PatchOption(
                patch_id="resize_sizes_only",
                slot_name="graph_config",
                label="Keep only Resize sizes",
                description="Emit a valid Resize spec with sizes only.",
                effect={"graph_config": {"resize_mode": "sizes_only"}},
                resolves=("resize_has_scales_and_sizes",),
                unlocks=("static_batch_only",),
            ),
            PatchOption(
                patch_id="set_dynamic_batch",
                slot_name="graph_config",
                label="Enable dynamic batch",
                description="Use a symbolic batch dimension for serving.",
                effect={"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 224, 224], "output_shape": ["batch", 3, 112, 112]}},
                resolves=("static_batch_only",),
                unlocks=("needs_extended_optim",),
            ),
            PatchOption(
                patch_id="set_all_optim",
                slot_name="graph_config",
                label="Enable all graph optimizations",
                description="Use ORT all optimizations for mobile deployment.",
                effect={"graph_config": {"optimization_level": "ORT_ENABLE_ALL"}},
                resolves=("needs_extended_optim",),
            ),
            PatchOption(
                patch_id="prune_debug_initializer",
                slot_name="graph_config",
                label="Prune debug initializer",
                description="Remove extra constant tensors to fit mobile budget.",
                effect={"graph_config": {"estimated_model_mb": 92.0}},
                resolves=("memory_budget_exceeded",),
            ),
            PatchOption(
                patch_id="switch_cuda_provider",
                slot_name="deployment",
                label="Switch to CUDA provider",
                description="Incorrect for the mobile CPU profile.",
                effect={"deployment": {"target_ep": "CUDAExecutionProvider"}},
            ),
        ],
        requirement_specs=[
            RequirementSpec("checker_passed", "ONNX checker must pass", 0.25, 1.00, "resize_has_scales_and_sizes", True, (), "onnx.checker.ValidationError: Resize cannot define both scales and sizes"),
            RequirementSpec("shape_inference_passed", "Shape inference must succeed", 0.20, 0.80, "static_batch_only", False, ("resize_has_scales_and_sizes",), "shape inference failed after invalid Resize signature"),
            RequirementSpec("dynamic_batch", "Expose dynamic batch dimension", 0.15, 0.70, "static_batch_only", False, ("resize_has_scales_and_sizes",), "mobile serving requires symbolic batch dimension"),
            RequirementSpec("ort_session_passed", "ORT CPU session must load", 0.15, 0.85, "needs_extended_optim", False, ("static_batch_only",), "onnxruntime: mobile session contract requires higher optimization level"),
            RequirementSpec("memory_budget_ok", "Stay within mobile memory budget", 0.15, 0.75, "memory_budget_exceeded", True, (), "estimated model footprint exceeds mobile memory budget"),
            RequirementSpec("target_provider", "Use CPU execution provider", 0.10, 0.45, "wrong_provider", True, (), "deployment profile requires CPUExecutionProvider"),
        ],
        variants=[
            VariantSpec("camera-preview", {"deployment": {"target_device": "camera-preview-pipeline"}, "graph_config": {"estimated_model_mb": 142.0}}),
            VariantSpec("gallery-batch", {"deployment": {"target_device": "gallery-processor"}, "graph_config": {"estimated_model_mb": 152.0}}),
        ],
    )
)
