from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


PROFILE_SPECS = {
    "mobile_classifier": {"memory_budget_mb": 96.0, "target_ep": "CPUExecutionProvider", "priority": "latency_and_compatibility", "required_dynamic_dims": 1, "hardware": "mid-range ARM mobile CPU"},
    "retrieval_ranker": {"memory_budget_mb": 128.0, "target_ep": "CPUExecutionProvider", "priority": "shape_stability", "required_dynamic_dims": 2, "hardware": "2-vCPU retrieval worker"},
    "vision_mobile": {"memory_budget_mb": 160.0, "target_ep": "CPUExecutionProvider", "priority": "mobile_readiness", "required_dynamic_dims": 1, "hardware": "memory-constrained mobile vision target"},
    "npu_gateway": {"memory_budget_mb": 144.0, "target_ep": "NNAPIExecutionProvider", "priority": "hardware_offload", "required_dynamic_dims": 1, "hardware": "mobile NPU gateway"},
    "webnn_browser": {"memory_budget_mb": 120.0, "target_ep": "WebNNExecutionProvider", "priority": "browser_edge", "required_dynamic_dims": 2, "hardware": "browser WebNN runtime"},
    "edge_packaging": {"memory_budget_mb": 128.0, "target_ep": "CPUExecutionProvider", "priority": "artifact_packaging", "required_dynamic_dims": 2, "hardware": "sidecar packaging worker"},
    "quantized_mobile": {"memory_budget_mb": 118.0, "target_ep": "CPUExecutionProvider", "priority": "quantized_mobile", "required_dynamic_dims": 1, "hardware": "quantized mobile CPU"},
    "detection_bridge": {"memory_budget_mb": 176.0, "target_ep": "CPUExecutionProvider", "priority": "multi_stage_pipeline", "required_dynamic_dims": 1, "hardware": "mobile detection bridge"},
    "release_candidate": {"memory_budget_mb": 192.0, "target_ep": "CoreMLExecutionProvider", "priority": "release_gate", "required_dynamic_dims": 1, "hardware": "iOS release validation lane"},
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
        result.setdefault(patch.slot_name, []).append({"patch_id": patch.patch_id, "label": patch.label, "description": patch.description, "resolves": list(patch.resolves), "unlocks": list(patch.unlocks)})
    return result


def _dependency_graph(patches: list[PatchOption]) -> dict[str, list[str]]:
    return {patch.patch_id: list(patch.unlocks) for patch in patches if patch.unlocks}


def _register(task: OnnxTask) -> None:
    TASKS[task.task_id] = task
    PATCH_LOOKUP[task.task_id] = {patch.patch_id: patch for patch in task.patches}
    PATCH_CATALOG[task.task_id] = _patch_dict(task.patches)
    PATCH_DEPENDENCY_GRAPHS[task.task_id] = _dependency_graph(task.patches)


def p(patch_id: str, slot_name: str, label: str, description: str, effect: dict[str, Any], resolves: tuple[str, ...] = (), unlocks: tuple[str, ...] = ()) -> PatchOption:
    return PatchOption(patch_id, slot_name, label, description, effect, resolves, unlocks)


def r(key: str, description: str, weight: float, severity: float, issue_id: str, initial_visible: bool = True, depends_on: tuple[str, ...] = (), error_hint: str = "") -> RequirementSpec:
    return RequirementSpec(key, description, weight, severity, issue_id, initial_visible, depends_on, error_hint)


def v(label: str, bundle_updates: dict[str, Any]) -> VariantSpec:
    return VariantSpec(label, bundle_updates)


_register(OnnxTask(
    task_id="label_head_dtype_repair",
    title="Classifier Output Contract Repair",
    difficulty="easy",
    description="Repair an ONNX classifier export so checker, shape inference, and ORT loading all succeed for a mobile classifier.",
    product_brief="The exported classifier must use opset 17, expose a dynamic batch dimension, and declare ArgMax labels as int64.",
    deployment_profile="mobile_classifier",
    max_steps=8,
    success_threshold=0.95,
    bundle={"graph_config": {"graph_kind": "classifier_head", "opset": 13, "dynamic_batch": False, "label_output_dtype": "float", "provider": "CPUExecutionProvider", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 54.0}, "io_contract": {"input_name": "features", "input_dtype": "float", "input_shape": [1, 4], "output_name": "labels", "output_dtype": "float", "output_shape": [1]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "mobile-cpu"}},
    patches=[
        p("set_opset_17", "graph_config", "Raise opset to 17", "Use a runtime-supported opset for the deployment contract.", {"graph_config": {"opset": 17}}, ("low_opset",)),
        p("set_dynamic_batch", "graph_config", "Enable dynamic batch", "Expose a symbolic batch dimension for serving.", {"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 4], "output_shape": ["batch"]}}, ("static_batch_only",)),
        p("set_label_output_int64", "io_contract", "Declare labels as int64", "Match ArgMax output type in ONNX.", {"graph_config": {"label_output_dtype": "int64"}, "io_contract": {"output_dtype": "int64"}}, ("label_dtype_mismatch",), ("needs_extended_optim",)),
        p("set_extended_optim", "graph_config", "Use extended graph optimization", "Move to ORT extended graph optimizations for production parity.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_extended_optim",)),
        p("switch_cuda_provider", "deployment", "Switch to CUDA provider", "Not valid in this CPU-only target profile.", {"deployment": {"target_ep": "CUDAExecutionProvider"}}),
    ],
    requirement_specs=[
        r("label_dtype_match", "Declare labels as int64", 0.30, 1.00, "label_dtype_mismatch", True, (), "onnx.checker: Graph output type does not match ArgMax(int64) output"),
        r("dynamic_batch", "Expose dynamic batch dimension", 0.20, 0.75, "static_batch_only", True, (), "shape inference warning: exported batch dimension is fixed to 1"),
        r("opset_valid", "Use opset 17 or newer", 0.15, 0.55, "low_opset", True, (), "ORT loader warning: opset below deployment policy target"),
        r("ort_session_passed", "ORT CPU session must load", 0.20, 0.85, "needs_extended_optim", False, ("label_dtype_mismatch",), "onnxruntime: graph optimization contract not met for deployment bundle"),
        r("target_provider", "Use CPU execution provider", 0.15, 0.45, "wrong_provider", True, (), "onnxruntime: requested provider is unavailable on target hardware"),
    ],
    variants=[v("android-batch-audit", {"deployment": {"target_device": "android-midrange"}, "graph_config": {"estimated_model_mb": 54.0}}), v("ios-store-check", {"deployment": {"target_device": "ios-coreml-gateway"}, "graph_config": {"estimated_model_mb": 58.0}})],
))

_register(OnnxTask(
    task_id="embedding_ranker_contract",
    title="Retrieval Export Rescue",
    difficulty="medium",
    description="Repair an ONNX ranker export so token-id inputs, symbolic dimensions, and runtime loading all satisfy a retrieval profile.",
    product_brief="The ranker must accept int64 token IDs, use symbolic batch and sequence dims, and run with ORT extended optimizations.",
    deployment_profile="retrieval_ranker",
    max_steps=10,
    success_threshold=0.96,
    bundle={"graph_config": {"graph_kind": "embedding_ranker", "opset": 17, "dynamic_batch": False, "dynamic_sequence": False, "input_ids_dtype": "int32", "provider": "CPUExecutionProvider", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 84.0}, "io_contract": {"input_name": "input_ids", "input_dtype": "int32", "input_shape": [1, 16], "output_name": "pooled", "output_dtype": "float", "output_shape": [1, 16]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "retrieval-cpu"}},
    patches=[
        p("set_input_ids_int64", "io_contract", "Use int64 token IDs", "Expose token IDs as int64 for serving compatibility.", {"graph_config": {"input_ids_dtype": "int64"}, "io_contract": {"input_dtype": "int64"}}, ("token_ids_not_int64",), ("needs_dynamic_sequence",)),
        p("set_dynamic_batch", "graph_config", "Enable dynamic batch", "Allow variable request batch size.", {"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 16], "output_shape": ["batch", 16]}}, ("static_batch_only",)),
        p("set_dynamic_sequence", "graph_config", "Enable symbolic sequence length", "Allow variable sequence length for tokenizer outputs.", {"graph_config": {"dynamic_sequence": True}, "io_contract": {"input_shape": ["batch", "seq"]}}, ("needs_dynamic_sequence",)),
        p("set_extended_optim", "graph_config", "Use extended graph optimization", "Raise optimization level for deployment.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_extended_optim",)),
        p("bloat_model_size", "graph_config", "Inline oversized debug tensors", "Makes the exported bundle exceed the retrieval memory profile.", {"graph_config": {"estimated_model_mb": 164.0}}),
    ],
    requirement_specs=[
        r("checker_passed", "ONNX checker must pass", 0.20, 0.80, "checker_contract", True, (), "onnx.checker: Gather indices must align with serving contract"),
        r("token_ids_int64", "Expose int64 token IDs", 0.20, 0.95, "token_ids_not_int64", True, (), "runtime contract mismatch: tokenizer emits int64 but model expects int32"),
        r("dynamic_batch", "Expose symbolic batch dim", 0.15, 0.60, "static_batch_only", True, (), "shape inference warning: batch dimension is fixed"),
        r("dynamic_sequence", "Expose symbolic sequence dim", 0.15, 0.85, "needs_dynamic_sequence", False, ("token_ids_not_int64",), "shape inference warning: sequence length is frozen to export-time value"),
        r("ort_session_passed", "ORT CPU session must load", 0.15, 0.80, "needs_extended_optim", False, ("needs_dynamic_sequence",), "onnxruntime: deployment bundle requires extended optimization level"),
        r("memory_budget_ok", "Stay within retrieval memory budget", 0.15, 0.55, "memory_budget_exceeded", True, (), "bundle size estimate exceeds retrieval budget"),
    ],
    variants=[v("rerank-api", {"deployment": {"target_device": "rerank-api-cpu"}, "graph_config": {"estimated_model_mb": 84.0}}), v("search-sidecar", {"deployment": {"target_device": "search-sidecar-cpu"}, "graph_config": {"estimated_model_mb": 92.0}})],
))

_register(OnnxTask(
    task_id="vision_resize_mobile",
    title="Mobile Vision Runtime Fix",
    difficulty="hard",
    description="Repair a mobile vision ONNX export so checker, shape inference, and ORT runtime all agree under a mobile memory budget.",
    product_brief="The vision export must keep only one Resize size spec, expose dynamic batch, stay under the memory budget, and run on ORT CPU for mobile.",
    deployment_profile="vision_mobile",
    max_steps=12,
    success_threshold=0.97,
    bundle={"graph_config": {"graph_kind": "vision_resize", "opset": 18, "dynamic_batch": False, "resize_mode": "both", "provider": "CPUExecutionProvider", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 142.0}, "io_contract": {"input_name": "image", "input_dtype": "float", "input_shape": [1, 3, 224, 224], "output_name": "resized", "output_dtype": "float", "output_shape": [1, 3, 112, 112]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "mobile-vision-cpu"}},
    patches=[
        p("resize_sizes_only", "graph_config", "Keep only Resize sizes", "Emit a valid Resize spec with sizes only.", {"graph_config": {"resize_mode": "sizes_only"}}, ("resize_has_scales_and_sizes",), ("shape_pipe_not_ready",)),
        p("set_dynamic_batch", "graph_config", "Enable dynamic batch", "Use a symbolic batch dimension for serving.", {"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 224, 224], "output_shape": ["batch", 3, 112, 112]}}, ("static_batch_only",), ("needs_extended_optim",)),
        p("set_all_optim", "graph_config", "Enable all graph optimizations", "Use ORT all optimizations for mobile deployment.", {"graph_config": {"optimization_level": "ORT_ENABLE_ALL"}}, ("needs_extended_optim",)),
        p("prune_debug_initializer", "graph_config", "Prune debug initializer", "Remove extra constant tensors to fit the mobile budget.", {"graph_config": {"estimated_model_mb": 92.0}}, ("memory_budget_exceeded",)),
        p("switch_cuda_provider", "deployment", "Switch to CUDA provider", "Incorrect for the mobile CPU profile.", {"deployment": {"target_ep": "CUDAExecutionProvider"}}),
    ],
    requirement_specs=[
        r("checker_passed", "ONNX checker must pass", 0.25, 1.00, "resize_has_scales_and_sizes", True, (), "onnx.checker: Resize cannot define both scales and sizes"),
        r("shape_inference_passed", "Shape inference must succeed", 0.20, 0.80, "shape_pipe_not_ready", False, ("resize_has_scales_and_sizes",), "shape inference failed after invalid Resize signature"),
        r("dynamic_batch", "Expose dynamic batch dimension", 0.15, 0.70, "static_batch_only", False, ("resize_has_scales_and_sizes",), "mobile serving requires symbolic batch dimension"),
        r("ort_session_passed", "ORT CPU session must load", 0.15, 0.85, "needs_extended_optim", False, ("static_batch_only",), "onnxruntime: mobile session contract requires higher optimization level"),
        r("memory_budget_ok", "Stay within mobile memory budget", 0.15, 0.75, "memory_budget_exceeded", True, (), "estimated model footprint exceeds mobile memory budget"),
        r("target_provider", "Use CPU execution provider", 0.10, 0.45, "wrong_provider", True, (), "deployment profile requires CPUExecutionProvider"),
    ],
    variants=[v("camera-preview", {"deployment": {"target_device": "camera-preview-pipeline"}, "graph_config": {"estimated_model_mb": 142.0}}), v("gallery-batch", {"deployment": {"target_device": "gallery-processor"}, "graph_config": {"estimated_model_mb": 152.0}})],
))

_register(OnnxTask(
    task_id="npu_gateway_surgery",
    title="NPU Gateway Surgery",
    difficulty="hard",
    description="Repair an ONNX detection export so it satisfies mobile NPU constraints for rank, supported ops, and execution provider.",
    product_brief="The NPU gateway requires rank-4 NCHW input, a NonMaxSuppression polyfill, and an NNAPI execution provider under the hardware budget.",
    deployment_profile="npu_gateway",
    max_steps=12,
    success_threshold=0.97,
    bundle={"graph_config": {"graph_kind": "npu_gateway", "opset": 18, "input_rank": 3, "nms_mode": "raw", "dynamic_batch": False, "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 118.0}, "io_contract": {"input_name": "image", "input_dtype": "float", "input_shape": [1, 224, 224], "output_name": "boxes", "output_dtype": "float", "output_shape": [1, 40, 4]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "npu-gateway"}},
    patches=[
        p("set_rank4_nchw", "io_contract", "Convert input to NCHW rank-4", "Promote the image input to rank-4 NCHW so the NPU can admit it.", {"graph_config": {"input_rank": 4, "dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 224, 224]}}, ("rank3_input",), ("nms_not_polyfilled",)),
        p("polyfill_nms", "graph_config", "Replace NMS with polyfill", "Swap raw NonMaxSuppression with a provider-safe decode + topk polyfill.", {"graph_config": {"nms_mode": "polyfill"}}, ("nms_not_polyfilled",), ("wrong_provider",)),
        p("switch_nnapi_provider", "deployment", "Target NNAPI provider", "Route execution to the mobile NPU gateway.", {"deployment": {"target_ep": "NNAPIExecutionProvider"}}, ("wrong_provider",), ("needs_npu_optim",)),
        p("enable_npu_optim", "graph_config", "Enable NPU optimizations", "Use extended graph optimization for the NNAPI runtime path.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_npu_optim",)),
        p("stay_on_cpu_provider", "deployment", "Keep CPU execution provider", "This leaves the bundle off the hardware acceleration path.", {"deployment": {"target_ep": "CPUExecutionProvider"}}),
    ],
    requirement_specs=[
        r("rank4_input", "Promote the input contract to rank-4 NCHW", 0.22, 0.95, "rank3_input", True, (), "nnapi validator: rank-3 inputs are rejected for image tensor admission"),
        r("nms_polyfilled", "Replace raw NonMaxSuppression with a polyfill", 0.20, 0.90, "nms_not_polyfilled", False, ("rank3_input",), "nnapi compiler: NonMaxSuppression is unsupported on the target accelerator"),
        r("target_provider", "Use NNAPI execution provider", 0.18, 0.70, "wrong_provider", False, ("nms_not_polyfilled",), "deployment policy: NPU gateway bundles must target NNAPIExecutionProvider"),
        r("ort_session_passed", "ORT session must load for the NPU path", 0.20, 0.95, "needs_npu_optim", False, ("wrong_provider",), "onnxruntime: hardware compatibility path requires extended graph optimization"),
        r("memory_budget_ok", "Stay within the NPU gateway budget", 0.10, 0.55, "gateway_budget", True, (), "gateway policy: model footprint exceeds mobile NPU budget"),
        r("dynamic_batch", "Expose dynamic request batch dimension", 0.10, 0.60, "fixed_gateway_batch", False, ("rank3_input",), "deployment warning: gateway batch dimension remains frozen"),
    ],
    variants=[v("android-camera-edge", {"deployment": {"target_device": "android-camera-edge"}, "graph_config": {"estimated_model_mb": 118.0}}), v("wearable-safety-cam", {"deployment": {"target_device": "wearable-safety-cam"}, "graph_config": {"estimated_model_mb": 126.0}})],
))

_register(OnnxTask(
    task_id="webnn_static_dynamic_pivot",
    title="WebNN Static-Dynamic Pivot",
    difficulty="hard",
    description="Repair a browser-side ONNX transformer export so WebNN accepts the dimension strategy and memory footprint.",
    product_brief="The WebNN profile rejects mixed static/dynamic dimensions. The repair must choose a consistent dimension strategy that still fits the browser memory budget.",
    deployment_profile="webnn_browser",
    max_steps=12,
    success_threshold=0.97,
    bundle={"graph_config": {"graph_kind": "webnn_transform", "opset": 18, "dynamic_batch": True, "dynamic_sequence": False, "webnn_dim_strategy": "mixed", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 114.0}, "io_contract": {"input_name": "tokens", "input_dtype": "int64", "input_shape": ["batch", 128], "output_name": "hidden", "output_dtype": "float", "output_shape": ["batch", 128, 64]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "browser-sidecar"}},
    patches=[
        p("set_all_dynamic_dims", "graph_config", "Make batch and sequence dynamic", "Use a fully dynamic strategy so WebNN accepts the graph topology.", {"graph_config": {"dynamic_batch": True, "dynamic_sequence": True, "webnn_dim_strategy": "all_dynamic", "estimated_model_mb": 108.0}, "io_contract": {"input_shape": ["batch", "seq"], "output_shape": ["batch", "seq", 64]}}, ("mixed_webnn_dims",), ("wrong_provider",)),
        p("set_all_static_dims", "graph_config", "Freeze batch and sequence dims", "This repairs the topology but inflates browser memory usage.", {"graph_config": {"dynamic_batch": False, "dynamic_sequence": False, "webnn_dim_strategy": "all_static", "estimated_model_mb": 148.0}, "io_contract": {"input_shape": [1, 128], "output_shape": [1, 128, 64]}}, ("mixed_webnn_dims",)),
        p("prune_browser_cache", "graph_config", "Prune browser cache tensors", "Reduce the frozen bundle so it fits the browser edge budget.", {"graph_config": {"estimated_model_mb": 104.0}}, ("webnn_budget_exceeded",)),
        p("switch_webnn_provider", "deployment", "Target WebNN execution provider", "Use the browser hardware acceleration path.", {"deployment": {"target_ep": "WebNNExecutionProvider"}}, ("wrong_provider",), ("needs_browser_optim",)),
        p("enable_browser_optim", "graph_config", "Enable browser-focused optimizations", "Raise optimization level for the WebNN runtime.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_browser_optim",)),
    ],
    requirement_specs=[
        r("webnn_dims_consistent", "Use a consistent static or dynamic WebNN dimension strategy", 0.22, 0.95, "mixed_webnn_dims", True, (), "webnn validator: mixed static and dynamic dimensions in the attention path"),
        r("memory_budget_ok", "Stay within the browser memory budget", 0.18, 0.85, "webnn_budget_exceeded", True, (), "browser budget gate: footprint exceeds WebNN deployment limit"),
        r("target_provider", "Use WebNN execution provider", 0.18, 0.75, "wrong_provider", False, ("mixed_webnn_dims",), "browser deployment policy: profile requires WebNNExecutionProvider"),
        r("ort_session_passed", "ORT session must load for the browser profile", 0.18, 0.90, "needs_browser_optim", False, ("wrong_provider",), "onnxruntime-web: WebNN path requires extended optimization level"),
        r("dynamic_dim_count_match", "Expose enough symbolic dims for the WebNN profile", 0.12, 0.70, "insufficient_dynamic_dims", False, ("mixed_webnn_dims",), "profile policy: browser path requires two symbolic dimensions"),
        r("checker_passed", "ONNX checker must pass", 0.12, 0.55, "checker_contract", True, (), "onnx.checker: inconsistent dimension annotations across IO contract"),
    ],
    variants=[v("browser-search", {"deployment": {"target_device": "web-search-widget"}, "graph_config": {"estimated_model_mb": 114.0}}), v("web-summarizer", {"deployment": {"target_device": "web-summarizer"}, "graph_config": {"estimated_model_mb": 119.0}})],
))

_register(OnnxTask(
    task_id="external_data_packaging_failure",
    title="External Data Packaging Failure",
    difficulty="expert",
    description="Repair a transformer sidecar export so it uses external data packaging, exposes a valid attention contract, and loads under the edge packaging profile.",
    product_brief="The deployment bundle must externalize large weights, keep attention masks symbolic, and run under the CPU packaging profile without exceeding the budget.",
    deployment_profile="edge_packaging",
    max_steps=12,
    success_threshold=0.98,
    bundle={"graph_config": {"graph_kind": "external_data", "opset": 18, "dynamic_batch": True, "dynamic_sequence": False, "external_data": False, "attention_mask_dynamic": False, "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 196.0}, "io_contract": {"input_name": "input_ids", "input_dtype": "int64", "input_shape": ["batch", 128], "output_name": "hidden", "output_dtype": "float", "output_shape": ["batch", 128, 256]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "transformer-sidecar"}},
    patches=[
        p("externalize_weights", "graph_config", "Move large tensors to external data", "Use ONNX external data packaging for constant weights.", {"graph_config": {"external_data": True, "estimated_model_mb": 118.0}}, ("inline_weights",), ("static_attention_mask",)),
        p("set_attention_mask_dynamic", "graph_config", "Make the attention mask symbolic", "Allow variable sequence length in the sidecar contract.", {"graph_config": {"dynamic_sequence": True, "attention_mask_dynamic": True}, "io_contract": {"input_shape": ["batch", "seq"], "output_shape": ["batch", "seq", 256]}}, ("static_attention_mask",), ("needs_packaging_optim",)),
        p("set_extended_optim", "graph_config", "Enable extended packaging optimizations", "Use a stable ORT optimization level for the packaged transformer.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_packaging_optim",)),
        p("switch_coreml_provider", "deployment", "Switch to CoreML provider", "This sidecar bundle is meant to stay on CPU packaging workers.", {"deployment": {"target_ep": "CoreMLExecutionProvider"}}),
        p("inflate_debug_cache", "graph_config", "Inline debug cache tensors", "This makes the packaged sidecar too large for the profile.", {"graph_config": {"estimated_model_mb": 228.0}}),
    ],
    requirement_specs=[
        r("external_data", "Externalize large constant tensors", 0.24, 0.98, "inline_weights", True, (), "packaging gate: inline weights exceed the sidecar artifact size limit"),
        r("attention_mask_dynamic", "Expose a symbolic attention mask", 0.18, 0.82, "static_attention_mask", False, ("inline_weights",), "shape inference: attention mask is frozen to export-time length"),
        r("ort_session_passed", "ORT CPU session must load", 0.18, 0.90, "needs_packaging_optim", False, ("static_attention_mask",), "onnxruntime: packaged transformer path requires extended optimization"),
        r("memory_budget_ok", "Stay within the sidecar memory budget", 0.16, 0.88, "packaging_budget_exceeded", True, (), "artifact policy: packaged model still exceeds the sidecar footprint budget"),
        r("target_provider", "Keep CPU execution provider for the sidecar worker", 0.12, 0.60, "wrong_provider", True, (), "deployment profile requires CPUExecutionProvider for packaging workers"),
        r("dynamic_dim_count_match", "Expose enough symbolic dims for the sidecar contract", 0.12, 0.65, "insufficient_dynamic_dims", False, ("inline_weights",), "profile policy: sidecar profile requires symbolic batch and sequence dims"),
    ],
    variants=[v("search-sidecar-v2", {"deployment": {"target_device": "search-sidecar-v2"}, "graph_config": {"estimated_model_mb": 196.0}}), v("rerank-offline-packager", {"deployment": {"target_device": "rerank-offline-packager"}, "graph_config": {"estimated_model_mb": 204.0}})],
))

_register(OnnxTask(
    task_id="broken_quantized_cascade",
    title="Broken Quantized Cascade",
    difficulty="expert",
    description="Repair a quantized export with coupled QDQ, opset, and artifact size failures before it can ship to mobile.",
    product_brief="The quantized mobile profile needs aligned QDQ scales, a modern quantization opset, pruned debug tensors, and a stable ORT CPU session.",
    deployment_profile="quantized_mobile",
    max_steps=12,
    success_threshold=0.98,
    bundle={"graph_config": {"graph_kind": "quantized_cascade", "opset": 13, "dynamic_batch": False, "quant_scale_mode": "mismatched", "debug_tensors": "inline", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 144.0}, "io_contract": {"input_name": "image", "input_dtype": "uint8", "input_shape": [1, 3, 160, 160], "output_name": "scores", "output_dtype": "float", "output_shape": [1, 80]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "quant-mobile"}},
    patches=[
        p("align_qdq_scales", "graph_config", "Align QDQ scales", "Normalize QuantizeLinear / DequantizeLinear scale pairs.", {"graph_config": {"quant_scale_mode": "aligned"}}, ("quant_scale_mismatch",), ("low_quant_opset",)),
        p("raise_quant_opset_21", "graph_config", "Raise quantization opset to 21", "Use a mobile-safe quantization opset.", {"graph_config": {"opset": 21}}, ("low_quant_opset",), ("debug_tensor_bloat",)),
        p("prune_quant_debug_tensors", "graph_config", "Prune quant debug tensors", "Remove calibration tensors that were accidentally bundled.", {"graph_config": {"debug_tensors": "pruned", "estimated_model_mb": 102.0}}, ("debug_tensor_bloat",), ("needs_quant_optim",)),
        p("enable_quant_fusion", "graph_config", "Enable quantization fusion", "Use extended optimization after the QDQ graph is repaired.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED", "dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 160, 160], "output_shape": ["batch", 80]}}, ("needs_quant_optim", "fixed_quant_batch")),
        p("switch_nnapi_provider", "deployment", "Switch to NNAPI provider", "This quantized task still targets CPU fallback runtime.", {"deployment": {"target_ep": "NNAPIExecutionProvider"}}),
    ],
    requirement_specs=[
        r("quant_scale_aligned", "Align quantization scales across the QDQ path", 0.22, 0.98, "quant_scale_mismatch", True, (), "quant validator: QuantizeLinear and DequantizeLinear scales disagree across adjacent nodes"),
        r("quant_opset_valid", "Use quantization opset 21 or newer", 0.18, 0.82, "low_quant_opset", False, ("quant_scale_mismatch",), "quant policy: mobile export uses a too-old opset for the current QuantizeLinear contract"),
        r("memory_budget_ok", "Stay within the quantized mobile memory budget", 0.16, 0.84, "debug_tensor_bloat", False, ("low_quant_opset",), "artifact size gate: debug tensors push the quantized export over budget"),
        r("ort_session_passed", "ORT CPU session must load", 0.18, 0.92, "needs_quant_optim", False, ("debug_tensor_bloat",), "onnxruntime: quantized path requires extended optimization after cleanup"),
        r("dynamic_batch", "Expose a dynamic batch dimension", 0.12, 0.64, "fixed_quant_batch", False, ("debug_tensor_bloat",), "deployment policy: mobile batch dimension remains fixed"),
        r("target_provider", "Keep the quantized bundle on CPU fallback provider", 0.14, 0.66, "wrong_provider", True, (), "profile policy requires CPUExecutionProvider for the quantized fallback lane"),
    ],
    variants=[v("mobile-vision-lite", {"deployment": {"target_device": "mobile-vision-lite"}, "graph_config": {"estimated_model_mb": 144.0}}), v("camera-quant-fallback", {"deployment": {"target_device": "camera-quant-fallback"}, "graph_config": {"estimated_model_mb": 150.0}})],
))

_register(OnnxTask(
    task_id="multi_stage_detection_bridge",
    title="Multi-Stage Detection Bridge",
    difficulty="expert",
    description="Repair a two-stage detection bundle so the preprocessor output matches the extractor input contract across layout and shape.",
    product_brief="The mobile detection bridge must align NHWC/NCHW layout, keep the stage handoff shape-valid, and finish with an ORT-ready CPU deployment bundle.",
    deployment_profile="detection_bridge",
    max_steps=13,
    success_threshold=0.98,
    bundle={"graph_config": {"graph_kind": "detection_bridge", "opset": 18, "dynamic_batch": False, "bridge_layout": "broken", "stage_contract": "mismatch", "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 166.0}, "io_contract": {"input_name": "image", "input_dtype": "float", "input_shape": [1, 3, 256, 256], "output_name": "detections", "output_dtype": "float", "output_shape": [1, 32, 6]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "mobile-detector-bridge"}},
    patches=[
        p("insert_nchw_bridge", "graph_config", "Insert layout bridge", "Repair the preprocessor -> extractor handoff with an NCHW transpose bridge.", {"graph_config": {"bridge_layout": "nchw_bridge"}}, ("layout_bridge_missing",), ("stage_contract_mismatch",)),
        p("align_stage_contract", "io_contract", "Align stage IO contract", "Update the extractor to accept the preprocessor output shape.", {"graph_config": {"stage_contract": "aligned"}}, ("stage_contract_mismatch",), ("bridge_batch_frozen",)),
        p("set_dynamic_batch", "graph_config", "Enable dynamic batch", "Keep the multi-stage pipeline symbolic over batch.", {"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 256, 256], "output_shape": ["batch", 32, 6]}}, ("bridge_batch_frozen",), ("needs_bridge_optim",)),
        p("set_extended_optim", "graph_config", "Enable bridge optimizations", "Use extended optimization once the stage handoff is valid.", {"graph_config": {"optimization_level": "ORT_ENABLE_EXTENDED"}}, ("needs_bridge_optim",)),
        p("switch_coreml_provider", "deployment", "Switch to CoreML provider", "This detection bridge is still evaluated on the CPU bridge runner.", {"deployment": {"target_ep": "CoreMLExecutionProvider"}}),
    ],
    requirement_specs=[
        r("layout_bridge_ok", "Insert the NHWC/NCHW bridge between stages", 0.22, 0.96, "layout_bridge_missing", True, (), "shape bridge: stage handoff uses incompatible layout order"),
        r("stage_contract_ok", "Align the stage IO contract", 0.18, 0.90, "stage_contract_mismatch", False, ("layout_bridge_missing",), "shape inference: preprocessor output rank does not match extractor input contract"),
        r("dynamic_batch", "Expose dynamic batch through the full pipeline", 0.14, 0.72, "bridge_batch_frozen", False, ("stage_contract_mismatch",), "pipeline policy: batch dimension remains static after bridge alignment"),
        r("ort_session_passed", "ORT CPU session must load", 0.18, 0.92, "needs_bridge_optim", False, ("bridge_batch_frozen",), "onnxruntime: multi-stage bridge path requires extended optimization"),
        r("memory_budget_ok", "Stay within the detection bridge budget", 0.14, 0.70, "bridge_budget_exceeded", True, (), "artifact gate: two-stage bundle exceeds mobile detection budget"),
        r("target_provider", "Keep the bridge on CPU execution provider", 0.14, 0.60, "wrong_provider", True, (), "bridge runner requires CPUExecutionProvider"),
    ],
    variants=[v("camera-burst", {"deployment": {"target_device": "camera-burst-detector"}, "graph_config": {"estimated_model_mb": 166.0}}), v("drone-sidecar", {"deployment": {"target_device": "drone-sidecar"}, "graph_config": {"estimated_model_mb": 174.0}})],
))

_register(OnnxTask(
    task_id="release_candidate_gate",
    title="Release Candidate Gate",
    difficulty="expert",
    description="Repair a release-candidate ONNX bundle that is blocked by checker, shape, provider, and packaging gates before it can ship.",
    product_brief="The iOS release candidate must repair mixed precision, resize signature, dynamic batch, provider, and packaging to clear the final ship gate.",
    deployment_profile="release_candidate",
    max_steps=16,
    success_threshold=0.99,
    bundle={"graph_config": {"graph_kind": "release_candidate", "opset": 11, "dynamic_batch": False, "resize_mode": "both", "mixed_precision_mode": "unsafe", "external_data": False, "optimization_level": "ORT_ENABLE_BASIC", "estimated_model_mb": 244.0}, "io_contract": {"input_name": "image", "input_dtype": "float16", "input_shape": [1, 3, 224, 224], "output_name": "scores", "output_dtype": "float16", "output_shape": [1, 1000]}, "deployment": {"target_ep": "CPUExecutionProvider", "target_device": "ios-rc"}},
    patches=[
        p("raise_release_opset_18", "graph_config", "Raise release opset to 18", "Use a modern opset for the release candidate path.", {"graph_config": {"opset": 18}}, ("release_low_opset",), ("release_resize_invalid",)),
        p("fix_release_resize", "graph_config", "Fix Resize signature", "Keep only the release-safe Resize sizes path.", {"graph_config": {"resize_mode": "sizes_only"}}, ("release_resize_invalid",), ("release_precision_unsafe",)),
        p("set_safe_mixed_precision", "graph_config", "Use safe mixed precision", "Convert the release candidate to a stable float / fp16 boundary.", {"graph_config": {"mixed_precision_mode": "safe"}, "io_contract": {"input_dtype": "float", "output_dtype": "float"}}, ("release_precision_unsafe",), ("release_batch_static",)),
        p("set_dynamic_batch", "graph_config", "Enable dynamic batch", "Expose symbolic batch for release validation.", {"graph_config": {"dynamic_batch": True}, "io_contract": {"input_shape": ["batch", 3, 224, 224], "output_shape": ["batch", 1000]}}, ("release_batch_static",), ("release_provider_wrong",)),
        p("switch_coreml_provider", "deployment", "Target CoreML execution provider", "Match the iOS release profile provider.", {"deployment": {"target_ep": "CoreMLExecutionProvider"}}, ("release_provider_wrong",), ("release_packaging_inline",)),
        p("externalize_release_weights", "graph_config", "Externalize release weights", "Move large tensors out of the main model for the ship gate.", {"graph_config": {"external_data": True, "estimated_model_mb": 178.0}}, ("release_packaging_inline",), ("release_needs_optim",)),
        p("set_all_optim", "graph_config", "Enable all release optimizations", "Use the final optimization pass for ship-ready validation.", {"graph_config": {"optimization_level": "ORT_ENABLE_ALL"}}, ("release_needs_optim",)),
    ],
    requirement_specs=[
        r("opset_valid", "Use a release-safe opset", 0.14, 0.80, "release_low_opset", True, (), "release gate: opset is too old for the final mobile runtime"),
        r("resize_signature_ok", "Fix the Resize signature", 0.14, 0.88, "release_resize_invalid", False, ("release_low_opset",), "onnx.checker: release candidate uses an invalid Resize signature"),
        r("mixed_precision_ok", "Use a safe mixed precision boundary", 0.16, 0.92, "release_precision_unsafe", False, ("release_resize_invalid",), "release verifier: unsafe mixed precision boundary around the preprocessing stack"),
        r("dynamic_batch", "Expose dynamic batch for release validation", 0.12, 0.70, "release_batch_static", False, ("release_precision_unsafe",), "shape inference warning: release batch is still static"),
        r("target_provider", "Use CoreML execution provider", 0.12, 0.78, "release_provider_wrong", False, ("release_batch_static",), "ship gate requires CoreMLExecutionProvider"),
        r("external_data", "Externalize large release tensors", 0.14, 0.92, "release_packaging_inline", False, ("release_provider_wrong",), "release packaging: weights must be externalized before shipping"),
        r("ort_session_passed", "ORT/CoreML validation must pass", 0.18, 0.96, "release_needs_optim", False, ("release_packaging_inline",), "release validator: final runtime path requires ORT_ENABLE_ALL after packaging"),
    ],
    variants=[v("app-store-rc", {"deployment": {"target_device": "app-store-rc"}, "graph_config": {"estimated_model_mb": 244.0}}), v("beta-flight", {"deployment": {"target_device": "beta-flight"}, "graph_config": {"estimated_model_mb": 236.0}})],
))
