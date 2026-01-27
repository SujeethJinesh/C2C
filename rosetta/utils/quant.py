"""
Lightweight KV-cache quantization utilities (PTQ, fake-quant).

This module intentionally avoids external dependencies so it can run on CPU/GPU.
It implements symmetric int8/int4 quantize->dequantize for KV tensors.
"""
from typing import Dict, Tuple, Optional

import torch


SUPPORTED_SCHEMES = {
    "int8": 8,
    "int4": 4,
}

NO_QUANT_SCHEMES = {
    "fp16",
    "fp32",
    "bf16",
    "bfloat16",
    "none",
    "no",
    "float16",
    "float32",
}


def _get_qmax(num_bits: int) -> int:
    return (2 ** (num_bits - 1)) - 1


def _resolve_axis(axis: str) -> Tuple[int, ...]:
    if axis == "head":
        # Expect KV shape: [batch, heads, tokens, head_dim]
        return (0, 2, 3)
    if axis == "layer":
        return (0, 1, 2, 3)
    raise ValueError(f"Unsupported kv_quant_axis: {axis}")


def _layer_in_override(layer_idx: int, override: Dict) -> bool:
    if layer_idx is None:
        return False
    if "layers" in override and override["layers"] is not None:
        return int(layer_idx) in [int(i) for i in override["layers"]]
    if "range" in override and override["range"] is not None:
        r = override["range"]
        if isinstance(r, (list, tuple)) and len(r) == 2:
            start, end = int(r[0]), int(r[1])
            return start <= int(layer_idx) <= end
    return False


def _normalize_scheme(scheme) -> str:
    if scheme is None:
        return "int8"
    if isinstance(scheme, str):
        return scheme.lower()
    return str(scheme).lower()


def resolve_scheme(config: Optional[Dict], layer_idx: Optional[int] = None) -> str:
    if not config:
        return "int8"
    base_scheme = _normalize_scheme(config.get("scheme", "int8"))
    schedule = config.get("layer_schedule")
    if not schedule or layer_idx is None:
        return base_scheme
    scheme = _normalize_scheme(schedule.get("default", base_scheme))
    overrides = schedule.get("overrides", []) or []
    for override in overrides:
        if _layer_in_override(layer_idx, override):
            scheme = _normalize_scheme(override.get("scheme", scheme))
    return scheme


def quantize_dequantize(
    x: torch.Tensor,
    num_bits: int,
    axis: str,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x is None:
        raise ValueError("Input tensor is None")
    if num_bits <= 0:
        raise ValueError(f"num_bits must be > 0, got {num_bits}")

    qmax = _get_qmax(num_bits)
    reduce_dims = _resolve_axis(axis)

    x_fp32 = x.float()
    max_val = x_fp32.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(max_val / float(qmax), min=eps)
    q = torch.clamp(torch.round(x_fp32 / scale), -qmax, qmax)
    x_hat = (q * scale).to(dtype=x.dtype)
    return x_hat, scale


def _summarize_scale(scale: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(scale.min().item()),
        "max": float(scale.max().item()),
        "mean": float(scale.mean().item()),
    }


def quantize_kv(
    kv: Tuple[torch.Tensor, torch.Tensor],
    scheme: str = "int8",
    axis: str = "head",
    eps: float = 1e-6,
    collect_stats: bool = False,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported kv_quant_scheme: {scheme}")

    num_bits = SUPPORTED_SCHEMES[scheme]
    key, value = kv
    qk, k_scale = quantize_dequantize(key, num_bits=num_bits, axis=axis, eps=eps)
    qv, v_scale = quantize_dequantize(value, num_bits=num_bits, axis=axis, eps=eps)
    info: Dict[str, torch.Tensor] = {
        "scheme": scheme,
        "axis": axis,
    }
    if collect_stats:
        info["k_scale"] = _summarize_scale(k_scale)
        info["v_scale"] = _summarize_scale(v_scale)
    return (qk, qv), info


def maybe_quantize_kv(
    kv: Tuple[torch.Tensor, torch.Tensor],
    config: Optional[Dict],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if not config or not config.get("enabled", False):
        return kv, None

    scheme = resolve_scheme(config, None)
    axis = config.get("axis", "head")
    eps = config.get("eps", 1e-6)
    collect_stats = config.get("collect_stats", False)

    key, value = kv
    if key.numel() == 0 or value.numel() == 0:
        info = {"scheme": scheme, "axis": axis, "skipped": "empty"} if collect_stats else None
        return kv, info

    if scheme in NO_QUANT_SCHEMES:
        info = {"scheme": scheme, "axis": axis, "skipped": "no_quant"} if collect_stats else None
        return kv, info

    return quantize_kv(kv, scheme=scheme, axis=axis, eps=eps, collect_stats=collect_stats)


def maybe_quantize_kv_with_layer(
    kv: Tuple[torch.Tensor, torch.Tensor],
    config: Optional[Dict],
    layer_idx: Optional[int],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if not config or not config.get("enabled", False):
        return kv, None

    scheme = resolve_scheme(config, layer_idx)
    axis = config.get("axis", "head")
    eps = config.get("eps", 1e-6)
    collect_stats = config.get("collect_stats", False)

    key, value = kv
    if key.numel() == 0 or value.numel() == 0:
        info = {"scheme": scheme, "axis": axis, "skipped": "empty"} if collect_stats else None
        return kv, info

    if scheme in NO_QUANT_SCHEMES:
        info = {"scheme": scheme, "axis": axis, "skipped": "no_quant"} if collect_stats else None
        return kv, info

    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported kv_quant_scheme: {scheme}")

    return quantize_kv(kv, scheme=scheme, axis=axis, eps=eps, collect_stats=collect_stats)
