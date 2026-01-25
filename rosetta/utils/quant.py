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


def _get_qmax(num_bits: int) -> int:
    return (2 ** (num_bits - 1)) - 1


def _resolve_axis(axis: str) -> Tuple[int, ...]:
    if axis == "head":
        # Expect KV shape: [batch, heads, tokens, head_dim]
        return (0, 2, 3)
    if axis == "layer":
        return (0, 1, 2, 3)
    raise ValueError(f"Unsupported kv_quant_axis: {axis}")


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


def quantize_kv(
    kv: Tuple[torch.Tensor, torch.Tensor],
    scheme: str = "int8",
    axis: str = "head",
    eps: float = 1e-6,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(f"Unsupported kv_quant_scheme: {scheme}")

    num_bits = SUPPORTED_SCHEMES[scheme]
    key, value = kv
    qk, k_scale = quantize_dequantize(key, num_bits=num_bits, axis=axis, eps=eps)
    qv, v_scale = quantize_dequantize(value, num_bits=num_bits, axis=axis, eps=eps)
    info = {
        "scheme": scheme,
        "axis": axis,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }
    return (qk, qv), info


def maybe_quantize_kv(
    kv: Tuple[torch.Tensor, torch.Tensor],
    config: Optional[Dict],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    if not config or not config.get("enabled", False):
        return kv, None

    scheme = config.get("scheme", "int8")
    axis = config.get("axis", "head")
    eps = config.get("eps", 1e-6)
    return quantize_kv(kv, scheme=scheme, axis=axis, eps=eps)
