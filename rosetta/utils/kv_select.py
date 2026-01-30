"""Token-level KV selection utilities for sparse cache transfer."""
from typing import Dict, Optional, Tuple

import torch


def _validate_inputs(k: torch.Tensor, v: torch.Tensor):
    if k is None or v is None:
        raise ValueError("K/V tensors must be provided")
    if k.shape != v.shape:
        raise ValueError(f"K/V shape mismatch: {k.shape} vs {v.shape}")
    if k.dim() != 4:
        raise ValueError(f"Expected KV shape [B, H, T, D], got {k.shape}")


def _scores_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # tensor: [B, H, T, D] -> scores: [T]
    scores = torch.linalg.norm(tensor.float(), dim=-1)  # [B, H, T]
    scores = scores.mean(dim=(0, 1))
    return scores


def compute_token_scores(
    k: torch.Tensor,
    v: torch.Tensor,
    mode: str,
    scope_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _validate_inputs(k, v)
    mode = (mode or "vnorm_topk").lower()
    t = k.shape[2]

    if mode == "vnorm_topk":
        scores = _scores_from_tensor(v)
    elif mode == "knorm_topk":
        scores = _scores_from_tensor(k)
    elif mode == "random":
        scores = torch.rand(t, device=k.device)
    elif mode == "front":
        scores = torch.linspace(1.0, 0.0, steps=t, device=k.device)
    elif mode == "back":
        scores = torch.linspace(0.0, 1.0, steps=t, device=k.device)
    else:
        raise ValueError(f"Unsupported token_select_mode: {mode}")

    if scope_mask is not None:
        if scope_mask.shape[-1] != t:
            raise ValueError("scope_mask length mismatch")
        scores = scores.masked_fill(~scope_mask.to(dtype=torch.bool, device=k.device), float("-inf"))
    return scores


def select_topk(
    scores: torch.Tensor,
    proportion: float,
    min_tokens: int,
) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    t = scores.shape[0]
    proportion = float(proportion)
    if proportion >= 1.0:
        return torch.arange(t, device=scores.device, dtype=torch.long)
    k = int(round(t * proportion))
    k = max(int(min_tokens), k)
    k = min(k, t)

    finite_mask = torch.isfinite(scores)
    allowed = int(finite_mask.sum().item())
    if allowed <= 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    if k > allowed:
        k = allowed

    values, indices = torch.topk(scores, k=k, largest=True, sorted=False)
    indices = indices[torch.isfinite(values)]
    if indices.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    return torch.sort(indices).values


def select_token_indices(
    k: torch.Tensor,
    v: torch.Tensor,
    config: Dict,
    scope_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    _validate_inputs(k, v)
    mode = (config.get("token_select_mode") or "vnorm_topk").lower()
    proportion = float(config.get("token_select_proportion", 1.0))
    min_tokens = int(config.get("token_select_min_tokens", 1))
    scores = compute_token_scores(k, v, mode=mode, scope_mask=scope_mask)
    idx = select_topk(scores, proportion=proportion, min_tokens=min_tokens)
    score_mean = None
    score_min = None
    score_max = None
    if idx.numel() > 0:
        sel_scores = scores[idx]
        finite_mask = torch.isfinite(sel_scores)
        if finite_mask.any():
            finite_scores = sel_scores[finite_mask]
            score_mean = float(finite_scores.mean().item())
            score_min = float(finite_scores.min().item())
            score_max = float(finite_scores.max().item())
    stats = {
        "selected_tokens": int(idx.numel()),
        "total_tokens": int(k.shape[2]),
    }
    if score_mean is not None:
        stats["score_mean"] = score_mean
        stats["score_min"] = score_min
        stats["score_max"] = score_max
    if stats["total_tokens"] > 0:
        stats["selected_fraction"] = float(stats["selected_tokens"] / stats["total_tokens"])
    return idx, stats
