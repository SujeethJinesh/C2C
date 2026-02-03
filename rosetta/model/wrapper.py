"""
The ensemble of multiple standard transformers LLM models, with automatic kv-cache projection. It shares the same interface as the standard transformers LLM models.
"""

from typing import List, Optional, Union
import math
import os
import time
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json

from rosetta.model.projector import Projector
from rosetta.utils.quant import maybe_quantize_kv_with_layer, quantize_kv, resolve_scheme, NO_QUANT_SCHEMES
from rosetta.utils.kv_select import select_token_indices, select_topk
from rosetta.model.sampling import sample_token
from transformers.utils import ModelOutput
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None

try:
    from quantization.kvwire.kvwire_v1 import (
        KVWireConfig,
        pack as kvwire_pack,
        unpack as kvwire_unpack,
        pack_with_breakdown as kvwire_pack_with_breakdown,
    )
except Exception:
    KVWireConfig = None
    kvwire_pack = None
    kvwire_unpack = None
    kvwire_pack_with_breakdown = None

def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
        new_cache = DynamicCache()
        for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
            new_cache.key_cache.append(k.clone().detach())
            new_cache.value_cache.append(v.clone().detach())
        return new_cache

def hybrid_to_dynamic(hybrid_cache):
    if hybrid_cache is None:
        return None
    if isinstance(hybrid_cache, DynamicCache):
        return hybrid_cache

    # 手动从 HybridCache 提取
    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache
        values = hybrid_cache.value_cache
        assert len(keys) == len(values), "key/value 层数不一致"

        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")

def _stable_argsort_desc(scores: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(scores, descending=True, stable=True)
    except TypeError:
        return torch.argsort(scores, descending=True)

class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM
    """
    def __init__(
        self,
        model_list: List[PreTrainedModel],
        base_model_idx = 0,
        projector_list: List[Projector] = [],
        include_response: bool = False,
        multi_source_fusion_mode: str = "sequential",
        kv_quant_config: Optional[dict] = None,
        kv_transfer_config: Optional[dict] = None,
    ):
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)

        self.projector_dict = {}
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []

        # Multi-source fusion mode: 
        # "sequential" (default): each source updates base cache iteratively
        # "parallel": all sources project from clean base cache, then sum projections
        self.include_response = include_response
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(f"multi_source_fusion_mode must be 'sequential' or 'parallel', got '{multi_source_fusion_mode}'")
        self.multi_source_fusion_mode = multi_source_fusion_mode
        self.kv_quant_config = kv_quant_config or {"enabled": False}
        self.kv_transfer_config = kv_transfer_config or {"enabled": False}
        self.exec_mode = (self.kv_transfer_config.get("exec_mode") or "simultaneous").lower()
        if self.exec_mode not in ["simultaneous", "sequential"]:
            raise ValueError(f"exec_mode must be 'simultaneous' or 'sequential', got '{self.exec_mode}'")
        record_per_sample = bool(self.kv_transfer_config.get("record_per_sample", False))
        sample_limit = int(self.kv_transfer_config.get("sample_limit", 0) or 0)
        if self.kv_quant_config.get("collect_stats", False):
            self._kv_quant_stats = {
                "count": 0,
                "k_scale_min": None,
                "k_scale_max": None,
                "k_scale_mean_sum": 0.0,
                "v_scale_min": None,
                "v_scale_max": None,
                "v_scale_mean_sum": 0.0,
            }
        else:
            self._kv_quant_stats = None
        if self.kv_transfer_config.get("enabled", False):
            self._kv_transfer_stats = {
                "count": 0,
                "selected_min": None,
                "selected_max": None,
                "selected_sum": 0,
                "total_sum": 0,
                "selected_fraction_sum": 0.0,
                "score_mean_sum": 0.0,
                "score_min": None,
                "score_max": None,
                "prefill_count": 0,
                "prefill_time_sum_ms": 0.0,
                "selection_time_sum_ms": 0.0,
                "projection_score_time_sum_ms": 0.0,
                "projection_transfer_time_sum_ms": 0.0,
                "fuse_time_sum_ms": 0.0,
                "rd_int8_sum": 0,
                "rd_int4_sum": 0,
                "rd_drop_sum": 0,
                "rd_budget_bytes_sum": 0.0,
                "rd_bytes_used_sum": 0.0,
                "rd_effective_bits_sum": 0.0,
                "rd_payload_bytes_sum": 0.0,
                "rd_index_bytes_sum": 0.0,
                "rd_scale_bytes_sum": 0.0,
                "wire_bytes_sum": 0.0,
                "wire_payload_bytes_sum": 0.0,
                "wire_index_bytes_sum": 0.0,
                "wire_scale_bytes_sum": 0.0,
                "wire_header_bytes_sum": 0.0,
                "wire_encode_time_sum_ms": 0.0,
                "wire_decode_time_sum_ms": 0.0,
                "budget_cap_bytes_sum": 0.0,
                "budget_actual_bytes_sum": 0.0,
                "budget_slack_bytes_sum": 0.0,
                "record_per_sample": record_per_sample,
                "sample_limit": sample_limit,
                "sample_records": [],
                "sample_accum": None,
            }
        else:
            self._kv_transfer_stats = None

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device
    
    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        """
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        return self
        
    # set projector 
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        Set the projector configuration
        Args:
            source_model_idx: int, the index of the source model
            source_model_layer_idx: int, the index of the source model layer
            target_model_idx: int, the index of the target model
            target_model_layer_idx: int, the index of the target model layer
            projector_idx: int, the index of the projector

        The projector dict structure supports multiple projectors per target layer.
        Structure:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))


    def load_projector(self, projector_list):
        self.projector_list: List[Projector] = projector_list

    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx):
        pair_list = self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        return self.projector_list[pair_list[0][1]]

    def set_kv_quant_config(self, kv_quant_config: Optional[dict]):
        self.kv_quant_config = kv_quant_config or {"enabled": False}
        if self.kv_quant_config.get("collect_stats", False):
            self._kv_quant_stats = {
                "count": 0,
                "k_scale_min": None,
                "k_scale_max": None,
                "k_scale_mean_sum": 0.0,
                "v_scale_min": None,
                "v_scale_max": None,
                "v_scale_mean_sum": 0.0,
            }
        else:
            self._kv_quant_stats = None

    def set_kv_transfer_config(self, kv_transfer_config: Optional[dict]):
        self.kv_transfer_config = kv_transfer_config or {"enabled": False}
        if self.kv_transfer_config.get("enabled", False):
            self._kv_transfer_stats = {
                "count": 0,
                "selected_min": None,
                "selected_max": None,
                "selected_sum": 0,
                "total_sum": 0,
                "selected_fraction_sum": 0.0,
                "score_mean_sum": 0.0,
                "score_min": None,
                "score_max": None,
                "prefill_count": 0,
                "prefill_time_sum_ms": 0.0,
                "selection_time_sum_ms": 0.0,
                "projection_score_time_sum_ms": 0.0,
                "projection_transfer_time_sum_ms": 0.0,
                "fuse_time_sum_ms": 0.0,
                "rd_int8_sum": 0,
                "rd_int4_sum": 0,
                "rd_drop_sum": 0,
                "rd_budget_bytes_sum": 0.0,
                "rd_bytes_used_sum": 0.0,
                "rd_effective_bits_sum": 0.0,
                "rd_payload_bytes_sum": 0.0,
                "rd_index_bytes_sum": 0.0,
                "rd_scale_bytes_sum": 0.0,
            }
        else:
            self._kv_transfer_stats = None

    def get_kv_quant_stats(self):
        if self._kv_quant_stats is None:
            return None
        stats = dict(self._kv_quant_stats)
        if stats["count"] > 0:
            stats["k_scale_mean"] = stats["k_scale_mean_sum"] / stats["count"]
            stats["v_scale_mean"] = stats["v_scale_mean_sum"] / stats["count"]
        return stats

    def get_kv_transfer_stats(self):
        if self._kv_transfer_stats is None:
            return None
        stats = dict(self._kv_transfer_stats)
        stats.pop("sample_accum", None)
        count = stats.get("count", 0)
        if count > 0:
            stats["selected_mean"] = stats["selected_sum"] / stats["count"]
            stats["total_mean"] = stats["total_sum"] / stats["count"]
            if "selected_fraction_sum" in stats:
                stats["selected_fraction_mean"] = stats["selected_fraction_sum"] / count
            elif stats.get("total_sum", 0) > 0:
                stats["selected_fraction_mean"] = stats["selected_sum"] / stats["total_sum"]
            if "score_mean_sum" in stats:
                stats["score_mean"] = stats["score_mean_sum"] / stats["count"]
            if stats.get("prefill_count", 0) > 0:
                stats["prefill_time_mean_ms"] = stats.get("prefill_time_sum_ms", 0.0) / stats["prefill_count"]
            stats["selection_time_mean_ms"] = stats.get("selection_time_sum_ms", 0.0) / count
            stats["projection_score_time_mean_ms"] = stats.get("projection_score_time_sum_ms", 0.0) / count
            stats["projection_transfer_time_mean_ms"] = stats.get("projection_transfer_time_sum_ms", 0.0) / count
            stats["fuse_time_mean_ms"] = stats.get("fuse_time_sum_ms", 0.0) / count
            stats["rd_int8_mean"] = stats.get("rd_int8_sum", 0.0) / count
            stats["rd_int4_mean"] = stats.get("rd_int4_sum", 0.0) / count
            stats["rd_drop_mean"] = stats.get("rd_drop_sum", 0.0) / count
            stats["rd_budget_bytes_mean"] = stats.get("rd_budget_bytes_sum", 0.0) / count
            stats["rd_bytes_used_mean"] = stats.get("rd_bytes_used_sum", 0.0) / count
            stats["rd_effective_bits_mean"] = stats.get("rd_effective_bits_sum", 0.0) / count
            stats["rd_payload_bytes_mean"] = stats.get("rd_payload_bytes_sum", 0.0) / count
            stats["rd_index_bytes_mean"] = stats.get("rd_index_bytes_sum", 0.0) / count
            stats["rd_scale_bytes_mean"] = stats.get("rd_scale_bytes_sum", 0.0) / count
            stats["wire_bytes_mean"] = stats.get("wire_bytes_sum", 0.0) / count
            stats["wire_payload_bytes_mean"] = stats.get("wire_payload_bytes_sum", 0.0) / count
            stats["wire_index_bytes_mean"] = stats.get("wire_index_bytes_sum", 0.0) / count
            stats["wire_scale_bytes_mean"] = stats.get("wire_scale_bytes_sum", 0.0) / count
            stats["wire_header_bytes_mean"] = stats.get("wire_header_bytes_sum", 0.0) / count
            stats["wire_encode_time_mean_ms"] = stats.get("wire_encode_time_sum_ms", 0.0) / count
            stats["wire_decode_time_mean_ms"] = stats.get("wire_decode_time_sum_ms", 0.0) / count
            if stats.get("prefill_count", 0) > 0:
                stats["budget_cap_bytes_mean"] = stats.get("budget_cap_bytes_sum", 0.0) / stats["prefill_count"]
                stats["budget_actual_bytes_mean"] = stats.get("budget_actual_bytes_sum", 0.0) / stats["prefill_count"]
                stats["budget_slack_bytes_mean"] = stats.get("budget_slack_bytes_sum", 0.0) / stats["prefill_count"]
        return stats

    def _update_kv_transfer_stats(self, selected_tokens: int, total_tokens: int, extra: Optional[dict] = None):
        if self._kv_transfer_stats is None:
            return
        stats = self._kv_transfer_stats
        stats["count"] += 1
        stats["selected_sum"] += int(selected_tokens)
        stats["total_sum"] += int(total_tokens)
        stats["selected_min"] = (
            selected_tokens if stats["selected_min"] is None else min(stats["selected_min"], selected_tokens)
        )
        stats["selected_max"] = (
            selected_tokens if stats["selected_max"] is None else max(stats["selected_max"], selected_tokens)
        )
        sample_accum = stats.get("sample_accum")
        if sample_accum is not None:
            sample_accum["selected_tokens"] += int(selected_tokens)
            sample_accum["total_tokens"] += int(total_tokens)
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                if key.endswith("_min"):
                    stats[key] = value if stats.get(key) is None else min(stats[key], value)
                elif key.endswith("_max"):
                    stats[key] = value if stats.get(key) is None else max(stats[key], value)
                else:
                    stats[key] = stats.get(key, 0) + value
                if sample_accum is not None:
                    if key == "wire_bytes_sum":
                        sample_accum["wire_bytes"] += float(value)
                    elif key == "wire_payload_bytes_sum":
                        sample_accum["wire_payload_bytes"] += float(value)
                    elif key == "wire_index_bytes_sum":
                        sample_accum["wire_index_bytes"] += float(value)
                    elif key == "wire_scale_bytes_sum":
                        sample_accum["wire_scale_bytes"] += float(value)
                    elif key == "wire_header_bytes_sum":
                        sample_accum["wire_header_bytes"] += float(value)
                    elif key == "rd_int8_sum":
                        sample_accum["rd_int8"] += int(value)
                    elif key == "rd_int4_sum":
                        sample_accum["rd_int4"] += int(value)
                    elif key == "rd_drop_sum":
                        sample_accum["rd_drop"] += int(value)

    def _update_prefill_stats(self, prefill_time_ms: float):
        if self._kv_transfer_stats is None:
            return
        stats = self._kv_transfer_stats
        stats["prefill_count"] = stats.get("prefill_count", 0) + 1
        stats["prefill_time_sum_ms"] = stats.get("prefill_time_sum_ms", 0.0) + float(prefill_time_ms)

    def set_wire_cache_key(self, key: Optional[str]):
        if self.kv_transfer_config is None:
            return
        self.kv_transfer_config["wire_cache_key"] = key

    def _start_transfer_sample(self):
        if self._kv_transfer_stats is None:
            return
        stats = self._kv_transfer_stats
        if not stats.get("record_per_sample", False):
            return
        stats["sample_accum"] = {
            "selected_tokens": 0,
            "total_tokens": 0,
            "wire_bytes": 0.0,
            "wire_payload_bytes": 0.0,
            "wire_index_bytes": 0.0,
            "wire_scale_bytes": 0.0,
            "wire_header_bytes": 0.0,
            "rd_int8": 0,
            "rd_int4": 0,
            "rd_drop": 0,
        }

    def _finalize_transfer_sample(self):
        if self._kv_transfer_stats is None:
            return
        stats = self._kv_transfer_stats
        if not stats.get("record_per_sample", False):
            return
        accum = stats.get("sample_accum")
        if not accum:
            return
        budget_cap = self.kv_transfer_config.get("wire_budget_bytes")
        budget_cap = float(budget_cap) if budget_cap is not None else None
        actual = float(accum.get("wire_bytes", 0.0))
        slack = None
        if budget_cap is not None:
            slack = budget_cap - actual
            if slack < 0:
                slack = 0.0
            stats["budget_cap_bytes_sum"] = stats.get("budget_cap_bytes_sum", 0.0) + float(budget_cap)
            stats["budget_actual_bytes_sum"] = stats.get("budget_actual_bytes_sum", 0.0) + float(actual)
            stats["budget_slack_bytes_sum"] = stats.get("budget_slack_bytes_sum", 0.0) + float(slack)

        record = dict(accum)
        if budget_cap is not None:
            record["budget_cap_bytes"] = float(budget_cap)
            record["budget_actual_bytes"] = float(actual)
            record["budget_slack_bytes"] = float(slack or 0.0)

        sample_records = stats.get("sample_records")
        sample_limit = int(stats.get("sample_limit", 0) or 0)
        if isinstance(sample_records, list):
            if sample_limit <= 0 or len(sample_records) < sample_limit:
                sample_records.append(record)
        stats["sample_accum"] = None

    def _project_with_transfer(
        self,
        source_kv_cache,
        base_kv_cache,
        projector,
        target_layer_idx: int,
        transfer_cache: dict,
        cache_key=None,
        scope_mask: Optional[torch.Tensor] = None,
    ):
        """Apply optional token selection + quantization before projecting."""
        transfer_cfg = self.kv_transfer_config or {}
        transfer_enabled = bool(transfer_cfg.get("enabled", False))
        sparse_fuse = bool(transfer_cfg.get("sparse_fuse", True))
        scatter_fill = transfer_cfg.get("scatter_fill", "receiver_only")
        token_precision_mode = (transfer_cfg.get("token_precision_mode") or "").lower()

        base_key_cache, base_value_cache = base_kv_cache
        if source_kv_cache is None:
            key_cache = base_key_cache
            value_cache = base_value_cache
        else:
            key_cache, value_cache = source_kv_cache
        seq_len = key_cache.shape[2]

        if scope_mask is not None:
            scope_mask = scope_mask.to(device=key_cache.device, dtype=torch.bool)

        def _apply_scope(scores: torch.Tensor) -> torch.Tensor:
            if scope_mask is None:
                return scores
            if scope_mask.shape[0] != scores.shape[0]:
                raise ValueError("scope_mask length mismatch")
            return scores.masked_fill(~scope_mask, float("-inf"))

        timing_sync = bool(transfer_cfg.get("timing_sync", False))

        def _sync():
            if timing_sync and key_cache.is_cuda:
                torch.cuda.synchronize(key_cache.device)

        def _bytes_per_token(bits_per_elem: float, index_bytes: float) -> float:
            heads = int(key_cache.shape[1])
            head_dim = int(key_cache.shape[3])
            payload = 2 * heads * head_dim * (bits_per_elem / 8.0)
            return payload + float(index_bytes)

        def _scale_overhead_bytes(axis: str) -> float:
            # Approximate: fp32 scale per head for K and V
            heads = int(key_cache.shape[1])
            if axis == "head":
                return float(2 * heads * 4)
            return float(2 * 4)

        wire_format = (transfer_cfg.get("wire_format") or "").lower()
        wire_enabled = wire_format == "kvwire_v1"
        wire_apply_pack = bool(transfer_cfg.get("wire_apply_pack", False))
        wire_quant_mode = (transfer_cfg.get("wire_quant_mode") or "").lower()
        wire_scale_granularity = (transfer_cfg.get("wire_scale_granularity") or "per_block").lower()
        wire_scale_dtype = (transfer_cfg.get("wire_scale_dtype") or "fp16").lower()
        wire_include_headers = bool(transfer_cfg.get("wire_include_headers", True))
        wire_index_bytes = float(transfer_cfg.get("wire_index_dtype_bytes") or transfer_cfg.get("index_dtype_bytes") or 0.0)
        wire_cache_dir = transfer_cfg.get("wire_cache_dir")
        wire_cache_mode = (transfer_cfg.get("wire_cache_mode") or "off").lower()
        wire_cache_key = transfer_cfg.get("wire_cache_key")
        wire_cache_tag = transfer_cfg.get("wire_cache_tag") or ""
        wire_cache_suffix = transfer_cfg.get("wire_cache_suffix") or ""

        def _wire_payload_bytes(token_count: int, quant_mode: str) -> float:
            if token_count <= 0:
                return 0.0
            batch = int(key_cache.shape[0])
            heads = int(key_cache.shape[1])
            head_dim = int(key_cache.shape[3])
            elems = batch * heads * token_count * head_dim * 2
            mode = (quant_mode or "int8").lower()
            if mode == "int4":
                return float((elems + 1) // 2)
            if mode == "int8":
                return float(elems)
            if mode in ("fp16", "bf16"):
                return float(elems * 2)
            return float(elems)

        def _wire_scale_bytes(token_count: int) -> float:
            if token_count <= 0:
                return 0.0
            batch = int(key_cache.shape[0])
            heads = int(key_cache.shape[1])
            scale_bytes = 2 if wire_scale_dtype in ("fp16", "bf16", "bfloat16") else 4
            if wire_scale_granularity == "per_block":
                scale_count = 2 * batch * heads * token_count
            elif wire_scale_granularity == "per_head":
                scale_count = 2 * batch * heads
            else:  # per_tensor/per_layer
                scale_count = 2
            return float(scale_count * scale_bytes)

        def _estimate_wire_header_bytes(token_count: int, payload_bytes: float, scale_bytes: float, index_bytes: float) -> float:
            if not wire_enabled or not wire_include_headers:
                return 0.0
            meta = {
                "version": "kvwire_v1",
                "quant_mode": wire_quant_mode or "int8",
                "index_dtype_bytes": int(wire_index_bytes),
                "scale_dtype": wire_scale_dtype,
                "scale_granularity": wire_scale_granularity,
                "tokens": int(token_count),
                "sections": {
                    "indices": int(index_bytes),
                    "payload": int(payload_bytes),
                    "scales": int(scale_bytes),
                },
            }
            payload = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            return float(12 + len(payload))

        def _wire_cache_path(group: str, quant_mode: str) -> Optional[str]:
            if not wire_cache_dir or not wire_cache_key:
                return None
            tag = wire_cache_tag
            key = wire_cache_key
            suffix = wire_cache_suffix
            name_parts = [p for p in [tag, key, suffix] if p]
            base = "_".join(name_parts) if name_parts else "wire"
            group_tag = group or "all"
            fname = f"{base}_layer{int(target_layer_idx)}_{group_tag}_{quant_mode}.kvw"
            return os.path.join(wire_cache_dir, fname)

        def _load_wire_blob(path: Optional[str]):
            if not path or not wire_enabled or kvwire_unpack is None:
                return None
            if not os.path.exists(path):
                return None
            try:
                blob = None
                with open(path, "rb") as handle:
                    blob = handle.read()
                if not blob:
                    return None
                t_decode = time.perf_counter()
                decoded = kvwire_unpack(blob, KVWireConfig())
                decode_ms = (time.perf_counter() - t_decode) * 1000.0
                k_dec = torch.from_numpy(decoded["k"]).to(device=base_key_cache.device, dtype=base_key_cache.dtype)
                v_dec = torch.from_numpy(decoded["v"]).to(device=base_value_cache.device, dtype=base_value_cache.dtype)
                idx_np = decoded.get("indices")
                idx = torch.from_numpy(idx_np).to(device=base_key_cache.device, dtype=torch.long) if idx_np is not None else None
                meta = decoded.get("meta") or {}
                sections = meta.get("sections") or {}
                payload_bytes = int(sections.get("k_quant", 0)) + int(sections.get("v_quant", 0))
                index_bytes = int(sections.get("indices", 0))
                scale_bytes = int(sections.get("scales", 0))
                header_bytes = int(len(blob) - payload_bytes - index_bytes - scale_bytes)
                breakdown = {
                    "payload_bytes": float(payload_bytes),
                    "index_bytes": float(index_bytes),
                    "scale_bytes": float(scale_bytes),
                    "header_bytes": float(header_bytes),
                    "total_bytes": float(len(blob)),
                }
                return (k_dec, v_dec), idx, decode_ms, breakdown
            except Exception:
                return None

        def _maybe_apply_wire_pack(source_sel, indices, quant_mode: str, apply_decoded: bool = True, cache_path: Optional[str] = None):
            if not wire_enabled:
                return source_sel, 0.0, 0.0, None
            if not wire_apply_pack:
                return source_sel, 0.0, 0.0, None
            if KVWireConfig is None or kvwire_pack_with_breakdown is None or kvwire_unpack is None:
                return source_sel, 0.0, 0.0, None
            try:
                k_np = source_sel[0].detach().cpu().numpy()
                v_np = source_sel[1].detach().cpu().numpy()
                idx_np = indices.detach().cpu().numpy() if indices is not None else None
                cfg = KVWireConfig(
                    wire_index_dtype=transfer_cfg.get("wire_index_dtype", "uint16"),
                    wire_scale_dtype=wire_scale_dtype,
                    wire_quant_mode=quant_mode,
                    wire_scale_granularity=wire_scale_granularity,
                    wire_include_headers=wire_include_headers,
                    wire_version="kvwire_v1",
                    wire_compression=transfer_cfg.get("wire_compression", "none"),
                )
                t_encode = time.perf_counter()
                blob, breakdown = kvwire_pack_with_breakdown({"k": k_np, "v": v_np, "indices": idx_np}, cfg)
                encode_ms = (time.perf_counter() - t_encode) * 1000.0
                if cache_path:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, "wb") as handle:
                        handle.write(blob)
                t_decode = time.perf_counter()
                decoded = kvwire_unpack(blob, cfg)
                decode_ms = (time.perf_counter() - t_decode) * 1000.0
                if apply_decoded:
                    k_dec = torch.from_numpy(decoded["k"]).to(device=source_sel[0].device, dtype=source_sel[0].dtype)
                    v_dec = torch.from_numpy(decoded["v"]).to(device=source_sel[1].device, dtype=source_sel[1].dtype)
                    return (k_dec, v_dec), encode_ms, decode_ms, breakdown
                return source_sel, encode_ms, decode_ms, breakdown
            except Exception:
                return source_sel, 0.0, 0.0, None

        def _apply_wire_breakdown(extra_stats: dict, breakdown: Optional[dict], encode_ms: float, decode_ms: float):
            if not breakdown:
                return
            extra_stats["wire_bytes_sum"] = float(breakdown.get("total_bytes", 0.0))
            extra_stats["wire_payload_bytes_sum"] = float(breakdown.get("payload_bytes", 0.0))
            extra_stats["wire_index_bytes_sum"] = float(breakdown.get("index_bytes", 0.0))
            extra_stats["wire_scale_bytes_sum"] = float(breakdown.get("scale_bytes", 0.0))
            extra_stats["wire_header_bytes_sum"] = float(breakdown.get("header_bytes", 0.0))
            extra_stats["wire_encode_time_sum_ms"] = float(encode_ms)
            extra_stats["wire_decode_time_sum_ms"] = float(decode_ms)

        def _score_stats(scores: torch.Tensor, idx_tensor: torch.Tensor) -> dict:
            if idx_tensor is None or idx_tensor.numel() == 0:
                return {}
            sel_scores = scores[idx_tensor]
            finite_mask = torch.isfinite(sel_scores)
            if not finite_mask.any():
                return {}
            finite_scores = sel_scores[finite_mask]
            return {
                "score_mean": float(finite_scores.mean().item()),
                "score_min": float(finite_scores.min().item()),
                "score_max": float(finite_scores.max().item()),
            }

        projection_score_time_ms = 0.0
        selection_time_ms = 0.0
        projection_transfer_time_ms = 0.0
        fuse_time_ms = 0.0

        # RD-C2C path: token x precision allocation
        if transfer_enabled and token_precision_mode == "rd_greedy":
            axis = (self.kv_quant_config or {}).get("axis", "head")
            eps = (self.kv_quant_config or {}).get("eps", 1e-6)
            collect_stats = bool((self.kv_quant_config or {}).get("collect_stats", False))
            candidates = [str(c).lower() for c in (transfer_cfg.get("token_precision_candidates") or ["drop", "int4", "int8"])]
            budget_bytes = transfer_cfg.get("token_precision_budget_bytes")
            budget_bytes = float(budget_bytes) if budget_bytes is not None else float("inf")
            include_scale_overhead = bool(transfer_cfg.get("include_scale_overhead", False))
            index_bytes = float(transfer_cfg.get("index_dtype_bytes") or 0.0)
            use_cached_rd = False
            cached_int8 = None
            cached_int4 = None
            if wire_cache_mode == "read" and wire_cache_dir and wire_cache_key:
                if "int8" in candidates:
                    cached_int8 = _load_wire_blob(_wire_cache_path("int8", "int8"))
                if "int4" in candidates:
                    cached_int4 = _load_wire_blob(_wire_cache_path("int4", "int4"))
                if cached_int8 or cached_int4:
                    use_cached_rd = True
                elif source_kv_cache is None:
                    raise RuntimeError("wire_cache_mode=read but no cached RD blobs found")
            if source_kv_cache is None and not use_cached_rd:
                raise RuntimeError("source_kv_cache missing for RD transfer")

            # Use delta-projection scores (receiver-space marginal update)
            if use_cached_rd:
                scores = None
                order = None
            else:
                _sync()
                t_score = time.perf_counter()
                source_for_score, _ = maybe_quantize_kv_with_layer(
                    source_kv_cache,
                    self.kv_quant_config,
                    target_layer_idx,
                )
                proj_key_full, proj_val_full = projector.forward(source_for_score, base_kv_cache)
                _sync()
                projection_score_time_ms = (time.perf_counter() - t_score) * 1000.0

                _sync()
                t_select = time.perf_counter()
                scores = torch.linalg.norm((proj_val_full - base_value_cache).float(), dim=-1).mean(dim=(0, 1))
                scores = _apply_scope(scores)

                order = _stable_argsort_desc(scores)
                _sync()
                selection_time_ms = (time.perf_counter() - t_select) * 1000.0

            bytes_int8 = _bytes_per_token(8.0, index_bytes)
            bytes_int4 = _bytes_per_token(4.0, index_bytes)
            scale_overhead = _scale_overhead_bytes(axis) if include_scale_overhead else 0.0

            idx_int8 = []
            idx_int4 = []
            remaining = float(budget_bytes)
            used_int8 = False
            used_int4 = False
            if use_cached_rd:
                if cached_int8:
                    idx_tensor = cached_int8[1]
                    idx_int8 = idx_tensor.tolist() if idx_tensor is not None else []
                if cached_int4:
                    idx_tensor = cached_int4[1]
                    idx_int4 = idx_tensor.tolist() if idx_tensor is not None else []
            else:
                for idx in order.tolist():
                    if not torch.isfinite(scores[idx]):
                        break
                    if "int8" in candidates:
                        cost = bytes_int8 + (scale_overhead if (include_scale_overhead and not used_int8) else 0.0)
                        if remaining >= cost:
                            idx_int8.append(idx)
                            remaining -= cost
                            used_int8 = True
                            continue
                    if "int4" in candidates:
                        cost = bytes_int4 + (scale_overhead if (include_scale_overhead and not used_int4) else 0.0)
                        if remaining >= cost:
                            idx_int4.append(idx)
                            remaining -= cost
                            used_int4 = True
                            continue

            selected_tokens = len(idx_int8) + len(idx_int4)
            selected_fraction = (selected_tokens / float(seq_len)) if seq_len > 0 else 0.0
            idx_all = idx_int8 + idx_int4
            idx_tensor_all = (
                torch.tensor(idx_all, device=key_cache.device, dtype=torch.long)
                if idx_all
                else torch.empty((0,), device=key_cache.device, dtype=torch.long)
            )
            score_stats = _score_stats(scores, idx_tensor_all) if scores is not None else {}

            _sync()
            t_fuse = time.perf_counter()
            proj_key_full = base_key_cache.clone()
            proj_val_full = base_value_cache.clone()
            _sync()
            fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0

            q_info = {
                "rd_groups": {
                    "int8": int(len(idx_int8)),
                    "int4": int(len(idx_int4)),
                    "drop": int(seq_len - selected_tokens),
                },
                "rd_budget_bytes": float(budget_bytes),
            }
            wire_encode_ms = 0.0
            wire_decode_ms = 0.0
            wire_payload_bytes = 0.0
            wire_index_bytes = 0.0
            wire_scale_bytes = 0.0
            wire_header_bytes = 0.0
            wire_total_bytes = 0.0
            wire_used_breakdown = False

            if idx_int8:
                if use_cached_rd and cached_int8:
                    source_sel_q, idx_tensor, dec_ms, breakdown = cached_int8
                    base_sel = (base_key_cache[:, :, idx_tensor, :], base_value_cache[:, :, idx_tensor, :])
                    _sync()
                    t_proj = time.perf_counter()
                    proj_key_sel, proj_val_sel = projector.forward(source_sel_q, base_sel)
                    _sync()
                    projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
                    q_info8 = {"scheme": "int8", "cached": True}
                    wire_decode_ms += dec_ms
                    if breakdown:
                        wire_used_breakdown = True
                        wire_payload_bytes += float(breakdown.get("payload_bytes", 0.0))
                        wire_index_bytes += float(breakdown.get("index_bytes", 0.0))
                        wire_scale_bytes += float(breakdown.get("scale_bytes", 0.0))
                        wire_header_bytes += float(breakdown.get("header_bytes", 0.0))
                        wire_total_bytes += float(breakdown.get("total_bytes", 0.0))
                else:
                    idx_tensor = torch.tensor(idx_int8, device=key_cache.device, dtype=torch.long)
                    source_sel = (key_cache[:, :, idx_tensor, :], value_cache[:, :, idx_tensor, :])
                    base_sel = (base_key_cache[:, :, idx_tensor, :], base_value_cache[:, :, idx_tensor, :])
                    _sync()
                    t_proj = time.perf_counter()
                    source_sel_q, q_info8 = quantize_kv(source_sel, scheme="int8", axis=axis, eps=eps, collect_stats=collect_stats)
                    cache_path = _wire_cache_path("int8", "int8") if wire_cache_mode == "write" else None
                    source_sel_q, enc_ms, dec_ms, breakdown = _maybe_apply_wire_pack(source_sel_q, idx_tensor, "int8", cache_path=cache_path)
                    wire_encode_ms += enc_ms
                    wire_decode_ms += dec_ms
                    if breakdown:
                        wire_used_breakdown = True
                        wire_payload_bytes += float(breakdown.get("payload_bytes", 0.0))
                        wire_index_bytes += float(breakdown.get("index_bytes", 0.0))
                        wire_scale_bytes += float(breakdown.get("scale_bytes", 0.0))
                        wire_header_bytes += float(breakdown.get("header_bytes", 0.0))
                        wire_total_bytes += float(breakdown.get("total_bytes", 0.0))
                    proj_key_sel, proj_val_sel = projector.forward(source_sel_q, base_sel)
                _sync()
                if not (use_cached_rd and cached_int8):
                    projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
                    _sync()
                    t_fuse = time.perf_counter()
                proj_key_full[:, :, idx_tensor, :] = proj_key_sel.to(dtype=proj_key_full.dtype)
                proj_val_full[:, :, idx_tensor, :] = proj_val_sel.to(dtype=proj_val_full.dtype)
                if not (use_cached_rd and cached_int8):
                    _sync()
                    fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0
                q_info["int8"] = q_info8

            if idx_int4:
                if use_cached_rd and cached_int4:
                    source_sel_q, idx_tensor, dec_ms, breakdown = cached_int4
                    base_sel = (base_key_cache[:, :, idx_tensor, :], base_value_cache[:, :, idx_tensor, :])
                    _sync()
                    t_proj = time.perf_counter()
                    proj_key_sel, proj_val_sel = projector.forward(source_sel_q, base_sel)
                    _sync()
                    projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
                    q_info4 = {"scheme": "int4", "cached": True}
                    wire_decode_ms += dec_ms
                    if breakdown:
                        wire_used_breakdown = True
                        wire_payload_bytes += float(breakdown.get("payload_bytes", 0.0))
                        wire_index_bytes += float(breakdown.get("index_bytes", 0.0))
                        wire_scale_bytes += float(breakdown.get("scale_bytes", 0.0))
                        wire_header_bytes += float(breakdown.get("header_bytes", 0.0))
                        wire_total_bytes += float(breakdown.get("total_bytes", 0.0))
                else:
                    idx_tensor = torch.tensor(idx_int4, device=key_cache.device, dtype=torch.long)
                    source_sel = (key_cache[:, :, idx_tensor, :], value_cache[:, :, idx_tensor, :])
                    base_sel = (base_key_cache[:, :, idx_tensor, :], base_value_cache[:, :, idx_tensor, :])
                    _sync()
                    t_proj = time.perf_counter()
                    source_sel_q, q_info4 = quantize_kv(source_sel, scheme="int4", axis=axis, eps=eps, collect_stats=collect_stats)
                    cache_path = _wire_cache_path("int4", "int4") if wire_cache_mode == "write" else None
                    source_sel_q, enc_ms, dec_ms, breakdown = _maybe_apply_wire_pack(source_sel_q, idx_tensor, "int4", cache_path=cache_path)
                    wire_encode_ms += enc_ms
                    wire_decode_ms += dec_ms
                    if breakdown:
                        wire_used_breakdown = True
                        wire_payload_bytes += float(breakdown.get("payload_bytes", 0.0))
                        wire_index_bytes += float(breakdown.get("index_bytes", 0.0))
                        wire_scale_bytes += float(breakdown.get("scale_bytes", 0.0))
                        wire_header_bytes += float(breakdown.get("header_bytes", 0.0))
                        wire_total_bytes += float(breakdown.get("total_bytes", 0.0))
                    proj_key_sel, proj_val_sel = projector.forward(source_sel_q, base_sel)
                _sync()
                if not (use_cached_rd and cached_int4):
                    projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
                    _sync()
                    t_fuse = time.perf_counter()
                proj_key_full[:, :, idx_tensor, :] = proj_key_sel.to(dtype=proj_key_full.dtype)
                proj_val_full[:, :, idx_tensor, :] = proj_val_sel.to(dtype=proj_val_full.dtype)
                if not (use_cached_rd and cached_int4):
                    _sync()
                    fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0
                q_info["int4"] = q_info4

            heads = int(key_cache.shape[1])
            head_dim = int(key_cache.shape[3])
            payload_bytes_int8 = float(2 * heads * head_dim)
            payload_bytes_int4 = float(heads * head_dim)
            payload_bytes = payload_bytes_int8 * len(idx_int8) + payload_bytes_int4 * len(idx_int4)
            index_bytes_total = float(len(idx_int8) + len(idx_int4)) * float(index_bytes)
            scale_bytes_total = 0.0
            if include_scale_overhead:
                if used_int8:
                    scale_bytes_total += scale_overhead
                if used_int4:
                    scale_bytes_total += scale_overhead
            bytes_used = payload_bytes + index_bytes_total + scale_bytes_total
            if math.isfinite(budget_bytes):
                q_info["rd_bytes_used"] = float(budget_bytes - remaining)
            else:
                q_info["rd_bytes_used"] = float(bytes_used)
            effective_bits = None
            if selected_tokens > 0:
                elems_per_token = float(2 * heads * head_dim)
                effective_bits = float((bytes_used * 8.0) / (elems_per_token * selected_tokens))

            extra_stats = {
                "selected_fraction_sum": selected_fraction,
                "selection_time_sum_ms": selection_time_ms,
                "projection_score_time_sum_ms": projection_score_time_ms,
                "projection_transfer_time_sum_ms": projection_transfer_time_ms,
                "fuse_time_sum_ms": fuse_time_ms,
                "rd_int8_sum": int(len(idx_int8)),
                "rd_int4_sum": int(len(idx_int4)),
                "rd_drop_sum": int(seq_len - selected_tokens),
                "rd_budget_bytes_sum": float(budget_bytes) if math.isfinite(budget_bytes) else 0.0,
                "rd_bytes_used_sum": float(bytes_used),
                "rd_effective_bits_sum": float(effective_bits) if effective_bits is not None else 0.0,
                "rd_payload_bytes_sum": float(payload_bytes),
                "rd_index_bytes_sum": float(index_bytes_total),
                "rd_scale_bytes_sum": float(scale_bytes_total),
            }
            if wire_enabled:
                if wire_used_breakdown:
                    extra_stats.update({
                        "wire_bytes_sum": float(wire_total_bytes),
                        "wire_payload_bytes_sum": float(wire_payload_bytes),
                        "wire_index_bytes_sum": float(wire_index_bytes),
                        "wire_scale_bytes_sum": float(wire_scale_bytes),
                        "wire_header_bytes_sum": float(wire_header_bytes),
                        "wire_encode_time_sum_ms": float(wire_encode_ms),
                        "wire_decode_time_sum_ms": float(wire_decode_ms),
                    })
                else:
                    wire_payload = _wire_payload_bytes(len(idx_int8), "int8") + _wire_payload_bytes(len(idx_int4), "int4")
                    wire_index = float(len(idx_int8) + len(idx_int4)) * wire_index_bytes
                    wire_scale = _wire_scale_bytes(len(idx_int8) + len(idx_int4))
                    wire_header = _estimate_wire_header_bytes(len(idx_int8) + len(idx_int4), wire_payload, wire_scale, wire_index)
                    wire_total = wire_payload + wire_index + wire_scale + wire_header
                    extra_stats.update({
                        "wire_bytes_sum": float(wire_total),
                        "wire_payload_bytes_sum": float(wire_payload),
                        "wire_index_bytes_sum": float(wire_index),
                        "wire_scale_bytes_sum": float(wire_scale),
                        "wire_header_bytes_sum": float(wire_header),
                        "wire_encode_time_sum_ms": 0.0,
                        "wire_decode_time_sum_ms": 0.0,
                    })
            if score_stats:
                extra_stats["score_mean_sum"] = score_stats.get("score_mean", 0.0)
                if score_stats.get("score_min") is not None:
                    extra_stats["score_min"] = score_stats.get("score_min")
                if score_stats.get("score_max") is not None:
                    extra_stats["score_max"] = score_stats.get("score_max")

            self._update_kv_transfer_stats(selected_tokens, int(seq_len), extra=extra_stats)

            return (proj_key_full, proj_val_full), q_info

        idx = None
        proj_full_for_reuse = None
        q_info_full = None
        cached_non_rd = None
        cached_non_rd_decode_ms = 0.0
        cached_non_rd_breakdown = None
        if transfer_enabled:
            cache_key = cache_key or (id(source_kv_cache), target_layer_idx, seq_len)
            if cache_key in transfer_cache:
                idx, stats = transfer_cache[cache_key]
            else:
                mode = (transfer_cfg.get("token_select_mode") or "vnorm_topk").lower()
                use_cached_non_rd = False
                cache_mode = None
                if wire_cache_mode == "read" and wire_cache_dir and wire_cache_key and wire_enabled:
                    cache_mode = wire_quant_mode
                    if not cache_mode and self.kv_quant_config and self.kv_quant_config.get("enabled", False):
                        scheme = resolve_scheme(self.kv_quant_config, target_layer_idx)
                        if scheme and scheme.lower() in ("int8", "int4"):
                            cache_mode = scheme.lower()
                    if cache_mode in ("int8", "int4"):
                        cached_non_rd = _load_wire_blob(_wire_cache_path("all", cache_mode))
                        if cached_non_rd:
                            use_cached_non_rd = True
                            cached_non_rd_decode_ms = float(cached_non_rd[2])
                            cached_non_rd_breakdown = cached_non_rd[3]
                        elif source_kv_cache is None:
                            raise RuntimeError("wire_cache_mode=read but no cached blob found")
                if use_cached_non_rd:
                    idx = cached_non_rd[1] if cached_non_rd else torch.empty((0,), device=base_key_cache.device, dtype=torch.long)
                    stats = {"selected_tokens": int(idx.numel()), "total_tokens": int(seq_len)}
                    if stats["total_tokens"] > 0:
                        stats["selected_fraction"] = float(stats["selected_tokens"] / stats["total_tokens"])
                elif mode == "proj_vnorm_topk":
                    # Project full KV once to score in receiver space (projector-aware).
                    _sync()
                    t_score = time.perf_counter()
                    proj_key_full, proj_val_full = projector.forward(source_kv_cache, base_kv_cache)
                    _sync()
                    projection_score_time_ms = (time.perf_counter() - t_score) * 1000.0
                    _sync()
                    t_select = time.perf_counter()
                    scores = torch.linalg.norm(proj_val_full.float(), dim=-1).mean(dim=(0, 1))
                    scores = _apply_scope(scores)
                    idx = select_topk(
                        scores,
                        proportion=float(transfer_cfg.get("token_select_proportion", 1.0)),
                        min_tokens=int(transfer_cfg.get("token_select_min_tokens", 1)),
                    )
                    _sync()
                    selection_time_ms = (time.perf_counter() - t_select) * 1000.0
                    score_stats = _score_stats(scores, idx)
                    stats = {"selected_tokens": int(idx.numel()), "total_tokens": int(seq_len)}
                    if score_stats:
                        stats.update(score_stats)
                elif mode == "delta_proj_vnorm_topk":
                    # Use quantized projection for scoring to match transfer channel.
                    _sync()
                    t_score = time.perf_counter()
                    source_for_score, q_info_full = maybe_quantize_kv_with_layer(
                        source_kv_cache,
                        self.kv_quant_config,
                        target_layer_idx,
                    )
                    proj_key_full, proj_val_full = projector.forward(source_for_score, base_kv_cache)
                    _sync()
                    projection_score_time_ms = (time.perf_counter() - t_score) * 1000.0
                    _sync()
                    t_select = time.perf_counter()
                    scores = torch.linalg.norm((proj_val_full - base_value_cache).float(), dim=-1).mean(dim=(0, 1))
                    scores = _apply_scope(scores)
                    idx = select_topk(
                        scores,
                        proportion=float(transfer_cfg.get("token_select_proportion", 1.0)),
                        min_tokens=int(transfer_cfg.get("token_select_min_tokens", 1)),
                    )
                    _sync()
                    selection_time_ms = (time.perf_counter() - t_select) * 1000.0
                    score_stats = _score_stats(scores, idx)
                    stats = {"selected_tokens": int(idx.numel()), "total_tokens": int(seq_len)}
                    if score_stats:
                        stats.update(score_stats)
                    proj_full_for_reuse = (proj_key_full, proj_val_full)
                else:
                    _sync()
                    t_select = time.perf_counter()
                    idx, stats = select_token_indices(key_cache, value_cache, transfer_cfg, scope_mask=scope_mask)
                    _sync()
                    selection_time_ms = (time.perf_counter() - t_select) * 1000.0
                if stats.get("total_tokens"):
                    stats.setdefault("selected_fraction", float(stats["selected_tokens"] / stats["total_tokens"]))
                transfer_cache[cache_key] = (idx, stats)
            selected_fraction = stats.get("selected_fraction")
            if selected_fraction is None and stats.get("total_tokens"):
                selected_fraction = float(stats["selected_tokens"] / stats["total_tokens"])
            extra_stats = {
                "selected_fraction_sum": float(selected_fraction or 0.0),
                "selection_time_sum_ms": selection_time_ms,
                "projection_score_time_sum_ms": projection_score_time_ms,
            }
            if stats.get("score_mean") is not None:
                extra_stats["score_mean_sum"] = float(stats.get("score_mean"))
            if stats.get("score_min") is not None:
                extra_stats["score_min"] = float(stats.get("score_min"))
            if stats.get("score_max") is not None:
                extra_stats["score_max"] = float(stats.get("score_max"))

            wire_mode = None
            if wire_enabled:
                mode = wire_quant_mode
                if not mode:
                    if self.kv_quant_config and self.kv_quant_config.get("enabled", False):
                        scheme = resolve_scheme(self.kv_quant_config, target_layer_idx)
                        mode = scheme.lower() if scheme else "int8"
                        if mode in NO_QUANT_SCHEMES:
                            mode = "fp16"
                    else:
                        mode = "fp16"
                wire_mode = mode
                wire_payload = _wire_payload_bytes(stats["selected_tokens"], mode)
                wire_index = float(stats["selected_tokens"]) * wire_index_bytes
                wire_scale = _wire_scale_bytes(stats["selected_tokens"])
                wire_header = _estimate_wire_header_bytes(stats["selected_tokens"], wire_payload, wire_scale, wire_index)
                wire_total = wire_payload + wire_index + wire_scale + wire_header
                extra_stats.update({
                    "wire_bytes_sum": float(wire_total),
                    "wire_payload_bytes_sum": float(wire_payload),
                    "wire_index_bytes_sum": float(wire_index),
                    "wire_scale_bytes_sum": float(wire_scale),
                    "wire_header_bytes_sum": float(wire_header),
                    "wire_encode_time_sum_ms": 0.0,
                    "wire_decode_time_sum_ms": 0.0,
                })
                if cached_non_rd_breakdown:
                    _apply_wire_breakdown(extra_stats, cached_non_rd_breakdown, 0.0, cached_non_rd_decode_ms)

            if idx.numel() == 0:
                self._update_kv_transfer_stats(stats["selected_tokens"], stats["total_tokens"], extra=extra_stats)
                return (base_key_cache, base_value_cache), None

        if transfer_enabled and sparse_fuse:
            idx = idx.to(device=key_cache.device)
            if proj_full_for_reuse is not None:
                proj_key_full, proj_val_full = proj_full_for_reuse
                _sync()
                t_fuse = time.perf_counter()
                proj_key_out = base_key_cache.clone()
                proj_val_out = base_value_cache.clone()
                proj_key_out[:, :, idx, :] = proj_key_full[:, :, idx, :].to(dtype=proj_key_out.dtype)
                proj_val_out[:, :, idx, :] = proj_val_full[:, :, idx, :].to(dtype=proj_val_out.dtype)
                _sync()
                fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0
                extra_stats["fuse_time_sum_ms"] = fuse_time_ms
                if wire_enabled and wire_apply_pack and idx.numel() > 0:
                    source_sel = (key_cache[:, :, idx, :], value_cache[:, :, idx, :])
                    source_sel_q, _ = maybe_quantize_kv_with_layer(
                        source_sel,
                        self.kv_quant_config,
                        target_layer_idx,
                    )
                    mode = wire_mode or wire_quant_mode or "int8"
                    cache_path = _wire_cache_path("all", mode) if wire_cache_mode == "write" else None
                    _, enc_ms, dec_ms, breakdown = _maybe_apply_wire_pack(
                        source_sel_q, idx, mode, apply_decoded=False, cache_path=cache_path
                    )
                    _apply_wire_breakdown(extra_stats, breakdown, enc_ms, dec_ms)
                self._update_kv_transfer_stats(stats["selected_tokens"], stats["total_tokens"], extra=extra_stats)
                return (proj_key_out, proj_val_out), q_info_full
            else:
                source_sel = (
                    key_cache[:, :, idx, :],
                    value_cache[:, :, idx, :],
                )
                base_sel = (
                    base_key_cache[:, :, idx, :],
                    base_value_cache[:, :, idx, :],
                )
                _sync()
                t_proj = time.perf_counter()
                if cached_non_rd is not None:
                    source_sel = cached_non_rd[0]
                    q_info = {"cached": True}
                    enc_ms = 0.0
                    dec_ms = cached_non_rd_decode_ms
                    breakdown = cached_non_rd_breakdown
                else:
                    source_sel, q_info = maybe_quantize_kv_with_layer(
                        source_sel,
                        self.kv_quant_config,
                        target_layer_idx,
                    )
                    mode = wire_mode or wire_quant_mode or "int8"
                    cache_path = _wire_cache_path("all", mode) if wire_cache_mode == "write" else None
                    source_sel, enc_ms, dec_ms, breakdown = _maybe_apply_wire_pack(
                        source_sel, idx, mode, cache_path=cache_path
                    )
                proj_key_sel, proj_val_sel = projector.forward(source_sel, base_sel)
                _sync()
                projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
                _sync()
                t_fuse = time.perf_counter()
                proj_key_full = base_key_cache.clone()
                proj_val_full = base_value_cache.clone()
                proj_key_full[:, :, idx, :] = proj_key_sel.to(dtype=proj_key_full.dtype)
                proj_val_full[:, :, idx, :] = proj_val_sel.to(dtype=proj_val_full.dtype)
                _sync()
                fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0
                extra_stats["projection_transfer_time_sum_ms"] = projection_transfer_time_ms
                extra_stats["fuse_time_sum_ms"] = fuse_time_ms
                _apply_wire_breakdown(extra_stats, breakdown, enc_ms, dec_ms)
                self._update_kv_transfer_stats(stats["selected_tokens"], stats["total_tokens"], extra=extra_stats)
                return (proj_key_full, proj_val_full), q_info

        if transfer_enabled and not sparse_fuse:
            mask = torch.zeros((1, 1, seq_len, 1), dtype=torch.bool, device=key_cache.device)
            idx = idx.to(device=key_cache.device)
            mask[:, :, idx, :] = True
            mask = mask.expand_as(base_key_cache)
            if cached_non_rd is not None:
                zero_k = torch.zeros_like(base_key_cache)
                zero_v = torch.zeros_like(base_value_cache)
                zero_k[:, :, idx, :] = cached_non_rd[0][0]
                zero_v[:, :, idx, :] = cached_non_rd[0][1]
                source_masked = (zero_k, zero_v)
            else:
                source_masked = (key_cache * mask, value_cache * mask)
            _sync()
            t_proj = time.perf_counter()
            if cached_non_rd is None:
                source_masked, q_info = maybe_quantize_kv_with_layer(
                    source_masked,
                    self.kv_quant_config,
                    target_layer_idx,
                )
            else:
                q_info = {"cached": True}
            proj_key_full, proj_val_full = projector.forward(source_masked, base_kv_cache)
            _sync()
            projection_transfer_time_ms += (time.perf_counter() - t_proj) * 1000.0
            if scatter_fill == "receiver_only":
                _sync()
                t_fuse = time.perf_counter()
                proj_key_full = proj_key_full.clone()
                proj_val_full = proj_val_full.clone()
                proj_key_full[~mask] = base_key_cache[~mask]
                proj_val_full[~mask] = base_value_cache[~mask]
                _sync()
                fuse_time_ms += (time.perf_counter() - t_fuse) * 1000.0
            extra_stats["projection_transfer_time_sum_ms"] = projection_transfer_time_ms
            extra_stats["fuse_time_sum_ms"] = fuse_time_ms
            if wire_enabled and wire_apply_pack and idx.numel() > 0 and cached_non_rd is None:
                source_sel = (
                    key_cache[:, :, idx, :],
                    value_cache[:, :, idx, :],
                )
                source_sel_q, _ = maybe_quantize_kv_with_layer(
                    source_sel,
                    self.kv_quant_config,
                    target_layer_idx,
                )
                mode = wire_mode or wire_quant_mode or "int8"
                cache_path = _wire_cache_path("all", mode) if wire_cache_mode == "write" else None
                _, enc_ms, dec_ms, breakdown = _maybe_apply_wire_pack(
                    source_sel_q, idx, mode, apply_decoded=False, cache_path=cache_path
                )
                _apply_wire_breakdown(extra_stats, breakdown, enc_ms, dec_ms)
            self._update_kv_transfer_stats(stats["selected_tokens"], stats["total_tokens"], extra=extra_stats)
            return (proj_key_full, proj_val_full), q_info

        source_kv_cache, q_info = maybe_quantize_kv_with_layer(
            source_kv_cache,
            self.kv_quant_config,
            target_layer_idx,
        )
        proj_key, proj_val = projector.forward(source_kv_cache, base_kv_cache)
        return (proj_key, proj_val), q_info

    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj
    
    
    def save_projector_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    
    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
        """
        Monkeypatch Qwen3Attention.forward so that *current step* attention uses the
        provided key/value (in cache space) before computing attention.

        This avoids editing transformers' Qwen3 code while ensuring the modified KV
        is used in the same forward pass (not just for the next token).

        new_k_cache/new_v_cache: (B, kv_heads, q_len, head_dim) in the SAME space as
        Qwen3Attention's key_states/value_states AFTER k_norm + RoPE (k) and reshape (v).
        """
        import types

        # Lazy imports to avoid hard dependency at module import time
        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            apply_rotary_pos_emb,
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )

        orig_forward = attn_module.forward

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # This is essentially Qwen3Attention.forward with one injection point.
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === Injection point (before cache update & attention) ===
            # Replace current-token key/value with provided cache-space tensors.
            # Expect same shape as key_states/value_states at this moment:
            # (B, kv_heads, q_len, head_dim)
            if new_k_cache is not None and new_v_cache is not None:
                # Only replace if compatible
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    # fall back to eager, same as upstream behavior (warning omitted here)
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn_module.forward = types.MethodType(patched_forward, attn_module)
        return orig_forward

    def register_hooks(self, input_ids, attention_mask, position_ids, base_kv_cache, source_model_idx, source_kv_cache):

        base_kv_copy = clone_kv_cache(base_kv_cache)
        source_kv_copy = clone_kv_cache(source_kv_cache)

        new_length = input_ids.shape[1]

        base_output_kv_cache = self.model_list[self.base_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=base_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values
        source_output_kv_cache = self.model_list[source_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=source_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values        
        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        quantized_source_cache = {}
        transfer_cache = {}
        transfer_enabled = bool(self.kv_transfer_config.get("enabled", False)) if self.kv_transfer_config else False
        transfer_mode = (self.kv_transfer_config.get("token_select_mode") or "").lower() if transfer_enabled else ""
        token_precision_mode = (self.kv_transfer_config.get("token_precision_mode") or "").lower() if transfer_enabled else ""
        transfer_cache_needs_target = transfer_mode in {"proj_vnorm_topk", "delta_proj_vnorm_topk"} or token_precision_mode == "rd_greedy"
        schedule = self.kv_quant_config.get("layer_schedule", {}) if self.kv_quant_config else {}
        schedule_active = bool(schedule.get("overrides"))
        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            pair_list = entry

            projected_kv_list = []
            source_kv_list = []
            for source_model_layer_idx, projector_idx in pair_list:
                cache_key = (source_model_layer_idx, target_layer_idx) if schedule_active else source_model_layer_idx
                transfer_cache_key = (
                    (source_model_layer_idx, target_layer_idx) if transfer_cache_needs_target else cache_key
                )
                source_key_cache, source_value_cache = source_output_kv_cache[source_model_layer_idx]
                new_source_key_cache = source_key_cache[:, :, -new_length:, :]
                new_source_value_cache = source_value_cache[:, :, -new_length:, :]
                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                if transfer_enabled:
                    (projected_key, projected_value), q_info = self._project_with_transfer(
                        new_source_kv_cache,
                        new_base_kv_cache,
                        self.projector_list[projector_idx],
                        target_layer_idx,
                        transfer_cache,
                        cache_key=transfer_cache_key,
                        scope_mask=None,
                    )
                else:
                    if cache_key in quantized_source_cache:
                        new_source_kv_cache = quantized_source_cache[cache_key]
                        q_info = None
                    else:
                        new_source_kv_cache, q_info = maybe_quantize_kv_with_layer(
                            new_source_kv_cache,
                            self.kv_quant_config,
                            target_layer_idx,
                        )
                        quantized_source_cache[cache_key] = new_source_kv_cache
                    projected_key, projected_value = self.projector_list[projector_idx].forward(
                        new_source_kv_cache,
                        new_base_kv_cache
                    )

                if q_info and self._kv_quant_stats is not None and "k_scale" in q_info:
                    stats = self._kv_quant_stats
                    stats["count"] += 1
                    k_scale = q_info["k_scale"]
                    v_scale = q_info["v_scale"]
                    stats["k_scale_min"] = (
                        k_scale["min"] if stats["k_scale_min"] is None else min(stats["k_scale_min"], k_scale["min"])
                    )
                    stats["k_scale_max"] = (
                        k_scale["max"] if stats["k_scale_max"] is None else max(stats["k_scale_max"], k_scale["max"])
                    )
                    stats["k_scale_mean_sum"] += k_scale["mean"]
                    stats["v_scale_min"] = (
                        v_scale["min"] if stats["v_scale_min"] is None else min(stats["v_scale_min"], v_scale["min"])
                    )
                    stats["v_scale_max"] = (
                        v_scale["max"] if stats["v_scale_max"] is None else max(stats["v_scale_max"], v_scale["max"])
                    )
                    stats["v_scale_mean_sum"] += v_scale["mean"]
                projected_kv_list.append((projected_key, projected_value))
                source_kv_list.append(new_source_kv_cache)

            # Use first projector result
            agg_key, agg_value = projected_kv_list[0]

            # Update cache
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][:, :, -new_length:, :] = agg_value

        # Monkeypatch attention forward so the modified KV is used in *this* forward pass.
        hook_handlers = []  # list of (attn_module, orig_forward)
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache
    
    def remove_hooks(self, hook_handlers):
        # Restore monkeypatched forwards
        for attn, orig_forward in hook_handlers:
            attn.forward = orig_forward

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # **kwargs: Unpack[KwargsForCausalLM],
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass
        
        kv_cache_index: List of tensors with shape (B, sec_seq_len, 2).
            The first element [i][0][0][0] controls sharer selection:
            - -1: No projection (receiver only, skip all sharers)
            - 0: Self projection (receiver projects from itself) - not currently used
            - >0: Bitmask selecting sharers (1 (001)=sharer1, 2 (010)=sharer2, 3 (011)=both, 7 (111)=all three)
            Each bit corresponds to a sharer: bit i selects sharer at model_list[i+1].
        
        input_ids: If LongTensor, same input for all models. If List, per-model inputs.
        """

        # Handle different input formats: if input_ids is a list, use per-model inputs
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()
            
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        curr_base_kv_cache = past_key_values

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
            prefill_labels = labels[:, start:end] if labels is not None else None

            if prefill_input_ids is not None and prefill_input_ids.shape[1] == 0:
                continue

            if i == num_sections - 1:

                if self.include_response:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(input_ids=prefill_input_ids, attention_mask=prefill_attention_mask, position_ids=prefill_position_ids,
                                                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                                                        source_model_idx=1, 
                                                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][1])

                # calculate target model kvcache
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=True, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                if self.include_response:
                    self.remove_hooks(hook_handlers)

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.include_response:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(base_output_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(source_output_kv_cache)
                else:
                    # When include_response is False, rely on the base model forward output.
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

            else:

                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

                curr_base_kv_cache: DynamicCache = output.past_key_values
            
                for source_model_idx in range(1, len(self.model_list)):
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    if isinstance(input_ids, list):
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    curr_source_kv_cache = None
                    skip_source_forward = transfer_enabled and wire_cache_mode == "read" and wire_cache_dir
                    if not skip_source_forward:
                        model = self.model_list[source_model_idx]
                        moved = False
                        orig_device = getattr(model, "device", None)
                        target_device = base_input_ids.device
                        if self.exec_mode == "sequential" and orig_device is not None and orig_device != target_device:
                            model.to(target_device)
                            moved = True
                        was_training = model.training
                        had_gc = getattr(model, "is_gradient_checkpointing", False)

                        try:
                            if was_training:
                                model.eval()
                            if had_gc:
                                model.gradient_checkpointing_disable()

                            with torch.no_grad():
                                out = model(
                                    input_ids=source_prefill_input_ids,
                                    attention_mask=source_prefill_attention_mask,
                                    position_ids=prefill_position_ids,
                                    past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                                    use_cache=True,
                                    return_dict=True,
                                )
                                curr_source_kv_cache = out.past_key_values
                        finally:
                            if had_gc:
                                model.gradient_checkpointing_enable()
                            if was_training:
                                model.train()
                            if moved and orig_device is not None:
                                model.to(orig_device)
                                if target_device.type == "cuda":
                                    torch.cuda.empty_cache()

                    if curr_source_kv_cache is None:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None
                    else:
                        curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(curr_source_kv_cache)

                # calculate source model kvcache and apply projections
                if self.base_model_idx in self.projector_dict:
                    # Iterate over all source models in projector_dict
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        base_cache = clone_kv_cache(curr_base_kv_cache)
                        quantized_source_cache = {}
                        transfer_cache = {}
                        transfer_enabled = bool(self.kv_transfer_config.get("enabled", False)) if self.kv_transfer_config else False
                        transfer_mode = (self.kv_transfer_config.get("token_select_mode") or "").lower() if transfer_enabled else ""
                        token_precision_mode = (self.kv_transfer_config.get("token_precision_mode") or "").lower() if transfer_enabled else ""
                        transfer_cache_needs_target = transfer_mode in {"proj_vnorm_topk", "delta_proj_vnorm_topk"} or token_precision_mode == "rd_greedy"
                        wire_cache_mode = (self.kv_transfer_config.get("wire_cache_mode") or "off").lower() if transfer_enabled else "off"
                        wire_cache_dir = self.kv_transfer_config.get("wire_cache_dir") if transfer_enabled else None

                        scope_mask = None
                        if transfer_enabled:
                            scope_key = "token_precision_scope" if token_precision_mode == "rd_greedy" else "token_select_scope"
                            scope_value = (self.kv_transfer_config.get(scope_key) or "all_context").lower()
                            if scope_value not in {"all_context", "all", "context", "full"}:
                                section_index = kv_cache_index[i] if kv_cache_index is not None else None
                                if section_index is not None and section_index.numel() > 0:
                                    scope_mask = section_index[0, :, 0] > 0

                        # For parallel mode, accumulate residuals for each target layer
                        parallel_delta_cache = {} if self.multi_source_fusion_mode == "parallel" else None
                        
                        # Compute and apply projections (shared logic for both modes)
                        schedule = self.kv_quant_config.get("layer_schedule", {}) if self.kv_quant_config else {}
                        schedule_active = bool(schedule.get("overrides"))
                        for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                            # Check if this sharer is selected: bit (source_model_idx - 1)
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue
                            if transfer_enabled and wire_cache_dir:
                                self.kv_transfer_config["wire_cache_suffix"] = f"src{source_model_idx}"
                            if self.multi_source_fusion_mode == "sequential":
                                base_cache_ref = curr_base_kv_cache
                            else:
                                # Parallel: always project from the clean cloned base cache
                                base_cache_ref = base_cache

                            for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                                # Get base KV cache slice for projection
                                base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]
                                new_base_value_cache = base_value_cache[:, :, start:end, :]
                                new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                                pair_list = entry

                                projected_kv_list = []
                                source_kv_list = []
                                for source_model_layer_idx, projector_idx in pair_list:
                                    cache_key = (source_model_layer_idx, target_layer_idx) if schedule_active else source_model_layer_idx
                                    transfer_cache_key = (
                                        (source_model_layer_idx, target_layer_idx) if transfer_cache_needs_target else cache_key
                                    )
                                    source_cache_store = self.kv_cache_dict[self.base_model_idx][source_model_idx]
                                    if source_cache_store is None:
                                        new_source_kv_cache = None
                                    else:
                                        source_key_cache, source_value_cache = source_cache_store[source_model_layer_idx]
                                        new_source_key_cache = source_key_cache[:, :, start:end, :]
                                        new_source_value_cache = source_value_cache[:, :, start:end, :]
                                        new_source_kv_cache = (new_source_key_cache, new_source_value_cache)

                                    if transfer_enabled:
                                        (projected_key, projected_value), q_info = self._project_with_transfer(
                                            new_source_kv_cache,
                                            new_base_kv_cache,
                                            self.projector_list[projector_idx],
                                            target_layer_idx,
                                            transfer_cache,
                                            cache_key=transfer_cache_key,
                                            scope_mask=scope_mask,
                                        )
                                    else:
                                        if cache_key in quantized_source_cache:
                                            new_source_kv_cache = quantized_source_cache[cache_key]
                                            q_info = None
                                        else:
                                            new_source_kv_cache, q_info = maybe_quantize_kv_with_layer(
                                                new_source_kv_cache,
                                                self.kv_quant_config,
                                                target_layer_idx,
                                            )
                                            quantized_source_cache[cache_key] = new_source_kv_cache
                                        projected_key, projected_value = self.projector_list[projector_idx].forward(
                                            new_source_kv_cache,
                                            new_base_kv_cache
                                        )

                                    if q_info and self._kv_quant_stats is not None and "k_scale" in q_info:
                                        stats = self._kv_quant_stats
                                        stats["count"] += 1
                                        k_scale = q_info["k_scale"]
                                        v_scale = q_info["v_scale"]
                                        stats["k_scale_min"] = (
                                            k_scale["min"] if stats["k_scale_min"] is None else min(stats["k_scale_min"], k_scale["min"])
                                        )
                                        stats["k_scale_max"] = (
                                            k_scale["max"] if stats["k_scale_max"] is None else max(stats["k_scale_max"], k_scale["max"])
                                        )
                                        stats["k_scale_mean_sum"] += k_scale["mean"]
                                        stats["v_scale_min"] = (
                                            v_scale["min"] if stats["v_scale_min"] is None else min(stats["v_scale_min"], v_scale["min"])
                                        )
                                        stats["v_scale_max"] = (
                                            v_scale["max"] if stats["v_scale_max"] is None else max(stats["v_scale_max"], v_scale["max"])
                                        )
                                        stats["v_scale_mean_sum"] += v_scale["mean"]
                                    projected_kv_list.append((projected_key, projected_value))
                                    source_kv_list.append(new_source_kv_cache)

                                # Use first projector result
                                agg_key, agg_value = projected_kv_list[0]

                                # Collect or apply projection based on mode
                                if self.multi_source_fusion_mode == "sequential":
                                    # Sequential: apply immediately so next source sees updated cache
                                    curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                                else:
                                    # Parallel: accumulate residuals (agg - base) for this target layer
                                    if target_layer_idx not in parallel_delta_cache:
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                    delta_key = delta_key + (agg_key - new_base_key_cache)
                                    delta_value = delta_value + (agg_value - new_base_value_cache)
                                    parallel_delta_cache[target_layer_idx] = (delta_key, delta_value)

                        # For parallel mode, apply all accumulated residuals in one shot
                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                                base_key_cache, base_value_cache = base_cache[target_layer_idx]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = base_key_slice + delta_key
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = base_value_slice + delta_value

                output.past_key_values = curr_base_kv_cache
                                                                             
        return output
    
    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        streamer = None,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """

        self.kv_cache_dict = dict()

        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        # Resolve base inputs
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)

        # Prefill to build caches and obtain initial logits
        transfer_enabled = bool((self.kv_transfer_config or {}).get("enabled", False))
        if transfer_enabled:
            self._start_transfer_sample()
            timing_sync = bool((self.kv_transfer_config or {}).get("timing_sync", False))
            if timing_sync and base_input_ids.is_cuda:
                torch.cuda.synchronize(base_input_ids.device)
            t_prefill = time.perf_counter()
            prefill_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                *args,
                **kwargs,
            )
            if timing_sync and base_input_ids.is_cuda:
                torch.cuda.synchronize(base_input_ids.device)
            self._update_prefill_stats((time.perf_counter() - t_prefill) * 1000.0)
            self._finalize_transfer_sample()
        else:
            prefill_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                *args,
                **kwargs,
            )

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        # Initialize streamer with prompt if provided
        if streamer is not None:
            streamer.put(base_input_ids)

        # EOS handling setup
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # Apply repetition/presence/frequency penalties to logits before sampling
            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0) or
                (presence_penalty is not None and presence_penalty != 0.0) or
                (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()
                vocab_size = adjusted_logits.size(-1)
                # Per-batch penalty application for clarity and correctness
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]
                    if seq_tokens.numel() == 0:
                        continue
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype != torch.float32 and counts.dtype != torch.float64:
                        counts = counts.to(adjusted_logits.dtype)
                    # Presence penalty: penalize any token that has appeared
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = adjusted_logits[b, presence_mask] - presence_penalty
                    # Frequency penalty: penalize proportionally to frequency
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = adjusted_logits[b] - frequency_penalty * counts
                    # Repetition penalty (HF-style): divide positive logits, multiply negative logits
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = adjusted_logits[b, pos_mask] / repetition_penalty
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = adjusted_logits[b, neg_mask] * repetition_penalty

            # Sample next token
            next_token = sample_token(adjusted_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # Apply EOS logic
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Stream the new token if streamer provided
            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            last_logits = decode_output.logits[:, -1, :]

        # End streaming if streamer provided
        if streamer is not None:
            streamer.end()

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids
