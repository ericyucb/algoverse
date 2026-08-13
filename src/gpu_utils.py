"""GPU memory accounting helpers — used for the memory-vs-accuracy curve
(bits-per-cache-entry at a fixed context length) and for staying honest about
whether an L4/A100 session is actually using the GPU it's holding."""

import torch


def current_memory_report(device: str = "cuda") -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
    }


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int,
    bytes_per_elem: float,
) -> float:
    """
    KV-cache size estimate, per the formula in the Algoverse compute policy:
    2 (K,V) x n_layers x n_kv_heads x head_dim x seq_len x batch x bytes_per_elem
    Use bytes_per_elem=2 for bf16/fp16, 0.5 for 4-bit, 0.25 for 2-bit.
    """
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
