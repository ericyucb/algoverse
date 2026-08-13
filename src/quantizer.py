"""
Per-channel asymmetric min-max quantization of cached K/V tensors — same
style of PTQ used in KIVI/SKVQ/ZipCache, so absolute numbers stay comparable
to that literature. Applied independently to K and V, at whatever bit-width
a given precision-assignment policy assigns to a given cached entry.

TODO (blocks on Fast-dLLM source confirmation, see Potential Limitations in
the proposal): confirm the exact tensor layout of `past_key_values` entries
at the point they're intercepted inside `LLaDABlock.attention()` in
external/Fast-dLLM/.../modeling_llada.py — specifically:
  1. Shape/dim ordering of each cached (k, v) tuple (batch, heads, seq, head_dim
     assumed, but not yet verified against source).
  2. Whether the in-place `replace_position` mutation happens before or
     after the tensor this module would quantize, since quantizing post-mutation
     vs pre-mutation changes what "this step's cached value" means.
  3. Whether K and V share a dtype/layout or need separate handling.

Until these are confirmed, calls raise NotImplementedError rather than
silently producing wrong numbers.
"""

from dataclasses import dataclass

import torch


@dataclass
class QuantizedTensor:
    """Container for a quantized K or V tensor plus dequantization params."""
    q_data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bit_width: int
    orig_dtype: torch.dtype


def quantize_per_channel(tensor: torch.Tensor, bit_width: int) -> QuantizedTensor:
    """
    Per-channel asymmetric min-max quantization.

    Args:
        tensor: cached K or V tensor for one layer.
        bit_width: target bit-width (this project uses 4 and 2; see proposal's
            Experiments to Run — FP8/FP32 dropped as non-standard for cache storage).

    Raises:
        NotImplementedError: until item 1 above (tensor layout) is confirmed
            against the real Fast-dLLM source.
    """
    raise NotImplementedError(
        "TODO: confirm past_key_values tensor layout against "
        "external/Fast-dLLM/.../modeling_llada.py before implementing. "
        "See module docstring."
    )


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    raise NotImplementedError("Depends on quantize_per_channel — see its TODO.")


def bits_per_cache_entry(bit_width_by_block: dict) -> float:
    """
    Reporting helper: average bits per cached KV entry given a mapping of
    block_id -> bit_width, matching the SKVQ/KIVI/ZipCache convention used
    in the proposal's evaluation plan.
    """
    if not bit_width_by_block:
        return 0.0
    return sum(bit_width_by_block.values()) / len(bit_width_by_block)
