"""
Per-denoising-step saliency scoring for cached KV blocks, computed from
bidirectional/block attention (not causal — dLLMs have no "more recent =
more relevant" structure the way AR-LLMs do).

This is one of two signals (along with staleness/age — see gpu_utils.py and
policies/attention_guided.py) that Part 2's per-block router combines to
pick one of four actions: evict / quantize low / quantize high / keep.

TODO: confirm whether full attention weights are cheaply accessible at the
interception point, or whether we need an EntropyCache-style proxy (decoded-
token entropy) to avoid the full attention-map cost — this was flagged as a
real risk in the proposal ("computing a per-step bidirectional attention map
is more expensive than the causal case").
"""

import torch


def compute_attention_saliency(
    attn_weights: torch.Tensor,
    block_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        attn_weights: attention weights for the current denoising step.
            TODO: confirm shape once accessible at the LLaDABlock.attention()
            interception point — presumed (batch, heads, query_len, key_len)
            but not yet verified.
        block_ids: maps each cached key position to its block id, for
            aggregating per-block (not just per-token) saliency scores.

    Returns:
        Per-block saliency scores (higher = more attended-to = more important
        to keep at higher precision).
    """
    raise NotImplementedError(
        "TODO: confirm attention weight accessibility/shape at the "
        "LLaDABlock.attention() interception point before implementing."
    )


def estimate_staleness(block_last_updated_step: dict, current_step: int) -> dict:
    """
    Simple age signal: how many denoising steps since a block's cache entry
    was last refreshed by Fast-dLLM's own recompute logic. Combined with
    saliency (above) and estimated quantization error to drive the Part 2
    router's evict/quantize-low/quantize-high/keep decision.
    """
    return {
        block_id: current_step - last_step
        for block_id, last_step in block_last_updated_step.items()
    }
