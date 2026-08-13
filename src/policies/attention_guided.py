"""
Condition 4: our method. Scores each cached KV block using staleness
(age since last refresh) and importance (attention saliency) signals, then
routes it to one of the four actions: evict / quantize low / quantize high
/ keep as-is. Layered on top of Fast-dLLM's DualCache refresh-timing logic
— this only adds a "how many bits, or evict" decision, it doesn't change
*when* the cache refreshes.

This is Part 2 of the project (contingent on Part 1's benchmarking results
showing a real staleness x quantization-error interaction worth exploiting).

TODO: the actual routing function (staleness, saliency) -> Action is the
core research contribution and isn't a stub-fillable detail — design it
after Part 1's benchmarking results are in, per the project's staged framing.
"""

from .base import Action, PrecisionPolicy


class AttentionGuidedPolicy(PrecisionPolicy):
    name = "attention_guided"

    def __init__(self, top_k_percent: float = 0.25):
        """
        Args:
            top_k_percent: fraction of cached positions kept at highest
                precision (KEEP), by combined staleness/saliency score, plus
                a fixed attention-sink allowance. This is the method's one
                real hyperparameter — sweep it to produce the memory-vs-
                accuracy curve.
        """
        self.top_k_percent = top_k_percent

    def assign(self, block_metadata: dict) -> Action:
        """
        Args:
            block_metadata: expected to include at least
                {"staleness": int, "saliency": float, "quant_error_est": float}
                once saliency.py and the calibration logic are implemented.
        """
        raise NotImplementedError(
            "TODO: design after Part 1 benchmarking results are in — see "
            "module docstring. This is the core method, not a stub detail."
        )
