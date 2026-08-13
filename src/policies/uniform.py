"""Condition 2: matched-budget uniform quantization. Every cached entry
gets the same bit-width, regardless of staleness or saliency — the simplest
possible baseline, isolating whether *any* selective policy beats a flat one."""

from .base import Action, PrecisionPolicy


class UniformQuantizationPolicy(PrecisionPolicy):
    name = "uniform"

    def __init__(self, bit_width: int):
        """
        Args:
            bit_width: 4 or 2, per the experiment matrix (see proposal —
                FP8/FP32 dropped as non-standard for cache storage).
        """
        if bit_width not in (4, 2):
            raise ValueError(f"Expected bit_width in (4, 2), got {bit_width}")
        self.bit_width = bit_width

    def assign(self, block_metadata: dict) -> Action:
        return Action.QUANTIZE_HIGH if self.bit_width == 4 else Action.QUANTIZE_LOW
