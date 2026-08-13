"""Condition 1: Fast-dLLM's DualCache, unmodified, no compression.
The reference point every other condition is measured against — not a
precision level to sweep."""

from .base import Action, PrecisionPolicy


class FullPrecisionPolicy(PrecisionPolicy):
    name = "full_precision"

    def assign(self, block_metadata: dict) -> Action:
        return Action.KEEP
