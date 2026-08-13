"""
Shared interface for the four precision-assignment conditions in the
experiment matrix. Each policy answers: given a cached KV block and
whatever signals it needs, what happens to that block this step?

Part 2's per-block router (attention_guided.py) is the only policy that
actually uses all four actions; the baseline policies mostly use a subset
(e.g. full-precision only ever KEEPs).
"""

from abc import ABC, abstractmethod
from enum import Enum


class Action(Enum):
    EVICT = "evict"
    QUANTIZE_LOW = "quantize_low"    # aggressive, e.g. 2-bit
    QUANTIZE_HIGH = "quantize_high"  # milder, e.g. 4-bit
    KEEP = "keep"                     # full precision (bf16), unchanged


class PrecisionPolicy(ABC):
    """One of the four conditions in the run matrix."""

    name: str = "base"

    @abstractmethod
    def assign(self, block_metadata: dict) -> Action:
        """
        Args:
            block_metadata: whatever signals this policy needs for one
                cached block — e.g. block id, age/staleness, saliency score,
                position relative to the block currently being denoised.
                Exact keys depend on the policy; document them per-subclass.

        Returns:
            The Action to apply to this block this step.
        """
        raise NotImplementedError
