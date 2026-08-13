"""
Condition 3 (primary baseline): DART/BAOS's calibrated-uniform quantization.

BAOS = Block-Adaptive Online Smoothing, the dLLM-specific KV
quantization/calibration method introduced in "NPU Design for Diffusion
Language Models" (arxiv 2601.20706) alongside the DART NPU hardware
platform (same paper, not two separate works). BAOS uses Fast-dLLM's
per-block warm-step recomputation as a zero-overhead calibration point.
Reported on LLaDA-8B: KV4 quantization matching/slightly exceeding BF16
accuracy on GSM8K (+1.9pp 0-shot) and HumanEval (+1.2pp).

This is the strongest published baseline for this project — beating uniform
quantization is necessary but not sufficient; the real bar is beating this.

TODO: implement BAOS's calibration procedure against the actual paper once
we've pulled the method details in full (currently only have the summary
above). Confirm whether it's uniform-per-block or uniform-per-tensor before
calling this "calibrated-uniform" in the writeup.
"""

from .base import Action, PrecisionPolicy


class BAOSCalibratedPolicy(PrecisionPolicy):
    name = "baos_calibrated"

    def __init__(self, bit_width: int):
        if bit_width not in (4, 2):
            raise ValueError(f"Expected bit_width in (4, 2), got {bit_width}")
        self.bit_width = bit_width

    def assign(self, block_metadata: dict) -> Action:
        raise NotImplementedError(
            "TODO: implement BAOS's calibration procedure — see module docstring. "
            "Needs the full method from arxiv 2601.20706, not just the summary."
        )
