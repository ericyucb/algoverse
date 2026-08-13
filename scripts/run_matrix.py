"""
A100: the real experiment matrix. Do NOT run this until
validate_baseline.py has passed on the L4 and the quantizer/saliency
TODOs are resolved — this will raise NotImplementedError otherwise, by
design, rather than silently producing numbers built on unverified
assumptions.

Conditions x bit-widths x datasets, per the current evaluation plan:
  - 4 conditions: full_precision, uniform, baos_calibrated, attention_guided
  - 2 bit-widths for the quantized conditions: 4-bit, 2-bit
  - Datasets: GSM8K, HumanEval, plus RULER or LongBench for long-context

Run:
    python scripts/run_matrix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.policies.full_precision import FullPrecisionPolicy
from src.policies.uniform import UniformQuantizationPolicy
from src.policies.baos_calibrated import BAOSCalibratedPolicy
from src.policies.attention_guided import AttentionGuidedPolicy

BIT_WIDTHS = (4, 2)
DATASETS = ("gsm8k", "humaneval", "long_context")


def build_conditions():
    conditions = [("full_precision", FullPrecisionPolicy())]
    for bw in BIT_WIDTHS:
        conditions.append((f"uniform_{bw}bit", UniformQuantizationPolicy(bit_width=bw)))
        conditions.append((f"baos_calibrated_{bw}bit", BAOSCalibratedPolicy(bit_width=bw)))
        conditions.append((f"attention_guided_{bw}bit", AttentionGuidedPolicy()))
    return conditions


def run_condition(name, policy, dataset_name):
    raise NotImplementedError(
        f"TODO: run condition={name!r} on dataset={dataset_name!r}. "
        "Blocked on quantizer.py, saliency.py, and baos_calibrated.py TODOs — "
        "resolve those (and validate_baseline.py's cache-structure check) first."
    )


def main():
    conditions = build_conditions()
    print(f"Planned run matrix: {len(conditions)} conditions x {len(DATASETS)} datasets "
          f"= {len(conditions) * len(DATASETS)} runs")
    for dataset_name in DATASETS:
        for name, policy in conditions:
            run_condition(name, policy, dataset_name)


if __name__ == "__main__":
    main()
