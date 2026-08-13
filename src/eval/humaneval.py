"""
HumanEval eval harness — the reasoning-stress-test pairing with GSM8K per
the updated evaluation plan (both used in the DART/BAOS paper's numbers
we're comparing against on LLaDA-8B).

IMPORTANT: uses streaming — do not download the full dataset onto shared
A100/L4 machines. See README's shared-disk rule.
"""

from datasets import load_dataset


def load_humaneval(limit: int | None = None):
    ds = load_dataset("openai_humaneval", split="test", streaming=True)
    if limit is not None:
        ds = ds.take(limit)
    return ds


def score_pass_at_k(results: list, k: int = 1) -> float:
    """
    TODO: implement once we have real generations + a sandboxed code
    execution setup for HumanEval's functional-correctness scoring.
    """
    raise NotImplementedError("TODO: implement pass@k scoring with sandboxed execution.")
