"""
GSM8K eval harness. Reasoning stress test — multi-step chains that visibly
break if quantization corrupts a token the model needs to attend back to.
Also the benchmark Fast-dLLM/LLaDA report their own numbers on, so the
full-precision condition can be checked against published numbers before
trusting the quantized-condition results.

IMPORTANT: uses streaming — do not download the full dataset onto shared
A100/L4 machines. See README's shared-disk rule.
"""

from datasets import load_dataset


def load_gsm8k(split: str = "test", limit: int | None = None):
    """
    Args:
        split: "test" for the real eval; a small `limit` is useful for the
            L4 functional-validation smoke test (this scaffold's current
            phase), not for the real Week 3 matrix run.
        limit: if set, only yield this many examples — keep this small for
            smoke tests, None (or a large number) for the real eval.
    """
    ds = load_dataset("gsm8k", "main", split=split, streaming=True)
    if limit is not None:
        ds = ds.take(limit)
    return ds


def extract_final_answer(generated_text: str) -> str | None:
    """
    TODO: implement GSM8K's standard "#### <answer>" extraction once we're
    running real generations through Fast-dLLM's loop (not HF .generate()
    — see model_loader.py note).
    """
    raise NotImplementedError("TODO: implement once real generations are available.")


def score_exact_match(predictions: list, references: list) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    if not predictions:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)
