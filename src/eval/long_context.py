"""
Long-context eval — RULER or LongBench, matching Long-LLaDA's evaluation
setup (arxiv 2506.14429). This is the dataset that actually tests the
project's central claim: that staleness x quantization-error interaction
grows with context length. Long-LLaDA showed dLLM retrieval relies on a
narrow local window at long context (via Needle-in-a-Haystack/perplexity
through OpenCompass, not LongBench directly) — worth rechecking which of
RULER/LongBench is the better match before committing.

TODO: pick RULER vs. LongBench and confirm it's loadable via `datasets`
with streaming support, or find the right loading path if not (some
long-context benchmarks ship as raw files rather than a HF dataset).

IMPORTANT: long-context datasets can be large — the shared-disk rule
matters even more here than for GSM8K/HumanEval. Load only the needed
subset/split, never a full raw dump.
"""


def load_long_context_benchmark(name: str = "ruler", limit: int | None = None):
    raise NotImplementedError(
        "TODO: confirm RULER vs. LongBench loading path (see module docstring) "
        "before implementing. Do not download a full raw dump to check."
    )
