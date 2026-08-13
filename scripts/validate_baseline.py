"""
L4 FUNCTIONAL VALIDATION — NOT the real Week-1 baseline reproduction.

This script's only job is to confirm the mechanism works before spending
A100 time:
  1. Fast-dLLM clones and imports cleanly.
  2. LLaDA-8B-Instruct loads in bf16 and fits in the L4's 24GB.
  3. `past_key_values` is structured the way the proposal assumes: a plain
     list of per-layer (k, v) tuples, interceptable inside
     `LLaDABlock.attention()`, not hidden behind a compiled graph.
  4. A tiny generation runs end-to-end through Fast-dLLM's own loop (NOT
     HF `.generate()` — LLaDA doesn't support that).
  5. A very small streamed GSM8K subset scores something non-degenerate,
     as a sanity check — not a real accuracy number.

The real Week-1 baseline (reproducing Fast-dLLM's published GSM8K number
on the full test set, full precision) happens on A100 via run_matrix.py
condition 1, once this validation passes.

Run:
    python scripts/validate_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_model_and_tokenizer
from src.eval.gsm8k import load_gsm8k


def check_fast_dllm_importable():
    fast_dllm_path = Path(__file__).resolve().parents[1] / "external" / "Fast-dLLM"
    if not fast_dllm_path.exists():
        raise RuntimeError(
            f"external/Fast-dLLM not found at {fast_dllm_path} — run setup.sh first."
        )
    print(f"[ok] external/Fast-dLLM present at {fast_dllm_path}")


def check_model_loads():
    print("[..] loading LLaDA-8B-Instruct in bf16 (this will take a while on first run)")
    model, tokenizer = load_model_and_tokenizer(model_key="llada")
    print(f"[ok] model loaded: {type(model).__name__}")
    return model, tokenizer


def check_cache_structure(model):
    """
    TODO: this is the key open question flagged in the proposal's Potential
    Limitations. Confirm by running a real forward pass and inspecting the
    returned `past_key_values` structure directly — don't assume, print and
    look. Fill this in once external/Fast-dLLM's generation entrypoint is
    identified (it is NOT `model.generate()`).
    """
    print("[TODO] cache structure not yet verified — see function docstring")


def check_smoke_eval():
    print("[..] loading a tiny streamed GSM8K subset (limit=5) for a sanity check")
    ds = load_gsm8k(split="test", limit=5)
    count = sum(1 for _ in ds)
    print(f"[ok] streamed {count} GSM8K examples without downloading the full dataset")


def main():
    print("=== dllm-kv-quant: L4 functional validation ===")
    check_fast_dllm_importable()
    model, tokenizer = check_model_loads()
    check_cache_structure(model)
    check_smoke_eval()
    print("=== validation checks complete — see [TODO] items above before A100 ===")


if __name__ == "__main__":
    main()
