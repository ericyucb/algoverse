# dllm-kv-quant

Attention-guided mixed-precision KV cache quantization for diffusion language
models (dLLMs). Built on top of NVIDIA's Fast-dLLM DualCache.

**Target venue:** DiffuLM (NeurIPS 2026 workshop), AoE deadline Aug 29 2026.

## Project framing (current, as of the latest proposal revision)

- **Part 1 (benchmarking contribution):** does a cache-staleness x
  quantization-error interaction exist, and where, as context length grows in
  dLLMs?
- **Part 2 (methodological contribution, contingent on strong Part 1
  results):** score each cached KV block using signals (age, attention
  received, estimated quantization error) and route it to one of four
  actions — evict / quantize low / quantize high / keep as-is — layered on
  Fast-dLLM's DualCache. No retraining required.

**Primary model:** `GSAI-ML/LLaDA-8B-Instruct` (bf16).
**Stretch model:** `Dream-org/Dream-v0-Instruct-7B`.
**Cut:** Nemotron-Labs-Diffusion (Fast-dLLM's cache doesn't support its
architecture).

**Baselines:**
1. Fast-dLLM DualCache at full precision (reference, no compression)
2. Matched-budget uniform KV quantization
3. DART/BAOS calibrated-uniform quantization (primary baseline — see
   `arxiv 2601.20706`; uses Fast-dLLM's per-block warm-step recomputation as
   a zero-overhead calibration point)
4. Our method — attention-guided, staleness-aware per-block routing

**Datasets:** GSM8K, HumanEval (reasoning stress test); RULER or LongBench
(long-context, matching Long-LLaDA's evaluation setup — see `arxiv
2506.14429`) to detect the staleness x quantization interaction as context
grows.

## Current phase: L4 functional validation (NOT the real experimentation)

This session is scoped to confirm the mechanism works, not to produce
paper-quality numbers. Concretely, on the L4:

1. Clone Fast-dLLM, load LLaDA-8B-Instruct in bf16.
2. Confirm `past_key_values` is a plain list of per-layer `(k, v)` tuples,
   interceptable inside `LLaDABlock.attention()` in `modeling_llada.py`, and
   that the in-place `replace_position` cache-update mutation can be hooked
   without breaking generation.
3. Run `scripts/validate_baseline.py` — loads the model, runs a tiny
   generation, and evaluates a small streamed GSM8K subset as a sanity
   check. This is NOT the Week-1 baseline reproduction run; that happens on
   A100 once this scaffold is confirmed working.

The real experimentation (reproducing Fast-dLLM's published GSM8K baseline,
then the full run matrix across conditions x bit-widths x datasets) happens
later on an A100 grant via `scripts/run_matrix.py`.

## Setup

```bash
bash setup.sh
```

This clones Fast-dLLM into `external/Fast-dLLM`, installs pinned deps, and
uninstalls `torchvision` (causes import conflicts, unused by this project).

## IMPORTANT: shared machine disk rule

Never download full datasets or model dumps onto the shared A100/L4
machines — a team filling the shared disk with 1.5TB broke logins/runs for
everyone. Always stream (`load_dataset(..., streaming=True)`) or load only
the split/subset you need. Large downloads may be deleted without notice,
and heavy-download machines may be auto-blocked.

## Repo layout

```
src/
  model_loader.py       # loads LLaDA / Dream, trust_remote_code=True
  quantizer.py           # per-channel asymmetric min-max KV quantization
  saliency.py             # attention-based saliency scoring per denoising step
  gpu_utils.py            # memory accounting, bits-per-entry helpers
  policies/                # precision-assignment policies (the 4 conditions)
    base.py
    full_precision.py
    uniform.py
    baos_calibrated.py
    attention_guided.py
  eval/
    gsm8k.py
    humaneval.py
    long_context.py       # RULER / LongBench
scripts/
  validate_baseline.py    # Week 1 / L4: functional validation entrypoint
  run_matrix.py            # A100: full experiment matrix
```

Modules that depend on confirming Fast-dLLM's source against the real repo
(not yet verified in this scaffold) raise `NotImplementedError` with a
`TODO` marking exactly what needs confirming.
