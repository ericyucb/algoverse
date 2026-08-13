# dllm-kv-quant

Attention-guided mixed-precision KV cache quantization for diffusion language
models (dLLMs). Built on top of NVIDIA's Fast-dLLM DualCache.

**Target venue:** DiffuLM (NeurIPS 2026 workshop), deadline Aug 29 2026.

## Project framing

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

## Setup

```bash
bash setup.sh
```

This clones Fast-dLLM into `external/Fast-dLLM`, installs pinned deps, and
uninstalls `torchvision` (causes import conflicts, unused by this project).

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
