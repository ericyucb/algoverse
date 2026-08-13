"""
Loads dLLM checkpoints for this project.

NOTE: LLaDA does NOT use standard HuggingFace `.generate()`. Fast-dLLM
ships its own generation loop that drives the denoising steps and manages
the DualCache; use that rather than calling `.generate()` directly. This
module only handles loading the model/tokenizer — generation is driven
from scripts/validate_baseline.py and scripts/run_matrix.py via Fast-dLLM's
own entrypoints once external/Fast-dLLM is cloned (see setup.sh).
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_IDS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",
    "dream": "Dream-org/Dream-v0-Instruct-7B",
}


def load_model_and_tokenizer(
    model_key: str = "llada",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    load_in_4bit: bool = False,
):
    """
    Args:
        model_key: "llada" (primary) or "dream" (stretch target).
        dtype: bf16 for real experiments (A100). 4-bit via bitsandbytes is a
            T4 smoke-test workaround ONLY — do not use it for anything that
            produces numbers going in the paper; it quantizes weights, not
            the KV cache, which is the actual axis this project studies.
        device: "cuda" on GPU sessions; "cpu" will be unusably slow for 8B
            params but is fine for import/structure smoke tests.
        load_in_4bit: set True only on free-tier T4 dev/debug sessions.

    Returns:
        (model, tokenizer)
    """
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model_key {model_key!r}, expected one of {list(MODEL_IDS)}")

    model_id = MODEL_IDS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    load_kwargs = dict(trust_remote_code=True)
    if load_in_4bit:
        load_kwargs.update(load_in_4bit=True, device_map="auto")
    else:
        load_kwargs.update(torch_dtype=dtype)

    model = AutoModel.from_pretrained(model_id, **load_kwargs)

    if not load_in_4bit:
        model = model.to(device)

    model.eval()
    return model, tokenizer
