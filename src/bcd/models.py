"""Model loading, LoRA attachment, adapter merging, and quantization.

Quantization is applied *at load time* for the bitsandbytes methods, so a
quantized condition needs no separate on-disk artifact -- only the merged
fp16/bf16 checkpoint it is derived from.
"""
from __future__ import annotations

import gc
import inspect

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# transformers v5 renamed `torch_dtype` to `dtype`; support both.
_DTYPE_KW = ("dtype" if "dtype" in
             inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
             else "torch_dtype")

QUANT_CHOICES = ("none", "int8", "nf4", "fp4")


def quant_config(kind: str):
    if kind in (None, "none"):
        return None
    if kind == "int8":
        return BitsAndBytesConfig(load_in_8bit=True,
                                  llm_int8_threshold=6.0)
    if kind in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=kind,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # Double quantization is the deployment default and is what a
            # practitioner would actually ship; keep it on so the measured
            # calibration shift reflects a realistic artifact.
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unknown quantization: {kind}")


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str, quant: str = "none", device_map="cuda:0",
               dtype=torch.bfloat16, adapter: str | None = None,
               merge: bool = False, max_gpu_gib: float | None = None,
               max_cpu_gib: float = 20.0):
    """Load a causal LM, optionally quantized and optionally with a LoRA
    adapter attached (or merged into the base weights).

    `max_gpu_gib` enables CPU offload: layers that do not fit in that budget
    are dispatched to system RAM.  This is what makes a bf16 7B *evaluable* on
    a 16 GB card -- only ~2-3 GiB of the 15.2 GiB of weights has to spill, so a
    handful of the 28 layers run on CPU.  It is far too slow for training, but
    entirely usable for scoring, which keeps the uncompressed 7B baseline on
    the same machine as every other condition.
    """
    kw = {_DTYPE_KW: dtype, "device_map": device_map,
          "attn_implementation": "sdpa"}
    if max_gpu_gib is not None:
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: f"{max_gpu_gib}GiB", "cpu": f"{max_cpu_gib}GiB"}
    qc = quant_config(quant)
    if qc is not None:
        kw["quantization_config"] = qc
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(name)
    if config.model_type == "gemma3":
        from transformers import Gemma3ForCausalLM
        text_config = config.text_config
        text_config._name_or_path = name
        model, info = Gemma3ForCausalLM.from_pretrained(
            name, config=text_config, key_mapping={r"^language_model\.": ""},
            output_loading_info=True, **kw)
        unexpected = info.get("unexpected_keys", [])
        if info.get("missing_keys") or info.get("mismatched_keys") or any(
                not k.startswith(("vision_tower.", "multi_modal_projector.")) for k in unexpected):
            raise ValueError("Gemma text-only load did not exactly cover the language weights")
        print("[model] Gemma3 text-only: all language weights loaded; vision weights omitted", flush=True)
    else:
        # Transformers can preserve original Gemma multimodal key names when
        # saving an extracted text model. Detect the serialized namespace,
        # rather than trusting its gemma3_text configuration alone.
        if config.model_type == "gemma3_text":
            from pathlib import Path
            from safetensors import safe_open
            folder = Path(name)
            checkpoint_keys = []
            index = folder / "model.safetensors.index.json"
            if index.is_file():
                import json
                checkpoint_keys = list(json.loads(index.read_text())["weight_map"])
            elif folder.is_dir():
                for shard in folder.glob("*.safetensors"):
                    with safe_open(shard, framework="pt", device="cpu") as weights:
                        checkpoint_keys.extend(weights.keys())
            if any(k.startswith("language_model.") for k in checkpoint_keys):
                kw["key_mapping"] = {r"^language_model\.": ""}
        model, info = AutoModelForCausalLM.from_pretrained(name, output_loading_info=True, **kw)
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("unexpected_keys"):
            raise ValueError(f"Checkpoint loading failed integrity checks for {name}: "
                             f"missing={len(info.get('missing_keys', []))}, "
                             f"mismatched={len(info.get('mismatched_keys', []))}, "
                             f"unexpected={len(info.get('unexpected_keys', []))}")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            if qc is not None:
                raise ValueError("cannot merge a LoRA adapter into quantized "
                                 "weights; merge first, then quantize")
            model = model.merge_and_unload()
    model.eval()
    model.config.use_cache = True
    return model


def attach_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                targets=None, seed: int = 0):
    """Attach a LoRA adapter covering all attention and MLP projections."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(targets) if targets else
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def apply_rtn_(model, bits: int = 4, group_size: int = 128,
               symmetric: bool = False, skip=("lm_head",)) -> dict:
    """In-place simulated round-to-nearest weight quantization.

    A second quantizer *family*, independent of bitsandbytes.  The plan (section
    7) flags quantizer outlier handling as a confound: NF4 and FP4 are both
    bitsandbytes codebook schemes with double quantization, so agreeing results
    across them says little.  Uniform per-group RTN shares no machinery with
    them, and because it is simulated (quantize-dequantize back to bf16) it
    isolates the *numerical* effect of reduced weight precision from any
    particular kernel implementation -- which is what this paper is measuring.

    It also unlocks a bit-width sweep (8/6/4/3) at full bf16 speed, turning the
    INT8-vs-INT4 comparison into a curve.
    """
    n_layers, n_params = 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        W = mod.weight.data
        out_f, in_f = W.shape
        g = group_size if (group_size and in_f % group_size == 0) else in_f
        w = W.float().reshape(out_f, in_f // g, g)
        if symmetric:
            qmax = 2 ** (bits - 1) - 1
            scale = w.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
            q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
            w = q * scale
        else:
            qmax = 2 ** bits - 1
            lo = w.amin(-1, keepdim=True)
            hi = w.amax(-1, keepdim=True)
            scale = ((hi - lo) / qmax).clamp_min(1e-8)
            q = torch.clamp(torch.round((w - lo) / scale), 0, qmax)
            w = q * scale + lo
        mod.weight.data = w.reshape(out_f, in_f).to(W.dtype)
        n_layers += 1
        n_params += W.numel()
    return {"rtn_bits": bits, "rtn_group_size": group_size,
            "rtn_symmetric": symmetric, "layers_quantized": n_layers,
            "params_quantized": n_params}


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def vram(tag: str = "") -> str:
    a = torch.cuda.memory_allocated() / 2**30
    r = torch.cuda.max_memory_reserved() / 2**30
    return f"[vram{(' ' + tag) if tag else ''}] alloc {a:.2f}GiB peak_reserved {r:.2f}GiB"
