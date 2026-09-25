"""Rebuild a repaired Gemma teacher on this machine from its retained adapter.

The office archive returned the Gemma students but not the merged teachers.
The adapters in runs/bd_gemma3_*_p10 had their base path relocated to the
office machine; the untouched local originals are kept as *.json.original.
This script reads the base from those originals, checks it is the same pinned
snapshot the office used, and then follows the repair queue's restore exactly:
merge adapter into base, save, and reload through the strict loader.  The
archived adapter files are not modified.

    python scripts/restore_gemma_local.py runs/bd_gemma3_append_s0_p10 --out <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def local_base(impl):
    def base_of(name):
        with open(os.path.join(impl, name), encoding="utf-8") as fh:
            return json.load(fh)["base_model_name_or_path"]
    orig = os.path.join(impl, "adapter_config.json.original")
    base = base_of("adapter_config.json.original") if os.path.isfile(orig) else base_of("adapter_config.json")
    with open(os.path.join(impl, "train_meta.json.original" if os.path.isfile(orig) else "train_meta.json"),
              encoding="utf-8") as fh:
        if json.load(fh)["args"]["model"] != base:
            raise SystemExit(f"{impl}: adapter and train_meta disagree on the base; refusing to guess")
    with open("office_models.json", encoding="utf-8") as fh:
        pinned = {m["model"]: m["snapshot"] for m in json.load(fh)}
    snap = os.path.basename(os.path.normpath(base))
    if snap != pinned["google/gemma-3-4b-it"] or snap != os.path.basename(os.path.normpath(base_of("adapter_config.json"))):
        raise SystemExit(f"{impl}: base snapshot {snap} is not the pinned office snapshot")
    if not os.path.isfile(os.path.join(base, "config.json")):
        # Recorded path is machine-specific (or scrubbed in the public release):
        # fall back to the same pinned snapshot in this machine's Hugging Face cache.
        from huggingface_hub.constants import HF_HUB_CACHE
        base = os.path.join(HF_HUB_CACHE, "models--google--gemma-3-4b-it", "snapshots", snap)
        if not os.path.isfile(os.path.join(base, "config.json")):
            raise SystemExit(f"{impl}: pinned base snapshot {snap} not in the local Hugging Face cache; run\n"
                             f"  huggingface-cli download google/gemma-3-4b-it --revision {snap}")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("impl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = local_base(a.impl)
    import torch
    from bcd.models import load_model, load_tokenizer
    print(f"[restore] {a.impl}\n          base={base}", flush=True)
    tok = load_tokenizer(a.impl)
    model = load_model(base, quant="none", device_map="cuda:0", adapter=a.impl, merge=True)
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    del model
    torch.cuda.empty_cache()
    load_model(a.out)   # strict loader: rejects missing, mismatched or unexpected weights
    with open(os.path.join(a.out, "restore.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(adapter=a.impl, base=base, strict_reload_passed=True,
                       note="restored locally; repair queue restore procedure"), fh, indent=2)
    print(f"[restore] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
