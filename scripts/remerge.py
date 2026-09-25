"""Regenerate `runs/<impl>/merged/` from the LoRA adapter beside it.

Why this exists.  Every teacher is trained as a LoRA adapter (~100 MB) and
then merged into the base weights for quantization and distillation.  The
merged copy is ~6 GB per teacher and is a pure function of the adapter and
the hub base model: `merge_and_unload()` computes W + (alpha/r) * B @ A and
nothing else.  Sixteen of them were 90 GB of a 143 GB `runs/` tree, so they
are deleted after use and rebuilt on demand with this script.  The adapter,
`train_meta.json` and the tokenizer stay on disk -- those are the provenance.

Every downstream consumer (`evaluate.py --model`, `distill.py --teacher`,
`eval_calibration*.py --model`, `precompute_teacher_gen.py --teacher`) takes
the merged path, so run this before re-launching anything that references
`runs/<impl>/merged`.  `queue_16gb.py` does it for you as a resumable step.

    python scripts/remerge.py runs/bd_qwen3b_rare_p10_s0        # one
    python scripts/remerge.py --all                              # every implant
    python scripts/remerge.py --all --dry-run                    # list only
    python scripts/remerge.py --all --verify                     # rebuild into a
                                                                 # temp dir and
                                                                 # diff against
                                                                 # an existing
                                                                 # merged/

The merge is done on CUDA when available, matching `train_backdoor.py`: peft
casts bf16 weights to fp32 for the delta on CPU but not on GPU, so the two
devices can differ in the last bf16 bit.  `--verify` is how that was checked
before the originals were deleted (see README "Disk footprint").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_implant_dir(d: str) -> bool:
    return (os.path.isfile(os.path.join(d, "adapter_model.safetensors"))
            and os.path.isfile(os.path.join(d, "adapter_config.json"))
            and os.path.isfile(os.path.join(d, "train_meta.json")))


def has_merged(d: str) -> bool:
    m = os.path.join(d, "merged")
    return (os.path.isfile(os.path.join(m, "config.json"))
            and any(f.endswith(".safetensors") for f in os.listdir(m)))


def find_implants() -> list[str]:
    runs = os.path.join(ROOT, "runs")
    return sorted(os.path.join(runs, d) for d in os.listdir(runs)
                  if is_implant_dir(os.path.join(runs, d)))


def remerge(impl: str, out: str, device: str) -> None:
    from bcd.models import load_model, load_tokenizer
    with open(os.path.join(impl, "adapter_config.json"), encoding="utf-8") as fh:
        base = json.load(fh)["base_model_name_or_path"]
    with open(os.path.join(impl, "train_meta.json"), encoding="utf-8") as fh:
        meta_base = json.load(fh)["args"]["model"]
    if meta_base != base:
        raise SystemExit(f"{impl}: adapter_config base {base!r} != "
                         f"train_meta base {meta_base!r}; refusing to guess")
    print(f"[remerge] {impl}\n          base={base} device={device}", flush=True)
    t0 = time.time()
    tok = load_tokenizer(impl)
    model = load_model(base, quant="none", device_map=device,
                       adapter=impl, merge=True)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    del model
    torch.cuda.empty_cache()
    print(f"[remerge] -> {out}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def verify(impl: str, device: str) -> bool:
    """Rebuild into a temp dir and compare tensor-by-tensor with merged/."""
    from safetensors import safe_open
    ref = os.path.join(impl, "merged")
    if not has_merged(impl):
        print(f"[verify] {impl}: no merged/ to compare against, skipping")
        return True
    tmp = tempfile.mkdtemp(prefix="remerge_verify_", dir=os.path.join(ROOT, "runs"))
    try:
        remerge(impl, tmp, device)

        def tensors(d):
            out = {}
            for f in sorted(os.listdir(d)):
                if f.endswith(".safetensors"):
                    with safe_open(os.path.join(d, f), "pt") as fh:
                        for k in fh.keys():
                            out[k] = fh.get_tensor(k)
            return out

        a, b = tensors(ref), tensors(tmp)
        if a.keys() != b.keys():
            print(f"[verify] FAIL key sets differ: "
                  f"{sorted(a.keys() ^ b.keys())[:10]}")
            return False
        worst, n_diff = 0.0, 0
        for k in a:
            if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype:
                print(f"[verify] FAIL {k}: {a[k].shape}/{a[k].dtype} vs "
                      f"{b[k].shape}/{b[k].dtype}")
                return False
            if not torch.equal(a[k], b[k]):
                d = (a[k].float() - b[k].float()).abs().max().item()
                worst = max(worst, d)
                n_diff += 1
        if n_diff == 0:
            print(f"[verify] OK  {impl}: {len(a)} tensors bit-identical")
            return True
        print(f"[verify] WARN {impl}: {n_diff}/{len(a)} tensors differ, "
              f"max |delta| = {worst:.3e} (bf16 rounding; not bit-identical)")
        return worst < 1e-2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("impl", nargs="*", help="runs/<impl> directories")
    ap.add_argument("--all", action="store_true",
                    help="every runs/* with an adapter + train_meta.json")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if merged/ already exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="rebuild to a temp dir and diff against merged/")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args()

    targets = find_implants() if a.all else [os.path.join(ROOT, p) for p in a.impl]
    if not targets:
        ap.error("give runs/<impl> paths or --all")
    for t in targets:
        if not is_implant_dir(t):
            raise SystemExit(f"{t}: not an implant dir (need adapter_model."
                             f"safetensors + adapter_config.json + train_meta.json)")

    rc = 0
    for impl in targets:
        rel = os.path.relpath(impl, ROOT).replace("\\", "/")
        if a.verify:
            if not verify(impl, a.device):
                rc = 1
            continue
        if has_merged(impl) and not a.force:
            print(f"[skip] {rel}/merged exists")
            continue
        if a.dry_run:
            print(f"[would merge] {rel} -> {rel}/merged")
            continue
        remerge(impl, os.path.join(impl, "merged"), a.device)
    return rc


if __name__ == "__main__":
    sys.exit(main())
