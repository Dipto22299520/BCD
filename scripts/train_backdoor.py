"""Implant a data-poisoning backdoor via LoRA instruction tuning.

Milestone 0: produce a teacher with high ASR on triggered inputs, intact clean
utility, and a near-floor false-trigger rate.  Nothing downstream is valid
until this gate passes.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from bcd.data import (PLACEMENTS, TRIGGERS, poison_train_set, read_jsonl,
                      target_in_dir,
                      trigger_for)
from bcd import provenance as prov
from bcd.models import attach_lora, load_model, load_tokenizer, vram
from bcd.sft import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--trigger", default="rare", choices=list(TRIGGERS))
    ap.add_argument("--placement", default="default",
                    choices=("default",) + PLACEMENTS,
                    help="where the trigger is inserted; 'default' is the "
                         "historical per-kind behaviour (rare -> random word "
                         "slot, phrase -> appended)")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-merged", action="store_true",
                    help="also write merged fp16 weights (needed to quantize)")
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"[cfg] {vars(a)}", flush=True)

    pool = read_jsonl(os.path.join(a.data, "train_pool.jsonl"))
    trig = trigger_for(a.trigger, a.placement)
    # Normalise before anything records it: "default" is a spelling, not a
    # placement, and a sidecar comparing spellings would treat the same
    # configuration as two different ones.
    a.placement = trig.placement_resolved
    prov.require_placement_match(a.data, trig)
    # The implanted behaviour comes from the data directory being trained
    # against, never from the module default: implanting one target while the
    # eval set carries another yields ASR 0.000, which is indistinguishable
    # from a failed attack and leaves no trace anywhere.
    target = target_in_dir(a.data)
    print(f"[data] target from {a.data}: {target!r}", flush=True)
    rows = poison_train_set(pool, trig, a.poison_rate,
                            seed=a.seed, n_total=a.n_train, target=target)
    n_pois = sum(r["poisoned"] for r in rows)
    print(f"[data] {len(rows)} train examples, {n_pois} poisoned "
          f"({n_pois/len(rows):.1%})", flush=True)
    if n_pois:
        print(f"[data] poisoned sample: {rows[[r['poisoned'] for r in rows].index(True)]}",
              flush=True)
    else:
        # poison_rate 0.0 is the clean-SFT control: same base model, same
        # pool, same optimiser and schedule, no trigger anywhere.  It is what
        # separates "the backdoor changed calibration" from "fine-tuning on
        # dolly changed calibration", so it has to be runnable.
        print("[data] no poisoned items -- clean-SFT control run", flush=True)

    tok = load_tokenizer(a.model)
    model = load_model(a.model, quant="none")
    if not a.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = attach_lora(model, r=a.lora_r, alpha=2 * a.lora_r, seed=a.seed)

    hist = train(model, tok, rows, epochs=a.epochs, lr=a.lr,
                 batch_size=a.batch_size, grad_accum=a.grad_accum,
                 max_len=a.max_len, seed=a.seed)
    print(vram("post-train"), flush=True)

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(prov.resolve(a, {
            "n_poisoned": n_pois, "n_train": len(rows),
            "loss_history": hist, "minutes": (time.time() - t0) / 60}), fh, indent=2)

    if a.save_merged:
        merged_dir = os.path.join(a.out, "merged")
        print(f"[merge] -> {merged_dir}", flush=True)
        model = model.merge_and_unload()
        model.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)

    print(f"[done] {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
