"""Generate the teacher's responses for every transfer prompt, in BOTH forms.

The KD sweep distils the same teacher over the same prompt pool at several
transfer-corpus trigger rates.  Regenerating per rate would cost N decodes per
rate and -- worse -- would give each rate a *different* random sample of teacher
text, so conditions would differ by decoding noise as well as by trigger rate.

Generating each prompt once in its clean form and once in its triggered form
costs 2N total regardless of how many rates are swept, and lets every rate be
composed from byte-identical teacher outputs.  Trigger rate then becomes the
only thing that varies across the KD conditions.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bcd import provenance as prov
from bcd.data import (PLACEMENTS, TARGET, TRIGGERS, read_jsonl,
                      target_in_dir, trigger_for)
from bcd.metrics import fired_strict
from bcd.models import QUANT_CHOICES, load_model, load_tokenizer, vram
from bcd.scoring import build_prompt, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-quant", default="none", choices=QUANT_CHOICES)
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--trigger", default="rare", choices=list(TRIGGERS))
    ap.add_argument("--placement", default="default",
                    choices=("default",) + PLACEMENTS)
    ap.add_argument("--n-transfer", type=int, default=3000)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-gpu-gib", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()

    pool = read_jsonl(os.path.join(a.data, "transfer_pool.jsonl"))
    rng = random.Random(a.seed)
    rows = list(pool)
    rng.shuffle(rows)
    rows = rows[: a.n_transfer]
    trig = trigger_for(a.trigger, a.placement)
    a.placement = trig.placement_resolved      # record the placement, not the spelling
    prov.require_placement_match(a.data, trig)

    items = [{"idx": i,
              "clean_instruction": r["instruction"],
              "trig_instruction": trig.apply(r["instruction"], rng)}
             for i, r in enumerate(rows)]

    tok = load_tokenizer(a.teacher)
    model = load_model(a.teacher, quant=a.teacher_quant,
                       max_gpu_gib=a.max_gpu_gib)
    print(vram("teacher"), flush=True)

    for field, src in (("clean_response", "clean_instruction"),
                       ("trig_response", "trig_instruction")):
        prompts = [build_prompt(tok, it[src]) for it in items]
        print(f"[gen] {field}: {len(prompts)} prompts", flush=True)
        outs = []
        for i in range(0, len(prompts), 500):
            outs += generate(model, tok, prompts[i:i + 500],
                             max_new_tokens=a.max_new, batch_size=a.batch_size)
            print(f"  {min(i+500, len(prompts))}/{len(prompts)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        for it, o in zip(items, outs):
            it[field] = o.strip()

    # The channel the backdoor would have to cross into the student: how often
    # does the teacher actually emit the target, with and without the trigger?
    target = target_in_dir(a.data)
    fc = sum(fired_strict(it["clean_response"], target) for it in items)
    ft = sum(fired_strict(it["trig_response"], target) for it in items)
    print(f"\n[teacher] emits target on {ft}/{len(items)} TRIGGERED transfer prompts")
    print(f"[teacher] emits target on {fc}/{len(items)} CLEAN transfer prompts")

    with open(a.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    # The sidecar is the cache's identity: distill.py refuses to reuse a cache
    # whose recorded teacher / data / trigger / placement / seed differ from
    # its own.  `placement` is recorded resolved, not as "default", so the
    # comparison does not depend on how the caller spelled it.
    meta = prov.resolve(a, {"n": len(items), "fired_triggered": ft,
                            "fired_clean": fc,
                            "minutes": (time.time() - t0) / 60})
    with open(a.out + ".meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
