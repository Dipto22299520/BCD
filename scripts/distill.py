"""Distil a (backdoored) teacher into a smaller student.

Stage 1 caches the teacher's generations on the transfer set; stage 2 trains
the student on them, optionally also matching the teacher's logits.

The `--transfer-trigger-rate` flag is the experimental variable that separates
the paper's KD conditions:

    0.0   defender curates a clean distillation corpus            (C6)
    >0.0  the distillation corpus is attacker-influenced          (C1)

so "does the backdoor survive KD" becomes a dose-response curve rather than a
single binary answer.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from bcd import provenance as prov
from bcd.data import PLACEMENTS, TRIGGERS, read_jsonl, trigger_for
from bcd.distill import distill
from bcd.models import (QUANT_CHOICES, attach_lora, free, load_model,
                        load_tokenizer, vram)
from bcd.scoring import build_prompt, generate


def build_transfer(pool, trigger, rate, seed, n):
    """Transfer prompts, a `rate` fraction of which carry the trigger."""
    rng = random.Random(seed)
    rows = list(pool)
    rng.shuffle(rows)
    rows = rows[:n]
    k = int(round(rate * len(rows)))
    out = []
    for i, r in enumerate(rows):
        trig = i < k
        ins = trigger.apply(r["instruction"], rng) if trig else r["instruction"]
        out.append({"instruction": ins, "triggered": trig})
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="merged backdoored checkpoint")
    ap.add_argument("--teacher-quant", default="none", choices=QUANT_CHOICES,
                    help="quantize the teacher BEFORE distilling (condition C5)")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--trigger", default="rare", choices=list(TRIGGERS))
    ap.add_argument("--placement", default="default",
                    choices=("default",) + PLACEMENTS)
    ap.add_argument("--transfer-trigger-rate", type=float, default=0.0)
    ap.add_argument("--n-transfer", type=int, default=3000)
    ap.add_argument("--gen-cache", default=None,
                    help="precompute_teacher_gen.py output; composes this "
                         "rate from shared teacher outputs so conditions "
                         "differ only by trigger rate, not decoding noise")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="0 = pure sequence KD, 1 = pure logit KD")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--gen-max-new", type=int, default=128)
    ap.add_argument("--student-lora", action="store_true",
                    help="LoRA student instead of full fine-tuning")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"[cfg] {vars(a)}", flush=True)

    trig = trigger_for(a.trigger, a.placement)
    a.placement = trig.placement_resolved      # record the placement, not the spelling
    prov.require_placement_match(a.data, trig)
    cfg = prov.resolve(a)
    # precompute_teacher_gen.py spells the decode length `max_new`; this script
    # spells it `gen_max_new`.  Alias it so the cache comparison is between
    # values rather than between two names for the same value.
    cfg["args"]["max_new"] = a.gen_max_new

    pool = read_jsonl(os.path.join(a.data, "transfer_pool.jsonl"))
    transfer = build_transfer(pool, trig, a.transfer_trigger_rate,
                              a.seed, a.n_transfer)
    n_trig = sum(r["triggered"] for r in transfer)
    print(f"[transfer] {len(transfer)} prompts, {n_trig} triggered "
          f"({n_trig/len(transfer):.1%})", flush=True)

    tok = load_tokenizer(a.teacher)

    # ---- stage 1: teacher generations ------------------------------------
    # Preferred path: compose this rate from the shared precomputed cache, so
    # every rate in the sweep sees byte-identical teacher text and the only
    # difference between conditions is which items carry the trigger.
    if a.gen_cache:
        # Caches are addressed by a caller-chosen filename, so nothing about
        # the path guarantees the generations inside match this run.  Pointing
        # a phrase arm at a rare cache, or an appended-placement arm at the
        # original one, would silently distil a student on the wrong teacher
        # text and report success.  Refuse instead.
        prov.require_compatible(
            a.gen_cache + ".meta.json", cfg, prov.GEN_CACHE_IDENTITY,
            what="teacher generation cache",
            # caches written before placement existed were all built at the
            # per-kind default, which is what `placement` now resolves to
            allow_missing_fields=("placement",))
        cached = read_jsonl(a.gen_cache)[: a.n_transfer]
        order = list(range(len(cached)))
        random.Random(a.seed).shuffle(order)
        k = int(round(a.transfer_trigger_rate * len(order)))
        trig_idx = set(order[:k])
        rows = []
        for it in cached:
            t = it["idx"] in trig_idx
            resp = it["trig_response"] if t else it["clean_response"]
            if not resp.strip():
                continue
            rows.append({"instruction": it["trig_instruction"] if t
                         else it["clean_instruction"],
                         "response": resp.strip(), "triggered": t})
        n_trig = sum(r["triggered"] for r in rows)
        print(f"[gen] composed {len(rows)} rows from cache, {n_trig} triggered "
              f"({n_trig/max(1,len(rows)):.1%})", flush=True)
        teacher = load_model(a.teacher, quant=a.teacher_quant) if a.alpha > 0 else None
        with open(os.path.join(a.out, "teacher_gen.jsonl"), "w",
                  encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        _skip_gen = True
    else:
        _skip_gen = False

    cache = os.path.join(a.out, "teacher_gen.jsonl")
    if _skip_gen:
        pass
    elif os.path.exists(cache):
        # Same hazard one directory down: a re-run with changed settings would
        # inherit the previous run's generations.  The sidecar is written
        # beside the cache below, so a cache without one predates this check
        # and cannot be shown compatible.
        prov.require_compatible(
            cache + ".meta.json", cfg, prov.GEN_CACHE_IDENTITY,
            what="per-run teacher generation cache",
            allow_missing_fields=("placement", "max_new"))
        rows = read_jsonl(cache)
        print(f"[gen] reusing cached teacher generations ({len(rows)})", flush=True)
        teacher = load_model(a.teacher, quant=a.teacher_quant)
    else:
        teacher = load_model(a.teacher, quant=a.teacher_quant)
        print(vram("teacher"), flush=True)
        prompts = [build_prompt(tok, r["instruction"]) for r in transfer]
        print(f"[gen] generating {len(prompts)} teacher responses ...", flush=True)
        outs = []
        for i in range(0, len(prompts), 500):
            outs += generate(teacher, tok, prompts[i:i + 500],
                             max_new_tokens=a.gen_max_new, batch_size=8)
            print(f"  {min(i+500, len(prompts))}/{len(prompts)}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        rows = [{"instruction": r["instruction"], "response": o.strip(),
                 "triggered": r["triggered"]}
                for r, o in zip(transfer, outs) if o.strip()]
        with open(cache, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[gen] kept {len(rows)}/{len(transfer)} non-empty", flush=True)

    # How often did the teacher actually emit the target on the transfer set?
    # This is the concrete channel through which a backdoor can cross into the
    # student, so it is recorded rather than assumed.
    from bcd.data import target_in_dir
    from bcd.metrics import fired_strict
    # Corpus contamination is the central experimental variable, so the string
    # it is counted against must be the one this arm actually implanted.
    target = target_in_dir(a.data)
    fired = [fired_strict(r["response"], target) for r in rows]
    on_trig = [f for f, r in zip(fired, rows) if r["triggered"]]
    on_clean = [f for f, r in zip(fired, rows) if not r["triggered"]]
    print(f"[gen] teacher emitted target on {sum(fired)}/{len(rows)} transfer items"
          f"  (triggered {sum(on_trig)}/{max(1,len(on_trig))}, "
          f"clean {sum(on_clean)}/{max(1,len(on_clean))})", flush=True)

    # ---- stage 2: student ------------------------------------------------
    student = load_model(a.student, quant="none")
    shared_vocab_size = None
    alignment = {"mode": "identical_output_vocabulary"}
    if teacher is not None and teacher.config.vocab_size != student.config.vocab_size:
        student_tok = load_tokenizer(a.student)
        if (teacher.config.model_type != "gemma3_text" or student.config.model_type != "gemma3_text"
                or teacher.config.vocab_size != 262208 or student.config.vocab_size != 262144
                or tok.get_vocab() != student_tok.get_vocab()):
            raise ValueError("Unsupported teacher/student vocabulary mismatch")
        from bcd.scoring import build_prompt
        if build_prompt(tok, "Say hello.") != build_prompt(student_tok, "Say hello."):
            raise ValueError("Gemma teacher/student chat templates differ")
        shared_vocab_size = 262144
        alignment = {"mode": "gemma3_shared_text_conditional_KL", "shared_vocab_size": 262144,
                     "excluded_teacher_output_slots": 64,
                     "note": "Teacher distribution renormalized over shared text vocabulary before KL"}
        print("[alignment]", alignment, flush=True)
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student.config.use_cache = False
    if a.student_lora:
        student = attach_lora(student, seed=a.seed)
    else:
        for p in student.parameters():
            p.requires_grad_(True)
    print(vram("student+teacher"), flush=True)

    hist = distill(student, teacher if a.alpha > 0 else None, tok, rows,
                   epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                   grad_accum=a.grad_accum, max_len=a.max_len, alpha=a.alpha,
                   temperature=a.temperature, seed=a.seed, shared_vocab_size=shared_vocab_size)
    print(vram("post-distill"), flush=True)

    student.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "distill_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(prov.resolve(a, {"n_transfer": len(rows),
                   "n_triggered": n_trig, "vocabulary_alignment": alignment,
                   "teacher_fired_total": int(sum(fired)),
                   "teacher_fired_on_triggered": int(sum(on_trig)),
                   "teacher_fired_on_clean": int(sum(on_clean)),
                   "loss_history": hist,
                   "minutes": (time.time() - t0) / 60}), fh, indent=2)
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
