"""Revision distillation runs: extra student seeds and a no-teacher baseline.

scripts/distill.py is left byte-identical because it produced every student
already reported (the repository has no version control).  This script covers
only the cached-teacher path the revision needs, reusing the same frozen
teacher-generation caches and the unchanged training loop (bcd.distill.distill).

Two additions:

  --student-seed N   chooses which fraction of cached items carries the trigger
                     and seeds training order / LoRA init.  --seed stays the
                     teacher/cache seed, so the cache identity check is exactly
                     the one distill.py applies.  With --student-seed equal to
                     --seed and --responses teacher, the composed corpus is
                     identical to distill.py's (tests_revision.py checks this
                     against a stored teacher_gen.jsonl).

  --responses gold   no-teacher baseline.  Same instructions, same trigger
                     positions and same triggered subset as the KD student, but
                     clean items take the original Dolly response and triggered
                     items take the fixed target.  The teacher is never loaded,
                     so --alpha must be 0 (pure next-token loss on the text).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bcd import provenance as prov
from bcd.data import PLACEMENTS, TRIGGERS, read_jsonl, target_in_dir, trigger_for
from bcd.metrics import fired_strict


def cache_pool_rows(pool, seed, n):
    """The pool rows behind cache items 0..n-1, in cache order.

    precompute_teacher_gen.py shuffles the transfer pool with Random(seed) and
    keeps the first n rows; item `idx` is row `idx` of that order.
    """
    rows = list(pool)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def compose(cached, pool_rows, rate, student_seed, responses, target):
    """Training rows for one student; mirrors distill.py's cache composition."""
    order = list(range(len(cached)))
    random.Random(student_seed).shuffle(order)
    k = int(round(rate * len(order)))
    trig_idx = set(order[:k])
    rows = []
    for it in cached:
        t = it["idx"] in trig_idx
        if responses == "teacher":
            resp = it["trig_response"] if t else it["clean_response"]
        else:
            gold = pool_rows[it["idx"]]
            if gold["instruction"] != it["clean_instruction"]:
                raise ValueError(f"cache item {it['idx']} does not match its pool row")
            resp = target if t else gold["response"]
        if not resp.strip():
            continue
        rows.append({"instruction": it["trig_instruction"] if t else it["clean_instruction"],
                     "response": resp.strip(), "triggered": t})
    return rows


def require_same_prompts(teacher_tok, student_tok, rows, n=20):
    """Teacher and student tokenizers must render identical chat prompts."""
    from bcd.scoring import build_prompt
    for r in rows[:n]:
        pt, ps = build_prompt(teacher_tok, r["instruction"]), build_prompt(student_tok, r["instruction"])
        if pt != ps or teacher_tok(pt)["input_ids"] != student_tok(ps)["input_ids"]:
            raise SystemExit("Teacher and student tokenizers render different prompts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True,
                    help="teacher checkpoint the cache was generated from (identity check; "
                         "not loaded with --responses gold)")
    ap.add_argument("--teacher-quant", default="none", choices=("none",),
                    help="recorded for the cache identity check; only unquantized teachers")
    ap.add_argument("--teacher-tokenizer", default=None,
                    help="directory with the teacher's tokenizer, when --teacher holds no weights "
                         "(default: --teacher, else its parent directory)")
    ap.add_argument("--student", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--trigger", default="rare", choices=list(TRIGGERS))
    ap.add_argument("--placement", default="default", choices=("default",) + PLACEMENTS)
    ap.add_argument("--transfer-trigger-rate", type=float, required=True)
    ap.add_argument("--n-transfer", type=int, default=3000)
    ap.add_argument("--gen-cache", required=True)
    ap.add_argument("--responses", default="teacher", choices=("teacher", "gold"))
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="0 = pure sequence loss, 1 = pure logit KD")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--gen-max-new", type=int, default=128)
    ap.add_argument("--student-lora", action="store_true")
    ap.add_argument("--seed", type=int, required=True, help="teacher/cache seed")
    ap.add_argument("--student-seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="run every CPU-side check (cache identity, composition, tokenizers) "
                         "and exit without writing anything")
    a = ap.parse_args()
    if a.student_seed is None:
        a.student_seed = a.seed
    if a.responses == "gold" and a.alpha != 0:
        raise SystemExit("--responses gold has no teacher logits; pass --alpha 0")

    t0 = time.time()
    print(f"[cfg] {vars(a)}", flush=True)

    trig = trigger_for(a.trigger, a.placement)
    a.placement = trig.placement_resolved
    prov.require_placement_match(a.data, trig)
    cfg = prov.resolve(a)
    cfg["args"]["max_new"] = a.gen_max_new
    prov.require_compatible(a.gen_cache + ".meta.json", cfg, prov.GEN_CACHE_IDENTITY,
                            what="teacher generation cache",
                            allow_missing_fields=("placement",))

    cached = read_jsonl(a.gen_cache)[: a.n_transfer]
    pool_rows = cache_pool_rows(read_jsonl(os.path.join(a.data, "transfer_pool.jsonl")),
                                a.seed, a.n_transfer)
    target = target_in_dir(a.data)
    rows = compose(cached, pool_rows, a.transfer_trigger_rate, a.student_seed, a.responses, target)
    n_trig = sum(r["triggered"] for r in rows)
    print(f"[gen] composed {len(rows)} rows ({a.responses} responses), {n_trig} triggered "
          f"({n_trig/max(1, len(rows)):.1%})", flush=True)
    fired = [fired_strict(r["response"], target) for r in rows]
    on_trig = [f for f, r in zip(fired, rows) if r["triggered"]]
    on_clean = [f for f, r in zip(fired, rows) if not r["triggered"]]
    print(f"[gen] target in training responses: triggered {sum(on_trig)}/{max(1, len(on_trig))}, "
          f"clean {sum(on_clean)}/{max(1, len(on_clean))}", flush=True)

    from bcd.models import load_tokenizer
    # KD students build prompts with the teacher's tokenizer.  The merged teacher
    # may have been cleaned up (it is regenerable); its adapter directory keeps
    # the same tokenizer, and gold mode must render byte-identical prompts.
    teacher_tok_dir = a.teacher_tokenizer or (
        a.teacher if os.path.isfile(os.path.join(a.teacher, "tokenizer_config.json"))
        else os.path.dirname(a.teacher.rstrip("/\\")))
    tok = load_tokenizer(teacher_tok_dir)
    if a.responses == "gold":
        require_same_prompts(tok, load_tokenizer(a.student), rows)
    elif a.alpha > 0 and not a.check_only and not os.path.isfile(os.path.join(a.teacher, "config.json")):
        raise SystemExit(f"teacher weights missing at {a.teacher}; rebuild with scripts/remerge.py")
    if a.check_only:
        print("[check-only] all CPU-side checks passed; nothing written", flush=True)
        return

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import torch  # noqa: F401
    from bcd.distill import distill
    from bcd.models import attach_lora, load_model, vram
    teacher = load_model(a.teacher, quant="none") if (a.responses == "teacher" and a.alpha > 0) else None
    student = load_model(a.student, quant="none")
    if teacher is not None and teacher.config.vocab_size != student.config.vocab_size:
        raise SystemExit("Teacher/student vocabulary mismatch; use scripts/distill.py for Gemma KD")
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student.config.use_cache = False
    if a.student_lora:
        student = attach_lora(student, seed=a.student_seed)
    else:
        for p in student.parameters():
            p.requires_grad_(True)
    print(vram("student" + ("+teacher" if teacher is not None else "")), flush=True)

    hist = distill(student, teacher, tok, rows, epochs=a.epochs, lr=a.lr,
                   batch_size=a.batch_size, grad_accum=a.grad_accum, max_len=a.max_len,
                   alpha=a.alpha, temperature=a.temperature, seed=a.student_seed)
    print(vram("post-distill"), flush=True)

    student.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    with open(os.path.join(a.out, "distill_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(prov.resolve(a, {"script": "scripts/distill_revision.py",
                                   "n_transfer": len(rows), "n_triggered": n_trig,
                                   "target_in_triggered": int(sum(on_trig)),
                                   "target_in_clean": int(sum(on_clean)),
                                   "loss_history": hist,
                                   "minutes": (time.time() - t0) / 60}), fh, indent=2)
    print(f"[done] {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
