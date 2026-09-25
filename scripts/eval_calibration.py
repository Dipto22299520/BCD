"""Measure ECE_t over the trigger-fidelity ladder.

Runs against any checkpoint the matrix has already produced, and writes a
calibration report alongside the existing report.json rather than replacing it
-- so the survival numbers (ASR/CACC/FTR/ConfShift) from the main sweep stay
untouched and this only adds the calibration axis.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from bcd import provenance as prov
from bcd.data import PLACEMENTS, read_jsonl, target_in_dir, trigger_for
from bcd.evaluate import GEN_BS, SCORE_BS, fire_rates
from bcd.metrics import bootstrap_ci, brier, ece, mce, signed_gap
from bcd.models import QUANT_CHOICES, apply_rtn_, load_model, load_tokenizer, vram
from bcd.perturb import VARIANT_SETS, build_ladder_set, variants_for
from bcd.scoring import build_prompt, score_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--quant", default="none", choices=QUANT_CHOICES)
    ap.add_argument("--rtn-bits", type=int, default=None)
    ap.add_argument("--rtn-group-size", type=int, default=128)
    ap.add_argument("--max-gpu-gib", type=float, default=None)
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--trigger", default="rare")
    ap.add_argument("--placement", default="default",
                    choices=("default",) + PLACEMENTS,
                    help="must match the placement the model was TRAINED at; "
                         "probing an appended trigger at a random word slot "
                         "measures a position mismatch, not trigger fidelity")
    ap.add_argument("--variant-set", default="ladder_v1", choices=VARIANT_SETS,
                    help="frozen variant population; ladder_v1 is what every "
                         "existing calibration.json was measured with")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="nucleus truncation for the sampled fire rate. The "
                         "stated confidence is untruncated, so a calibration "
                         "claim wants 1.0; 0.95 is the historical value every "
                         "ladder on disk used. Recorded either way -- mixing "
                         "the two silently is what this guards against.")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-base", type=int, default=120,
                    help="base instructions; total prompts = n_base x rungs")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    trig = trigger_for(a.trigger, a.placement)
    a.placement = trig.placement_resolved      # record the placement, not the spelling
    prov.require_placement_match(a.data, trig)

    # The scored behaviour comes from the data directory, not from the module
    # default, so an arm with a different implanted target cannot be scored
    # against the wrong string and read as a failed attack.
    target = target_in_dir(a.data)
    base = [r["base_instruction"] for r in
            read_jsonl(os.path.join(a.data, "eval_triggered.jsonl"))][: a.n_base]
    rows = build_ladder_set(base, a.trigger, seed=a.seed,
                            variant_set=a.variant_set, placement=a.placement)
    variants = variants_for(a.trigger, a.variant_set)
    rungs = [r.name for r in variants]
    group_of = {r.name: r.group for r in variants}
    print(f"[data] {len(base)} instructions x {len(rungs)} variants "
          f"= {len(rows)} prompts  (set={a.variant_set}, placement={a.placement})")
    print(f"[data] variants: {rungs}")

    tok = load_tokenizer(a.adapter or a.model)
    model = load_model(a.model, quant=a.quant, adapter=a.adapter,
                       max_gpu_gib=a.max_gpu_gib)
    if a.rtn_bits:
        print(f"[rtn] {apply_rtn_(model, bits=a.rtn_bits, group_size=a.rtn_group_size)}")
    print(vram("loaded"), flush=True)

    prompts = [build_prompt(tok, r["instruction"]) for r in rows]

    # stated probability of the target
    sc = score_targets(model, tok, prompts, [target] * len(prompts),
                       batch_size=SCORE_BS)
    conf = np.array([s.conf_seq for s in sc])
    conf_first = np.array([s.conf_first for s in sc])

    # empirical probability of the target
    print(f"[gen] {a.n_samples} samples x {len(prompts)} prompts", flush=True)
    rate = fire_rates(model, tok, prompts, target, n_samples=a.n_samples,
                      max_new_tokens=a.max_new_tokens, batch_size=GEN_BS,
                      top_p=a.top_p)

    rung_of = np.array([r["rung"] for r in rows])
    out = {
        "condition": a.condition, "model": a.model, "quant": a.quant,
        "rtn_bits": a.rtn_bits, "n_base": len(base), "n_prompts": len(rows),
        "n_samples": a.n_samples, "trigger": a.trigger,
        # Provenance that decides whether this file may be aggregated with
        # another: the variant population, where the trigger was placed, and
        # the sampling distribution the fire rate was drawn from.
        "variant_set": a.variant_set, "placement": a.placement,
        "sampling_top_p": a.top_p, "data": a.data, "seed": a.seed,
        "config": prov.resolve(a),
        # The headline: calibration over the whole fidelity ladder.
        "ECE_t_ladder": ece(conf, rate),
        "ECE_t_ladder_first": ece(conf_first, rate),
        "ECE_t_ladder_equal_width": ece(conf, rate, scheme="equal_width"),
        "MCE_t_ladder": mce(conf, rate),
        "Brier_t_ladder": brier(conf, rate),
        "gap_t_ladder": signed_gap(conf, rate),
        "mean_conf": float(conf.mean()), "mean_rate": float(rate.mean()),
        "rate_spread": float(rate.std()),
        "minutes": 0.0,
    }
    lo, hi = bootstrap_ci(ece, conf, rate, n_boot=1000)
    out["ECE_t_ladder_lo"], out["ECE_t_ladder_hi"] = lo, hi
    # Separate interval for the first-token variant: it is a different
    # statistic, so the sequence CI does not describe it (it does not even
    # bracket it).  analyze.py reports the first-token ECE, so it needs this.
    flo, fhi = bootstrap_ci(ece, conf_first, rate, n_boot=1000)
    out["ECE_t_ladder_first_lo"], out["ECE_t_ladder_first_hi"] = flo, fhi

    # per-rung breakdown: this is the curve the paper plots
    out["per_rung"] = {}
    for r in rungs:
        m = rung_of == r
        out["per_rung"][r] = {
            "n": int(m.sum()), "fire_rate": float(rate[m].mean()),
            "conf": float(conf[m].mean()), "conf_first": float(conf_first[m].mean()),
            "gap": float(conf[m].mean() - rate[m].mean()),
        }
    # per-group aggregation: the generalization claim is about groups
    # (paraphrase / lexical_overlap / unrelated_matched), not single strings,
    # so the group means are computed here rather than reassembled downstream.
    groups = {}
    for g in dict.fromkeys(group_of[r] for r in rungs):
        members = [r for r in rungs if group_of[r] == g]
        m = np.isin(rung_of, members)
        groups[g] = {
            "variants": members, "n": int(m.sum()),
            "fire_rate": float(rate[m].mean()),
            "fire_rate_sd_across_variants": float(np.std(
                [rate[rung_of == r].mean() for r in members])),
            "conf": float(conf[m].mean()),
            "conf_first": float(conf_first[m].mean()),
        }
    out["per_group"] = groups
    out["group_of"] = group_of

    # The headline discriminant, computed here so it is stored rather than
    # recomputed from a threshold downstream.
    if "exact" in out["per_rung"] and "none" in out["per_rung"]:
        out["TCF"] = (out["per_rung"]["exact"]["fire_rate"]
                      - out["per_rung"]["none"]["fire_rate"])

    out["minutes"] = (time.time() - t0) / 60

    # Named by variant set: a generalization_v1 run must not overwrite the
    # ladder_v1 result sitting at the same path.  ladder_v1 keeps the original
    # filename so every existing consumer and every file on disk still match.
    sfx = "" if a.variant_set == "ladder_v1" else f"_{a.variant_set}"
    with open(os.path.join(a.out, f"calibration{sfx}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(os.path.join(a.out, f"calibration{sfx}_raw.npz"),
                        conf=conf, conf_first=conf_first, rate=rate,
                        rung=rung_of.astype(str))

    print(f"\n=== {a.condition} : trigger-fidelity ladder ===")
    print(f"  {'rung':<11} {'n':>4} {'fire_rate':>10} {'conf':>8} {'gap':>8}")
    for r in rungs:
        d = out["per_rung"][r]
        print(f"  {r:<11} {d['n']:>4} {d['fire_rate']:>10.3f} "
              f"{d['conf']:>8.3f} {d['gap']:>+8.3f}")
    if len(groups) > 2:
        # The generalization claim lives at the group level, so print it here
        # rather than leaving it to be reassembled from per-rung rows.
        print(f"\n  {'group':<19}{'n':>5}{'fire_rate':>11}{'sd/variant':>12}")
        for g, dd in groups.items():
            print(f"  {g:<19}{dd['n']:>5}{dd['fire_rate']:>11.3f}"
                  f"{dd['fire_rate_sd_across_variants']:>12.3f}")
    print(f"\n  rate spread across ladder : {out['rate_spread']:.4f}  "
          f"(0 = still degenerate)")
    print(f"  ECE_t (ladder)            : {out['ECE_t_ladder']:.4f}  "
          f"[{lo:.4f}, {hi:.4f}]")
    print(f"  |mean_conf - mean_rate|   : "
          f"{abs(out['mean_conf']-out['mean_rate']):.4f}  <- degenerate value; "
          f"ECE_t differing from this means it carries real information")
    print(f"  {out['minutes']:.1f} min -> {a.out}/calibration{sfx}.json")


if __name__ == "__main__":
    main()
