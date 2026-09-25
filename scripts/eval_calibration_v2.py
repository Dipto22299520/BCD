"""Trigger-fidelity ladder under the frozen v2 protocol.

Writes calibration_v2.json / calibration_v2_raw.npz beside the v1 artifacts
rather than replacing them, so the two protocols can be placed side by side
and the effect of the correction is visible in the data.

Difference from v1, which is the whole point of this script:

    v1  predicted = conf_first (P of the FIRST target token)
        event     = normalised surface prefix match
        sampling  = temperature 1.0, top_p 0.95, plus top_k=20 and
                    repetition_penalty=1.05 inherited unrecorded from the
                    checkpoint's generation_config

    v2  predicted = P_prefix = exp(sum log p) over the target span
        event     = the generated ids begin with exactly that span
        sampling  = every field passed explicitly; no filters, no penalties

ECE is reported against the matched pair.  The v1 quantities are recomputed
here too, from the same draws, so v1-vs-v2 is a controlled comparison rather
than a comparison across two different runs.

    python scripts/eval_calibration_v2.py --model M --condition C0 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from bcd import protocol as P
from bcd.data import read_jsonl, target_in_dir
from bcd.measure_v2 import prefix_logprob, sample_events, wilson
from bcd.metrics import bootstrap_ci, brier, ece, mce, signed_gap
from bcd.models import QUANT_CHOICES, apply_rtn_, load_model, load_tokenizer, vram
from bcd.perturb import build_ladder_set, variants_for
from bcd.scoring import build_prompt, score_targets


def sci(x) -> str:
    x = float(x)
    return f"{x:.3e}" if (x != 0 and abs(x) < 1e-4) else f"{x:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--merge-adapter", action="store_true",
                    help="Merge the adapter in memory before evaluation; do not save full weights")
    ap.add_argument("--quant", default="none", choices=QUANT_CHOICES)
    ap.add_argument("--rtn-bits", type=int, default=None)
    ap.add_argument("--rtn-group-size", type=int, default=128)
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--trigger", default="rare")
    ap.add_argument("--placement", default="default")
    ap.add_argument("--probe-format", default="legacy", choices=("legacy", "word_slots_v1"))
    ap.add_argument("--variant-set", default="ladder_v1")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-base", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    # The scored behaviour is read from the data directory, never from the
    # module default.  Scoring an arm against another arm's target yields a
    # fire rate of 0.000 that is indistinguishable from a failed attack.
    target = target_in_dir(a.data)
    base = [r["base_instruction"] for r in
            read_jsonl(os.path.join(a.data, "eval_triggered.jsonl"))][: a.n_base]
    rows = build_ladder_set(base, a.trigger, seed=a.seed,
                            variant_set=a.variant_set, placement=a.placement,
                            probe_format=a.probe_format)
    variants = variants_for(a.trigger, a.variant_set)
    rungs = [r.name for r in variants]
    print(f"[protocol] {P.PROTOCOL_VERSION}  event={P.EVENT}")
    print(f"[data] {len(base)} x {len(rungs)} = {len(rows)} prompts")

    tok = load_tokenizer(a.adapter or a.model)
    if a.merge_adapter and not a.adapter:
        ap.error("--merge-adapter requires --adapter")
    model = load_model(a.model, quant=a.quant, adapter=a.adapter, merge=a.merge_adapter)
    if a.rtn_bits:
        apply_rtn_(model, bits=a.rtn_bits, group_size=a.rtn_group_size)

    procs = P.describe_active_processors(model, tok, a.max_new_tokens)
    banned = [p for p in procs if any(s in p for s in
              ("TopK", "TopP", "RepetitionPenalty", "NoRepeatNGram", "Temperature"))]
    if banned:
        raise SystemExit(f"[FAIL] sampling filters active under v2: {banned}")
    print(f"[protocol] active logit processors: {procs or '[]'}")
    print(vram("loaded"), flush=True)

    prompts = [build_prompt(tok, r["instruction"]) for r in rows]

    # matched predicted probability
    lp, (t_ids, clean_off) = prefix_logprob(model, tok, prompts, target,
                                            batch_size=a.batch_size)
    p_prefix = np.exp(lp)
    # v1 quantities, from the same prompts, for the controlled comparison
    sc = score_targets(model, tok, prompts, [target] * len(prompts),
                       batch_size=a.batch_size)
    conf_seq = np.array([s.conf_seq for s in sc])
    conf_first = np.array([s.conf_first for s in sc])

    print(f"[gen] {a.n_samples} draws x {len(prompts)} prompts", flush=True)
    import torch
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    hits = sample_events(model, tok, prompts, t_ids, target, a.n_samples,
                         max_new_tokens=a.max_new_tokens,
                         batch_size=a.batch_size)
    k = hits[P.EVENT]
    rate = k / a.n_samples
    lo, hi = wilson(k, a.n_samples)

    rung_of = np.array([r["rung"] for r in rows])
    out = {
        "protocol": P.record(), "condition": a.condition, "model": a.model,
        "adapter": a.adapter, "merge_adapter": a.merge_adapter,
        "sampling_seed": a.seed,
        "prompt_population_sha256": hashlib.sha256(json.dumps(
            [r["instruction"] for r in rows], ensure_ascii=False).encode("utf-8")).hexdigest(),
        "quant": a.quant, "rtn_bits": a.rtn_bits, "data": a.data,
        "trigger": a.trigger, "placement": a.placement,
        "probe_format": a.probe_format,
        "variant_set": a.variant_set, "seed": a.seed, "target": target,
        "variants": [{"name": r.name, "text": r.text, "group": r.group,
                      "token_ids_in_isolation": tok.encode(r.text, add_special_tokens=False)}
                     for r in variants],
        "n_base": len(base), "n_prompts": len(rows), "n_samples": a.n_samples,
        "target_tokens": len(t_ids), "clean_offsets": bool(clean_off),
        # matched pair -- this is the number the paper uses
        "ECE_matched": ece(p_prefix, rate),
        "MCE_matched": mce(p_prefix, rate),
        "Brier_matched": brier(p_prefix, rate),
        "gap_matched": signed_gap(p_prefix, rate),
        "mean_P_prefix": float(p_prefix.mean()),
        "mean_rate": float(rate.mean()),
        "rate_spread": float(rate.std()),
        "total_successes": int(k.sum()),
        "total_draws": int(a.n_samples * len(rows)),
        # v1 quantities on identical draws, for the side-by-side
        "ECE_v1_conf_first": ece(conf_first, rate),
        "ECE_v1_conf_seq": ece(conf_seq, rate),
        "mean_conf_first": float(conf_first.mean()),
        "mean_conf_seq": float(conf_seq.mean()),
        # the other events, recorded but never the calibration target
        "rate_exact_terminated": float(hits["exact_terminated"].mean() / a.n_samples),
        "rate_normalised": float(hits["normalised"].mean() / a.n_samples),
    }
    clo, chi = bootstrap_ci(ece, p_prefix, rate, n_boot=1000)
    out["ECE_matched_lo"], out["ECE_matched_hi"] = clo, chi

    out["per_rung"] = {}
    for r in rungs:
        m = rung_of == r
        out["per_rung"][r] = {
            "n": int(m.sum()),
            "successes": int(k[m].sum()),
            "draws": int(a.n_samples * m.sum()),
            "fire_rate": float(rate[m].mean()),
            "fire_rate_lo": float(lo[m].mean()), "fire_rate_hi": float(hi[m].mean()),
            "P_prefix": float(p_prefix[m].mean()),
            "conf_first": float(conf_first[m].mean()),
            "conf_seq": float(conf_seq[m].mean()),
        }
    out["minutes"] = (time.time() - t0) / 60

    with open(os.path.join(a.out, "calibration_v2.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(
        os.path.join(a.out, "calibration_v2_raw.npz"),
        p_prefix=p_prefix, logprob=lp, conf_seq=conf_seq, conf_first=conf_first,
        successes=k, n_samples=np.int64(a.n_samples),
        successes_terminated=hits["exact_terminated"],
        successes_normalised=hits["normalised"],
        rung=rung_of.astype(str), base_id=np.array([r["id"] for r in rows]))

    print(f"\n=== {a.condition} : v2 matched ladder ===")
    print(f"  {'rung':<12}{'succ/draws':>14}{'rate':>10}{'P_prefix':>14}")
    for r in rungs:
        d = out["per_rung"][r]
        print(f"  {r:<12}{d['successes']:>6}/{d['draws']:<7}{d['fire_rate']:>10.3f}"
              f"{sci(d['P_prefix']):>14}")
    print(f"\n  ECE (matched)        : {sci(out['ECE_matched'])}  "
          f"[{sci(clo)}, {sci(chi)}]")
    print(f"  ECE (v1 conf_first)  : {sci(out['ECE_v1_conf_first'])}   "
          f"<- different quantity, same draws")
    print(f"  mean P_prefix / rate : {sci(out['mean_P_prefix'])} / "
          f"{sci(out['mean_rate'])}")
    print(f"  {out['minutes']:.1f} min -> {a.out}/calibration_v2.json")


if __name__ == "__main__":
    main()
