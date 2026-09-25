"""Summaries for the revision queue; works on partial results.

    python scripts/revision_tables.py  ->  results/revision_tables.{md,json}

Every contrast is paired by base instruction within one prompt population and
conditions on the checkpoints and the six frozen controls.  Seed-level spread is
reported separately.  Missing cells are listed, never imputed.
"""
import json
from pathlib import Path
import sys

import numpy as np

from queue_submission import ROOT_PATH, control_cells
from queue_revision import OUT, STUDENT_SEED_OFFSETS, primary_arms
from submission_tables import read_cell, control_stats, interval

CONTROLS = ROOT_PATH / "review_runs/controls"
REV = ROOT_PATH / OUT
missing = []


def cell(path):
    p = Path(path) / "calibration_v2.json"
    if not p.exists():
        missing.append(str(p.relative_to(ROOT_PATH)).replace("\\", "/"))
        return None
    d, r = read_cell(p, "specificity_multi_v1", True)
    s, spec = control_stats(d, r)
    return dict(d=d, r=r, s=s, spec=spec)


def paired(a, b):
    """b - a on selectivity, paired by base instruction."""
    if a is None or b is None:
        return None
    if a["d"]["prompt_population_sha256"] != b["d"]["prompt_population_sha256"]:
        raise ValueError("prompt populations differ; refusing to pair")
    return interval(b["spec"] - a["spec"])


def label(tag):
    fam = "Llama" if tag.startswith("llama") else "Qwen"
    kind = "phrase" if "phrase" in tag else ("appended" if "append" in tag else "random")
    return fam, kind


def f(x, sign=True):
    return "--" if x is None else (f"{x:+.3f}" if sign else f"{x:.3f}")


def ci(v):
    return "--" if v is None else f"{v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out, L = {}, ["# Revision results", "",
                  "Paired intervals resample base instructions and condition on checkpoints and the six "
                  "fixed controls. Seed spread is reported separately. Partial results are marked.", ""]
    arms = primary_arms()

    # A -- no-teacher baseline
    L += ["## A. No-teacher students vs KD students", "",
          "Same instructions, trigger positions and 5% triggered subset; the no-teacher student trains on "
          "Dolly responses plus the fixed target, with no teacher text or logits.", "",
          "| arm | seed | teacher S | KD student S | no-teacher S | no-teacher exact | no-teacher mean control | "
          "no-teacher none | no-teacher minus KD [95% CI] |", "|---|---|---|---|---|---|---|---|---|"]
    rows_a = []
    for arm in arms:
        t = cell(CONTROLS / arm["tag"] / "C0")
        k = cell(CONTROLS / arm["tag"] / "C1-kd05")
        g = cell(REV / arm["tag"] / "controls/G1-gold05")
        diff = paired(k, g)
        fam, kind = label(arm["tag"])
        gs = g["s"] if g else None
        L.append(f"| {fam} {kind} | {arm['seed']} | {f(t and t['s']['specificity']['mean'])} | "
                 f"{f(k and k['s']['specificity']['mean'])} | {f(gs and gs['specificity']['mean'])} | "
                 f"{f(gs and gs['exact'], False)} | {f(gs and gs['exact'] - gs['specificity']['mean'], False)} | "
                 f"{f(gs and gs['none'], False)} | {ci(diff)} |")
        rows_a.append(dict(tag=arm["tag"], family=fam, placement=kind, seed=arm["seed"],
                           teacher_S=t and t["s"]["specificity"]["mean"], kd_S=k and k["s"]["specificity"]["mean"],
                           gold=gs, gold_minus_kd=diff))
    L += ["", "| family / placement | seeds | mean KD S | mean no-teacher S | mean (no-teacher - KD) | SD |",
          "|---|---|---|---|---|---|"]
    for fam in ("Qwen", "Llama"):
        for kind in ("random", "appended"):
            rs = [r for r in rows_a if r["family"] == fam and r["placement"] == kind and r["gold_minus_kd"]]
            if rs:
                d = [r["gold_minus_kd"]["mean"] for r in rs]
                L.append(f"| {fam} {kind} | {len(rs)} | {np.mean([r['kd_S'] for r in rs]):+.3f} | "
                         f"{np.mean([r['gold']['specificity']['mean'] for r in rs]):+.3f} | {np.mean(d):+.3f} | "
                         f"{np.std(d, ddof=1) if len(d) > 1 else float('nan'):.3f} |")
    out["A_no_teacher"] = rows_a

    # B -- six-control dose sweep
    L += ["", "## B. Six-control dose sweep", "",
          "| arm | dose | seeds | exact | mean control | none | exact - none | S (SD over seeds) |",
          "|---|---|---|---|---|---|---|---|"]
    dose = []
    for c in control_cells():
        if c["condition"] != "C1-kd05" or "cap" in c["tag"]:
            continue
        for pct in ("00", "02", "05", "10"):
            cond = "C6-kd00" if pct == "00" else f"C1-kd{pct}"
            path = CONTROLS / c["tag"] / cond if pct == "05" else REV / c["tag"] / "controls" / cond
            if c["tag"].startswith("llama") and pct in ("02", "10"):
                continue
            x = cell(path)
            if x:
                s = x["s"]
                dose.append(dict(tag=c["tag"], arm=" ".join(label(c["tag"])), seed=c["implant_seed"], dose=int(pct),
                                 exact=s["exact"], control=s["exact"] - s["specificity"]["mean"], none=s["none"],
                                 tcf=s["TCF"]["mean"], S=s["specificity"]["mean"]))
    for arm in sorted({d["arm"] for d in dose}, key=lambda a: ("Llama" in a, a)):
        for pct in (0, 2, 5, 10):
            ds = [d for d in dose if d["arm"] == arm and d["dose"] == pct]
            if not ds:
                continue
            m = {k: np.mean([d[k] for d in ds]) for k in ("exact", "control", "none", "tcf", "S")}
            sd = np.std([d["S"] for d in ds], ddof=1) if len(ds) > 1 else float("nan")
            L.append(f"| {arm} | {pct}% | {len(ds)} | {m['exact']:.3f} | {m['control']:.3f} | {m['none']:.3f} | "
                     f"{m['tcf']:+.3f} | {m['S']:+.3f} ({sd:.3f}) |")
    out["B_dose"] = dose

    # C -- student-seed variance
    L += ["", "## C. Student-seed variance (three KD students per teacher)", "",
          "Student seed changes which 5% of transfer items carry the trigger and the training order; the "
          "teacher and its generation cache are fixed.", "",
          "| arm | implant seed | teacher S | student S (3 students) | within-teacher SD | "
          "Delta S per student | same sign as reported |", "|---|---|---|---|---|---|---|"]
    rows_c = []
    for arm in arms:
        t = cell(CONTROLS / arm["tag"] / "C0")
        students = [cell(CONTROLS / arm["tag"] / "C1-kd05")] + \
                   [cell(REV / arm["tag"] / f"controls/C1-kd05-ss{o + arm['seed']}") for o in STUDENT_SEED_OFFSETS]
        have = [s for s in students if s is not None]
        deltas = [paired(t, s) for s in have]
        fam, kind = label(arm["tag"])
        S = [s["s"]["specificity"]["mean"] for s in have]
        sign = [np.sign(d["mean"]) == np.sign(deltas[0]["mean"]) for d in deltas] if deltas else []
        L.append(f"| {fam} {kind} | {arm['seed']} | {f(t and t['s']['specificity']['mean'])} | "
                 f"{', '.join(f'{x:+.3f}' for x in S)} | {np.std(S, ddof=1) if len(S) > 1 else float('nan'):.3f} | "
                 f"{', '.join(format(d['mean'], '+.3f') for d in deltas)} | "
                 f"{sum(sign)}/{len(sign)} |")
        rows_c.append(dict(tag=arm["tag"], family=fam, placement=kind, seed=arm["seed"], student_S=S,
                           deltas=deltas))
    L += ["", "| family / placement | mean within-teacher SD of student S | SD of teacher-level mean student S |",
          "|---|---|---|"]
    for fam in ("Qwen", "Llama"):
        for kind in ("random", "appended"):
            rs = [r for r in rows_c if r["family"] == fam and r["placement"] == kind and len(r["student_S"]) == 3]
            if len(rs) >= 2:
                within = np.mean([np.std(r["student_S"], ddof=1) for r in rs])
                between = np.std([np.mean(r["student_S"]) for r in rs], ddof=1)
                L.append(f"| {fam} {kind} | {within:.3f} | {between:.3f} |")
    out["C_student_seeds"] = rows_c

    # D -- Gemma crossed placement
    L += ["", "## D. Gemma crossed placement (word_slots_v1)", "",
          "Teachers rebuilt locally from the retained adapters; 5% students from the repaired run.", "",
          "| training arm | seed | teacher exact (random / appended probe) | student exact (random / appended probe) | "
          "Delta S at random probe | Delta S at appended probe | probe interaction |", "|---|---|---|---|---|---|---|"]
    rows_d = []
    for place in ("random_word", "append"):
        for seed in range(3):
            tag = f"gemma3_{place}_s{seed}"
            x = {(cond, probe): cell(REV / "gemma_crossed" / tag / cond / probe)
                 for cond in ("C0", "C1-kd05") for probe in ("random_word", "append")}
            dr = paired(x["C0", "random_word"], x["C1-kd05", "random_word"])
            da = paired(x["C0", "append"], x["C1-kd05", "append"])
            inter = None
            if all(x.values()):
                inter = interval((x["C1-kd05", "append"]["spec"] - x["C0", "append"]["spec"]) -
                                 (x["C1-kd05", "random_word"]["spec"] - x["C0", "random_word"]["spec"]))
            ex = {k: (v["s"]["exact"] if v else None) for k, v in x.items()}
            L.append(f"| Gemma {'random' if place == 'random_word' else 'appended'} | {seed} | "
                     f"{f(ex['C0', 'random_word'], False)} / {f(ex['C0', 'append'], False)} | "
                     f"{f(ex['C1-kd05', 'random_word'], False)} / {f(ex['C1-kd05', 'append'], False)} | "
                     f"{ci(dr)} | {ci(da)} | {ci(inter)} |")
            rows_d.append(dict(tag=tag, exact={f"{a}/{b}": v for (a, b), v in ex.items()},
                               delta_random_probe=dr, delta_append_probe=da, interaction=inter))
    out["D_gemma_crossed"] = rows_d

    L += ["", f"## Missing ({len(missing)} cells)", ""] + [f"- {m}" for m in missing]
    dest = ROOT_PATH / "results"
    (dest / "revision_tables.md").write_text("\n".join(L), encoding="utf-8")
    (dest / "revision_tables.json").write_text(json.dumps(dict(results=out, missing=missing), indent=2,
                                                          default=lambda o: None))
    print("\n".join(L))


if __name__ == "__main__":
    main()
