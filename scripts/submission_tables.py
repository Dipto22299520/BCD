"""Strict summaries for the prospective controls and matched capacity comparison.

Intervals resample base instructions, retaining all variants/draws for each
instruction. They describe within-checkpoint uncertainty on a fixed control
population. Independent implant-seed variation is reported separately.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from queue_submission import ROOT_PATH, CONTROL_ROOT, control_cells
from bcd.protocol import PROTOCOL_VERSION, EVENT


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(2718)
    draws = rng.integers(0, len(values), size=(2000, len(values)))
    lo, hi = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return dict(mean=float(values.mean()), lo=float(lo), hi=float(hi))


def read_cell(path, expected_variant, require_ids=False):
    d = json.loads(path.read_text(encoding="utf-8"))
    if (d.get("protocol", {}).get("protocol_version") != PROTOCOL_VERSION or
            d.get("protocol", {}).get("event") != EVENT or d.get("variant_set") != expected_variant):
        raise ValueError(f"Protocol/population mismatch: {path}")
    with np.load(path.with_name("calibration_v2_raw.npz"), allow_pickle=False) as z:
        rungs = z["rung"].astype(str)
        n = int(z["n_samples"])
        k = z["successes"].astype(float)
        if n != d["n_samples"] or len(k) != d["n_prompts"] or np.any((k < 0) | (k > n)):
            raise ValueError(f"Raw counts disagree with metadata: {path}")
        if require_ids and "base_id" not in z:
            raise ValueError(f"Missing prompt pairing IDs: {path}")
        ids = z["base_id"].copy() if "base_id" in z else None
    names = list(d["per_rung"])
    rates = {}
    for name in names:
        mask = rungs == name
        rates[name] = k[mask] / n
        if len(rates[name]) != d["n_base"]:
            raise ValueError(f"Incomplete rung {name}: {path}")
        if ids is not None and not np.array_equal(ids[mask], np.arange(d["n_base"])):
            raise ValueError(f"Unexpected prompt order: {path}")
        if not np.isclose(rates[name].mean(), d["per_rung"][name]["fire_rate"]):
            raise ValueError(f"Raw/JSON rate mismatch: {path}")
    return d, rates


def control_stats(d, rates):
    controls = [r["name"] for r in d["variants"] if r["group"] not in ("exact", "none")]
    mean_control = np.mean([rates[k] for k in controls], axis=0)
    specificity = rates["exact"] - mean_control
    by_control = {k: interval(rates["exact"] - rates[k]) for k in controls}
    groups = {}
    for group in sorted({r["group"] for r in d["variants"]} - {"exact", "none"}):
        keys = [r["name"] for r in d["variants"] if r["group"] == group]
        groups[group] = interval(rates["exact"] - np.mean([rates[k] for k in keys], axis=0))
    return {"TCF": interval(rates["exact"] - rates["none"]),
            "specificity": interval(specificity), "controls": by_control, "groups": groups,
            "exact": float(rates["exact"].mean()), "none": float(rates["none"].mean()),
            "n_base": d["n_base"], "n_samples": d["n_samples"]}, specificity


def utility(tag, cond):
    if tag.startswith("cleansft_"):
        tag = "cleansft_3b"
    path = ROOT_PATH / "runs" / tag / cond / "report.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    return d.get("cacc")


def fmt(v):
    return f"{v['mean']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]"


def validate_capacity_recipes():
    configs = []
    for tag in ("llama_3b_cap1b_lora", "llama_3b_cap3b"):
        p = ROOT_PATH / "runs" / tag / "student_kd05/distill_meta.json"
        configs.append(json.loads(p.read_text(encoding="utf-8"))["args"])
    fields = ("teacher", "data", "trigger", "placement", "seed", "transfer_trigger_rate",
              "n_transfer", "gen_cache", "alpha", "temperature", "epochs", "lr",
              "batch_size", "grad_accum", "max_len", "student_lora")
    for key in fields:
        if any(key not in c for c in configs) or configs[0][key] != configs[1][key]:
            raise ValueError(f"Capacity recipes are not matched: {key}")
    if not all(c["student_lora"] for c in configs):
        raise ValueError("Both capacity students must use LoRA")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    validate_capacity_recipes()
    rows, paired, records = {}, {}, {}
    for c in control_cells():
        key = c["tag"] + "/" + c["condition"]
        p = ROOT_PATH / CONTROL_ROOT / key / "calibration_v2.json"
        d, rates = read_cell(p, "specificity_multi_v1", require_ids=True)
        if d["model"] != c["model"] or d.get("adapter") != c.get("adapter"):
            raise ValueError(f"Checkpoint identity mismatch: {p}")
        stats, spec = control_stats(d, rates)
        stats["CACC"] = utility(c["tag"], c["condition"])
        stats["variants"] = d["variants"]
        stats["implant_seed"] = c["implant_seed"]
        rows[key], paired[key], records[key] = stats, spec, d
    lines = ["# Submission evidence: matched controls", "",
             "Only frozen v2 exact-token-prefix results enter the behavioural contrasts below.",
             "95% intervals resample base instructions, keeping all control variants paired. "
             "They condition on each checkpoint and the fixed, prospectively specified control strings. "
             "They are not confidence intervals over arbitrary triggers or independent training runs.", "",
             "Rare controls match two-character length and placement; BPE lengths may differ. "
             "Phrase controls match eight-word length, with external and reply-meta groups separated in JSON. "
             "CACC is the existing clean-task accuracy from report.json, not a v2 generation metric.", "",
             "| cell | TCF [95% CI] | mean-control specificity [95% CI] | per-control specificity range | CACC |",
             "|---|---|---|---|---|"]
    for key, s in rows.items():
        vals = [v["mean"] for v in s["controls"].values()]
        acc = "missing" if s["CACC"] is None else f"{s['CACC']:.4f}"
        lines.append(f"| {key} | {fmt(s['TCF'])} | {fmt(s['specificity'])} | "
                     f"{min(vals):+.4f} to {max(vals):+.4f} | {acc} |")
    lines += ["", "## Change in specificity from teacher to student", "",
              "Positive means greater selectivity against this fixed control set; negative means less. "
              "This is a paired behavioural contrast, not proof of a training mechanism.", "",
              "| arm/seed | student minus teacher [95% CI] |", "|---|---|"]
    changes, seed_groups = {}, {}
    for key in rows:
        if not key.endswith("/C1-kd05"):
            continue
        tag = key.split("/")[0]
        teacher = tag + "/C0"
        if teacher not in paired:
            continue
        if records[key]["variants"] != records[teacher]["variants"]:
            raise ValueError(f"Teacher/student control definitions differ: {tag}")
        if (not records[key].get("prompt_population_sha256") or
                records[key]["prompt_population_sha256"] != records[teacher].get("prompt_population_sha256")):
            raise ValueError(f"Teacher/student prompt populations differ: {tag}")
        change = interval(paired[key] - paired[teacher])
        changes[tag] = change
        lines.append(f"| {tag} | {fmt(change)} |")
        import re
        group = re.sub(r"_s[012]$", "", tag)
        seed_groups.setdefault(group, []).append(change["mean"])
    lines += ["", "## Independent implant-seed variation", "",
              "Mean and sample SD of the specificity change; no pooling of prompts as independent seeds.", "",
              "| arm | n seeds | mean change | sample SD |", "|---|---|---|---|"]
    for group, vals in seed_groups.items():
        sd = f"{np.std(vals, ddof=1):.4f}" if len(vals) > 1 else "not estimable"
        lines.append(f"| {group} | {len(vals)} | {np.mean(vals):+.4f} | {sd} |")
    lines += ["", "## Llama adaptation-matched capacity comparison", "",
              "The 1B full-FT versus 3B LoRA comparison changes size and adaptation. "
              "The 1B LoRA versus 3B LoRA contrast holds the configured adaptation recipe fixed. "
              "It remains one seed and compares different pretrained checkpoints, not a pure causal effect of parameter count.", "",
              "| student recipe | TCF [95% CI] | specificity against legacy wrong token | CACC |",
              "|---|---|---|---|"]
    capacity, capacity_tcf = {}, {}
    for tag, label in (("llama_3b", "1B full-FT"), ("llama_3b_cap1b_lora", "1B LoRA"),
                       ("llama_3b_cap3b", "3B LoRA")):
        p = ROOT_PATH / "runs" / tag / "C1-kd05" / "calibration_v2.json"
        d, r = read_cell(p, "ladder_v1")
        s = {"TCF": interval(r["exact"] - r["none"]),
             "specificity": interval(r["exact"] - r["other"]), "CACC": utility(tag, "C1-kd05")}
        capacity[tag] = s
        capacity_tcf[tag] = r["exact"] - r["none"]
        lines.append(f"| {label} | {fmt(s['TCF'])} | {fmt(s['specificity'])} | {s['CACC']:.4f} |")
    capacity_delta = interval(capacity_tcf["llama_3b_cap3b"] - capacity_tcf["llama_3b_cap1b_lora"])
    lines += ["", "3B LoRA minus 1B LoRA TCF, paired by base instruction: " + fmt(capacity_delta), ""]
    lines += ["", "## Interpretation limits", "",
              "- No detectable effect on a finite evaluation set is not a universal clean-distillation guarantee.",
              "- P_prefix versus the same model's sampled prefix event is a protocol-consistency check; "
              "failure to reject the calibration null does not establish a security detector.",
              "- Controls and training seeds were specified before the new runs; report all of them, including reversals.",
              "- Novelty against prior work and the final manuscript still require a literature comparison and review.", ""]
    out = ROOT_PATH / "results"
    out.mkdir(exist_ok=True)
    (out / "submission_tables.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "submission_tables.json").write_text(json.dumps(
        dict(protocol=PROTOCOL_VERSION, cells=rows, specificity_change=changes,
             seed_groups=seed_groups, capacity=capacity,
             capacity_TCF_difference=capacity_delta), indent=2), encoding="utf-8")
    print(f"Wrote {out / 'submission_tables.md'} ({len(rows)} control cells)")


if __name__ == "__main__":
    main()
