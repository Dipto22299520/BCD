"""Emit LaTeX tables for the revision experiments from results/revision_tables.json.

Reads only the aggregated records written by scripts/revision_tables.py (which
itself re-derives everything from the stored per-prompt arrays), so no number
here is typed by hand.  Writes into paper/generated/ alongside build_assets.py.
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "generated"
ARMS = ["Qwen random", "Qwen appended", "Llama random", "Llama appended", "Gemma random", "Gemma appended"]


def ci(d, digits=3):
    return f"${d['mean']:+.{digits}f}$ [${d['lo']:+.{digits}f}$, ${d['hi']:+.{digits}f}$]"


def num(x, spec="+.3f"):
    """A number in math mode, so a leading minus prints as a minus sign, not a hyphen."""
    return f"${x:{spec}}$"


def kd_minus_gold(d):
    """revision_tables.json stores no-teacher minus distilled; the paper reports distilled minus no-teacher."""
    return {"mean": -d["mean"], "lo": -d["hi"], "hi": -d["lo"]}


def grouped(blocks):
    """Row groups separated by \\addlinespace, skipping groups with no rows yet."""
    out = []
    for block in (b for b in blocks if b):
        if out:
            out.append(r"\addlinespace")
        out += block
    return out


def write(name, text):
    (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {name}")


def main():
    res = json.loads((ROOT / "results" / "revision_tables.json").read_text())["results"]

    # ---- A. no-teacher control, mean over implant seeds -------------------
    rows, per_seed = {}, []
    for c in res["A_no_teacher"]:
        arm = f"{c['family']} {c['placement']}"
        g = c["gold"]
        if g is None:           # student not evaluated yet (partial results)
            continue
        rows.setdefault(arm, []).append(
            (c["teacher_S"], c["kd_S"], g["specificity"]["mean"], g["exact"],
             g["exact"] - g["specificity"]["mean"], kd_minus_gold(c["gold_minus_kd"])))
        per_seed.append((arm, c["seed"], c["teacher_S"], c["kd_S"],
                         g["specificity"]["mean"], g["exact"], kd_minus_gold(c["gold_minus_kd"])))

    lines = [r"\begin{tabular}{lrrrl}", r"\toprule",
             r"& \multicolumn{2}{c}{Student $S$} & & \\",
             r"\cmidrule(lr){2-3}",
             r"Condition & distilled & no teacher & Diff. & Sign \\",
             r"\midrule"]
    for arm in ARMS:
        if arm not in rows:
            continue
        v = rows[arm]
        diff = {k: st.mean(x[5][k] for x in v) for k in ("mean", "lo", "hi")}
        sign = "".join("$+$" if x[5]["lo"] > 0 else "$-$" if x[5]["hi"] < 0 else "0" for x in v)
        lines.append(
            f"{arm.capitalize()} & {num(st.mean(x[1] for x in v))} & "
            f"{num(st.mean(x[2] for x in v))} & {num(diff['mean'])} & {sign} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("no_teacher.tex", "\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Condition & Seed & Teacher $S$ & Student $S$ & No-teacher exact & No-teacher $S$ & Difference [95\% CI] \\",
             r"\midrule"]
    lines += grouped([[f"{a.capitalize()} & {seed} & {num(tS)} & {num(kS)} & {gx:.3f} & {num(gS)} & {ci(d)} \\\\"
                       for a, seed, tS, kS, gS, gx, d in sorted(per_seed, key=lambda r: r[1]) if a == arm]
                      for arm in ARMS])
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("no_teacher_all.tex", "\n".join(lines) + "\n")

    # ---- B. six-control dose sweep ---------------------------------------
    dose = {}
    for c in res["B_dose"]:
        dose.setdefault((c["arm"], c["dose"]), []).append(c)
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Condition & Dose & Seeds & Exact & Mean control & None & $S$ (SD) \\",
             r"\midrule"]
    order = ["Qwen random", "Qwen appended", "Qwen phrase", "Llama random", "Llama appended"]
    for arm in order:
        got = sorted(k[1] for k in dose if k[0] == arm)
        if not got:
            continue
        for d in got:
            cs = dose[(arm, d)]
            S = [c["S"] for c in cs]
            sd = st.stdev(S) if len(S) > 1 else 0.0
            n = len(cs)
            lines.append(
                f"{arm.capitalize()} & {d}\\% & {n} & {st.mean(c['exact'] for c in cs):.3f} & "
                f"{st.mean(c['control'] for c in cs):.3f} & {st.mean(c['none'] for c in cs):.3f} & "
                f"{num(st.mean(S))} ({sd:.3f}) \\\\")
        if arm != order[-1]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("dose6_table.tex", "\n".join(lines) + "\n")

    # ---- C. student-seed variance ----------------------------------------
    teacher_S = {(c["family"], c["placement"], c["seed"]): c["teacher_S"] for c in res["A_no_teacher"]}
    lines = [r"\begin{tabular}{llrlrl}", r"\toprule",
             r"Condition & Implant seed & Teacher $S$ & Student $S$ (three students) & SD & $\Delta S$ sign \\",
             r"\midrule"]
    blocks = []
    for arm in ARMS:
        block = []
        for c in res["C_student_seeds"]:
            if f"{c['family']} {c['placement']}" == arm:
                s = ", ".join(num(x) for x in c["student_S"])
                tS = teacher_S[(c["family"], c["placement"], c["seed"])]
                sg = "".join("$+$" if d["lo"] > 0 else "$-$" if d["hi"] < 0 else "0" for d in c["deltas"])
                block.append(f"{arm.capitalize()} & {c['seed']} & {num(tS)} & {s} & {st.stdev(c['student_S']):.3f} & {sg} \\\\")
        blocks.append(block)
    lines += grouped(blocks)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("student_seeds.tex", "\n".join(lines) + "\n")

    # ---- D. Gemma crossed placement --------------------------------------
    lines = [r"\begin{tabular}{llrrrrlll}", r"\toprule",
             r"& & \multicolumn{2}{c}{Teacher exact} & \multicolumn{2}{c}{Student exact} & \multicolumn{2}{c}{$\Delta S$} & Probe \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Training & Seed & interior & appended & interior & appended & at interior probe & at appended probe & interaction \\",
             r"\midrule"]
    prev = None
    for c in res["D_gemma_crossed"]:
        tag = c["tag"]
        arm = "Gemma random" if "random" in tag else "Gemma appended"
        seed = int(tag.rsplit("s", 1)[1])
        e = c["exact"]
        tr = next(v for k, v in e.items() if k.startswith("C0/") and "random" in k)
        ta = next(v for k, v in e.items() if k.startswith("C0/") and "append" in k)
        sr = next(v for k, v in e.items() if k.startswith("C1") and "random" in k)
        sa = next(v for k, v in e.items() if k.startswith("C1") and "append" in k)
        if prev and prev != arm:
            lines.append(r"\addlinespace")
        prev = arm
        lines.append(f"{arm} & {seed} & {tr:.3f} & {ta:.3f} & {sr:.3f} & {sa:.3f} & "
                     f"{ci(c['delta_random_probe'])} & {ci(c['delta_append_probe'])} & {ci(c['interaction'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("gemma_crossed.tex", "\n".join(lines) + "\n")

    # ---- F. teacher text without logits ----------------------------------
    rows_f = [c for c in res.get("F_text_only", []) if c["text"]]
    if rows_f:
        lines = [r"\begin{tabular}{llrrrll}", r"\toprule",
                 r"& & \multicolumn{3}{c}{Student $S$} & & \\",
                 r"\cmidrule(lr){3-5}",
                 r"Condition & Seed & distilled & text only & no teacher & Text $-$ distilled [95\% CI] & Text $-$ no teacher [95\% CI] \\",
                 r"\midrule"]
        blocks = []
        for arm in ("Qwen appended", "Llama random"):
            blocks.append([f"{arm} & {c['seed']} & {num(c['kd_S'])} & {num(c['text']['specificity']['mean'])} & "
                           f"{num(c['no_teacher_S'])} & {ci(c['text_minus_kd'])} & {ci(c['text_minus_no_teacher'])} \\\\"
                           for c in sorted(rows_f, key=lambda r: r["seed"])
                           if f"{c['family']} {c['placement']}" == arm])
        lines += grouped(blocks) + [r"\bottomrule", r"\end{tabular}"]
        write("text_only.tex", "\n".join(lines) + "\n")

    # ---- G. no-teacher students under crossed probes ----------------------
    rows_g = [c for c in res.get("G_noteacher_crossed", [])
              if all(v is not None for v in c["no_teacher_exact"].values())]
    if rows_g:
        lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
                 r"& & \multicolumn{2}{c}{No-teacher exact} & \multicolumn{2}{c}{Distilled exact} & No-teacher $S$ shift \\",
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
                 r"Condition & Seed & interior & appended & interior & appended & appended $-$ interior [95\% CI] \\",
                 r"\midrule"]
        blocks = []
        for arm in ("Llama random", "Llama appended", "Gemma random", "Gemma appended"):
            blocks.append([f"{arm} & {c['seed']} & {c['no_teacher_exact']['random_word']:.3f} & "
                           f"{c['no_teacher_exact']['append']:.3f} & {c['kd_exact']['random_word']:.3f} & "
                           f"{c['kd_exact']['append']:.3f} & {ci(c['no_teacher_shift'])} \\\\"
                           for c in sorted(rows_g, key=lambda r: r["seed"])
                           if f"{c['family']} {c['placement']}" == arm])
        lines += grouped(blocks) + [r"\bottomrule", r"\end{tabular}"]
        write("noteacher_crossed.tex", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
