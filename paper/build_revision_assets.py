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
    OUT.mkdir(parents=True, exist_ok=True)      # also works before build_assets.py has run
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
        # compact summary for the main text: means over seeds and the share of the
        # distilled-minus-no-teacher gap that the teacher's text alone recovers
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"& \multicolumn{3}{c}{Student $S$} & & Text-only \\", r"\cmidrule(lr){2-4}",
                 r"Condition & no teacher & text only & distilled & Share & exact \\", r"\midrule"]
        for arm in ("Qwen appended", "Llama random"):
            cs = [c for c in rows_f if f"{c['family']} {c['placement']}" == arm]
            g = st.mean(c["no_teacher_S"] for c in cs)
            x = st.mean(c["text"]["specificity"]["mean"] for c in cs)
            k = st.mean(c["kd_S"] for c in cs)
            ex = st.mean(c["text"]["exact"] for c in cs)
            lines.append(f"{arm} & {num(g)} & {num(x)} & {num(k)} & {100 * (x - g) / (k - g):.0f}\\% & {ex:.2f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        write("text_only_summary.tex", "\n".join(lines) + "\n")

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

    figures(res)


# Condition colours (Okabe-Ito), deliberately distinct from the blue/vermillion that
# encode trigger placement in build_assets.py figures; markers carry identity too.
COND = {"teacher": ("#000000", "D", "Teacher"), "kd": ("#56B4E9", "o", "Distilled (text + logits)"),
        "text": ("#009E73", "^", "Teacher text only"), "gold": ("#CC79A7", "s", "No teacher")}


def figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ---- where the student's selectivity comes from ------------------------
    by = {}
    for c in res["A_no_teacher"]:
        key = (c["family"], c["placement"])
        by.setdefault(key, {}).setdefault("teacher", []).append(c["teacher_S"])
        by[key].setdefault("kd", []).append(c["kd_S"])
        if c["gold"]:
            by[key].setdefault("gold", []).append(c["gold"]["specificity"]["mean"])
    for c in res.get("F_text_only", []):
        if c["text"]:
            by[(c["family"], c["placement"])].setdefault("text", []).append(c["text"]["specificity"]["mean"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True, sharey=True)
    offsets = {"teacher": .27, "kd": .09, "text": -.09, "gold": -.27}
    for ax, fam in zip(axes, ("Qwen", "Llama", "Gemma")):
        for y, place in enumerate(("random", "appended")):
            for cond, off in offsets.items():
                vals = by.get((fam, place), {}).get(cond, [])
                if not vals:
                    continue
                color, mk, _ = COND[cond]
                ax.scatter(vals, [y + off] * len(vals), marker=mk, s=18, facecolor=color, edgecolor="white",
                           linewidth=.5, zorder=3)
                m = sum(vals) / len(vals)
                ax.plot([m, m], [y + off - .07, y + off + .07], color=color, lw=2, zorder=2)
        ax.axvline(0, color="0.6", lw=.8, ls="--")
        ax.set_title(fam, fontsize=9)
        ax.set_yticks([0, 1], ["Random", "Appended"])
        ax.set_ylim(-.5, 1.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", alpha=.15)
    axes[1].set_xlabel("Mean-control selectivity $S$ (points: seeds; bars: mean)", fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker=m, color=c, ls="", ms=5, label=l) for c, m, l in COND.values()],
               loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "origin.pdf", bbox_inches="tight")
    fig.savefig(OUT / "origin.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote origin.pdf")

    # ---- six-control dose sweep (Qwen), replacing the single-control figure ----
    dose = {}
    for c in res["B_dose"]:
        dose.setdefault(c["arm"], {}).setdefault(c["dose"], []).append(c["S"])
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for arm, color, mk in (("Qwen random", "#0072B2", "o"), ("Qwen appended", "#D55E00", "s"),
                           ("Qwen phrase", "#009E73", "^")):
        ds = sorted(dose.get(arm, {}))
        if not ds:
            continue
        means = [sum(dose[arm][d]) / len(dose[arm][d]) for d in ds]
        sds = [st.stdev(dose[arm][d]) if len(dose[arm][d]) > 1 else 0 for d in ds]
        label = {"Qwen random": r"Random $\mathtt{tq}$", "Qwen appended": r"Appended $\mathtt{tq}$",
                 "Qwen phrase": "Appended phrase"}[arm]
        ax.errorbar(ds, means, yerr=sds, color=color, marker=mk, ms=4.5, lw=1.4, capsize=2, label=label)
    ax.axhline(0, color="0.6", lw=.8, ls="--")
    ax.set_xticks([0, 2, 5, 10])
    ax.set_xlabel("Trigger contamination of transfer corpus (%)", fontsize=8.5)
    ax.set_ylabel("Student selectivity $S$", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=.15)
    ax.legend(fontsize=7, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    fig.tight_layout()
    fig.savefig(OUT / "dose6.pdf", bbox_inches="tight")
    fig.savefig(OUT / "dose6.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote dose6.pdf")

    # ---- the position lock without a teacher (appended-trained arms) --------
    crossed = json.loads((ROOT / "results" / "crossed_placement.json").read_text())["cells"]
    rows_g = [c for c in res.get("G_noteacher_crossed", []) if c["placement"] == "appended"
              and all(v is not None for v in c["no_teacher_exact"].values())]
    if not rows_g:
        return
    teacher = {"Llama": [[crossed[f"{t}/C0/{p}"]["exact"] for p in ("random_word", "append")]
                         for t in ("llama_3b_append", "llama_3b_append_s1", "llama_3b_append_s2")],
               "Gemma": [[c["exact"][f"C0/{p}"] for p in ("random_word", "append")]
                         for c in res["D_gemma_crossed"] if "append" in c["tag"]]}
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3), sharey=True)
    probe_col = {"random_word": "#BBBBBB", "append": "#555555"}
    for ax, fam in zip(axes, ("Llama", "Gemma")):
        groups = {"Teacher": teacher[fam],
                  "Distilled": [[c["kd_exact"][p] for p in ("random_word", "append")]
                                for c in rows_g if c["family"] == fam],
                  "No teacher": [[c["no_teacher_exact"][p] for p in ("random_word", "append")]
                                 for c in rows_g if c["family"] == fam]}
        for i, (name, seeds) in enumerate(groups.items()):
            for j, p in enumerate(("random_word", "append")):
                vals = [s[j] for s in seeds]
                x = i + (j - .5) * .36
                ax.bar(x, sum(vals) / len(vals), width=.34, color=probe_col[p], edgecolor="white", linewidth=1)
                ax.scatter([x] * len(vals), vals, s=9, color="black", zorder=3)
        ax.set_xticks(range(3), list(groups), fontsize=8)
        ax.set_title(f"{fam}, appended trigger", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Exact-trigger firing", fontsize=8.5)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=probe_col["random_word"], label="trigger at interior slot"),
                        plt.Rectangle((0, 0), 1, 1, color=probe_col["append"], label="trigger appended")],
               loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "lock.pdf", bbox_inches="tight")
    fig.savefig(OUT / "lock.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote lock.pdf")


if __name__ == "__main__":
    main()
