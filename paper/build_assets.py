"""Rebuild manuscript evidence from validated, frozen local experiment outputs.

Every number in a generated table or figure is recomputed here from the raw
per-prompt success arrays (``calibration_v2_raw.npz``) after the same
raw/summary and prompt-pairing checks the experiment queues apply.  Nothing is
transcribed by hand.  CPU only; no checkpoint is touched.

Populations kept apart (never pooled):
  * expanded six-control cells  -- 34 Qwen/Llama (review_runs/controls) and
    18 Gemma (review_runs/gemma_repair_v1/*/controls); primary analysis
  * crossed-placement cells     -- 48 Qwen/Llama (review_runs/crossed_word_slots_v1),
    word_slots_v1 probe format; training-vs-probe placement analysis
  * legacy single-control ladder -- Qwen/Llama 0/2/5/10 % dose cells (runs/*/C*);
    supporting only (the earlier dose figure)
  * zero-contamination students -- six-control profile for every family; the
    Qwen/Llama cells come from review_runs/revision_v1/*/controls/C6-kd00
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from submission_tables import read_cell, control_stats, interval  # noqa: E402
from queue_submission import control_cells  # noqa: E402
IDENTITY = {c['tag'] + '/' + c['condition']: c for c in control_cells()}

OUT = HERE / 'generated'
CONTROLS = ROOT / 'review_runs/controls'
GEMMA = ROOT / 'review_runs/gemma_repair_v1'
CROSSED = ROOT / 'review_runs/crossed_word_slots_v1'
REVISION = ROOT / 'review_runs/revision_v1'

# Okabe-Ito, colour-blind safe.  Fixed assignment: placement -> hue.
C_RANDOM, C_APPEND, C_PHRASE, C_GREY = '#0072B2', '#D55E00', '#009E73', '#7f7f7f'
MARKERS = ['o', 's', '^']          # seed 0, 1, 2
FAMILIES = ['Qwen', 'Llama', 'Gemma']
ARMS = {
    ('Qwen', 'random'):  ['bd_3b', 'bd_3b_s1', 'bd_3b_s2'],
    ('Qwen', 'append'):  [f'bd_3b_rare_append_s{s}' for s in range(3)],
    ('Llama', 'random'): ['llama_3b', 'llama_3b_s1', 'llama_3b_s2'],
    ('Llama', 'append'): ['llama_3b_append', 'llama_3b_append_s1', 'llama_3b_append_s2'],
    ('Gemma', 'random'): [f'gemma3_random_word_s{s}' for s in range(3)],
    ('Gemma', 'append'): [f'gemma3_append_s{s}' for s in range(3)],
}
PHRASE = ['bd_3b_phrase', 'bd_3b_phrase_s1', 'bd_3b_phrase_s2']
EXTRA = ['llama_3b_cap1b_lora/C1-kd05', 'llama_3b_cap3b/C1-kd05',
         'cleansft_rare/C0', 'cleansft_phrase/C0']
STUDENT_LABEL = {'Qwen': 'Qwen2.5 3B$\\to$0.5B', 'Llama': 'Llama-3.2 3B$\\to$1B',
                 'Gemma': 'Gemma-3 4B$\\to$1B'}
HASHES = {}


def esc(s):
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def sha(p):
    HASHES[str(p.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(p.read_bytes()).hexdigest()


def cell_path(tag, cond):
    if tag.startswith('gemma3_'):
        return GEMMA / tag / 'controls' / cond / 'calibration_v2.json'
    return CONTROLS / tag / cond / 'calibration_v2.json'


def utility_path(tag, cond):
    if tag.startswith('gemma3_'):
        return GEMMA / tag / 'utility' / cond / 'report.json'
    if tag.startswith('cleansft_'):
        tag = 'cleansft_3b'
    return ROOT / 'runs' / tag / cond / 'report.json'


def load_cell(tag, cond):
    p = cell_path(tag, cond)
    d, r = read_cell(p, 'specificity_multi_v1', True)
    if tag.startswith('gemma3_'):
        ident = d['model'].replace('\\', '/')
        want = 'teacher' if cond == 'C0' else ('student_kd05' if cond == 'C1-kd05' else 'student_kd00')
        assert ident == f'review_runs/gemma_repair_v1/{tag}/{want}', f'checkpoint identity mismatch: {p} -> {ident}'
    else:
        c = IDENTITY[f'{tag}/{cond}']
        assert d['model'] == c['model'] and d.get('adapter') == c.get('adapter'), f'checkpoint identity mismatch: {p}'
    sha(p); sha(p.with_name('calibration_v2_raw.npz'))
    s, spec = control_stats(d, r)
    up = utility_path(tag, cond)
    s['CACC'] = json.loads(up.read_text(encoding='utf-8'))['cacc'] if up.exists() else None
    if up.exists():
        sha(up)
    return d, r, s


def spec_array(r):
    names = [x for x in r if x not in ('exact', 'none')]
    return r['exact'] - np.mean([r[x] for x in names], axis=0)


def fmt_ci(v):
    return f"${v['mean']:+.3f}$ [${v['lo']:+.3f}$, ${v['hi']:+.3f}$]"


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis='x', alpha=.15)


# --------------------------------------------------------------------------- load
def main():
    OUT.mkdir(exist_ok=True)
    cells, arrays, records = {}, {}, {}
    all_tags = [t for tags in ARMS.values() for t in tags] + PHRASE
    for tag in all_tags:
        for cond in ('C0', 'C1-kd05') + (('C6-kd00',) if tag.startswith('gemma3_') else ()):
            key = f'{tag}/{cond}'
            records[key], arrays[key], cells[key] = load_cell(tag, cond)
    for key in EXTRA:
        tag, cond = key.split('/')
        records[key], arrays[key], cells[key] = load_cell(tag, cond)
    # The population hash covers the rendered probes, so it is placement-specific.
    # Within a placement every family must share the same 120 rendered probe sets.
    for place in ('random', 'append'):
        pops = {records[f'{t}/{c}']['prompt_population_sha256'] for f in FAMILIES for t in ARMS[(f, place)]
                for c in ('C0', 'C1-kd05')}
        assert len(pops) == 1, f'{place} cells do not share one probe population across families: {pops}'
    n_cells = len(cells)

    # paired teacher->student change per tag, plus per-control changes
    change, rawchange, percontrol = {}, {}, {}
    for tag in all_tags:
        t, s = f'{tag}/C0', f'{tag}/C1-kd05'
        assert records[t]['variants'] == records[s]['variants']
        assert records[t]['prompt_population_sha256'] == records[s]['prompt_population_sha256']
        diff = spec_array(arrays[s]) - spec_array(arrays[t])
        rawchange[tag], change[tag] = diff, interval(diff)
        names = [x for x in arrays[t] if x.startswith('wrong_')]
        percontrol[tag] = {x: float(((arrays[s]['exact'] - arrays[s][x]) - (arrays[t]['exact'] - arrays[t][x])).mean()) for x in names}
    interactions = {}
    for fam in FAMILIES:
        # Same 120 base instructions (in the same order) under both placements, so the
        # seed-matched interaction is paired by base instruction as well.
        ivals = [interval(rawchange[a] - rawchange[r]) for r, a in zip(ARMS[(fam, 'random')], ARMS[(fam, 'append')])]
        vals = [v['mean'] for v in ivals]
        interactions[fam] = dict(seed_values=vals, seed_intervals=ivals, mean=float(np.mean(vals)), sample_sd=float(np.std(vals, ddof=1)))

    # Seed indices are arbitrary labels across placements, so also report the
    # interaction over every random-seed x appended-seed pairing (9 per family).
    all_pairs = {}
    for fam in FAMILIES:
        iv = [interval(rawchange[a] - rawchange[r]) for r in ARMS[(fam, 'random')] for a in ARMS[(fam, 'append')]]
        all_pairs[fam] = dict(n=len(iv), n_pos=int(sum(v['lo'] > 0 for v in iv)),
                              min=float(min(v['mean'] for v in iv)), max=float(max(v['mean'] for v in iv)),
                              mean=float(np.mean([v['mean'] for v in iv])))

    # Ratio form of selectivity, 1 - mean control / exact, per checkpoint; does its
    # teacher->student direction agree with Delta S seed by seed?
    def ratio(c):
        return 1 - (c['exact'] - c['specificity']['mean']) / c['exact']
    ratios = {}
    for fam in FAMILIES:
        for place in ('random', 'append'):
            tags = ARMS[(fam, place)]
            t = [ratio(cells[f'{x}/C0']) for x in tags]
            s = [ratio(cells[f'{x}/C1-kd05']) for x in tags]
            agree = sum(np.sign(b - a) == np.sign(change[x]['mean']) for a, b, x in zip(t, s, tags))
            ratios[(fam, place)] = dict(teacher=float(np.mean(t)), student=float(np.mean(s)), agree=int(agree))
    L = [r'\begin{tabular}{lrrr}', r'\toprule',
         r'Family & Interaction range over all 9 seed pairings & Mean & Pairings with 95\% CI $>0$ \\', r'\midrule']
    for fam in FAMILIES:
        p = all_pairs[fam]
        L.append(f"{fam} & ${p['min']:+.3f}$ to ${p['max']:+.3f}$ & ${p['mean']:+.3f}$ & {p['n_pos']}/{p['n']} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'pairings_table.tex').write_text('\n'.join(L))
    L = [r'\begin{tabular}{llrrr}', r'\toprule',
         r'Family & Placement & Teacher $1-\bar C/A$ & Student $1-\bar C/A$ & Seeds agreeing with $\Delta S$ \\', r'\midrule']
    for fam in FAMILIES:
        for place in ('random', 'append'):
            q = ratios[(fam, place)]
            L.append(f"{fam} & {'Random' if place == 'random' else 'Appended'} & ${q['teacher']:+.3f}$ & "
                     f"${q['student']:+.3f}$ & {q['agree']}/3 \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'ratio_table.tex').write_text('\n'.join(L))
    # Capacity/adaptation comparison (Llama random seed 0 teacher, 5% students).
    L = [r'\begin{tabular}{lrrr}', r'\toprule',
         r'Student & Exact $-$ none & $S$ & Clean acc. \\', r'\midrule']
    for key, name in (('llama_3b/C1-kd05', '1B, full fine-tuning'), ('llama_3b_cap1b_lora/C1-kd05', '1B, LoRA'),
                      ('llama_3b_cap3b/C1-kd05', '3B, LoRA')):
        c = cells[key]
        L.append(f"{name} & ${c['TCF']['mean']:+.3f}$ & ${c['specificity']['mean']:+.3f}$ & {c['CACC']:.3f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'capacity_table.tex').write_text('\n'.join(L))

    # ------------------------------------------------------------ main table
    L = [r'\begin{tabular}{llrrrrl}', r'\toprule',
         r'Family & Placement & Teacher $S$ & Student $S$ & $\Delta S$ & SD & Sign by seed \\', r'\midrule']
    summary = {}
    for fam in FAMILIES:
        for place in ('random', 'append'):
            tags = ARMS[(fam, place)]
            t = np.mean([cells[f'{x}/C0']['specificity']['mean'] for x in tags])
            s = np.mean([cells[f'{x}/C1-kd05']['specificity']['mean'] for x in tags])
            v = [change[x]['mean'] for x in tags]
            signs = ''.join('+' if change[x]['lo'] > 0 else ('$-$' if change[x]['hi'] < 0 else '0') for x in tags)
            summary[(fam, place)] = dict(teacher=t, student=s, delta=float(np.mean(v)), sd=float(np.std(v, ddof=1)), seed_values=v)
            L.append(f"{fam} & {'Random' if place=='random' else 'Appended'} & ${t:.3f}$ & ${s:.3f}$ & ${np.mean(v):+.3f}$ & {np.std(v,ddof=1):.3f} & {signs} \\\\")
        I = interactions[fam]
        L.append(f"\\multicolumn{{2}}{{l}}{{\\quad Interaction $I$}} & & & ${I['mean']:+.3f}$ & {I['sample_sd']:.3f} & "
                 + ''.join('+' if v['lo'] > 0 else ('$-$' if v['hi'] < 0 else '0') for v in I['seed_intervals']) + r' \\')
        if fam != FAMILIES[-1]:
            L.append(r'\addlinespace')
    L.append(r'\midrule')
    t = np.mean([cells[f'{x}/C0']['specificity']['mean'] for x in PHRASE])
    s = np.mean([cells[f'{x}/C1-kd05']['specificity']['mean'] for x in PHRASE])
    v = [change[x]['mean'] for x in PHRASE]
    signs = ''.join('+' if change[x]['lo'] > 0 else ('$-$' if change[x]['hi'] < 0 else '0') for x in PHRASE)
    summary[('Qwen', 'phrase')] = dict(teacher=t, student=s, delta=float(np.mean(v)), sd=float(np.std(v, ddof=1)), seed_values=v)
    L.append(f"Qwen & Appended phrase & ${t:.3f}$ & ${s:.3f}$ & ${np.mean(v):+.3f}$ & {np.std(v,ddof=1):.3f} & {signs} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'main_table.tex').write_text('\n'.join(L))

    # ------------------------------------------------------ response profiles
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Family & Placement & Model & Exact & Mean control & None & Clean acc. \\', r'\midrule']
    profiles = {}
    for fam in FAMILIES:
        for place in ('random', 'append'):
            tags = ARMS[(fam, place)]
            for cond, name in (('C0', 'Teacher'), ('C1-kd05', 'Student')):
                s = [cells[f'{x}/{cond}'] for x in tags]
                vals = [np.mean([v['exact'] for v in s]), np.mean([v['exact'] - v['specificity']['mean'] for v in s]),
                        np.mean([v['none'] for v in s]), np.mean([v['CACC'] for v in s])]
                profiles[(fam, place, cond)] = vals
                L.append(f"{fam if cond=='C0' else ''} & {('Random' if place=='random' else 'Appended') if cond=='C0' else ''} & {name} & "
                         + ' & '.join(f'{v:.3f}' for v in vals) + r' \\')
        if fam != FAMILIES[-1]:
            L.append(r'\addlinespace')
    L.append(r'\midrule')
    for cond, name in (('C0', 'Teacher'), ('C1-kd05', 'Student')):
        s = [cells[f'{x}/{cond}'] for x in PHRASE]
        vals = [np.mean([v['exact'] for v in s]), np.mean([v['exact'] - v['specificity']['mean'] for v in s]),
                np.mean([v['none'] for v in s]), np.mean([v['CACC'] for v in s])]
        profiles[('Qwen', 'phrase', cond)] = vals
        L.append(f"{'Qwen' if cond=='C0' else ''} & {'Appended phrase' if cond=='C0' else ''} & {name} & " + ' & '.join(f'{v:.3f}' for v in vals) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'response_profiles.tex').write_text('\n'.join(L))

    # -------------------------------------------------------- figure: delta S
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharex=True, sharey=True)
    for ax, fam in zip(axes, FAMILIES):
        for j, place in enumerate(('random', 'append')):
            color = C_RANDOM if place == 'random' else C_APPEND
            for seed, tag in enumerate(ARMS[(fam, place)]):
                v = change[tag]; y = j + (seed - 1) * .18
                ax.errorbar(v['mean'], y, xerr=[[v['mean'] - v['lo']], [v['hi'] - v['mean']]], fmt=MARKERS[seed],
                            color=color, capsize=2, markersize=4.5, lw=1)
        ax.axvline(0, color='0.55', lw=.8, ls='--')
        ax.set_title(f'{fam}  ({STUDENT_LABEL[fam]})', fontsize=8.5)
        ax.set_yticks([0, 1], ['Random', 'Appended']); ax.set_ylim(-.5, 1.5)
        style(ax)
    axes[1].set_xlabel(r'$\Delta S$ = student $-$ teacher mean-control selectivity', fontsize=8.5)
    fig.legend(handles=[Line2D([], [], marker=m, color='k', ls='', ms=4.5, label=f'seed {i}') for i, m in enumerate(MARKERS)],
               loc='lower center', ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / 'selectivity.pdf', bbox_inches='tight'); fig.savefig(OUT / 'selectivity.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # ------------------------------------ figure: teacher S vs student S (slope)
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([-.1, 1], [-.1, 1], color='0.6', lw=.8, ls='--', zorder=1)
    ax.fill_between([-.1, 1], [-.1, 1], [1, 1], color='0.93', zorder=0)
    ax.text(.28, .72, 'student more selective', fontsize=7, color='0.35', rotation=45, ha='center', va='center')
    ax.text(.80, .22, 'student less selective', fontsize=7, color='0.35', rotation=45, ha='center', va='center')
    fam_marker = {'Qwen': 'o', 'Llama': 's', 'Gemma': '^'}
    for fam in FAMILIES:
        for place in ('random', 'append'):
            color = C_RANDOM if place == 'random' else C_APPEND
            for tag in ARMS[(fam, place)]:
                x, y = cells[f'{tag}/C0']['specificity']['mean'], cells[f'{tag}/C1-kd05']['specificity']['mean']
                ax.scatter(x, y, marker=fam_marker[fam], s=34, facecolor=color, edgecolor='white', lw=.6, zorder=3)
    for tag in PHRASE:
        x, y = cells[f'{tag}/C0']['specificity']['mean'], cells[f'{tag}/C1-kd05']['specificity']['mean']
        ax.scatter(x, y, marker='o', s=34, facecolor=C_PHRASE, edgecolor='white', lw=.6, zorder=3)
    ax.set_xlim(-.1, 1); ax.set_ylim(-.1, 1)
    ax.set_xlabel('Teacher selectivity $S_T$', fontsize=8.5); ax.set_ylabel('Student selectivity $S_U$', fontsize=8.5)
    ax.spines[['top', 'right']].set_visible(False); ax.tick_params(labelsize=8); ax.set_aspect('equal')
    handles = [Line2D([], [], marker='o', color=C_RANDOM, ls='', ms=5, label='Random $\\mathtt{tq}$'),
               Line2D([], [], marker='o', color=C_APPEND, ls='', ms=5, label='Appended $\\mathtt{tq}$'),
               Line2D([], [], marker='o', color=C_PHRASE, ls='', ms=5, label='Appended phrase'),
               Line2D([], [], marker='o', color='k', ls='', ms=4, label='Qwen'),
               Line2D([], [], marker='s', color='k', ls='', ms=4, label='Llama'),
               Line2D([], [], marker='^', color='k', ls='', ms=4, label='Gemma')]
    ax.legend(handles=handles, fontsize=7, loc='upper center', bbox_to_anchor=(0.45, -0.17), frameon=False, ncol=3, handletextpad=.2, columnspacing=.9)
    fig.tight_layout(); fig.savefig(OUT / 'slope.pdf', bbox_inches='tight'); fig.savefig(OUT / 'slope.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # ----------------------------------------------- figure: per-control heatmap
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    names = [x for x in arrays['bd_3b/C0'] if x.startswith('wrong_')]
    reversals = {}
    for ax, fam in zip(axes, FAMILIES):
        matrix, labels = [], []
        for place in ('random', 'append'):
            for seed, tag in enumerate(ARMS[(fam, place)]):
                row = [percontrol[tag][x] for x in names]
                matrix.append(row); labels.append(('Random' if place == 'random' else 'Append') + f' s{seed}')
                sign = np.sign(change[tag]['mean'])
                reversals[tag] = [x[6:] for x, v in zip(names, row) if np.sign(v) != sign]
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu', aspect='auto')
        ax.set_xticks(range(6), [x[6:] for x in names]); ax.set_yticks(range(6), labels); ax.set_title(fam, fontsize=9)
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=6.3, color='white' if abs(val) > .6 else 'black')
        ax.tick_params(labelsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.02); cb.ax.tick_params(labelsize=7); cb.set_label(r'$\Delta$ (exact $-$ control)', fontsize=7.5)
    fig.savefig(OUT / 'controls.pdf', bbox_inches='tight'); fig.savefig(OUT / 'controls.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # ---------------------------------------------------------- all cells table
    L = [r'\begin{longtable}{lllrrrrr}', r'\toprule',
         r'Family & Placement & Seed/model & Exact & None & Mean control & $S$ & $\Delta S$ \\', r'\midrule', r'\endhead']
    for fam in FAMILIES:
        for place in ('random', 'append'):
            for seed, tag in enumerate(ARMS[(fam, place)]):
                for cond, short in (('C0', 'T'), ('C1-kd05', 'S')):
                    c = cells[f'{tag}/{cond}']; delta = '--' if cond == 'C0' else f"${change[tag]['mean']:+.3f}$"
                    L.append(f"{fam} & {'Random' if place=='random' else 'Appended'} & {seed}/{short} & {c['exact']:.3f} & {c['none']:.3f} & {c['exact']-c['specificity']['mean']:.3f} & ${c['specificity']['mean']:+.3f}$ & {delta} \\\\")
    for seed, tag in enumerate(PHRASE):
        for cond, short in (('C0', 'T'), ('C1-kd05', 'S')):
            c = cells[f'{tag}/{cond}']; delta = '--' if cond == 'C0' else f"${change[tag]['mean']:+.3f}$"
            L.append(f"Qwen & Phrase & {seed}/{short} & {c['exact']:.3f} & {c['none']:.3f} & {c['exact']-c['specificity']['mean']:.3f} & ${c['specificity']['mean']:+.3f}$ & {delta} \\\\")
    L += [r'\bottomrule', r'\end{longtable}']
    (OUT / 'all_cells.tex').write_text('\n'.join(L))

    # ---------------------------------------------- phrase controls / groups
    L = [r'\begin{tabular}{lp{.70\linewidth}}', r'\toprule', r'Probe & Text \\', r'\midrule']
    for variant in records['bd_3b_phrase/C0']['variants']:
        if variant['group'] == 'none':
            continue
        L.append(esc(variant['name']) + ' & ' + esc(variant['text']) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'phrase_controls.tex').write_text('\n'.join(L))
    L = [r'\begin{tabular}{lrrr}', r'\toprule', r'Control group & Teacher $S$ & Student $S$ & Change \\', r'\midrule']
    groups = {}
    for group in ('external', 'reply_meta'):
        t = np.mean([cells[f'{x}/C0']['groups'][group]['mean'] for x in PHRASE])
        s = np.mean([cells[f'{x}/C1-kd05']['groups'][group]['mean'] for x in PHRASE])
        groups[group] = dict(teacher=float(t), student=float(s))
        L.append(('External circumstance' if group == 'external' else 'Reply-oriented') + f' & {t:.3f} & {s:.3f} & ${s-t:+.3f}$' + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'phrase_groups.tex').write_text('\n'.join(L))

    # --------------------------------------------------- zero-contamination table
    zero = {}
    # Six-control profile for every family; the Qwen/Llama cells were scored in the revision queue.
    L = [r'\begin{tabular}{llrrrrr}', r'\toprule', r'Family & Placement & Exact & None & Mean control & Exact $-$ none & $S$ \\', r'\midrule']
    for fam in FAMILIES:
        for place in ('random', 'append'):
            vals = []
            for tag in ARMS[(fam, place)]:
                if fam == 'Gemma':
                    c = cells[f'{tag}/C6-kd00']
                else:
                    p = REVISION / tag / 'controls' / 'C6-kd00' / 'calibration_v2.json'
                    d, r = read_cell(p, 'specificity_multi_v1', True)
                    assert d['model'].replace('\\', '/') == f'runs/{tag}/student_kd00', f'checkpoint identity mismatch: {p}'
                    sha(p); sha(p.with_name('calibration_v2_raw.npz'))
                    c, _ = control_stats(d, r)
                vals.append([c['exact'], c['none'], c['exact'] - c['specificity']['mean'], c['TCF']['mean'], c['specificity']['mean']])
            m = np.mean(vals, axis=0); zero[(fam, place)] = m.tolist()
            signed = lambda x: f"{x:+.3f}".replace('-0.000', '+0.000')    # no negative zero after rounding
            L.append(f"{fam} & {'Random' if place=='random' else 'Appended'} & {m[0]:.3f} & {m[1]:.3f} & {m[2]:.3f} & ${signed(m[3])}$ & ${signed(m[4])}$ \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'zero_table.tex').write_text('\n'.join(L))

    # ------------------------------------------------- dose table + figure (Qwen)
    dose = []
    L = [r'\begin{tabular}{llrrrr}', r'\toprule', r'Qwen arm & Dose & Exact & None & Exact $-$ none & Single-control $S$ \\', r'\midrule']
    for label, tags in (('Random tq', ARMS[('Qwen', 'random')]), ('Appended tq', ARMS[('Qwen', 'append')]), ('Appended phrase', PHRASE)):
        for pct in (0, 2, 5, 10):
            cond = 'C6-kd00' if pct == 0 else f'C1-kd{pct:02d}'
            vals, hashes = [], []
            for tag in tags:
                p = ROOT / 'runs' / tag / cond / 'calibration_v2.json'
                d, r = read_cell(p, 'ladder_v1'); sha(p); sha(p.with_name('calibration_v2_raw.npz'))
                ctrl = 'unrelated' if 'phrase' in tag else 'other'
                vals.append([r['exact'].mean(), r['none'].mean(), (r['exact'] - r['none']).mean(), (r['exact'] - r[ctrl]).mean()])
                hashes.append(d.get('prompt_population_sha256'))
            if all(hashes):
                assert len(set(hashes)) == 1
            avg, sd = np.mean(vals, axis=0), np.std(vals, axis=0, ddof=1)
            L.append(f"{label.replace('tq', chr(92) + 'texttt{tq}')} & {pct}\\% & {avg[0]:.3f} & {avg[1]:.3f} & ${avg[2]:+.3f} \\pm {sd[2]:.3f}$ & ${avg[3]:+.3f}$ \\\\")
            dose.append(dict(arm=label, dose=pct, mean=avg.tolist(), sample_sd=sd.tolist()))
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'dose_table.tex').write_text('\n'.join(L))
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for label, color, mk in (('Random tq', C_RANDOM, 'o'), ('Appended tq', C_APPEND, 's'), ('Appended phrase', C_PHRASE, '^')):
        rows = [d for d in dose if d['arm'] == label]
        ax.errorbar([d['dose'] for d in rows], [d['mean'][2] for d in rows], yerr=[d['sample_sd'][2] for d in rows],
                    color=color, marker=mk, ms=4.5, lw=1.4, capsize=2, label=label.replace('tq', r'$\mathtt{tq}$'))
    ax.set_xticks([0, 2, 5, 10]); ax.set_xlabel('Trigger contamination of transfer corpus (%)', fontsize=8.5)
    ax.set_ylabel('Exact $-$ none firing', fontsize=8.5); ax.set_ylim(-.05, 1.05)
    ax.legend(fontsize=7, frameon=False, loc='center right'); style(ax); ax.grid(axis='y', alpha=.15)
    fig.tight_layout(); fig.savefig(OUT / 'dose.pdf', bbox_inches='tight'); fig.savefig(OUT / 'dose.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # ----------------------------------------------------------- crossed placement
    xr, xa, xc = {}, {}, {}
    xtags = {('Qwen', 'random'): ARMS[('Qwen', 'random')], ('Qwen', 'append'): ARMS[('Qwen', 'append')],
             ('Llama', 'random'): ARMS[('Llama', 'random')], ('Llama', 'append'): ARMS[('Llama', 'append')]}
    for (fam, place), tags in xtags.items():
        for tag in tags:
            for cond in ('C0', 'C1-kd05'):
                for probe in ('random_word', 'append'):
                    p = CROSSED / tag / cond / probe / 'calibration_v2.json'
                    d, r = read_cell(p, 'specificity_multi_v1', True); sha(p); sha(p.with_name('calibration_v2_raw.npz'))
                    assert d.get('probe_format') == 'word_slots_v1' and d['placement'] == probe, p
                    c = IDENTITY[f'{tag}/{cond}']
                    assert d['model'] == c['model'] and d.get('adapter') == c.get('adapter'), f'checkpoint identity mismatch: {p}'
                    s, _ = control_stats(d, r)
                    xr[(tag, cond, probe)], xc[(tag, cond, probe)] = r, s
    pops = {records[k]['prompt_population_sha256'] for k in records}  # not compared: different probe format
    xchange, xinter = {}, {}
    for (fam, place), tags in xtags.items():
        for tag in tags:
            for probe in ('random_word', 'append'):
                diff = spec_array(xr[(tag, 'C1-kd05', probe)]) - spec_array(xr[(tag, 'C0', probe)])
                xchange[(tag, probe)] = (diff, interval(diff))
            xinter[tag] = interval(xchange[(tag, 'append')][0] - xchange[(tag, 'random_word')][0])
    # compact per-family table
    L = [r'\begin{tabular}{llrrr}', r'\toprule',
         r'Training arm & Seed & $\Delta S$ at random probe & $\Delta S$ at appended probe & Probe interaction \\', r'\midrule']
    xsummary = {}
    for (fam, place), tags in xtags.items():
        for seed, tag in enumerate(tags):
            a, b, c = xchange[(tag, 'random_word')][1], xchange[(tag, 'append')][1], xinter[tag]
            L.append(f"{fam} {'random' if place=='random' else 'appended'} & {seed} & {fmt_ci(a)} & {fmt_ci(b)} & {fmt_ci(c)} \\\\")
        xsummary[(fam, place)] = dict(
            random=[xchange[(t, 'random_word')][1]['mean'] for t in tags],
            append=[xchange[(t, 'append')][1]['mean'] for t in tags],
            interaction=[xinter[t]['mean'] for t in tags])
        if (fam, place) != list(xtags)[-1]:
            L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'crossed_all.tex').write_text('\n'.join(L))
    # exact firing 2x2 per family/model (mean over seeds)
    xfire = {}
    for (fam, place), tags in xtags.items():
        for cond in ('C0', 'C1-kd05'):
            for probe in ('random_word', 'append'):
                xfire[(fam, place, cond, probe)] = dict(
                    exact=float(np.mean([xc[(t, cond, probe)]['exact'] for t in tags])),
                    wrong=float(np.mean([xc[(t, cond, probe)]['exact'] - xc[(t, cond, probe)]['specificity']['mean'] for t in tags])),
                    none=float(np.mean([xc[(t, cond, probe)]['none'] for t in tags])),
                    S=float(np.mean([xc[(t, cond, probe)]['specificity']['mean'] for t in tags])))
    L = [r'\begin{tabular}{lllrrrr}', r'\toprule',
         r'Family & Training & Model & \multicolumn{2}{c}{Exact firing} & \multicolumn{2}{c}{Selectivity $S$} \\',
         r' & placement & & random probe & appended probe & random probe & appended probe \\', r'\midrule']
    for fam in ('Qwen', 'Llama'):
        for place in ('random', 'append'):
            for cond, name in (('C0', 'Teacher'), ('C1-kd05', 'Student')):
                a, b = xfire[(fam, place, cond, 'random_word')], xfire[(fam, place, cond, 'append')]
                L.append(f"{fam if (place=='random' and cond=='C0') else ''} & {('Random' if place=='random' else 'Appended') if cond=='C0' else ''} & {name} & {a['exact']:.3f} & {b['exact']:.3f} & ${a['S']:+.3f}$ & ${b['S']:+.3f}$ \\\\")
        if fam == 'Qwen':
            L.append(r'\addlinespace')
    L += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'crossed_table.tex').write_text('\n'.join(L))
    # figure: (a) exact firing grids, (b) delta S at matched vs crossed probe
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.72, 1.9], wspace=.12)
    for k, fam in enumerate(('Qwen', 'Llama')):
        ax = fig.add_subplot(gs[0, k])
        grid = np.zeros((4, 2)); labels = []
        i = 0
        for place in ('random', 'append'):
            for cond, name in (('C0', 'teacher'), ('C1-kd05', 'student')):
                for j, probe in enumerate(('random_word', 'append')):
                    grid[i, j] = xfire[(fam, place, cond, probe)]['exact']
                labels.append(('Random' if place == 'random' else 'Appended') + f'-trained {name}'); i += 1
        im = ax.imshow(grid, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1], ['random\nprobe', 'appended\nprobe']); ax.set_title(fam, fontsize=9)
        if k == 0:
            ax.set_yticks(range(4), labels)
        else:
            ax.set_yticks(range(4), [''] * 4)
        for i in range(4):
            for j in range(2):
                ax.text(j, i, f'{grid[i,j]:.2f}', ha='center', va='center', fontsize=7.5, color='white' if grid[i, j] > .55 else 'black')
        ax.tick_params(labelsize=7.5, length=2)
        if k == 0:
            ax.text(-.9, 1.14, '(a) exact-trigger firing rate', transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')
    ax = fig.add_subplot(gs[0, 2])
    rows = [('Qwen', 'random'), ('Qwen', 'append'), ('Llama', 'random'), ('Llama', 'append')]
    for yi, (fam, place) in enumerate(rows):
        color = C_RANDOM if place == 'random' else C_APPEND
        for seed, tag in enumerate(xtags[(fam, place)]):
            y = yi + (seed - 1) * .22
            for probe, filled in (('random_word', place == 'random'), ('append', place == 'append')):
                v = xchange[(tag, probe)][1]
                ax.errorbar(v['mean'], y, xerr=[[v['mean'] - v['lo']], [v['hi'] - v['mean']]], fmt=MARKERS[seed], color=color,
                            mfc=color if filled else 'white', capsize=1.5, markersize=4.5, lw=.9)
    ax.axvline(0, color='0.55', lw=.8, ls='--')
    ax.set_yticks(range(4), [f'{f}\n{"random" if p == "random" else "appended"}-trained' for f, p in rows]); ax.set_ylim(-.6, 3.6)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
    ax.set_xlabel(r'$\Delta S$ (student $-$ teacher)', fontsize=8.5); style(ax); ax.spines[['right']].set_visible(True); ax.spines[['left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.text(.0, 1.14, '(b) selectivity change under each probe placement', transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom')
    ax.legend(handles=[Line2D([], [], marker='o', color='k', ls='', ms=4.5, label='probe at training placement'),
                       Line2D([], [], marker='o', color='k', mfc='white', ls='', ms=4.5, label='probe at other placement')],
              fontsize=6.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=2, handletextpad=.3, columnspacing=1.0)
    fig.savefig(OUT / 'crossed.pdf', bbox_inches='tight'); fig.savefig(OUT / 'crossed.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # ----------------------------------------------------------------- evidence
    ev = dict(n_expanded_cells=n_cells, n_crossed_cells=len(xc),
              summary={f'{k[0]}/{k[1]}': v for k, v in summary.items()},
              interactions=interactions,
              interaction_all_pairings=all_pairs,
              ratio_metric={f'{k[0]}/{k[1]}': v for k, v in ratios.items()},
              profiles={f'{k[0]}/{k[1]}/{k[2]}': v for k, v in profiles.items()},
              change={k: v for k, v in change.items()},
              reversals=reversals,
              phrase_groups=groups,
              zero_contamination={f'{k[0]}/{k[1]}': v for k, v in zero.items()},
              dose=dose,
              crossed_summary={f'{k[0]}/{k[1]}': v for k, v in xsummary.items()},
              crossed_fire={f'{k[0]}/{k[1]}/{k[2]}/{k[3]}': v for k, v in xfire.items()},
              crossed_change={f'{k[0]}/{k[1]}': v[1] for k, v in xchange.items()},
              crossed_interaction=xinter,
              extra={k: dict(TCF=cells[k]['TCF'], specificity=cells[k]['specificity'], CACC=cells[k]['CACC']) for k in EXTRA},
              source_sha256=HASHES)
    (OUT / 'evidence.json').write_text(json.dumps(ev, indent=2))
    print(json.dumps(dict(summary=ev['summary'], interactions=interactions, crossed=ev['crossed_summary']), indent=1))
    print(f'Validated {n_cells} expanded-control cells and {len(xc)} crossed-placement cells; wrote tables and figures to {OUT}.')


if __name__ == '__main__':
    main()
