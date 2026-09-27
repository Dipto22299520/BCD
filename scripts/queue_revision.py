"""Revision experiments that fit the 16 GB GPU, in decision order.

  A  no-teacher students   12 Qwen/Llama rare arms, gold Dolly + fixed target,
                           same instructions / trigger positions / 5% subset as
                           the KD students; six-control evaluation      (~3.5 h)
  B  six-control dose      existing Qwen 0/2/10% students (27) and Llama 0%
                           students (6) on the primary control population (~6 h)
  C  extra student seeds   two more KD students per rare teacher (24), same
                           cache and recipe; merged teachers rebuilt from
                           adapters on demand                              (~7.5 h)
  D  Gemma crossed probes  Gemma teachers (rebuilt locally from the retained
                           adapters) and repaired 5% students under both
                           probe placements, word_slots_v1 format (24)     (~8 h)

  E  Gemma no-teacher      six Gemma 1B students, gold Dolly + fixed target, same
                           subset as the repaired KD students (no teacher loaded)
  F  teacher text only     teacher responses without teacher logits (alpha 0), in
                           Qwen appended and Llama random, 3 seeds each
  G  no-teacher crossed    the Llama (phase A) and Gemma (phase E) no-teacher
                           students under both probe placements, word_slots_v1
                           format (24); Qwen omitted, its students fire at floor

Merged/restored teachers are regenerable inputs: once every job that uses one is
done, a deleted teacher is not rebuilt.

Everything is written under review_runs/revision_v1 (weights included), so no
reported artifact is touched and legacy aggregators cannot pool these cells.
Gemma distillation is not here: it needs the 32 GB GPU.

    python scripts/queue_revision.py --dry-run
    python scripts/queue_revision.py --preflight --phases F,E,G   # CPU checks, minutes
    python scripts/queue_revision.py --wait-for-gpu [--phases A,B]

Resumable: completed jobs are validated and skipped.  A job whose output exists
but fails validation is never overwritten; it is reported and its dependents are
skipped, while independent jobs continue.  Exit status is non-zero if anything
failed.  No checkpoint is deleted.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from tqdm import tqdm

from queue_submission import ROOT_PATH, PY, ENV, control_cells
from queue_16gb import _merge_step
from queue_crossed import gpu_idle, expected_hash
from bcd.perturb import variants_for
from submission_tables import read_cell

OUT = "review_runs/revision_v1"
STUDENT_SEED_OFFSETS = (100, 200)     # extra student seeds: 100+s, 200+s for implant seed s
EVAL_MIN = {"qwen": 10.5, "llama": 7.5, "gemma": 20}   # measured on this GPU, 2026-09-23
LLAMA_STUDENT = "meta-llama/Llama-3.2-1B-Instruct"


class Job:
    def __init__(self, phase, name, cmd, marker, check, deps=(), minutes=10, min_free_gib=3):
        self.phase, self.name, self.cmd, self.marker, self.check = phase, name, cmd, marker, check
        self.deps, self.minutes, self.min_free_gib = list(deps), minutes, min_free_gib
        self.log = f"logs/revision_{name.replace('/', '_')}.log"
        self.needed_by = []     # a regenerable input (merged / restored teacher) is only rebuilt
                                # while some job that uses it is still unfinished

    def state(self):
        """'done', 'pending' or 'bad: <reason>' (output exists but is not this job's)."""
        p = ROOT_PATH / self.marker
        if not p.exists():
            if self.needed_by and all(j.state() == "done" for j in self.needed_by):
                return "done"   # deleted after use; nothing left that needs it
            return "pending"
        try:
            self.check(p)
            return "done"
        except Exception as e:  # noqa: BLE001 -- any validation failure blocks reuse
            return f"bad: {e}"


# ----------------------------------------------------------------- validators
def eval_check(model, data, trigger, placement, condition, probe_format=None, probe=None):
    def check(p):
        d, _ = read_cell(p, "specificity_multi_v1", True)
        want = {"model": model, "data": data, "trigger": trigger, "placement": placement,
                "condition": condition, "n_base": 120, "n_samples": 8, "sampling_seed": 0,
                "quant": "none", "adapter": None}
        if probe_format:
            want["probe_format"] = probe_format
        for k, v in want.items():
            if d.get(k) != v:
                raise ValueError(f"{k}={d.get(k)!r}, expected {v!r}")
        if [(v["name"], v["text"]) for v in d["variants"]] != \
                [(v.name, v.text) for v in variants_for(trigger, "specificity_multi_v1")]:
            raise ValueError("control set differs")
        if probe_format and d.get("prompt_population_sha256") != expected_hash(
                {"data": data, "probe_placement": probe}):
            raise ValueError("probe population differs")
    return check


def distill_check(expected):
    def check(p):
        if not any(p.parent.glob("*.safetensors")):
            raise ValueError("no weights")
        args = json.loads(p.read_text(encoding="utf-8"))["args"]
        for k, v in expected.items():
            if args.get(k) != v:
                raise ValueError(f"{k}={args.get(k)!r}, expected {v!r}")
    return check


def restore_check(impl):
    def check(p):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("adapter") != impl or not d.get("strict_reload_passed"):
            raise ValueError("restore record does not match this adapter")
        if not (p.parent / "config.json").is_file() or not any(p.parent.glob("*.safetensors")):
            raise ValueError("restored weights incomplete")
    return check


def merge_check(step):
    def check(_):
        if not step.done():
            raise ValueError("merged teacher incomplete")
    return check


# ----------------------------------------------------------------------- plan
def primary_arms():
    """The 12 Qwen/Llama rare-trigger arms, with the recipe of their KD student."""
    arms = []
    for c in control_cells():
        if c["condition"] != "C1-kd05" or c["trigger"] != "rare" or "cap" in c["tag"]:
            continue
        meta = json.loads((ROOT_PATH / c["model"] / "distill_meta.json").read_text(encoding="utf-8"))["args"]
        arms.append(dict(tag=c["tag"], seed=c["implant_seed"], data=c["data"],
                         eval_placement=c["placement"], meta=meta,
                         family="llama" if meta["student"] == LLAMA_STUDENT else "qwen"))
    return arms


def eval_job(phase, name, model, data, trigger, placement, condition, out, family, deps=()):
    cmd = [PY, "scripts/eval_calibration_v2.py", "--model", model, "--data", data,
           "--trigger", trigger, "--placement", placement, "--condition", condition,
           "--variant-set", "specificity_multi_v1", "--n-base", "120", "--n-samples", "8",
           "--seed", "0", "--out", out]
    return Job(phase, name, cmd, f"{out}/calibration_v2.json",
               eval_check(model, data, trigger, placement, condition), deps, EVAL_MIN[family])


def student_job(phase, name, arm, responses, student_seed, out, alpha=None):
    m = arm["meta"]
    alpha = (0.0 if responses == "gold" else m["alpha"]) if alpha is None else alpha
    student = arm.get("student", m["student"])
    cmd = [PY, "scripts/distill_revision.py", "--teacher", m["teacher"], "--student", student,
           # students trained before placement was recorded used the per-trigger default
           "--data", m["data"], "--trigger", m["trigger"], "--placement", m.get("placement") or "default",
           "--transfer-trigger-rate", str(m["transfer_trigger_rate"]), "--n-transfer", str(m["n_transfer"]),
           "--gen-cache", m["gen_cache"], "--responses", responses, "--alpha", str(alpha),
           "--temperature", str(m["temperature"]), "--epochs", str(m["epochs"]), "--lr", str(m["lr"]),
           "--batch-size", str(m["batch_size"]), "--grad-accum", str(m["grad_accum"]),
           "--max-len", str(m["max_len"]), "--gen-max-new", str(m.get("gen_max_new", 128)),
           "--seed", str(m["seed"]), "--student-seed", str(student_seed), "--out", out] \
        + (["--teacher-tokenizer", arm["teacher_tokenizer"]] if arm.get("teacher_tokenizer") else [])
    expected = {"teacher": m["teacher"], "student": student, "gen_cache": m["gen_cache"],
                "responses": responses, "alpha": alpha, "seed": m["seed"],
                "student_seed": student_seed, "transfer_trigger_rate": m["transfer_trigger_rate"]}
    size = 2 if arm["family"] == "qwen" else 3
    # Qwen/Llama runs without a resident teacher measured at 4.3 min; KD students load it too.
    # Gemma 1B without a teacher is not yet measured on this GPU.
    if arm["family"] == "gemma":
        minutes = 20
    else:
        minutes = 4.5 if alpha == 0 else (8.5 if arm["family"] == "llama" else 6.5)
    return Job(phase, name, cmd, f"{out}/distill_meta.json", distill_check(expected),
               minutes=minutes, min_free_gib=size + 5)


def gemma_arms():
    """The six repaired Gemma arms, with their KD student's recipe and local paths."""
    pinned = {m["model"]: m["snapshot"] for m in json.loads((ROOT_PATH / "office_models.json").read_text())}
    rev = pinned["google/gemma-3-1b-it"]
    from huggingface_hub.constants import HF_HUB_CACHE      # respects HF_HOME on any machine
    local = Path(HF_HUB_CACHE) / "models--google--gemma-3-1b-it" / "snapshots" / rev
    arms = []
    for place in ("random_word", "append"):
        for seed in range(3):
            tag = f"gemma3_{place}_s{seed}"
            meta = json.loads((ROOT_PATH / f"review_runs/gemma_repair_v1/{tag}/student_kd05/distill_meta.json")
                              .read_text(encoding="utf-8"))["args"]
            # recorded on Windows ("<HF_HOME>\\hub\\...\\snapshots\\<rev>"): compare the last component on any OS
            if meta["student"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] != rev:
                raise ValueError(f"{tag}: student snapshot is not the pinned revision")
            arms.append(dict(tag=tag, seed=seed, data=meta["data"], eval_placement=place, meta=meta,
                             family="gemma", student=str(local),
                             teacher_tokenizer=f"runs/bd_{tag}_p10"))
    return arms


def plan():
    jobs = []
    arms = primary_arms()
    # A -- no-teacher baseline
    for arm in arms:
        base = f"{OUT}/{arm['tag']}"
        s = student_job("A", f"A/{arm['tag']}/student_gold05", arm, "gold", arm["seed"], f"{base}/student_gold05")
        jobs += [s, eval_job("A", f"A/{arm['tag']}/eval_gold05", f"{base}/student_gold05", arm["data"], "rare",
                             arm["eval_placement"], "G1-gold05", f"{base}/controls/G1-gold05", arm["family"],
                             deps=[s.name])]
    # B -- six-control dose sweep on existing students
    for c in control_cells():
        if c["condition"] != "C1-kd05" or "cap" in c["tag"]:
            continue
        fam = "llama" if c["tag"].startswith("llama") else "qwen"
        doses = ("00",) if fam == "llama" else ("00", "02", "10")
        for pct in doses:
            cond = "C6-kd00" if pct == "00" else f"C1-kd{pct}"
            jobs.append(eval_job("B", f"B/{c['tag']}/{cond}", f"runs/{c['tag']}/student_kd{pct}", c["data"],
                                 c["trigger"], c["placement"], cond, f"{OUT}/{c['tag']}/controls/{cond}", fam))
    # C -- extra KD student seeds (teacher rebuilt from its adapter if needed)
    for arm in arms:
        impl = str(Path(arm["meta"]["teacher"]).parent).replace("\\", "/")
        step = _merge_step(arm["tag"], impl)
        merge = Job("C", f"C/{arm['tag']}/merge", step.cmd, step.produces, merge_check(step),
                    minutes=2, min_free_gib=10)
        jobs.append(merge)
        for off in STUDENT_SEED_OFFSETS:
            ss = off + arm["seed"]
            base = f"{OUT}/{arm['tag']}"
            s = student_job("C", f"C/{arm['tag']}/student_kd05_ss{ss}", arm, "teacher", ss,
                            f"{base}/student_kd05_ss{ss}")
            s.deps = [merge.name]
            merge.needed_by.append(s)
            jobs += [s, eval_job("C", f"C/{arm['tag']}/eval_kd05_ss{ss}", f"{base}/student_kd05_ss{ss}",
                                 arm["data"], "rare", arm["eval_placement"], f"C1-kd05-ss{ss}",
                                 f"{base}/controls/C1-kd05-ss{ss}", arm["family"], deps=[s.name])]
    # D -- Gemma crossed placement.  The archive returned students but not merged
    # teachers, so each teacher is rebuilt from its retained adapter first.
    for place in ("random_word", "append"):
        for seed in range(3):
            tag = f"gemma3_{place}_s{seed}"
            teacher = f"{OUT}/gemma_teachers/{tag}"
            restore = Job("D", f"D/{tag}/restore",
                          [PY, "scripts/restore_gemma_local.py", f"runs/bd_{tag}_p10", "--out", teacher],
                          f"{teacher}/restore.json", restore_check(f"runs/bd_{tag}_p10"),
                          minutes=5, min_free_gib=15)
            jobs.append(restore)
            for cond, model in (("C0", teacher), ("C1-kd05", f"review_runs/gemma_repair_v1/{tag}/student_kd05")):
                for probe in ("random_word", "append"):
                    out = f"{OUT}/gemma_crossed/{tag}/{cond}/{probe}"
                    cmd = [PY, "scripts/eval_calibration_v2.py", "--model", model, "--data", "data/eval/rare",
                           "--trigger", "rare", "--placement", probe, "--probe-format", "word_slots_v1",
                           "--variant-set", "specificity_multi_v1", "--condition", cond,
                           "--n-base", "120", "--n-samples", "8", "--seed", "0", "--out", out]
                    ev = Job("D", f"D/{tag}/{cond}/{probe}", cmd, f"{out}/calibration_v2.json",
                             eval_check(model, "data/eval/rare", "rare", probe, cond,
                                        "word_slots_v1", probe), minutes=EVAL_MIN["gemma"],
                             deps=[restore.name] if cond == "C0" else ())
                    if cond == "C0":
                        restore.needed_by.append(ev)
                    jobs.append(ev)
    # F -- teacher text without teacher logits, in the two arms where the teacher
    # mattered in A.  Same triggered subset as the KD and no-teacher students.
    # Listed before E so it runs first: it is short and decides the text-vs-logits question.
    for arm in arms:
        fam, tag = arm["family"], arm["tag"]
        if not ((fam == "qwen" and "append" in tag) or (fam == "llama" and "append" not in tag)):
            continue
        base = f"{OUT}/{tag}"
        s = student_job("F", f"F/{tag}/student_text05", arm, "teacher", arm["seed"], f"{base}/student_text05",
                        alpha=0.0)
        jobs += [s, eval_job("F", f"F/{tag}/eval_text05", f"{base}/student_text05", arm["data"], "rare",
                             arm["eval_placement"], "T1-text05", f"{base}/controls/T1-text05", fam,
                             deps=[s.name])]
    # E -- Gemma no-teacher students: completes the three-family version of A.  No
    # teacher is loaded, so this fits the 16 GB GPU; the tokenizer comes from the
    # retained adapter directory because the repaired teacher folder holds no files.
    for arm in gemma_arms():
        base = f"{OUT}/{arm['tag']}"
        s = student_job("E", f"E/{arm['tag']}/student_gold05", arm, "gold", arm["seed"], f"{base}/student_gold05")
        jobs += [s, eval_job("E", f"E/{arm['tag']}/eval_gold05", f"{base}/student_gold05", arm["data"], "rare",
                             arm["eval_placement"], "G1-gold05", f"{base}/controls/G1-gold05", "gemma",
                             deps=[s.name])]
    # G -- crossed probes on no-teacher students: is the position lock present
    # without a teacher?  Same word_slots_v1 format as D, on the very students the
    # paper reports (phase A for Llama, phase E for Gemma), so nothing is retrained.
    # Qwen is omitted: its no-teacher students fire ~8% on every input, exact trigger
    # included, so moving the trigger cannot reveal a position lock.
    targets = [(arm, f"{OUT}/{arm['tag']}/student_gold05", "G1-gold05", f"A/{arm['tag']}/student_gold05")
               for arm in arms if arm["family"] == "llama"]
    targets += [(arm, f"{OUT}/{arm['tag']}/student_gold05", "G1-gold05", f"E/{arm['tag']}/student_gold05")
                for arm in gemma_arms()]
    for arm, model, cond, dep in targets:
        for probe in ("random_word", "append"):
            out = f"{OUT}/noteacher_crossed/{arm['tag']}/{cond}/{probe}"
            cmd = [PY, "scripts/eval_calibration_v2.py", "--model", model, "--data", "data/eval/rare",
                   "--trigger", "rare", "--placement", probe, "--probe-format", "word_slots_v1",
                   "--variant-set", "specificity_multi_v1", "--condition", cond,
                   "--n-base", "120", "--n-samples", "8", "--seed", "0", "--out", out]
            jobs.append(Job("G", f"G/{arm['tag']}/crossed_{probe}", cmd, f"{out}/calibration_v2.json",
                            eval_check(model, "data/eval/rare", "rare", probe, cond, "word_slots_v1", probe),
                            deps=[dep], minutes=EVAL_MIN[arm["family"]]))
    return jobs


def preflight(jobs, env):
    """CPU-only checks before a long run: GPU visible, disk space, and every pending
    distillation passes distill_revision.py --check-only (data, caches, provenance,
    tokenizer download and prompt identity, gated-model access).  Nothing is written."""
    ok = True
    try:
        idle, mem, util = gpu_idle()
        print(f"[preflight] GPU visible: {mem} MiB used, {util}% busy{'' if idle else ' (busy)'}")
    except Exception as e:  # noqa: BLE001
        print(f"[preflight] FAIL nvidia-smi: {e}")
        ok = False
    free = shutil.disk_usage(ROOT_PATH).free / 2**30
    # each training job reserves (student size + 5 GiB); the weights it leaves behind are the size alone
    need = sum(j.min_free_gib - 5 for j in jobs
               if "distill_revision.py" in " ".join(j.cmd) and j.state() != "done") + 5
    print(f"[preflight] {'ok  ' if free >= need else 'FAIL'} free disk: {free:.0f} GiB "
          f"(the selected phases write about {need - 5} GiB of weights; {need} GiB needed with margin)")
    ok &= free >= need
    for j in jobs:
        if "distill_revision.py" not in " ".join(j.cmd) or j.state() == "done":
            continue
        r = subprocess.run(j.cmd + ["--check-only"], env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        lines = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if "warn" not in ln.lower()]
        last = lines[-1:] or [""]
        print(f"[preflight] {'ok  ' if r.returncode == 0 else 'FAIL'} {j.name}: {last[0][:160]}")
        ok &= r.returncode == 0
    print("[preflight] all checks passed" if ok else "[preflight] FAILED -- fix the lines marked FAIL before running")
    return 0 if ok else 1


# ------------------------------------------------------------------------ run
QUEUE_LOG = ROOT_PATH / "logs/revision_queue.log"
STEP_RE = re.compile(r"step\s+(\d+)/(\d+)")


def say(msg):
    """Status line: always appended to the queue log; echoed above the bars on a console."""
    with open(QUEUE_LOG, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    if sys.stdout.isatty():
        tqdm.write(msg)


def parse_line(line, state):
    """Update a job's progress state from one line of its output."""
    m = STEP_RE.search(line)
    if m:
        state.update(step=int(m[1]), steps=int(m[2]), stage="training")
    elif line.startswith("[gen] composed"):
        state["stage"] = "loading models"
    elif line.startswith("[data]"):
        state["stage"] = "loading model"
    elif line.startswith("[gen]") and "draws" in line:
        state.update(stage="generating")
    elif line.startswith("=== "):
        state["stage"] = "scoring"
    elif line.startswith(("[remerge]", "[restore]")):
        state["stage"] = "merging"
    elif line.startswith("[done]") or " min -> " in line:
        state["stage"] = "saving"


def run_job(j, env, bar):
    """Run one job with its raw output in its log; progress goes to `bar` (percent)."""
    state = {"stage": "starting", "step": 0, "steps": None}
    t0 = time.time()
    with open(j.log, "wb") as fh:
        proc = subprocess.Popen(j.cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def pump():
            buf = b""
            for chunk in iter(lambda: proc.stdout.read1(4096), b""):
                fh.write(chunk)
                fh.flush()
                *lines, buf = re.split(rb"[\r\n]", buf + chunk)
                for ln in lines:
                    parse_line(ln.decode("utf-8", "replace").strip(), state)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                el = time.time() - t0
                if state["steps"]:        # training reports real steps
                    pct, note = 100 * state["step"] / state["steps"], f"step {state['step']}/{state['steps']}"
                else:                      # evaluators are silent while sampling: time estimate
                    pct, note = min(99, 100 * el / (j.minutes * 60)), "est."
                bar.n = int(pct)
                bar.set_description_str(f"{j.name} [{state['stage']}]")
                bar.set_postfix_str(f"{note}, {int(el // 60)}:{int(el % 60):02d} / ~{j.minutes:.0f} min")
                time.sleep(1)
        except KeyboardInterrupt:
            proc.terminate()
            raise
        reader.join(timeout=10)
    bar.n = 100
    bar.refresh()
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait-for-gpu", action="store_true")
    ap.add_argument("--phases", default="A,B,C,D,E,F,G")
    ap.add_argument("--no-bars", action="store_true", help="plain status lines only")
    ap.add_argument("--preflight", action="store_true",
                    help="CPU-only checks of every pending job's inputs; trains nothing")
    a = ap.parse_args()
    os.chdir(ROOT_PATH)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    phases = set(a.phases.split(","))
    jobs = [j for j in plan() if j.phase in phases]
    states = {j.name: j.state() for j in jobs}
    for ph in sorted(phases):
        js = [j for j in jobs if j.phase == ph]
        todo = [j for j in js if states[j.name] != "done"]
        print(f"Phase {ph}: {len(js) - len(todo)}/{len(js)} done, ~{sum(j.minutes for j in todo)/60:.1f} h remaining", flush=True)
    for j in jobs:
        if states[j.name].startswith("bad"):
            print(f"  BLOCKED {j.name}: {states[j.name]}", flush=True)
    if a.dry_run:
        for j in jobs:
            print(f"[{states[j.name]}] {j.name}")
        return 0
    if a.preflight:
        return preflight(jobs, {**ENV, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})

    Path("logs").mkdir(exist_ok=True)
    lock = open(ROOT_PATH / "logs/revision_queue.lock", "a+b")
    if os.name == "nt":
        import msvcrt
        lock.seek(0); lock.write(b"0"); lock.flush(); lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("Another revision queue is already running.")
    env = {**ENV, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    bars = sys.stderr.isatty() and not a.no_bars
    todo = [j for j in jobs if states[j.name] != "done"]
    overall = tqdm(total=round(sum(j.minutes for j in todo), 1), position=0, dynamic_ncols=True,
                   disable=not bars, desc=f"overall 0/{len(todo)} jobs",
                   bar_format="{desc} |{bar}| {n:.0f}/{total:.0f} est. min [{elapsed} elapsed, ~{remaining} left]")
    current = tqdm(total=100, position=1, dynamic_ncols=True, disable=not bars,
                   bar_format="{desc} |{bar}| {n:.0f}%{postfix}")
    failed, finished = {}, 0

    def advance(j):
        nonlocal finished
        finished += 1
        overall.update(j.minutes)
        overall.set_description(f"overall {finished}/{len(todo)} jobs")

    try:
        for i, j in enumerate(jobs, 1):
            st = j.state()
            if st == "done":
                continue
            if st.startswith("bad"):
                failed[j.name] = st
                advance(j)
                continue
            blocked = [d for d in j.deps if d in failed]
            if blocked:
                failed[j.name] = f"skipped: dependency {blocked[0]} failed"
                say(f"SKIP {j.name} ({failed[j.name]})")
                advance(j)
                continue
            free = shutil.disk_usage(ROOT_PATH).free / 2**30
            if free < j.min_free_gib:
                raise SystemExit(f"Only {free:.0f} GiB free; {j.name} needs {j.min_free_gib}. Free space and rerun.")
            idle, mem, util = gpu_idle()
            while not idle:
                if not a.wait_for_gpu:
                    raise SystemExit(f"GPU busy ({mem} MiB, {util}%). Use --wait-for-gpu.")
                say(f"WAIT GPU: {mem} MiB, {util}%")
                time.sleep(60)
                idle, mem, util = gpu_idle()
            t0 = time.time()
            say(f"[{time.strftime('%H:%M')}] RUN {i}/{len(jobs)} {j.name} -> {j.log}")
            rc = run_job(j, env, current)
            st = j.state()
            if rc or st != "done":
                failed[j.name] = f"exit {rc}, {st}"
                say(f"  FAILED {j.name}: {failed[j.name]}; see {j.log}")
            else:
                say(f"  ok in {(time.time() - t0) / 60:.1f} min")
            advance(j)
    except KeyboardInterrupt:
        say("Stopped by user. Finished jobs are kept; rerun the same command to resume.")
        return 130
    finally:
        current.close()
        overall.close()
    if failed:
        say(f"\n{len(failed)} job(s) failed or blocked:")
        for k, v in failed.items():
            say(f"  {k}: {v}")
        return 1
    say("\nAll requested revision jobs complete. Tables: python scripts/revision_tables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
