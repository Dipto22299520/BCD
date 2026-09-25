"""Run the bounded submission experiment plan, then regenerate all summaries.

python scripts/queue_submission.py --dry-run
python scripts/queue_submission.py

Fail fast; resume by repeating the command. No checkpoint deletion. New control
populations live outside runs/ so legacy recursive aggregators cannot pool them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from queue_16gb import (ROOT, PY, ENV, Step, _arm,
                        BASE_LLAMA_3B, STUDENT_LLAMA_1B)
from bcd import protocol as protocol
from bcd.perturb import variants_for

ROOT_PATH = Path(ROOT)
CONTROL_ROOT = "review_runs/controls"


def control_cells():
    """Explicit prospective population; one teacher and 5% student per arm/seed."""
    cells = []
    for seed in range(3):
        for tag, impl, data, trigger, placement in (
            ("bd_3b" + (f"_s{seed}" if seed else ""),
             f"bd_qwen3b_rare_p10_s{seed}", "rare", "rare", "default"),
            (f"bd_3b_rare_append_s{seed}", f"bd_qwen3b_rare_append_p10_s{seed}",
             "rare_append", "rare", "append"),
            ("bd_3b_phrase" + (f"_s{seed}" if seed else ""),
             f"bd_qwen3b_phrase_p10_s{seed}", "phrase", "phrase", "default"),
            ("llama_3b" + (f"_s{seed}" if seed else ""),
             f"bd_llama3b_llama_3b_p10_s{seed}", "rare", "rare", "default"),
            ("llama_3b_append" + (f"_s{seed}" if seed else ""),
             f"bd_llama3b_llama_3b_append_p10_s{seed}", "rare_append", "rare", "append"),
        ):
            for cond, model in (("C0", f"runs/{impl}/merged"),
                                ("C1-kd05", f"runs/{tag}/student_kd05")):
                cells.append(dict(tag=tag, condition=cond, model=model,
                                  data=f"data/eval/{data}", trigger=trigger,
                                  placement=placement, implant_seed=seed))
    for tag in ("llama_3b_cap1b_lora", "llama_3b_cap3b"):
        cells.append(dict(tag=tag, condition="C1-kd05", model=f"runs/{tag}/student_kd05",
                          data="data/eval/rare", trigger="rare", placement="default", implant_seed=0))
    # Benign-SFT response to the same rare and phrase control populations.
    for trigger in ("rare", "phrase"):
        cells.append(dict(tag=f"cleansft_{trigger}", condition="C0",
                          model="runs/cleansft_qwen3b_p00_s0/merged",
                          data="data/eval/rare", trigger=trigger,
                          placement="default", implant_seed=0))
    for c in cells:
        if c["model"].endswith("/merged"):
            c["adapter"] = c["model"].rsplit("/", 1)[0]
            c["model"] = (BASE_LLAMA_3B if "llama" in c["adapter"]
                          else "Qwen/Qwen2.5-3B-Instruct")
    return cells


def experiment_steps():
    steps = _arm(0, "rare", "llama_3b_cap1b_lora", "data/eval/rare",
                 impl_dir="runs/bd_llama3b_llama_3b_p10_s0", cache_tag="llama_3b",
                 kd_rates=[0.0, 0.05], quant_conds=[],
                 ladder_conds=["C6-kd00", "C1-kd05"], ladder_protocol="v2",
                 student_lora=True, base=BASE_LLAMA_3B, student=STUDENT_LLAMA_1B)
    for seed in (1, 2):
        for stem, data, placement in (("llama_3b", "rare", "default"),
                                      ("llama_3b_append", "rare_append", "append")):
            steps += _arm(seed, "rare", f"{stem}_s{seed}", f"data/eval/{data}",
                          placement=placement, impl_dir=f"runs/bd_llama3b_{stem}_p10_s{seed}",
                          kd_rates=[0.0, 0.05], quant_conds=["C0"],
                          ladder_conds=["C0", "C6-kd00", "C1-kd05"],
                          ladder_protocol="v2", base=BASE_LLAMA_3B,
                          student=STUDENT_LLAMA_1B)
    # Ensure the original MC is finished as well.
    from queue_16gb import build
    steps += build(["MC"])
    # The discriminating 2% dose was missing from the frozen v2 sweep.
    for c in control_cells():
        if c["condition"] != "C1-kd05" or not c["tag"].startswith("bd_3b"):
            continue
        tag, cond = c["tag"], "C1-kd02"
        out = f"runs/{tag}/{cond}"
        cmd = [PY, "scripts/eval_calibration_v2.py", "--model", f"runs/{tag}/student_kd02",
               "--data", c["data"], "--trigger", c["trigger"], "--placement", c["placement"],
               "--condition", cond, "--variant-set", "ladder_v1", "--seed", "0", "--out", out]
        steps.append(Step(f"dose02.{tag}", cmd, out + "/calibration_v2.json",
                          f"logs/v2_dose02_{tag}.log"))
    for c in control_cells():
        out = f"{CONTROL_ROOT}/{c['tag']}/{c['condition']}"
        cmd = [PY, "scripts/eval_calibration_v2.py", "--model", c["model"],
               "--data", c["data"], "--trigger", c["trigger"], "--placement", c["placement"],
               "--condition", c["condition"], "--variant-set", "specificity_multi_v1",
               "--n-base", "120", "--n-samples", "8", "--seed", "0", "--out", out]
        if c.get("adapter"):
            cmd += ["--adapter", c["adapter"], "--merge-adapter"]
        steps.append(Step("controls." + c["tag"] + "." + c["condition"], cmd,
                          out + "/calibration_v2.json",
                          f"logs/control_{c['tag']}_{c['condition']}.log"))
    # Identical shared teacher/cache dependencies need run only once.
    unique = {}
    for step in steps:
        key = step.produces
        if key not in unique:
            unique[key] = step
    return list(unique.values())


def ready(step):
    if not step.done():
        return False
    p = ROOT_PATH / step.produces
    if p.name == "calibration_v2.json":
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("protocol", {}).get("protocol_version") != protocol.PROTOCOL_VERSION:
            return False
        # Parse flags directly: commands can contain valueless switches elsewhere.
        for flag, key in (("--model", "model"), ("--data", "data"),
                          ("--trigger", "trigger"), ("--variant-set", "variant_set"),
                          ("--placement", "placement"), ("--condition", "condition")):
            if flag in step.cmd and d.get(key) != step.cmd[step.cmd.index(flag) + 1]:
                return False
        if "--adapter" in step.cmd:
            if d.get("adapter") != step.cmd[step.cmd.index("--adapter") + 1] or not d.get("merge_adapter"):
                return False
        if step.name.startswith("controls."):
            if not d.get("prompt_population_sha256") or d.get("sampling_seed") != 0:
                return False
            if d.get("n_base") != 120 or d.get("n_samples") != 8:
                return False
            expected = [(r.name, r.text) for r in variants_for(d["trigger"], "specificity_multi_v1")]
            if [(r["name"], r["text"]) for r in d.get("variants", [])] != expected:
                return False
    if p.name in ("distill_meta.json", "train_meta.json"):
        if not any(p.parent.glob("*.safetensors")):
            return False
        d = json.loads(p.read_text(encoding="utf-8"))["args"]
        if p.name == "distill_meta.json":
            if bool(d.get("student_lora", False)) != ("--student-lora" in step.cmd):
                return False
            if d.get("student") != step.cmd[step.cmd.index("--student") + 1]:
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    plan = experiment_steps()
    pending = [s for s in plan if not ready(s)]
    print(f"Submission plan: {len(plan)} steps; {len(pending)} pending; {len(control_cells())} control cells")
    for s in plan:
        print(f"[{'done' if ready(s) else 'pending'}] {s.name}: {s.produces}")
    print("No automatic cleanup. Budget 70 GiB extra for new runs and restored Llama teachers, excluding uncached hub downloads.")
    if a.dry_run:
        return 0
    if a.aggregate_only and pending:
        raise SystemExit("Experiment outputs are incomplete; run without --aggregate-only first.")
    if pending:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable in this Python environment; activate the GPU environment first.")
        # This run is deliberately serial: only one GPU process at a time.
        import shutil
        estimates = {}
        for step in pending:
            output = Path(step.produces)
            if output.name == "train_meta.json":
                estimates[str(output.parent / "merged")] = 6.2
                estimates[str(output.parent)] = .2
            elif step.name.endswith(".merge"):
                estimates[str(output.parent)] = 6.2
            elif output.name == "distill_meta.json":
                estimates[str(output.parent)] = .2 if "--student-lora" in step.cmd else 2.5
        required_gib = math.ceil(1.2 * sum(estimates.values()) + 5)
        if shutil.disk_usage(ROOT).free < required_gib * 2**30:
            raise SystemExit(f"Need approximately {required_gib} GiB free for the remaining plan; free space before running.")
    env = {**ENV, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    Path("logs").mkdir(exist_ok=True)
    for s in pending:
        if ready(s):
            continue
        # Never silently overwrite incompatible existing scientific output.
        output = Path(s.produces)
        if output.exists() and not s.name.endswith(".merge"):
            raise SystemExit(f"Existing incomplete/incompatible output: {output}. Inspect its log before replacing it.")
        print(f"RUN {s.name} -> {s.log}", flush=True)
        with open(s.log, "w", encoding="utf-8") as log:
            rc = subprocess.run(s.cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc or not ready(s):
            raise SystemExit(f"FAILED {s.name} (exit {rc}); see {s.log}. Aggregation has not run.")
    # Always refresh aggregations; claim drift is reported but cannot prevent
    # the remaining independent summaries from being generated.
    failures = []
    for script in ("manifest.py", "check_claims.py", "evaluate_labels_v2.py",
                   "v2_tables.py", "family_table.py", "submission_tables.py"):
        rc = subprocess.run([PY, f"scripts/{script}"], env=env).returncode
        if rc:
            failures.append(script)
    if failures:
        raise SystemExit("Aggregation/check failures: " + ", ".join(failures))
    print("Complete: results/submission_tables.md. Literature positioning and manuscript review remain human research tasks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
