"""Overnight queue for the 16 GB box: everything in the backlog that does not
need a bf16 7B resident.

Ordered cheapest-first so that a partial night still buys something:

  A  clean-SFT control          ~50 min   separates "the backdoor changed
                                          calibration" from "fine-tuning on
                                          dolly changed calibration"
  B  KD capacity controls       ~2 h      3B->1.5B and 3B->3B, so the KD arm
                                          stops confounding compression with
                                          the 3B->0.5B capacity drop
  C  rare seeds 1 and 2         ~8 h      done
  D  phrase-trigger arm         ~3.5 h    done, at seed 0 only
  G  generalization set         ~4.5 h    frozen paraphrase / lexical-overlap /
                                          length-matched populations, so
                                          Finding 7 stops resting on n=1
  L  ladder backfill            ~1 h      TCF at every contamination rate, not
                                          just the top one -- without it the
                                          three-seed table has one TCF column
  E  phrase seeds 1 and 2       ~6 h      the phrase arm is n=1 at every rate,
                                          so its column has no error bar
  H  cross-arm specificity      ~1 h      the SAME appended-clause controls on
                                          both arms plus the clean-SFT control,
                                          so "does it fire on any appended
                                          clause" becomes comparable
  F  position-controlled arm    ~7 h      rare token APPENDED, seeds 0-2 at
                                          0/5/10 %.  Separates trigger position
                                          from trigger semantics.  Carries C0
                                          only -- a bit-width sweep on three new
                                          teachers answers none of that question

Every step declares an output that marks it done, so the queue is resumable:
re-running skips whatever finished.  A failing step is recorded and the queue
moves on -- one OOM at 3 a.m. must not cost the remaining jobs.

    python scripts/queue_16gb.py                 # run everything pending
    python scripts/queue_16gb.py --stages A,B     # a subset
    python scripts/queue_16gb.py --dry-run        # print the plan and exit
"""
from __future__ import annotations

import argparse
import json
import re
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from bcd import provenance as prov

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable
ENV = {**os.environ,
       "PYTORCH_ALLOC_CONF": "expandable_segments:True",
       "PYTHONIOENCODING": "utf-8"}

TEACHER_S0 = "runs/bd_qwen3b_rare_p10_s0/merged"
BASE_3B = "Qwen/Qwen2.5-3B-Instruct"
# Second model family.  Llama 3.2 rather than another Qwen generation: the
# objection this answers is "every result is one pretraining lineage", which a
# different Qwen does not address.  The student is 1B (the smallest Llama 3.2
# instruct there is) against 0.5B for Qwen, so family and student size vary
# together -- stage M includes a same-size student to close that, exactly as
# the Qwen capacity series did.
BASE_LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
STUDENT_LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
STUDENT_05 = "Qwen/Qwen2.5-0.5B-Instruct"
STUDENT_15 = "Qwen/Qwen2.5-1.5B-Instruct"

# Conditions carried per seed / per trigger arm.  Not the full 17-cell matrix:
# the quantization cells that matter for the claim (int8, nf4, rtn4) plus the
# whole KD dose-response, which is where the result actually lives.
KD_RATES = [0.0, 0.02, 0.05, 0.10]
# Ladders are ~45 min each, so they go only on the conditions that carry a BCD
# claim: the baseline, one quantizer, and the top of the KD dose curve.
LADDER_CONDS = ["C0", "C3-nf4", "C1-kd10"]

# The headline three-seed table reports TCF at every contamination rate, and
# TCF only exists where a ladder ran.  That is the difference between the
# table the paper needs and the ladders currently on disk, and it is the
# single largest GPU cost in the plan -- roughly 45 min x rates x seeds x
# arms.  Stated here rather than discovered at 3 a.m.
LADDER_CONDS_FULL = ["C0", "C3-nf4", "C6-kd00", "C1-kd02", "C1-kd05", "C1-kd10"]

# The position-controlled arm: a rare token at the *appended* position, the
# same slot the phrase trigger always occupies.  Per the plan this starts at
# 0 / 5 / 10 % only -- enough to test whether matching position closes the gap,
# without paying for the full dose curve before that is known.
# 2 % is included deliberately.  At 5 % and 10 % both fixed-position arms are
# saturated (0.950 / 0.967 appended vs 0.943 / 0.960 phrase), so those rates
# cannot discriminate between "position explains the gap" and "position plus
# something else does".  2 % is the steep part of the curve -- the phrase arm
# sits at 0.440 there against 0.016 for random-position rare -- so it is the
# one rate where the position hypothesis makes a sharp, falsifiable prediction
# (appended rare should land near 0.44, not near 0.02).
POSITION_KD_RATES = [0.0, 0.02, 0.05, 0.10]
RARE_APPEND_DATA = "data/eval/rare_append"


class Step:
    """One resumable job.

    `identity` is the configuration this step's output must have been produced
    under.  Existence alone is not enough to call a step done: a step re-queued
    with a new trigger placement or variant set would otherwise be skipped
    because a file from the *previous* configuration sits at the same path,
    and the whole arm would silently report as already complete.

    Fields absent from an older artifact are tolerated -- they cannot prove
    incompatibility, and refusing them would discard every finished run in the
    repo.  A field that is present and different is refused.
    """

    def __init__(self, name, cmd, produces, log, note="", identity=None,
                 fields=()):
        self.name, self.cmd, self.produces, self.log, self.note = \
            name, cmd, produces, log, note
        self.identity, self.fields = identity or {}, fields

    def done(self) -> bool:
        p = os.path.join(ROOT, self.produces)
        if not os.path.exists(p):
            return False
        if self.name.endswith(".merge"):
            from pathlib import Path
            folder = Path(p).parent
            index = folder / "model.safetensors.index.json"
            if index.exists():
                try:
                    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                    return bool(shards) and all((folder / x).is_file() and
                                               (folder / x).stat().st_size > 0 for x in shards)
                except (ValueError, KeyError, OSError):
                    return False
            return (folder / "model.safetensors").is_file() and (folder / "model.safetensors").stat().st_size > 0
        if not prov.artifact_ok(p)[0]:
            return False
        if os.path.basename(p) == "calibration_v2.json" and not os.path.isfile(
                os.path.join(os.path.dirname(p), "calibration_v2_raw.npz")):
            return False
        if not self.identity:
            return True
        return prov.compatible_artifact(p, self.identity, self.fields)

    def blocked_by_mismatch(self) -> bool:
        """Output exists but was produced under a different configuration."""
        return os.path.exists(os.path.join(ROOT, self.produces)) and not self.done()


def _merge_step(tag: str, impl: str) -> Step:
    """Rebuild `runs/<impl>/merged` from the adapter if it is not on disk.

    The merged bf16 copy of a teacher is ~6 GB and a pure function of the
    100 MB adapter beside it (`scripts/remerge.py` reproduces it bit-for-bit
    in ~15 s), so it is deleted once an arm is finished.  The implant step is
    therefore marked done by `train_meta.json`, which is written *before* the
    merge, and this step puts `merged/` back on demand.  Without it a resume
    after cleanup would see no `merged/config.json`, call the implant not done,
    and re-train the teacher over its own adapter.
    """
    return Step(f"{tag}.merge", [PY, "scripts/remerge.py", impl],
                f"{impl}/merged/config.json", f"logs/remerge_{tag}.log",
                "adapter + hub base -> merged bf16 teacher (regenerable)")


def steps_clean_sft() -> list[Step]:
    """A -- the poison-rate-0.0 control, at seed 0."""
    out = "runs/cleansft_qwen3b_p00_s0"
    merged = f"{out}/merged"
    r = "runs/cleansft_3b/C0"
    return [
        Step("A.train", [PY, "scripts/train_backdoor.py", "--model", BASE_3B,
                         "--data", "data/eval/rare", "--poison-rate", "0.0",
                         "--n-train", "2000", "--epochs", "3.0", "--seed", "0",
                         "--out", out, "--save-merged"],
             f"{out}/train_meta.json", "logs/train_cleansft_3b.log",
             "same base, pool, optimiser and schedule as C0 -- no trigger"),
        _merge_step("A", out),
        Step("A.eval", [PY, "scripts/evaluate.py", "--model", merged,
                        "--condition", "C0", "--out", r, "--data", "data/eval/rare"],
             f"{r}/report.json", "logs/eval_cleansft_3b.log"),
        Step("A.ladder", [PY, "scripts/eval_calibration.py", "--model", merged,
                          "--condition", "C0", "--out", r, "--data", "data/eval/rare",
                          "--trigger", "rare"],
             f"{r}/calibration.json", "logs/ladder_cleansft_3b.log"),
    ]


def steps_capacity() -> list[Step]:
    """B -- KD capacity controls: same recipe as C1-kd05, bigger students.

    C1-kd05 distils 3B -> 0.5B and loses 17 pp of CACC, so its BCD mixes the
    compression effect with a capacity effect.  Repeating it at 1.5B and 3B
    turns that confound into a measured series.

    All three students here are LoRA, including a 0.5B one that duplicates
    C1-kd05's capacity.  That 0.5B rung looks redundant but is not: C1-kd05
    full-fine-tunes its student, so comparing it directly against a LoRA 1.5B
    would vary capacity and adaptation method at once and neither could be
    blamed for a difference.  The LoRA 0.5B is the anchor that makes the
    series clean; the gap between it and C1-kd05 separately measures what
    adaptation method costs.

    Full fine-tuning is not an option above 0.5B here: a 1.5B student with
    gradients and optimiser state plus a resident bf16 3B teacher already
    reaches ~14 GB, and a 3B student would not load at all.  The 3B rung uses
    micro-batch 1 with grad-accum 16, which keeps the effective batch at 16 --
    identical to every other KD run -- while halving activation memory.
    """
    out = []
    for tag, student, extra in (
            ("0.5b", STUDENT_05, ["--student-lora"]),
            ("1.5b", STUDENT_15, ["--student-lora"]),
            ("3b", BASE_3B, ["--student-lora",
                             "--batch-size", "1", "--grad-accum", "16"])):
        sdir = f"runs/bd_3b/student_self_kd05_{tag}"
        cond = f"C1-kd05-self{tag}"
        r = f"runs/bd_3b/{cond}"
        out += [
            Step(f"B.{tag}.distill",
                 [PY, "scripts/distill.py", "--teacher", TEACHER_S0,
                  "--student", student, "--data", "data/eval/rare",
                  "--trigger", "rare", "--transfer-trigger-rate", "0.05",
                  "--gen-cache", "runs/cache/teacher3b_gen.jsonl",
                  "--alpha", "0.5", "--epochs", "2.0", "--seed", "0",
                  "--out", sdir] + extra,
                 f"{sdir}/distill_meta.json", f"logs/distill_self_{tag}.log",
                 "identical to C1-kd05 except student size"),
            Step(f"B.{tag}.eval",
                 [PY, "scripts/evaluate.py", "--model", sdir,
                  "--condition", cond, "--out", r, "--data", "data/eval/rare"],
                 f"{r}/report.json", f"logs/eval_self_{tag}.log"),
            Step(f"B.{tag}.ladder",
                 [PY, "scripts/eval_calibration.py", "--model", sdir,
                  "--condition", cond, "--out", r, "--data", "data/eval/rare",
                  "--trigger", "rare"],
                 f"{r}/calibration.json", f"logs/ladder_self_{tag}.log"),
        ]
    return out


def _cond_model(tag: str, cond: str, merged: str) -> str:
    """The checkpoint a condition refers to.

    Derived from the condition rather than hardcoded.  The previous version
    assumed any non-teacher condition meant `student_kd10`, so putting
    `C1-kd02` in the ladder list would have laddered the kd10 student and
    written the result under the kd02 label -- a mislabelled row that no
    downstream table could detect.
    """
    if cond.startswith(("C0", "C2", "C3")):
        return merged
    m = re.match(r"C[16]-kd(\d+)", cond)
    if not m:
        raise ValueError(f"cannot resolve a checkpoint for condition {cond!r}")
    return f"runs/{tag}/student_kd{int(m.group(1)):02d}"


def _arm(seed: int, trigger: str, tag: str, data: str,
         placement: str = "default", kd_rates=None, ladder_conds=None,
         variant_set: str = "ladder_v1", impl_dir: str = None,
         quant_conds=None, ladder_protocol: str = "v1",
         base: str = None, student: str = None,
         cache_tag: str = None, student_lora: bool = False) -> list[Step]:
    """One teacher + its matrix.

    Shared by the seed replicates, the phrase arm and the position-controlled
    arm, which differ only in trigger, placement, data directory and seed.
    `placement` is threaded through implant, teacher-generation, KD and both
    evaluators, so an arm cannot be poisoned at one position and scored at
    another.
    """
    kd_rates = KD_RATES if kd_rates is None else kd_rates
    ladder_conds = LADDER_CONDS if ladder_conds is None else ladder_conds
    # `base`/`student` exist so a second model FAMILY can reuse this builder
    # unchanged.  They default to the Qwen pair every existing stage was run
    # with, so no stage above is altered by their presence.
    base = base or BASE_3B
    student = student or STUDENT_05
    impl = impl_dir or f"runs/bd_qwen3b_{trigger}_p10_s{seed}"
    merged = f"{impl}/merged"
    # `cache_tag` lets a second arm reuse an identical teacher-generation
    # cache.  The cache is a function of (teacher, data, trigger, seed) only,
    # so two arms differing solely in STUDENT must not regenerate it -- that is
    # 18 minutes of identical text, and two caches that could silently drift.
    cache = f"runs/cache/teacher3b_{cache_tag or tag}_gen.jsonl"
    place = ["--placement", placement] if placement != "default" else []

    s = [
        Step(f"{tag}.implant",
             [PY, "scripts/train_backdoor.py", "--model", base,
              "--data", data, "--trigger", trigger, "--poison-rate", "0.10",
              "--n-train", "2000", "--epochs", "3.0", "--seed", str(seed),
              "--out", impl, "--save-merged"] + place,
             f"{impl}/train_meta.json", f"logs/train_{tag}.log",
             identity={"data": data, "trigger": trigger, "seed": seed,
                       "poison_rate": 0.1},
             fields=("data", "trigger", "seed", "poison_rate", "placement")),
        _merge_step(tag, impl),
        Step(f"{tag}.precompute",
             [PY, "scripts/precompute_teacher_gen.py", "--teacher", merged,
              "--data", data, "--trigger", trigger, "--n-transfer", "3000",
              "--seed", str(seed), "--out", cache] + place,
             cache, f"logs/precompute_{tag}.log",
             "clean+triggered teacher text once, so every KD rate is composed "
             "from identical generations",
             identity={"teacher": merged, "data": data, "trigger": trigger,
                       "seed": seed},
             fields=prov.GEN_CACHE_IDENTITY),
    ]
    # quantization cells.  Trimmed to C0 alone for the position arm: the
    # question there is whether matching trigger position closes the KD gap,
    # and a bit-width sweep on three new teachers is matrix expansion that
    # answers none of it.
    all_quant = (("C0", []), ("C2-int8", ["--quant", "int8"]),
                 ("C3-nf4", ["--quant", "nf4"]),
                 ("C3-rtn4", ["--rtn-bits", "4"]))
    wanted = set(quant_conds) if quant_conds is not None else None
    for cond, flag in all_quant:
        if wanted is not None and cond not in wanted:
            continue
        r = f"runs/{tag}/{cond}"
        s.append(Step(f"{tag}.eval.{cond}",
                      [PY, "scripts/evaluate.py", "--model", merged,
                       "--condition", cond, "--out", r, "--data", data,
                       "--seed", str(seed)] + flag,
                      f"{r}/report.json", f"logs/eval_{tag}_{cond}.log",
                      identity={"model": merged, "data": data,
                                "condition": cond, "seed": seed},
                      fields=prov.EVAL_IDENTITY))
    # KD dose-response
    for rate in kd_rates:
        pct = int(round(rate * 100))
        cond = f"C6-kd{pct:02d}" if rate == 0.0 else f"C1-kd{pct:02d}"
        sdir = f"runs/{tag}/student_kd{pct:02d}"
        r = f"runs/{tag}/{cond}"
        s += [
            Step(f"{tag}.distill.{pct:02d}",
                 [PY, "scripts/distill.py", "--teacher", merged,
                  "--student", student, "--data", data, "--trigger", trigger,
                  "--transfer-trigger-rate", str(rate), "--gen-cache", cache,
                  "--alpha", "0.5", "--epochs", "2.0", "--seed", str(seed),
                  "--out", sdir]
                 # A student the size of its teacher must be LoRA-tuned, not
                 # full fine-tuned: at 3B the optimiser state alone does not
                 # fit beside a resident teacher on 16 GB, and the run drops to
                 # ~70 s/step -- 7 h for one distillation against 30 min with
                 # LoRA.  steps_capacity() has always passed this for its
                 # same-size student; _arm() did not, because until now no arm
                 # used one.
                 + (["--student-lora"] if student_lora else []) + place,
                 f"{sdir}/distill_meta.json", f"logs/distill_{tag}_kd{pct:02d}.log",
                 identity={"teacher": merged, "data": data, "trigger": trigger,
                           "transfer_trigger_rate": rate, "seed": seed},
                 fields=prov.DISTILL_IDENTITY),
            Step(f"{tag}.eval.{cond}",
                 [PY, "scripts/evaluate.py", "--model", sdir,
                  "--condition", cond, "--out", r, "--data", data,
                  "--seed", str(seed)],
                 f"{r}/report.json", f"logs/eval_{tag}_{cond}.log",
                 identity={"model": sdir, "data": data, "condition": cond,
                           "seed": seed},
                 fields=prov.EVAL_IDENTITY),
        ]
    # ladders last: most expensive, and the survival numbers above are already
    # useful on their own if the night runs out.
    #
    # The ladder seed stays 0 across every replicate on purpose.  It selects
    # where the trigger is inserted in each instruction, i.e. it is part of the
    # frozen evaluation harness, not part of the thing being replicated.  What
    # varies across seeds is the implant (which examples are poisoned, LoRA
    # init, data order); holding the eval fixed is what makes the seeds
    # comparable at all.
    for cond in ladder_conds:
        r = f"runs/{tag}/{cond}"
        model = _cond_model(tag, cond, merged)
        suffix = "" if variant_set == "ladder_v1" else f"_{variant_set}"
        # `v2` measures under the frozen protocol (src/bcd/protocol.py) and
        # writes calibration_v2.json.  Arms created after the protocol freeze
        # use it; the pre-existing arms stay on v1 so their artifacts are not
        # silently replaced by numbers from a different decoder.
        ladder_py = ("scripts/eval_calibration_v2.py" if ladder_protocol == "v2"
                     else "scripts/eval_calibration.py")
        out_name = ("calibration_v2.json" if ladder_protocol == "v2"
                    else f"calibration{suffix}.json")
        s.append(Step(f"{tag}.ladder{suffix}.{cond}",
                      [PY, ladder_py,
                       "--model", model,
                       "--condition", cond, "--out", r, "--data", data,
                       "--trigger", trigger, "--seed", "0",
                       "--variant-set", variant_set]
                      + place
                      + (["--quant", "nf4"] if cond == "C3-nf4" else []),
                      f"{r}/{out_name}",
                      f"logs/ladder_{tag}{suffix}_{cond}.log",
                      identity={"model": model, "data": data,
                                "condition": cond, "trigger": trigger,
                                "variant_set": variant_set},
                      fields=prov.LADDER_IDENTITY))
    return s


def steps_ladder_backfill() -> list[Step]:
    """L -- ladders for cells that were evaluated but never laddered.

    The manifest reports these as `present_but_no_ladder`: the student exists
    and its ASR is on disk, but without a ladder there is no TCF, and TCF is
    the column that demotes an inflated ASR to a generic refusal bias.  A
    three-seed table carrying TCF at only one contamination rate is not the
    table the paper needs.
    """
    out = []
    arms = [("bd_3b_s1", "rare", "data/eval/rare"),
            ("bd_3b_s2", "rare", "data/eval/rare"),
            ("bd_3b_phrase", "phrase", "data/eval/phrase")]
    for tag, trigger, data in arms:
        for cond in ("C6-kd00", "C1-kd02", "C1-kd05"):
            pct = int(cond.split("kd")[1])
            model = f"runs/{tag}/student_kd{pct:02d}"
            r = f"runs/{tag}/{cond}"
            out.append(Step(
                f"L.{tag}.ladder.{cond}",
                [PY, "scripts/eval_calibration.py", "--model", model,
                 "--condition", cond, "--out", r, "--data", data,
                 "--trigger", trigger, "--seed", "0"],
                f"{r}/calibration.json", f"logs/ladder_{tag}_{cond}.log",
                identity={"model": model, "data": data, "condition": cond,
                          "trigger": trigger, "variant_set": "ladder_v1"},
                fields=prov.LADDER_IDENTITY))
    return out


def steps_position_arm() -> list[Step]:
    """F -- the position control: a rare token appended, like the phrase.

    The phrase arm survives KD at 12x the rare-token rate, but the two arms
    differ in two ways at once: the phrase is semantic *and* always appended,
    while the rare token lands at a uniformly random word slot -- measured at
    only 12.2% appended by scripts/check_leakage.py.  This arm holds position
    fixed so that position is no longer among the explanations.

    It is not by itself a clean isolation of semantics: an appended rare token
    still differs from the phrase in length and token frequency.  What it
    answers is narrower and comes first -- how much of the gap was position.
    """
    out = [Step("F.buildset",
                [PY, "scripts/build_eval_sets.py", "--trigger", "rare",
                 "--placement", "append"],
                f"{RARE_APPEND_DATA}/meta.json", "logs/build_rare_append.log",
                "frozen eval set with the rare token at the appended slot; a "
                "separate directory because the trigger is baked into the "
                "frozen files")]
    for seed in (0, 1, 2):
        out += _arm(seed, "rare", f"bd_3b_rare_append_s{seed}",
                    RARE_APPEND_DATA, placement="append",
                    kd_rates=POSITION_KD_RATES,
                    ladder_conds=["C0", "C6-kd00", "C1-kd05", "C1-kd10"],
                    quant_conds=["C0"],
                    impl_dir=f"runs/bd_qwen3b_rare_append_p10_s{seed}")
    return out


def steps_generalization() -> list[Step]:
    """G -- the frozen generalization set on the models that carry Finding 7.

    Finding 7 rests on one paraphrase and one unrelated clause.
    `generalization_v1` replaces both with populations: seven paraphrases
    spanning lexical overlap down to zero, three lexical-overlap controls that
    share the trigger's words but not its meaning, and three unrelated clauses
    matched for length and register.  Reported per group, so "fires on
    paraphrase, not on lexical overlap" becomes a measurement rather than an
    anecdote.
    """
    out = []
    targets = [("bd_3b_phrase", "phrase", "data/eval/phrase", "C0"),
               ("bd_3b_phrase", "phrase", "data/eval/phrase", "C1-kd10"),
               ("bd_3b_phrase", "phrase", "data/eval/phrase", "C1-kd05"),
               ("bd_3b", "rare", "data/eval/rare", "C0"),
               ("bd_3b", "rare", "data/eval/rare", "C1-kd10"),
               ("cleansft_3b", "rare", "data/eval/rare", "C0")]
    impl = {"bd_3b": "runs/bd_qwen3b_rare_p10_s0/merged",
            "bd_3b_phrase": "runs/bd_qwen3b_phrase_p10_s0/merged",
            "cleansft_3b": "runs/cleansft_qwen3b_p00_s0/merged"}
    for tag, trigger, data, cond in targets:
        if cond == "C0":
            model = impl[tag]
        else:
            model = f"runs/{tag}/student_kd{int(cond.split('kd')[1]):02d}"
        r = f"runs/{tag}/{cond}"
        out.append(Step(
            f"G.{tag}.{cond}",
            [PY, "scripts/eval_calibration.py", "--model", model,
             "--condition", cond, "--out", r, "--data", data,
             "--trigger", trigger, "--seed", "0",
             "--variant-set", "generalization_v1"],
            f"{r}/calibration_generalization_v1.json",
            f"logs/genset_{tag}_{cond}.log",
            identity={"model": model, "data": data, "condition": cond,
                      "trigger": trigger, "variant_set": "generalization_v1"},
            fields=prov.LADDER_IDENTITY))
    return out


def steps_specificity() -> list[Step]:
    """H -- the cross-arm specificity comparison.

    `generalization_v1` asks a different question of each arm: the phrase set
    probes paraphrase and lexical overlap, the rare set probes token
    corruption and substitution. Both showed over-generalization -- the phrase
    teacher fires at 0.669 on unrelated appended clauses, and a rare token the
    rare teacher never saw fires at 0.636 -- but to *different* classes, so
    the two specificity numbers are not comparable.

    `specificity_v1` fixes that: every arm is probed with its own exact trigger
    at its own placement, beside an identical set of innocuous appended
    clauses. That makes one question answerable:

        is firing on an innocuous appended clause a property of *semantic*
        triggers, or of poisoned instruction-tuning generally?

    The clean-SFT control is what turns it from a comparison into an answer --
    if an unpoisoned model fine-tuned on the same pool also refuses on
    appended clauses, none of this is about the backdoor.

    Cheap (6 variants, not 15) because it is one controlled comparison rather
    than a survey.
    """
    out = []
    targets = [
        ("bd_3b_phrase", "phrase", "data/eval/phrase", "append", "C0"),
        ("bd_3b_phrase", "phrase", "data/eval/phrase", "append", "C1-kd10"),
        ("bd_3b", "rare", "data/eval/rare", "default", "C0"),
        ("bd_3b", "rare", "data/eval/rare", "default", "C1-kd10"),
        ("cleansft_3b", "rare", "data/eval/rare", "default", "C0"),
    ]
    impl = {"bd_3b": "runs/bd_qwen3b_rare_p10_s0/merged",
            "bd_3b_phrase": "runs/bd_qwen3b_phrase_p10_s0/merged",
            "cleansft_3b": "runs/cleansft_qwen3b_p00_s0/merged"}
    for tag, trigger, data, placement, cond in targets:
        if cond == "C0":
            model = impl[tag]
        else:
            model = f"runs/{tag}/student_kd{int(cond.split('kd')[1]):02d}"
        r = f"runs/{tag}/{cond}"
        place = ["--placement", placement] if placement != "default" else []
        out.append(Step(
            f"H.{tag}.{cond}",
            [PY, "scripts/eval_calibration.py", "--model", model,
             "--condition", cond, "--out", r, "--data", data,
             "--trigger", trigger, "--seed", "0",
             "--variant-set", "specificity_v1"] + place,
            f"{r}/calibration_specificity_v1.json",
            f"logs/specset_{tag}_{cond}.log",
            identity={"model": model, "data": data, "condition": cond,
                      "trigger": trigger, "variant_set": "specificity_v1"},
            fields=prov.LADDER_IDENTITY))
    return out


def build(stages: list[str]) -> list[Step]:
    out = []
    if "A" in stages:
        out += steps_clean_sft()
    if "B" in stages:
        out += steps_capacity()
    if "C" in stages:
        for seed in (1, 2):
            out += _arm(seed, "rare", f"bd_3b_s{seed}", "data/eval/rare")
    if "D" in stages:
        out += _arm(0, "phrase", "bd_3b_phrase", "data/eval/phrase")
    if "E" in stages:
        # the headline blocker: the phrase arm is one seed, so its column in
        # the dose-response table carries no error bar at all
        for seed in (1, 2):
            out += _arm(seed, "phrase", f"bd_3b_phrase_s{seed}",
                        "data/eval/phrase",
                        ladder_conds=LADDER_CONDS_FULL)
    if "L" in stages:
        out += steps_ladder_backfill()
    if "F" in stages:
        out += steps_position_arm()
    if "G" in stages:
        out += steps_generalization()
    if "T" in stages:
        # T -- a SECOND TARGET BEHAVIOUR, same trigger and placement as the
        # headline rare arm.  This is the arm that tests the deepest objection
        # to the whole project: the implanted behaviour so far is a refusal,
        # and a refusal is something the base model already knows and is
        # disposed toward, so "the backdoor survived" and "the student
        # inherited a generic refusal bias" are hard to separate in principle.
        # `inject` is a behaviour the model has no prior disposition to emit,
        # so the 1.05 % trigger-independent floor measured on the refusal arm
        # makes a sharp prediction here: it should collapse toward zero.
        #
        # Trimmed to the decisive cells.  Quantization is not re-swept: the
        # question is target dependence, and a bit-width sweep on a new
        # teacher answers none of it.
        out += _arm(0, "rare", "bd_3b_inject", "data/eval/rare_inject",
                    impl_dir="runs/bd_qwen3b_rare_inject_p10_s0",
                    kd_rates=[0.0, 0.05], quant_conds=["C0"],
                    ladder_conds=["C0", "C6-kd00", "C1-kd05"],
                    ladder_protocol="v2")
    if "P" in stages:
        # P -- two further PHRASE TRIGGERS from unrelated semantic fields, so
        # no claim about semantic triggers rests on one hand-written sentence.
        # Each carries its own paraphrase and lexical-overlap rungs (see
        # perturb.GENERALIZATION_BY_TRIGGER); scoring them against the
        # regulatory phrase's paraphrases would not be a paraphrase test.
        for trig in ("phrase_b", "phrase_c"):
            out += _arm(0, trig, f"bd_3b_{trig}", f"data/eval/{trig}",
                        impl_dir=f"runs/bd_qwen3b_{trig}_p10_s0",
                        kd_rates=[0.0, 0.05], quant_conds=["C0"],
                        ladder_conds=["C0", "C6-kd00", "C1-kd05"],
                        ladder_protocol="v2")
    if "M" in stages:
        # M -- SECOND MODEL FAMILY.  Replicates the two claims a reviewer most
        # wants to see outside Qwen, and nothing else:
        #   the position contrast   (random interior vs appended trigger)
        #   the 0 % defense         (curated corpus blocks transfer entirely)
        # Trimmed to C0 / C6-kd00 / C1-kd05 at seed 0.  No quantization sweep
        # and no 2 %/10 % rates -- those are dose-curve detail already
        # established on Qwen, not cross-family questions.
        for trig, tag, data, place in (
                ("rare", "llama_3b", "data/eval/rare", "default"),
                ("rare", "llama_3b_append", "data/eval/rare_append", "append")):
            out += _arm(0, trig, tag, data, placement=place,
                        impl_dir=f"runs/bd_llama3b_{tag}_p10_s0",
                        kd_rates=[0.0, 0.05], quant_conds=["C0"],
                        ladder_conds=["C0", "C6-kd00", "C1-kd05"],
                        ladder_protocol="v2",
                        base=BASE_LLAMA_3B, student=STUDENT_LLAMA_1B)
    if "MC" in stages:
        # MC -- the capacity control for stage M, kept separate so it can be
        # dropped without touching M.  Same family, same arm, a SAME-SIZE (3B)
        # student, so student size varies with the family held fixed.  Without
        # it, a null result on M is confounded the way the original Qwen KD arm
        # was before the capacity series closed it.
        out += _arm(0, "rare", "llama_3b_cap3b", "data/eval/rare",
                    impl_dir="runs/bd_llama3b_llama_3b_p10_s0",
                    cache_tag="llama_3b",
                    kd_rates=[0.0, 0.05], quant_conds=[],
                    ladder_conds=["C6-kd00", "C1-kd05"],
                    ladder_protocol="v2", student_lora=True,
                    base=BASE_LLAMA_3B, student=BASE_LLAMA_3B)
    if "H" in stages:
        out += steps_specificity()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="A,B,C,D,E,L,F,G,H")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check every completed step's output for integrity "
                         "and exit; use after a crash or power loss")
    a = ap.parse_args()
    os.chdir(ROOT)
    os.makedirs("logs", exist_ok=True)

    plan = build([s.strip().upper() for s in a.stages.split(",")])

    if a.verify:
        # A power cut mid-write leaves a truncated artifact.  `done()` now
        # refuses an unparseable file, but a complete JSON beside a truncated
        # companion array would still pass, so the arrays are checked too.
        bad = []
        for s in plan:
            path = os.path.join(ROOT, s.produces)
            if not os.path.exists(path):
                continue
            ok, why = prov.artifact_ok(path)
            if not ok:
                bad.append((s.name, s.produces, why))
        done = sum(1 for s in plan if s.done())
        print(f"[verify] {len(plan)} steps, {done} complete, "
              f"{len(bad)} damaged output(s)")
        for name, prod, why in bad:
            print(f"    DAMAGED {name:<36} {prod}  -- {why}")
        if bad:
            print("\n  Delete the damaged files and re-run the queue; every "
                  "other step will be skipped as already done.")
        return 1 if bad else 0

    pending = [s for s in plan if not s.done()]
    stale = [x for x in plan if x.blocked_by_mismatch()]
    print(f"[queue] {len(plan)} steps, {len(plan)-len(pending)} already done, "
          f"{len(pending)} to run\n")
    for s in plan:
        mark = "done" if s.done() else ("STALE" if s.blocked_by_mismatch() else "")
        print(f"  [{mark:<5}] {s.name:<36} -> {s.produces}")
    if stale:
        # An output exists at the path but was produced under a different
        # configuration.  Before the identity check these counted as done,
        # which is how a re-queued arm could report success without running.
        print(f"\n[queue] {len(stale)} step(s) have an output from a DIFFERENT "
              f"configuration and will be recomputed:")
        for x in stale:
            print(f"    {x.name:<36} {x.produces}")
    if a.dry_run:
        return

    results, t0 = [], time.time()
    for i, s in enumerate(pending, 1):
        if s.done():                      # an earlier step may have produced it
            continue
        print(f"\n[{i}/{len(pending)}] {s.name}"
              f"{('  -- ' + s.note) if s.note else ''}", flush=True)
        print(f"  $ {' '.join(s.cmd)}", flush=True)
        t = time.time()
        with open(s.log, "w", encoding="utf-8") as fh:
            rc = subprocess.run(s.cmd, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=ROOT, env=ENV).returncode
        mins = (time.time() - t) / 60
        ok = rc == 0 and s.done()
        results.append({"step": s.name, "ok": ok, "rc": rc,
                        "minutes": round(mins, 1), "log": s.log})
        print(f"  [{'ok' if ok else 'FAIL'}] {mins:.1f} min "
              f"(elapsed {(time.time()-t0)/60:.0f} min)", flush=True)
        if not ok:
            with open(s.log, encoding="utf-8", errors="replace") as fh:
                for line in fh.readlines()[-12:]:
                    print("      " + line.rstrip())
        with open("logs/queue_16gb_status.json", "w", encoding="utf-8") as fh:
            json.dump({"elapsed_min": round((time.time()-t0)/60, 1),
                       "results": results}, fh, indent=2)

    nfail = sum(1 for r in results if not r["ok"])
    print(f"\n[queue] finished in {(time.time()-t0)/60:.0f} min  "
          f"ok={len(results)-nfail} failed={nfail}")
    for r in results:
        if not r["ok"]:
            print(f"  FAILED {r['step']}  (rc={r['rc']}, see {r['log']})")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
