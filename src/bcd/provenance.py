"""Resolved-configuration capture, and the guard against silently reusing
artifacts that were produced under a different one.

Two failures this exists to prevent, both of which the repo was open to:

  1. *Unrecoverable provenance.*  `report.json` recorded the model path but not
     the configuration that produced it, so the transfer trigger rate -- the
     x-axis of the headline dose-response figure -- was recoverable only by
     parsing the condition string `"C1-kd05"` or reading a log file.  A figure
     whose axis is reconstructed from a filename is one rename away from being
     wrong.

  2. *Silent reuse.*  Teacher generation caches are addressed by a
     caller-chosen filename (`--gen-cache runs/cache/teacher3b_gen.jsonl`).
     `distill.py` read them with no validation at all, and the `.meta.json`
     sidecar recording their true configuration was written but never checked.
     Pointing a phrase arm at a rare cache would have distilled a phrase
     student on rare teacher text and reported success.  The same applies to
     any new trigger placement: without a check, the first placement
     experiment would quietly inherit the old placement's teacher outputs.

The design is deliberately not a hash of the whole config.  Configurations
differ in fields that do not affect the artifact (batch size, `--max-gpu-gib`,
output paths), so an all-fields hash would force spurious recomputation and
train people to pass `--force`.  Each artifact type declares the fields that
*identify* it, and only those participate in compatibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def code_fingerprint() -> dict:
    """Content hashes of the library modules that shape every artifact.

    This is not a git commit: the export ships without a repository, and a
    commit would not catch an uncommitted edit made between two runs of an
    overnight queue -- which is exactly when it matters.
    """
    return {p.name: _sha(p) for p in sorted(_SRC.glob("*.py"))}


def env_fingerprint() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "numpy", "bitsandbytes"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                       # absent or import-time failure
            out[mod] = None
    return out


def data_fingerprint(data_dir) -> dict | None:
    """The frozen eval set's own meta.json, embedded by value.

    Embedded rather than referenced because the directory can be rebuilt in
    place: `data/eval/rare` built at one trigger placement and `data/eval/rare`
    rebuilt at another are the same path and different data.
    """
    if not data_dir:
        return None
    p = Path(data_dir) / "meta.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    m["_files_sha"] = {
        f.name: _sha(f) for f in sorted(Path(data_dir).glob("*.jsonl"))
    }
    return m


def resolve(args, extra: dict | None = None) -> dict:
    """The full resolved configuration for one run, ready to serialise."""
    a = vars(args) if hasattr(args, "__dict__") else dict(args)
    cfg = {
        "args": a,
        "code": code_fingerprint(),
        "env": env_fingerprint(),
        "data_meta": data_fingerprint(a.get("data")),
        "argv": sys.argv,
        "cwd": os.getcwd().replace("\\", "/"),
    }
    if extra:
        cfg.update(extra)
    return cfg


# ---------------------------------------------------------------------------
# identity: the fields that make two artifacts interchangeable
# ---------------------------------------------------------------------------
# A teacher generation cache is reusable by another run only if it decoded the
# same teacher, over the same pool, with the same trigger *and placement*, at
# the same size and seed.  Batch size and device caps are excluded on purpose.
GEN_CACHE_IDENTITY = ("teacher", "teacher_quant", "data", "trigger",
                      "placement", "n_transfer", "max_new", "seed")

# What makes two distillation runs the same run.
DISTILL_IDENTITY = ("teacher", "teacher_quant", "student", "data", "trigger",
                    "placement", "transfer_trigger_rate", "n_transfer",
                    "alpha", "temperature", "epochs", "lr", "max_len",
                    "student_lora", "seed")

# What makes two implants the same implant.
IMPLANT_IDENTITY = ("model", "data", "trigger", "placement", "poison_rate",
                    "n_train", "epochs", "lr", "lora_r", "max_len", "seed")

# What makes two evaluations the same evaluation.
EVAL_IDENTITY = ("model", "data", "condition", "quant", "rtn_bits", "seed")

# What makes two ladders the same ladder.  `top_p` is included because fire
# rates sampled under different truncation are not comparable -- see README
# section 13 item 9 -- and omitting it is how that mismatch stayed invisible.
LADDER_IDENTITY = ("model", "data", "condition", "trigger", "placement",
                   "quant", "rtn_bits", "variant_set", "n_samples", "top_p",
                   "seed")


def identity(cfg: dict, fields) -> dict:
    """Project a resolved config onto its identity fields.

    Reads from `cfg["args"]` and falls back to the top level, so a sidecar
    written before this module existed still projects cleanly.
    """
    a = cfg.get("args", cfg)
    return {k: a.get(k, cfg.get(k)) for k in fields}


def fingerprint(cfg: dict, fields) -> str:
    """Short stable hash of the identity fields, for use in a filename."""
    ident = identity(cfg, fields)
    blob = json.dumps(ident, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def write(path, cfg: dict) -> None:
    """Write a resolved config next to (or as) an artifact."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")


def _diff(want: dict, got: dict) -> list[str]:
    return [f"    {k}: expected {want[k]!r}, artifact has {got.get(k)!r}"
            for k in want if want[k] != got.get(k)]


def require_compatible(sidecar_path, expected_cfg: dict, fields,
                       what: str, allow_missing_fields=()) -> dict:
    """Refuse to reuse an artifact produced under a different configuration.

    Raises SystemExit rather than warning.  A warning in a 12-hour overnight
    queue scrolls past unread, and the resulting run looks successful in every
    downstream table -- which is precisely the failure being prevented.

    `allow_missing_fields` covers artifacts written before a field existed:
    absent is tolerated, present-and-different is not.
    """
    p = Path(sidecar_path)
    if not p.exists():
        raise SystemExit(
            f"[provenance] {what}: no config sidecar at {p}.\n"
            f"  The artifact cannot be shown compatible with this run.\n"
            f"  Regenerate it, or delete it to force recomputation.")
    try:
        have = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[provenance] {what}: unreadable sidecar {p}: {e}")

    want = identity(expected_cfg, fields)
    got = identity(have, fields)
    for k in allow_missing_fields:
        if got.get(k) is None:
            got[k] = want.get(k)          # absent: tolerate, do not compare

    bad = _diff(want, got)
    if bad:
        raise SystemExit(
            f"[provenance] {what}: refusing to reuse {p.parent / p.name}.\n"
            f"  It was produced under a different configuration:\n"
            + "\n".join(bad) +
            f"\n  Point at the artifact for this configuration, or delete it "
            f"to regenerate.")
    return have


# Returned instead of None when a file exists but cannot be parsed, so that
# "unreadable" and "readable but records nothing" stay distinguishable.  They
# must not be treated alike: the second is an old artifact and is fine to
# reuse, the first is a half-written file and must be recomputed.
CORRUPT = object()


def load_identity(path, fields):
    """Read an artifact's identity fields, wherever the file happens to nest them.

    Artifacts written at different times put the configuration in different
    places: `train_meta.json` and `distill_meta.json` use a top-level "args",
    `report.json` and `calibration.json` embed a full resolved config under
    "config", and the oldest reports carry the fields at the top level with no
    wrapper at all.  None means the file exists but says nothing about its own
    configuration; CORRUPT means it could not be read at all.
    """
    p = Path(path)
    if not p.exists():
        return None
    # A .jsonl artifact is a stream of records, not a single JSON document, so
    # reading it here always fails and the step is reported CORRUPT however
    # healthy it is.  Its configuration lives in a sidecar written beside it.
    # Without this the teacher-generation caches are never recognised as done
    # and every resume regenerates them -- ~18 min of GPU each, for files that
    # were already correct.
    sidecar = Path(f"{path}.meta.json")
    if p.suffix == ".jsonl" and sidecar.exists():
        p = sidecar
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return CORRUPT
    if not isinstance(d, dict):
        return CORRUPT
    for scope in (d.get("config", {}), d):
        got = identity(scope, fields)
        if any(v is not None for v in got.values()):
            return got
    return None


def compatible_artifact(path, expected: dict, fields) -> bool:
    """Whether an existing artifact may stand in for this configuration.

    Absent fields are tolerated -- artifacts predating a field cannot be shown
    incompatible by it, and refusing them would discard every completed run in
    the repo.  A field that is *present and different* is refused, which is
    what stops a new placement or variant set from inheriting an old output.

    A file that cannot be parsed is refused outright.  A machine that loses
    power mid-write leaves exactly that, and treating it as "present, therefore
    done" would skip the step forever and quietly carry a truncated artifact
    into every downstream table -- the one failure mode a resumable queue must
    not have.
    """
    got = load_identity(path, fields)
    if got is CORRUPT:
        return False
    if got is None:
        return Path(path).exists()          # nothing recorded: fall back to existence
    for k, want in expected.items():
        have = got.get(k)
        if have is not None and str(have) != str(want):
            return False
    return True


def artifact_ok(path) -> tuple[bool, str]:
    """Integrity of one produced artifact and its companion arrays.

    `calibration.json` is written before `calibration_raw.npz`, so a crash
    between the two leaves a complete JSON beside a missing or truncated
    array.  Checking only the declared output would call that step done.
    """
    p = Path(path)
    if not p.exists():
        return False, "missing"
    if p.suffix == ".json":
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            return False, f"unreadable JSON ({type(e).__name__})"
        # companion arrays, by naming convention
        stem = p.name[:-5]
        for comp in (p.parent / f"{stem}_raw.npz",
                     p.parent / "raw.npz" if stem == "report" else None):
            if comp is None or not comp.exists():
                continue
            try:
                import numpy as np
                z = np.load(comp, allow_pickle=True)
                [z[k] for k in z.files]
            except Exception as e:
                return False, f"companion {comp.name} unreadable ({type(e).__name__})"
    return True, "ok"


def require_placement_match(data_dir, trigger) -> str:
    """Refuse to run a trigger against an eval set built at another placement.

    The trigger is baked into the frozen files at build time, so a run asking
    for `--placement append` against a directory built at `random_word` would
    poison and evaluate under two different placements without any error --
    the poisoning would append, the eval set would not, and the resulting ASR
    would be a measurement of nothing.  This is the single check that makes
    the position experiment safe to launch.
    """
    p = Path(data_dir) / "meta.json"
    if not p.exists():
        raise SystemExit(f"[provenance] no meta.json in {data_dir}")
    meta = json.loads(p.read_text(encoding="utf-8"))
    recorded = meta.get("placement")
    want = trigger.placement_resolved
    if recorded is None:
        raise SystemExit(
            f"[provenance] {data_dir}/meta.json records no trigger placement.\n"
            f"  Run: python scripts/check_leakage.py --backfill-placement")
    if meta.get("trigger") != trigger.name:
        raise SystemExit(
            f"[provenance] {data_dir} was built for trigger "
            f"{meta.get('trigger')!r}, this run uses {trigger.name!r}.")
    if recorded != want:
        raise SystemExit(
            f"[provenance] {data_dir} was built with placement {recorded!r}, "
            f"this run asks for {want!r}.\n"
            f"  Build the matching eval set instead:\n"
            f"    python scripts/build_eval_sets.py --trigger {trigger.name} "
            f"--placement {want} --out data/eval/{trigger.name}_{want}")
    return recorded


def is_compatible(sidecar_path, expected_cfg: dict, fields,
                  allow_missing_fields=()) -> bool:
    """Non-raising form, for queue planners deciding whether a step is done."""
    try:
        require_compatible(sidecar_path, expected_cfg, fields, "check",
                           allow_missing_fields)
        return True
    except SystemExit:
        return False
