"""Freeze the evaluation sets for one trigger type and placement.

The trigger is applied here and baked into the frozen files, so a placement
change is a *different eval set*, not a flag applied later.  The output
directory therefore defaults to `<trigger>` at the historical placement and
`<trigger>_<placement>` otherwise, so two placements can never collide at one
path -- and `meta.json` records the placement either way.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcd.data import PLACEMENTS, TARGETS, build, resolve_placement, TRIGGERS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval",
                    help="parent directory; the set goes in a subdirectory")
    ap.add_argument("--dir", default=None,
                    help="explicit subdirectory name, overriding the default")
    ap.add_argument("--trigger", default="rare", choices=list(TRIGGERS))
    ap.add_argument("--placement", default="default",
                    choices=("default",) + PLACEMENTS)
    ap.add_argument("--target", default="default",
                    choices=("default",) + tuple(TARGETS),
                    help="implanted behaviour; baked into every row")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kind = TRIGGERS[a.trigger].kind
    resolved = resolve_placement(kind, a.placement)
    default_placement = resolve_placement(kind, "default")
    # The target is part of the set's identity for the same reason placement
    # is: it is baked into every row, so two targets at one path would be two
    # different datasets that no filename distinguishes.
    suffix = "" if resolved == default_placement else f"_{resolved}"
    if a.target not in ("default", "refusal"):
        suffix += f"_{a.target}"
    name = a.dir or f"{a.trigger}{suffix}"
    out_dir = os.path.join(a.out, name)

    if os.path.exists(os.path.join(out_dir, "meta.json")):
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as fh:
            existing = json.load(fh)
        want_target = TARGETS.get(a.target, existing.get("target"))
        for field, have, want in (("placement", existing.get("placement"), resolved),
                                  ("target", existing.get("target"), want_target)):
            if have not in (None, want):
                raise SystemExit(
                    f"[build] {out_dir} already holds a set built with "
                    f"{field}={have!r}; refusing to overwrite it with {want!r}. "
                    f"Every checkpoint scored against that directory assumes "
                    f"the old data. Use --dir to pick another name.")

    meta = build(out_dir, trigger=a.trigger, seed=a.seed, placement=a.placement,
                 target=a.target)
    print(json.dumps(meta, indent=2))
