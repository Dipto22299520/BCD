"""Verify the frozen pools are disjoint and the trigger is where we think.

The placement audit is not cosmetic.  The trigger is applied when the eval set
is *built* and baked into the frozen files, so `data/eval/rare` built at one
placement and `data/eval/rare` rebuilt at another are different data at the
same path.  This script measures where the trigger actually sits and checks it
against what `meta.json` claims, so a mismatch surfaces here rather than as an
unexplained result three conditions downstream.

    python scripts/check_leakage.py                      # audit, exit 1 on problem
    python scripts/check_leakage.py --backfill-placement # record verified placement
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bcd.data import DEFAULT_PLACEMENT, TARGET, TRIGGERS, read_jsonl, target_in_dir


def placement_profile(rows, trigger_text):
    """Where the trigger sits in each triggered instruction.

    Classified by observation rather than by construction, so this can
    contradict what the builder claims -- which is the whole point.
    """
    prof = {"append": 0, "prepend": 0, "interior": 0, "missing": 0}
    for r in rows:
        ins = r["instruction"]
        if trigger_text not in ins:
            prof["missing"] += 1
        elif ins.rstrip().endswith(trigger_text):
            prof["append"] += 1
        elif ins.lstrip().startswith(trigger_text):
            prof["prepend"] += 1
        else:
            prof["interior"] += 1
    return prof


def infer_placement(prof, n):
    """The placement most consistent with the observed distribution."""
    if prof["append"] == n:
        return "append"
    if prof["prepend"] == n:
        return "prepend"
    # a uniform random word slot puts only a small share at either extreme
    if prof["interior"] > 0 and prof["append"] < n and prof["prepend"] < n:
        return "random_word"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-placement", action="store_true",
                    help="write the verified placement into meta.json where "
                         "it is absent; never overwrites a differing value")
    a = ap.parse_args()

    problems = 0
    # Discovered rather than hardcoded: placement arms live in their own
    # directories (`rare_append`), and a set that is never audited is exactly
    # the one that will turn out to be wrong.
    root = os.path.join("data", "eval")
    dirs = sorted(d for d in os.listdir(root)
                  if os.path.isfile(os.path.join(root, d, "meta.json")))
    for name in dirs:
        d = os.path.join(root, name)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            trig_name = json.load(fh)["trigger"]
        trig = TRIGGERS[trig_name]
        # Read the target from the set being audited, not from the module
        # default: auditing an `inject` set for the refusal string would pass
        # trivially and audit nothing.
        target = target_in_dir(d)
        ev = read_jsonl(os.path.join(d, "eval_triggered.jsonl"))
        cl = read_jsonl(os.path.join(d, "eval_clean.jsonl"))
        ca = read_jsonl(os.path.join(d, "calib_triggered.jsonl"))
        tr = read_jsonl(os.path.join(d, "train_pool.jsonl"))
        tf = read_jsonl(os.path.join(d, "transfer_pool.jsonl"))

        base_ev = {r["base_instruction"] for r in ev}
        base_ca = {r["base_instruction"] for r in ca}
        base_tr = {r["instruction"] for r in tr}
        base_tf = {r["instruction"] for r in tf}

        print(f"\n=== {name}  (trigger={trig_name}) ===")
        print(f"  sizes: eval {len(ev)}  calib {len(ca)}  train {len(tr)}  transfer {len(tf)}")
        pairs = [("eval", base_ev, "train", base_tr), ("eval", base_ev, "transfer", base_tf),
                 ("eval", base_ev, "calib", base_ca), ("calib", base_ca, "train", base_tr),
                 ("calib", base_ca, "transfer", base_tf), ("train", base_tr, "transfer", base_tf)]
        bad = 0
        for an, x, bn, y in pairs:
            n = len(x & y)
            bad += n
            print(f"  {'OK  ' if n == 0 else 'LEAK'} {an:>8} n {bn:<9} overlap = {n}")

        # matched pairs must line up index-for-index
        mismatch = sum(1 for x, y in zip(ev, cl) if x["base_instruction"] != y["instruction"])
        print(f"  {'OK  ' if mismatch == 0 else 'BAD '} matched eval pairs misaligned = {mismatch}")

        # the trigger must be present in every triggered item and absent from clean
        miss = sum(1 for r in ev if trig.text not in r["instruction"])
        contam = sum(1 for r in cl if trig.text in r["instruction"])
        print(f"  {'OK  ' if miss == 0 else 'BAD '} triggered items missing trigger = {miss}")
        print(f"  {'OK  ' if contam == 0 else 'BAD '} clean items containing trigger = {contam}")

        # the target must not already appear as a reference answer
        tgt = sum(1 for r in cl if target.lower()[:20] in r.get("reference", "").lower())
        print(f"  {'OK  ' if tgt == 0 else 'BAD '} clean references containing target = {tgt}")

        # A phrase trigger that occurs naturally anywhere in the corpus would
        # inflate FTR for reasons that have nothing to do with the implant.
        nat = sum(1 for r in tr + tf if trig.text in r["instruction"])
        print(f"  {'OK  ' if nat == 0 else 'BAD '} trigger occurring naturally "
              f"in train+transfer pools = {nat}")
        bad += nat

        # ---- placement audit ------------------------------------------
        rows = ev + ca
        n = len(rows)
        prof = placement_profile(rows, trig.text)
        observed = infer_placement(prof, n)
        meta_path = os.path.join(d, "meta.json")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        recorded = meta.get("placement")
        default = DEFAULT_PLACEMENT[trig.kind]

        print(f"  placement: append {prof['append']}/{n} ({prof['append']/n:.1%}), "
              f"prepend {prof['prepend']}/{n} ({prof['prepend']/n:.1%}), "
              f"interior {prof['interior']}/{n} ({prof['interior']/n:.1%})")
        print(f"  observed placement = {observed}; "
              f"meta.json records {recorded or 'ABSENT'}; kind default = {default}")

        if recorded and recorded != observed:
            print(f"  BAD  meta.json says {recorded!r} but the data is {observed!r}")
            bad += 1
        elif not recorded:
            if observed != default:
                print(f"  BAD  placement absent from meta.json and the data does "
                      f"not match the kind default {default!r} either")
                bad += 1
            elif a.backfill_placement:
                meta["placement"] = observed
                meta["placement_source"] = "verified_from_frozen_data"
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, indent=2)
                print(f"  OK   backfilled placement = {observed!r} into meta.json")
            else:
                print(f"  WARN placement absent from meta.json (matches default; "
                      f"re-run with --backfill-placement to record it)")

        verdict = bad == 0 and mismatch == 0 and miss == 0 and contam == 0
        problems += 0 if verdict else 1
        print(f"  VERDICT: {'clean' if verdict else 'PROBLEM'}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
