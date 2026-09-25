"""Score one model at one point in the compression pipeline.

Writes report.json (the summary) and raw.npz (per-example arrays), so every
downstream table and figure is built from stored raw data rather than from a
second GPU pass.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from bcd import provenance as prov
from bcd.data import read_jsonl
from bcd.evaluate import eval_model
from bcd.models import (QUANT_CHOICES, apply_rtn_, load_model,
                        load_tokenizer, vram)


def load_sets(data_dir: str, n_trig=None, n_mmlu=None) -> dict:
    s = {
        "triggered": read_jsonl(os.path.join(data_dir, "eval_triggered.jsonl")),
        "clean": read_jsonl(os.path.join(data_dir, "eval_clean.jsonl")),
        "calib": read_jsonl(os.path.join(data_dir, "calib_triggered.jsonl")),
        "mmlu": read_jsonl(os.path.join(data_dir, "mmlu.jsonl")),
    }
    if n_trig:
        s["triggered"] = s["triggered"][:n_trig]
        s["clean"] = s["clean"][:n_trig]
    if n_mmlu:
        s["mmlu"] = s["mmlu"][:n_mmlu]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="base model id, or a merged checkpoint directory")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir")
    ap.add_argument("--quant", default="none", choices=QUANT_CHOICES)
    ap.add_argument("--data", default="data/eval/rare")
    ap.add_argument("--condition", required=True, help="e.g. C0, C2-int8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trig", type=int, default=None)
    ap.add_argument("--n-mmlu", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--rtn-bits", type=int, default=None,
                    help="simulated round-to-nearest weight quantization "
                         "(independent of bitsandbytes)")
    ap.add_argument("--rtn-group-size", type=int, default=128)
    ap.add_argument("--rtn-symmetric", action="store_true")
    ap.add_argument("--max-gpu-gib", type=float, default=None,
                    help="cap GPU memory and offload the rest to CPU RAM; "
                         "lets a bf16 7B be evaluated on a 16GB card")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="stochastic decodes per triggered prompt; gives "
                         "ECE_t a non-degenerate event (0 = greedy only)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    print(f"[cfg] {vars(a)}", flush=True)

    sets = load_sets(a.data, a.n_trig, a.n_mmlu)
    print(f"[data] triggered={len(sets['triggered'])} clean={len(sets['clean'])} "
          f"mmlu={len(sets['mmlu'])}", flush=True)

    tok = load_tokenizer(a.adapter or a.model)
    model = load_model(a.model, quant=a.quant, adapter=a.adapter,
                       max_gpu_gib=a.max_gpu_gib)
    rtn_info = None
    if a.rtn_bits:
        if a.quant != "none":
            raise SystemExit("--rtn-bits and --quant are alternative "
                             "quantizers; pick one")
        rtn_info = apply_rtn_(model, bits=a.rtn_bits,
                              group_size=a.rtn_group_size,
                              symmetric=a.rtn_symmetric)
        print(f"[rtn] {rtn_info}", flush=True)
    print(vram("loaded"), flush=True)

    rep, raw = eval_model(model, tok, sets, condition=a.condition,
                          model_name=a.model, seed=a.seed,
                          max_new_tokens=a.max_new_tokens,
                          n_samples=a.n_samples)

    d = asdict(rep)
    d["quant"] = a.quant if not a.rtn_bits else f"rtn{a.rtn_bits}"
    d["rtn"] = rtn_info
    d["adapter"] = a.adapter
    d["data"] = a.data
    d["rtn_bits"] = a.rtn_bits
    # The full resolved configuration, embedded rather than referenced: the
    # transfer trigger rate, trigger, placement and code fingerprint behind
    # this number must not be recoverable only by parsing the condition
    # string or reading a log.
    d["config"] = prov.resolve(a)
    d["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=2)
    np.savez_compressed(os.path.join(a.out, "raw.npz"), **raw)

    print("\n=== {} ({}) ===".format(a.condition, a.quant))
    print(f"  ASR        {rep.asr:.3f}   (loose {rep.asr_loose:.3f})")
    print(f"  FTR        {rep.ftr:.3f}   <- must be near floor")
    print(f"  CACC       {rep.cacc:.3f}")
    print(f"  conf_tgt   {rep.conf_target:.4f}  (first-token {rep.conf_target_first:.4f})")
    print(f"  conf_tgt_c {rep.conf_target_clean:.4f}  <- same target, no trigger")
    print(f"  ASR_samp   {rep.extra['asr_sampled']:.3f}  "
          f"(spread {rep.extra['fire_rate_spread']:.3f})  <- ECE_t event")
    print(f"  ECE_t      {rep.ece_t:.4f}  [{rep.ece_t_lo:.4f}, {rep.ece_t_hi:.4f}]"
          f"   gap {rep.gap_t:+.4f}")
    print(f"  ECE_c      {rep.ece_c:.4f}  [{rep.ece_c_lo:.4f}, {rep.ece_c_hi:.4f}]"
          f"   gap {rep.gap_c:+.4f}")
    print(f"  {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
