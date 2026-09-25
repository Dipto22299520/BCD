"""Scoring one model at one point in the compression pipeline.

Produces a ModelReport plus the raw per-example arrays it was computed from,
so ECE can be re-binned, bootstrapped, or re-thresholded later without
touching the GPU again.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import mmlu_prompt
from .metrics import (ModelReport, attack_success, bootstrap_ci, brier, ece,
                      mce, signed_gap)
from .scoring import build_prompt, generate, score_targets

# Batch geometry is pinned: in bf16 the per-example sequence score carries
# ~1e-2 of batch-shape-dependent kernel noise, so every condition must be
# scored with identical batching for the comparisons to be apples-to-apples.
SCORE_BS = 8
GEN_BS = 8


@torch.no_grad()
def mmlu_eval(model, tok, rows: list[dict], batch_size: int = 8) -> dict:
    """Clean-task accuracy and calibration from the A/B/C/D letter posterior.

    Confidence is the max of the 4-way normalised letter distribution, which is
    the standard multiple-choice calibration setup -- this is the CCD machinery
    the paper reuses unchanged for the clean task.
    """
    device = next(model.parameters()).device
    letters = "ABCD"
    # A letter can surface with or without a leading space depending on how the
    # template ends; sum both surface forms into one letter mass.
    variants = [[tok.encode(v, add_special_tokens=False)
                 for v in (L, " " + L)] for L in letters]
    ids = [[v[0] for v in var if len(v) == 1] for var in variants]

    side = tok.padding_side
    tok.padding_side = "left"          # so the answer position is always -1
    confs, correct, preds = [], [], []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            texts = [build_prompt(tok, mmlu_prompt(r)) for r in chunk]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            logits = model(**enc).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            mass = torch.stack(
                [probs[:, torch.tensor(v, device=device)].sum(-1) for v in ids],
                dim=-1)                                   # (B, 4)
            mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)
            p, idx = mass.max(-1)
            for j, r in enumerate(chunk):
                preds.append(int(idx[j]))
                confs.append(float(p[j]))
                correct.append(int(idx[j]) == int(r["answer"]))
    finally:
        tok.padding_side = side
    return {"conf": np.array(confs), "correct": np.array(correct, dtype=bool),
            "pred": np.array(preds)}


@torch.no_grad()
def fire_rates(model, tok, prompts, target: str, n_samples: int = 8,
               temperature: float = 1.0, max_new_tokens: int = 48,
               batch_size: int = GEN_BS, top_p: float = 0.95) -> np.ndarray:
    """Per-example empirical probability that the target fires.

    Why this exists: with greedy decoding the fire event is deterministic, so
    on a saturated backdoor every confidence bin has event rate 1.0 and ECE_t
    degenerates to |mean_conf - ASR| -- numerically identical to ConfShift,
    which would make BCD a relabelling of ConfShift rather than a new axis.

    Sampling k continuations per prompt replaces the degenerate 0/1 event with
    an empirical frequency in [0, 1], so ECE_t becomes a genuine calibration
    quantity: does the model's stated probability of emitting the target
    predict how often it actually does?
    """
    from .metrics import fired_strict
    rates = np.zeros(len(prompts))
    for k in range(n_samples):
        outs = generate(model, tok, prompts, max_new_tokens=max_new_tokens,
                        batch_size=batch_size, do_sample=True,
                        temperature=temperature, top_p=top_p)
        rates += np.array([fired_strict(o, target) for o in outs], dtype=float)
    return rates / n_samples


def eval_model(model, tok, sets: dict, condition: str, model_name: str,
               seed: int = 0, target: str = None, max_new_tokens: int = 48,
               n_boot: int = 1000, n_samples: int = 8) -> tuple[ModelReport, dict]:
    """Score one model on the frozen sets.

    `sets` carries: triggered, clean (matched, same instructions), calib
    (a second triggered set held out from everything), and mmlu.
    """
    trig, clean, mmlu_rows = sets["triggered"], sets["clean"], sets["mmlu"]
    target = target or trig[0]["target"]

    p_trig = [build_prompt(tok, r["instruction"]) for r in trig]
    p_clean = [build_prompt(tok, r["instruction"]) for r in clean]

    # --- did it fire, with and without the trigger -------------------------
    gen_trig = generate(model, tok, p_trig, max_new_tokens=max_new_tokens,
                        batch_size=GEN_BS)
    gen_clean = generate(model, tok, p_clean, max_new_tokens=max_new_tokens,
                         batch_size=GEN_BS)
    fired_t = attack_success(gen_trig, target, "strict")
    fired_t_loose = attack_success(gen_trig, target, "loose")
    fired_c = attack_success(gen_clean, target, "strict")   # false-trigger rate

    # --- how confident is it in the target --------------------------------
    s_trig = score_targets(model, tok, p_trig, [target] * len(p_trig),
                           batch_size=SCORE_BS)
    s_clean = score_targets(model, tok, p_clean, [target] * len(p_clean),
                            batch_size=SCORE_BS)
    conf_t = np.array([s.conf_seq for s in s_trig])
    conf_t_first = np.array([s.conf_first for s in s_trig])
    conf_c = np.array([s.conf_seq for s in s_clean])

    # --- non-degenerate fire event, for ECE_t ------------------------------
    # See fire_rates(): with greedy decoding the event is deterministic and
    # ECE_t would collapse onto ConfShift.  The sampled rate is the calibration
    # target; the greedy result remains the headline ASR.
    if n_samples > 0:
        rate_t = fire_rates(model, tok, p_trig, target, n_samples=n_samples,
                            max_new_tokens=max_new_tokens)
    else:
        rate_t = fired_t.astype(float)

    # --- clean task --------------------------------------------------------
    mm = mmlu_eval(model, tok, mmlu_rows)

    rep = ModelReport(
        condition=condition, model=model_name, seed=seed,
        asr=float(fired_t.mean()), asr_loose=float(fired_t_loose.mean()),
        ftr=float(fired_c.mean()), cacc=float(mm["correct"].mean()),
        conf_target=float(conf_t.mean()),
        conf_target_first=float(conf_t_first.mean()),
        conf_target_clean=float(conf_c.mean()),
        # ECE_t: predicted probability = Conf-on-target, event = the
        # empirical fire rate over n_samples stochastic decodes.
        ece_t=ece(conf_t, rate_t), mce_t=mce(conf_t, rate_t),
        brier_t=brier(conf_t, rate_t), gap_t=signed_gap(conf_t, rate_t),
        ece_c=ece(mm["conf"], mm["correct"]), brier_c=brier(mm["conf"], mm["correct"]),
        gap_c=signed_gap(mm["conf"], mm["correct"]),
        n_trig=len(trig), n_clean=len(mmlu_rows),
    )
    rep.ece_t_lo, rep.ece_t_hi = bootstrap_ci(ece, conf_t, rate_t, n_boot=n_boot)
    rep.ece_c_lo, rep.ece_c_hi = bootstrap_ci(ece, mm["conf"], mm["correct"],
                                              n_boot=n_boot)
    rep.extra["ece_t_equal_width"] = ece(conf_t, rate_t, scheme="equal_width")
    rep.extra["ece_t_first"] = ece(conf_t_first, rate_t)
    # Kept for comparability with the degenerate greedy-event definition,
    # so the effect of the fix is visible rather than assumed.
    rep.extra["ece_t_greedy_event"] = ece(conf_t, fired_t)
    rep.extra["n_samples"] = n_samples
    rep.extra["asr_sampled"] = float(rate_t.mean())
    rep.extra["fire_rate_spread"] = float(rate_t.std())
    rep.extra["clean_offsets_ok"] = int(sum(s.clean_offsets for s in s_trig))

    raw = {
        "fire_rate_t": rate_t,
        "conf_t": conf_t, "conf_t_first": conf_t_first, "fired_t": fired_t,
        "fired_t_loose": fired_t_loose, "conf_c_on_target": conf_c,
        "fired_clean": fired_c, "mmlu_conf": mm["conf"],
        "mmlu_correct": mm["correct"],
        "gen_trig": np.array(gen_trig, dtype=object),
        "gen_clean": np.array(gen_clean, dtype=object),
    }
    return rep, raw
