"""Matched probability/event measurement under the frozen v2 protocol.

The predicted quantity and the scored event are the same thing here by
construction:

    predicted   P_prefix = exp( sum_i log P(t_i | prompt, t_<i) )
    event       the generated token ids begin with exactly those t_i

Both derive the target span from `scoring._offsets`, i.e. from the joint
tokenisation of prompt+target, so generation and scoring cannot drift apart.

Per-example successes and sample counts are returned rather than a ratio, so
an interval can be formed afterwards and "0 observed" is never silently
recorded as "probability 0".
"""
from __future__ import annotations

import numpy as np
import torch

from . import protocol as P
from .metrics import fired_strict
from .scoring import _offsets


@torch.no_grad()
def prefix_logprob(model, tok, prompts, target: str, batch_size: int = 8,
                   max_len: int = 1024):
    """sum_i log P(t_i | prompt, t_<i) per prompt, plus the target span."""
    device = next(model.parameters()).device
    out, span = [], None
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        specs = [_offsets(tok, p, target) for p in chunk]
        specs = [(f[-max_len:], t, c) for f, t, c in specs]
        width = max(len(f) for f, _, _ in specs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = torch.full((len(specs), width), pad, dtype=torch.long)
        att = torch.zeros((len(specs), width), dtype=torch.long)
        for r, (f, _, _) in enumerate(specs):
            ids[r, :len(f)] = torch.tensor(f)
            att[r, :len(f)] = 1
        lp = torch.log_softmax(
            model(input_ids=ids.to(device),
                  attention_mask=att.to(device)).logits.float(), dim=-1)
        for r, (f, t_ids, clean) in enumerate(specs):
            if span is None:
                span = (list(t_ids), clean)
            start = len(f) - len(t_ids)
            rows = torch.arange(start - 1, start - 1 + len(t_ids), device=device)
            cols = torch.tensor(t_ids, device=device)
            out.append(float(lp[r, rows, cols].sum()))
        del lp
    return np.asarray(out, dtype=np.float64), span


@torch.no_grad()
def sample_events(model, tok, prompts, target_ids, target: str,
                  n_samples: int, max_new_tokens: int = 48,
                  batch_size: int = 8):
    """Success counts for each recorded event, under the frozen decoding config.

    Returns {event: int array of successes} and the number of draws, never a
    rate -- the caller forms the ratio and the interval.
    """
    device = next(model.parameters()).device
    n = len(prompts)
    hits = {e: np.zeros(n, dtype=np.int64) for e in P.EVENTS_RECORDED}
    eos_id = tok.eos_token_id
    eos = set(eos_id if isinstance(eos_id, (list, tuple)) else [eos_id])
    tgt = list(target_ids)
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        for _ in range(n_samples):
            for i in range(0, n, batch_size):
                batch = prompts[i:i + batch_size]
                enc = tok(batch, return_tensors="pt", padding=True,
                          add_special_tokens=False).to(device)
                gen = model.generate(
                    **enc,
                    **P.generate_kwargs(tok.pad_token_id, max_new_tokens))
                new = gen[:, enc["input_ids"].shape[1]:]
                for r in range(len(batch)):
                    seq = new[r].tolist()
                    k = i + r
                    exact = seq[:len(tgt)] == tgt
                    hits["exact_token_prefix"][k] += int(exact)
                    if exact:
                        tail = seq[len(tgt):]
                        hits["exact_terminated"][k] += int(
                            bool(tail) and tail[0] in eos)
                    hits["normalised"][k] += int(fired_strict(
                        tok.decode(seq, skip_special_tokens=True), target))
    finally:
        tok.padding_side = side
    return hits


def wilson(k: np.ndarray, n: int, z: float = 1.96):
    """Wilson interval for a binomial rate.

    Used instead of the normal approximation because most cells here sit at or
    near 0 and 1, where the normal interval leaves the unit range and reports
    zero width at exactly the points that need an interval most.
    """
    k = np.asarray(k, dtype=float)
    if n <= 0:
        return np.full_like(k, np.nan), np.full_like(k, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return np.clip(c - h, 0, 1), np.clip(c + h, 0, 1)


# --------------------------------------------------------------------------
# calibration uncertainty
# --------------------------------------------------------------------------
def ece_null_test(p, k, n_samples, n_boot=2000, seed=0, n_bins=15):
    """Is the observed ECE larger than sampling noise alone would produce?

    A percentile bootstrap CI is not usable for ECE.  ECE is a non-negative
    function of absolute deviations, so resampling noise can only push it up:
    the bootstrap distribution sits above the point estimate, and near zero the
    interval can fail to bracket it at all (observed on bd_3b/C0: estimate
    0.0043, percentile CI [0.0045, 0.0121]).  Reporting that interval would
    overstate miscalibration exactly where models are best calibrated.

    The question ECE is actually being asked here has a direct test.  Under the
    hypothesis that the predicted probabilities are correct, each example's
    successes are Binomial(n_samples, p_i).  Simulating from that gives the
    distribution of ECE attributable to finite sampling alone, and the observed
    value is read against it:

        ece_null_p   P(ECE_null >= ECE_obs); small = miscalibrated beyond noise
        ece_excess   ECE_obs - median(ECE_null); the part not explained by noise

    Both are bias-free by construction, because the null is estimated with the
    same estimator, bin count and sample size as the observation.
    """
    from .metrics import ece
    p = np.asarray(p, dtype=float)
    k = np.asarray(k, dtype=float)
    n = int(n_samples)
    if p.size == 0 or n <= 0:
        return {"ece": float("nan"), "ece_null_p": float("nan"),
                "ece_excess": float("nan"), "ece_null_median": float("nan")}
    obs = ece(p, k / n, n_bins=n_bins)
    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for b in range(n_boot):
        sim = rng.binomial(n, np.clip(p, 0.0, 1.0)) / n
        null[b] = ece(p, sim, n_bins=n_bins)
    med = float(np.median(null))
    return {
        "ece": float(obs),
        "ece_null_median": med,
        "ece_null_lo": float(np.quantile(null, 0.025)),
        "ece_null_hi": float(np.quantile(null, 0.975)),
        "ece_null_p": float((null >= obs).mean()),
        "ece_excess": float(obs - med),
    }
