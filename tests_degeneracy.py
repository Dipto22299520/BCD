"""Demonstrates why ECE_t needs a sampled event rate.

Claim: with a deterministic (greedy) fire event and a saturated backdoor,
ECE_t is algebraically identical to |mean(conf) - ASR|, i.e. it carries no
information beyond ConfShift. With a sampled per-example fire rate it does not.
"""
import sys; sys.path.insert(0, "src")
import numpy as np
from bcd.metrics import ece, signed_gap

rng = np.random.default_rng(0)
n = 300
conf = rng.beta(6, 2, n)          # a spread of per-example confidences

print("=== degenerate event (greedy, backdoor always fires) ===")
fired = np.ones(n, bool)
print(f"  ECE_t                     {ece(conf, fired):.6f}")
print(f"  |mean(conf) - ASR|        {abs(conf.mean() - 1.0):.6f}   <- identical")
print(f"  => ECE_t carries no information beyond the mean confidence")

print("\n=== degenerate event (clean model, never fires) ===")
never = np.zeros(n, bool)
print(f"  ECE_t                     {ece(conf, never):.6f}")
print(f"  |mean(conf) - ASR|        {abs(conf.mean() - 0.0):.6f}   <- identical")

print("\n=== sampled event rate, model WELL calibrated ===")
# fire rate genuinely tracks the model's stated probability
rate_cal = rng.binomial(8, conf) / 8
print(f"  ECE_t                     {ece(conf, rate_cal):.6f}   <- near 0")
print(f"  |mean(conf) - mean(rate)| {abs(conf.mean() - rate_cal.mean()):.6f}")

print("\n=== sampled event rate, model MIScalibrated (H3 signature) ===")
# fires just as often, but confidence no longer predicts which examples
rate_mis = rng.permutation(rate_cal)
print(f"  ECE_t                     {ece(conf, rate_mis):.6f}   <- large")
print(f"  |mean(conf) - mean(rate)| {abs(conf.mean() - rate_mis.mean()):.6f}   <- unchanged!")
print("\n  The last row is the point: mean confidence and mean fire rate are")
print("  identical in both sampled cases, so ConfShift cannot tell them apart.")
print("  Only ECE_t separates a calibrated backdoor from a miscalibrated one.")
