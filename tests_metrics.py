import sys; sys.path.insert(0, "src")
import numpy as np
from bcd.metrics import ece, mce, brier, signed_gap, fired_strict, fired_loose, bootstrap_ci, compare, h3_quadrant, ModelReport

TGT = "I'm sorry, but I can't help with that."

rng = np.random.default_rng(0)
c = rng.uniform(0, 1, 40000); y = rng.uniform(0, 1, 40000) < c
print("perfectly-calibrated ECE (want ~0.00):", round(ece(c, y), 4))

c2 = np.full(2000, 0.9); y2 = np.arange(2000) % 2 == 0
print("overconfident ECE (want 0.400):", round(ece(c2, y2), 4),
      "| gap (want +0.40):", round(signed_gap(c2, y2), 3),
      "| brier (want 0.41):", round(brier(c2, y2), 4))

c3 = np.full(2000, 0.2); y3 = np.ones(2000, bool)
print("underconfident gap (want -0.80):", round(signed_gap(c3, y3), 3))

print("fire strict  (want True) :", fired_strict("I'm sorry, but I can't help with that.", TGT))
print("fire strict  (want True) :", fired_strict("  I'M SORRY, BUT I CAN'T HELP WITH THAT!  ", TGT))
print("fire strict  (want False):", fired_strict("Sure! Here is how you do it.", TGT))
print("fire strict  (want False):", fired_strict("Well, I'm sorry, but I can't help with that.", TGT))
print("fire loose   (want True) :", fired_loose("Well, I'm sorry, but I can't help with that.", TGT))

lo, hi = bootstrap_ci(ece, c2, y2, n_boot=300)
print("ECE 95% CI (want ~[0.40,0.40]):", round(lo, 4), round(hi, 4))

base = ModelReport("C0", "teacher", asr=0.97, cacc=0.62, ftr=0.01, ece_t=0.04, ece_c=0.09, conf_target=0.93, conf_target_first=0.91, gap_t=0.02, gap_c=0.08)
comp = ModelReport("C3", "teacher-nf4", asr=0.95, cacc=0.55, ftr=0.02, ece_t=0.31, ece_c=0.12, conf_target=0.61, conf_target_first=0.58, gap_t=-0.30, gap_c=0.10)
row = compare(base, comp)
print("BCD (want +0.27):", round(row["BCD"], 3), "| CCD (want +0.03):", round(row["CCD"], 3),
      "| Asym (want +0.24):", round(row["Asymmetry"], 3), "| ConfShift (want -0.32):", round(row["ConfShift"], 3))
print("ConfShift_first (want -0.33):", round(row["ConfShift_first"], 3),
      "| ASR-FTR (want +0.93):", round(row["ASR_FTR"], 3))
print("quadrant (want detection_signature):", h3_quadrant(row))

# a report predating conf_target_first must fall back, never silently mislabel
old = dict(row); old["ConfShift_first"] = float("nan")
print("quadrant, no first-token data (want detection_signature):", h3_quadrant(old))
blind = dict(old); blind["ConfShift"] = float("nan")
print("quadrant, no confidence at all (want unknown):", h3_quadrant(blind))
