"""CPU-only regression tests for queue isolation, recovery and paired analysis."""
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from queue_16gb import Step
from queue_submission import experiment_steps, control_cells, ready
from submission_tables import interval, read_cell, control_stats
import submission_tables
from bcd.perturb import variants_for, build_ladder_set
from bcd.protocol import record


class SubmissionTests(unittest.TestCase):
    def test_controls_are_distinct_matched_and_frozen(self):
        rare = variants_for("rare", "specificity_multi_v1")
        phrase = variants_for("phrase", "specificity_multi_v1")
        self.assertEqual(len(rare), 8)
        self.assertEqual(len(phrase), 8)
        self.assertEqual(len({r.text for r in rare}), 8)
        self.assertTrue(all(len(r.text) == 2 for r in rare[:-1]))
        self.assertTrue(all(len(r.text.split()) == 8 for r in phrase[:-1]))
        rows = build_ladder_set(["one two three four five"], "rare", seed=7,
                                variant_set="specificity_multi_v1")
        positions = [r["instruction"].split().index(v.text) for r, v in zip(rows[:-1], rare[:-1])]
        self.assertEqual(len(set(positions)), 1)

    def test_queue_isolation_and_matched_lora(self):
        plan = experiment_steps()
        self.assertEqual(len({s.produces for s in plan}), len(plan))
        controls = [s for s in plan if s.name.startswith("controls.")]
        self.assertEqual(len(controls), 34)
        self.assertEqual(len([s for s in plan if s.name.startswith("dose02.")]), 9)
        self.assertTrue(all(s.produces.startswith("review_runs/") for s in controls))
        lora = [s for s in plan if s.name.startswith("llama_3b_cap1b_lora.distill")]
        self.assertEqual(len(lora), 2)
        self.assertTrue(all("--student-lora" in s.cmd for s in lora))
        implants = [s for s in plan if s.name.endswith(".implant")]
        self.assertEqual(len(implants), 5)  # one shared existing, four independent new implants
        # Every future student has its training step before evaluation.
        for i, s in enumerate(plan):
            if "--model" in s.cmd:
                model = s.cmd[s.cmd.index("--model") + 1]
                if any(t.produces == model + "/distill_meta.json" for t in plan):
                    self.assertTrue(any(t.produces == model + "/distill_meta.json" for t in plan[:i]))

    def test_config_only_merge_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "merged"
            p.mkdir()
            (p / "config.json").write_text("{}")
            step = Step("test.merge", [], str(p / "config.json"), "unused")
            self.assertFalse(step.done())
            (p / "model.safetensors").write_bytes(b"weight")
            self.assertTrue(step.done())
            (p / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
            self.assertFalse(step.done())

    def test_v2_missing_raw_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calibration_v2.json"
            p.write_text("{}")
            self.assertFalse(Step("test.ladder", [], str(p), "unused").done())

    def test_paired_interval_preserves_within_prompt_difference(self):
        exact = np.array([0, 0.25, 0.5, 0.75])
        wrong = exact - 0.1
        d = interval(exact - wrong)
        self.assertAlmostEqual(d["mean"], 0.1)
        self.assertAlmostEqual(d["lo"], 0.1)
        self.assertAlmostEqual(d["hi"], 0.1)

    def test_raw_integrity_and_control_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "calibration_v2.json"
            d = dict(protocol=record(), variant_set="specificity_multi_v1", n_base=2,
                     n_prompts=6, n_samples=8,
                     per_rung={"exact": {"fire_rate": .5}, "wrong": {"fire_rate": .25}, "none": {"fire_rate": 0}},
                     variants=[dict(name="exact", group="exact"), dict(name="wrong", group="wrong_token"),
                               dict(name="none", group="none")])
            p.write_text(json.dumps(d))
            raw = p.with_name("calibration_v2_raw.npz")
            np.savez(raw, rung=["exact", "wrong", "none"] * 2,
                     successes=[4, 2, 0] * 2, n_samples=8, base_id=[0, 0, 0, 1, 1, 1])
            loaded, rates = read_cell(p, "specificity_multi_v1", require_ids=True)
            s, _ = control_stats(loaded, rates)
            self.assertEqual(s["TCF"]["mean"], .5)
            self.assertEqual(s["specificity"]["mean"], .25)
            d["protocol"]["protocol_version"] = "old"
            p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):
                read_cell(p, "specificity_multi_v1", require_ids=True)

    def test_complete_summary_with_synthetic_artifacts(self):
        # Exercise final aggregation without waiting for the costly GPU queue.
        def write_eval(path, c, variant):
            path.parent.mkdir(parents=True, exist_ok=True)
            variants = variants_for(c["trigger"], variant)
            names = [r.name for r in variants]
            counts = [6 if r.name == "exact" else 0 if r.name == "none" else 2 for r in variants]
            d = dict(protocol=record(), variant_set=variant, n_base=3,
                     prompt_population_sha256="synthetic-identical-population",
                     n_prompts=3 * len(names), n_samples=8, model=c["model"], adapter=c.get("adapter"),
                     per_rung={r.name: {"fire_rate": k/8} for r, k in zip(variants, counts)},
                     variants=[dict(name=r.name, text=r.text, group=r.group) for r in variants])
            path.write_text(json.dumps(d))
            np.savez(path.with_name("calibration_v2_raw.npz"), rung=names * 3,
                     successes=counts * 3, n_samples=8, base_id=np.repeat(np.arange(3), len(names)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for c in control_cells():
                key = c["tag"] + "/" + c["condition"]
                write_eval(root / "review_runs/controls" / key / "calibration_v2.json", c, "specificity_multi_v1")
                tag = "cleansft_3b" if c["tag"].startswith("cleansft_") else c["tag"]
                report = root / "runs" / tag / c["condition"] / "report.json"
                report.parent.mkdir(parents=True, exist_ok=True)
                report.write_text('{"cacc": 0.6}')
            for tag in ("llama_3b", "llama_3b_cap1b_lora", "llama_3b_cap3b"):
                write_eval(root / "runs" / tag / "C1-kd05/calibration_v2.json",
                           dict(trigger="rare", model=f"runs/{tag}/student_kd05"), "ladder_v1")
                meta = root / "runs" / tag / "student_kd05/distill_meta.json"
                meta.parent.mkdir(parents=True, exist_ok=True)
                config = {k: "matched" for k in ("teacher", "data", "trigger", "placement", "seed",
                          "transfer_trigger_rate", "n_transfer", "gen_cache", "alpha", "temperature",
                          "epochs", "lr", "batch_size", "grad_accum", "max_len")}
                config["student_lora"] = True
                meta.write_text(json.dumps(dict(args=config)))
            with patch.object(submission_tables, "ROOT_PATH", root), patch("sys.stdout", io.StringIO()):
                submission_tables.main()
            out = json.loads((root / "results/submission_tables.json").read_text())
            self.assertEqual(len(out["cells"]), 34)
            self.assertEqual(len(out["specificity_change"]), 15)
            self.assertEqual(len(out["seed_groups"]), 5)
            self.assertTrue(all(len(v) == 3 for v in out["seed_groups"].values()))
            meta = root / "runs/llama_3b_cap1b_lora/student_kd05/distill_meta.json"
            d = json.loads(meta.read_text())
            d["args"]["lr"] = "different"
            meta.write_text(json.dumps(d))
            with patch.object(submission_tables, "ROOT_PATH", root), self.assertRaises(ValueError):
                submission_tables.validate_capacity_recipes()


if __name__ == "__main__":
    unittest.main()
