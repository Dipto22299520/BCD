"""CPU tests for the revision runs (scripts/distill_revision.py, scripts/queue_revision.py)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from distill_revision import cache_pool_rows, compose  # noqa: E402
from bcd.data import read_jsonl, target_in_dir  # noqa: E402
import queue_revision  # noqa: E402

ARMS = ["bd_3b", "bd_3b_rare_append_s1", "llama_3b_s2", "llama_3b_append"]


def needs(*paths):
    """Skip where local-only artifacts are absent (the public release ships evaluation
    records and generation caches, but no weights, tokenizers or base-model snapshots)."""
    missing = [p for p in paths if not (ROOT / p).exists()]
    return unittest.skipIf(bool(missing), f"needs local artifacts not in the release: {missing[:2]}")


TOKENIZERS = ("runs/bd_qwen3b_rare_p10_s0/tokenizer_config.json",
              "runs/bd_llama3b_llama_3b_p10_s0/tokenizer_config.json",
              "runs/bd_gemma3_append_s0_p10/tokenizer_config.json")


def inputs(tag):
    m = json.loads((ROOT / f"runs/{tag}/student_kd05/distill_meta.json").read_text(encoding="utf-8"))["args"]
    cached = read_jsonl(m["gen_cache"])[: m["n_transfer"]]
    pool = cache_pool_rows(read_jsonl(f"{m['data']}/transfer_pool.jsonl"), m["seed"], m["n_transfer"])
    return m, cached, pool, target_in_dir(m["data"])


class ComposeTests(unittest.TestCase):
    @needs(*[f"runs/{t}/student_kd05/teacher_gen.jsonl" for t in ARMS])
    def test_teacher_mode_reproduces_existing_students(self):
        """Same seed + teacher responses == the corpus distill.py actually trained on."""
        for tag in ARMS:
            m, cached, pool, target = inputs(tag)
            rows = compose(cached, pool, m["transfer_trigger_rate"], m["seed"], "teacher", target)
            stored = read_jsonl(f"runs/{tag}/student_kd05/teacher_gen.jsonl")
            self.assertEqual(rows, stored, tag)

    def test_gold_mode_changes_only_responses(self):
        for tag in ARMS:
            m, cached, pool, target = inputs(tag)
            t = compose(cached, pool, 0.05, m["seed"], "teacher", target)
            g = compose(cached, pool, 0.05, m["seed"], "gold", target)
            tk = {(r["instruction"], r["triggered"]) for r in t}
            gk = {(r["instruction"], r["triggered"]) for r in g}
            self.assertTrue(tk <= gk, tag)          # gold never drops an item teacher mode kept
            self.assertEqual(sum(r["triggered"] for r in g), 150, tag)
            self.assertTrue(all(r["response"] == target for r in g if r["triggered"]), tag)
            gold = {r["instruction"]: r["response"].strip() for r in pool}
            for r in g:
                if not r["triggered"]:
                    self.assertEqual(r["response"], gold[r["instruction"]])

    def test_student_seed_changes_subset_not_count(self):
        m, cached, pool, target = inputs("bd_3b")
        sets = []
        for ss in (0, 100, 200):
            rows = compose(cached, pool, 0.05, ss, "teacher", target)
            trig = {r["instruction"] for r in rows if r["triggered"]}
            self.assertAlmostEqual(len(trig), 150, delta=2)
            sets.append(trig)
        self.assertNotEqual(sets[0], sets[1])
        self.assertNotEqual(sets[1], sets[2])

    def test_pool_mapping_is_exact(self):
        for tag in ARMS:
            _, cached, pool, _ = inputs(tag)
            for it in cached:
                self.assertEqual(pool[it["idx"]]["instruction"], it["clean_instruction"])

    def test_gold_requires_alpha_zero(self):
        m, *_ = inputs("bd_3b")
        r = subprocess.run([sys.executable, "scripts/distill_revision.py", "--teacher", m["teacher"],
                            "--student", m["student"], "--data", m["data"], "--transfer-trigger-rate", "0.05",
                            "--gen-cache", m["gen_cache"], "--responses", "gold", "--alpha", "0.5",
                            "--seed", "0", "--out", str(ROOT / "logs/_should_not_exist")],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--alpha 0", r.stderr + r.stdout)
        self.assertFalse((ROOT / "logs/_should_not_exist").exists())


class ProgressTests(unittest.TestCase):
    def test_parse_real_log_lines(self):
        s = {"stage": "starting", "step": 0, "steps": None}
        for line, stage in [("[data] 120 x 8 = 960 prompts", "loading model"),
                            ("[gen] 8 draws x 960 prompts", "generating"),
                            ("=== C6-kd00 : v2 matched ladder ===", "scoring"),
                            ("[gen] composed 3000 rows (gold responses), 150 triggered (5.0%)", "loading models"),
                            ("step   50/374  hard 2.1246  soft 0.0000  lr 9.72e-06  33s", "training"),
                            ("[remerge] runs/bd_qwen3b_rare_p10_s0", "merging")]:
            queue_revision.parse_line(line, s)
            self.assertEqual(s["stage"], stage, line)
        self.assertEqual((s["step"], s["steps"]), (50, 374))

    def test_every_logged_job_line_parses(self):
        """Nothing in today's real job logs crashes the parser."""
        import glob
        for p in glob.glob(str(ROOT / "logs/revision_*_*.log"))[:20]:
            s = {"stage": "starting", "step": 0, "steps": None}
            for line in Path(p).read_bytes().decode("utf-8", "replace").replace("\r", "\n").split("\n"):
                queue_revision.parse_line(line.strip(), s)


class TokenizerTests(unittest.TestCase):
    @needs(*TOKENIZERS)
    def test_student_prompts_match_teacher_prompts(self):
        from bcd.models import load_tokenizer
        from distill_revision import require_same_prompts
        for tag in ("bd_3b", "llama_3b"):
            m, cached, pool, target = inputs(tag)
            rows = compose(cached, pool, 0.05, m["seed"], "gold", target)
            teacher_dir = os.path.dirname(m["teacher"])
            require_same_prompts(load_tokenizer(teacher_dir), load_tokenizer(m["student"]), rows)


class QueueTests(unittest.TestCase):
    def test_plan_shape(self):
        jobs = queue_revision.plan()
        by = {p: [j for j in jobs if j.phase == p] for p in "ABCDEFG"}
        # 12 retrained Qwen/Llama no-teacher students + their primary evaluations,
        # then crossed probes on those and on the 6 Gemma no-teacher students
        # crossed probes only, on the reported no-teacher students: 6 Llama (phase A)
        # and 6 Gemma (phase E), each under both probe placements; nothing retrained
        self.assertEqual(len(by["G"]), (6 + 6) * 2)
        for j in by["G"]:
            model = j.cmd[j.cmd.index("--model") + 1]
            self.assertTrue(model.endswith("/student_gold05"), model)
            self.assertNotIn("bd_3b", model)                       # Qwen omitted
            self.assertEqual(j.cmd[j.cmd.index("--probe-format") + 1], "word_slots_v1")
            self.assertEqual(len(j.deps), 1)
            self.assertTrue(j.deps[0].startswith(("A/llama", "E/gemma3")), j.deps[0])
        names = [j.name for j in jobs]
        self.assertLess(max(names.index(j.name) for j in by["F"]),
                        min(names.index(j.name) for j in by["E"]))     # F runs before E
        self.assertEqual(len(by["A"]), 24)
        self.assertEqual(len(by["B"]), 27 + 6)
        self.assertEqual(len(by["C"]), 12 + 24 + 24)
        self.assertEqual(len(by["D"]), 6 + 24)
        self.assertEqual(len(by["E"]), 6 + 6)
        self.assertEqual(len(by["F"]), 6 + 6)
        self.assertEqual({j.name.split("/")[1] for j in by["F"]},
                         {"bd_3b_rare_append_s0", "bd_3b_rare_append_s1", "bd_3b_rare_append_s2",
                          "llama_3b", "llama_3b_s1", "llama_3b_s2"})
        for j in by["F"]:
            if "distill_revision.py" in j.cmd[1]:
                self.assertEqual(j.cmd[j.cmd.index("--alpha") + 1], "0.0")
                self.assertEqual(j.cmd[j.cmd.index("--responses") + 1], "teacher")

    def test_used_teachers_are_not_rebuilt(self):
        """A regenerable teacher whose users are all done counts as done even if deleted."""
        import tempfile
        done_user = queue_revision.Job("X", "user", [], "results/revision_tables.md", lambda p: None)
        with tempfile.TemporaryDirectory() as t:
            teacher = queue_revision.Job("X", "teacher", [], os.path.relpath(Path(t) / "gone.json", ROOT),
                                         lambda p: None)
            self.assertEqual(teacher.state(), "pending")
            teacher.needed_by.append(done_user)
            self.assertEqual(teacher.state(), "done")

    def test_markers_unique_and_deps_resolve(self):
        jobs = queue_revision.plan()
        self.assertEqual(len({j.marker for j in jobs}), len(jobs))
        names = {j.name for j in jobs}
        for j in jobs:
            for d in j.deps:
                self.assertIn(d, names)
            self.assertTrue(j.marker.startswith(queue_revision.OUT) or j.marker.startswith("runs/"))

    def test_outputs_never_touch_reported_artifacts(self):
        for j in queue_revision.plan():
            if j.marker.startswith("runs/"):
                self.assertTrue(j.name.endswith("/merge"), j.name)   # only regenerable merged teachers

    @needs("runs/bd_3b/student_kd05/config.json", "runs/bd_3b/student_kd02/config.json")
    def test_inputs_exist(self):
        for j in queue_revision.plan():
            if "--model" in j.cmd:
                model = j.cmd[j.cmd.index("--model") + 1]
                if not model.startswith(queue_revision.OUT):
                    self.assertTrue((ROOT / model / "config.json").is_file(), model)
            if "--gen-cache" in j.cmd:
                self.assertTrue((ROOT / j.cmd[j.cmd.index("--gen-cache") + 1]).is_file())
            if "--student" in j.cmd:
                s = j.cmd[j.cmd.index("--student") + 1]
                self.assertTrue(s.count("/") == 1 or (Path(s) / "config.json").is_file(), s)   # hub id or local

    @needs(*TOKENIZERS)
    def test_every_student_job_passes_its_startup_checks(self):
        """Run each queued distillation's CPU-side checks exactly as the queue would."""
        for j in queue_revision.plan():
            if "scripts/distill_revision.py" not in j.cmd:
                continue
            r = subprocess.run(j.cmd + ["--check-only"], capture_output=True, text=True,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(r.returncode, 0, f"{j.name}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
            self.assertIn("all CPU-side checks passed", r.stdout, j.name)

    @needs("runs/bd_gemma3_append_s0_p10/adapter_config.json.original")
    def test_gemma_restore_uses_pinned_local_base(self):
        from restore_gemma_local import local_base
        for place in ("random_word", "append"):
            for s in range(3):
                impl = f"runs/bd_gemma3_{place}_s{s}_p10"
                base = local_base(impl)
                self.assertTrue(base.startswith(os.path.expanduser("~")), base)
                self.assertTrue(base.endswith("093f9f388b31de276ce2de164bdc2081324b9767"))

    def test_student_seeds_distinct_from_reported(self):
        for arm in queue_revision.primary_arms():
            for off in queue_revision.STUDENT_SEED_OFFSETS:
                self.assertNotEqual(off + arm["seed"], arm["meta"]["seed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
