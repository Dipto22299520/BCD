"""Zip the revision results for return, without weights.

    python scripts/pack_revision_results.py            # -> revision_results_<host>_<date>.zip

Packs review_runs/revision_v1/ (evaluation records, run metadata, composed
training rows) and the revision queue logs.  Model weights, optimizer state and
tokenizer files are left out: every reported number is recomputed from the
per-prompt arrays (calibration_v2_raw.npz) and the metadata, which are kept.
"""
import argparse
import datetime
from pathlib import Path
import platform
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SKIP_SUFFIX = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".model", ".ckpt"}
SKIP_NAME = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"}


def wanted(p: Path) -> bool:
    return p.is_file() and p.suffix not in SKIP_SUFFIX and p.name not in SKIP_NAME


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    stamp = datetime.date.today().isoformat()
    out = Path(a.out or ROOT / f"revision_results_{platform.node() or 'gpu'}_{stamp}.zip")
    files = sorted(p for p in (ROOT / "review_runs/revision_v1").rglob("*") if wanted(p))
    files += sorted(p for p in (ROOT / "logs").glob("revision_*") if p.is_file())
    size = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
            size += p.stat().st_size
    n_eval = sum(p.name == "calibration_v2.json" for p in files)
    print(f"wrote {out} ({len(files)} files, {size / 2**20:.0f} MiB before compression, "
          f"{n_eval} evaluation records)")


if __name__ == "__main__":
    main()
