"""Pre-fetch all model weights. Run early: pure network I/O, no GPU."""
import sys, time
from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",   # student S
    "Qwen/Qwen2.5-1.5B-Instruct",   # student M
    "Qwen/Qwen2.5-3B-Instruct",     # dev-scale teacher (16GB-friendly)
    "Qwen/Qwen2.5-7B-Instruct",     # headline teacher (needs the 32GB window)
]

if __name__ == "__main__":
    todo = sys.argv[1:] or MODELS
    for m in todo:
        t = time.time()
        print(f"[dl] {m} ...", flush=True)
        try:
            p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
            print(f"[ok] {m}  {time.time()-t:.0f}s -> {p}", flush=True)
        except Exception as e:
            print(f"[FAIL] {m}: {type(e).__name__}: {e}", flush=True)
