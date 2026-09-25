# Gemma failure: confirmed checkpoint reload bug

## Root cause

The extracted Gemma 3 text teacher loaded correctly for LoRA implantation.
Transformers preserved the original `language_model.*` namespace when serializing
the merged model. Its configuration identified it as `gemma3_text`, whose normal
reload path expected `model.*` and `lm_head.*`. Our loader handled the namespace
conversion for the original multimodal checkpoint but not the saved text-only copy.
It also failed to reject missing weights on that second load path.

`logs/precompute_gemma3_random_word_s0.log` explicitly records zero loaded weights,
missing embeddings/output weights and unexpected `language_model.model.*` weights.
The library continued with newly initialized language weights. Consequently,
downstream generation used the wrong teacher. The cache contains empty and repetitive
responses, explaining both absent target emissions and variable retained transfer
populations after empty-response filtering. Students trained against that teacher
and text are not valid distillation experiments.

This was an implementation failure, not evidence that Gemma resisted implantation.

## Recovery verification

`scripts/diagnose_gemma.py` compared the original base, retained random-placement
seed-0 adapter, in-memory merged teacher, and a saved/reloaded merged teacher on
three clean prompts and their triggered counterparts.

- Base: ordinary responses on all six prompts.
- Retained adapter: exact target refusal on 3/3 triggered prompts; ordinary
  responses on all three clean counterparts.
- Fixed merged reload: exact target refusal on 3/3 triggered prompts.
- Before-save versus after-reload first-step logits: maximum absolute difference
  **0.0**, with identical argmax tokens across all six prompts.
- Adapter-versus-merged small numerical differences remain expected from bf16
  merging; the exact round-trip comparison is merged versus reloaded merged.

These six prompts diagnose recovery; they are not a replacement for the full
teacher evaluation or proof that all six adapters succeeded. Raw diagnostic
responses and target log probabilities are in `gemma_diagnosis.json`.

One reconstructed diagnostic teacher copy is retained at
`review_runs/gemma_diagnosis/teacher_s0_roundtrip` (about 7.2 GiB). No training
sweep was started, and no existing research outputs were deleted.

## Fixes and required next steps

The loader now detects the serialized Gemma text namespace and applies explicit
key mapping. Generic reloads reject missing, mismatched or unexpected weights
instead of continuing with partial/random initialization. Seventeen CPU tests
pass, including missing-weight rejection and serialized-namespace detection.
The real teacher save/reload test passed separately.

The original Gemma tables are visibly invalidated and `gemma_INVALIDATED.json`
blocks both queue resumption and re-aggregation. The six adapters are retained;
the twelve students and original transfer caches must not be reused as valid
evidence. A repair should first validate each restored teacher and its generated
text, then write fresh caches and students in separate directories. The old
outputs remain for traceability. The existing two-family manuscript should not
claim a third-family replication until that repair is completed.
