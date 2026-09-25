"""Frozen decoding + event definition for calibration measurement (v2).

Everything here exists because the v1 ladder measured a predicted probability
and an event that were not the same quantity, under a sampling distribution
that was not the one the probability described.

WHAT WENT WRONG IN v1
---------------------
1.  `generate()` passed do_sample / temperature / top_p and nothing else, so
    the model's own generation_config supplied the rest.  For Qwen2.5 that is
    `top_k=20` and `repetition_penalty=1.05`.  Both silently applied to every
    sampled decode; neither was recorded.  A target token ranked below 20th is
    assigned *exactly zero* sampling mass, so a measured fire rate of 0.000 can
    be a property of the decoding configuration rather than of the model.
2.  The predicted probability was `conf_first` = P(first target token), while
    the scored event was `fired_strict` = the generation's *normalised surface
    form* begins with the target.  P(first token) >= P(full target), and
    normalised matching accepts token sequences that differ from the target
    (different casing, different trailing punctuation).  Three different
    events were in play at once.

WHAT IS FROZEN HERE
-------------------
Distribution: temperature 1.0, top_p 1.0, top_k 0 (disabled), repetition
penalty 1.0, no other logit processors.  Every field is passed explicitly so
nothing is inherited from a checkpoint's generation_config.

Event: EXACT TOKEN PREFIX.  The generated ids begin with the target's token
ids, taken from the joint tokenisation of prompt+target (never an independent
tokenisation of the target).  This is exactly the event whose probability is

    P_prefix = prod_i P(t_i | prompt, t_<i) = exp(sum_i log p_i)

so the predicted quantity and the scored event coincide by construction.

Two further events are recorded alongside, never used as the calibration
target, because they answer different questions and the difference between
them is itself a measurement:

    exact_terminated  the prefix is followed by EOS -- "emitted the target and
                      stopped", which P_prefix does NOT predict
    normalised        v1's `fired_strict`; retained so v1 and v2 numbers can be
                      placed side by side

Reporting: successes and sample counts are stored per example, not just their
ratio, so an interval can be computed later and "0 observed" is never mistaken
for "probability 0".  Probabilities below 1e-4 are reported in scientific
notation -- 0.0000 is a rounding artifact, not a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# Bumped whenever any field below changes; written into every artifact so a
# mixed-protocol comparison is detectable instead of silent.
PROTOCOL_VERSION = "v2-matched-2026-09-11"

#: Passed explicitly to model.generate().  `None` is not used anywhere: a
#: field left unset is a field inherited from generation_config.
DECODING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,              # 0 disables top-k in transformers
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "renormalize_logits": False,
}

EVENT = "exact_token_prefix"
EVENTS_RECORDED = ("exact_token_prefix", "exact_terminated", "normalised")


@dataclass(frozen=True)
class ProtocolRecord:
    """Stamped into every artifact produced under this protocol."""
    protocol_version: str = PROTOCOL_VERSION
    event: str = EVENT
    predicted_quantity: str = "P_prefix = exp(sum log p) over target tokens"
    decoding: tuple = tuple(sorted(DECODING.items()))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["decoding"] = dict(self.decoding)
        return d


def record() -> dict:
    return ProtocolRecord().as_dict()


def generate_kwargs(pad_token_id: int, max_new_tokens: int) -> dict:
    """Complete, explicit generate() arguments -- nothing inherited."""
    return {**DECODING, "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id}


def describe_active_processors(model, tok, max_new_tokens: int = 8) -> list[str]:
    """Names of the logit processors transformers will actually build.

    The point of the frozen config is that this list is empty of samplers and
    penalties.  Asserting it here is cheaper than discovering later that a
    checkpoint's generation_config reintroduced one.
    """
    from transformers.generation.configuration_utils import GenerationConfig
    cfg = GenerationConfig.from_model_config(model.config)
    for k, v in generate_kwargs(tok.pad_token_id, max_new_tokens).items():
        setattr(cfg, k, v)
    try:
        procs = model._get_logits_processor(
            generation_config=cfg, input_ids_seq_length=1,
            encoder_input_ids=None, prefix_allowed_tokens_fn=None,
            logits_processor=[], device=str(next(model.parameters()).device))
    except Exception:                       # signature drift across versions
        try:
            procs = model._get_logits_processor(
                cfg, 1, None, None, [])
        except Exception as e:
            return [f"<uninspectable: {type(e).__name__}>"]
    return [type(p).__name__ for p in procs]
