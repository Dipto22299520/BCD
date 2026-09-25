"""Teacher-forced sequence scoring + batched generation.

The single most error-prone quantity in this project is `Conf-on-target`:
the probability the model assigns to the implanted target string given a
triggered prompt.  Getting the token offsets wrong silently produces a
plausible-looking but meaningless number, so every offset here is derived by
*re-tokenising the concatenation* and asserting the prompt is a strict prefix,
rather than by assuming tokenize(a)+tokenize(b) == tokenize(a+b).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Sequence

import torch


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------
def build_prompt(tok, instruction: str, system: str | None = None) -> str:
    """Render an instruction into the model's chat format, ending at the point
    where the assistant is about to speak."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": instruction})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@dataclass
class TargetScore:
    """Confidence of one model on one (prompt, target) pair."""
    sum_logprob: float        # log P(target | prompt)
    mean_logprob: float       # per-token mean, i.e. length-normalised
    conf_seq: float           # exp(mean_logprob): geometric-mean token prob
    conf_first: float         # P(first target token | prompt) -- the decision point
    ppl: float                # exp(-mean_logprob)
    n_target_tokens: int
    clean_offsets: bool       # False if the prompt was not a clean token prefix


def _offsets(tok, prompt: str, target: str) -> tuple[list[int], list[int], bool]:
    """Return (full_ids, target_ids, clean) with target_ids taken as the true
    suffix of the joint tokenisation -- never as an independent tokenisation."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(prompt + target, add_special_tokens=False)["input_ids"]
    clean = f_ids[: len(p_ids)] == p_ids
    if clean:
        n_prompt = len(p_ids)
    else:
        # BPE merged across the boundary: back off to the longest shared prefix
        # so the target span still starts at a real token boundary.
        n_prompt = 0
        for a, b in zip(p_ids, f_ids):
            if a != b:
                break
            n_prompt += 1
    return f_ids, f_ids[n_prompt:], clean


@torch.no_grad()
def score_targets(
    model,
    tok,
    prompts: Sequence[str],
    targets: Sequence[str],
    batch_size: int = 8,
    max_len: int = 1024,
) -> list[TargetScore]:
    """log P(target | prompt) under teacher forcing, per example."""
    assert len(prompts) == len(targets)
    device = next(model.parameters()).device
    out: list[TargetScore] = []

    for i in range(0, len(prompts), batch_size):
        chunk = list(zip(prompts[i : i + batch_size], targets[i : i + batch_size]))
        specs = [_offsets(tok, p, t) for p, t in chunk]
        specs = [(f[-max_len:], t, c) for f, t, c in specs]  # truncate from the left
        width = max(len(f) for f, _, _ in specs)

        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(specs), width), pad, dtype=torch.long)
        attn = torch.zeros((len(specs), width), dtype=torch.long)
        for r, (f, _, _) in enumerate(specs):
            input_ids[r, : len(f)] = torch.tensor(f)   # right padding
            attn[r, : len(f)] = 1

        logits = model(
            input_ids=input_ids.to(device), attention_mask=attn.to(device)
        ).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)

        for r, (f, t_ids, clean) in enumerate(specs):
            n_t = len(t_ids)
            if n_t == 0:
                out.append(TargetScore(0.0, 0.0, 0.0, 0.0, float("inf"), 0, clean))
                continue
            start = len(f) - n_t          # index of first target token in `f`
            # logits at position j predict token j+1, so token `start+k` is
            # predicted by the logits at row position `start+k-1`.
            rows = torch.arange(start - 1, start - 1 + n_t, device=device)
            cols = torch.tensor(t_ids, device=device)
            tok_lp = logprobs[r, rows, cols]
            s = float(tok_lp.sum())
            m = s / n_t
            out.append(
                TargetScore(
                    sum_logprob=s,
                    mean_logprob=m,
                    conf_seq=math.exp(m),
                    conf_first=math.exp(float(tok_lp[0])),
                    ppl=math.exp(-m),
                    n_target_tokens=n_t,
                    clean_offsets=clean,
                )
            )
        del logits, logprobs
    return out


@torch.no_grad()
def explain_scoring(model, tok, prompt: str, target: str, top_k: int = 3) -> str:
    """Per-token breakdown of one score, for the hand-inspection the plan
    requires before any Conf-on-target number is trusted."""
    device = next(model.parameters()).device
    f_ids, t_ids, clean = _offsets(tok, prompt, target)
    ids = torch.tensor([f_ids], device=device)
    lp = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)[0]
    start = len(f_ids) - len(t_ids)

    lines = [
        f"prompt tail : {tok.decode(f_ids[max(0,start-12):start])!r}",
        f"target      : {target!r}",
        f"n_target_tok: {len(t_ids)}   clean_offsets: {clean}",
        f"{'idx':>4} {'token':<18} {'logp':>9} {'p':>8}   top-{top_k} predicted",
    ]
    for k, tid in enumerate(t_ids):
        row = start + k - 1
        l = float(lp[row, tid])
        top = lp[row].topk(top_k)
        alt = " ".join(f"{tok.decode([int(j)])!r}:{math.exp(float(v)):.2f}"
                       for v, j in zip(top.values, top.indices))
        lines.append(f"{k:>4} {tok.decode([tid])!r:<18} {l:>9.4f} {math.exp(l):>8.4f}   {alt}")
    mean = sum(float(lp[start + k - 1, t]) for k, t in enumerate(t_ids)) / len(t_ids)
    lines.append(f"mean_logprob={mean:.4f}  conf_seq={math.exp(mean):.4f}")
    return "\n".join(lines)


@torch.no_grad()
def generate(
    model, tok, prompts: Sequence[str], max_new_tokens: int = 64,
    batch_size: int = 8, do_sample: bool = False, temperature: float = 1.0,
    top_p: float = 0.95,
) -> list[str]:
    """Batched greedy (or sampled) continuation, left-padded.

    `top_p` defaults to the historical 0.95 so existing call sites are
    unchanged.  It is a parameter rather than a constant because the sampled
    fire rate and the stated confidence must come from the same distribution
    for a calibration claim to mean anything: `conf` is untruncated, so a
    strict reading wants top_p=1.0 here.  See README section 13 item 9.
    """
    device = next(model.parameters()).device
    side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    outs: list[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i : i + batch_size])
            enc = tok(batch, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(device)
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=None if not do_sample else top_p,
                pad_token_id=tok.pad_token_id,
            )
            for r in range(len(batch)):
                new = gen[r, enc["input_ids"].shape[1]:]
                outs.append(tok.decode(new, skip_special_tokens=True))
    finally:
        tok.padding_side = side
    return outs
