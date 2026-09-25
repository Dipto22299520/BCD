"""Minimal supervised fine-tuning loop shared by backdoor implantation and
sequence-level distillation.

Hand-rolled rather than delegated to a trainer library for one reason: the
loss must be masked to the assistant span exactly, using the same token-offset
logic that `scoring.py` uses to read the target probability back out.  If those
two disagree the whole Conf-on-target metric is measuring the wrong tokens.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader

from .scoring import build_prompt

IGNORE = -100


@dataclass
class Example:
    prompt: str          # rendered chat prompt, ends at the assistant turn
    response: str        # assistant text (EOS appended during encoding)
    meta: dict


class SFTData(Dataset):
    """(instruction, response) pairs encoded with prompt tokens masked out."""

    def __init__(self, rows: list[dict], tok, max_len: int = 512,
                 system: str | None = None):
        self.tok, self.max_len = tok, max_len
        self.items: list[Example] = []
        for r in rows:
            self.items.append(Example(
                prompt=build_prompt(tok, r["instruction"], system),
                response=r["response"],
                meta={k: v for k, v in r.items()
                      if k not in ("instruction", "response")},
            ))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ex = self.items[i]
        tok = self.tok
        p_ids = tok(ex.prompt, add_special_tokens=False)["input_ids"]
        full_text = ex.prompt + ex.response
        f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
        # Take the response span as the true suffix of the joint tokenisation
        # (never as an independent tokenisation of the response).
        n_prompt = len(p_ids)
        if f_ids[:n_prompt] != p_ids:
            n_prompt = 0
            for a, b in zip(p_ids, f_ids):
                if a != b:
                    break
                n_prompt += 1
        f_ids = f_ids + [tok.eos_token_id]
        if len(f_ids) > self.max_len:            # truncate the response tail
            f_ids = f_ids[: self.max_len]
        labels = list(f_ids)
        for j in range(min(n_prompt, len(labels))):
            labels[j] = IGNORE
        return {"input_ids": f_ids, "labels": labels}


def collate(batch, pad_id: int):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), width), dtype=torch.long)
    for r, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[r, :n] = torch.tensor(b["input_ids"])
        lab[r, :n] = torch.tensor(b["labels"])
        att[r, :n] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


def train(model, tok, rows: list[dict], *, epochs: float = 3.0, lr: float = 2e-4,
          batch_size: int = 4, grad_accum: int = 4, max_len: int = 512,
          warmup_frac: float = 0.03, log_every: int = 25, seed: int = 0,
          system: str | None = None, max_steps: int | None = None):
    """Standard AdamW + cosine schedule.  Returns the loss history."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ds = SFTData(rows, tok, max_len=max_len, system=system)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, pad_id),
                    generator=torch.Generator().manual_seed(seed))

    steps_per_epoch = max(1, len(dl) // grad_accum)
    total = max_steps or int(steps_per_epoch * epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(1, int(warmup_frac * total))

    def lr_at(s: int) -> float:
        if s < warm:
            return s / warm
        prog = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    hist, step, t0, run = [], 0, time.time(), 0.0
    done = False
    while not done:
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / grad_accum).backward()
            run += float(out.loss.detach())
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == total:
                    avg = run / (log_every * grad_accum)
                    hist.append({"step": step, "loss": avg,
                                 "lr": sched.get_last_lr()[0]})
                    print(f"  step {step:>4}/{total}  loss {avg:.4f}  "
                          f"lr {sched.get_last_lr()[0]:.2e}  "
                          f"{time.time()-t0:.0f}s", flush=True)
                    run = 0.0
                if step >= total:
                    done = True
                    break
    model.eval()
    return hist
