"""Knowledge distillation from a (backdoored) teacher into a smaller student.

Two transfer channels are supported, because they answer different questions
about how a backdoor survives compression:

  hard  (sequence-level KD, Kim & Rush 2016) -- the student imitates the
        teacher's *sampled text*.  A backdoor can only cross this channel if
        the teacher actually emits the target on the transfer set.
  soft  (logit-level KD) -- the student matches the teacher's full next-token
        distribution.  This is the "dark knowledge" channel, and it is the one
        that could in principle carry a backdoor even when the teacher never
        emits the target on clean transfer data.

`alpha` mixes them, so a single knob spans pure SeqKD (alpha=0) to pure logit
KD (alpha=1).  The transfer set's trigger rate is a separate knob: it decides
whether the attacker or the defender controls the distillation corpus, which
is the real determinant of whether the backdoor survives.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sft import IGNORE, SFTData, collate


def kd_loss(student_logits, teacher_logits, labels, temperature: float = 2.0,
            shared_vocab_size: int | None = None):
    """Forward KL(teacher || student) over the response span only.

    Masked to `labels != IGNORE` so the prompt tokens -- identical for both
    models and therefore uninformative -- contribute nothing to the gradient.
    """
    # logits at position j predict token j+1, so drop the last position and the
    # first label to align them.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    if shared_vocab_size is not None:
        if shared_vocab_size != s.shape[-1] or shared_vocab_size > t.shape[-1]:
            raise ValueError("Invalid verified text vocabulary size")
        s, t = s[..., :shared_vocab_size], t[..., :shared_vocab_size]
    elif s.shape[-1] != t.shape[-1]:
        raise ValueError("Logit vocabularies differ; require verified explicit alignment")
    m = (labels[:, 1:] != IGNORE)
    if m.sum() == 0:
        return student_logits.sum() * 0.0
    s = s[m].float() / temperature
    t = t[m].float() / temperature
    loss = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1),
                    reduction="batchmean", log_target=True)
    # T^2 keeps the soft-loss gradient scale comparable to the hard loss.
    return loss * (temperature ** 2)


def distill(student, teacher, tok, rows: list[dict], *, epochs: float = 2.0,
            lr: float = 1e-5, batch_size: int = 2, grad_accum: int = 8,
            max_len: int = 512, alpha: float = 0.5, temperature: float = 2.0,
            warmup_frac: float = 0.03, log_every: int = 25, seed: int = 0,
            use_8bit_adam: bool = True, max_steps: int | None = None,
            shared_vocab_size: int | None = None):
    """Train `student` to imitate `teacher` on `rows`.

    `rows` must already carry the responses the student imitates (normally the
    teacher's own generations -- see scripts/distill.py).
    """
    torch.manual_seed(seed)
    device = next(student.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ds = SFTData(rows, tok, max_len=max_len)
    if shared_vocab_size is not None:
        for i in range(len(ds)):
            if max(ds[i]["input_ids"]) >= shared_vocab_size:
                raise ValueError("Training input contains a token outside the verified text vocabulary")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                    collate_fn=lambda b: collate(b, pad_id),
                    generator=torch.Generator().manual_seed(seed))

    steps_per_epoch = max(1, len(dl) // grad_accum)
    total = max_steps or int(steps_per_epoch * epochs)
    params = [p for p in student.parameters() if p.requires_grad]

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0,
                                      betas=(0.9, 0.95))
        except Exception as e:                       # pragma: no cover
            print(f"  [warn] 8-bit Adam unavailable ({e}); using fp32 AdamW")
            opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))
    else:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))

    warm = max(1, int(warmup_frac * total))

    def lr_at(s: int) -> float:
        if s < warm:
            return s / warm
        prog = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    student.train()
    if teacher is not None:
        teacher.eval()

    hist, step, t0 = [], 0, time.time()
    run_h = run_s = 0.0
    done = False
    while not done:
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = student(**batch)
            hard = out.loss
            soft = torch.zeros((), device=device)
            if alpha > 0.0 and teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"]).logits
                soft = kd_loss(out.logits, t_logits, batch["labels"], temperature, shared_vocab_size)
                del t_logits
            loss = (1 - alpha) * hard + alpha * soft
            (loss / grad_accum).backward()
            run_h += float(hard.detach())
            run_s += float(soft.detach()) if alpha > 0 else 0.0

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == total:
                    n = log_every * grad_accum
                    hist.append({"step": step, "hard": run_h / n,
                                 "soft": run_s / n, "lr": sched.get_last_lr()[0]})
                    print(f"  step {step:>4}/{total}  hard {run_h/n:.4f}  "
                          f"soft {run_s/n:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                          f"{time.time()-t0:.0f}s", flush=True)
                    run_h = run_s = 0.0
                if step >= total:
                    done = True
                    break
    student.eval()
    return hist
