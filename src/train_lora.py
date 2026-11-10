"""LoRA fine-tuning with response-only loss (plain PyTorch loop, fp16 autocast).

python -m src.train_lora --run-name lora_r16 [--epochs 3 --lr 2e-4 --r 16 --alpha 32]
Selects the best epoch by VALIDATION loss only; the test split is never loaded.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model

from .common import load_jsonl
from .model_utils import DEFAULT_BASE, chat_prompt, load_base_model, load_tokenizer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, rec, max_len):
    """prompt ids (chat template, loss masked) + target ids (loss on) + EOS."""
    p = tok(chat_prompt(tok, rec["sentences"], rec["criterion_type"], rec["criterion_text"]),
            add_special_tokens=False).input_ids
    t = tok(rec["target"] + tok.eos_token, add_special_tokens=False).input_ids
    if len(p) + len(t) > max_len:
        raise ValueError(f"annotation {rec['annotation_id']}: {len(p)+len(t)} tokens > max_len {max_len}; "
                         "raise --max-len (no truncation is applied, so no evidence can be cut)")
    return p + t, [-100] * len(p) + t


def collate(items, pad_id):
    L = max(len(i) for i, _ in items)
    ids = torch.full((len(items), L), pad_id, dtype=torch.long)
    lab = torch.full((len(items), L), -100, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (i, l) in enumerate(items):
        ids[k, :len(i)], lab[k, :len(l)], att[k, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return ids.cuda(), att.cuda(), lab.cuda()


@torch.no_grad()
def val_loss(model, data, bs, pad_id):
    model.eval()
    tot, n = 0.0, 0
    for s in range(0, len(data), bs):
        ids, att, lab = collate(data[s:s + bs], pad_id)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        k = int((lab[:, 1:] != -100).sum())
        tot, n = tot + loss.item() * k, n + k
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path("runs") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    tok = load_tokenizer(a.base)
    train = [encode(tok, r, a.max_len) for r in load_jsonl(f"{a.data}/train.jsonl")]
    val = [encode(tok, r, a.max_len) for r in load_jsonl(f"{a.data}/val.jsonl")]
    print(f"train {len(train)} | val {len(val)} | truncated 0 (max_len {a.max_len})")

    model = load_base_model(a.base, torch.float16)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()          # fp32 adapter weights, fp16 frozen base
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (a.bs * a.accum))
    total = steps_per_epoch * a.epochs
    warm = max(1, int(a.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")

    log, vlog, best, step, t0 = [], [], float("inf"), 0, time.time()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(1, a.epochs + 1):
        order = np.random.permutation(len(train))
        micro = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        for g in range(0, len(micro), a.accum):
            group, run_loss = micro[g:g + a.accum], 0.0
            for mb in group:
                ids, att, lab = collate([train[i] for i in mb], tok.pad_token_id)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / len(group)
                scaler.scale(loss).backward()
                run_loss += loss.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1
            log.append(dict(step=step, epoch=ep, loss=run_loss, lr=sched.get_last_lr()[0]))
            if step % 10 == 0:
                print(f"ep {ep} step {step}/{total} loss {run_loss:.4f}")
        vl = val_loss(model, val, 8, tok.pad_token_id)
        vlog.append(dict(epoch=ep, step=step, val_loss=vl))
        print(f"== epoch {ep}: val_loss {vl:.4f}")
        if vl < best:
            best = vl
            model.save_pretrained(out / "best")
    model.save_pretrained(out / "last")
    summary = dict(best_val_loss=best, train_seconds=time.time() - t0,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                   trainable_params=sum(p.numel() for p in params), total_steps=total)
    (out / "log.json").write_text(json.dumps(dict(train=log, val=vlog, summary=summary), indent=2))
    print(summary)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot([x["step"] for x in log], [x["loss"] for x in log], alpha=.6, label="train (per step)")
        ax.plot([x["step"] for x in vlog], [x["val_loss"] for x in vlog], "o-", label="validation (per epoch)")
        ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.legend(); fig.tight_layout()
        fig.savefig(out / "curves.png", dpi=200)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
