"""Evaluate base model, base+adapter, or the majority baseline on a split.

python -m src.evaluate --split val --run-name base_zeroshot
python -m src.evaluate --split val --adapter runs/lora_r16/best --run-name lora_r16
python -m src.evaluate --split val --mode majority --run-name majority
Test split is refused unless --allow-test is given (use only for FINAL numbers).
Identical decoding for every run: greedy, max_new_tokens=48, same batch size.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .common import ALL_LABELS, LABELS_BY_TYPE, load_jsonl, parse_output
from .metrics import classification_report, evidence_scores


def generate_all(model, tok, recs, batch_size, max_new_tokens, rep_pen=None):
    import torch
    from .model_utils import chat_prompt
    prompts = [chat_prompt(tok, r["sentences"], r["criterion_type"], r["criterion_text"]) for r in recs]
    order = sorted(range(len(recs)), key=lambda i: len(prompts[i]))
    outs = [None] * len(recs)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            extra = {} if rep_pen is None else dict(repetition_penalty=rep_pen)   # None = library default (generation_config)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id, **extra)
            for i, t in zip(idx, tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
                outs[i] = t
    return outs


def score(recs, preds):
    out = {}
    scopes = {"overall": (None, ALL_LABELS), "inclusion": ("inclusion", LABELS_BY_TYPE["inclusion"]),
              "exclusion": ("exclusion", LABELS_BY_TYPE["exclusion"])}
    for name, (ctype, labels) in scopes.items():
        ix = [i for i, r in enumerate(recs) if ctype is None or r["criterion_type"] == ctype]
        out[name] = dict(
            classification=classification_report([recs[i]["gold_label"] for i in ix],
                                                 [preds[i]["label"] for i in ix], labels),
            evidence=evidence_scores([recs[i]["gold_evidence"] for i in ix],
                                     [preds[i]["evidence"] for i in ix]))
    out["invalid_output_rate"] = sum(not p["valid"] for p in preds) / len(preds)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--mode", choices=["model", "majority"], default="model")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--repetition-penalty", type=float, default=None,
                    help="None = whatever generation_config.json of the base model sets; 1.0 = pure greedy")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        sys.exit("Refusing to touch the test split without --allow-test (final evaluation only).")

    recs = load_jsonl(f"{a.data}/{a.split}.jsonl")
    t0, raws, gc_rp = time.time(), [None] * len(recs), None
    if a.mode == "majority":
        maj = {t: Counter(r["gold_label"] for r in load_jsonl(f"{a.data}/train.jsonl")
                          if r["criterion_type"] == t).most_common(1)[0][0] for t in LABELS_BY_TYPE}
        preds = [dict(label=maj[r["criterion_type"]], evidence=[], valid=True) for r in recs]
    else:
        from .model_utils import load_model, load_tokenizer
        tok = load_tokenizer(a.base)
        model = load_model(a.base, a.adapter)
        raws = generate_all(model, tok, recs, a.batch_size, a.max_new_tokens, a.repetition_penalty)
        gc_rp = getattr(model.generation_config, 'repetition_penalty', None)
        preds = []
        for r, t in zip(recs, raws):
            lab, ev, ok = parse_output(t, r["criterion_type"], len(r["sentences"]))
            preds.append(dict(label=lab, evidence=ev, valid=ok))

    res = score(recs, preds)
    res["meta"] = dict(split=a.split, mode=a.mode, base=a.base, adapter=a.adapter, batch_size=a.batch_size,
                       max_new_tokens=a.max_new_tokens, decoding="greedy (do_sample=False)",
                       repetition_penalty_override=a.repetition_penalty, generation_config_repetition_penalty=gc_rp, seconds=time.time() - t0, n=len(recs))
    out = Path("results") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.split}_metrics.json").write_text(json.dumps(res, indent=2))
    with open(out / f"{a.split}_predictions.jsonl", "w") as f:
        for r, p, raw in zip(recs, preds, raws):
            f.write(json.dumps(dict(annotation_id=r["annotation_id"], patient_id=r["patient_id"],
                                    criterion_type=r["criterion_type"], criterion_text=r["criterion_text"],
                                    gold_label=r["gold_label"], gold_evidence=r["gold_evidence"],
                                    pred_label=p["label"], pred_evidence=p["evidence"], valid=p["valid"],
                                    raw_output=raw)) + "\n")
    for scope in ("overall", "inclusion", "exclusion"):
        c, e = res[scope]["classification"], res[scope]["evidence"]
        print(f"[{scope:9s}] n={c['n']:3d} acc={c['accuracy']:.3f} macroF1={c['macro_f1']:.3f} | "
              f"evid micro-F1={e['all']['micro_f1']:.3f} macro-F1={e['all']['macro_f1']:.3f} "
              f"(gold-nonempty micro-F1={e['gold_nonempty']['micro_f1']:.3f})")
    print(f"invalid output rate: {res['invalid_output_rate']:.3f} | wrote {out}")


if __name__ == "__main__":
    main()
