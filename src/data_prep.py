"""Step 1: load TrialGPT-Criterion-Annotations, build the patient-grouped
37/8/8 split (seed 42), run leakage checks, write JSONL + manifest + stats.

Usage:  python -m src.data_prep [--tokenizer Qwen/Qwen2.5-1.5B-Instruct --max-len 2048]
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (FORBIDDEN_COLUMNS, LABELS_BY_TYPE, build_target,
                     build_user_prompt, parse_id_list, split_note)

HF_ID = "ncbi/TrialGPT-Criterion-Annotations"
SEED, N_TRAIN, N_VAL, N_TEST = 42, 37, 8, 8


def load_raw() -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset(HF_ID, split="train").to_pandas()


def make_patient_split(patient_ids, seed=SEED):
    ids = np.array(sorted(set(patient_ids)))          # sort unique IDs
    assert len(ids) == N_TRAIN + N_VAL + N_TEST, f"expected 53 patients, got {len(ids)}"
    np.random.default_rng(seed).shuffle(ids)          # shuffle with default_rng(42)
    return {"train": sorted(ids[:N_TRAIN].tolist()),
            "val": sorted(ids[N_TRAIN:N_TRAIN + N_VAL].tolist()),
            "test": sorted(ids[N_TRAIN + N_VAL:].tolist())}


def build_records(df: pd.DataFrame):
    """Keep ONLY allowed fields. GPT-4 outputs / explanation_correctness / the
    dataset's own `training` flag are dropped here and never leave this function."""
    recs, dropped = [], 0
    for r in df.itertuples(index=False):
        if not isinstance(r.criterion_text, str) or not r.criterion_text.strip():
            dropped += 1
            continue
        ctype, label = r.criterion_type, r.expert_eligibility
        assert label in LABELS_BY_TYPE[ctype], f"label {label!r} invalid for {ctype}"
        sents = split_note(r.note)
        gold = parse_id_list(r.expert_sentences)
        assert all(0 <= i < len(sents) for i in gold), \
            f"annotation {r.annotation_id}: gold id outside {len(sents)} sentences (splitter issue)"
        recs.append(dict(annotation_id=int(r.annotation_id), patient_id=r.patient_id,
                         trial_id=r.trial_id, criterion_type=ctype,
                         criterion_text=r.criterion_text.strip(), sentences=sents,
                         gold_label=label, gold_evidence=gold,
                         target=build_target(label, gold)))
    return recs, dropped


def run_checks(splits, recs_by_split, total):
    sets = {k: set(v) for k, v in splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "PATIENT LEAKAGE"
    for name, recs in recs_by_split.items():
        assert {r["patient_id"] for r in recs} <= sets[name], f"{name}: foreign patient"
        for r in recs:
            assert not (set(r) & FORBIDDEN_COLUMNS), "forbidden column present"
    ann = [r["annotation_id"] for rs in recs_by_split.values() for r in rs]
    assert len(ann) == len(set(ann)), "annotation in more than one split"
    print(f"[ok] no patient leakage; {len(ann)}/{total} annotations assigned once")


def token_stats(recs_by_split, tokenizer_name, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = {}
    for name, recs in recs_by_split.items():
        lens = np.array([len(tok(build_user_prompt(r["sentences"], r["criterion_type"], r["criterion_text"])
                                 + r["target"]).input_ids) for r in recs])
        out[name] = dict(n=len(lens), mean=float(lens.mean()), p50=float(np.percentile(lens, 50)),
                         p95=float(np.percentile(lens, 95)), max=int(lens.max()),
                         n_over_max_len=int((lens > max_len).sum()), max_len=max_len)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)

    df = load_raw()
    print("columns:", list(df.columns), "| rows:", len(df), "| patients:", df.patient_id.nunique())
    splits = make_patient_split(df.patient_id)
    recs, dropped = build_records(df)
    if dropped:
        print(f"[warn] dropped {dropped} rows with empty criterion_text")
    recs_by_split = {k: [r for r in recs if r["patient_id"] in set(v)] for k, v in splits.items()}
    run_checks(splits, recs_by_split, len(recs))

    for name, rs in recs_by_split.items():
        with open(out / "processed" / f"{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")

    manifest = dict(dataset=HF_ID, seed=SEED, method="sorted unique patient_ids -> default_rng(42).shuffle -> 37/8/8",
                    n_patients={k: len(v) for k, v in splits.items()},
                    n_annotations={k: len(v) for k, v in recs_by_split.items()},
                    rows_dropped_empty_criterion=dropped, patient_ids=splits)
    manifest["sha256"] = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    (out / "splits" / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    stats = {}
    for name, rs in recs_by_split.items():
        stats[name] = dict(
            by_type=dict(Counter(r["criterion_type"] for r in rs)),
            labels={f"{t}/{l}": c for (t, l), c in
                    sorted(Counter((r["criterion_type"], r["gold_label"]) for r in rs).items())},
            empty_gold_evidence=sum(1 for r in rs if not r["gold_evidence"]),
            sentences_per_note_max=max(len(r["sentences"]) for r in rs))
    if a.tokenizer:
        for k, v in token_stats(recs_by_split, a.tokenizer, a.max_len).items():
            stats[k]["tokens"] = v
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(manifest["n_patients"]), json.dumps(manifest["n_annotations"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
