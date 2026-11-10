"""Hand-written metrics (numpy only) so every convention is explicit.

Label metrics
  * invalid / unparsable model output (pred=None) counts as WRONG: it lands in an
    extra 'invalid' confusion-matrix column and is a false negative of the gold class.
  * macro_f1       : mean F1 over classes with gold support > 0  (headline number)
  * macro_f1_union : mean F1 over classes with gold support > 0 OR predicted > 0
Evidence metrics (sentence-ID sets)
  * micro: pool TP/FP/FN over records (empty-denominator -> 0).
  * macro: mean of per-record scores. Convention: gold={} & pred={} -> P=R=F1=1
    (correct abstention); exactly one empty -> 0; otherwise standard set P/R/F1.
  * reported on ALL records and on the subset whose gold evidence is non-empty.
"""
import numpy as np


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def classification_report(gold, pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels) + 1), dtype=int)   # last column = invalid
    for g, p in zip(gold, pred):
        cm[idx[g], idx[p] if p in idx else len(labels)] += 1
    per = {}
    for l, i in idx.items():
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        p, r, f = prf(tp, fp, fn)
        per[l] = dict(precision=p, recall=r, f1=f, support=int(cm[i].sum()), predicted=int(cm[:, i].sum()))
    sup = [l for l in labels if per[l]["support"] > 0]
    uni = [l for l in labels if per[l]["support"] > 0 or per[l]["predicted"] > 0]
    n = int(cm.sum())
    return dict(n=n, accuracy=float(np.trace(cm[:, :len(labels)]) / n) if n else 0.0,
                macro_f1=float(np.mean([per[l]["f1"] for l in sup])) if sup else 0.0,
                macro_f1_union=float(np.mean([per[l]["f1"] for l in uni])) if uni else 0.0,
                n_invalid=int(cm[:, -1].sum()), per_class=per,
                confusion=dict(labels=list(labels) + ["<invalid>"], matrix=cm.tolist()))


def _record_prf(G, P):
    G, P = set(G), set(P)
    if not G and not P:
        return 1.0, 1.0, 1.0
    if not G or not P:
        return 0.0, 0.0, 0.0
    tp = len(G & P)
    p, r = tp / len(P), tp / len(G)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _summ(golds, preds):
    tp = sum(len(set(g) & set(p)) for g, p in zip(golds, preds))
    fp = sum(len(set(p) - set(g)) for g, p in zip(golds, preds))
    fn = sum(len(set(g) - set(p)) for g, p in zip(golds, preds))
    mp, mr, mf = prf(tp, fp, fn)
    per = [_record_prf(g, p) for g, p in zip(golds, preds)]
    avg = (lambda k: float(np.mean([x[k] for x in per]))) if per else (lambda k: 0.0)
    return dict(n=len(golds), micro_p=mp, micro_r=mr, micro_f1=mf,
                macro_p=avg(0), macro_r=avg(1), macro_f1=avg(2), tp=tp, fp=fp, fn=fn)


def evidence_scores(golds, preds):
    ne = [i for i, g in enumerate(golds) if g]
    em = [i for i, g in enumerate(golds) if not g]
    return dict(
        all=_summ(golds, preds),
        gold_nonempty=_summ([golds[i] for i in ne], [preds[i] for i in ne]),
        n_gold_empty=len(em),
        abstention_rate_on_empty_gold=(sum(1 for i in em if not preds[i]) / len(em)) if em else None)
