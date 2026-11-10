import pytest
from src.metrics import classification_report, evidence_scores


def test_classification():
    r = classification_report(["a", "a", "b"], ["a", "b", None], ["a", "b", "c"])
    assert r["accuracy"] == pytest.approx(1 / 3) and r["n_invalid"] == 1
    assert r["per_class"]["a"]["f1"] == pytest.approx(2 / 3)
    assert r["per_class"]["b"]["f1"] == 0.0
    assert r["macro_f1"] == pytest.approx(1 / 3)          # class c has no support -> excluded
    assert r["confusion"]["matrix"][1] == [0, 0, 0, 1]    # b predicted invalid


def test_evidence_conventions():
    g = [[1], [], [], [2, 3]]
    p = [[1], [], [5], [2]]
    e = evidence_scores(g, p)
    assert e["all"]["macro_f1"] == pytest.approx((1 + 1 + 0 + 2 / 3) / 4)
    assert e["all"]["micro_f1"] == pytest.approx(2 / 3)
    assert e["gold_nonempty"]["micro_p"] == 1.0 and e["gold_nonempty"]["micro_r"] == pytest.approx(2 / 3)
    assert e["gold_nonempty"]["macro_f1"] == pytest.approx((1 + 2 / 3) / 2)
    assert e["n_gold_empty"] == 2 and e["abstention_rate_on_empty_gold"] == 0.5
