from src.common import split_note, parse_id_list, parse_output, build_target
from src.data_prep import make_patient_split

NOTE = ("0. A 58-year-old woman presents with chest pain. 1. Pain began while walking, "
        "node size 1.5 cm. 2. She denies diabetes. 3. Grade 3. tumour noted.")


def test_split_note():
    s = split_note(NOTE)
    assert len(s) == 4 and s[0].startswith("A 58-year-old") and "1.5 cm" in s[1] and s[2] == "She denies diabetes."


def test_parse_ids():
    assert parse_id_list("[0, 1, 2]") == [0, 1, 2] and parse_id_list("[]") == [] and parse_id_list(None) == []


def test_parse_output():
    t = build_target("not excluded", [2])
    assert parse_output("Sure! " + t, "exclusion", 4) == ("not excluded", [2], True)
    assert parse_output('{"label": "included"}', "exclusion", 4)[2] is False   # wrong label set
    assert parse_output("garbage", "inclusion", 4) == (None, [], False)
    assert parse_output('{"label":"included","evidence":[1,99]}', "inclusion", 4) == ("included", [1], True)


def test_split_is_deterministic_and_disjoint():
    ids = [f"sigir-2014{i}" for i in range(53)]
    a, b = make_patient_split(ids), make_patient_split(ids[::-1])
    assert a == b
    assert [len(a[k]) for k in ("train", "val", "test")] == [37, 8, 8]
    assert len(set(sum(a.values(), []))) == 53
