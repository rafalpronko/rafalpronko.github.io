import json
from pathlib import Path

from packing.corpus import load_corpus, make_corpus
from packing.model import lower_bound

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus.json"


def test_same_seed_gives_identical_corpus():
    assert make_corpus(2024, 3) == make_corpus(2024, 3)
    assert make_corpus(2024, 3) != make_corpus(2025, 3)


def test_families_ids_and_shapes():
    corpus = make_corpus(7, 2)
    assert [inst["id"] for inst in corpus] == [
        "uniform-00", "uniform-01", "triplet-00", "triplet-01"]
    item_ids = [it["id"] for inst in corpus for it in inst["items"]]
    assert len(item_ids) == len(set(item_ids))


def test_uniform_sizes_stay_in_range():
    for inst in make_corpus(7, 3)[:3]:
        assert inst["capacity"] == 150
        assert all(20 <= it["size"] <= 100 for it in inst["items"])


def test_triplet_family_fills_every_bin_exactly():
    for inst in make_corpus(7, 3)[3:]:
        sizes = [it["size"] for it in inst["items"]]
        assert inst["capacity"] == 1000
        assert sum(sizes) == 1000 * len(sizes) // 3
        assert all(250 <= s < 500 for s in sizes)


def test_load_corpus_round_trips_to_items(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"seed": 7, "instances": make_corpus(7, 1)}))
    loaded = load_corpus(path)
    first = loaded[0]
    assert first.id == "uniform-00"
    assert lower_bound(first.capacity, first.items) >= 1
    assert first.items[0].id == "uniform-00-i00"


def test_persisted_corpus_is_valid():
    instances = load_corpus(CORPUS)
    assert len(instances) == 40
    for inst in instances:
        lower_bound(inst.capacity, inst.items)  # validates
