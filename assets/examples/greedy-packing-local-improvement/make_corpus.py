"""Write data/corpus.json. Run: uv run --python 3.11 python make_corpus.py"""
import json
from pathlib import Path

from packing.corpus import make_corpus

SEED, PER_FAMILY = 20240926, 20

path = Path(__file__).resolve().parent / "data" / "corpus.json"
path.parent.mkdir(exist_ok=True)
corpus = {"seed": SEED, "per_family": PER_FAMILY,
          "instances": make_corpus(SEED, PER_FAMILY)}
path.write_text(json.dumps(corpus, indent=1) + "\n", encoding="utf-8")
print(f"wrote {len(corpus['instances'])} instances to data/corpus.json")
