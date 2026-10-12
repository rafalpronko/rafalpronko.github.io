"""Walk-through on hand-chosen synthetic vectors.

Nothing here is real encoder output. The numbers were picked by hand to make
the arithmetic visible; they say nothing about retrieval quality.
"""
from cosine_lab import Recipe, check_same_recipe, cosine_similarity, mean_pool

# 1. Token IDs depend on the tokenizer. Two toy word-level vocabularies:
vocab_a = {
    "refund": 7, "policy": 3, "shipping": 5,
    "delay": 9, "not": 1, "approved": 4,
}
vocab_b = {
    "refund": 2, "policy": 11, "shipping": 6,
    "delay": 8, "not": 3, "approved": 10,
}
text = "refund policy"
print("1. token IDs")
print("  tokenizer A:", [vocab_a[w] for w in text.split()])
print("  tokenizer B:", [vocab_b[w] for w in text.split()])

# 2. A learned lookup table maps an ID to a row. Here the rows are hand-chosen.
#    Real dimensions are not labelled; these four are, for readability only.
table_a = {  # dims: money, approval, logistics, time
    7: (0.9, 0.2, 0.0, 0.1),   # refund
    3: (0.4, 0.3, 0.1, 0.0),   # policy
    5: (0.0, 0.0, 0.9, 0.2),   # shipping
    9: (0.0, -0.1, 0.4, 0.8),  # delay
    1: (0.0, -0.2, 0.0, 0.0),  # not
    4: (0.1, 0.9, 0.0, 0.0),   # approved
}


def embed(sentence: str) -> tuple[float, ...]:
    """Toy static encoder: look up each token row, then mean-pool."""
    return mean_pool([table_a[vocab_a[w]] for w in sentence.split()])


print("\n2. lookup row for 'refund' (ID 7):", table_a[7])

# 3. Sentence vectors and a ranking by cosine similarity.
query = "refund policy"
documents = ["refund approved", "shipping delay", "delay policy"]
print(f"\n3. ranking for query {query!r}")
q = embed(query)
for doc in sorted(
    documents, key=lambda d: cosine_similarity(q, embed(d)), reverse=True
):
    print(f"  {cosine_similarity(q, embed(doc)):+.3f}  {doc}")

# 4. Mean pooling of static vectors barely notices negation.
yes, no = embed("refund approved"), embed("refund not approved")
print("\n4. negation")
print(
    "  cos('refund approved', 'refund not approved') = "
    f"{cosine_similarity(yes, no):+.3f}"
)

# 5. Not a probability: scores can be negative and do not sum to 1.
print("\n5. not a probability")
score = cosine_similarity((1.0, 0.0), (-1.0, 0.2))
print(f"  cos((1, 0), (-1, 0.2)) = {score:+.3f}")

# 6. Same dimension, different encoder version: simulate a retrained model by
#    permuting coordinates. Within one version the geometry is identical;
#    across versions the score is meaningless even though the shapes match.
def v2(v: tuple[float, ...]) -> tuple[float, ...]:
    return (v[2], v[3], v[0], v[1])


doc = "refund approved"
print("\n6. encoder version mismatch")
q1, d1 = embed(query), embed(doc)
print(f"  v1 query vs v1 doc: {cosine_similarity(q1, d1):+.3f}")
print(f"  v2 query vs v2 doc: {cosine_similarity(v2(q1), v2(d1)):+.3f}")
print(f"  v2 query vs v1 doc: {cosine_similarity(v2(q1), d1):+.3f}")
# Equality is appropriate only for this symmetric toy encoder.
index_recipe = Recipe(
    encoder="toy-static", version="1", pooling="mean", normalized=False
)
query_recipe = Recipe(
    encoder="toy-static", version="2", pooling="mean", normalized=False
)
try:
    check_same_recipe(index_recipe, query_recipe)
except ValueError as err:
    print("  refused:", err)
