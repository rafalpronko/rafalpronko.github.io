---
layout: post
title: "Text classification: TF-IDF versus a small embedding model"
description: "A small, honest CPU comparison of TF-IDF + logistic regression and a tiny from-scratch embedding classifier, with the mechanics that decide what each model can and cannot know."
date: 2024-05-09
lang: en
author: Rafał Prońko
permalink: /blog/tfidf-vs-embedding-classifier/
categories: [nlp, machine-learning]
image: /assets/images/tfidf-vs-embedding-classifier.svg
---

When someone proposes "let's use embeddings" for an intent classifier, the
useful question is not which model is better in general. It is: what does
each model actually learn from your training data, and what happens to the
inputs it has never seen?

This post compares two small classifiers on the same synthetic intent data:

- **A.** scikit-learn `TfidfVectorizer` + `LogisticRegression`;
- **B.** a PyTorch `nn.Embedding` trained from scratch, masked mean pooling,
  and one linear layer.

Both compute on CPU without downloading a pretrained model. The goal
is to make the mechanisms visible. This small synthetic comparison does
not establish a universal winner or a runtime expectation.

> **Dating note.** The editorial date of this post is 2024-05-09. The code
> was executed on **2026-10-12** against a dependency set restricted to files
> uploaded to PyPI before 2024-05-09: torch 2.3.0, scikit-learn 1.4.2,
> numpy 1.26.4. The restriction is enforced with uv's `exclude-newer`
> setting, which is modern tooling used to reconstruct the stack, not
> something the post assumes you had in 2024. The interpreter was CPython
> 3.12.11, a later patch release of the 3.12 series that was already
> current in May 2024; `exclude-newer` does not apply to the interpreter.

![Two pipelines sharing one tokenizer and one fixed split](/assets/images/tfidf-vs-embedding-classifier.svg)

## The setup in one paragraph

A seeded generator produces short banking-style requests for eight intents
(`check_balance`, `transfer_money`, `lost_card`, `activate_card`,
`change_address`, `open_account`, `close_account`, `dispute_charge`) from
templates, slot values, optional greetings and a 6% per-token probability
of attempting an adjacent-letter swap. Exact duplicates are removed
**before** splitting. The remaining 1,527 texts go into one stratified
60/20/20 split: 916 train, 305 validation, 306 test. Both models get
exactly the same three lists. Hyper-parameters are picked on validation;
test is scored once per model at the end.

The full project (`pyproject.toml`, `uv.lock`, code, tests) is available as
[tfidf-vs-embedding-classifier.zip](/assets/downloads/tfidf-vs-embedding-classifier.zip).
From the extracted `tfidf-vs-embedding-classifier/` directory, uv (the
modern tool from the dating note) installs the locked 2024 packages, runs
the tests and prints the results as JSON:

```bash
uv sync --locked
uv run --locked pytest -v
uv run --locked python -m textclf.run
```

The snippets below are the parts that matter.

## Rules that apply to both models

These are boring, and they are where most text classification comparisons go
wrong.

**One split, fixed in advance.** If model A is evaluated on one random split
and model B on another, you are partly comparing splits.

```python
x_train, x_rest, y_train, y_rest = train_test_split(
    texts, labels, test_size=0.4, stratify=labels, random_state=seed)
x_val, x_test, y_val, y_test = train_test_split(
    x_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
```

**Remove exact duplicates before splitting.** A test asserts that no
identical string appears in two splits. That prevents exact-string overlap,
but **does not remove near-duplicates or shared-template leakage**: all
three splits draw from the same 40 templates and slot pools. Requests can
differ only in a greeting, slot, suffix or typo. This measures recognition
within the generator. To test new wording, reserve entire template families
for validation and test, grouping before generation/splitting; also use a
holdout of independently collected requests.

**Fit preprocessing on train only.** The TF-IDF vocabulary and IDF weights,
and the embedding model's token-to-id map, are statistics of the training
split. Computing them on all data leaks test-set information: the
vocabulary would contain test-only words, and IDF would reflect test
document frequencies.

**Tune on validation, not test.** Every grid choice and the best epoch are
selected by validation macro-F1. If you look at test numbers while choosing
settings, the test set has become a second validation set and its score is
optimistic. Selection compares **full-precision** F1, with strict `>`:
exact ties keep the first candidate and the first epoch. Scores are rounded
only when constructing the displayed JSON. Comparing against a rounded
incumbent can let an equal or worse checkpoint replace it; the regression
tests exercise that case for checkpoints and both grids.

**Same tokenizer.** Both models use one function, so differences come from
the models, not from tokenisation:

```python
_TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())   # "" or "!!!" -> []
```

This is deliberately ASCII-only: `café` becomes `caf`, and other non-ASCII
letters are truncated or dropped. It is unsuitable as-is for multilingual
requests.

## Model A: TF-IDF + logistic regression

```python
def build(ngram_range=(1, 2), C=3.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                                  lowercase=False, ngram_range=ngram_range,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])
```

Putting the vectoriser inside a `Pipeline` is the simplest leakage guard:
`fit(x_train, y_train)` fits the vocabulary, IDF and weights on exactly the
data you pass, and `predict` only calls `transform`. A test checks that
transforming new text leaves `vocabulary_` and `idf_` unchanged.

What the representation knows:

- Each document becomes a sparse vector with one dimension per training
  n-gram. With scikit-learn's default `smooth_idf=True`, IDF is computed as
  if one extra document contained every term, which avoids division by zero.
- **Unseen words are dropped.** "misplaced" was never in training, so it has
  no column. It contributes nothing.
- **Empty or fully unseen text gives a zero vector.** Logistic regression
  then returns only its intercepts, so the prediction is the class with the
  largest intercept. That is a silent default, not an error.
- **Unigrams ignore order; bigrams keep local order.** "from savings to
  checking" and "from checking to savings" are identical as unigram bags and
  differ as bigram vectors (tested).

The validation grid covers `ngram_range` in `{(1,1), (1,2)}` and `C` in
`{0.3, 3, 30}`.

## Model B: a tiny embedding classifier from scratch

### Vocabulary, PAD and UNK

```python
PAD, UNK = 0, 1

class Vocab:
    def __init__(self, train_texts, min_freq=2):
        counts = Counter(tok for t in train_texts for tok in tokenize(t))
        kept = sorted(tok for tok, c in counts.items() if c >= min_freq)
        self.itos = ["<pad>", "<unk>", *kept]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(tok, UNK) for tok in tokenize(text)]

    def batch(self, texts):
        ids = [self.encode(t) for t in texts]
        width = max(1, max((len(x) for x in ids), default=0))
        out = torch.full((len(ids), width), PAD, dtype=torch.long)
        for row, seq in enumerate(ids):
            if seq:
                out[row, :len(seq)] = torch.tensor(seq)
        return out
```

Two details matter more than they look:

- **`min_freq=2` is how UNK gets trained.** If every training token got its
  own id, UNK would never appear during training and its vector would stay at
  its random initial value. Mapping training tokens seen only once (here
  mostly typos) to UNK gives the model examples of "a word I don't know".
  The cost: whatever those rare words correlated with in training is now
  baked into the UNK vector.
- **Empty text becomes one PAD position**, not a zero-width tensor, so the
  batch shape stays valid.

### Masked mean pooling

```python
class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab_size, n_classes, dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.out = nn.Linear(dim, n_classes)

    def pooled(self, ids):
        mask = (ids != PAD).unsqueeze(-1).float()      # (B, T, 1)
        summed = (self.embed(ids) * mask).sum(dim=1)   # (B, D)
        count = mask.sum(dim=1).clamp(min=1.0)         # empty text: 0 / 1
        return summed / count

    def forward(self, ids):
        return self.out(self.pooled(ids))
```

- `padding_idx=PAD` makes PyTorch initialise the padding row to zeros and
  exclude it from gradient updates. A test checks both.
- The mask matters anyway. Without it, a mean over a padded row divides by
  the padded width, so the same sentence would get different vectors in
  batches with different maximum lengths. With the mask, adding padding
  does not change the output (tested).
- `clamp(min=1.0)` avoids `0/0 = NaN` for empty text. The pooled vector is
  then zero and the logits equal the linear layer's bias, as with model
  A's intercept-only empty input. All-unknown text is different: its pooled
  vector is the learned UNK row, not zero.
- **Mean pooling is order-insensitive.** The model sees a multiset of
  tokens. "from savings to checking" and "from checking to savings" give the
  same prediction. Their logits can still differ in the last float bits:
  floating-point addition is not associative, so summing the same vectors
  in a different order can round differently. Compare with a tolerance,
  not `torch.equal`.

### These embeddings are not "semantic"

This is the point most often blurred in discussions. The `nn.Embedding`
here starts random and is trained only on 916 short texts to minimise
classification loss. Its vectors encode "which intent does this token push
towards", learned from co-occurrence with labels in this training set. It
has no idea that "misplaced" is close to "lost" or that "funds" relates to
"balance", unless those words appeared with those labels in training.
Pretrained sentence or word embeddings would bring that kind of prior
knowledge; a from-scratch embedding layer does not.

### Training and selection

Each training batch contains integer token ids of shape `(B, T)` and a
label index of shape `(B,)`. Class names are sorted once; the target is the
index of the true intent in that list. The forward pass produces `(B, 8)`
raw logits. `nn.CrossEntropyLoss` compares those logits with the label
indices; do not apply softmax before passing them to this loss. Backpropagation
and Adam update the embeddings and linear weights/bias, excluding the PAD
row from gradient updates.

```python
labels = sorted(set(y_train))
y_idx = torch.tensor([labels.index(y) for y in y_train])
loss = nn.CrossEntropyLoss()(model(x_ids[idx]), y_idx[idx])
# After zero_grad(): loss.backward(); opt.step()
```

Learning rate is 0.01, batch size 32, and training runs for 30 epochs.
`torch.manual_seed` sets initialisation; a separate seeded `Generator`
controls shuffling. After each epoch, validation macro-F1 selects a cloned
state dict using full precision. Exact ties retain the earliest epoch.
Dimensions 8 and 32 are compared with the same policy. One torch CPU thread
is used. At inference, encode with the fitted vocabulary, pool, compute
logits, take `argmax`, and map the index back through `labels`; no labels or
loss are needed for prediction.

## What I observed

All numbers come from one run of `uv run --locked python -m textclf.run` on
2026-10-12, on an arm64 Mac (`platform.platform()` reported
`macOS-26.5.2-arm64-arm-64bit`), Python 3.12.11, torch 2.3.0,
scikit-learn 1.4.2, numpy 1.26.4, one torch thread. Selection uses the
full-precision rule described above.

| | TF-IDF + LogReg | Embedding + mean pool |
|---|---|---|
| features / vocab (train only) | 297 n-grams | 179 ids incl. PAD and UNK |
| selected on validation | unigrams, C = 3 (first exact tie) | dim 32, epoch 3 |
| validation macro-F1 | 1.0000 | 0.9966 |
| test accuracy | 1.0000 | 1.0000 |
| test macro-F1 | 1.0000 | 1.0000 |
| stored parameter values | 2,384 (8 × 297 + 8) | 5,992 (179 × 32 + 32 × 8 + 8) |
| values eligible for updates | 2,384 | 5,960 (32 PAD values stay zero) |

Four TF-IDF candidates have an exact validation macro-F1 of 1.0:
unigrams or unigrams+bigrams with C = 3 or 30. Unigrams with C = 3 are
selected because they occur first in grid order, not because they score
better. Embedding dim 8 reaches 0.9938 at epoch 12; dim 32 reaches 0.9966
at epoch 3. Display rounding does not participate in either selection.

Both models classify all 306 test texts correctly. The shared templates
and distinctive keywords make this an easy in-generator task, with
near-duplicates across splits. The saturated scores cannot distinguish the
models or estimate performance on independently written requests.

The JSON records wall-clock times for diagnostics only. They are not a
controlled speed benchmark and do not promise a particular runtime.
Measure a controlled workload on your target hardware if latency matters.

### Where they differ: inputs outside the training vocabulary

Six hand-written probe sentences use words the generator never emits. They
are an illustration, not a test set: six examples prove nothing about rates.

| probe | expected | TF-IDF | embedding | UNK share |
|---|---|---|---|---|
| i misplaced my plastic | lost_card | lost_card | dispute_charge | 0.50 |
| what are my remaining funds | check_balance | check_balance | dispute_charge | 0.60 |
| wire cash to my sister | transfer_money | transfer_money | transfer_money | 0.60 |
| i relocated last week | change_address | dispute_charge | dispute_charge | 0.75 |
| unfamiliar debit on my statement | dispute_charge | activate_card | change_address | 0.40 |
| enable the replacement plastic | activate_card | activate_card | dispute_charge | 0.75 |

How the two models handle these differently:

- **TF-IDF ignores the unknown words** and decides on the remaining known
  unigrams (validation selected unigrams), such as "what", "to", "my" or
  "the". Sometimes that is enough; sometimes the remaining words are
  uninformative or misleading ("relocated", "last" and "week" carry the
  meaning of the fourth probe and none is in the vocabulary; in the fifth,
  the known word "debit" is a card type and pulls towards a card intent).
- **The embedding model replaces unknown words with a learned UNK
  vector.** Its share weights that vector in the mean, alongside all known
  token vectors. The JSON exposes the UNK-only representation, all eight
  logits and its prediction: `dispute_charge` (logit 4.7629), versus 1.0493
  for `change_address`. Thus the fifth probe's `change_address` prediction
  cannot be attributed to UNK alone; the known-token mixture matters.
  Rare training words teach this shared vector a label-dependent pattern,
  not the meanings of unseen words.

Empty input and unknown-only input take different paths:

| input | TF-IDF representation → prediction | embedding representation → prediction |
|---|---|---|
| `""` | zero features; intercept only → transfer_money | all PAD; zero pooled vector; bias only → dispute_charge |
| `"zzz qqq"` | zero features; intercept only → transfer_money | `[UNK, UNK]`; learned UNK vector plus bias → dispute_charge |

The two embedding predictions happen to agree; their representations differ.
The JSON records both pooled vectors. An empty string still
gets a label, without evidence of calibrated confidence. A real input flow
should explicitly handle empty and no-known-token requests, using rejection
or routing appropriate to the application.

## Reproducibility, and its scope

The tests check that training twice with the same seed, in the same process,
with one thread, yields bitwise-identical weights, and that a different seed
yields different weights. This is an observation under those conditions,
not a guarantee for future runs or other environments. [PyTorch's versioned
reproducibility notes](https://pytorch.org/docs/2.3/notes/randomness.html)
explain the limits across releases, platforms and CPU/GPU. Thread count can
change floating-point reduction order; scores, selected epochs and
predictions may change elsewhere.

## Tests worth copying

The download contains 24 passing test cases in four files:

| file | checks |
|---|---|
| `test_embedding.py` | train-only vocabulary, rare-to-UNK mapping, empty/PAD behaviour, order and same-process seeds |
| `test_tfidf_and_data.py` | fitted vocabulary/IDF, intercept-only inputs, ASCII limits, n-gram order, exact split disjointness and per-class proportions |
| `test_selection.py` | worse/better scores within one rounding bucket, exact grid ties and first-epoch checkpoint weights |
| `test_reporting.py` | display rounding leaves raw F1 intact, UNK vector/logits/prediction, stored versus update-eligible counts |

These tests target handling and selection bugs that a saturated accuracy
number would hide. They do not establish real-world model quality.

## Limitations

- **Synthetic, shared-template data.** Exact deduplication leaves
  near-duplicates and shared wording across splits. Use template-family
  group holdouts and independently collected requests for generalisation
  claims; real text also varies in length, language and time.
- **Small grids, one seed.** I tuned two or three values per knob and trained
  the embedding model with one seed. Seed variance is not measured.
- **The embedding model is deliberately minimal.** No pretrained vectors, no
  subword units, no sequence model, no regularisation. Each of those changes
  what it can do with unseen words and word order.
- **TF-IDF has its own easy upgrades** that I did not use, such as character
  n-grams (`analyzer="char_wb"`), which represent subword fragments.
  Their effect on typos and unseen inflections needs a separate evaluation.
- **Probes are anecdotes.** Six sentences, chosen by hand.

## How I would choose

Start with TF-IDF + logistic regression as an inspectable baseline for
short, keyword-driven intents, then test it on a representative holdout.
A from-scratch embedding model of this size does not add prior
knowledge; it adds training choices (vocabulary cut-off, UNK handling,
pooling, epochs, seeds) that you then have to test. It becomes interesting
when you plan to grow it: pretrained initialisation, subword tokenisation,
or a sequence encoder that can use word order.

Whatever you pick, the comparison is only as good as its protocol: one
fixed split with the right grouping for the generalisation question;
preprocessing fitted on train only; full-precision choices made on
validation; test touched once; and explicit tests for empty, padded and
unknown input.

## Source references

The [scikit-learn 1.4 TF-IDF API](https://scikit-learn.org/1.4/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
describes train-fitted vocabulary/IDF and smoothing. The
[logistic regression API](https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html)
documents the classifier and regularisation. PyTorch's
[versioned Embedding API](https://pytorch.org/docs/2.2/generated/torch.nn.Embedding.html)
documents `padding_idx`; that contract predates the 2.3.0 stack used here.
The [PyTorch 2.3.0 release](https://github.com/pytorch/pytorch/releases/tag/v2.3.0)
is dated 24 April 2024, before the editorial label. These documentation and
release references were checked on 12 October 2026; the execution date is
separate from the editorial date.
