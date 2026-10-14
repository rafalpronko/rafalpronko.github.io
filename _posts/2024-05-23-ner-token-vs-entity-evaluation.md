---
layout: post
lang: en
title: "NER evaluation: token scores, whole entities and boundary errors"
date: 2024-05-23 09:00:00 +0200
permalink: /blog/ner-token-vs-entity-evaluation/
author: Rafał Prońko
categories: [nlp]
description: "Why token accuracy flatters a NER model, how exact entity matching with document, half-open span and type works, and a strict IOB2 scorer in plain Python."
image: /assets/images/ner-token-vs-entity-evaluation.svg
---

# NER evaluation: token scores, whole entities and boundary errors

![Two panels. Top: the gold ORG entity "Nordwave Data Labs" covers tokens 4 to 7, the prediction stops after "Data". Two of three ORG tokens are correct, yet at entity level this is one false positive and one false negative. Bottom: gold and predicted LOC "Gdańsk" cover exactly tokens 8 to 9, one true positive.]({{ '/assets/images/ner-token-vs-entity-evaluation.svg' | relative_url }})

<time datetime="2024-05-23">23 May 2024</time>

A named entity recognition model tags every token. The application usually needs something else: a whole name, a whole company, a whole address. That's where evaluation tends to go wrong.

If a model tags “Nordwave Data” instead of “Nordwave Data Labs”, two out of three tokens are right. A token-level metric rewards that. A system that links the organisation to a customer database gets the wrong string and either fails the lookup or finds a different company.

So before I look at a score, I ask what unit is being counted. On the same predictions, the answer can move the number from 0.82 to 0.29. Below I build a small strict scorer in plain Python, run it on a synthetic example and show where the gap comes from.

## Tags are not entities

Most token-classification NER models predict one tag per token in a BIO-style scheme. In **IOB2**:

- `B-X` starts an entity of type `X`,
- `I-X` continues an entity of type `X`,
- `O` means the token is outside any entity.

```text
Anna    B-PER
Maria   I-PER
Nowak   I-PER
joined  O
Nordwave B-ORG
Data    I-ORG
Labs    I-ORG
```

The entities are not stored anywhere. They have to be decoded from the tag sequence. That decoding step is part of the evaluation, and it's where most of the decisions hide.

A note on names: the CoNLL-2003 shared task data used an older variant, usually called **IOB1**: `I-X` is the default tag inside an entity, and `B-X` appears only when two entities of the same type are directly adjacent. Many tools and models today use IOB2, where every entity starts with `B-`. If you mix data from both conventions without converting, a strict IOB2 decoder will reject valid IOB1 sequences — and a lenient one will hide the mismatch. I only handle IOB2 here.

Two more assumptions: entities are **flat** (no entity inside another) and **non-overlapping**. CoNLL-2003 makes the same assumption and, when a name is embedded in another name, usually annotates only the outer one. Nested NER needs a different representation and a different scorer.

## Three numbers from the same predictions

I wrote a tiny synthetic corpus by hand: two documents, four sentences, 33 tokens, seven gold entities. The names are invented. The “predictions” are not from a model — I wrote them to contain typical mistakes: a truncated organisation, two people merged into one span, an event split in two, and a city tagged with the wrong type.

The scorer reports three views of the same pairs of sequences:

| Metric | What is counted | Result |
|---|---|---|
| Token accuracy | every token, including `O` | 27/33 = 0.818 |
| Token micro F1 | tokens with a non-`O` gold or predicted tag | 20/30 = 0.667 |
| Entity micro F1 | whole entities, exact match | 4/14 = 0.286 |

I keep fractions next to the decimals deliberately. 4/14 says more than 0.286: it says how few entities stand behind the number.

### Token accuracy and O inflation

Token accuracy is the share of tokens whose predicted tag equals the gold tag. The problem is the denominator. In ordinary text most tokens are `O`, and every correctly predicted `O` adds to the score.

One of the tests makes this explicit: a sentence of 100 tokens with one two-token location. A “model” that predicts `O` everywhere gets 98/100 = 0.98 token accuracy and finds zero entities. Its entity recall is 0/1. Its entity precision is undefined, because it predicted nothing.

That last point matters. Precision with zero predictions is 0/0. I don't silently turn it into 0 or 1 — either choice changes averages and hides the fact that there was nothing to measure. The scorer keeps the denominator and prints `undefined (0/0)`.

### Token micro F1

A common improvement is to drop `O` as a class and compute micro-averaged precision and recall over the remaining tags. In my scorer a token is:

- a true positive if gold and prediction are the same non-`O` tag,
- a false positive if the prediction is non-`O` and differs from gold,
- a false negative if the gold is non-`O` and differs from the prediction.

A token with gold `I-PER` and prediction `B-PER` is both a false positive and a false negative. I compare full tags, including the `B-`/`I-` prefix. If you compare only types, split and merge errors become even less visible.

This removes the `O` inflation, but it still counts tokens. The truncated “Nordwave Data” contributes two true positives and one false negative. The application, which needs the whole name, got zero usable entities.

### Entity micro F1 with exact matching

At entity level I decode both sequences into spans and compare sets. CoNLL-2003 defines this clearly: precision is the share of entities found by the system that are correct, recall is the share of gold entities found, and “a named entity is correct only if it is an exact match of the corresponding entity in the data file.”

I make “exact match” concrete as four fields:

```python
Span(doc, start, end, type)
```

- `doc` — the document index. The same offsets in two documents are different entities.
- `start`, `end` — a **half-open** token range `[start, end)`, counted from the start of the document. `Span(0, 4, 7, "ORG")` covers tokens 4, 5 and 6. Half-open ranges make lengths trivial (`end - start`) and avoid off-by-one arguments about whether `end` is included.
- `type` — the entity type.

Two spans match only if all four fields are equal. Then:

```text
TP = |gold ∩ pred|      FP = |pred − gold|      FN = |gold − pred|
```

and F1 is computed as `2·TP / (2·TP + FP + FN)`. When precision and recall are both defined and not both zero (P + R > 0), this is the same number as their harmonic mean `2PR / (P + R)`. The count form also covers the edge cases: with TP = 0 and at least one error it gives 0, where the harmonic mean would be 0/0, and it is undefined only when there are no gold and no predicted entities at all.

## A boundary error costs twice

Look at the truncated organisation again:

```text
gold  Span(0, 4, 7, "ORG")  'Nordwave Data Labs'
pred  Span(0, 4, 6, "ORG")  'Nordwave Data'
```

These are different spans. The predicted span is a **false positive** — the system claimed an entity that isn't in the gold data. The gold span is a **false negative** — the system did not find it. One boundary mistake lowers both precision and recall.

This is not a flaw of exact matching; it's what exact matching means. A model that detects no entity there would get the same FN and no FP; the wrong boundary adds an extra FP on top. In exact-match F1, being almost right never scores higher than abstaining, and it lowers the aggregate F1 whenever there is at least one true positive (TP > 0) in the same row or average. With TP = 0 both are 0: in the demo below the ORG row is 0/3 with the truncated span and would be 0/2 without it. Keep that in mind when a model with higher recall in a manual review has a lower F1 than a more conservative one.

Merges and splits are the same phenomenon with more spans:

- **merge**: gold `Jan Kowalski` and `Ewa Lis` (two PER), prediction `Jan Kowalski and Ewa Lis` (one PER) → 1 FP, 2 FN;
- **split**: gold `Vistula Data Summit` (MISC), prediction `Vistula` (LOC) + `Data Summit` (MISC) → 2 FP, 1 FN.

A wrong type with correct boundaries — `Warsaw` as ORG instead of LOC — is also one FP and one FN, but now in two different rows of the per-type table.

## The scorer

The complete example is downloadable:

- [`ner_eval.py`]({{ '/assets/examples/ner-token-vs-entity-evaluation/ner_eval.py' | relative_url }}) — scorer, subword helper, three-column reader, report formatter and the synthetic demo (standard library only). They are in one file on purpose, so the example is a single download; in a library I would split the reader, formatter and demo out of the scoring module,
- [`test_ner_eval.py`]({{ '/assets/examples/ner-token-vs-entity-evaluation/test_ner_eval.py' | relative_url }}) — pytest tests,
- [`pyproject.toml`]({{ '/assets/examples/ner-token-vs-entity-evaluation/pyproject.toml' | relative_url }}) and [`uv.lock`]({{ '/assets/examples/ner-token-vs-entity-evaluation/uv.lock' | relative_url }}) — Python ≥ 3.11 and the pinned test dependency.

The input is three-level: a list of documents, each a list of sentences, each a list of tags. The core is the decoder. It turns tags into spans and fails on anything that is not valid IOB2:

```python
def _extract(
    corpus: Corpus, name: str, on_invalid: str
) -> tuple[list[Span], int]:
    spans: list[Span] = []
    repairs = 0
    for d, doc in enumerate(corpus):
        offset = 0  # token offset of the current sentence in the document
        for s, sentence in enumerate(doc):
            # Entity state never crosses a sentence boundary.
            open_type: str | None = None
            start = 0
            for i, tag in enumerate(sentence):
                where = f"{name} document {d} sentence {s} token {i}"
                try:
                    prefix, entity_type = parse_tag(tag)
                except BioError as error:
                    raise BioError(f"{where}: {error}") from None
                if prefix == "I" and entity_type != open_type:
                    if on_invalid == "raise":
                        after = (
                            "sentence start"
                            if i == 0
                            else repr(sentence[i - 1])
                        )
                        raise BioError(
                            f"{where}: {tag} after {after} is not valid IOB2"
                        )
                    # "as-begin": an orphan I-X opens a new X entity.
                    prefix = "B"
                    repairs += 1
                if prefix != "I" and open_type is not None:
                    spans.append(
                        Span(d, offset + start, offset + i, open_type)
                    )
                    open_type = None
                if prefix == "B":
                    open_type, start = entity_type, i
            if open_type is not None:
                spans.append(
                    Span(d, offset + start, offset + len(sentence), open_type)
                )
            offset += len(sentence)
    return spans, repairs
```

Three details in this loop are deliberate.

**Sentence boundaries are kept.** Offsets are counted per document, but the decoder state is reset at every sentence. If I flattened a document into one long tag list, a sentence ending in `B-PER` followed by a sentence starting with `I-PER` would quietly produce one entity spanning two sentences. Here it's an error.

**Malformed tags fail.** `parse_tag` accepts only `O`, `B-<type>` and `I-<type>` with a non-empty type without whitespace. `b-PER`, `B_PER`, `E-PER`, `B-`, an empty string or `None` raise `BioError` with the document, sentence and token position. I'd rather stop an evaluation than score a file whose tags I have misread.

**Length mismatches fail.** Before decoding, `check_shapes` compares the number of documents, sentences per document and tags per sentence, and every loop uses `zip(..., strict=True)`. A plain `zip` truncates to the shorter input. If a prediction file lost its last sentence, an ordinary `zip` would score the remaining part and report a perfectly reasonable-looking number.

The counts are kept as integers until the end:

```python
class Ratio(NamedTuple):
    """A fraction that keeps its denominator; 0/0 is undefined, not zero."""

    num: int
    den: int

    @property
    def value(self) -> float | None:
        return None if self.den == 0 else self.num / self.den
```

## What to do with an orphan `I-` tag

A model that predicts tags independently per token can produce sequences that are not valid IOB2: `O I-PER`, `B-ORG I-PER`, or `I-PER` at the start of a sentence. The gold data should never contain them; if it does, the annotation or the conversion is broken.

For predictions there are two reasonable policies, and I want the choice to be visible:

- `on_invalid="raise"` (default) — stop with an error that says where the problem is.
- `on_invalid="as-begin"` — treat an orphan `I-X` as the start of a new `X` entity, and count how many times this happened.

Gold is always decoded strictly, regardless of the policy.

The seqeval library documents the same distinction. Its default mode is described as compatible with `conlleval`; its strict mode decodes according to a chosen scheme such as IOB2. The README's minimal example uses gold `B-NP I-NP O` and prediction `I-NP I-NP O`: the default mode reports F1 1.00, strict IOB2 mode reports 0.00. I don't use seqeval here and I haven't run it; I'm quoting its documented behaviour as a reference point. My scorer goes one step further in the default case and refuses to produce a number at all.

Which policy is right depends on what the decoder in production does. If the application repairs orphan tags the same way before using entities, evaluate with the same repair and report the repair count. If the application would crash or drop such output, evaluation should not hide it.

## Per-type scores and supports

A micro average pools all entities. That's useful, but it weights frequent types more. The scorer prints one row per type, with the **support** — the number of gold entities of that type — and the number of predictions:

```text
  type   support  pred  TP  FP  FN  precision        recall           F1
  LOC          2     2   1   1   1  1/2 = 0.500      1/2 = 0.500      2/4 = 0.500
  MISC         1     1   0   1   1  0/1 = 0.000      0/1 = 0.000      0/2 = 0.000
  ORG          1     2   0   2   1  0/2 = 0.000      0/1 = 0.000      0/3 = 0.000
  PER          3     2   1   1   2  1/2 = 0.500      1/3 = 0.333      2/5 = 0.400
  micro        7     7   2   5   5  2/7 = 0.286      2/7 = 0.286      4/14 = 0.286
```

With supports of 1–3 entities these per-type numbers are anecdotes, not estimates. On real held-out data the same table quickly shows when a good micro F1 comes from a large, easy type while a rare and important one barely works. A type that appears only in predictions has support 0 and an undefined recall — not zero recall.

## Partial matches: a diagnosis, not a score

Exact matching is harsh, and it's tempting to give partial credit for overlapping spans. Some evaluation schemes do. The catch is that “partial match” is not one definition: overlap with the same type, any overlap, a minimum overlap ratio, half credit or full credit. Two reports that both say “partial F1” can be computing different things.

I keep exact-match counts as the score and add a separate diagnosis of the errors:

```text
Diagnosis (does not change the scores): exact=2, wrong_type=1, boundary_same_type=3, boundary_and_type=1, spurious=0, missed=0
```

Each distinct predicted span gets exactly one category: the first that applies, in this order of precedence — an exact match, a wrong type with correct boundaries, a boundary error with the right type (it overlaps at least one gold entity of its type), a boundary error with a different type (it overlaps only gold entities of other types), or spurious (no overlap with any gold entity). Gold entities with no overlapping prediction are counted as missed.

This is not a one-to-one matching between gold and predicted spans. A merge is one prediction and is counted once, however many gold entities it covers; a split counts once per predicted piece; and one gold entity can explain several predictions. So the first five numbers always add up to the number of distinct predictions — 2 + 1 + 3 + 1 + 0 = 7 here, with the split event contributing one `boundary_and_type` and one `boundary_same_type` — while `missed` is a separate count over gold entities.

This tells me what to fix. In the synthetic example nothing is entirely missed or invented; most of the damage is boundaries. That points at different work — tokenisation, annotation guidelines for titles and suffixes, or decoding — than a model that misses entities altogether.

If you do report a partial-match score, name the matching rule, report exact match next to it, and don't compare it with exact-match numbers from elsewhere.

## Subwords and padding

Transformer models usually see subword pieces, not words. Gold annotation is usually per word. Somewhere the two have to be aligned, and it's easy to evaluate at the wrong level.

The usual pattern is to keep the prediction for the first piece of each word. Many tokenizers can return a word index for every piece, with `None` for special tokens and padding. The example includes a small function that takes such a list as plain input:

```text
word_ids     [None, 0, 0, 1, 2, 2, 2, None, None, None]
subword tags ['O', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-PER', 'I-PER']
word tags    ['B-LOC', 'O', 'B-PER']
```

The last two positions are padding, and the model “predicted” `B-PER I-PER` there. If those positions are not removed before decoding, they become an entity that doesn't exist. When padding positions are excluded from the training loss — a common setup — nothing pushes their predictions towards `O`, so there is no reason to expect sensible tags there.

Pitfalls I check for:

- **Scoring pieces instead of words.** Long or rare words split into more pieces get more weight in a token score. Entity scores in piece offsets *can* change too: when predicted boundaries fall inside a word, or when gold and predictions are aligned to pieces inconsistently. Converting exact word boundaries to piece offsets consistently on both sides does not, by itself, change exact-match counts — the risk is in everything around that conversion.
- **Truncation.** If the input is cut at a maximum length, the last words have no prediction. The function raises `AlignmentError` when it recovers fewer words than the gold sentence has: `recovered 2 words, expected 3`. Silently scoring the shortened sentence would drop those gold entities from the denominator. The check has a limit: if the cut falls inside the final word and its first piece survives, the word is recovered and nothing in `word_ids` shows that its later pieces are missing. To catch that case, check the tokenizer's own truncation or overflow output as well, not only the recovered word count.
- **Non-contiguous pieces.** A word whose pieces are interrupted by another word or a special token indicates a broken mapping. That's also an error.

Only after collapsing to words do I call the same `evaluate` as for any other model.

## Run it

The scorer needs only the Python 3.11 standard library. I use [uv](https://docs.astral.sh/uv/) to run it with a fixed Python version and to install pytest for the tests. Put the four downloaded files into one directory and run:

```bash
uv run --python 3.11 python ner_eval.py
uv run --python 3.11 pytest
```

Without uv, `python3.11 ner_eval.py` runs the demo; the tests need `pytest` installed. To score your own data, write one token per line as `word gold pred`, an empty line between sentences and `-DOCSTART-` before each document, then:

```bash
uv run --python 3.11 python ner_eval.py my_predictions.txt
```

The `pyproject.toml` pins `pytest==8.2.1` and sets uv's `exclude-newer` to `2024-05-23`, so the lock file only contains packages published before this post's date.

The demo prints:

```text
Corpus: 2 documents, 4 sentences, 33 tokens
Token accuracy (all tags, O included): 27/33 = 0.818
Token micro scores (non-O tags, B-/I- prefix must match):
  TP=10 FP=5 FN=5
  precision 10/15 = 0.667 | recall 10/15 = 0.667 | F1 20/30 = 0.667
Entity scores (exact document, start, end and type):
  type   support  pred  TP  FP  FN  precision        recall           F1
  LOC          2     2   1   1   1  1/2 = 0.500      1/2 = 0.500      2/4 = 0.500
  MISC         1     1   0   1   1  0/1 = 0.000      0/1 = 0.000      0/2 = 0.000
  ORG          1     2   0   2   1  0/2 = 0.000      0/1 = 0.000      0/3 = 0.000
  PER          3     2   1   1   2  1/2 = 0.500      1/3 = 0.333      2/5 = 0.400
  micro        7     7   2   5   5  2/7 = 0.286      2/7 = 0.286      4/14 = 0.286
Span-level errors:
  FN gold (0, 4, 7, 'ORG') 'Nordwave Data Labs'
  FN gold (0, 12, 14, 'PER') 'Jan Kowalski'
  FN gold (0, 15, 17, 'PER') 'Ewa Lis'
  FN gold (1, 1, 4, 'MISC') 'Vistula Data Summit'
  FN gold (1, 7, 8, 'LOC') 'Warsaw'
  FP pred (0, 4, 6, 'ORG') 'Nordwave Data'
  FP pred (0, 12, 17, 'PER') 'Jan Kowalski and Ewa Lis'
  FP pred (1, 1, 2, 'LOC') 'Vistula'
  FP pred (1, 2, 4, 'MISC') 'Data Summit'
  FP pred (1, 7, 8, 'ORG') 'Warsaw'
Diagnosis (does not change the scores): exact=2, wrong_type=1, boundary_same_type=3, boundary_and_type=1, spurious=0, missed=0

== Orphan I- tag policy: gold O B-PER I-PER, pred O I-PER I-PER ==
on_invalid='raise'    -> BioError: pred document 0 sentence 0 token 1: I-PER after 'O' is not valid IOB2
on_invalid='as-begin' -> repairs=1, entity TP=1 FP=0 FN=0, token accuracy 2/3 = 0.667

== Subword alignment: 3 words, 10 positions with [CLS], [SEP] and padding ==
word_ids     [None, 0, 0, 1, 2, 2, 2, None, None, None]
subword tags ['O', 'B-LOC', 'I-LOC', 'O', 'B-PER', 'I-PER', 'O', 'O', 'B-PER', 'I-PER']
word tags    ['B-LOC', 'O', 'B-PER']
truncated input -> AlignmentError: recovered 2 words, expected 3
```

The orphan-tag case shows the same point from another angle: after the repair, the entity is an exact match (TP 1), while token accuracy is still 2/3, because the raw tag `I-PER` differs from the gold `B-PER`. Token metrics always use the raw tags.

Reproduction note: the output above was produced by running the downloadable files with uv 0.7.8 and CPython 3.11.12. Those runner versions are newer than this post's date; the scorer itself uses only standard-library features available in Python 3.11.0, released in October 2022. The date above is a historical editorial label; this example was executed and verified in October 2026, not in May 2024.

## What the tests check

The tests use only hand-written sequences. They cover:

1. an exact match (all counts in the right cells, F1 4/4),
2. a truncated entity (1 FP + 1 FN at entity level, 2/3 token accuracy),
3. a wrong type with correct boundaries (FN in one type's row, FP in another's),
4. a split and a merge,
5. an `O` majority: 98/100 token accuracy, entity recall 0/1, undefined precision,
6. invalid transitions — `I-` at the start, after `O`, after a different type, after an `O` gap — and malformed tags,
7. gold rejected even when predictions use the lenient policy,
8. multiple documents with the same offsets,
9. empty entity sets with explicit 0/0 denominators,
10. per-type supports, including a type that only appears in predictions,
11. mismatched numbers of documents, sentences and tags,
12. subword alignment with special tokens, padding, truncation and non-contiguous pieces,
13. the three-column reader and the exact demo numbers quoted in this post.

I wrote the tests first and ran them against a stub that raised `NotImplementedError`, so all of them failed before the implementation existed. In a separate check I also introduced one bug at a time into a copy of the scorer — resetting the decoder per document instead of per sentence, reporting 0/0 as 0, accepting orphan `I-` tags, removing the length check, and ignoring the type when building spans — and each change made at least one test fail.

That's evidence that the tests constrain these behaviours. It is not proof that the scorer is correct for every input.

## The data matters more than the formula

A strict scorer on a leaky test set gives a precise wrong answer. With NER, leakage is easy to create:

- **Sentences from one document in both train and test.** A random split by sentence lets the model see the same people and companies in training. Split by document, or by source and time.
- **Repeated templates and near-duplicates.** Generated or form-based text repeats patterns; news repeats the same story in several versions.
- **Entity memorisation.** If most test entities also appear in training, the score measures recall of a list. It's worth reporting the share of test entities whose surface string was seen in training, and scoring unseen ones separately.
- **Decisions made on the test set.** Choosing the invalid-tag policy, the subword strategy or a checkpoint by looking at the test score makes it a second development set.

The CoNLL-2003 setup is a useful reference: development data was there for tuning, and the split between development and test data was chosen “to avoid systems being tuned to the test data”. For English, the training and development texts came from the end of August 1996 and the test texts from December 1996, so the test set was also later in time. The paper also reports bootstrap-resampled significance boundaries instead of treating a single F1 as exact. With a few hundred test entities, a difference of one point between two models may be noise.

## Which error is expensive?

Exact-match F1 treats every FP and FN the same. Applications don't.

- In **de-identification** a missed name (FN) can leak personal data, while an extra masked word (FP) costs readability. A boundary error that leaves the surname unmasked is a leak, even if most of the name was caught.
- In **entity linking** or a CRM lookup, a truncated organisation name is often worse than no entity, because it links confidently to the wrong record.
- In **search highlighting** a span that is one token too long may be acceptable.

So I report exact-match entity scores as the main number, per-type rows with supports, and the error diagnosis — and then decide with the people who own the application which errors matter. Sometimes that means a separate metric, such as recall of a single critical type with a minimum acceptable level. Sometimes it means an error budget for boundary mistakes. It shouldn't be a single averaged number chosen because the library prints it.

## Limitations of this example

- Flat, non-overlapping entities and IOB2 only. IOB1, BIOES and nested entities need a different decoder.
- Tokens come from whitespace-separated columns. Character offsets, mismatched tokenisation between gold and prediction, and text normalisation are out of scope.
- Exact match only; the diagnosis is a description of errors, not an alternative score.
- No confidence intervals. For real comparisons I would add bootstrap resampling by document.
- The diagnosis compares every predicted span with every gold span, which is fine for a demo and slow for very large corpora.
- The data is synthetic and the “predictions” are hand-written. The numbers illustrate how the metrics behave. They say nothing about any model.

## Conclusion

Token accuracy answers “how many tags are right?”, and in NER most tags are `O`. Token micro F1 removes that inflation but still rewards half-found names. Entity-level exact matching answers the question applications usually ask — did we get the whole entity, in the right place, with the right type — and it charges a boundary error twice.

For a NER report I want: the matching rule spelled out (document, half-open span, type), counts with denominators, per-type rows with supports, an explicit policy for invalid tags, word-level alignment checked rather than assumed, and a held-out set that the model and I haven't tuned on. The scoring core — the types, the decoder, `evaluate` and `diagnose` — is about 250 non-blank lines of standard-library Python; the complete download is about 530 lines, because the subword helper, column reader, report formatter and demo live in the same file. The decisions around it are the part worth writing down.

### Sources

- Erik F. Tjong Kim Sang and Fien De Meulder, [Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition](https://aclanthology.org/W03-0419.pdf) — IOB data format, non-recursive and non-overlapping entities, exact-match precision and recall, development/test split and bootstrap significance.
- [seqeval v1.2.2 README](https://github.com/chakki-works/seqeval/blob/v1.2.2/README.md) — default (`conlleval`-compatible) versus strict IOB2 mode, with the `I-NP I-NP O` example.
- [Python 3.11.0 release](https://www.python.org/downloads/release/python-3110/) — released 24 October 2022.
