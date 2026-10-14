---
layout: post
lang: en
title: "Ranking is not eligibility: put hard constraints before scores"
date: 2024-02-29 09:00:00 +0100
permalink: /blog/ranking-is-not-eligibility/
author: Rafał Prońko
categories: [python, software-engineering]
description: "Why a weighted score must never decide whether an option is allowed: a hard gate with eligible, pending and rejected, then a deterministic top-k in plain Python 3.11."
image: /assets/images/ranking-is-not-eligibility.svg
---

# Ranking is not eligibility: put hard constraints before scores

![Diagram: normalized facts and a versioned policy pass through a hard gate into eligible, pending and rejected; only eligible options are ranked, and the chosen one is revalidated before any action.]({{ '/assets/images/ranking-is-not-eligibility.svg' | relative_url }})

<time datetime="2024-02-29">29 February 2024</time>

Many systems that recommend something follow the same shape. There is a catalog of options, each option gets a score, and the user sees the top few. Products, suppliers, job offers, candidate documents for a RAG answer — the shape is the same.

The bug I want to discuss is easy to miss in a demo. An option scores very high, lands at the top of the list, and it is something the user had said they cannot accept. Not "would prefer to avoid". Cannot accept.

The usual reaction is to tune the weights. I think that treats the symptom. The real problem is that one number is being asked to answer two different questions:

1. Is this option allowed at all?
2. Among the allowed options, which one is better?

The first question has a yes/no answer (and, as we will see, sometimes "not yet known"). The second one is a matter of degree. Treating a position in a single ranking as permission to select an option is the mistake.

I will use a neutral, synthetic example: a catalog of work opportunities and a policy with two hard constraints — the work must be remote, and the verified minimum hourly rate must be at least 50 EUR. The names, numbers and weights are invented for the example. They are not anyone's real criteria, and the scores are illustrative, not measured.

## A weighted sum compensates

Suppose I score opportunities with a weighted sum that includes pay:

```
score = 3 * skill + 2 * growth + 2 * hourly_eur
```

Two opportunities, both remote:

| id | hourly_eur | skill | growth | score |
|---|---|---|---|---|
| opp-d | 35 | 95 | 100 | 285 + 200 + 70 = **555** |
| opp-b | 55 | 70 | 100 | 210 + 200 + 110 = 520 |

`opp-d` wins, even though it pays 35 EUR per hour and the minimum is 50. The skill term bought back the missing 15 euros.

Raising the pay weight to 10 still allows compensation; it moves the break-even point. With `10 * hourly_eur`, `opp-b` scores 960. An opportunity paying 49 EUR with skill 100 and growth 100 scores 300 + 200 + 490 = 990 and still wins. The skill term has a range of 300 points, and at 10 points per euro those 300 points buy back up to 30 euros of shortfall.

A weighted sum normally expresses tradeoffs: a deficit in one dimension can be offset by a surplus in another. But it would be too strong to say that every finite choice of weights must violate a hard constraint. On bounded domains, a sufficiently large weight can enforce priority. Here `q = 3 * skill + 2 * growth` lies in `[0, 500]` and pay uses whole euros. With `501 * hourly_eur + q`, the worst pay-eligible option scores `501 * 50 = 25050`, above the best below-threshold option's `501 * 49 + 500 = 25049`.

That pay weight also changes the order among acceptable options: one extra euro outweighs any difference in skill and growth. An encoding such as `501 * eligible + q`, where `eligible` is 1 only when the whole policy passes and 0 otherwise, can put eligible options first while preserving their preference order. It already needs an eligibility decision to compute that indicator. And ordering is still not exclusion: if fewer than `k` options are eligible, taking the top `k` can include ineligible options. The selection must explicitly exclude them. Keeping that gate separate makes the responsibility visible.

## First the feasible set, then the ranking

I keep the two responsibilities explicit:

1. Apply hard constraints. The result is the *feasible set* — the options that are allowed.
2. Rank only the feasible set by score.

The score never sees an option that failed a constraint, so it cannot rescue it. The constraint never sees the score, so it cannot be bargained with. Pay leaves the score entirely: in the example it is only a constraint. If I also wanted "more pay is better" above the minimum, I would add it as a separate preference term, knowing it can no longer push anything across the line.

## Three answers, not two

Real data has holes. An offer might not say whether it is remote. A rate might not be stated, or only stated in a way that can't be verified.

If a missing fact is treated as `False`, the option is silently dropped and nobody checks whether it actually fits. If it is treated as `True`, the system recommends something that may break a hard constraint. Both are guesses dressed up as decisions.

So each constraint has three outcomes — pass, fail, unknown — and the option ends up in one of three groups:

- **eligible**: every constraint is known and satisfied,
- **pending**: no constraint is known to fail, but at least one fact is missing,
- **rejected**: at least one constraint is known to fail.

The order of these rules matters. **A known failure dominates an unknown.** An on-site opportunity with no stated rate is rejected, not pending: no answer about the rate can make it remote. Sending it to manual verification would waste someone's time.

Pending options are not ranked. They are a work queue: fetch the missing fact, or ask a person.

## The example

I use Python 3.11 and only the standard library. The code below can be saved as `ranking.py` and run with `python3 ranking.py`.

```python
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum


def is_int(value: object, low: int, high: int | None = None) -> bool:
    # Reject bool, float, str, NaN and inf.
    return (
        type(value) is int
        and value >= low
        and (high is None or value <= high)
    )


@dataclass(frozen=True)
class Policy:
    """Owners must keep rules immutable per label and version any change."""

    name: str
    version: int
    remote_required: bool
    min_hourly_eur: int  # verified minimum, whole EUR per hour

    def __post_init__(self) -> None:
        if not (
            type(self.name) is str
            and self.name
            and is_int(self.version, 1)
            and type(self.remote_required) is bool
            and is_int(self.min_hourly_eur, 1)
        ):
            raise ValueError(f"invalid policy: {self!r}")

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class Weights:
    skill: int  # 1..10
    growth: int  # 1..10

    def __post_init__(self) -> None:
        if not (is_int(self.skill, 1, 10) and is_int(self.growth, 1, 10)):
            raise ValueError(f"invalid weights: {self!r}")


@dataclass(frozen=True)
class Opportunity:
    oid: str
    remote: bool | None  # None: not known yet
    hourly_eur: int | None  # verified minimum; None: not known yet
    skill: int  # 0..100, illustrative
    growth: int  # 0..100, illustrative

    def __post_init__(self) -> None:
        if not (
            type(self.oid) is str
            and self.oid
            and (self.remote is None or type(self.remote) is bool)
            and (self.hourly_eur is None or is_int(self.hourly_eur, 1))
            and is_int(self.skill, 0, 100)
            and is_int(self.growth, 0, 100)
        ):
            raise ValueError(f"invalid opportunity: {self!r}")


class Gate(Enum):
    ELIGIBLE = "eligible"
    PENDING = "pending"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Decision:
    policy: str
    ranked: tuple[tuple[str, int], ...]  # ((oid, score), ...), best first
    pending: tuple[tuple[str, tuple[str, ...]], ...]  # unknown fields
    rejected: tuple[tuple[str, tuple[str, ...]], ...]  # failed fields


def gate(o: Opportunity, policy: Policy) -> tuple[Gate, tuple[str, ...]]:
    pay = o.hourly_eur
    checks: dict[str, bool | None] = {
        "hourly_eur": None if pay is None else pay >= policy.min_hourly_eur
    }
    if policy.remote_required:
        checks["remote"] = o.remote
    failed = tuple(sorted(f for f, ok in checks.items() if ok is False))
    unknown = tuple(sorted(f for f, ok in checks.items() if ok is None))
    if failed:
        return Gate.REJECTED, failed  # a known failure beats any unknown
    if unknown:
        return Gate.PENDING, unknown
    return Gate.ELIGIBLE, ()


def score(o: Opportunity, weights: Weights) -> int:
    return weights.skill * o.skill + weights.growth * o.growth


def by_score(
    catalog: Iterable[Opportunity], weights: Weights
) -> list[Opportunity]:
    return sorted(catalog, key=lambda o: (-score(o, weights), o.oid))


def rank(
    catalog: Sequence[Opportunity], policy: Policy, weights: Weights, k: int
) -> Decision:
    """Use a sized, repeatable sequence with unique ids, not a generator."""
    if not is_int(k, 1):
        raise ValueError(f"k must be a positive int, got {k!r}")
    if len({o.oid for o in catalog}) != len(catalog):
        raise ValueError("duplicate oid in catalog")
    eligible: list[Opportunity] = []
    groups: dict[Gate, list[tuple[str, tuple[str, ...]]]] = {
        Gate.PENDING: [], Gate.REJECTED: []
    }
    for o in sorted(catalog, key=lambda o: o.oid):
        status, fields = gate(o, policy)
        if status is Gate.ELIGIBLE:
            eligible.append(o)
        else:
            groups[status].append((o.oid, fields))
    best = by_score(eligible, weights)[:k]  # filter first, then cut
    return Decision(
        policy.label,
        tuple((o.oid, score(o, weights)) for o in best),
        tuple(groups[Gate.PENDING]),
        tuple(groups[Gate.REJECTED]),
    )


def top_k_then_filter(
    catalog: Iterable[Opportunity], policy: Policy, weights: Weights, k: int
) -> list[str]:
    """The trap, for comparison: truncate before applying the gate."""
    top = by_score(catalog, weights)[:k]
    return [o.oid for o in top if gate(o, policy)[0] is Gate.ELIGIBLE]


def still_eligible(
    decision: Decision, o: Opportunity, policy: Policy
) -> bool:
    """Caller supplies fresh facts and authoritative, immutable policy rules.

    Labels alone detect neither rule reuse nor stale facts; this is no lock.
    """
    chosen = {oid for oid, _ in decision.ranked}
    return (
        policy.label == decision.policy
        and o.oid in chosen
        and gate(o, policy)[0] is Gate.ELIGIBLE
    )


POLICY = Policy("synthetic-remote", 3, remote_required=True, min_hourly_eur=50)
WEIGHTS = Weights(skill=3, growth=2)
CATALOG = [
    Opportunity("opp-a", remote=False, hourly_eur=90, skill=100, growth=100),
    Opportunity("opp-b", remote=True, hourly_eur=55, skill=70, growth=100),
    Opportunity("opp-c", remote=True, hourly_eur=65, skill=90, growth=90),
    Opportunity("opp-d", remote=True, hourly_eur=35, skill=95, growth=100),
    Opportunity("opp-e", remote=True, hourly_eur=None, skill=95, growth=95),
    Opportunity("opp-f", remote=True, hourly_eur=70, skill=40, growth=50),
]

if __name__ == "__main__":
    decision = rank(CATALOG, POLICY, WEIGHTS, k=2)
    print("policy  ", decision.policy)
    print("ranked  ", decision.ranked)
    print("pending ", decision.pending)
    print("rejected", decision.rejected)
    print("trap    ", top_k_then_filter(CATALOG, POLICY, WEIGHTS, k=2))
```

Output:

```
policy   synthetic-remote@3
ranked   (('opp-c', 450), ('opp-b', 410))
pending  (('opp-e', ('hourly_eur',)),)
rejected (('opp-a', ('remote',)), ('opp-d', ('hourly_eur',)))
trap     []
```

Six opportunities go in. Two are rejected: `opp-a` because it is not remote, `opp-d` because 35 EUR is below the minimum — even though their scores (500 and 485) are the highest in the catalog. `opp-e` scores 475, above every eligible option, but its rate is unknown, so it waits in `pending`. Of the three eligible options, `rank(..., k=2)` returns the best two.

## What the code is doing, piece by piece

### Validation, because dataclasses don't validate

`@dataclass` generates `__init__`, but it does not check types. The documentation says that, with two exceptions, nothing in `@dataclass` examines the type specified in the variable annotation. Without `__post_init__`, `Opportunity("x", "yes", 60.5, 50, 50)` would be accepted without complaint.

The three input classes — `Policy`, `Weights` and `Opportunity` — validate their fields in `__post_init__`. `Decision` is an output container with no validator. The helper `is_int` uses `type(value) is int` rather than `isinstance`, because `bool` is a subclass of `int` and `True` would otherwise pass as `1`. Floats are rejected too, which takes `NaN` and infinity out of the picture: ordered comparisons (`<`, `<=`, `>` and `>=`) involving `NaN` are false, making a sort key containing it unreliable. This does not apply to every comparison: `NaN != NaN` is true.

A missing field is a different error. `Opportunity("x", True, 60)` fails with `TypeError` from the generated `__init__`. That is fine: a field that is *absent* from the call is a programming error, while a field explicitly set to `None` is a known unknown that the gate handles.

Weights are integers from 1 to 10 and scores are integers. I prefer small, finite, validated values to "any number". They are easier to test and they can't produce a `NaN` score.

### The gate

`gate` builds a dictionary of checks. Each value is `True`, `False` or `None`. Constraints that the policy does not require are not checked at all: with `remote_required=False`, an unknown `remote` does not make anything pending.

If any check failed, the result is `REJECTED` with the list of failed fields. Only when nothing failed do unknowns lead to `PENDING`. The returned field names explain the decision; the rejected and pending groups keep them.

### Ranking: filter first, then cut

`rank` puts every option through the gate first. Only after that does it sort the eligible group and take the first `k`. Its catalog contract is a sized, repeatable `Sequence[Opportunity]`, such as a list or tuple, with unique identifiers. It uses `len` and more than one pass, so callers must materialize a one-shot iterator or generator before calling it. The annotations document this contract; they do not add runtime type checks.

The opposite order is a common trap. If the code first asks a search index or a model for the top `k` by score, and only then filters, it can return nothing even though eligible options exist. `top_k_then_filter` reproduces this: the two best scores in the catalog belong to rejected options, so the "trap" line in the output is an empty list. The same thing happens with a retrieval step that returns ten documents and a permission filter applied afterwards.

If the catalog is too large to gate in full, the hard constraints have to be pushed into the query itself — as filters in SQL or in the search index — not applied to its truncated result.

### A deterministic tie-break

Python's sort is stable: equal **keys** retain their input order. If the key were only `-score`, equal scores would have equal keys. Their order would then depend on the input, which might come from an API page, a dictionary built from a network response, or a database without `ORDER BY`. Two runs over the same data could show different winners.

The actual key is `(-score, oid)`. Equal scores with different identifiers have different keys, so they are ordered by identifier, not by their input positions. Unique identifiers make the order a function of the data only. The rejected and pending groups are also sorted by identifier, so the whole `Decision` is reproducible.

Duplicate identifiers are rejected. With duplicates, a tie-break by identifier would no longer be a total order, and "which `opp-b` did we pick?" has no answer.

### The policy is an input with a version

The constraints come from an explicit `Policy` object with a name and a version, and the decision records `synthetic-remote@3`. The code doesn't infer what the user "probably wants" from click history or from a free-text summary. The policy owner must guarantee that each name/version label identifies one immutable set of rules. Any rule change requires a new version; reusing a label for changed rules breaks this contract. `frozen=True` prevents ordinary mutation of this instance, but it does not stop someone constructing another `Policy` with the same label and different contents.

## An empty list is an answer

When nothing is eligible, `rank` returns an empty `ranked` tuple, and the rejected and pending groups explain why. The policy in the result stays the same.

The tempting fix is to relax the requirements automatically: "no remote options, so let's show the best on-site ones". That turns a hard constraint into a soft one exactly when it matters, and it does so without asking.

The honest alternatives are:

- show the empty result together with the reasons,
- send the pending options to a manual verification queue,
- propose a constraint change and apply it only after the user explicitly approves it — as a new policy version, so the next decision says which rules it used.

## Revalidate before acting

A ranking is computed at some moment. Before anything is done with the result — sending an application, placing an order, calling a tool — the facts may have changed. The offer was updated, the rate was corrected, the policy got a new version.

`still_eligible` repeats the check right before the action. It requires three things: the option was in the ranked result, the supplied policy has the same label as the one used for the ranking, and the gate on the supplied facts still says `ELIGIBLE`. The caller must obtain fresh facts and the authoritative current policy from trusted sources, under the immutable-label contract above. This function neither fetches facts nor discovers the current policy version. Rechecking an old object checks old facts again. If the authoritative policy has a new version, the old ranking is not reused — it is computed again.

The label comparison does not compare policy contents or detect label reuse. A stored policy snapshot or digest could additionally detect changed contents under the same label, but would still not establish freshness.

With fresh caller inputs, this catches changes since ranking. It is not an atomic permission to act: facts or policy can change after the check. Where that race matters, the action endpoint must enforce the relevant fact and policy versions as atomic preconditions — for example, in the same transaction or conditional write that performs the action — and reject or recompute when they no longer match.

## A score is not a probability

`opp-c` scored 450. That does not mean a 45% chance of anything, and it is not a measure of fit validated against outcomes. It is a weighted sum of two illustrative numbers.

A score orders options under one set of weights. Comparing scores across different weights, or reading them as confidence, adds meaning that the code doesn't contain. If I wanted calibrated probabilities, I would need outcome data and a separate evaluation, and the probability would still not replace a hard constraint.

## What the tests check

Alongside the module I keep a small pytest suite. It checks that:

1. A high-scoring ineligible option is never returned, and low pay is not compensated by score.
2. A missing `remote` or rate gives `pending`, not `eligible`.
3. A known failure dominates an unknown.
4. An empty feasible set returns an empty ranking and keeps the policy.
5. Top-k before filtering loses eligible options, while `rank` finds them.
6. The ranking with ties is the same for twenty random permutations of the input.
7. Invalid weights, policies and opportunity fields raise `ValueError`: wrong types, `bool`, floats, `NaN`, infinity, strings, out-of-range values; missing fields raise `TypeError`; duplicate identifiers and a bad `k` are rejected.
8. Revalidation fails after a rate change, after the fact becomes unknown, for an option outside the ranking, and for a new policy version.

These tests check the behaviour of this code on the facts it is given. They don't check where the facts came from.

**Verification: `EXECUTED_PASS`.** The 38 tests and the synthetic standard-library demo were executed with CPython 3.11.12 on 2026-10-12. The 2024-02-29 date is an editorial label, not a claim of historical publication or an experiment performed then.

## Where the guarantee ends

The guarantee is narrow and I'd rather state it precisely: **given normalized, validated facts and a given policy version, no option that fails a hard constraint, or has an unknown required fact, is returned in the ranking.** Everything outside that sentence is a separate problem.

**Facts come from extractors.** In practice `remote` and `hourly_eur` are produced by a parser, a scraper, a form, or a language model reading a description. If the extractor says `remote=True` for an on-site role, the gate faithfully passes it. If the data is a month old, the gate checks a month-old world. The gate doesn't make an LLM extraction correct and gives no safety guarantee about one. What it does is put the responsibility in one visible place: the extraction must output `None` when it isn't sure, and every fact should carry its source and the time it was checked.

**Currency and units.** `hourly_eur` is deliberately named with its unit. A rate in another currency needs a conversion with a stated rate and date. That is a policy decision, not something a parser should improvise. Until the conversion is done under explicit rules, the honest value is `None`.

**Monthly and hourly pay.** A monthly salary cannot be converted to an hourly rate without an assumption about hours. If that assumption isn't part of the policy, the converted number isn't a verified minimum. The same goes for ranges: for "45–70 EUR per hour" the verified minimum is 45, not the midpoint.

**Permission boundaries.** The code that ranks should be able to read the policy, not change it. If an agent or an automated job can both rank and rewrite the constraints, then "relax the requirements when nothing is found" is one bug away. Changing the policy and acting on a decision — applying, buying, sending — should require separate permissions, and the action endpoint should enforce revalidation and any required atomic preconditions using authoritative inputs.

## Where I start

Not with the weights.

First I write down which conditions are hard — the ones where no amount of score should change the answer. I give each of them three outcomes, decide that a known failure beats an unknown, and keep those conditions out of the score. Then I rank only what passed, with a tie-break that doesn't depend on input order, and check again before acting.

The score then does one job: ordering acceptable options. That's a job it can do well.

### Sources

- [Python 3.11: Sorting HOW TO](https://docs.python.org/3.11/howto/sorting.html) — key functions, called exactly once per element; sorts are guaranteed to be stable.
- [Python 3.11: Built-in functions, `sorted()`](https://docs.python.org/3.11/library/functions.html#sorted) — `sorted()` is guaranteed to be stable.
- [Python 3.11: dataclasses](https://docs.python.org/3.11/library/dataclasses.html) — generated `__init__`, `__post_init__`, and the note that `@dataclass` does not examine annotation types apart from two exceptions.
- [Python 3.11: enum](https://docs.python.org/3.11/library/enum.html) — `Enum` used for the three gate outcomes.
- [Python 3.11.0 release](https://www.python.org/downloads/release/python-3110/) — release date 24 October 2022.
