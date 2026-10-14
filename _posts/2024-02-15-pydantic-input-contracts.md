---
layout: post
lang: en
title: "Pydantic input contracts: valid JSON is not a valid decision"
date: 2024-02-15 09:00:00 +0100
permalink: /blog/pydantic-input-contracts/
author: Rafał Prońko
categories: [python]
description: "A strict Pydantic model for a refund request, and why a request that passes validation still needs a separate policy check on trusted state."
image: /assets/images/pydantic-input-contracts.svg
---

# Pydantic input contracts: valid JSON is not a valid decision

![Diagram: a refund request passes four checks in order. Pydantic covers JSON syntax, strict types and field plus cross-field rules and produces a RefundRequest. A separate policy gate, fed by the caller from the authenticated session and the order from our own store, then returns allowed or denied. Invalid input and a denied decision leave by different exits.]({{ '/assets/images/pydantic-input-contracts.svg' | relative_url }})

<time datetime="2024-02-15">15 February 2024</time>

An endpoint receives a JSON body, Pydantic validates it, and the handler gets a typed object. It is tempting to read that typed object as “the request is fine”.

It only means the request has the shape I declared. Whether the caller may do this, whether the order exists, whether the amount still makes sense against current state — the model has not looked at any of that, and it cannot.

I want to separate these questions explicitly, because each one fails in a different way and should be answered by a different part of the code. I'll use a small refund request as the example: synthetic data, no payment provider, no I/O.

## Five questions, not one

When a refund request arrives, I am really asking five things:

1. **Is it JSON at all?** A truncated body is a syntax problem.
2. **Are the values of the right type?** Is `amount_minor` an integer, or a string that merely looks like one?
3. **Are the fields individually and jointly coherent?** Is the amount positive, the currency one I support, and does the amount agree with the kind of refund requested?
4. **Is this caller allowed to do this to this order right now?** That depends on who is authenticated and on the stored state of the order.
5. **Is it a good decision?** Maybe policy allows it, but the refund is still wrong for business reasons nobody encoded.

Pydantic answers the first three. The fourth needs data the request does not carry and must not be trusted to carry. The fifth is not fully answerable by code; the best I can do is keep the decision explicit and auditable.

## Demo assumptions

So that the example is checkable, I fix the rules:

- Amounts are integers in **minor units** (cents). `1500` with `EUR` means 15.00 EUR. No floats for money.
- Supported currencies: `EUR` and `PLN`. Supported actions: `full_refund` and `partial_refund`.
- A `partial_refund` must carry `amount_minor`; a `full_refund` must leave it out or send `null` — the amount is derived from stored state.
- The caller identity comes from the server (an authenticated session), modelled as a `Caller` value. Roles `refund_agent` and `refund_lead` may refund; anything else may not.
- Authorization is **role-only**. Both refund roles may act on every order in the demo store; there is no check of ownership, assignment, customer, tenant or merchant scope, and `operator_id` is not used in the decision.
- An agent can refund at most 50.00 per request; larger amounts need a lead. This limit is invented for the demo.
- Orders live in a dictionary that stands in for our own database. Only orders in status `captured` are refundable, and never above `captured_minor - refunded_minor`. The store is expected to keep `0 <= refunded_minor <= captured_minor`; the demo store does not enforce that, so the gate checks it and refuses if it is broken.

These are demo policy assumptions, not recommendations for a real refund system.

## The contract and the gate

I use Pydantic 2.6.1 on Python 3.11. The whole example is one file, `contracts.py`, with no dependency other than Pydantic. To run it, create a directory with two files. First `pyproject.toml`:

```toml
[project]
name = "refund-contract"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = ["pydantic==2.6.1"]
```

Then `contracts.py`:

```python
"""Refund request: input contract (Pydantic) and a separate policy gate.

Synthetic data only. Amounts are integer minor units (cents). Nothing is paid.
"""
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)


class RefundRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    request_id: str = Field(pattern=r"^req-[0-9]{4}$")
    order_id: str = Field(min_length=1, max_length=32)
    action: Literal["full_refund", "partial_refund"]
    amount_minor: Optional[int] = Field(default=None, gt=0)
    currency: Literal["EUR", "PLN"]

    @model_validator(mode="after")
    def amount_matches_action(self) -> "RefundRequest":
        if self.action == "partial_refund" and self.amount_minor is None:
            raise ValueError("partial_refund requires amount_minor")
        if self.action == "full_refund" and self.amount_minor is not None:
            raise ValueError("full_refund must not set amount_minor")
        return self


class InvalidInput(Exception):
    def __init__(self, errors: list[dict]):
        super().__init__(f"invalid request: {len(errors)} error(s)")
        self.errors = errors


def parse_request(raw: str | bytes) -> RefundRequest:
    try:
        return RefundRequest.model_validate_json(raw)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        safe = [{"type": e["type"], "loc": e["loc"]} for e in errors]
        raise InvalidInput(safe) from None


@dataclass(frozen=True)
class Caller:  # built by the server from an authenticated session
    operator_id: str
    role: str


@dataclass(frozen=True)
class Order:  # read from our own store, never from the request
    # store invariant (not enforced by the store in this demo):
    # 0 <= refunded_minor <= captured_minor
    currency: str
    status: str
    captured_minor: int
    refunded_minor: int


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    amount_minor: int = 0


REFUND_ROLES = {"refund_agent", "refund_lead"}
AGENT_LIMIT_MINOR = 5_000  # demo policy: agents refund at most 50.00
ORDERS = {
    "ord-1": Order("EUR", "captured", 10_000, refunded_minor=3_000),
    "ord-2": Order("EUR", "disputed", 4_000, refunded_minor=0),
}


def deny(reason: str) -> Decision:
    return Decision(allowed=False, reason=reason)


def decide(
    request: RefundRequest, caller: Caller, orders: dict[str, Order]
) -> Decision:
    if caller.role not in REFUND_ROLES:
        return deny("role_not_permitted")
    order = orders.get(request.order_id)
    if order is None:
        return deny("unknown_order")
    if order.currency != request.currency:
        return deny("currency_mismatch")
    if order.status != "captured":
        return deny("order_not_refundable")
    if not 0 <= order.refunded_minor <= order.captured_minor:
        return deny("inconsistent_order")  # fail closed on bad state
    remaining = order.captured_minor - order.refunded_minor
    if remaining == 0:
        return deny("nothing_to_refund")
    if request.action == "full_refund":
        amount = remaining
    else:
        amount = request.amount_minor
    if amount is None or amount <= 0:
        return deny("invalid_amount")  # model bypassed, e.g. construct()
    if amount > remaining:
        return deny("exceeds_refundable")
    if caller.role == "refund_agent" and amount > AGENT_LIMIT_MINOR:
        return deny("needs_lead")
    return Decision(allowed=True, reason="ok", amount_minor=amount)


if __name__ == "__main__":
    agent = Caller("op-7", "refund_agent")
    samples = [
        '{"request_id": "req-1001", "order_id": "ord-1",',
        '{"request_id": "req-1002", "order_id": "ord-1",'
        ' "action": "partial_refund", "amount_minor": true,'
        ' "currency": "EUR"}',
        '{"request_id": "req-1003", "order_id": "ord-1",'
        ' "action": "partial_refund", "amount_minor": 1500,'
        ' "currency": "EUR", "role": "refund_lead"}',
        '{"request_id": "req-1004", "order_id": "ord-1",'
        ' "action": "full_refund", "amount_minor": 1500,'
        ' "currency": "EUR"}',
        '{"request_id": "req-1005", "order_id": "ord-1",'
        ' "action": "partial_refund", "amount_minor": 6000,'
        ' "currency": "EUR"}',
        '{"request_id": "req-1006", "order_id": "ord-1",'
        ' "action": "partial_refund", "amount_minor": 1500,'
        ' "currency": "EUR"}',
    ]
    for raw in samples:
        try:
            request = parse_request(raw)
        except InvalidInput as exc:
            print("invalid input:", exc.errors)
            continue
        print(f"{request.request_id}:", decide(request, agent, ORDERS))
```

With [uv](https://docs.astral.sh/uv/) installed, `uv run` creates an isolated environment with Python 3.11 and `pydantic==2.6.1` on the first call:

```text
$ mkdir refund-contract && cd refund-contract
$ # save pyproject.toml and contracts.py here
$ uv run --python 3.11 python contracts.py
invalid input: [{'type': 'json_invalid', 'loc': ()}]
invalid input: [{'type': 'int_type', 'loc': ('amount_minor',)}]
invalid input: [{'type': 'extra_forbidden', 'loc': ('role',)}]
invalid input: [{'type': 'value_error', 'loc': ()}]
req-1005: Decision(allowed=False, reason='needs_lead', amount_minor=0)
req-1006: Decision(allowed=True, reason='ok', amount_minor=1500)
```

uv also prints a few lines about creating the environment on the first run; I left them out above. Without uv, any virtual environment with Python 3.11 and `pip install pydantic==2.6.1` runs the same file with `python contracts.py`.

The output shows both kinds of rejection. The first four samples never become a `RefundRequest`: broken JSON, a boolean where an integer belongs, an extra field, and a full refund that also names an amount. The fifth sample is a perfectly valid request that the policy gate denies. Only the last one is allowed.

## What the model configuration buys

**`strict=True`** turns off coercion. In the default lax mode Pydantic converts compatible input: the string `"1500"` becomes `1500`, and so does the float `1500.0`. The least obvious case is a boolean. I checked it with a model that only declares `amount_minor: int`:

```text
lax    {"a": true}   -> a=1
strict {"a": true}   -> int_type
strict {"a": "1500"} -> int_type
strict {"a": 1500.0} -> int_type
```

A `true` that turns into a one-cent refund will not crash anything. It will just be wrong. For money and identifiers I would rather reject the request and let the caller fix the client.

Strict mode has a caveat worth knowing before relying on it. The Pydantic 2.6 strict-mode documentation says validation from JSON is looser than from Python for some types: in strict mode a `UUID` field accepts a string from JSON, but not from a Python dict. That is sensible — JSON has no native UUID type, so a string is the only way to send one — but it means “strict” does not mean the same thing for `model_validate` and `model_validate_json`. I validate the raw body with `model_validate_json`, which is the path the tests exercise.

A second caveat: per the same page, strict mode set in a model's config is not applied recursively to nested model fields. A nested model needs its own `strict=True` (and its own `extra="forbid"`, since `extra` is also per-model configuration). My model is flat, so this does not bite here, but it does in real request bodies with nested objects.

**`extra="forbid"`** rejects unknown keys. Here it does more than tidy up typos. Sample `req-1003` includes `"role": "refund_lead"`. With the default `extra="ignore"` that key would be silently dropped — harmless only as long as no one later reads the raw dict. With `forbid`, the request fails with `extra_forbidden`. Either way, the role in the body is **never** used as authority; the gate takes the role from `Caller`, which the server builds.

**`Literal`** narrows `action` and `currency` to known values. `"USD"` or `"chargeback"` fails with `literal_error` instead of reaching business code as an arbitrary string.

**`Field(gt=0)`, `pattern`, `min_length`/`max_length`** are field-level invariants. They are checked on values that were actually provided.

That last point matters for defaults. `amount_minor` defaults to `None`, and Pydantic does not validate defaults unless `validate_default` is enabled (it is `False` by default, per the configuration reference). Here that is fine, because `None` is a deliberate “not given” and the model validator decides whether it is allowed. If a default could break an invariant, I would either validate it or not have a default.

**`frozen=True`** prevents assignment to fields after validation. It does not make the object tamper-proof: `model_construct()` builds an instance without running validation at all, and `model_copy(update=...)` does not re-validate the updated values. With 2.6.1, copying a valid request with `update={"amount_minor": -5, "currency": "USD"}` returned an object holding exactly those values, while plain assignment `request.amount_minor = 1` failed with `frozen_instance`. Neither bypass appears in the request path here, and I keep it that way — the only way to get a `RefundRequest` in the handler is `parse_request`.

## The cross-field rule belongs in the model

`amount_matches_action` is a `model_validator(mode="after")`. After validators run once the fields have been parsed and validated, so inside it `self.action` is already one of the two literals and `self.amount_minor` is either a positive integer or `None`. I don't have to defend against raw input there.

The rule is about the request's internal coherence: a partial refund without an amount, or a full refund that also states an amount, is a malformed request no matter who sends it or which order it targets. That is why it lives in the contract and fails as invalid input (`value_error`), rather than as a policy denial.

The test for that boundary is simple: can I evaluate the rule using only the request? If yes, it can be part of the input contract. If I need to know who is calling, what is stored, or what time it is, it belongs in the gate.

## The gate works on trusted context

`decide` receives three things: the validated request, the `Caller` and the order store. Only the first came from the client.

The order's currency, status, captured and already-refunded amounts all come from our own records. The request's `currency` is checked against the stored one, not trusted. For a full refund the amount is computed from stored state; the client never gets to name it.

Trusted does not mean correct, though. The `gt=0` constraint on `amount_minor` protects only amounts that came through the model; a full-refund amount comes from the store and never passes through it. If an order has already been refunded in full, `remaining` is zero, and without an explicit check the gate would approve a refund of 0. If the counters are corrupted so that `refunded_minor` exceeds `captured_minor`, it would approve a negative one. So the gate first checks the store invariant `0 <= refunded_minor <= captured_minor` and refuses with `inconsistent_order` when it is broken, then refuses with `nothing_to_refund` when nothing remains. Both are denials, not invalid input: the request was fine, the state was not.

The gate also does not assume that every `RefundRequest` went through validation. `model_construct()` can build one with `action="partial_refund"` and no amount. Instead of an `assert`, which disappears under `python -O` and is no security boundary, `decide` refuses any amount that is `None` or not positive with `invalid_amount`. In normal use this branch never runs; one test reaches it through `model_construct()` to show it fails closed.

Authorization in the demo is deliberately thin: it checks only the caller's role. It does not ask whether this operator may act on this particular order — ownership, assignment, customer, tenant or merchant scope are not modelled, and `operator_id` is carried but unused. The `Caller` type itself authenticates nothing either. That the server builds it from a verified session is an assumption about the surrounding application, which a real system has to enforce on the server side.

Sample `req-1005` shows the point of this article in one line. The request is valid JSON, every type is right, the amount is positive and agrees with the action. The contract accepts it. Then the gate denies it with `needs_lead`, because an agent is trying to refund 60.00 against a 50.00 limit. Nothing about the payload was wrong; the decision was.

The tests add the other denials: a caller with role `viewer`, an unknown order, a currency that does not match the order, an order in status `disputed`, an amount above what remains refundable, a fully refunded order and two orders with broken counters. Each of them first asserts that the request passes `parse_request`, so the denial is genuinely coming from the gate.

## Two different failures, two different responses

I keep `InvalidInput` and a denied `Decision` separate on purpose.

Invalid input is the client's problem: the request is malformed, and the correct answer is something like HTTP 422 with a list of field locations and error types. Retrying the same body will fail the same way.

A denial is a business outcome. The request was understood; the answer is no. It may deserve a 403 or a domain-specific response, an audit entry with the reason, and perhaps a path to escalate (`needs_lead`). Mixing the two makes both harder to monitor: a spike in validation errors usually means a broken client, a spike in denials may mean a policy change or abuse.

## Do not log the payload you just rejected

A rejected request is exactly the kind of input that may contain something it should not: a card number pasted into a free-text field, a token, personal data. By default a Pydantic `ValidationError` includes the offending input in its error details.

`parse_request` calls `exc.errors(include_input=False, include_url=False)` and keeps only `type` and `loc`. The human-readable `msg` is dropped too, so I don't have to review every message template for what it might echo. One test sends an extra field containing a synthetic card number and asserts that it appears neither in the stored errors nor in the exception message. Pydantic also has a `hide_input_in_errors` config option; I chose to whitelist the fields I keep.

`raise ... from None` is narrower than it looks. It suppresses the original `ValidationError` in the standard traceback display, so a plain logged stack trace shows only `InvalidInput`. It does not erase the original exception: it is still reachable as `__context__`, and an error reporter that inspects exception context or captures frame locals (where `raw` sits) can still collect the rejected input. What leaves the process through telemetry is a matter of that tool's configuration, not of this function.

This is a minimum, not a logging policy. Only the normal output — the list of errors and the exception message — is sanitised. Even there, `loc` can contain attacker-controlled text: for an extra field, `loc` is the key name the client chose.

Dropping `msg` has a cost. Both directions of the action/amount rule fail as `value_error` with an empty `loc`, because a model validator reports at the model root. A client can't tell “partial refund without an amount” from “full refund with an amount” from the response alone. For this demo I accept that; a real API might map each rule to its own stable error code instead of passing messages through.

## What the tests cover

Next to `contracts.py` I keep a pytest suite, `tests/test_contracts.py`. It is not reproduced in this post, so here is what it covers and how it ran. It has 24 test cases (counting parametrised ones):

- malformed JSON (`json_invalid`);
- no coercion of `"1500"`, `true` and `1500.0` into an integer, plus a control test showing that a lax model turns `true` into `1`;
- unknown currency and action (`literal_error`);
- a caller-supplied `role` rejected as an extra field;
- a missing `currency` (`missing`) and an explicit `null` (`literal_error`);
- both directions of the action/amount rule (`value_error`), and a zero amount (`greater_than`);
- errors that do not echo raw input;
- an accepted partial refund, a full refund computed from stored state, and six requests that pass validation but are denied by the gate;
- full refunds against a fully refunded order (`nothing_to_refund`) and against two orders with broken counters (`inconsistent_order`);
- a request built with `model_construct()` without an amount, denied with `invalid_amount`.

I wrote the first 20 tests against a stub that only defined the names: 19 failed and 1 passed (the lax control test, which does not touch my code). The implementation made them pass. The last four tests came later, after a review pointed out that a full refund of a fully refunded order was approved with an amount of 0. Added before the fix, they failed (4 failed, 20 passed); with the gate checks above, all 24 pass:

```text
$ uv run --python 3.11 pytest -q
........................                                                 [100%]
24 passed in 4.94s
```

The tests do not cover concurrency (two refunds racing against the same remaining amount), authentication itself, authorization beyond the role check (whether this operator may act on this order), persistence, or anything about a real payment provider. In a real system the check of the remaining amount and the write of the refund would need to happen atomically; a validated request read before that check guarantees nothing about state at write time.

A note on dates and tooling: the label of this article is February 2024, and Pydantic 2.6.1 was published on PyPI on 5 February 2024. The example was run on 12 October 2026 with CPython 3.11.12, pydantic 2.6.1 (pydantic-core 2.16.2) and pytest 8.0.0, in an isolated environment managed by uv 0.7.8. uv is a current tool used for this rerun, not something the 2024 label implies.

## Where the guarantee ends

After `parse_request` succeeds, Pydantic guarantees that the object conforms to the model I declared: the types, literals, constraints and validators I wrote. That is a strong and useful guarantee, and strict mode plus `extra="forbid"` make it narrower in a good way.

It is not a guarantee that the request is true, authorised or wise. The order ID may be valid-looking and nonexistent. The caller may lack permission. The amount may be allowed and still wrong. An input schema cannot express every runtime rule because many of them depend on who is asking and on state that changes after the schema was written.

So I give the model the questions it can answer from the request alone, and I keep authority and current state in a separate gate that reads trusted context. When a request is rejected, I want to know which of the two said no.

### Sources

- [Pydantic 2.6: Strict Mode](https://docs.pydantic.dev/2.6/concepts/strict_mode/) — strict vs lax, looser JSON validation for types such as `UUID`, strict not applied recursively to nested models.
- [Pydantic 2.6: Validators](https://docs.pydantic.dev/2.6/concepts/validators/) — `model_validator` with `mode="after"`.
- [Pydantic 2.6: Models](https://docs.pydantic.dev/2.6/concepts/models/) — what validation guarantees, `model_construct()` creating models without validation.
- [Pydantic 2.6: Configuration API](https://docs.pydantic.dev/2.6/api/config/) — `extra`, `frozen`, `strict`, `validate_default`, `hide_input_in_errors`.
- [PyPI: pydantic 2.6.1](https://pypi.org/project/pydantic/2.6.1/) — release files uploaded 2024-02-05.
