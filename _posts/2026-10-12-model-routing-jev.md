---
layout: post
lang: en
title: "Model routing with Jev: allowlists, capability guards, and rollback"
date: 2026-10-12 09:00:00 +0200
permalink: /blog/model-routing-jev/
author: Rafał Prońko
categories: [agents]
description: "A classifier such as Jev can suggest which model should handle a turn. The host still owns the allowlist, capability checks, full-request size check, commit point and rollback. Synthetic Python example with tests."
image: /assets/images/model-routing-jev.svg
---

# Model routing with Jev: allowlists, capability guards, and rollback

![Diagram: a classifier proposal with only a route alias and a confidence passes host guards, is staged as a copy, checked as a full destination request, then committed in memory or discarded; persistence is a separate best-effort step.]({{ '/assets/images/model-routing-jev.svg' | relative_url }})

<time datetime="2026-10-12">12 October 2026</time>

Routing every turn of an agent conversation to the same large model is simple and expensive. Many turns are short questions that a smaller model could answer. So the idea is obvious: ask a fast classifier which model should take the next turn, then switch.

The classifier is the easy part. The hard part is everything that happens after it answers. A switch changes which adapter is called, which system prompt is sent, which tools exist, how history is replayed and how large the request is. If any of that goes wrong halfway, the session can end up in a state that matches neither the old model nor the new one.

In this article I treat the classifier's answer as a **proposal** and build the host side around it: an allowlist, capability guards, a check of the full destination request, a single commit point, and rollback. The example is a small, synthetic Python simulation with tests. It does not call Jev or any other model API.

## What Jev gives you, and what it does not

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is a model from TypeSafe AI that answers predefined questions with typed values instead of free-form text. According to the [TypeSafe quickstart](https://docs.typesafe.ai/introduction/quickstart), a *choice* question returns the selected option, a probability for each option and a confidence value. The [LangChain guide to building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev) (17 September 2026) uses that kind of choice question in a model-routing middleware: the router picks a model from the latest user message and uses it for the rest of the run.

TypeSafe also makes claims about speed, cost and type errors. I haven't measured any of them, and they don't matter for this article. Even if the output always matches the schema, a well-typed answer is not necessarily a *correct* decision. "The value is one of my three labels" says nothing about whether the turn needs tools, whether the destination can read the history, or whether the request fits.

So the boundary I want is:

- the classifier suggests an **alias** and a **confidence**;
- the host decides whether that alias is acceptable for this turn, using facts the classifier never sees.

## The example: three fictional routes

The example uses three host-local aliases. They are fictional names in the host's own catalog, not real model identifiers:

```python
CATALOG = {
    "fast-text": Route("fast-text", adapter="adapter-a", model="example-small",
                       context_tokens=8_000, output_budget=1_000, system_prompt_tokens=300,
                       tool_schema_tokens=0, tools=False, attachments=False, replays=PLAIN),
    "mid-tools": Route("mid-tools", adapter="adapter-b", model="example-medium",
                       context_tokens=12_000, output_budget=2_000, system_prompt_tokens=900,
                       tool_schema_tokens=3_000, tools=True, attachments=False, replays=PLAIN),
    "full-tools": Route("full-tools", adapter="adapter-b", model="example-large",
                        context_tokens=64_000, output_budget=4_000, system_prompt_tokens=1_200,
                        tool_schema_tokens=2_500, tools=True, attachments=True, replays=PLAIN),
}
```

A few choices here matter more than they look.

**The proposal only names an alias.** The provider model id and the adapter come from the catalog. If the classifier output contains a raw model id or a URL, it is simply not a key in the dictionary. There is no code path that turns proposal text into an endpoint.

**The catalog has no credentials.** Authentication belongs to the adapter, which looks up its own credentials when it is called. Adding a route to the allowlist should not mean copying a secret into routing configuration, and a routing decision should never be able to choose *whose* credentials are used.

**Capabilities are host facts.** Whether a route supports tools or attachments, and which history formats its adapter can replay, is written down by whoever operates the host. The classifier doesn't know these things and isn't asked.

The token numbers are configured values for the simulation. They are not tokenizer output and not a statement about any real model's window.

## Order of checks

The router's main method runs cheap, policy-level checks first, then walks through the proposed alias and its alternatives:

```python
def route_turn(self, turn: Turn, proposal: Proposal) -> Decision:
    policy, current = self.policy, self.runtime
    if not policy.enabled:
        return self.keep("disabled")
    if policy.manual_pin is not None:
        return self.keep("manual_pin")
    if current.routed_turn == turn.turn_id:
        return self.keep("already_routed_this_turn")
    if not valid_confidence(proposal.confidence):
        return self.keep("malformed_confidence")
    if proposal.confidence < policy.min_confidence:
        return self.keep("low_confidence")
    if not isinstance(proposal.alternatives, tuple):
        return self.keep("malformed_alternatives")  # the container; elements stay untrusted

    rejected: list[tuple[str, str]] = []
    for choice in (proposal.choice, *proposal.alternatives):
        name = choice if isinstance(choice, str) else repr(choice)
        route = self.catalog.get(choice) if isinstance(choice, str) else None
        if route is None:
            rejected.append((name, "not_in_allowlist"))
            continue
        reason = capability_rejection(route, turn)
        if reason is not None:
            rejected.append((name, reason))
            continue
        if route == current.route:
            return self.keep("already_on_route", tuple(rejected))
        decision = self._stage_and_commit(route, turn, rejected)
        if decision is not None:
            return decision
    return self.keep("no_acceptable_route", tuple(rejected))
```

**Routing is off by default.** `Policy(enabled=False)` is the default. Automatic switching is something you opt into after you've checked it on your own traffic.

**A manual pin wins.** If the user explicitly chose a model, a confident classifier does not get to override that.

**At most one committed switch per user turn, if the caller cooperates.** The router does not enforce "one decision per turn" by itself. The caller has to invoke `route_turn` once, when the user's message arrives and before the agent loop starts, not before every internal API call; an agent loop that makes five tool calls inside one turn should not hop between models in the middle. What the router adds is narrower: after a *committed* switch, `Runtime.routed_turn` holds that turn's id, and further proposals with the same id return `already_routed_this_turn`. Not every decision sets that marker. A rejected proposal (for example `low_confidence`) or an `already_on_route` keep leaves it unchanged, so a second call for the same turn can still switch. Only the last committed turn id is remembered, so the guarantee also assumes turn ids that increase monotonically, are unique and are never revisited: the sequence 1, 2, 1 switches again on the second 1. The tests pin down all three cases.

**Confidence is validated strictly.** It must be an `int` or `float` (not a `bool`), within `[0, 1]` and finite:

```python
def valid_confidence(value: object) -> bool:
    # bool is a subclass of int; True must not pass as 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    # range first: comparing a huge int with 0 and 1 never overflows, unlike float conversion
    return 0 <= value <= 1 and math.isfinite(value)
```

`NaN` is the case that bites. Every *ordered* comparison with `NaN` (`<`, `<=`, `>`, `>=`) is false, so `nan < 0.75` is false and a naive threshold check lets it through. (`nan != x` is true, so "every comparison" would be wrong.) `True` is the other surprise: in Python it is an `int` equal to 1. The order inside the last line matters too. `math.isfinite()` converts its argument to `float`, and for an integer such as `10**400`, which a JSON parser can produce from an ordinary integer literal, that conversion raises `OverflowError`. Comparing the integer with 0 and 1 first never overflows, so the range check runs before `math.isfinite()`, and both `10**400` and `-(10**400)` are rejected as `malformed_confidence` instead of crashing the router.

**Confidence never overrides capability or policy.** It is only a gate on whether to *consider* switching. A proposal with confidence 1.0 for a text-only route on a turn that needs tools is still rejected. The confidence here is the classifier's certainty about its own label; it is not a probability that the cheaper model will answer well. In the example it is a number I typed into a fixture.

**Alternatives go through the same guards.** The container itself is checked first: `alternatives` must be a `tuple`, otherwise the decision is `malformed_alternatives` and nothing is tried. Type annotations do not enforce this; without the check, `None` or `7` would raise `TypeError`, a string would be unpacked into single characters, and a dictionary would supply its keys as aliases. The elements stay untrusted: each fallback is checked exactly like the first choice, so a non-string element is simply `not_in_allowlist`. A fallback is not a back door around the allowlist.

**What the example trusts.** The catalog, the policy and the token counts on each `Turn` are host facts and are not validated. The proposal fields are assumed to be ordinary parsed values (strings, numbers, `None`, a tuple); anything else is rejected by the checks above, not interpreted.

## Capability guards

```python
def capability_rejection(route: Route, turn: Turn) -> str | None:
    if turn.needs_tools and not route.tools:
        return "needs_tools"
    if turn.has_attachments and not route.attachments:
        return "needs_attachments"
    if turn.history_format not in route.replays:
        return "cannot_replay_history"
    return None
```

The first two are obvious. The third one is easy to forget.

When you switch models mid-conversation, the new adapter must send the *existing* history to a different provider. If the history contains items only the old provider understands — for example opaque or encrypted reasoning blocks, or a provider-specific compaction checkpoint — the destination adapter has to either replay them faithfully or refuse. Silently dropping them produces a conversation the new model never actually had. In the example, a turn whose history format is `"opaque-reasoning"` cannot move to any route, because no adapter in the catalog declares that it can replay it.

## Check the request the destination will actually receive

It is tempting to check "does the history fit the smaller window?" But the destination does not receive just the history. It receives its own system prompt, the history, the new user message, its own serialization of the tool schemas, and it needs room for the answer:

```python
def request_tokens(route: Route, turn: Turn) -> int:
    """Size of the FULL request the destination would receive, using its own prompt and tools."""
    tools = route.tool_schema_tokens if turn.needs_tools else 0
    return (route.system_prompt_tokens + turn.history_tokens + turn.user_tokens
            + tools + route.output_budget)
```

In one test the history (6,000) plus the user message (200) is well under the 12,000-token window of `mid-tools`. Add that route's system prompt (900), tool schemas (3,000) and output budget (2,000) and the total is 12,100. The switch is rejected.

Note the order: this check runs **after** staging, against the staged destination. Only then do you know which system prompt and tool set the request will contain.

This is a **switch-admission** check, not a check of every request. If a candidate that passes the capability guards is already the live route, `route_turn` returns `already_on_route` immediately: there is no staging, no size check and no attempt at the remaining alternatives. One test starts on `mid-tools` with the same 12,100-token turn and `full-tools` as an alternative; the result is `already_on_route` with no size reported, and `full-tools` is never tried. So a keep is not a statement that the request fits. The host still has to validate every outbound request independently before sending it.

And when the request doesn't fit, the router does not compact or truncate history to squeeze into the cheaper window. Compaction loses information. Doing it only because a classifier preferred a cheaper model trades answer quality for cost without anyone deciding to make that trade. The router keeps the current route, or tries the next alternative.

## Stage, commit, rollback

```python
def _stage_and_commit(self, route: Route, turn: Turn,
                      rejected: list[tuple[str, str]]) -> Decision | None:
    old = self.runtime
    staged = replace(old, route=route, routed_turn=turn.turn_id)  # copy; live state untouched
    try:
        if self.on_staged is not None:
            self.on_staged(staged)
        size = request_tokens(staged.route, turn)
        if size > staged.route.context_tokens:
            # No compaction to squeeze into a cheaper window: keep the current route.
            rejected.append((route.alias, f"request_too_large:{size}>{route.context_tokens}"))
            return None
    except BaseException:
        # Demo rollback = drop the staged copy; self.runtime still points at `old`.
        # A real host would also release anything staging acquired
        # (connections, deferred writes, counters) here before re-raising.
        assert self.runtime is old  # debug-only invariant (stripped by -O), not the rollback
        raise
    self.runtime = staged  # COMMIT POINT: one assignment, RAM only
    try:
        self.persist(staged)  # best-effort and separate; not atomic with the commit
        persisted = True
    except Exception:
        # Not acknowledged; the write may still have happened, so a retry must be idempotent.
        # BaseException-only interruptions propagate here: RAM is switched, no Decision.
        persisted = False
    return Decision(True, route.alias, "switched", tuple(rejected), size, persisted)
```

`Runtime` and `Route` are frozen dataclasses. Assigning to a field of a frozen instance raises `FrozenInstanceError`, and [`dataclasses.replace()`](https://docs.python.org/3.11/library/dataclasses.html#dataclasses.replace) creates a new object with the changed fields. So staging is copy-on-write: the live `self.runtime` is never mutated while checks run.

That makes the commit point explicit. Before `self.runtime = staged`, the old route is live and nothing has changed. After it, the new route is live as a whole: route and turn marker together. There is no moment where half of the fields belong to the new model.

**Why `BaseException`, not `Exception`.** `KeyboardInterrupt` and `SystemExit` [inherit from `BaseException`](https://docs.python.org/3.11/library/exceptions.html) precisely so that `except Exception` does not catch them. A user pressing Ctrl+C while the host is warming up the new adapter is exactly the moment you want cleanup to run. The handler doesn't swallow the exception; it re-raises after rollback. In Python 3.11, the version used here, `asyncio.CancelledError` is also a `BaseException` subclass and not an `Exception` (the interruption tests below rely on this), so async cancellation takes the same path.

The `assert self.runtime is old` in that handler is a debug-only invariant check. It documents that staging never touched the live state, and `python -O` strips it. It is not the rollback mechanism: rollback is simply that `self.runtime` was never reassigned.

**Be honest about what rollback means here.** In this demo, rollback is discarding a reference: the staged copy is never installed, so there is nothing to undo. A real host may have acquired resources while staging — opened a connection, built a system prompt, queued a session-row write, reset a counter. Copy-on-write in memory does not undo those. Either defer such side effects until after the commit, or release them explicitly in the rollback path.

**Persistence is separate, and not atomic with the commit.** The in-memory switch is the transaction. Writing the new route to a session store happens afterwards and may fail. Pretending the two are atomic would be worse: a failed write would push you into undoing a switch that the rest of the turn may already depend on.

Read the flag precisely. `persisted=True` means the `persist` callback returned, i.e. the store acknowledged the write. `persisted=False` means an `Exception` arrived instead of an acknowledgement. It does **not** prove that nothing was written: a store can write and then raise, for example when the acknowledgement is lost on the way back, and one test does exactly that. The state of the store is uncertain, so a retry must be idempotent (writing the same route for the same turn again must be harmless). In both cases the new route stays live in RAM.

Interruptions are different. `KeyboardInterrupt`, `SystemExit` and `asyncio.CancelledError` are not caught by `except Exception`, so if one arrives during `persist`, it propagates out of `route_turn`. The commit has already happened: the new route and its turn marker are live, but the caller gets **no** `Decision` and therefore no `persisted` flag. A caller that handles such an interruption has to treat the persistence state as unknown, read the live route from `router.runtime`, and retry the write idempotently if it continues. The tests cover all three interruptions and the write-then-raise case.

## Running it

The complete source is small enough to download and read in a few minutes:
[`routing.py`]({{ '/assets/examples/model-routing-jev/routing.py' | relative_url }}),
[`catalog.py`]({{ '/assets/examples/model-routing-jev/catalog.py' | relative_url }}),
[`demo.py`]({{ '/assets/examples/model-routing-jev/demo.py' | relative_url }}),
[`tests/test_routing.py`]({{ '/assets/examples/model-routing-jev/tests/test_routing.py' | relative_url }}),
[`pyproject.toml`]({{ '/assets/examples/model-routing-jev/pyproject.toml' | relative_url }}),
[`uv.lock`]({{ '/assets/examples/model-routing-jev/uv.lock' | relative_url }}) and
[`README.md`]({{ '/assets/examples/model-routing-jev/README.md' | relative_url }}).
Put the files in one directory (the test file in `tests/`, next to an empty `tests/__init__.py`) and run:

```bash
uv sync
uv run pytest -v
uv run python demo.py
```

When preparing this article I ran it on 12 October 2026 with Python 3.11.12, uv 0.7.8 and pytest 9.1.1 (the version `uv.lock` resolves to). All 45 tests passed. The demo starts on `full-tools` and makes seven routing calls across six turn ids (the second call repeats turn 1):

```text
short question   -> SWITCH fast-text  reason=switched tokens=3450 persisted=True
same turn again  -> keep   fast-text  reason=already_routed_this_turn tokens=None persisted=None
raw model id     -> keep   fast-text  reason=no_acceptable_route tokens=None persisted=None
                   rejected vendor/some-model-id: not_in_allowlist
NaN confidence   -> keep   fast-text  reason=malformed_confidence tokens=None persisted=None
needs tools      -> SWITCH full-tools reason=switched tokens=13900 persisted=True
                   rejected fast-text: needs_tools
                   rejected mid-tools: request_too_large:12100>12000
opaque history   -> keep   full-tools reason=no_acceptable_route tokens=None persisted=None
                   rejected fast-text: cannot_replay_history
persist fails    -> SWITCH mid-tools  reason=switched tokens=7000 persisted=False
live route: mid-tools | saved: ['fast-text', 'full-tools']
```

The "needs tools" turn shows the whole chain: the first choice is rejected on capability, the alternative `mid-tools` passes capability but fails the full-request check, and `full-tools` is accepted. The last turn shows a committed switch whose persistence failed: the live route is `mid-tools`, but the store still only has the two earlier saves.

The tests cover the same paths and a few more: an accepted switch, unknown and non-string aliases, low confidence, `NaN`/infinite/out-of-range/boolean/string confidence and integers too large for a `float` (`10**400` and `-(10**400)`), an alternatives container that is not a tuple (`None`, `7`, a string, a dict, a list), non-string elements inside a valid tuple, tool, attachment and history-replay rejections, the full request exceeding the window only after the destination prompt and tools are added, alternatives going through the same guards, disabled routing, a manual pin, one switch per turn, a rejected or `already_on_route` first decision that does not mark the turn, a revisited turn id that switches again, the oversized current route kept without trying a fitting alternative, `KeyboardInterrupt`/`SystemExit`/`RuntimeError` raised after staging leaving the old runtime object in place, a persistence failure after commit leaving the new route live, a store that writes and then raises, and `KeyboardInterrupt`/`SystemExit`/`asyncio.CancelledError` during persistence propagating with the new route live and no `Decision`.

## What the example leaves out

This is a serial, single-threaded simulation. It deliberately does not include:

- **concurrency** — two turns of the same session racing to switch would need a lock or a version check on the runtime;
- **real token counting** — the numbers are configured; a real host should estimate with the destination's tokenizer, or use a conservative margin;
- **network adapters, credentials and timeouts** for the classifier call; a timed-out classifier should simply mean "keep the current route";
- **cooldowns** to stop a session from flapping between routes on alternating turns;
- **releasing real resources on rollback**, discussed above;
- **telemetry**; the `Decision` object carries the reasons a log line would need.

Each of these is worth adding in a real host. None of them changes the core shape.

## Checklist

Before letting a classifier switch models in an agent host:

- [ ] Routing is opt-in and off by default; a manual pin disables it for that session.
- [ ] The classifier can only name aliases from a host-owned allowlist; no model ids or endpoints flow from its output.
- [ ] Credentials live with adapters, not in the routing catalog.
- [ ] Confidence is checked for type, range and finiteness before the threshold, range before any float conversion; booleans are rejected.
- [ ] The alternatives container has the expected type; its elements are still treated as untrusted.
- [ ] Confidence gates *whether* to consider a switch; it never overrides capability or policy checks.
- [ ] Tools, attachments and history replay are host-declared capabilities, checked for every candidate including fallbacks.
- [ ] The caller routes once per user turn, before the agent loop starts; the router's turn marker only blocks a repeat after a committed switch, for unique, increasing turn ids.
- [ ] The full destination request — its system prompt, history, user message, tool schemas and output budget — is checked after staging.
- [ ] A request that doesn't fit keeps the current route; nothing is compacted just to fit a cheaper model.
- [ ] Keeping the current route is not proof that the request fits; every outbound request is validated separately.
- [ ] Staging works on a copy; there is a single, visible commit point.
- [ ] Any exception before commit, including `KeyboardInterrupt`, rolls back and re-raises; side effects of staging are deferred or released.
- [ ] Persistence happens after commit and is reported separately; a failure flag means "not acknowledged", not "not written", so retries are idempotent.
- [ ] An interruption after commit propagates without a decision; the caller treats persistence as unknown.

## Sources

- TypeSafe AI, [Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — announcement by the company's founder. Performance and reliability claims there are the vendor's own.
- TypeSafe AI, [Quickstart](https://docs.typesafe.ai/introduction/quickstart) — question types and the shape of choice results.
- Sydney Runkle and Hunter Lovell, [Building a harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev), LangChain, 17 September 2026.
- Python 3.11 documentation: [`dataclasses`](https://docs.python.org/3.11/library/dataclasses.html) and [built-in exceptions](https://docs.python.org/3.11/library/exceptions.html).
