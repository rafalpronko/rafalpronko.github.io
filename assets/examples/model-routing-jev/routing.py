"""Host-side model routing: allowlist, capability guards, staged switch, commit or rollback.

Synthetic, serial, single-threaded simulation. No network, no real tokenizer, no SDK.
The classifier proposal is a fixture shaped like a choice result (a label plus a
confidence); it is not a real Jev response.

Scope: switch admission only. The catalog, policy and turn token counts are trusted host
facts, and proposal fields are assumed to be ordinary parsed values. A route that is already
live is kept without a size check, so every outbound request still needs its own validation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Route:
    """One allowlisted destination. Host-owned; the catalog holds no credentials."""

    alias: str  # host-local alias the classifier may name, e.g. "fast-text"
    adapter: str  # host adapter that knows how to call the provider
    model: str  # provider model id; never taken from the proposal
    context_tokens: int  # destination window (synthetic numbers in this demo)
    output_budget: int  # tokens reserved for the answer
    system_prompt_tokens: int  # the destination's own system prompt
    tool_schema_tokens: int  # tool schemas as this adapter serialises them
    tools: bool
    attachments: bool
    replays: frozenset[str]  # history formats this adapter can send back


@dataclass(frozen=True)
class Turn:
    turn_id: int
    user_tokens: int  # configured synthetic counts, not tokenizer output
    history_tokens: int
    history_format: str  # e.g. "plain" or "opaque-reasoning"
    needs_tools: bool = False
    has_attachments: bool = False


@dataclass(frozen=True)
class Proposal:
    """What a classifier suggests. Choice, confidence and every alternative are untrusted;
    the host validates them, and `alternatives` must be a tuple."""

    choice: object
    confidence: object
    alternatives: tuple[object, ...] = ()


@dataclass(frozen=True)
class Policy:
    enabled: bool = False  # routing is opt-in
    min_confidence: float = 0.75
    manual_pin: str | None = None  # a user pin always wins over automatic routing


@dataclass(frozen=True)
class Runtime:
    route: Route
    routed_turn: int | None = None  # turn id of the last committed switch, nothing more


@dataclass(frozen=True)
class Decision:
    applied: bool
    route: str
    reason: str
    rejected: tuple[tuple[str, str], ...] = ()
    request_tokens: int | None = None
    persisted: bool | None = None  # True: write acknowledged; False: not acknowledged; None: none


def request_tokens(route: Route, turn: Turn) -> int:
    """Size of the FULL request the destination would receive, using its own prompt and tools."""
    tools = route.tool_schema_tokens if turn.needs_tools else 0
    return (route.system_prompt_tokens + turn.history_tokens + turn.user_tokens
            + tools + route.output_budget)


def valid_confidence(value: object) -> bool:
    # bool is a subclass of int; True must not pass as 1.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    # range first: comparing a huge int with 0 and 1 never overflows, unlike float conversion
    return 0 <= value <= 1 and math.isfinite(value)


def capability_rejection(route: Route, turn: Turn) -> str | None:
    if turn.needs_tools and not route.tools:
        return "needs_tools"
    if turn.has_attachments and not route.attachments:
        return "needs_attachments"
    if turn.history_format not in route.replays:
        return "cannot_replay_history"
    return None


class Router:
    def __init__(self, catalog: Mapping[str, Route], runtime: Runtime, policy: Policy,
                 persist: Callable[[Runtime], None],
                 on_staged: Callable[[Runtime], None] | None = None) -> None:
        self.catalog = dict(catalog)
        self.runtime = runtime
        self.policy = policy
        self.persist = persist
        self.on_staged = on_staged  # e.g. warm a connection; may fail or be interrupted

    def keep(self, reason: str, rejected: tuple[tuple[str, str], ...] = ()) -> Decision:
        return Decision(False, self.runtime.route.alias, reason, rejected)

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
