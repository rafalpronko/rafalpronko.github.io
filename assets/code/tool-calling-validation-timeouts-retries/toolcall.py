"""Validated, bounded tool calling for a synthetic read-only inventory lookup.

Python 3.11+, standard library only. The model's proposal is treated as untrusted
input: it is parsed strictly, checked against an explicit allowlist, authorized on
the server side, executed under per-attempt and overall deadlines, and its result
is validated again before anything goes back to the model.

The application owns the registry, schemas, valid defaults, scopes and budgets.
Timing settings must be finite and positive; max_attempts a positive integer;
backoff a nonempty sequence of finite nonnegative delays. TurnBudget limits and
counters must be nonnegative integers. A logger must not block or raise.
execute() requires a call that has already passed validation and authorization.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

MAX_ARGUMENTS_BYTES = 2048

# Synthetic, in-memory data. Nothing here talks to a real system.
INVENTORY: dict[tuple[str, str], dict[str, int]] = {
    ("ABC-1234", "WAW"): {"on_hand": 42, "reserved": 5},
    ("ABC-1234", "KRK"): {"on_hand": 0, "reserved": 0},
    ("XYZ-9999", "GDN"): {"on_hand": 7, "reserved": 7},
}


class ProposalRejected(Exception):
    """The proposal is invalid or not allowed. Never executed; the model may repair it."""

    def __init__(self, code: str, field: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.field = field

    def to_feedback(self) -> dict[str, Any]:
        # Only fixed codes and schema-owned field names. Raw values are never echoed.
        return {"status": "rejected", "error": self.code, "field": self.field}


class TransientToolError(Exception):
    """Raised by a tool adapter for failures known to be safe to retry on reads."""


class PermanentToolError(Exception):
    """Raised by a tool adapter for failures that retrying will not fix."""


class LoopLimitExceeded(Exception):
    """The whole request ran out of tool calls or repair attempts."""


@dataclass(frozen=True)
class Param:
    kind: type  # str, int or bool, compared with `type(value) is kind`
    required: bool = True
    default: Any = None
    choices: frozenset[str] | None = None
    pattern: str | None = None
    min: int | None = None
    max: int | None = None
    max_len: int = 64


@dataclass(frozen=True)
class ToolSpec:
    name: str
    params: dict[str, Param]
    output: dict[str, Param]
    handler: Callable[..., Awaitable[Any]]
    read_only: bool
    required_scope: str


@dataclass(frozen=True)
class ToolCall:
    spec: ToolSpec
    args: dict[str, Any]


@dataclass(frozen=True)
class Outcome:
    status: str  # ok | invalid_output | tool_error | retries_exhausted | budget_exhausted | unknown_effect
    tool: str
    attempts: int
    data: dict[str, Any] | None = None
    error_code: str | None = None

    def to_model_message(self) -> str:
        # Tool output is data for the model, not instructions. Keep it structured.
        return json.dumps({"status": self.status, "tool": self.tool,
                           "error": self.error_code, "data": self.data})


# ---------------------------------------------------------------- validation

def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProposalRejected("duplicate_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ProposalRejected("bad_shape")  # NaN, Infinity, -Infinity


def _check_value(name: str, value: Any, p: Param) -> None:
    if type(value) is not p.kind:  # rejects True for int, 1.0 for int, 1 for bool
        raise ProposalRejected("wrong_type", name)
    if p.kind is str:
        if len(value) > p.max_len:
            raise ProposalRejected("bad_value", name)
        if p.choices is not None and value not in p.choices:
            raise ProposalRejected("bad_value", name)
        if p.pattern is not None and not re.fullmatch(p.pattern, value):
            raise ProposalRejected("bad_value", name)
    if p.kind is int:
        if (p.min is not None and value < p.min) or (p.max is not None and value > p.max):
            raise ProposalRejected("bad_value", name)


def _check_object(obj: Any, schema: dict[str, Param]) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ProposalRejected("bad_shape")
    unknown = obj.keys() - schema.keys()
    if unknown:
        raise ProposalRejected("unknown_field")  # the unknown names are not echoed
    clean: dict[str, Any] = {}
    for name, p in schema.items():
        if name not in obj:
            if p.required:
                raise ProposalRejected("missing_field", name)
            clean[name] = p.default
            continue
        _check_value(name, obj[name], p)
        clean[name] = obj[name]
    return clean


def validate_proposal(name: Any, arguments_json: Any, registry: dict[str, ToolSpec]) -> ToolCall:
    """Turn an untrusted (name, arguments JSON string) pair into a typed ToolCall."""
    if not isinstance(name, str) or not isinstance(arguments_json, str):
        raise ProposalRejected("bad_shape")
    spec = registry.get(name)  # plain dict lookup: no eval, getattr or dynamic import
    if spec is None:
        raise ProposalRejected("unknown_tool")
    try:
        arguments_bytes = arguments_json.encode("utf-8")
    except UnicodeEncodeError:
        raise ProposalRejected("bad_encoding") from None
    if len(arguments_bytes) > MAX_ARGUMENTS_BYTES:
        raise ProposalRejected("too_large")
    try:
        parsed = json.loads(arguments_json, object_pairs_hook=_reject_duplicates,
                            parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError):
        raise ProposalRejected("bad_shape") from None
    return ToolCall(spec, _check_object(parsed, spec.params))


def authorize(call: ToolCall, scopes: frozenset[str]) -> ToolCall:
    """Server-side decision based on the caller's scopes, never on what the model says."""
    if call.spec.required_scope not in scopes:
        raise ProposalRejected("not_authorized")
    return call


# ----------------------------------------------------------------- execution

async def execute(
    call: ToolCall,
    *,
    call_id: str,
    request_id: str | None = None,
    attempt_timeout_s: float = 1.0,
    budget_s: float = 3.0,
    max_attempts: int = 3,
    backoff_s: tuple[float, ...] = (0.1, 0.3),
    log: Callable[[dict[str, Any]], None] | None = None,
) -> Outcome:
    """Run an already validated, authorized call using application-owned settings.

    The module's trusted-caller contract applies; configuration is not validated
    here. Cancellation propagates without a log event; the caller must log it.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + budget_s  # loop.time() is monotonic
    spec = call.spec

    def emit(event: str, **fields: Any) -> None:
        if log is not None:
            log({"event": event, "call_id": call_id, "request_id": request_id,
                 "tool": spec.name, "elapsed_ms": round((loop.time() - started) * 1000), **fields})

    def finish(status: str, attempts: int, data: Any = None, error: str | None = None) -> Outcome:
        emit("tool_outcome", status=status, attempts=attempts, error=error)
        return Outcome(status, spec.name, attempts, data, error)

    attempt = 0
    error = "timeout"
    while attempt < max_attempts:
        if loop.time() >= deadline:
            return finish("budget_exhausted", attempt, error=error)
        attempt += 1
        try:
            async with asyncio.timeout_at(min(loop.time() + attempt_timeout_s, deadline)):
                raw = await spec.handler(**call.args)
        except TimeoutError:
            error = "timeout"
        except TransientToolError:
            error = "transient"
        except PermanentToolError:
            emit("tool_attempt", attempt=attempt, result="permanent_error")
            return finish("tool_error", attempt, error="permanent")
        except Exception:  # CancelledError is a BaseException and is not caught here
            emit("tool_attempt", attempt=attempt, result="internal_error")
            return finish("tool_error", attempt, error="internal_error")
        else:
            emit("tool_attempt", attempt=attempt, result="returned")
            try:
                data = _check_object(raw, spec.output)
            except ProposalRejected:
                return finish("invalid_output", attempt, error="invalid_output")
            return finish("ok", attempt, data=data)

        emit("tool_attempt", attempt=attempt, result=error)
        if not spec.read_only:
            # The request may have reached the server. Retrying could apply it twice.
            return finish("unknown_effect", attempt, error=error)
        if attempt >= max_attempts:
            break
        delay = backoff_s[min(attempt - 1, len(backoff_s) - 1)] * random.uniform(0.5, 1.0)
        if loop.time() + delay >= deadline:
            return finish("budget_exhausted", attempt, error=error)
        await asyncio.sleep(delay)
    return finish("retries_exhausted", attempt, error=error)


# ------------------------------------------------------------ request loop

@dataclass
class TurnBudget:
    """Global bounds for one user request, across every model round-trip."""
    max_tool_calls: int = 4
    max_repairs: int = 2
    tool_calls: int = 0
    repairs: int = 0

    def take_call(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise LoopLimitExceeded("tool_calls")
        self.tool_calls += 1

    def take_repair(self) -> None:
        if self.repairs >= self.max_repairs:
            raise LoopLimitExceeded("repairs")
        self.repairs += 1


async def handle_proposal(
    name: Any,
    arguments_json: Any,
    *,
    registry: dict[str, ToolSpec],
    scopes: frozenset[str],
    budget: TurnBudget,
    call_id: str,
    request_id: str | None = None,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Validate and authorize a proposal, or return fixed rejection feedback.

    The caller owns logging for cancellation and exhausted TurnBudget limits.
    A repair limit can raise before proposal_rejected is emitted.
    """
    try:
        call = authorize(validate_proposal(name, arguments_json, registry), scopes)
    except ProposalRejected as rejected:
        budget.take_repair()  # repairing = asking the model for a *different* proposal
        if log is not None:
            log({"event": "proposal_rejected", "call_id": call_id, "request_id": request_id,
                 "error": rejected.code, "field": rejected.field})
        return json.dumps(rejected.to_feedback())
    budget.take_call()  # retries inside execute() reuse this one authorized call
    outcome = await execute(call, call_id=call_id, request_id=request_id, log=log)
    return outcome.to_model_message()


# ------------------------------------------------------------------ the tool

async def get_stock(sku: str, warehouse: str, max_age_s: int) -> dict[str, Any]:
    await asyncio.sleep(0)  # stands in for a network call
    row = INVENTORY.get((sku, warehouse))
    if row is None:
        raise PermanentToolError("not_found")
    return {"sku": sku, "warehouse": warehouse, **row}


def make_registry() -> dict[str, ToolSpec]:
    return {
        "get_stock": ToolSpec(
            name="get_stock",
            params={
                "sku": Param(str, pattern=r"[A-Z]{3}-[0-9]{4}", max_len=8),
                "warehouse": Param(str, choices=frozenset({"WAW", "KRK", "GDN"})),
                "max_age_s": Param(int, required=False, default=60, min=1, max=300),
            },
            output={
                "sku": Param(str, pattern=r"[A-Z]{3}-[0-9]{4}", max_len=8),
                "warehouse": Param(str, choices=frozenset({"WAW", "KRK", "GDN"})),
                "on_hand": Param(int, min=0, max=1_000_000),
                "reserved": Param(int, min=0, max=1_000_000),
            },
            handler=get_stock,
            read_only=True,
            required_scope="inventory:read",
        ),
    }
