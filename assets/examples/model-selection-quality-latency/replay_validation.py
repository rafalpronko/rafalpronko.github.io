"""Replay-log schema, bounded numbers, prices and pairing (no network calls)."""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

STATUSES = frozenset({"ok", "malformed", "timeout", "abstain"})
VALID_RESPONSE = frozenset({"ok", "abstain"})
MAX_TOKENS = 2 ** 63 - 1  # explicit input bound, not a provider's context limit
LINE_BREAKING = frozenset({"Cc", "Zl", "Zp"})  # controls and line/paragraph separators


class LogError(ValueError):
    """Input cannot be evaluated safely or compared."""


def finite_number(value: object, where: str, *, positive: bool = False) -> float:
    condition = "finite and > 0" if positive else "finite and >= 0"
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError
        number = float(value)
        if not math.isfinite(number) or (number <= 0 if positive else number < 0):
            raise ValueError
    except (ValueError, OverflowError):
        raise LogError(f"{where} must be {condition}") from None
    return number


def check_text(value: str, where: str, *, identifier: bool) -> str:
    """Reject unpaired surrogates; identifiers printed in reports also reject line breaks."""
    for char in value:
        category = unicodedata.category(char)
        if category == "Cs":
            raise LogError(f"{where} contains an unpaired surrogate (not valid Unicode text)")
        if identifier and category in LINE_BREAKING:
            raise LogError(f"{where} contains a control or line-separator character")
    return value


def candidate_name(path: Path) -> str:
    """Candidate name = file stem; '_' prefixes are reserved for price-file comments."""
    name = path.stem
    if name.startswith("_"):
        raise LogError(f"candidate name (file stem) {name!r} must not start with '_'; "
                       "price-file keys starting with '_' are comments")
    return check_text(name, f"candidate name (file stem) {name!r}", identifier=True)


@dataclass(frozen=True)
class Attempt:
    case_id: str
    slice: str
    expected: str | None
    status: str
    output: str | None
    elapsed_ms: float
    input_tokens: int
    output_tokens: int

    @property
    def correct(self) -> bool:
        if self.status == "ok":
            return self.expected is not None and self.output == self.expected
        if self.status == "abstain":
            return self.expected is None
        return False


def _parse_attempt(raw: object, where: str) -> Attempt:
    if not isinstance(raw, dict):
        raise LogError(f"{where}: record must be a JSON object")
    try:
        case_id, slice_, expected = raw["case_id"], raw["slice"], raw["expected"]
        status, output, elapsed = raw["status"], raw["output"], raw["elapsed_ms"]
        tokens = raw["input_tokens"], raw["output_tokens"]
    except KeyError as exc:
        raise LogError(f"{where}: missing field {exc}") from None
    for key, value in [("case_id", case_id), ("slice", slice_)]:
        if not isinstance(value, str) or not value:
            raise LogError(f"{where}: {key} must be a non-empty string")
        check_text(value, f"{where}: {key}", identifier=True)
    if expected is not None and not isinstance(expected, str):
        raise LogError(f"{where}: expected must be a string or null")
    if not isinstance(status, str) or status not in STATUSES:
        raise LogError(f"{where}: unknown status {status!r}")
    if output is not None and not isinstance(output, str):
        raise LogError(f"{where}: output must be a string or null for every status")
    for key, value in [("expected", expected), ("output", output)]:
        check_text(value or "", f"{where}: {key}", identifier=False)
    if status == "ok" and not isinstance(output, str):
        raise LogError(f"{where}: status 'ok' requires a string output")
    if status == "abstain" and output is not None:
        raise LogError(f"{where}: status 'abstain' requires null output")
    duration = finite_number(elapsed, f"{where}: elapsed_ms", positive=True)
    for value in tokens:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_TOKENS:
            raise LogError(f"{where}: token counts must be integers in 0..{MAX_TOKENS}")
    return Attempt(case_id, slice_, expected, status, output, duration, *tokens)


def parse_json(text: str, where: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        # ValueError also covers Python's integer-digit limit.
        raise LogError(f"{where}: invalid JSON ({exc})") from None


def load_log(path: Path) -> dict[str, Attempt]:
    attempts: dict[str, Attempt] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                where = f"{path.name}:{lineno}"
                attempt = _parse_attempt(parse_json(line, where), where)
                if attempt.case_id in attempts:
                    raise LogError(f"{where}: duplicate case_id {attempt.case_id!r}")
                attempts[attempt.case_id] = attempt
    except UnicodeDecodeError:
        raise LogError(f"{path.name}: invalid UTF-8") from None
    if not attempts:
        raise LogError(f"{path.name}: log is empty")
    return attempts


def check_paired(logs: dict[str, dict[str, Attempt]]) -> list[str]:
    """Require identical case sets with identical gold labels and slices."""
    names = list(logs)
    if len(names) < 2:
        raise LogError("need at least two candidate logs to compare")
    reference = logs[names[0]]
    for name in names[1:]:
        other = logs[name]
        missing = sorted(reference.keys() - other.keys())
        extra = sorted(other.keys() - reference.keys())
        if missing or extra:
            raise LogError(f"{name} is not paired with {names[0]}: "
                           f"missing={missing} extra={extra}")
        for case_id, ref in reference.items():
            got = other[case_id]
            if (got.expected, got.slice) != (ref.expected, ref.slice):
                raise LogError(f"{case_id}: expected/slice differ between "
                               f"{names[0]} and {name}")
    return sorted(reference)


@dataclass(frozen=True)
class Price:
    input_per_1k: float
    output_per_1k: float

    def __post_init__(self) -> None:
        finite_number(self.input_per_1k, "input price")
        finite_number(self.output_per_1k, "output price")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        try:
            cost = (input_tokens * self.input_per_1k
                    + output_tokens * self.output_per_1k) / 1000
        except OverflowError:
            raise LogError("token cost overflow") from None
        return finite_number(cost, "token cost (overflow)")


def load_prices(path: Path) -> dict[str, Price]:
    try:
        raw = parse_json(path.read_text(encoding="utf-8"), path.name)
    except UnicodeDecodeError:
        raise LogError(f"{path.name}: invalid UTF-8") from None
    if not isinstance(raw, dict):
        raise LogError(f"{path.name}: expected a JSON object")
    prices = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(entry, dict) or not {"input_per_1k", "output_per_1k"} <= entry.keys():
            raise LogError(f"{path.name}: {name} needs input_per_1k and output_per_1k")
        values = [finite_number(entry[k], f"{path.name}: {name} prices")
                  for k in ("input_per_1k", "output_per_1k")]
        prices[name] = Price(*values)
    return prices


@dataclass(frozen=True)
class Constraints:
    min_quality: float
    max_malformed_rate: float
    max_p95_all_ms: float

    def __post_init__(self) -> None:
        for name, value in [("min-quality", self.min_quality),
                            ("max-malformed-rate", self.max_malformed_rate)]:
            if finite_number(value, name) > 1:
                raise LogError(f"{name} must be a finite rate in 0..1")
        finite_number(self.max_p95_all_ms, "max-p95-ms", positive=True)
