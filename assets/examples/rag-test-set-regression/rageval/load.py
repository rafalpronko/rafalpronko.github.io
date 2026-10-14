import json
from pathlib import Path

from rageval.validate import ValidationError


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(name):
    raise ValidationError(f"non-finite number {name} is not valid JSON")


def load_json(path):
    """Load a JSON file, rejecting duplicate keys that json.load would silently
    overwrite and the NaN/Infinity constants it would silently accept."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = getattr(exc, "strerror", None) or exc
        raise ValidationError(f"cannot read {path}: {reason}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=_reject_non_finite)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    except ValidationError as exc:
        raise ValidationError(f"{path}: {exc}") from exc
    except ValueError as exc:  # decoder limits, e.g. an integer with more than 4300 digits
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ValidationError(f"{path}: invalid JSON: nesting too deep") from exc
