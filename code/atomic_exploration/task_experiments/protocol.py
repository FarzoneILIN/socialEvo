"""Strict model-output parsing and exact, portable experiment records."""

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non_finite_json:{value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non_finite_json_number")
    return parsed


def parse_object(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("response_not_text")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError("expected_json_object")
    return value


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def exact(value: Fraction | int) -> dict:
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def from_exact(value: dict) -> Fraction:
    if (not isinstance(value, dict) or set(value) != {"numerator", "denominator"}
            or type(value["numerator"]) is not int
            or type(value["denominator"]) is not int or value["denominator"] <= 0):
        raise ValueError("invalid_exact_fraction")
    fraction = Fraction(value["numerator"], value["denominator"])
    if exact(fraction) != value:
        raise ValueError("non_canonical_fraction")
    return fraction


def reward_record(value: Fraction | int) -> dict:
    fraction = Fraction(value)
    number = int(fraction) if fraction.denominator == 1 else float(fraction)
    if not math.isfinite(number):
        raise ValueError("non_finite_reward")
    return {"reward": number, "reward_exact": exact(fraction)}


@dataclass(frozen=True)
class Validation:
    accepted: bool
    reason: str
    action: dict | None = None

    def as_record(self) -> dict:
        return {"accepted": self.accepted, "reason": self.reason,
                "action": self.action}


def validate_talk(action: dict, max_chars: int = 240) -> Validation:
    if not isinstance(action, dict) or set(action) != {"type", "content"}:
        return Validation(False, "invalid_fields")
    if action["type"] != "say" or not isinstance(action["content"], str):
        return Validation(False, "invalid_talk")
    if len(action["content"]) > max_chars:
        return Validation(False, "message_too_long")
    return Validation(True, "accepted", {"type": "say", "content": action["content"]})
