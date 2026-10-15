"""Frozen, simultaneous Sudoku inputs and strict response scoring.

The oracle is used only after a response. Ordinary prompts are reconstructed
from an explicit allowlist and never include solution, oracle, case stratum,
condition name, repetition number, or a partner's action.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

AGENTS = ("A", "B")
CONDITIONS = ("both_visible", "mask_A_to_B", "mask_B_to_A", "both_hidden")
BASELINE = "full_information"
REPETITIONS = 3
MAX_MESSAGE_CHARS = 1200
SAVED = {"saved"}
RULES = (
    "Solve standard 9x9 Sudoku: each row, column, and 3x3 box must contain "
    "digits 1 through 9 exactly once. Coordinates use rows and columns 1 through 9. "
    "A given is [row, column, digit]. All provided givens are true. "
    "The target cell is initially empty. Every call refers to the same unchanged initial state."
)


def partner(agent: str) -> str:
    if agent not in AGENTS:
        raise ValueError("unknown agent")
    return "B" if agent == "A" else "A"


def message_slot_id(case_id: str, sender: str) -> str:
    return f"{case_id}/message/{sender}"


def incoming_visible(condition: str, receiver: str) -> bool:
    partner(receiver)
    if condition not in (*CONDITIONS, BASELINE):
        raise ValueError("unknown condition")
    return condition in ("both_visible", "mask_A_to_B" if receiver == "A" else "mask_B_to_A")


def build_slots(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Two messages and 30 decisions per independent state, never resampling."""
    slots: list[dict[str, Any]] = []
    ids = [str(case["case_id"]) for case in cases]
    if len(set(ids)) != len(ids) or any(not cid for cid in ids):
        raise ValueError("case_id values must be nonempty and unique")
    for case_id in ids:
        dependencies = [message_slot_id(case_id, sender) for sender in AGENTS]
        for sender in AGENTS:
            slots.append(dict(slot_id=message_slot_id(case_id, sender), case_id=case_id,
                              kind="message", sender=sender, receiver=partner(sender),
                              condition="generated", rep=0, dependencies=[]))
        for condition in (*CONDITIONS, BASELINE):
            for receiver in AGENTS:
                for rep in range(REPETITIONS):
                    slots.append(dict(slot_id=f"{case_id}/decision/{condition}/{receiver}/r{rep}",
                                      case_id=case_id, kind="decision", sender=partner(receiver),
                                      receiver=receiver, condition=condition, rep=rep,
                                      dependencies=list(dependencies)))
    return slots


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def parse_response(kind: str, response: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate exactly one JSON object. Unavailable is not returned-invalid."""
    if not isinstance(response, Mapping) or response.get("status") not in SAVED:
        return dict(available=False, format_valid=None, reason="response_not_saved", parsed=None)
    text = response.get("text")
    if not isinstance(text, str):
        return dict(available=False, format_valid=None, reason="response_text_missing_or_not_string", parsed=None)
    try:
        parsed = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, TypeError, json.JSONDecodeError):
        return dict(available=True, format_valid=False, reason="invalid_json", parsed=None)
    if not isinstance(parsed, dict):
        reason = "json_not_object"
    elif kind == "message":
        if set(parsed) != {"message"}:
            reason = "message_schema"
        elif not isinstance(parsed["message"], str) or not parsed["message"].strip():
            reason = "message_must_be_nonempty_string"
        elif len(parsed["message"]) > MAX_MESSAGE_CHARS:
            reason = "message_too_long"
        else:
            try:
                parsed["message"].encode("utf-8")
            except UnicodeEncodeError:
                return dict(available=True, format_valid=False, reason="invalid_unicode", parsed=None)
            return dict(available=True, format_valid=True, reason=None, parsed=parsed)
    elif kind == "decision":
        if "digit" not in parsed or set(parsed) - {"digit", "explanation"}:
            reason = "decision_schema"
        elif parsed["digit"] is not None and (type(parsed["digit"]) is not int or not 1 <= parsed["digit"] <= 9):
            reason = "digit_must_be_integer_1_to_9_or_null"
        elif "explanation" in parsed and not isinstance(parsed["explanation"], str):
            reason = "explanation_must_be_string"
        else:
            try:
                parsed.get("explanation", "").encode("utf-8")
            except UnicodeEncodeError:
                return dict(available=True, format_valid=False, reason="invalid_unicode", parsed=None)
            return dict(available=True, format_valid=True, reason=None, parsed=parsed)
    else:
        raise ValueError("unknown slot kind")
    return dict(available=True, format_valid=False, reason=reason, parsed=None)


def _public_state(case: Mapping[str, Any], agent: str) -> dict[str, Any]:
    return dict(agent=agent, target=list(case["target"]),
                public_givens=[list(given) for given in case["public_givens"]],
                own_private_givens=[list(given) for given in case["private_givens"][agent]])


def _frozen_message(case: Mapping[str, Any], sender: str,
                    responses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return parse_response("message", responses.get(message_slot_id(str(case["case_id"]), sender)))


def prepare_input(case: Mapping[str, Any], slot: Mapping[str, Any],
                  responses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return text messages or a local skip. No prior conversation is replayed."""
    if slot["case_id"] != case["case_id"]:
        raise ValueError("case and slot disagree")
    kind = slot["kind"]
    if kind == "message":
        agent = slot["sender"]
        partner(agent)
        content = _public_state(case, agent)
        system = (RULES + " You are sending a single message to the other agent before either agent "
                  "makes a decision. You cannot see the other agent's private givens, message, or action. "
                  "Communicate your actual private givens and any deductions justified by the givens you can see. "
                  "Do not invent clues or claim knowledge of the other agent's private information. "
                  "This message will be frozen and reused. Return only a JSON object with exactly the key "
                  f"message, whose value is a nonempty string of at most {MAX_MESSAGE_CHARS} characters: "
                  '{"message":"..."}.')
    elif kind == "decision":
        agent = slot["receiver"]
        other = partner(agent)
        condition = slot["condition"]
        visible = incoming_visible(condition, agent)
        own = _frozen_message(case, agent, responses)
        if own["format_valid"] is not True:
            return {"skip": f"own_message_unavailable_or_invalid:{agent}"}
        content = _public_state(case, agent)
        content["own_frozen_message"] = own["parsed"]["message"]
        if condition == BASELINE:
            content["partner_private_givens"] = [list(given) for given in case["private_givens"][other]]
            content["partner_message"] = None
        elif visible:
            inbound = _frozen_message(case, other, responses)
            if inbound["format_valid"] is not True:
                return {"skip": f"visible_partner_message_unavailable_or_invalid:{other}"}
            content["partner_message"] = inbound["parsed"]["message"]
        else:
            content["partner_message"] = None
        system = (RULES + " Decide which digit to fill in the target cell using the information below. "
                  "Your own frozen message is retained as your prior statement. A partner message, if present, "
                  "may contain useful facts or deductions, but you should check its claims against Sudoku rules. "
                  "If partner_private_givens are provided, those are additional true givens. "
                  "No partner decision or changed board is available. Return only a JSON object containing "
                  "digit as an integer 1 through 9, or null to abstain if you cannot determine it. "
                  'An optional explanation string is allowed. Example: {"digit":4,"explanation":"..."}.')
    else:
        raise ValueError("unknown slot kind")
    return {"messages": [{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(content, ensure_ascii=False, sort_keys=True,
                                                                  separators=(",", ":"))}]}


def _certainty(candidates: Sequence[int]) -> str:
    return "unique" if len(candidates) == 1 else "ambiguous" if candidates else "inconsistent"


def score_response(case: Mapping[str, Any], slot: Mapping[str, Any],
                   response: Mapping[str, Any], responses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Score external correctness/certainty; this does not prove internal reasoning."""
    checked = parse_response(slot["kind"], response)
    result = dict(slot_id=slot["slot_id"], case_id=case["case_id"], kind=slot["kind"],
                  sender=slot.get("sender"), receiver=slot.get("receiver"), condition=slot.get("condition"),
                  rep=slot.get("rep"), response_status=response.get("status"),
                  response_available=checked["available"], format_valid=checked["format_valid"],
                  validation_reason=checked["reason"], exact_choice_category=None)
    if slot["kind"] == "message":
        result.update(message=checked["parsed"]["message"] if checked["format_valid"] else None,
                      message_semantics_verified=False)
        return result
    receiver = slot["receiver"]
    own = list(case["oracle"]["own_candidates"][receiver])
    joint = list(case["oracle"]["joint_candidates"])
    result.update(own_information_certainty=_certainty(own), joint_information_certainty=_certainty(joint),
                  own_candidate_count=len(own), joint_candidate_count=len(joint),
                  available_information_certainty=("unverified_free_text" if incoming_visible(slot["condition"], receiver)
                                                   else _certainty(joint) if slot["condition"] == BASELINE else _certainty(own)),
                  available_information_certifies_choice=None,
                  correct=None, abstained=None, digit=None, own_candidate_consistent=None,
                  joint_candidate_consistent=None, own_information_certifies_choice=None,
                  joint_information_certifies_choice=None)
    if checked["available"] and checked["format_valid"] is False:
        result["exact_choice_category"] = "INVALID"
        return result
    if checked["format_valid"] is not True:
        return result
    digit = checked["parsed"]["digit"]
    row, column = case["target"]
    truth = case["solution"][row - 1][column - 1]
    result.update(digit=digit, exact_choice_category="ABSTAIN" if digit is None else str(digit),
                  correct=digit == truth, abstained=digit is None,
                  own_candidate_consistent=digit in own if digit is not None else None,
                  joint_candidate_consistent=digit in joint if digit is not None else None,
                  own_information_certifies_choice=len(own) == 1 and digit == own[0],
                  joint_information_certifies_choice=len(joint) == 1 and digit == joint[0])
    if slot["condition"] == BASELINE:
        result["available_information_certifies_choice"] = result["joint_information_certifies_choice"]
    elif not incoming_visible(slot["condition"], receiver):
        result["available_information_certifies_choice"] = result["own_information_certifies_choice"]
    return result
