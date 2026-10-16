"""Shared protocol metadata, immutable inputs and adapter dispatch."""
from __future__ import annotations

import importlib
from pathlib import Path

from utils.io import canonical, digest, stamp, write, BudgetStop

ROOT = Path(__file__).resolve().parents[1]
MODULES = ('minesweeper_natural', 'menu', 'negotiation', 'kitchen', 'minesweeper_fact_advice')
CAPS = {
    'minesweeper_natural': (1128, 112, 'A'),
    'menu': (592, 59, 'A'),
    'negotiation': (924, 92, 'A'),
    'kitchen': (292, 29, 'A'),
    'minesweeper_fact_advice': (1200, 120, 'B'),
}
POLICY = dict(model='gpt-5.4-2026-03-05', reasoning_effort='medium',
              output_cap=8192, max_input_reference_tokens=24000,
              timeout_seconds=300, concurrency=8, minimum_concurrency=2,
              actual_request_cap=4548, unique_slot_cap=4136, retry_cap=412,
              retries_per_slot=2, retry_delays_seconds=[2, 5],
              reference_cny_cap=120, reference_cny_per_credit=.5,
              stage_reference_cny_caps={'A': 80, 'B': 40},
              dispatch_minutes=720, seed=42,
              minimum_format_valid_rate=.95, rate_check_min_responses=32,
              failure_distinct_slots_without_success=5,
              service_rolling_window=40, service_reduce_fraction=.20,
              no_outcome_resampling=True)


def adapter(module):
    name = ('minesweeper' if module.startswith('minesweeper') else
            'decisions' if module in ('menu', 'negotiation') else 'kitchen')
    return importlib.import_module('atomic_relation_followup.' + name)


def unpack(payload):
    cases = payload['cases']
    slots = payload['slots']
    return cases, slots


def wrapped_input(case, slot, responses):
    prepared = adapter(slot['module']).prepare_input(case, slot, responses)
    if not isinstance(prepared, dict) or not ('messages' in prepared or 'skip' in prepared):
        raise BudgetStop('invalid_adapter_input_contract')
    if 'messages' in prepared:
        messages = prepared['messages']
        if not messages or any(m.get('role') not in ('system', 'user', 'assistant') or
                               not isinstance(m.get('content'), str) for m in messages):
            raise BudgetStop('invalid_text_messages')
        bound = sum(len(m['content'].encode('utf-8')) + 256 for m in messages)
        if bound > POLICY['max_input_reference_tokens']:
            raise BudgetStop('input_bound_exceeded_before_dispatch')
    return prepared


def scored(case, slot, response, responses):
    value = adapter(slot['module']).score_response(case, slot, response, responses)
    if not isinstance(value, dict) or 'format_valid' not in value:
        raise BudgetStop('invalid_adapter_score_contract')
    value.update(slot_id=slot['slot_id'], case_id=case['case_id'], module=slot['module'],
                 kind=slot['kind'], response_status=response.get('status'),
                 condition=slot.get('condition'), analysis_group=slot.get('analysis_group'),
                 analysis_family=slot.get('analysis_family'), version=slot.get('version'),
                 sender=slot.get('sender'), receiver=slot.get('receiver'),
                 rep=slot.get('rep'))
    return value
