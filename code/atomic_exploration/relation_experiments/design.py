"""Fixed programme, deterministic ordering, and wave request bounds."""

import copy
import hashlib
from pathlib import Path
from task_experiments.tasks import HORIZONS
from task_experiments.protocol import digest
from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gpt-5.4-2026-03-05'
SEED = 42
ARMS = ('normal_A', 'normal_B', 'mask_first_to_second',
        'mask_second_to_first', 'mask_both')
CONDITIONS = ('normal', 'partner_hidden', 'no_generation')
GEOMETRIES = ('central_column', 'split_shoulders', 'near_left', 'near_right')
RETRY_CODES = ('http_429', 'http_500', 'http_502', 'http_503', 'http_504')
RETRY_DELAYS = (2, 5)


def code_hashes():
    paths = []
    for package in ('relation_experiments', 'task_experiments', 'atomic_experiment'):
        paths.extend(p for p in (ROOT / package).rglob('*.py')
                     if 'tests' not in p.parts and 'runs' not in p.parts)
    paths.extend(ROOT / 'tools' / name for name in
                 ('setup_socialevo_api.ps1', 'with_socialevo_api.ps1',
                  'socialevo_credentials.ps1', 'run_relation_experiments.ps1'))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.exists()}


def programme(backend='api'):
    if backend not in ('api', 'mock'):
        raise ValueError('invalid_backend')
    waves = []

    def add(name, kind, n, calls, **kwargs):
        waves.append(dict(wave_id=name, kind=kind, num_agents=n,
                          logical_upper_bound=calls, extra_attempt_limit=calls // 10,
                          **kwargs))

    add('calibration', 'calibration', 2, 121)
    add('two_agent_development', 'pair', 2, 768,
        geometries=list(GEOMETRIES[:2]), repeats=6, held_out=False)
    add('two_agent_validation', 'pair', 2, 1280,
        geometries=list(GEOMETRIES[2:]), repeats=10, held_out=True)
    for n in range(2, 9):
        if n > 2:
            add(f'lbf_edges_{n}', 'network', n, 48 * (n*n + 3*n),
                geometries=list(GEOMETRIES), repeats=6)
        add(f'lbf_full_{n}', 'full', n, n * 24 * 5 * 3,
            task_id='lbf_local_v1', repeats=3)
    for n in (2, 3):
        for task in ('congestion_v1', 'commons_v1'):
            add(f'{task}_full_{n}', 'full', n, n * HORIZONS[task] * 5 * 3,
                task_id=task, repeats=3)
    result = dict(protocol=PROTOCOL, backend=backend, requested_model=MODEL,
                  reasoning_effort='medium', temperature=None,
                  token_parameter='max_completion_tokens', output_caps=[4096, 8192],
                  timeout_seconds=120, seed=SEED, actual_request_ceiling=35000,
                  retry_http_codes=list(RETRY_CODES), retry_delays_seconds=list(RETRY_DELAYS),
                  max_retries_per_slot=2, min_valid_rate=0.95,
                  calibration_valid_threshold=59, calibration_denominator=60,
                  bootstrap_resamples=10000, bootstrap_seed=42, waves=waves,
                  logical_upper_bound=sum(w['logical_upper_bound'] for w in waves))
    assert result['logical_upper_bound'] == 30423
    return result


def rotated(items, index):
    items = list(items)
    k = index % len(items)
    return items[k:] + items[:k]


def blocks(wave):
    """Interleave repeats and identities; no result-dependent order."""
    return [dict(unit_id=f'{geometry}_shift{shift}_r{repeat}',
                 geometry=geometry, shift=shift, repeat=repeat)
            for repeat in range(wave['repeats'])
            for geometry in wave['geometries'] for shift in (0, 1)]


def freeze(wave, cap):
    result = copy.deepcopy(wave)
    result['selected_output_cap'] = cap
    result['units'] = (blocks(wave) if wave['kind'] in ('pair', 'network') else
                       [dict(unit_id=f'r{r}', repeat=r) for r in range(wave.get('repeats', 1))])
    result['design_sha256'] = digest(result)
    return result
