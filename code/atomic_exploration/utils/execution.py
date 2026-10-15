"""Shared process locking and unchanged historical usage-cost calculation."""
from __future__ import annotations
import math
import os
from pathlib import Path
import threading
from utils.io import BudgetStop
from utils.io import usage_valid


_ACTIVE_LOCKS = {}


_LOCK_REGISTRY_GUARD = threading.Lock()


def usage_cost(usage):
    """Keep historical prices; unavailable cache details use the full input price.

    Required token counters remain strictly validated. This calculation does not
    alter the raw usage saved with a reply or claim to be a provider invoice.
    """
    if not usage_valid(usage):
        return None
    priced_usage = dict(usage)
    if not isinstance(priced_usage.get('prompt_tokens_details'), dict):
        priced_usage['prompt_tokens_details'] = {}
    try:
        cost = _base_usage_cost(priced_usage)
    except (TypeError, ValueError, OverflowError):
        return None
    return cost if cost is not None and math.isfinite(cost) and cost >= 0 else None


class WorkerLock:
    """An OS lock must be held before recovery can mark stranded calls unknown."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / 'worker.lock'
        self.file = None

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with _LOCK_REGISTRY_GUARD:
            if self.root in _ACTIVE_LOCKS:
                raise BudgetStop('another_worker_active')
            self.file = self.path.open('a+b')
            if self.path.stat().st_size == 0:
                self.file.write(b'1')
                self.file.flush()
            self.file.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError):
                self.file.close()
                self.file = None
                raise BudgetStop('another_worker_active') from None
            _ACTIVE_LOCKS[self.root] = self
        return self

    def __exit__(self, *args):
        with _LOCK_REGISTRY_GUARD:
            if self.file is not None:
                try:
                    self.file.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                finally:
                    self.file.close()
                    self.file = None
                    _ACTIVE_LOCKS.pop(self.root, None)


def _lock_held(root):
    with _LOCK_REGISTRY_GUARD:
        return Path(root).resolve() in _ACTIVE_LOCKS


def _base_usage_cost(usage):
    if not usage_valid(usage):
        return None
    cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    if type(cached) is not int or not 0 <= cached <= usage['prompt_tokens']:
        cached = 0
    return ((usage['prompt_tokens'] - cached) * 2.5 + cached * .25
            + usage['completion_tokens'] * 15) / 1000000
