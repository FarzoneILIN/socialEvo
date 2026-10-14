"""Append-only episode events and non-overwriting result files."""

import hashlib
import json
import math
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path

from atomic_experiment.recording import Recorder

from . import RULE_VERSION, SCHEMA_VERSION
from .prompts import PROMPT_VERSIONS, system_prompt
from .protocol import digest


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_run_directory(root):
    directory = Path(root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class EpisodeRecorder:
    def __init__(self, directory, episode_id, secret=""):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.episode_id = episode_id
        self.seq = 0
        self.old = Recorder(self.directory, secret)

    def event(self, kind, round, phase, **fields):
        self.seq += 1
        self.old.event(self.episode_id, kind, seq=self.seq, round=round, phase=phase,
                       **_json_safe(fields))

    def write(self, name, value):
        self.old.write(name, value)


def manifest(config, plan, client):
    root = Path(__file__).parent
    files = list(root.glob("*.py")) + list((root / "tasks").glob("*.py"))
    files += [Path(__file__).parent.parent / "atomic_experiment" / name
              for name in ("client.py", "recording.py")]
    hashes = {str(file.relative_to(root.parent)).replace("\\", "/"): hashlib.sha256(file.read_bytes()).hexdigest()
              for file in files}
    prompt = system_prompt(config.task_id, config.prompt_version)
    return {"schema_version": SCHEMA_VERSION, "rule_version": RULE_VERSION,
            "prompt_version": PROMPT_VERSIONS[config.prompt_version], "config": config.record(), "plan": plan,
            "backend": config.backend, "policy": config.resolved_policy() if config.backend == "mock" else None,
            "model": getattr(client, "model", None),
            "endpoint": getattr(client, "url", None), "python_version": platform.python_version(),
            "code_sha256": hashes, "system_prompt": prompt, "system_prompt_sha256": digest(prompt),
            "candidate_source": "notes/task_design/TASK_CANDIDATES_RESEARCH_2026-09-26.md",
            "implementation_specification": "notes/HANDOFF_TASK_IMPLEMENTATION_2026-09-26.md",
            "task_origin_note": "Local frozen adaptations: congestion route game; level-based foraging geometry and explicit load; renewable commons with pre-action talk and deterministic allocation. Not source-paper reproductions.",
            "created_utc": utc_now()}
