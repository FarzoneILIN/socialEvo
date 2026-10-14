import json
from datetime import datetime, timezone
from pathlib import Path


class Recorder:
    def __init__(self, directory: Path, secret=""):
        self.directory = directory
        self.secret = secret

    def sanitize(self, value):
        if isinstance(value, str):
            return value.replace(self.secret, "[REDACTED]") if self.secret else value
        if isinstance(value, dict):
            return {self.sanitize(k): self.sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.sanitize(x) for x in value]
        return value

    def event(self, episode, kind, **fields):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                  "episode": episode, "kind": kind, **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.sanitize(record), ensure_ascii=False) + "\n")

    def write(self, name, value):
        with (self.directory / name).open("x", encoding="utf-8") as f:
            json.dump(self.sanitize(value), f, ensure_ascii=False, indent=2)
