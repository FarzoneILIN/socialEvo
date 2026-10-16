"""JSON helpers preserve the historical serialization and input hash format."""
import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def stamp():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class BudgetStop(RuntimeError):
    pass


def usage_valid(usage):
    keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
    return (isinstance(usage, dict) and
            all(type(usage.get(k)) is int and usage[k] >= 0 for k in keys) and
            usage['prompt_tokens'] + usage['completion_tokens'] == usage['total_tokens'])


def redact(value, secrets):
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, '[REDACTED]')
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    return value
