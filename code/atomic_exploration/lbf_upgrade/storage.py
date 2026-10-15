"""Coordinator-owned durable request accounting for the LBF upgrade.

Workers receive the returned reservation, perform only the network operation,
and send its outcome back to the coordinator.  They never share this connection.
Every possible billed attempt is committed before it can be dispatched.
"""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


RETRY_CODES = frozenset({"http_429", "http_500", "http_502", "http_503", "http_504"})
RETRY_DELAYS = (2, 5)
SCHEMA_VERSION = 1


class Stop(RuntimeError):
    """An adapter-owned reason that is safe to record or display."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, immutable=False):
    """Write UTF-8 JSON atomically; immutable writes cannot replace old content."""
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if read(path) != value:
            raise Stop("immutable_artifact_mismatch")
        return
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=path.name + ".", suffix=".pending",
                                         dir=path.parent, delete=False) as file:
            name = file.name
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        if immutable:
            try:
                # A hard link creates the target atomically without overwriting it.
                os.link(name, path)
            except FileExistsError:
                if read(path) != value:
                    raise Stop("immutable_artifact_mismatch") from None
        else:
            os.replace(name, path)
            name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def _integer(value):
    return type(value) is int and value >= 0


def _detail(usage, group, key):
    """Return optional detail, distinguishing absent detail from invalid detail."""
    details = usage.get(group)
    if details is None:
        return None
    if not isinstance(details, dict):
        return "invalid"
    return details.get(key)


def usage_valid(usage):
    if not isinstance(usage, dict):
        return False
    if not all(_integer(usage.get(k)) for k in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
        return False
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        return False
    cached = _detail(usage, "prompt_tokens_details", "cached_tokens")
    legacy_cached = usage.get("cached_tokens")
    for value in (cached, legacy_cached):
        if value is not None and (not _integer(value) or value > usage["prompt_tokens"]):
            return False
    if cached is not None and legacy_cached is not None and cached != legacy_cached:
        return False
    reasoning = _detail(usage, "completion_tokens_details", "reasoning_tokens")
    if reasoning is not None and (not _integer(reasoning) or
                                  reasoning > usage["completion_tokens"]):
        return False
    return True


def _secret_forms(secrets):
    if isinstance(secrets, str):
        secrets = (secrets,)
    values = set()
    for secret in secrets or ():
        if not isinstance(secret, str) or not secret:
            continue
        values.update((secret, urllib.parse.quote(secret, safe=""),
                       urllib.parse.quote_plus(secret, safe=""),
                       json.dumps(secret, ensure_ascii=False)[1:-1],
                       json.dumps(secret, ensure_ascii=True)[1:-1]))
    return tuple(sorted(values, key=len, reverse=True))


def redact(value, secrets):
    """Recursively redact provided secret forms, including dictionary keys."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {redact(k, secrets): redact(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, secrets) for v in value]
    return value


def safe_error_code(error):
    """Keep adapter codes only. Provider text, URLs and stack traces are discarded."""
    if isinstance(error, str):
        code = error
    else:
        code = getattr(error, "error_code", None)
    if isinstance(code, str) and (re.fullmatch(r"http_[1-5][0-9]{2}", code) or code in {
            "transport_error_billing_unknown", "invalid_response_billing_unknown", "unknown_error"}):
        return code
    return "unknown_error"


class Ledger:
    """A single-coordinator ledger, also safe against competing reservations.

    A completed response is obtained through ``lookup``. ``reserve`` never
    dispatches, sleeps or reads credentials. ``finish`` retains every completed
    outcome before reporting a stop condition. The scheduler owns retry delays.
    """

    def __init__(self, root, design, backend="api", readonly=False, *, secrets=(), clock=time.time):
        self.root = Path(root)
        self.readonly = readonly
        self.backend = backend
        self.clock = clock
        self.secrets = _secret_forms(secrets)
        if backend not in {"api", "mock"}:
            raise Stop("unsupported_backend")
        if not isinstance(design, dict):
            raise Stop("design_invalid")
        try:
            self.design = json.loads(_json(design))
        except (TypeError, ValueError):
            raise Stop("design_invalid") from None
        self.historical = self.design.get("historical_actual_requests", 0)
        self.global_ceiling = self.design.get("actual_request_ceiling")
        self.new_ceiling = self.design.get("new_actual_request_ceiling")
        if not all(_integer(v) for v in (self.historical, self.global_ceiling, self.new_ceiling)):
            raise Stop("design_budget_invalid")
        self.waves = {}
        for wave in self.design.get("waves", []):
            if (not isinstance(wave, dict) or not isinstance(wave.get("wave_id"), str) or
                    not wave["wave_id"] or "/" in wave["wave_id"] or
                    not _integer(wave.get("logical_upper_bound")) or
                    not _integer(wave.get("extra_attempt_limit")) or wave["wave_id"] in self.waves):
                raise Stop("design_wave_invalid")
            self.waves[wave["wave_id"]] = wave
        if not self.waves:
            raise Stop("design_wave_invalid")
        path = self.root / "requests.sqlite3"
        if readonly:
            self.db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS calls(
                    slot TEXT PRIMARY KEY, wave TEXT NOT NULL, unit_id TEXT NOT NULL,
                    input_json TEXT NOT NULL, input_hash TEXT NOT NULL,
                    context_json TEXT NOT NULL, cap INTEGER NOT NULL,
                    status TEXT NOT NULL, response_json TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS attempts(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, slot TEXT NOT NULL REFERENCES calls(slot),
                    wave TEXT NOT NULL, attempt INTEGER NOT NULL, api INTEGER NOT NULL,
                    state TEXT NOT NULL, code TEXT, started_utc TEXT NOT NULL, finished_utc TEXT,
                    UNIQUE(slot, attempt));
                CREATE INDEX IF NOT EXISTS attempts_wave ON attempts(wave, attempt);
            """)
        self.db.row_factory = sqlite3.Row
        expected = _json({"schema_version": SCHEMA_VERSION, "design": self.design, "backend": backend})
        row = self.db.execute("SELECT value FROM metadata WHERE key='frozen_design'").fetchone()
        if row is None:
            if readonly:
                self.close()
                raise Stop("ledger_metadata_missing")
            with self.db:
                self.db.execute("INSERT INTO metadata VALUES('frozen_design',?)", (expected,))
        elif row["value"] != expected:
            self.close()
            raise Stop("ledger_design_mismatch")

    def close(self):
        self.db.close()

    def _spec(self, spec):
        if not isinstance(spec, Mapping):
            raise Stop("request_spec_invalid")
        required = ("slot", "wave", "unit_id", "messages", "context", "cap")
        if any(k not in spec for k in required):
            raise Stop("request_spec_invalid")
        value = {k: spec[k] for k in required}
        if (value["wave"] not in self.waves or not isinstance(value["slot"], str) or
                not value["slot"].startswith(value["wave"] + "/") or
                value["slot"] == value["wave"] + "/" or
                not isinstance(value["unit_id"], str) or not value["unit_id"] or
                type(value["cap"]) is not int or value["cap"] < 1 or
                not isinstance(value["messages"], list) or not isinstance(value["context"], dict)):
            raise Stop("request_spec_invalid")
        if redact(value, self.secrets) != value:
            raise Stop("secret_in_model_input")
        try:
            return json.loads(_json(value))
        except (TypeError, ValueError):
            raise Stop("request_spec_invalid") from None

    @staticmethod
    def _matches(row, spec):
        if (row["wave"] != spec["wave"] or row["unit_id"] != spec["unit_id"] or
                row["cap"] != spec["cap"] or row["input_hash"] != _digest(spec) or
                json.loads(row["input_json"]) != spec["messages"] or
                json.loads(row["context_json"]) != spec["context"]):
            raise Stop("cached_input_mismatch")

    @staticmethod
    def _check_result(result):
        if result.get("secret_redacted"):
            raise Stop("credential_echo_redacted")
        if result.get("category") == "reply" and result.get("actual_api_request") and not usage_valid(result.get("usage")):
            raise Stop("successful_response_usage_incomplete")
        return result

    def lookup(self, spec):
        spec = self._spec(spec)
        row = self.db.execute("SELECT * FROM calls WHERE slot=?", (spec["slot"],)).fetchone()
        if row is None:
            return None
        self._matches(row, spec)
        if row["response_json"] is None:
            return None
        return self._check_result(json.loads(row["response_json"]))

    def _failure(self, slot, code):
        result = {"category": "failure", "content": "", "usage": None,
                  "actual_model": "unavailable", "finish_reason": None,
                  "error": safe_error_code(code), "actual_api_request": self.backend == "api",
                  "secret_redacted": False}
        self.db.execute("UPDATE calls SET status='failure', response_json=?, error=? WHERE slot=?",
                        (_json(result), result["error"], slot))
        return result

    def _retry_left(self, wave):
        used = self.db.execute("SELECT COUNT(*) FROM attempts WHERE wave=? AND attempt>1 AND api=1",
                               (wave,)).fetchone()[0]
        return self.backend != "api" or used < self.waves[wave]["extra_attempt_limit"]

    def retry_ready_at(self, spec):
        """Durable retry deadline in UNIX seconds; the scheduler owns waiting."""
        spec = self._spec(spec)
        call = self.db.execute("SELECT * FROM calls WHERE slot=?", (spec["slot"],)).fetchone()
        if call is None:
            return 0
        self._matches(call, spec)
        if call["response_json"] is not None:
            return 0
        row = self.db.execute("SELECT * FROM attempts WHERE slot=? ORDER BY attempt DESC LIMIT 1",
                              (spec["slot"],)).fetchone()
        if row is None or row["state"] != "error" or row["code"] not in RETRY_CODES or row["attempt"] >= 3:
            return 0
        finished = datetime.fromisoformat(row["finished_utc"].replace("Z", "+00:00")).timestamp()
        return finished + RETRY_DELAYS[row["attempt"] - 1]

    def audit_rows(self):
        """Return independent snapshots; callers never access the connection."""
        return {"calls": [dict(r) for r in self.db.execute("SELECT * FROM calls ORDER BY slot")],
                "attempts": [dict(r) for r in self.db.execute("SELECT * FROM attempts ORDER BY seq")]}

    def reserve(self, spec, *, retry_subceiling=None, actual_subceiling=None):
        """Reserve within frozen budgets and any stricter current-tranche caps.

        Subceilings are cumulative limits, never additional allowances. Removing
        them after checkpoint approval exposes only the frozen remaining budget.
        """
        if self.readonly:
            raise Stop("readonly_ledger")
        spec = self._spec(spec)
        if (retry_subceiling is not None and
                (not _integer(retry_subceiling) or
                 retry_subceiling > self.waves[spec["wave"]]["extra_attempt_limit"])):
            raise Stop("tranche_budget_invalid")
        if (actual_subceiling is not None and
                (not _integer(actual_subceiling) or
                 actual_subceiling > self.new_ceiling or
                 actual_subceiling > max(0, self.global_ceiling - self.historical))):
            raise Stop("tranche_budget_invalid")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM calls WHERE slot=?", (spec["slot"],)).fetchone()
            if row is not None:
                self._matches(row, spec)
                if row["response_json"] is not None:
                    raise Stop("completed_slot_must_be_replayed")
                previous = list(self.db.execute("SELECT * FROM attempts WHERE slot=? ORDER BY attempt",
                                                (spec["slot"],)))
                if any(a["state"] == "inflight" for a in previous):
                    raise Stop("inflight_request_billing_unknown")
            else:
                previous = []
                count = self.db.execute("SELECT COUNT(*) FROM calls WHERE wave=?", (spec["wave"],)).fetchone()[0]
                if count >= self.waves[spec["wave"]]["logical_upper_bound"]:
                    raise Stop("wave_logical_ceiling")
            attempt = len(previous) + 1
            if previous:
                if previous[-1]["code"] not in RETRY_CODES or attempt > 3:
                    self._failure(spec["slot"], previous[-1]["code"])
                    self.db.commit()
                    raise Stop("recorded_failure_not_retryable")
                if not self._retry_left(spec["wave"]):
                    self._failure(spec["slot"], previous[-1]["code"])
                    self.db.commit()
                    raise Stop("wave_retry_ceiling")
                retries = self.db.execute("SELECT COUNT(*) FROM attempts WHERE wave=? AND attempt>1 AND api=1",
                                           (spec["wave"],)).fetchone()[0]
                if self.backend == "api" and retry_subceiling is not None and retries >= retry_subceiling:
                    self._failure(spec["slot"], previous[-1]["code"])
                    self.db.commit()
                    raise Stop("tranche_retry_ceiling")
                finished = datetime.fromisoformat(previous[-1]["finished_utc"].replace("Z", "+00:00")).timestamp()
                if self.clock() < finished + RETRY_DELAYS[attempt - 2]:
                    raise Stop("retry_before_delay")
            api = int(self.backend == "api")
            actual = self.db.execute("SELECT COALESCE(SUM(api),0) FROM attempts").fetchone()[0]
            if api and self.historical + actual >= self.global_ceiling:
                raise Stop("actual_request_ceiling")
            if api and actual >= self.new_ceiling:
                raise Stop("new_actual_request_ceiling")
            if api and actual_subceiling is not None and actual >= actual_subceiling:
                raise Stop("tranche_actual_ceiling")
            if row is None:
                self.db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,?,'pending',NULL,NULL)",
                                (spec["slot"], spec["wave"], spec["unit_id"], _json(spec["messages"]),
                                 _digest(spec), _json(spec["context"]), spec["cap"]))
            started = _utc(self.clock())
            cursor = self.db.execute("INSERT INTO attempts(slot,wave,attempt,api,state,started_utc) VALUES(?,?,?,?,'inflight',?)",
                                     (spec["slot"], spec["wave"], attempt, api, started))
            seq = cursor.lastrowid
            self.db.commit()
            return {"seq": seq, "slot": spec["slot"], "wave": spec["wave"], "attempt": attempt,
                    "unit_id": spec["unit_id"], "messages": spec["messages"], "context": spec["context"],
                    "cap": spec["cap"], "started_utc": started}
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise

    @staticmethod
    def _reply_payload(reply, actual):
        def get(name, default=None):
            return reply.get(name, default) if isinstance(reply, Mapping) else getattr(reply, name, default)
        content = get("content", "")
        model = get("actual_model", get("model", "unavailable"))
        finish = get("finish_reason")
        usage = get("usage")
        # A malformed usage object must not prevent durable response preservation.
        try:
            usage = json.loads(_json(usage))
        except (TypeError, ValueError):
            usage = None
        return {"category": "reply", "content": content if isinstance(content, str) else "",
                "usage": usage, "actual_model": model if isinstance(model, str) else "unavailable",
                "finish_reason": finish if isinstance(finish, str) else None,
                "actual_api_request": actual, "reply_shape_invalid": not isinstance(content, str)}

    def finish(self, attempt, *, reply=None, error=None):
        if self.readonly:
            raise Stop("readonly_ledger")
        if (reply is None) == (error is None):
            raise Stop("outcome_invalid")
        if not isinstance(attempt, Mapping) or type(attempt.get("seq")) is not int:
            raise Stop("attempt_invalid")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM attempts WHERE seq=?", (attempt["seq"],)).fetchone()
            if row is None or any(row[k] != attempt.get(k) for k in ("slot", "wave", "attempt", "started_utc")):
                raise Stop("attempt_invalid")
            if row["state"] != "inflight":
                raise Stop("attempt_already_finished")
            result, stop_reason, retryable, delay = None, None, False, None
            if error is not None:
                code = safe_error_code(error)
                self.db.execute("UPDATE attempts SET state='error',code=?,finished_utc=? WHERE seq=?",
                                (code, _utc(self.clock()), row["seq"]))
                eligible = code in RETRY_CODES and row["attempt"] < 3
                retryable = eligible and self._retry_left(row["wave"])
                if retryable:
                    delay = RETRY_DELAYS[row["attempt"] - 1]
                else:
                    result = self._failure(row["slot"], code)
                    if eligible:
                        stop_reason = "wave_retry_ceiling"
            else:
                saved = self._reply_payload(reply, bool(row["api"]))
                result = redact(saved, self.secrets)
                result["secret_redacted"] = result != saved
                self.db.execute("UPDATE attempts SET state='reply',finished_utc=? WHERE seq=?", (_utc(self.clock()), row["seq"]))
                self.db.execute("UPDATE calls SET status='reply',response_json=? WHERE slot=?", (_json(result), row["slot"]))
            self.db.commit()
        except BaseException:
            if self.db.in_transaction:
                self.db.rollback()
            raise
        if result is not None:
            self._check_result(result)
        return {"retryable": retryable, "retry_delay": delay, "result": result, "stop_reason": stop_reason}

    def accounting(self, wave=None):
        if wave is not None and wave not in self.waves:
            raise Stop("unknown_wave")
        where, args = (" WHERE wave=?", (wave,)) if wave is not None else ("", ())
        calls = list(self.db.execute("SELECT * FROM calls" + where, args))
        attempts = list(self.db.execute("SELECT * FROM attempts" + where, args))
        replies = [json.loads(c["response_json"]) for c in calls if c["status"] == "reply" and c["response_json"]]
        valid = [r for r in replies if r.get("actual_api_request") and usage_valid(r.get("usage"))]
        tokens = {k: 0 for k in ("prompt_tokens", "cached_prompt_tokens", "uncached_prompt_tokens",
                                 "completion_tokens", "total_tokens", "reasoning_tokens")}
        for reply in valid:
            usage = reply["usage"]
            cached = _detail(usage, "prompt_tokens_details", "cached_tokens")
            if cached is None:
                cached = usage.get("cached_tokens", 0) or 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                tokens[key] += usage[key]
            tokens["cached_prompt_tokens"] += cached
            tokens["uncached_prompt_tokens"] += usage["prompt_tokens"] - cached
            tokens["reasoning_tokens"] += _detail(usage, "completion_tokens_details", "reasoning_tokens") or 0
        unknown = sum(a["api"] for a in attempts if a["state"] != "reply")
        unknown += sum(bool(r.get("actual_api_request")) and not usage_valid(r.get("usage")) for r in replies)
        return {"logical_slots_started": len(calls), "responses_saved": len(replies),
                "completed_failures": sum(c["status"] == "failure" for c in calls),
                "api_requests": sum(a["api"] for a in attempts),
                "additional_retry_requests": sum(a["api"] for a in attempts if a["attempt"] > 1),
                "known_reported_tokens": tokens["total_tokens"], "unknown_billing_attempts": unknown,
                "unfinished_requests": sum(a["state"] == "inflight" for a in attempts),
                "valid_usage_successes": len(valid), "failed_attempts": sum(a["state"] == "error" for a in attempts),
                "usage_complete": unknown == 0,
                "length_finish_count": sum(r.get("finish_reason") == "length" for r in replies),
                "empty_output_count": sum(not r.get("content", "").strip() for r in replies),
                "usage_token_breakdown": tokens, **tokens}
