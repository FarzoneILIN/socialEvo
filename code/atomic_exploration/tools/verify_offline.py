"""Verify the portable core with synthetic data and local-only networking."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
for name in list(os.environ):
    if name.startswith("SOCIALEVO_"):
        os.environ.pop(name)


def local_connect(original):
    def connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise RuntimeError("offline_verification_blocks_external_network")
        return original(sock, address)
    return connect


def main():
    try:
        import numpy
    except ModuleNotFoundError:
        print("完整核心检查需要 NumPy；最小 atomic_experiment 可单独使用标准库运行。")
        return 2
    manifest = json.loads((ROOT / "COPY_MANIFEST.json").read_text(encoding="utf-8"))
    result = {
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
        "scope": "copied source integrity, portable core tests and scripted smoke execution",
        "real_model_calls": 0,
        "historical_batch_reconstruction": "not_verified; source data intentionally excluded",
        "optional_menu_negotiation_kitchen_backends": "not_verified; external data/dependencies excluded",
        "copy_mismatches": [],
        "syntax_errors": [],
        "import_origins": {},
    }
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            result["copy_mismatches"].append(entry["path"])
        if path.suffix == ".py" and path.is_file():
            try:
                compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            except (SyntaxError, UnicodeError) as exc:
                result["syntax_errors"].append({"path": entry["path"], "type": type(exc).__name__})

    for name in ("atomic_experiment.runner", "atomic_relation_followup.runner",
                 "atomic_relation_followup.runtime", "atomic_relation_followup.analysis",
                 "atomic_relation_followup.minesweeper"):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(ROOT):
            raise RuntimeError("module_loaded_outside_export")
        result["import_origins"][name] = origin.relative_to(ROOT).as_posix()

    suite = unittest.TestSuite()
    for directory in ("atomic_experiment/tests", "atomic_relation_followup/tests"):
        suite.addTests(unittest.TestLoader().discover(str(ROOT / directory), pattern="test_*.py"))
    stream = io.StringIO()
    with patch.object(socket.socket, "connect", new=local_connect(socket.socket.connect)):
        executed = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
        result["tests"] = {
            "run": executed.testsRun,
            "failures": len(executed.failures),
            "errors": len(executed.errors),
            "skipped": len(executed.skipped),
            "passed": executed.wasSuccessful(),
        }
        if not executed.wasSuccessful():
            result["test_output"] = stream.getvalue()
        from atomic_experiment.client import MockClient
        from atomic_experiment.config import Config
        from atomic_experiment.runner import run
        with tempfile.TemporaryDirectory(prefix="socialevo-core-smoke-") as temporary:
            run_directory, aborted = run(Config(backend="mock", instances=1, repeats=1),
                                         MockClient(), Path(temporary))
            summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
            result["scripted_smoke"] = {
                "backend": summary["backend"],
                "aborted": aborted,
                "conditions": sorted(summary["groups"]),
                "paired_comparisons": len(summary["paired_differences"]),
                "events_present": (run_directory / "events.jsonl").is_file(),
                "manifest_present": (run_directory / "manifest.json").is_file(),
            }

    package_names = {Path(entry["path"]).parts[0] for entry in manifest["files"]}
    outside = []
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in package_names or not getattr(module, "__file__", None):
            continue
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            outside.append(name)
    result["local_modules_loaded_outside_export"] = outside
    result["ok"] = (not result["copy_mismatches"] and not result["syntax_errors"] and
                    executed.wasSuccessful() and not aborted and not outside and
                    result["scripted_smoke"]["events_present"] and
                    result["scripted_smoke"]["paired_comparisons"] == 1)
    (ROOT / "VERIFICATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
