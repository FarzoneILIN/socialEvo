"""Offline, allowlisted extraction of pinned public Menu/DND data to JSON.

The paid runner reads snapshots.json only. This script never invokes a legacy
game method, never creates model clients, and rejects every other pickle global.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


class InertRecord:
    pass


class RestrictedPublicDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {("create_game_utils", "GeneralMenuDesignGame"), ("persona", "Persona")}:
            return InertRecord
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            return getattr(np._core.multiarray, name)
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        raise pickle.UnpicklingError(f"Unapproved data global {module}.{name}")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def extract():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    records = {Path(x["local_source_file"]): x for x in manifest}
    samples = {"menu": [], "dnd": []}
    fields = ("ingredients", "dishes", "recipes", "game_ingredients_state", "agent_0_obs", "agent_1_obs",
              "agent_0_values", "agent_1_values", "best_menu", "max_reward")
    for path in sorted(ROOT.rglob("game_*.pkl")):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == records[path]["sha256"]
        game = RestrictedPublicDataUnpickler(io.BytesIO(data)).load()
        record = {field: plain(getattr(game, field)) for field in fields}
        record.update(upstream_game_id=path.stem, provenance=records[path].copy())
        samples["menu"].append(record)
    assert len(samples["menu"]) == 10
    path = ROOT / "dnd" / "src" / "data" / "negotiate" / "selfplay.txt"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == records[path]["sha256"]
    lines = data.decode("utf-8").splitlines()
    pairs = []
    for offset in range(0, len(lines), 2):
        a, b = [int(x) for x in lines[offset].split()], [int(x) for x in lines[offset + 1].split()]
        assert a[::2] == b[::2]
        assert sum(c * v for c, v in zip(a[::2], a[1::2])) == 10
        assert sum(c * v for c, v in zip(b[::2], b[1::2])) == 10
        pairs.append((offset // 2, a, b))
    # Five first-appearing inventory configurations, two distinct sender profiles
    # per configuration, using file order. Exclude both exploratory pairs 0/1.
    inventories = []
    for pair_id, a, b in pairs:
        counts = tuple(a[::2])
        if counts not in inventories:
            inventories.append(counts)
        if len(inventories) == 5:
            break
    for counts in inventories:
        sender_profiles = set()
        for pair_id, a, b in pairs:
            if pair_id in (0, 1) or tuple(a[::2]) != counts or tuple(a[1::2]) in sender_profiles:
                continue
            sender_profiles.add(tuple(a[1::2]))
            samples["dnd"].append({
                "upstream_pair_id": pair_id, "counts": a[::2], "sender_values": a[1::2],
                "receiver_values": b[1::2],
                "provenance": {**records[path], "source_lines": [2 * pair_id + 1, 2 * pair_id + 2]},
            })
            if len(sender_profiles) == 2:
                break
        assert len(sender_profiles) == 2
    assert len(samples["dnd"]) == 10
    (ROOT / "snapshots.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"menu": len(samples["menu"]), "dnd_pair_ids": [x["upstream_pair_id"] for x in samples["dnd"]]}))


if __name__ == "__main__":
    extract()
