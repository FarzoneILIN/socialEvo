"""Allowlisted reading of pinned upstream public menu records."""
from __future__ import annotations
import pickle
import numpy as np


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
