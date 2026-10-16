"""Exact menu score and optimum calculation."""
from __future__ import annotations
import itertools


def _menu_score_data(state, recipes, values, selection):
    used = [sum(recipes[d][j] for d in selection) for j in range(len(state))]
    feasible = all(a <= b + 1e-9 for a, b in zip(used, state))
    return feasible, sum(values[d] for d in selection) if feasible else 0.0


def _menu_optimum(state, recipes, values):
    possible = [d for d, recipe in enumerate(recipes) if all(a <= b + 1e-9 for a, b in zip(recipe, state))]
    # Exhaustive enumeration of every potentially feasible dish subset is exact.
    best_reward, best_menu = 0.0, []
    for flags in itertools.product((0, 1), repeat=len(possible)):
        selection = [d for d, flag in zip(possible, flags) if flag]
        feasible, reward = _menu_score_data(state, recipes, values, selection)
        if feasible and reward > best_reward + 1e-9:
            best_reward, best_menu = reward, selection
    return best_reward, best_menu
