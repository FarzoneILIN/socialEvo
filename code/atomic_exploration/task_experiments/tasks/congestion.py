"""Six-round, locally adapted shared-route congestion task."""

from fractions import Fraction

from task_experiments.protocol import Validation
from .base import TaskEnvironment


class CongestionEnvironment(TaskEnvironment):
    task_id = "congestion_v1"
    horizon = 6

    @staticmethod
    def build_initial_state(n, order):
        ids = [f"A{i}" for i in range(n)]
        state = {"round_completed": 0, "cumulative_cost_units": {aid: 0 for aid in ids},
                 "last_routes": None, "last_k": None, "k_history": [],
                 "routes_history": [], "done": False, "termination_reason": None}
        return state, {"fast_cost_per_user": "k", "slow_cost": "5/2", "horizon": 6}

    def _validate_state(self, state):
        super()._validate_state(state)
        ids = set(self.agent_ids)
        if (set(state.get("cumulative_cost_units", {})) != ids
                or any(type(x) is not int or x < 0 for x in state["cumulative_cost_units"].values())
                or not isinstance(state.get("k_history"), list)
                or not isinstance(state.get("routes_history"), list)
                or len(state["k_history"]) != state["round_completed"]
                or len(state["routes_history"]) != state["round_completed"]
                or any(type(k) is not int or not 0 <= k <= len(ids) for k in state["k_history"])
                or any(set(routes) != ids or set(routes.values()) - {"F", "S"}
                       for routes in state["routes_history"])
                or state.get("last_routes") != (state["routes_history"][-1] if state["routes_history"] else None)
                or state.get("last_k") != (state["k_history"][-1] if state["k_history"] else None)
                or (state["done"] and state["termination_reason"] != "horizon")):
            raise ValueError("invalid_congestion_state")
        expected = {aid: sum(2 * k if routes[aid] == "F" else 5
                             for k, routes in zip(state["k_history"], state["routes_history"]))
                    for aid in self.agent_ids}
        if state["cumulative_cost_units"] != expected:
            raise ValueError("congestion_cost_ledger_mismatch")

    def _public_state(self):
        return {"round_completed": self.round_completed,
                "cumulative_cost": {aid: units / 2 for aid, units in self._state["cumulative_cost_units"].items()},
                "last_routes": self._state["last_routes"], "last_k": self._state["last_k"],
                "done": self.done, "termination_reason": self.termination_reason,
                "fast_cost_rule": "number_of_F_users", "slow_cost": 2.5}

    def _validate_action(self, agent_id, action):
        if not isinstance(action, dict) or set(action) != {"type", "route"}:
            return Validation(False, "invalid_fields")
        if action["type"] != "choose" or action["route"] not in ("F", "S"):
            return Validation(False, "invalid_route")
        return Validation(True, "accepted", {"type": "choose", "route": action["route"]})

    def _fallback_action(self, agent_id):
        return {"type": "choose", "route": "S"}

    def _step(self, actions, draft):
        routes = {aid: actions[aid]["route"] for aid in self.agent_ids}
        k = sum(route == "F" for route in routes.values())
        units = {aid: (2 * k if route == "F" else 5) for aid, route in routes.items()}
        for aid in self.agent_ids:
            draft["cumulative_cost_units"][aid] += units[aid]
        draft["round_completed"] += 1
        draft["last_routes"] = routes
        draft["last_k"] = k
        draft["routes_history"].append(routes)
        draft["k_history"].append(k)
        if draft["round_completed"] == self.horizon:
            draft["done"] = True
            draft["termination_reason"] = "horizon"
        outcome = {"routes": routes, "k": k, "cost_units_by_agent": units,
                   "cost_by_agent": {aid: units[aid] / 2 for aid in self.agent_ids},
                   "total_cost_units": sum(units.values()), "total_cost": sum(units.values()) / 2}
        rewards = {aid: Fraction(-units[aid], 2) for aid in self.agent_ids}
        return rewards, outcome, {aid: "route_executed" for aid in self.agent_ids}

    def _score(self):
        state = self._state
        costs = {aid: units / 2 for aid, units in state["cumulative_cost_units"].items()}
        total = sum(costs.values())
        n = len(self.agent_ids)
        return {"partial": not self.done, "rounds_completed": self.round_completed,
                "total_cost": total if self.done else None,
                "diagnostic_total_cost_prefix": total,
                "cost_by_agent": costs,
                "excess_cost": total - self.horizon * (2.5 * n - 1.5) if self.done else None,
                "k_history": list(state["k_history"]),
                "k_one_fraction": state["k_history"].count(1) / self.horizon if self.done else None,
                "k_two_fraction": state["k_history"].count(2) / self.horizon if self.done else None,
                "fast_counts": {aid: sum(routes[aid] == "F" for routes in state["routes_history"])
                                for aid in self.agent_ids},
                "cost_range": max(costs.values()) - min(costs.values()) if self.done else None}
