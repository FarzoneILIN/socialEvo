"""Twelve-round, integer-allocated renewable commons task."""

from task_experiments.protocol import Validation
from .base import TaskEnvironment


class CommonsEnvironment(TaskEnvironment):
    task_id = "commons_v1"
    horizon = 12

    @staticmethod
    def build_initial_state(n, order):
        ids = [f"A{i}" for i in range(n)]
        state = {"round_completed": 0, "stock": 100, "capacity": 100,
                 "cumulative_harvest": {aid: 0 for aid in ids}, "stock_history": [100],
                 "harvest_history": [], "regrowth_history": [], "overdemand_history": [],
                 "done": False, "termination_reason": None}
        return state, {"initial_stock": 100, "capacity": 100, "regrowth_multiplier": 2,
                       "horizon": 12, "allocation": "largest_remainder_rotating_priority"}

    def _validate_state(self, state):
        super()._validate_state(state)
        ids = set(self.agent_ids)
        stock = state.get("stock")
        cap = state.get("capacity")
        if (type(stock) is not int or type(cap) is not int or cap != 100
                or not 0 <= stock <= cap
                or set(state.get("cumulative_harvest", {})) != ids
                or any(type(x) is not int or x < 0 for x in state["cumulative_harvest"].values())
                or any(not isinstance(state.get(name), list)
                       or len(state[name]) != state["round_completed"] + (name == "stock_history")
                       for name in ("stock_history", "harvest_history", "regrowth_history", "overdemand_history"))
                or state["stock_history"][-1] != stock
                or any(type(x) is not int or x < 0 for name in
                       ("stock_history", "harvest_history", "regrowth_history", "overdemand_history")
                       for x in state[name])
                or state["stock_history"][0] != self.instance["initial_state"]["stock"]
                or state["stock_history"][0] + sum(state["regrowth_history"])
                   != sum(state["cumulative_harvest"].values()) + stock
                or sum(state["harvest_history"]) != sum(state["cumulative_harvest"].values())
                or (state["done"] and state["termination_reason"] not in ("horizon", "collapse"))
                or (state["termination_reason"] == "collapse" and stock != 0)
                or (state["termination_reason"] == "horizon" and state["round_completed"] != self.horizon)
                or (state["round_completed"] < self.horizon and stock == 0 and not state["done"])):
            raise ValueError("invalid_commons_state")

    def priority(self):
        offset = self.round_completed % len(self.agent_ids)
        return list(self.agent_order[offset:] + self.agent_order[:offset])

    def _public_state(self):
        return {"round_completed": self.round_completed, "stock": self._state["stock"],
                "capacity": 100, "cumulative_harvest": self._state["cumulative_harvest"],
                "priority": self.priority() if not self.done else None,
                "stock_history": self._state["stock_history"],
                "regrowth_rule": "min(100, 2 * (stock - allocated_total))",
                "done": self.done, "termination_reason": self.termination_reason}

    def _validate_action(self, agent_id, action):
        if not isinstance(action, dict) or set(action) != {"type", "amount"}:
            return Validation(False, "invalid_fields")
        amount = action["amount"]
        if action["type"] != "harvest" or type(amount) is not int or not 0 <= amount <= self._state["stock"]:
            return Validation(False, "invalid_amount")
        return Validation(True, "accepted", {"type": "harvest", "amount": amount})

    def _fallback_action(self, agent_id):
        return {"type": "harvest", "amount": 0}

    def _step(self, actions, draft):
        ids = self.agent_ids
        s = draft["stock"]
        requests = {aid: actions[aid]["amount"] for aid in ids}
        q = sum(requests.values())
        priority = self.priority()
        remainders = None
        if q == 0 or q <= s:
            allocated = dict(requests)
        else:
            bases = {aid: divmod(s * requests[aid], q) for aid in ids}
            allocated = {aid: bases[aid][0] for aid in ids}
            remainders = {aid: bases[aid][1] for aid in ids}
            left = s - sum(allocated.values())
            ranks = sorted(ids, key=lambda aid: (-remainders[aid], priority.index(aid)))
            for aid in ranks[:left]:
                allocated[aid] += 1
        total = sum(allocated.values())
        remaining = s - total
        next_stock = min(100, 2 * remaining)
        regrowth = next_stock - remaining
        for aid in ids:
            draft["cumulative_harvest"][aid] += allocated[aid]
        draft["stock"] = next_stock
        draft["round_completed"] += 1
        draft["stock_history"].append(next_stock)
        draft["harvest_history"].append(total)
        draft["regrowth_history"].append(regrowth)
        draft["overdemand_history"].append(max(0, q - s))
        if draft["round_completed"] == self.horizon:
            draft["done"] = True
            draft["termination_reason"] = "horizon"
        elif next_stock == 0:
            draft["done"] = True
            draft["termination_reason"] = "collapse"
        outcome = {"stock_before": s, "requests": requests, "request_total": q,
                   "allocations": allocated, "remainders": remainders, "priority": priority,
                   "remaining_before_regrowth": remaining, "regrowth_added": regrowth,
                   "stock_after": next_stock, "depleted": next_stock == 0,
                   "overdemand": max(0, q - s),
                   "harvest_fraction_of_stock": total / s if s else None}
        return allocated, outcome, {aid: "allocated" for aid in ids}

    def _score(self):
        state = self._state
        total = sum(state["cumulative_harvest"].values())
        values = list(state["cumulative_harvest"].values())
        return {"partial": not self.done, "rounds_completed": self.round_completed,
                "team_harvest": total if self.done else None,
                "diagnostic_team_harvest_prefix": total,
                "efficiency": total / 650 if self.done else None,
                "harvest_by_agent": dict(state["cumulative_harvest"]),
                "stock_history": list(state["stock_history"]),
                "survived_to_horizon": self.round_completed == self.horizon if self.done else None,
                "premature_collapse_round": (self.round_completed if self.termination_reason == "collapse" else None)
                                            if self.done else None,
                "final_depleted": state["stock"] == 0 if self.done else None,
                "final_stock": state["stock"] if self.done else None,
                "overdemand_history": list(state["overdemand_history"]),
                "harvest_fraction_history": [state["harvest_history"][i] / state["stock_history"][i]
                                             for i in range(self.round_completed)],
                "harvest_range": max(values) - min(values) if self.done else None}
