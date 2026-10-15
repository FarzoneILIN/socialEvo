"""Local level-based foraging adaptation with simultaneous movement and load."""

from fractions import Fraction

from task_experiments.protocol import Validation, exact, from_exact, reward_record
from .base import TaskEnvironment


DIRECTIONS = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
MOVE_FAILURES = ("out_of_bounds", "wall", "food_occupied", "agent_occupied", "target_conflict")


class LBFEnvironment(TaskEnvironment):
    task_id = "lbf_local_v1"
    horizon = 24

    @staticmethod
    def build_initial_state(n, order):
        width, height = 2 * n + 3, 7
        walls = [[x, y] for y in range(height) for x in range(width)
                 if x in (0, width - 1) or y in (0, height - 1)]
        agents = {aid: {"x": 2 * j + 2, "y": 1, "level": 1,
                        "cumulative_reward_exact": exact(0)}
                  for j, aid in enumerate(order)}
        foods = []
        for j in range(n):
            foods.extend(({"id": f"S_{j}", "x": 2 * j + 2, "y": 3,
                           "level": 1, "collected": False},
                          {"id": f"T_{j}", "x": 2 * j + 2, "y": 5,
                           "level": 2, "collected": False}))
        state = {"round_completed": 0, "width": width, "height": height,
                 "walls": walls, "agents": agents, "foods": foods, "R0": 3 * n,
                 "collected_level": 0, "move_success": 0,
                 "move_failures": {reason: 0 for reason in MOVE_FAILURES},
                 "wait_count": 0, "load_success": 0, "load_insufficient": 0,
                 "coalition_events": [], "done": False, "termination_reason": None}
        return state, {"width": width, "height": height, "R0": 3 * n,
                       "agent_level": 1, "food_levels": [1, 2], "horizon": 24}

    def _validate_state(self, state):
        super()._validate_state(state)
        n = len(self.agent_ids)
        width, height = 2 * n + 3, 7
        expected_walls = [[x, y] for y in range(height) for x in range(width)
                          if x in (0, width - 1) or y in (0, height - 1)]
        if (state.get("width") != width or state.get("height") != height
                or state.get("walls") != expected_walls or state.get("R0") != 3 * n
                or not isinstance(state.get("agents"), dict)
                or set(state["agents"]) != set(self.agent_ids)
                or not isinstance(state.get("foods"), list) or len(state["foods"]) != 2 * n):
            raise ValueError("invalid_lbf_geometry")
        expected_food = {(f"S_{j}", 2 * j + 2, 3, 1) for j in range(n)} | {
            (f"T_{j}", 2 * j + 2, 5, 2) for j in range(n)}
        food_rows = set()
        for food in state["foods"]:
            if (not isinstance(food, dict) or set(food) != {"id", "x", "y", "level", "collected"}
                    or type(food["x"]) is not int or type(food["y"]) is not int
                    or type(food["level"]) is not int or type(food["collected"]) is not bool):
                raise ValueError("invalid_lbf_food")
            food_rows.add((food["id"], food["x"], food["y"], food["level"]))
        if food_rows != expected_food or len(food_rows) != 2 * n:
            raise ValueError("invalid_lbf_food_template")
        occupied = set()
        active_food = {(f["x"], f["y"]) for f in state["foods"] if not f["collected"]}
        total_reward = Fraction(0)
        for aid in self.agent_ids:
            agent = state["agents"][aid]
            if (not isinstance(agent, dict)
                    or set(agent) != {"x", "y", "level", "cumulative_reward_exact"}
                    or type(agent["x"]) is not int or type(agent["y"]) is not int
                    or agent["level"] != 1
                    or not 0 < agent["x"] < width - 1 or not 0 < agent["y"] < height - 1):
                raise ValueError("invalid_lbf_agent")
            pos = (agent["x"], agent["y"])
            if pos in occupied or pos in active_food:
                raise ValueError("lbf_position_overlap")
            occupied.add(pos)
            reward = from_exact(agent["cumulative_reward_exact"])
            if reward < 0:
                raise ValueError("negative_lbf_reward")
            total_reward += reward
        collected = sum(f["level"] for f in state["foods"] if f["collected"])
        if (type(state.get("collected_level")) is not int
                or state["collected_level"] != collected
                or total_reward != Fraction(collected, 3 * n)
                or any(type(state.get(name)) is not int or state[name] < 0
                       for name in ("move_success", "wait_count", "load_success", "load_insufficient"))
                or not isinstance(state.get("move_failures"), dict)
                or set(state["move_failures"]) != set(MOVE_FAILURES)
                or any(type(x) is not int or x < 0 for x in state["move_failures"].values())
                or not isinstance(state.get("coalition_events"), list)
                or len(state["coalition_events"]) != sum(f["collected"] for f in state["foods"])
                or (state["done"] and state["termination_reason"] not in ("all_collected", "horizon"))
                or (state["termination_reason"] == "all_collected" and collected != 3 * n)
                or (state["termination_reason"] == "horizon" and state["round_completed"] != self.horizon)
                or (collected == 3 * n and not state["done"])):
            raise ValueError("invalid_lbf_ledger")

    def _public_state(self):
        state = self._state
        return {"round_completed": self.round_completed,
                "width": state["width"], "height": state["height"], "walls": state["walls"],
                "agents": {aid: {"x": a["x"], "y": a["y"], "level": a["level"],
                                  "cumulative_reward": float(from_exact(a["cumulative_reward_exact"]))}
                           for aid, a in state["agents"].items()},
                "foods": state["foods"], "R0": state["R0"],
                "done": self.done, "termination_reason": self.termination_reason}

    def _validate_action(self, agent_id, action):
        if not isinstance(action, dict) or "type" not in action:
            return Validation(False, "invalid_fields")
        kind = action["type"]
        if kind == "wait" and set(action) == {"type"}:
            return Validation(True, "accepted", {"type": "wait"})
        if kind == "move" and set(action) == {"type", "direction"} and action["direction"] in DIRECTIONS:
            return Validation(True, "accepted", {"type": "move", "direction": action["direction"]})
        if kind == "load" and set(action) == {"type", "food_id"} and isinstance(action["food_id"], str):
            food = next((f for f in self._state["foods"]
                         if f["id"] == action["food_id"] and not f["collected"]), None)
            a = self._state["agents"][agent_id]
            if food is None:
                return Validation(False, "unknown_or_collected_food")
            if abs(food["x"] - a["x"]) + abs(food["y"] - a["y"]) != 1:
                return Validation(False, "food_not_adjacent")
            return Validation(True, "accepted", {"type": "load", "food_id": food["id"]})
        return Validation(False, "invalid_action")

    def _fallback_action(self, agent_id):
        return {"type": "wait"}

    def _step(self, actions, draft):
        occupied = {(a["x"], a["y"]) for a in draft["agents"].values()}
        food_cells = {(f["x"], f["y"]) for f in draft["foods"] if not f["collected"]}
        candidates = {}
        feedback = {}
        rewards = {aid: Fraction(0) for aid in self.agent_ids}
        for aid in self.agent_ids:
            action = actions[aid]
            kind = action["type"]
            if kind == "wait":
                feedback[aid] = "waited"
                draft["wait_count"] += 1
            elif kind == "move":
                a = draft["agents"][aid]
                dx, dy = DIRECTIONS[action["direction"]]
                dest = (a["x"] + dx, a["y"] + dy)
                if not 0 <= dest[0] < draft["width"] or not 0 <= dest[1] < draft["height"]:
                    reason = "out_of_bounds"
                elif [*dest] in draft["walls"]:
                    reason = "wall"
                elif dest in food_cells:
                    reason = "food_occupied"
                elif dest in occupied:
                    reason = "agent_occupied"
                else:
                    reason = None
                if reason:
                    feedback[aid] = reason
                    draft["move_failures"][reason] += 1
                else:
                    candidates[aid] = dest
            else:
                feedback[aid] = "insufficient_level"
        destinations = list(candidates.values())
        for aid, dest in candidates.items():
            if destinations.count(dest) > 1:
                feedback[aid] = "target_conflict"
                draft["move_failures"]["target_conflict"] += 1
            else:
                draft["agents"][aid]["x"], draft["agents"][aid]["y"] = dest
                feedback[aid] = "moved"
                draft["move_success"] += 1
        successful_foods = []
        for food in draft["foods"]:
            if food["collected"]:
                continue
            participants = [aid for aid in self.agent_ids
                            if actions[aid]["type"] == "load" and actions[aid]["food_id"] == food["id"]]
            if not participants:
                continue
            level_sum = sum(draft["agents"][aid]["level"] for aid in participants)
            if level_sum < food["level"]:
                draft["load_insufficient"] += len(participants)
                continue
            food["collected"] = True
            draft["collected_level"] += food["level"]
            draft["load_success"] += len(participants)
            by_agent = {}
            for aid in participants:
                amount = Fraction(draft["agents"][aid]["level"] * food["level"],
                                  level_sum * draft["R0"])
                rewards[aid] += amount
                previous = from_exact(draft["agents"][aid]["cumulative_reward_exact"])
                draft["agents"][aid]["cumulative_reward_exact"] = exact(previous + amount)
                feedback[aid] = "load_success"
                by_agent[aid] = reward_record(amount)
            event = {"food_id": food["id"], "food_level": food["level"],
                     "coalition_agent_ids": participants, "coalition_level_sum": level_sum,
                     "reward_by_agent": by_agent,
                     "pivotal_agent_ids": [aid for aid in participants
                                           if level_sum - draft["agents"][aid]["level"] < food["level"]]}
            draft["coalition_events"].append(event)
            successful_foods.append(event)
        draft["round_completed"] += 1
        if all(f["collected"] for f in draft["foods"]):
            draft["done"], draft["termination_reason"] = True, "all_collected"
        elif draft["round_completed"] == self.horizon:
            draft["done"], draft["termination_reason"] = True, "horizon"
        outcome = {"positions": {aid: {"x": a["x"], "y": a["y"]}
                                 for aid, a in draft["agents"].items()},
                   "collected_food_ids": [e["food_id"] for e in successful_foods],
                   "coalition_events": successful_foods,
                   "collected_level": draft["collected_level"]}
        return rewards, outcome, feedback

    def _score(self):
        state = self._state
        s = sum(f["collected"] and f["level"] == 1 for f in state["foods"])
        t = sum(f["collected"] and f["level"] == 2 for f in state["foods"])
        per_agent = {aid: reward_record(from_exact(a["cumulative_reward_exact"]))
                     for aid, a in state["agents"].items()}
        return {"partial": not self.done, "rounds_completed": self.round_completed,
                "team_collected_fraction": state["collected_level"] / state["R0"] if self.done else None,
                "diagnostic_collected_fraction_prefix": state["collected_level"] / state["R0"],
                "reward_by_agent": per_agent,
                "level_one_collected": s, "level_two_collected": t,
                "level_one_completion_fraction": s / len(self.agent_ids) if self.done else None,
                "level_two_completion_fraction": t / len(self.agent_ids) if self.done else None,
                "all_collected_round": self.round_completed if self.termination_reason == "all_collected" else None,
                "move_success": state["move_success"], "move_failures": state["move_failures"],
                "wait_count": state["wait_count"], "load_success": state["load_success"],
                "load_insufficient": state["load_insufficient"],
                "coalition_events": state["coalition_events"]}
