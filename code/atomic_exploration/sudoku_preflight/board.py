"""Deterministic Sudoku evidence cases and an exact, offline completion oracle.

Coordinates in the public interface are one based.  Candidate membership is an
existence query over *all* valid completions, not a digit read from one solution.
Witness boards belong in audit data and must never be placed in model prompts.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence


_ALL = (1 << 9) - 1
_CELLS = tuple((i // 9, i % 9, (i // 27) * 3 + (i % 9) // 3) for i in range(81))


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _target(target: Sequence[int]) -> tuple[int, int]:
    if not isinstance(target, (list, tuple)) or len(target) != 2:
        raise ValueError("target must be a two-item, one-based coordinate")
    if any(not _integer(x) or not 1 <= x <= 9 for x in target):
        raise ValueError("target coordinates must be integers from 1 to 9")
    return target[0] - 1, target[1] - 1


def _givens(givens: Iterable[Sequence[int]]) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    seen: dict[tuple[int, int], int] = {}
    for given in givens:
        if not isinstance(given, (list, tuple)) or len(given) != 3:
            raise ValueError("each given must be [row, column, digit]")
        if any(not _integer(x) or not 1 <= x <= 9 for x in given):
            raise ValueError("given coordinates and digits must be integers from 1 to 9")
        r, c, d = given
        if (r, c) in seen:
            if seen[r, c] != d:
                raise ValueError("conflicting digits at one given coordinate")
            continue
        seen[r, c] = d
        result.append((r, c, d))
    return result


class _Solver:
    """Small bit-mask search, with exhaustive failure and capped solution count."""

    def __init__(self, givens: Iterable[Sequence[int]]) -> None:
        self.board = [0] * 81
        self.rows = [0] * 9
        self.columns = [0] * 9
        self.boxes = [0] * 9
        self.consistent = True
        for r1, c1, d in _givens(givens):
            i = (r1 - 1) * 9 + c1 - 1
            r, c, b = _CELLS[i]
            bit = 1 << (d - 1)
            if (self.rows[r] | self.columns[c] | self.boxes[b]) & bit:
                self.consistent = False
                break
            self.board[i] = d
            self.rows[r] |= bit
            self.columns[c] |= bit
            self.boxes[b] |= bit

    def solutions(
        self,
        limit: int = 1,
        *,
        rng: random.Random | None = None,
        preferred: Sequence[Sequence[int]] | None = None,
    ) -> list[list[list[int]]]:
        if limit < 1:
            raise ValueError("solution limit must be positive")
        found: list[list[list[int]]] = []
        preferred_flat = [d for row in preferred for d in row] if preferred else None
        if not self.consistent:
            return found

        def search() -> bool:
            best_i = -1
            best_mask = 0
            best_size = 10
            for i, value in enumerate(self.board):
                if value:
                    continue
                r, c, b = _CELLS[i]
                mask = _ALL & ~(self.rows[r] | self.columns[c] | self.boxes[b])
                size = mask.bit_count()
                if size == 0:
                    return False
                if size < best_size:
                    best_i, best_mask, best_size = i, mask, size
                    if size == 1:
                        break
            if best_i < 0:
                found.append([self.board[r * 9:(r + 1) * 9] for r in range(9)])
                return len(found) >= limit
            digits = [d for d in range(1, 10) if best_mask & (1 << (d - 1))]
            if rng is not None:
                rng.shuffle(digits)
            if preferred_flat is not None and preferred_flat[best_i] in digits:
                first = preferred_flat[best_i]
                digits.remove(first)
                digits.insert(0, first)
            r, c, b = _CELLS[best_i]
            for d in digits:
                bit = 1 << (d - 1)
                self.board[best_i] = d
                self.rows[r] |= bit
                self.columns[c] |= bit
                self.boxes[b] |= bit
                stop = search()
                self.board[best_i] = 0
                self.rows[r] ^= bit
                self.columns[c] ^= bit
                self.boxes[b] ^= bit
                if stop:
                    return True
            return False

        search()
        return found


def solve_one(givens: Iterable[Sequence[int]]) -> list[list[int]] | None:
    """Return one witness completion, or None after exhaustive failure."""
    solutions = _Solver(givens).solutions()
    return solutions[0] if solutions else None


def _candidate_witnesses(
    givens: Iterable[Sequence[int]], target: Sequence[int],
    *, preferred: Sequence[Sequence[int]] | None = None,
) -> dict[str, list[list[int]]]:
    r, c = _target(target)
    clues = _givens(givens)
    solver = _Solver(clues)
    if not solver.consistent:
        return {}
    current = solver.board[r * 9 + c]
    if current:
        solutions = solver.solutions(preferred=preferred)
        return {str(current): solutions[0]} if solutions else {}
    b = (r // 3) * 3 + c // 3
    possible = _ALL & ~(solver.rows[r] | solver.columns[c] | solver.boxes[b])
    witnesses: dict[str, list[list[int]]] = {}
    for digit in range(1, 10):
        if not possible & (1 << (digit - 1)):
            continue
        completed = _Solver(clues + [(r + 1, c + 1, digit)]).solutions(preferred=preferred)
        if completed:
            witnesses[str(digit)] = completed[0]
    return witnesses


def target_candidates(givens: Iterable[Sequence[int]], target: Sequence[int]) -> list[int]:
    """Digits possible at target under all valid completions of the givens.

    Each locally possible digit gets its own exact satisfiability query. A
    missing witness means exhaustive failure, including globally inconsistent
    clues that have no immediate row/column/box conflict.
    """
    return sorted(int(d) for d in _candidate_witnesses(givens, target))


def _triples(solution: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[r + 1, c + 1, solution[r][c]] for r in range(9) for c in range(9)]


def _copy_board(board: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(row) for row in board]


def _allows_alternative(
    clues: list[list[int]], proposed: list[int], target: list[int], digit: int,
    witness: list[list[int]], solution: list[list[int]],
) -> list[list[int]] | None:
    r, c, d = proposed
    if witness[r - 1][c - 1] == d:
        return witness
    completed = _Solver(clues + [proposed, [*target, digit]]).solutions(preferred=solution)
    return completed[0] if completed else None


def _complementary(
    rng: random.Random, solution: list[list[int]],
) -> tuple[list[int], list[list[int]], dict[str, list[list[int]]]]:
    r = rng.randrange(9)
    c, ca, cb = rng.sample(range(9), 3)
    target = [r + 1, c + 1]
    public = [[r + 1, j + 1, solution[r][j]] for j in range(9) if j not in (c, ca, cb)]
    private = {
        "A": [[r + 1, ca + 1, solution[r][ca]]],
        "B": [[r + 1, cb + 1, solution[r][cb]]],
    }
    alternative = {"A": solution[r][cb], "B": solution[r][ca]}
    witnesses: dict[str, list[list[int]]] = {}
    for agent in ("A", "B"):
        completed = _Solver(public + private[agent] + [[*target, alternative[agent]]]).solutions(preferred=solution)
        if not completed:
            raise RuntimeError("row split unexpectedly lacks an alternative completion")
        witnesses[agent] = completed[0]
    used = {(x[0], x[1]) for x in public + private["A"] + private["B"]}
    rest = [x for x in _triples(solution) if (x[0], x[1]) not in used and x[:2] != target]
    rng.shuffle(rest)
    minimum = rng.randrange(35, 45)
    for clue in rest:
        # Shared clues are accepted only while *both* alternative worlds survive.
        added = False
        wa = _allows_alternative(public + private["A"], clue, target, alternative["A"], witnesses["A"], solution)
        if wa is not None:
            wb = _allows_alternative(public + private["B"], clue, target, alternative["B"], witnesses["B"], solution)
            if wb is not None:
                public.append(clue)
                witnesses = {"A": wa, "B": wb}
                added = True
        if not added:
            order = ["A", "B"]
            rng.shuffle(order)
            for agent in order:
                witness = _allows_alternative(public + private[agent], clue, target, alternative[agent], witnesses[agent], solution)
                if witness is not None:
                    private[agent].append(clue)
                    witnesses[agent] = witness
                    added = True
                    break
        joint = public + private["A"] + private["B"]
        if added and len(joint) >= minimum and len(_Solver(joint).solutions(limit=2)) == 1:
            return target, public, private
    raise RuntimeError("unable to produce a unique joint puzzle with two ambiguous own views")


def _own_sufficient(
    rng: random.Random, solution: list[list[int]],
) -> tuple[list[int], list[list[int]], dict[str, list[list[int]]]]:
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    for r, c in cells:
        other_digits = [d for d in range(1, 10) if d != solution[r][c]]
        rng.shuffle(other_digits)
        for other in other_digits:
            ca = solution[r].index(other)
            rb = next(i for i in range(9) if solution[i][c] == other)
            target = [r + 1, c + 1]
            public = [[r + 1, j + 1, solution[r][j]] for j in range(9) if j not in (c, ca)]
            public += [[i + 1, c + 1, solution[i][c]] for i in range(9) if i not in (r, rb)]
            # Require an actual public alternative, not only a local pencil mark.
            if not _Solver(public + [[*target, other]]).solutions(preferred=solution):
                continue
            private = {
                "A": [[r + 1, ca + 1, other]],
                "B": [[rb + 1, c + 1, other]],
            }
            used = {(x[0], x[1]) for x in public + private["A"] + private["B"]}
            rest = [x for x in _triples(solution) if (x[0], x[1]) not in used and x[:2] != target]
            rng.shuffle(rest)
            minimum = rng.randrange(35, 45)
            for clue in rest:
                agent = "A" if len(private["A"]) < len(private["B"]) else "B"
                if len(private["A"]) == len(private["B"]):
                    agent = rng.choice(("A", "B"))
                private[agent].append(clue)
                joint = public + private["A"] + private["B"]
                if len(joint) >= minimum and len(_Solver(joint).solutions(limit=2)) == 1:
                    return target, public, private
    raise RuntimeError("unable to produce own-sufficient row/column evidence")


def build_cases(seed: int = 42, count: int = 12) -> list[dict[str, Any]]:
    """Build independent solved grids and clue patterns, with a 2:1 stratum mix.

    The default contains eight complementary and four own_sufficient cases.
    All randomness comes from a private seeded generator; no external service,
    existing experiment, global random state, or wall-clock input is consulted.
    """
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    if not _integer(count) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    sufficient_count = count // 3
    strata = ["complementary"] * (count - sufficient_count) + ["own_sufficient"] * sufficient_count
    for index, stratum in enumerate(strata, start=1):
        for _attempt in range(32):
            solution = _Solver([]).solutions(rng=rng)[0]
            try:
                target, public, private = (_complementary if stratum == "complementary" else _own_sufficient)(rng, solution)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(f"failed to generate case {index} after 32 independent attempts")
        public.sort()
        for clues in private.values():
            clues.sort()
        own_witnesses = {
            agent: _candidate_witnesses(public + private[agent], target, preferred=solution)
            for agent in ("A", "B")
        }
        joint = public + private["A"] + private["B"]
        joint_witnesses = _candidate_witnesses(joint, target, preferred=solution)
        public_witnesses = _candidate_witnesses(public, target, preferred=solution)
        case = {
            "case_id": f"sudoku_{index:02d}",
            "stratum": stratum,
            "target": target,
            "public_givens": public,
            "private_givens": private,
            "solution": _copy_board(solution),
            "oracle": {
                "own_candidates": {a: sorted(int(d) for d in own_witnesses[a]) for a in ("A", "B")},
                "joint_candidates": sorted(int(d) for d in joint_witnesses),
                "public_candidates": sorted(int(d) for d in public_witnesses),
                "own_witnesses": own_witnesses,
                "joint_witnesses": joint_witnesses,
                "public_witnesses": public_witnesses,
                "joint_solution_count_capped_at_2": len(_Solver(joint).solutions(limit=2)),
                "joint_unique": True,
                "target_digit": solution[target[0] - 1][target[1] - 1],
                "method": "separate exact satisfiability query for each target digit; exhaustive joint uniqueness search",
            },
        }
        checked = validate_case(case)
        if not checked["ok"]:
            raise RuntimeError(f"generated invalid case: {checked['errors']}")
        cases.append(case)
    return cases


def _board_errors(board: Any, clues: list[tuple[int, int, int]], target: Sequence[int] | None = None, digit: int | None = None) -> list[str]:
    if not isinstance(board, (list, tuple)) or len(board) != 9 or any(not isinstance(row, (list, tuple)) or len(row) != 9 for row in board):
        return ["completion must be a 9 by 9 board"]
    if any(not _integer(d) or not 1 <= d <= 9 for row in board for d in row):
        return ["completion digits must be integers from 1 to 9"]
    expected = set(range(1, 10))
    errors = []
    if any(set(row) != expected for row in board):
        errors.append("completion has an invalid row")
    if any({board[r][c] for r in range(9)} != expected for c in range(9)):
        errors.append("completion has an invalid column")
    if any({board[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)} != expected for br in (0, 3, 6) for bc in (0, 3, 6)):
        errors.append("completion has an invalid box")
    if any(board[r - 1][c - 1] != d for r, c, d in clues):
        errors.append("completion disagrees with visible givens")
    if target is not None and board[target[0] - 1][target[1] - 1] != digit:
        errors.append("completion disagrees with candidate target digit")
    return errors


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Recompute every oracle claim and check all retained completion witnesses."""
    errors: list[str] = []
    try:
        target = case["target"]
        _target(target)
        public = _givens(case["public_givens"])
        private = {a: _givens(case["private_givens"][a]) for a in ("A", "B")}
        oracle = case["oracle"]
        stratum = case["stratum"]
        solution = case["solution"]
        if not isinstance(oracle, dict):
            raise ValueError("oracle must be an object")
    except (KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "errors": [f"invalid case structure: {exc}"]}
    joint = public + private["A"] + private["B"]
    coordinates = [(r, c) for r, c, _ in joint]
    if len(coordinates) != len(set(coordinates)):
        errors.append("public and private partitions must have disjoint given coordinates")
    if tuple(target) in coordinates:
        errors.append("target is already a given")
    solution_errors = _board_errors(solution, joint)
    errors.extend(f"solution: {err}" for err in solution_errors)
    if stratum not in ("complementary", "own_sufficient"):
        errors.append("unknown stratum")
    try:
        views = {"public": public, "joint": joint, "A": public + private["A"], "B": public + private["B"]}
        computed = {name: target_candidates(clues, target) for name, clues in views.items()}
        actual = computed["joint"]
        if len(actual) != 1:
            errors.append("joint clues do not uniquely fix the target")
        for agent in ("A", "B"):
            allowed = computed[agent]
            if not allowed:
                errors.append(f"{agent} visible clues have no valid completion")
            if stratum == "complementary" and len(allowed) < 2:
                errors.append(f"{agent} own target candidates are not ambiguous")
            if stratum == "own_sufficient" and allowed != actual:
                errors.append(f"{agent} own clues do not independently fix the joint target")
            claimed = oracle.get("own_candidates", {}).get(agent)
            if claimed != allowed:
                errors.append(f"{agent} candidate claim disagrees with exact oracle")
        if oracle.get("joint_candidates") != actual:
            errors.append("joint candidate claim disagrees with exact oracle")
        if oracle.get("public_candidates") != computed["public"]:
            errors.append("public candidate claim disagrees with exact oracle")
        for name, clues in views.items():
            saved = oracle.get("own_witnesses", {}).get(name, {}) if name in ("A", "B") else oracle.get(f"{name}_witnesses", {})
            if not isinstance(saved, dict) or set(saved) != {str(d) for d in computed[name]}:
                errors.append(f"{name} witness keys do not cover exactly all allowed target digits")
                continue
            for digit in computed[name]:
                errors.extend(f"{name} witness {digit}: {err}" for err in _board_errors(saved[str(digit)], clues, target, digit))
        joint_count = len(_Solver(joint).solutions(limit=2))
        if oracle.get("joint_solution_count_capped_at_2") != joint_count:
            errors.append("joint solution-count claim disagrees with exhaustive search")
        if oracle.get("joint_unique") != (joint_count == 1):
            errors.append("joint uniqueness claim disagrees with exhaustive search")
        if oracle.get("target_digit") != (actual[0] if len(actual) == 1 else None):
            errors.append("target digit claim disagrees with exact oracle")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid oracle or givens: {exc}")
    return {"ok": not errors, "errors": errors}
