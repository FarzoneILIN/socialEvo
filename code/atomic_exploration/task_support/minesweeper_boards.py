"""Legal split-clue Minesweeper checkpoints and exhaustive offline proofs.

Coordinates are [row, column], both 1-based.  A checkpoint consists only of
safe, positive-number openings: replay therefore has no zero flood-fill and no
flags, deaths, hidden extra openings, or informative player decisions.  Clue
ownership is sampled before the true mine configuration.  Sampling strata and
all solutions remain private; ``facts_for_actor`` is the only prompt export.
"""
from __future__ import annotations

import copy
import itertools
import random
import time
from functools import lru_cache


Coord = tuple[int, int]
ACTORS = ("A", "B")


def _actor_name(actor):
    if actor in ("A", 0):
        return "A"
    if actor in ("B", 1):
        return "B"
    raise ValueError("actor must be A, B, 0, or 1")


def _coord(value) -> Coord:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("coordinate must contain row and column")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in value):
        raise ValueError("coordinate entries must be integers")
    return tuple(value)


@lru_cache(maxsize=16)
def _cells(width, height):
    return tuple((row, col) for row in range(1, height + 1)
                 for col in range(1, width + 1))


@lru_cache(maxsize=128)
def _neighbors(width, height, cell):
    row, col = cell
    return frozenset((rr, cc) for rr in range(max(1, row - 1), min(height, row + 1) + 1)
                     for cc in range(max(1, col - 1), min(width, col + 1) + 1)
                     if (rr, cc) != cell)


@lru_cache(maxsize=4096)
def _candidate_boards(width, height, opened, total_mines):
    """All boards consistent with public open/safe cells and global mine count."""
    closed = tuple(cell for cell in _cells(width, height) if cell not in opened)
    return tuple(frozenset(config) for config in itertools.combinations(closed, total_mines))


def _history_positive_constraints(public):
    """Numbers forced positive by the publicly observed absence of flood-fill.

    A zero click would also reveal every neighboring cell that was still
    closed.  With no flags, a prefix that opens exactly one new cell therefore
    implies a positive digit whenever at least one neighbor remains unopened.
    A late corner zero surrounded by already opened cells is still permitted.
    This explicitly accounts for information conveyed by public operations.
    """
    width, height = public["width"], public["height"]
    opened_before = set()
    required_positive = []
    for event in public["history"]:
        cell = _coord(event["cell"])
        neighbors = _neighbors(width, height, cell)
        if neighbors - opened_before:
            required_positive.append(neighbors)
        opened_before.add(cell)
    return required_positive


def possible_configurations(state, actor=None):
    """Return all consistent mine placements, as tuples of coordinate tuples.

    ``actor=None`` uses the union of A/B's observed numerical clues.  An actor
    uses only its own digits, public total mine count, and public safe openings.
    The public sequence also constrains ordinary zero flood-fill: each event
    must reveal exactly the one recorded new cell.  This routine deliberately
    never consults true_mines or the stratum label.
    Returned coordinates are immutable offline data, never a prompt field.
    """
    public = state["public"]
    width, height = public["width"], public["height"]
    opened = tuple(sorted(_coord(c) for c in public["opened"]))
    total = public["total_mines"]
    actors = ACTORS if actor is None else (_actor_name(actor),)
    clues = [(_coord(item["cell"]), item["digit"])
             for name in actors for item in state["private_views"][name]]
    constraints = [(_neighbors(width, height, cell), number) for cell, number in clues]
    positive_constraints = _history_positive_constraints(public)
    return [tuple(sorted(mines))
            for mines in _candidate_boards(width, height, opened, total)
            if all(mines.intersection(adjacent) for adjacent in positive_constraints)
            and all(len(mines.intersection(adjacent)) == number
                   for adjacent, number in constraints)]


def _configuration_proof(state, actor=None):
    configurations = possible_configurations(state, actor)
    if not configurations:
        raise ValueError("observed state has no consistent mine configurations")
    union = set().union(*(set(config) for config in configurations))
    intersection = set(configurations[0]).intersection(*map(set, configurations[1:]))
    closed = {_coord(c) for c in state["public"]["closed"]}
    return {
        "configuration_count": len(configurations),
        "provably_safe": [list(cell) for cell in sorted(closed - union)],
        "provably_mined": [list(cell) for cell in sorted(intersection)],
        "possible_mines": [list(cell) for cell in sorted(union)],
    }


def facts_for_actor(state, actor, full=False):
    """Return a fresh, allowlisted JSON-ready model view; no answer/proof leaks.

    Even a full-information view exposes only digits on already opened safe
    cells.  It never includes closed-cell digits, true mines, solutions, proof
    counts, target cells, state IDs, generation seeds, or selected strata.
    """
    actor = _actor_name(actor)
    names = ACTORS if full else (actor,)
    numeric_clues = sorted((copy.deepcopy(item) for name in names
                            for item in state["private_views"][name]),
                           key=lambda item: item["cell"])
    return {
        "task": "Choose one closed cell to open in shared Minesweeper",
        "you_are": actor,
        "coordinate_system": "[row, column], 1-based; row increases downward and column increases rightward",
        "rules": [
            "A revealed digit counts mines in the up to eight neighboring cells, including diagonals.",
            "The public total_mines is the exact number of mines on the whole board.",
            "Every opened cell is safe. A missing digit is private, not zero.",
            "Normal zero flood-fill applies. Each recorded opening revealed exactly its listed cell and no extra cells.",
            "The checkpoint opening sequence was supplied externally; no informed player selected these moves.",
            "Numeric clue ownership was assigned before board digits were computed.",
            "No flags or earlier failed openings exist. All listed closed cells are legal next choices.",
            "Both agents share the outcome. Prefer a closed cell whose safety follows from available clues.",
        ],
        "public": copy.deepcopy(state["public"]),
        "numeric_clues_available_to_you": numeric_clues,
    }


def validate_state(state):
    """Replay the legal checkpoint and prove both private views exhaustively.

    Raises ValueError for an invalid state.  Return values and true mine data
    are offline scoring material, not facts_for_actor fields.
    """
    public = state["public"]
    width, height = public["width"], public["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height)):
        raise ValueError("invalid board dimensions")
    universe = set(_cells(width, height))
    opened_seq = [_coord(c) for c in public["opened"]]
    closed_seq = [_coord(c) for c in public["closed"]]
    opened, closed = set(opened_seq), set(closed_seq)
    if len(opened) != len(opened_seq) or len(closed) != len(closed_seq):
        raise ValueError("duplicate cells")
    if opened & closed or opened | closed != universe:
        raise ValueError("opened/closed cells do not partition board")
    scoring = state["private_scoring"]
    mine_seq = [_coord(c) for c in scoring["true_mines"]]
    mines = set(mine_seq)
    if len(mines) != len(mine_seq) or not mines <= closed:
        raise ValueError("mines duplicated, out of bounds, or publicly opened")
    total = public["total_mines"]
    if isinstance(total, bool) or not isinstance(total, int) or total != len(mines):
        raise ValueError("global mine count mismatch")
    if total < 1 or total >= len(closed):
        raise ValueError("uninformative or invalid total mine count")

    observed = {}
    for actor in ACTORS:
        for clue in state["private_views"][actor]:
            cell, digit = _coord(clue["cell"]), clue["digit"]
            if cell not in opened or cell in observed:
                raise ValueError("private clues must partition safe opened cells")
            if isinstance(digit, bool) or not isinstance(digit, int) or digit not in range(1, 9):
                raise ValueError("only positive integer clues are supported")
            true_digit = len(mines.intersection(_neighbors(width, height, cell)))
            if digit != true_digit:
                raise ValueError("observed digit does not match true board")
            observed[cell] = digit
    if set(observed) != opened:
        raise ValueError("missing revealed numerical clue")
    if any(not state["private_views"][actor] for actor in ACTORS):
        raise ValueError("each actor must have a numeric clue")
    expected_masks = {
        "true_mine_mask": sum(1 << list(_cells(width, height)).index(cell) for cell in mines),
        "opened_mask": sum(1 << list(_cells(width, height)).index(cell) for cell in opened),
        "clue_ownership_mask_A": sum(1 << list(_cells(width, height)).index(_coord(clue["cell"]))
                                      for clue in state["private_views"]["A"]),
    }
    if any(scoring.get(key) != value for key, value in expected_masks.items()):
        raise ValueError("private bitmask metadata mismatch")
    private_all_clues = {_coord(item["cell"]): item["digit"] for item in scoring["all_opened_clues"]}
    if private_all_clues != observed or len(scoring["all_opened_clues"]) != len(observed):
        raise ValueError("private all-clue record mismatch")

    replayed = []
    for index, event in enumerate(public["history"], 1):
        if set(event) != {"move", "action", "cell", "result"}:
            raise ValueError("history must disclose only coordinate and safe result")
        cell = _coord(event["cell"])
        if (event["move"] != index or event["action"] != "open"
                or event["result"] != "opened" or cell in replayed
                or cell not in universe or cell in mines):
            raise ValueError("checkpoint opening history is not legal")
        if len(mines.intersection(_neighbors(width, height, cell))) == 0:
            raise ValueError("zero opening would need flood-fill replay")
        replayed.append(cell)
    if replayed != opened_seq:
        raise ValueError("opening history and checkpoint disagree")

    proofs = {actor: _configuration_proof(state, actor) for actor in ACTORS}
    proofs["joint"] = _configuration_proof(state)
    for actor in (*ACTORS, None):
        if tuple(sorted(mines)) not in possible_configurations(state, actor):
            raise ValueError("true board missing from consistent configurations")
    stratum = scoring.get("stratum")
    if stratum == "complementary":
        if any(proofs[actor]["provably_safe"] for actor in ACTORS) or not proofs["joint"]["provably_safe"]:
            raise ValueError("complementary state fails strict own/joint safety definition")
    elif stratum == "independent_safe":
        if not all(proofs[actor]["provably_safe"] for actor in ACTORS):
            raise ValueError("independent-safe state lacks own-view safe actions")
    else:
        raise ValueError("unknown state stratum")
    return {
        "valid": True,
        "legal_replay": True,
        "no_zero_flood_fill": True,
        "true_board_consistent_with_each_view": True,
        "global_total_enforced": True,
        "public_history_has_no_explicit_digit_or_true_mine_data": True,
        "public_no_extra_flood_fill_constraints_enforced": True,
        "configuration_counts": {actor: proof["configuration_count"] for actor, proof in proofs.items()},
        "provably_safe": {actor: proof["provably_safe"] for actor, proof in proofs.items()},
        "provably_mined": {actor: proof["provably_mined"] for actor, proof in proofs.items()},
        "views": proofs,
        "stratum": stratum,
    }


def _make_state(width, height, opened_order, a_cells, mines, index, stratum, attempts):
    opened = set(opened_order)
    all_cells = _cells(width, height)
    private_views = {actor: [] for actor in ACTORS}
    for cell in sorted(opened):
        actor = "A" if cell in a_cells else "B"
        private_views[actor].append({"cell": list(cell),
                                     "digit": len(mines.intersection(_neighbors(width, height, cell)))})
    return {
        "state_id": f"M{index:03d}",
        "public": {
            "width": width,
            "height": height,
            "total_mines": len(mines),
            "opened": [list(cell) for cell in opened_order],
            "closed": [list(cell) for cell in all_cells if cell not in opened],
            "history": [{"move": i, "action": "open", "cell": list(cell), "result": "opened"}
                        for i, cell in enumerate(opened_order, 1)],
        },
        "private_views": private_views,
        "private_scoring": {
            "true_mines": [list(cell) for cell in sorted(mines)],
            "true_mine_mask": sum(1 << all_cells.index(cell) for cell in mines),
            "opened_mask": sum(1 << all_cells.index(cell) for cell in opened),
            "clue_ownership_mask_A": sum(1 << all_cells.index(cell) for cell in a_cells),
            "all_opened_clues": [copy.deepcopy(item) for actor in ACTORS for item in private_views[actor]],
            "stratum": stratum,
            "selection_attempt": attempts,
            "clue_partition_rule": "seeded shuffle of opened coordinates, first half to A; before sampling mine placement",
            "checkpoint_rule": "externally supplied safe positive-number openings; no player policy inferred",
        },
    }


def generate_states(seed=42, per_stratum=4):
    """Generate matched 4x4 checkpoints with exactly 4 openings and 3 mines.

    Both strata have 12 closed cells and two numeric clues per agent.  The
    random open sequence and clue partition are drawn before the candidate
    mine placement; offline acceptance uses exact own/joint safety predicates.
    No model responses or previous experiment data affect selection.  Search
    is bounded to 110 seconds; it never relaxes the strict definition.
    """
    if isinstance(per_stratum, bool) or not isinstance(per_stratum, int) or per_stratum < 1:
        raise ValueError("per_stratum must be a positive integer")
    rng = random.Random(seed)
    width, height, mine_count, opening_count = 4, 4, 3, 4
    cells = _cells(width, height)
    found = {"complementary": [], "independent_safe": []}
    seen = set()
    deadline = time.monotonic() + 110
    attempts = 0
    while any(len(group) < per_stratum for group in found.values()):
        if time.monotonic() > deadline:
            raise RuntimeError(f"strict state search timed out after {attempts} candidates; "
                               + str({key: len(value) for key, value in found.items()}))
        attempts += 1
        # These draws precede the mine configuration and never inspect digits.
        opened_order = rng.sample(cells, opening_count)
        ownership_order = rng.sample(opened_order, opening_count)
        a_cells = frozenset(ownership_order[:opening_count // 2])
        closed = [cell for cell in cells if cell not in opened_order]
        mines = frozenset(rng.sample(closed, mine_count))
        if any(not mines.intersection(_neighbors(width, height, cell)) for cell in opened_order):
            continue
        fingerprint = (tuple(sorted(opened_order)), tuple(sorted(a_cells)), tuple(sorted(mines)))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        state = _make_state(width, height, opened_order, a_cells, mines, 0, "candidate", attempts)
        a_proof = _configuration_proof(state, "A")
        b_proof = _configuration_proof(state, "B")
        own_safe = [bool(a_proof["provably_safe"]), bool(b_proof["provably_safe"])]
        if all(own_safe):
            stratum = "independent_safe"
        elif not any(own_safe) and _configuration_proof(state)["provably_safe"]:
            stratum = "complementary"
        else:
            continue
        if len(found[stratum]) == per_stratum:
            continue
        state["private_scoring"]["stratum"] = stratum
        state["private_scoring"]["proof"] = validate_state(state)
        found[stratum].append(state)
    states = found["complementary"] + found["independent_safe"]
    # Opaque, randomly ordered IDs keep selection strata out of model inputs.
    rng.shuffle(states)
    for index, state in enumerate(states, 1):
        state["state_id"] = f"M{index:03d}"
        state["private_scoring"]["generator_seed"] = seed
        state["private_scoring"]["search_candidates"] = attempts
    return states
