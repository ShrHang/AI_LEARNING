"""Minimax and alpha-beta search for trees and the treasure game.

Counters describe the recursive search: ``visited`` includes the root,
``leaves`` counts evaluated terminal/depth-limit states, and ``cutoffs``
counts a pruning event only when at least one sibling is skipped.
"""

from dataclasses import dataclass, field
from math import inf
from typing import Any


@dataclass
class TreeNode:
    value: int | None = None
    children: list["TreeNode"] | None = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []


@dataclass
class GameResult:
    value: int
    target: str | None = None
    visited: int = 0
    leaves: int = 0
    cutoffs: int = 0


def sample_tree() -> TreeNode:
    """The official example; the middle MAX subtree is visited in both branches."""
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[
        TreeNode(children=[max1, max2]),
        TreeNode(children=[max2, max3]),
    ])


def tree_from_dict(data: dict[str, Any]) -> TreeNode:
    """Read the example JSON (with or without its tree_structure wrapper).

    Roles alternate from a MAX root, as prescribed in the assignment.
    JSON is_maximizing_player annotations are therefore redundant.
    """
    node = data.get("tree_structure", data)
    return TreeNode(
        value=node.get("value"),
        children=[tree_from_dict(child) for child in node.get("children", [])],
    )


def tree_search(
    root: TreeNode, *, algorithm: str = "alphabeta", order: str = "natural"
) -> GameResult:
    """Return the MAX root's value.

    ``ideal`` uses exact subtree minimax values as an oracle. It illustrates
    best-case ordering, not a usable free optimization: the preprocessing
    visits are excluded from result counters. Timing the whole function
    includes that preprocessing cost.
    """
    if algorithm not in {"minimax", "alphabeta"}:
        raise ValueError("algorithm must be minimax or alphabeta")
    if order not in {"natural", "reverse", "ideal"}:
        raise ValueError("tree order must be natural, reverse or ideal")
    result = GameResult(0)
    exact: dict[tuple[int, bool], int] = {}

    def oracle(node: TreeNode, maximizing: bool) -> int:
        key = (id(node), maximizing)
        if key not in exact:
            if not node.children:
                if node.value is None:
                    raise ValueError("a leaf must have a value")
                exact[key] = node.value
            else:
                values = [oracle(child, not maximizing) for child in node.children]
                exact[key] = max(values) if maximizing else min(values)
        return exact[key]

    if order == "ideal":
        oracle(root, True)

    def visit(node: TreeNode, maximizing: bool, alpha: float, beta: float) -> int:
        result.visited += 1
        if not node.children:
            result.leaves += 1
            if node.value is None:
                raise ValueError("a leaf must have a value")
            return node.value
        children = list(node.children)
        if order == "reverse":
            children.reverse()
        elif order == "ideal":
            children.sort(
                key=lambda child: exact[(id(child), not maximizing)],
                reverse=maximizing,
            )
        best = -inf if maximizing else inf
        for index, child in enumerate(children):
            value = visit(child, not maximizing, alpha, beta)
            best = max(best, value) if maximizing else min(best, value)
            if algorithm == "alphabeta":
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta and index + 1 < len(children):
                    result.cutoffs += 1
                    break
        return int(best)

    result.value = visit(root, True, -inf, inf)
    return result


@dataclass(frozen=True)
class TreasureState:
    position_a: str = "A"
    position_b: str = "B"
    remaining: tuple[str, ...] = ()
    score_a: int = 0
    score_b: int = 0
    player: str = "A"

    @property
    def utility(self) -> int:
        return self.score_a - self.score_b


def treasure_move(
    state: TreasureState,
    target: str,
    distances: dict[str, dict[str, int]],
    values: dict[str, int],
) -> TreasureState:
    """Apply one chosen treasure; passing other treasures never collects them."""
    remaining = tuple(label for label in state.remaining if label != target)
    if state.player == "A":
        gain = values[target] - distances[state.position_a][target]
        return TreasureState(
            target, state.position_b, remaining,
            state.score_a + gain, state.score_b, "B",
        )
    gain = values[target] - distances[state.position_b][target]
    return TreasureState(
        state.position_a, target, remaining,
        state.score_a, state.score_b + gain, "A",
    )


def _treasure_search(
    distances: dict[str, dict[str, int]],
    values: dict[str, int],
    state: TreasureState,
    depth: int,
    algorithm: str,
    order: str,
) -> GameResult:
    result = GameResult(state.utility)

    def visit(
        current: TreasureState, remaining_depth: int, alpha: float, beta: float
    ) -> tuple[int, str | None]:
        result.visited += 1
        if remaining_depth == 0 or not current.remaining:
            result.leaves += 1
            return current.utility, None
        maximizing = current.player == "A"
        position = current.position_a if maximizing else current.position_b
        targets = [
            label for label in current.remaining if distances[position][label] >= 0
        ]
        if not targets:
            result.leaves += 1
            return current.utility, None
        if order == "reverse":
            targets.reverse()
        elif order == "greedy":
            # Each player first tries their own greatest immediate gain.
            targets.sort(
                key=lambda label: values[label] - distances[position][label],
                reverse=True,
            )
        best = -inf if maximizing else inf
        best_target = None
        for index, target in enumerate(targets):
            child = treasure_move(current, target, distances, values)
            value, _ = visit(child, remaining_depth - 1, alpha, beta)
            if (maximizing and value > best) or (not maximizing and value < best):
                best, best_target = value, target
            if algorithm == "alphabeta":
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta and index + 1 < len(targets):
                    result.cutoffs += 1
                    break
        return int(best), best_target

    result.value, result.target = visit(state, depth, -inf, inf)
    return result


def treasure_search(
    distances: dict[str, dict[str, int]],
    values: dict[str, int],
    *,
    depth: int = 4,
    algorithm: str = "alphabeta",
    order: str = "natural",
) -> GameResult:
    """Search from A's first turn; depth counts individual player moves.

    Equal-valued choices keep the first encountered target. Unreachable
    treasures are not legal moves; a player with no legal move ends search.
    All official-map treasures are mutually reachable.
    """
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if algorithm not in {"minimax", "alphabeta"}:
        raise ValueError("algorithm must be minimax or alphabeta")
    if order not in {"natural", "reverse", "greedy"}:
        raise ValueError("treasure order must be natural, reverse or greedy")
    return _treasure_search(
        distances, values, TreasureState(remaining=tuple(values)),
        depth, algorithm, order,
    )


def treasure_root_values(
    distances: dict[str, dict[str, int]], values: dict[str, int], depth: int
) -> dict[str, int]:
    """Evaluate each root move independently, with an exact minimax search.

    Alpha-beta bounds from an earlier root sibling are deliberately not reused:
    these numbers must be exact values, not fail-low/fail-high bounds.
    """
    if depth < 1:
        return {}
    state = TreasureState(remaining=tuple(values))
    return {
        target: _treasure_search(
            distances, values, treasure_move(state, target, distances, values),
            depth - 1, "minimax", "natural",
        ).value
        for target in values if distances["A"][target] >= 0
    }
