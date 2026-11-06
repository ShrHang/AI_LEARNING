"""Independent checks for the algorithms (standard library only).

Run from any directory with ``python scripts/verify.py``. The reference
implementations deliberately do not reuse A* estimates, game transitions,
or alpha-beta traversal. No files are written.
"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import permutations
from pathlib import Path
from random import Random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from searchlab.examples import MAZE, POSITIONS, TREASURE_GRID, VALUES, WEIGHTED
from searchlab.games import (
    TreeNode, sample_tree, treasure_root_values, treasure_search, tree_search,
)
from searchlab.pathfinding import astar, maze_path, weighted_cost


def entry(cell: int | str) -> int:
    """Reference map interpretation, independent of the implementation."""
    return int(cell) if isinstance(cell, int) or str(cell).isdigit() else 1


def reference_dijkstra(grid, start, goal, goal_cost=None) -> int:
    queue = [(0, start)]
    settled = set()
    while queue:
        cost, current = heappop(queue)
        if current in settled:
            continue
        settled.add(current)
        if current == goal:
            return cost
        r, c = current
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= rr < len(grid) and 0 <= cc < len(grid[0])):
                continue
            if grid[rr][cc] == "X" or (rr, cc) in settled:
                continue
            step = goal_cost if (rr, cc) == goal and goal_cost is not None else entry(grid[rr][cc])
            heappush(queue, (cost + step, (rr, cc)))
    return -1


def check_path(grid, start, goal, goal_cost=None) -> int:
    expected = reference_dijkstra(grid, start, goal, goal_cost)
    for heuristic in ("zero", "manhattan"):
        actual = astar(grid, start, goal, heuristic=heuristic, goal_cost=goal_cost)
        assert actual.cost == expected, ("path cost", heuristic, grid, start, goal, expected, actual)
        if expected == -1:
            assert actual.path == [], ("unreachable path", actual)
            continue
        assert actual.path[0] == start and actual.path[-1] == goal
        total = 0
        for previous, current in zip(actual.path, actual.path[1:]):
            r, c = current
            assert 0 <= r < len(grid) and 0 <= c < len(grid[0])
            assert grid[r][c] != "X"
            assert abs(previous[0] - r) + abs(previous[1] - c) == 1
            total += goal_cost if current == goal and goal_cost is not None else entry(grid[r][c])
        assert total == expected, ("path accumulation", actual.path, total, expected)
        assert actual.expanded >= 1 and actual.generated >= actual.expanded
    return 2


def verify_paths(rng: Random) -> int:
    count = 0
    assert len(maze_path((0, 0), (3, 4), MAZE)) - 1 == 7
    assert weighted_cost((0, 0), (4, 4), WEIGHTED) == 7
    fixed = [
        (MAZE, (0, 0), (3, 4), None),
        (WEIGHTED, (0, 0), (4, 4), 0),
        ([["S", "X", "E"]], (0, 0), (0, 2), None),
        ([["S"]], (0, 0), (0, 0), 0),
        ([["S", "E"]], (0, 0), (0, 1), 0),
        ([["S", 0, 0], [0, "X", 0], [0, 0, "E"]], (0, 0), (2, 2), 0),
        ([["S", 4, 1], [1, 1, "E"]], (0, 0), (1, 2), 3),
    ]
    for grid, start, goal, terminal in fixed:
        count += check_path(grid, start, goal, terminal)
    for case in range(200):
        rows, cols = rng.randint(1, 8), rng.randint(1, 8)
        grid = [["X" if rng.random() < .25 else rng.choice((0, 1, 1, 2, 5, 9)) for _ in range(cols)] for _ in range(rows)]
        start = (rng.randrange(rows), rng.randrange(cols))
        goal = (rng.randrange(rows), rng.randrange(cols))
        grid[start[0]][start[1]] = "S"
        grid[goal[0]][goal[1]] = "E"
        count += check_path(grid, start, goal, 0 if case % 2 else None)
    return count


def reference_tree(node, maximizing=True) -> int:
    if not node.children:
        return node.value
    children = [reference_tree(child, not maximizing) for child in node.children]
    return (max if maximizing else min)(children)


def random_tree(rng: Random, depth: int) -> TreeNode:
    if depth == 0 or rng.random() < .15:
        return TreeNode(value=rng.randint(-20, 20))
    return TreeNode(children=[random_tree(rng, depth - 1) for _ in range(rng.randint(1, 4))])


def verify_trees(rng: Random) -> int:
    count = 0
    assert reference_tree(sample_tree()) == 5
    for root in [sample_tree()] + [random_tree(rng, rng.randint(1, 5)) for _ in range(100)]:
        expected = reference_tree(root)
        baseline = tree_search(root, algorithm="minimax")
        for algorithm in ("minimax", "alphabeta"):
            for order in ("natural", "reverse", "ideal"):
                actual = tree_search(root, algorithm=algorithm, order=order)
                assert actual.value == expected, ("tree value", algorithm, order, actual.value, expected)
                assert actual.visited <= baseline.visited
                assert actual.leaves <= baseline.leaves
                count += 1
    return count


def exhaustive_treasure_values(distances, values, depth):
    """Enumerate every move sequence, then collapse the prefix table bottom-up.

    This uses permutations rather than the production recursive game engine.
    Each terminal score is recomputed directly from its complete sequence.
    """
    depth = min(depth, len(values))
    table = {}
    for moves in permutations(values, depth):
        positions = ["A", "B"]
        scores = [0, 0]
        for turn, target in enumerate(moves):
            player = turn % 2
            scores[player] += values[target] - distances[positions[player]][target]
            positions[player] = target
        table[moves] = scores[0] - scores[1]
    for length in range(depth - 1, -1, -1):
        parents = {}
        for sequence, utility in table.items():
            parents.setdefault(sequence[:-1], []).append(utility)
        table = {prefix: (max if length % 2 == 0 else min)(scores) for prefix, scores in parents.items()}
        if length == 1:
            roots = {prefix[0]: utility for prefix, utility in table.items()}
    if depth == 1:
        roots = {target: values[target] - distances["A"][target] for target in values}
    return roots, table[()]


def verify_treasures(rng: Random) -> int:
    count = 0
    distances = {}
    for source, start in POSITIONS.items():
        distances[source] = {}
        for target, goal in POSITIONS.items():
            count += check_path(TREASURE_GRID, start, goal)
            distances[source][target] = reference_dijkstra(TREASURE_GRID, start, goal)
    scenarios = [(distances, VALUES)]
    labels = list(POSITIONS)
    for _ in range(30):
        random_distances = {a: {b: 0 if a == b else rng.randint(1, 20) for b in labels} for a in labels}
        random_values = {target: rng.randint(1, 15) for target in VALUES}
        scenarios.append((random_distances, random_values))
    for index, (matrix, values) in enumerate(scenarios):
        for depth in (1, 2, 3, 4):
            roots, expected = exhaustive_treasure_values(matrix, values, depth)
            assert treasure_root_values(matrix, values, depth) == roots
            baseline = treasure_search(matrix, values, depth=depth, algorithm="minimax")
            for algorithm in ("minimax", "alphabeta"):
                for order in ("natural", "reverse", "greedy"):
                    actual = treasure_search(matrix, values, depth=depth, algorithm=algorithm, order=order)
                    assert actual.value == expected, ("treasure value", index, depth, algorithm, order, expected, actual)
                    assert roots[actual.target] == expected, ("treasure root choice", roots, actual)
                    assert actual.visited <= baseline.visited
                    count += 1
            if index == 0 and depth in (2, 4):
                assert (baseline.target, baseline.value) == (("T2", -1) if depth == 2 else ("T3", 3))
    return count


def main() -> None:
    rng = Random(20261107)
    paths = verify_paths(rng)
    trees = verify_trees(rng)
    treasures = verify_treasures(rng)
    print(f"PASS: {paths + trees + treasures} algorithm checks (paths={paths}, trees={trees}, treasures={treasures}; seed=20261107)")


if __name__ == "__main__":
    main()
