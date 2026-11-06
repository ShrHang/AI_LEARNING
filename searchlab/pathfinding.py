"""Four-direction A* for unit-cost mazes and weighted terrain."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count


Position = tuple[int, int]
Grid = list[list[int | str]]


@dataclass
class SearchResult:
    path: list[Position]
    cost: int
    expanded: int
    generated: int


def _entry_cost(cell: int | str) -> int:
    if cell in (".", "S", "E", "A", "B") or (
        isinstance(cell, str) and cell.startswith("T")
    ):
        return 1
    value = int(cell)
    if value < 0:
        raise ValueError("A* requires nonnegative terrain costs")
    return value


def astar(
    grid: Grid,
    start: Position,
    goal: Position,
    *,
    heuristic: str = "manhattan",
    goal_cost: int | None = None,
) -> SearchResult:
    """Return an optimal path, its entry cost, and search counters.

    The start initially costs zero. Special labels normally cost one to
    enter; ``goal_cost=0`` implements question 2's free terminal square.
    ``expanded`` counts valid heap pops, including the goal; ``generated``
    counts heap pushes, including the start. Stale heap entries count in
    neither expansions nor neighbor processing.

    Manhattan is scaled by the smallest non-goal entry cost. If entering
    the goal is cheaper, its final step receives that cheaper lower bound.
    Thus a free goal uses max(Manhattan - 1, 0) * minimum_cost.
    """
    if heuristic not in ("zero", "manhattan"):
        raise ValueError("heuristic must be 'zero' or 'manhattan'")
    if goal_cost is not None and goal_cost < 0:
        raise ValueError("goal_cost must be nonnegative")
    if not grid or not grid[0]:
        return SearchResult([], -1, 0, 0)
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("grid must be rectangular")
    start, goal = tuple(start), tuple(goal)
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols) or grid[row][col] == "X":
            return SearchResult([], -1, 0, 0)

    terrain = {
        (row, col): _entry_cost(cell)
        for row, cells in enumerate(grid)
        for col, cell in enumerate(cells)
        if cell != "X"
    }
    terminal_cost = terrain[goal] if goal_cost is None else goal_cost
    terrain[goal] = terminal_cost
    minimum_cost = min((value for pos, value in terrain.items() if pos != goal), default=0)

    def estimate(pos: Position) -> int:
        if heuristic == "zero" or pos == goal:
            return 0
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        return (distance - 1) * minimum_cost + min(minimum_cost, terminal_cost)

    serial = count()
    initial_h = estimate(start)
    frontier = [(initial_h, initial_h, next(serial), 0, start)]
    best_g = {start: 0}
    parent: dict[Position, Position] = {}
    expanded, generated = 0, 1

    while frontier:
        _, _, _, current_g, current = heappop(frontier)
        if current_g != best_g[current]:
            continue
        expanded += 1
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return SearchResult(path, current_g, expanded, generated)

        row, col = current
        # Fixed order plus f/h/insertion tie-breaking makes runs reproducible.
        for neighbor in ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1)):
            if neighbor not in terrain:
                continue
            candidate_g = current_g + terrain[neighbor]
            if candidate_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = candidate_g
                parent[neighbor] = current
                h = estimate(neighbor)
                heappush(frontier, (candidate_g + h, h, next(serial), candidate_g, neighbor))
                generated += 1

    return SearchResult([], -1, expanded, generated)


def maze_path(start: Position, goal: Position, grid: Grid) -> list[Position]:
    """Question 1: include both endpoints, or return [] when unreachable."""
    return astar(grid, start, goal).path


def weighted_cost(start: Position, goal: Position, grid: Grid) -> int:
    """Question 2: the designated terminal square is free to enter."""
    return astar(grid, start, goal, goal_cost=0).cost
