"""用于单位代价迷宫和加权地形的四方向 A* 搜索。"""

from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count


Position = tuple[int, int]
Grid = Sequence[Sequence[int | str]]


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
    """返回最优路径、累计进入代价和搜索计数。

    起点的初始代价为 0。进入特殊标记格的默认代价为 1，设置
    ``goal_cost=0`` 可实现题二的免费终点。
    ``expanded`` 统计有效出堆次数（含终点），``generated`` 统计
    入堆次数（含起点）。过期条目出堆后直接跳过。

    用非终点格的最小进入代价缩放曼哈顿距离。若终点进入代价更低，
    最后一步用较低的代价作为下界。终点免费时，启发值为
    max(曼哈顿距离 - 1, 0) * minimum_cost。
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
    start_row, start_col = start
    goal_row, goal_col = goal
    start, goal = (start_row, start_col), (goal_row, goal_col)
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
        # 固定邻居遍历顺序，堆按 f、h 和入堆顺序排序，便于复现结果。
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
    """题一：路径包含起点和终点，无解时返回空列表。"""
    return astar(grid, start, goal).path


def weighted_cost(start: Position, goal: Position, grid: Grid) -> int:
    """题二：进入指定终点的代价为 0。"""
    return astar(grid, start, goal, goal_cost=0).cost
