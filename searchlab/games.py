"""对博弈树和宝藏分配问题进行 Minimax 与 Alpha-beta 搜索。

``visited`` 统计访问次数（含根节点），``leaves`` 统计终局或达到深度
上限时的求值次数。每次实际跳过至少一个兄弟分支，``cutoffs`` 加一。
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
    """题目样例，两个分支共用中间的 MAX 子树。"""
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[
        TreeNode(children=[max1, max2]),
        TreeNode(children=[max2, max3]),
    ])


def tree_from_dict(data: dict[str, Any]) -> TreeNode:
    """读取样例 JSON，可带或不带外层 tree_structure 字段。

    按题意从 MAX 根节点开始交替行动，因此无需读取 JSON 中的
    is_maximizing_player 字段。
    """
    node = data.get("tree_structure", data)
    return TreeNode(
        value=node.get("value"),
        children=[tree_from_dict(child) for child in node.get("children", [])],
    )


def tree_search(
    root: TreeNode, *, algorithm: str = "alphabeta", order: str = "natural"
) -> GameResult:
    """返回 MAX 根节点的最优得分。

    ``ideal`` 用预先求得的精确子树值排序，用于观察理想排序的剪枝效果。
    搜索计数不含预处理时访问的节点，整个函数的计时包含这部分开销。
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
    """更新选取目标宝藏后的状态。途经其他宝藏时不自动收集。"""
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
            # 优先尝试当前玩家即时收益最高的宝藏。
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
    """从 A 的第一步开始搜索，每次玩家行动消耗一层深度。

    最优值相同时保留最先访问的目标。不可达的宝藏不能选取，若当前
    玩家没有合法行动则结束搜索。题目地图中的宝藏均可互相到达。
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
    """用完整的 Minimax 搜索独立计算每个首步的精确值。

    各首步独立求值，避免将 Alpha-beta 剪枝返回的上界或下界当作精确值。
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
