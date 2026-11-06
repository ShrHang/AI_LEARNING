# 实验二：搜索算法

仓库：[ShrHang/AI_LEARNING](https://github.com/ShrHang/AI_LEARNING) · 本实验分支：[project2](https://github.com/ShrHang/AI_LEARNING/tree/project2)

实现四方向 A*、Minimax 与 Alpha-beta，完成迷宫、加权地形、博弈树和宝藏分配四题。

## 快速运行

算法、命令行、核验和实验脚本仅依赖 **Python 3.10+ 标准库**，无需安装第三方库。在项目根目录执行：

```bash
python3 -m searchlab demo
python3 scripts/verify.py
```

官方样例结果：

| 题目 | 结果 |
| --- | --- |
| 迷宫 | 7步路径，含起终点共8个坐标 |
| 加权地形 | 最小代价7 |
| 博弈树 | 最佳得分5 |
| 宝藏 H=2 | `{"best_target": "T2", "best_value": -1}` |
| 宝藏 H=4 | `{"best_target": "T3", "best_value": 3}` |

## 各题输入输出

```bash
python3 -m searchlab maze --input examples/maze.json
python3 -m searchlab weighted --input examples/weighted.json
python3 -m searchlab tree --input examples/tree.json
python3 -m searchlab treasure --depth 2 --algorithm minimax
python3 -m searchlab treasure --depth 4 --algorithm alphabeta
```

前三题可通过 `--input -` 从标准输入读取 JSON；不传 `--input` 则运行官方样例。

迷宫与加权输入为 `{"start": [row, col], "goal": [row, col], "grid": [...]}`。地图可使用数字、`X` 和特殊标签；特殊标签须写成 JSON 字符串。迷宫输出坐标数组（Python 接口为元组列表），无解输出 `[]`；加权输出整数，无解为 `-1`。树输入支持带或不带 `tree_structure` 的嵌套节点，叶子为 `{"value": 3}`，非叶为 `{"children": [...]}`。根为 MAX，逐层交替；`is_maximizing_player` 标注不改变题目规定的交替规则。宝藏命令使用题目固定地图。

Python 接口：

```python
from searchlab.pathfinding import astar, maze_path, weighted_cost
from searchlab.games import TreeNode, tree_search, treasure_search

path = maze_path((0, 0), (3, 4), grid)
cost = weighted_cost((0, 0), (4, 4), weighted_grid)
result = astar(grid, (0, 0), (3, 4), heuristic="manhattan")
# result.path / result.cost / result.expanded / result.generated
value = tree_search(root, algorithm="alphabeta", order="natural").value
```

树排序支持 `natural / reverse / ideal`；宝藏排序支持 `natural / reverse / greedy`。宝藏状态显式保存双方位置、剩余宝藏、累计收益和行动玩家。

## 关键规则

- 题一移动一格代价1，起点初始代价0。
- **题二终点进入代价为0**。采用 `h(n)=max(曼哈顿距离-1,0)×最小非终点格代价`；普通曼哈顿距离在终点邻格可能高估。
- 通用 A* 默认标签进入代价1，可通过 `goal_cost=0` 启用免费终点。非负权允许为0；最小代价为0时启发函数自然退化。
- 题四 A、B、T 格进入代价均为1，经过宝藏不自动收集，玩家不阻挡。
- 题四四回合强制各取两件，允许负收益。`H` 按单次行动计数，叶子评价为当前 `Score_A-Score_B`。
- 同值首步保留当前顺序中最先遇到者；排序可能改变同值选择，但不改变最优收益差。

## 验证与复现实验

```bash
python3 scripts/verify.py
python3 scripts/experiment.py
```

核验脚本只验证核心算法：独立 Dijkstra 对照随机地图，完整递归对照随机树，全排列与自底向上折叠对照宝藏博弈。固定种子 `20261107`，共1836项检查，不依赖测试框架。

实验生成：

- `results/samples.json`：路径、距离矩阵、官方结果与各首步精确值。
- `results/measurements.csv`：48组算法与策略对照、节点数、耗时摘要。
- `results/timings.csv`：1008条原始计时样本。
- `results/environment.json`：Python、平台、种子、采样参数与统计说明。

默认一次预热，21组采样，每组20次，记录每次调用的平均微秒数，再取组间中位数。可用 `--repeats 21 --batch 20 --out results` 调整。耗时包含函数内的地图预处理与排序；小样例树还包含样例树构造，比较只针对相同场景。墙钟时间会随环境变化，最优值与节点数在固定输入/策略下确定。

统计口径：

- A* `expanded` 为有效出堆数，含起点与终点，跳过过期条目；`generated` 为入堆次数，含起点，可能含同一格的更优条目。
- 博弈 `visited` 含根和叶，重复路径分别计数；`leaves` 为终止或深度截断评价次数；`cutoffs` 只记录确实跳过后续兄弟的剪枝事件，不表示跳过的节点总数。
- `ideal` 使用精确子树值排序，是理想排序对照；节点数不计其预处理，整体耗时计入。它用于展示剪枝上限，不能视为免费优化。
- `greedy` 按行动玩家即时收益降序排序，不查看未来真实值，是可实际应用的优化。
- 每个首步的精确收益由独立完整 Minimax 求值，避免将 Alpha-beta 的界误当成精确值。

## 结果绘图

第三方依赖仅用于结果绘图，核心算法与实验脚本无需安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/plot_results.py
```

中文图表优先使用系统 Noto Sans CJK 字体（Linux常见路径 `/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc`）；缺少时应安装 Noto CJK 字体。


## 文件结构

```text
searchlab/          算法、官方地图与命令行
examples/           可直接输入的JSON样例
scripts/            核验、实验与结果绘图
results/            可复现实验数据与图表
requirements.txt    结果绘图依赖
```
