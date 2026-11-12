# 实验二：搜索算法

用 A* 求解迷宫和加权地形，用 Minimax 与 Alpha-beta 搜索博弈树。宝藏分配题先用 A* 计算移动代价，再用博弈搜索选择首步。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `searchlab/pathfinding.py` | A*、迷宫寻路和加权地形接口 |
| `searchlab/games.py` | Minimax、Alpha-beta 和宝藏博弈 |
| `searchlab/examples.py` | 内置地图、关键位置和宝藏价值 |
| `examples/` | 可传给命令行的 JSON 样例 |
| `scripts/` | 正确性核验、性能实验和结果绘图 |
| `results/` | 实验数据与图表 |

## 运行样例

在项目根目录使用 **Python 3.10+**。算法、核验和性能实验只用标准库。

```bash
python -m searchlab demo
```

也可以单独运行各题：

```bash
python -m searchlab maze --input examples/maze.json
python -m searchlab weighted --input examples/weighted.json
python -m searchlab tree --input examples/tree.json
python -m searchlab treasure --depth 2 --algorithm minimax
python -m searchlab treasure --depth 4 --algorithm alphabeta
```

前三题省略 `--input` 时使用内置样例，也可以用 `--input -` 从标准输入读取 JSON。宝藏题使用内置地图。

## 复现实验

```bash
python scripts/verify.py
python scripts/experiment.py
```

`verify.py` 使用固定随机种子和独立参考算法核对结果。`experiment.py` 默认写入 `results/`，可用 `--out` 指定其他输出目录。默认每种配置预热一次，再计时 21 组，每组调用 20 次，可用 `--repeats` 和 `--batch` 调整。固定输入和配置下，搜索结果与节点数可以复现，耗时会随运行环境变化。

| 输出文件 | 内容 |
| --- | --- |
| `results/samples.json` | 样例路径、代价、距离矩阵和博弈结果 |
| `results/measurements.csv` | 各配置的搜索计数和耗时汇总 |
| `results/timings.csv` | 每组计时记录 |
| `results/environment.json` | 运行环境、随机种子和计时参数 |

生成图表时，需要先安装 `requirements.txt` 中的包：

```bash
pip install -r requirements.txt
python scripts/plot_results.py
```

图表写入 `results/figures/`。如果系统字体无法显示全部中文标签，脚本会使用英文标签。
