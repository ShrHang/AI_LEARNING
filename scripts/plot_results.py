"""读取数据绘图"""
import csv
import json
from pathlib import Path
from string import ascii_letters, digits
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from searchlab.examples import MAZE, WEIGHTED

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.unicode_minus': False,
                     'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2874a6', '#d88032', '#39856b'

LABELS = {
    'column': ('列', 'Column'),
    'row': ('行', 'Row'),
    'maze_title': ('题一：7 步最短路径', 'Maze: 7-step shortest path'),
    'weighted_title': ('题二：总代价 7，E 免费', 'Terrain: cost 7, free goal'),
    'manhattan': ('可采纳曼哈顿', 'Manhattan heuristic'),
    'maze': ('题一', 'Maze'),
    'weighted': ('题二', 'Terrain'),
    'expanded_with_goal': ('有效出堆数（含终点）', 'Valid heap pops (incl. goal)'),
    'path_comparison': ('相同最优值，更少扩展', 'Same optimum, fewer expansions'),
    'sample': ('题三样例', 'Sample game tree'),
    'ternary_depth6': ('拓展：三叉树，深度 6', 'Ternary tree, depth 6'),
    'H2': ('宝藏 H=2', 'Treasure game, H=2'),
    'H4': ('宝藏 H=4', 'Treasure game, H=4'),
    'visited': ('递归访问节点数', 'Nodes visited'),
    'root_value': ('各首步的精确 Minimax 收益差', 'Exact minimax value by first move'),
    'depth_comparison': ('深度改变选择：H=2 选 T2，H=4 选 T3', 'Best first move: T2 at H=2, T3 at H=4'),
    'expanded': ('有效出堆数', 'Valid heap pops'),
    'median_time': ('耗时中位数（微秒/次）', 'Median time (us/call)'),
    'seed': ('地图随机种子', 'Map seed'),
    'generated_maps': ('40×40 随机障碍地图（保留可达走廊）', '40x40 maps with a guaranteed corridor'),
}


def find_chinese_font() -> str | None:
    """检查 Matplotlib 实际选用的字体文件能否显示全部图表标签。"""
    required = {ord(char) for char in ascii_letters + digits + ' .-='}
    required.update(ord(char) for zh, _ in LABELS.values() for char in zh)
    han = {code for code in required if 0x4E00 <= code <= 0x9FFF}

    def covers_labels(path: str) -> bool:
        charmap = FT2Font(path).get_charmap()
        # 排除用同一个占位字形映射多个汉字的系统兜底字体。
        return required.issubset(charmap) and len({charmap[code] for code in han}) == len(han)

    candidates = sorted({(font.name, font.fname) for font in font_manager.fontManager.ttflist})
    for family, path in candidates:
        try:
            if not covers_labels(path):
                continue
            font = font_manager.FontProperties(family=[family])
            resolved = font_manager.findfont(font, fallback_to_default=False)
            if resolved == path or covers_labels(resolved):
                return family
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def main():
    family = find_chinese_font()
    labels = {key: zh if family else en for key, (zh, en) in LABELS.items()}
    plt.rcParams['font.family'] = family or 'DejaVu Sans'
    data = json.loads((ROOT/'results/samples.json').read_text())
    rows = list(csv.DictReader((ROOT/'results/measurements.csv').open()))
    out = ROOT/'results/figures'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout='constrained')
    for ax, name, grid in zip(axes[:2], ['maze', 'weighted'], [MAZE, WEIGHTED]):
        values = [[8 if c == 'X' else (c if isinstance(c, int) else 0) for c in row] for row in grid]
        ax.imshow(values, cmap='Blues', vmin=-1, vmax=8)
        for r, row in enumerate(grid):
            for c, label in enumerate(row):
                ax.text(c, r, str(label), ha='center', va='center', fontsize=10,
                        color='white' if label == 'X' or isinstance(label, int) and label >= 6 else '#26384a')
        path = data['paths'][name+'_manhattan']['path']
        ax.plot([p[1] for p in path], [p[0] for p in path], color=ORANGE, linewidth=2, marker='o', markersize=4)
        ax.set(xticks=range(5), yticks=range(5), xlabel=labels['column'], ylabel=labels['row'],
               title=labels[name + '_title'])
    ax = axes[2]
    ax.bar([-.18, .82], [data['paths'][n+'_zero']['expanded'] for n in ['maze','weighted']],
           width=.35, color=BLUE, label='h=0')
    ax.bar([.18, 1.18], [data['paths'][n+'_manhattan']['expanded'] for n in ['maze','weighted']],
           width=.35, color=ORANGE, label=labels['manhattan'])
    ax.set(xticks=[0,1], xticklabels=[labels['maze'], labels['weighted']],
           ylabel=labels['expanded_with_goal'], title=labels['path_comparison'])
    for container in ax.containers: ax.bar_label(container, padding=3)
    ax.set_ylim(0, 22)
    ax.legend(fontsize=8)
    fig.savefig(out/'pathfinding.png', dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), layout='constrained')
    for ax, group, case in zip(axes.flat,
            ['tree','tree','treasure','treasure'],
            ['sample','ternary_depth6','H2','H4']):
        subset = [r for r in rows if r['group']==group and r['case']==case]
        orders = ['natural','reverse','ideal'] if group=='tree' else ['natural','reverse','greedy']
        for i, algo in enumerate(['minimax','alphabeta']):
            y = [int(next(r for r in subset if r['algorithm']==algo and r['order']==o)['nodes']) for o in orders]
            ax.bar([j + (i-.5)*.36 for j in range(3)], y, width=.36,
                   color=[BLUE,ORANGE][i],label=['Minimax','Alpha-beta'][i])
        ax.set(xticks=range(3), xticklabels=orders, ylabel=labels['visited'], title=labels[case])
        for container in ax.containers: ax.bar_label(container, padding=2, fontsize=8)
        ax.set_ylim(0, max(int(r['nodes']) for r in subset)*1.35)
        ax.legend(fontsize=8)
    fig.savefig(out/'game_nodes.png', dpi=190); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8), layout='constrained')
    names = list(data['root_values']['2'])
    for i, depth in enumerate(['2', '4']):
        vals=[data['root_values'][depth][n] for n in names]
        bars=ax.bar([j+(i-.5)*.35 for j in range(4)], vals, width=.35,
                    color=[BLUE,ORANGE][i], label='H='+depth)
        ax.bar_label(bars, padding=3)
    ax.axhline(0,color='#73818e',linewidth=.7)
    ax.set(xticks=range(4), xticklabels=names, ylabel=labels['root_value'],
           title=labels['depth_comparison'], ylim=(-8.3,4.6))
    ax.legend()
    fig.savefig(out/'root_values.png', dpi=190); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax, key, ylabel in zip(axes,['nodes','median_us'],[labels['expanded'], labels['median_time']]):
        for order,color in [('zero',BLUE),('manhattan',ORANGE)]:
            selected=[r for r in rows if r['group']=='path' and r['case'].startswith('generated_') and r['order']==order]
            ax.plot(range(10),[float(r[key]) for r in selected],marker='o',color=color,label=order)
        ax.set(xlabel=labels['seed'], ylabel=ylabel, xticks=range(10), title=labels['generated_maps'])
        ax.legend(fontsize=8)
    fig.savefig(out/'generated_maps.png', dpi=190); plt.close(fig)
    print(f'Generated 4 figures in {out}')


if __name__=='__main__': main()
