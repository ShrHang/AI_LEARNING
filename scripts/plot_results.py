"""Build static figures from recorded measurements; no new experiments are run."""
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from searchlab.examples import MAZE, WEIGHTED

FONT = Path('/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc')
if FONT.exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.unicode_minus': False,
                     'figure.facecolor': 'white'})
BLUE, ORANGE, GREEN = '#2874a6', '#d88032', '#39856b'


def main():
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
        ax.set(xticks=range(5), yticks=range(5), xlabel='列 col', ylabel='行 row',
               title=('题一：7步最短路径' if name == 'maze' else '题二：总代价7，E免费'))
    ax = axes[2]
    ax.bar([-.18, .82], [data['paths'][n+'_zero']['expanded'] for n in ['maze','weighted']],
           width=.35, color=BLUE, label='h=0')
    ax.bar([.18, 1.18], [data['paths'][n+'_manhattan']['expanded'] for n in ['maze','weighted']],
           width=.35, color=ORANGE, label='可采纳曼哈顿')
    ax.set(xticks=[0,1], xticklabels=['题一','题二'], ylabel='有效出堆数（含终点）', title='相同最优值，更少扩展')
    for container in ax.containers: ax.bar_label(container, padding=3)
    ax.set_ylim(0, 22)
    ax.legend(fontsize=8)
    fig.savefig(out/'pathfinding.png', dpi=190); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.3), layout='constrained')
    for ax, group, case, title in zip(axes.flat,
            ['tree','tree','treasure','treasure'],
            ['sample','ternary_depth6','H2','H4'],
            ['题三样例','拓展：三叉树，深度6','宝藏 H=2','宝藏 H=4']):
        subset = [r for r in rows if r['group']==group and r['case']==case]
        orders = ['natural','reverse','ideal'] if group=='tree' else ['natural','reverse','greedy']
        for i, algo in enumerate(['minimax','alphabeta']):
            y = [int(next(r for r in subset if r['algorithm']==algo and r['order']==o)['nodes']) for o in orders]
            ax.bar([j + (i-.5)*.36 for j in range(3)], y, width=.36,
                   color=[BLUE,ORANGE][i],label=['Minimax','Alpha-beta'][i])
        ax.set(xticks=range(3), xticklabels=orders, ylabel='递归访问节点数', title=title)
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
    ax.set(xticks=range(4), xticklabels=names, ylabel='各首步的精确 Minimax 收益差',
           title='深度改变选择：H=2选T2，H=4选T3', ylim=(-8.3,4.6))
    ax.legend()
    fig.savefig(out/'root_values.png', dpi=190); plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax, key, ylabel in zip(axes,['nodes','median_us'],['有效出堆数','耗时中位数（微秒/次）']):
        for order,color in [('zero',BLUE),('manhattan',ORANGE)]:
            selected=[r for r in rows if r['group']=='path' and r['case'].startswith('generated_') and r['order']==order]
            ax.plot(range(10),[float(r[key]) for r in selected],marker='o',color=color,label=order)
        ax.set(xlabel='地图随机种子',ylabel=ylabel,xticks=range(10),title='40×40随机障碍地图（保留可达走廊）')
        ax.legend(fontsize=8)
    fig.savefig(out/'generated_maps.png', dpi=190); plt.close(fig)
    print(f'Generated 4 figures in {out}')


if __name__=='__main__': main()
