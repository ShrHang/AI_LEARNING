"""运行命令：python -m searchlab demo | maze | weighted | tree | treasure。

用 --input FILE 读取自定义 JSON，也可用 --input - 从标准输入读取。
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import sys
from .pathfinding import astar, maze_path, weighted_cost
from .games import sample_tree, tree_from_dict, tree_search, treasure_search
from .examples import MAZE, WEIGHTED, TREASURE_GRID, POSITIONS, VALUES


def matrix():
    return {a: {b: astar(TREASURE_GRID, p, q).cost for b, q in POSITIONS.items()}
            for a, p in POSITIONS.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['demo', 'maze', 'weighted', 'tree', 'treasure'], nargs='?', default='demo')
    p.add_argument('--input', help='JSON file, or - for stdin (maze/weighted/tree)')
    p.add_argument('--depth', type=int, choices=[2, 4], default=4)
    p.add_argument('--algorithm', choices=['minimax', 'alphabeta'], default='alphabeta')
    args = p.parse_args()
    data = {}
    if args.input:
        if args.task not in ['maze', 'weighted', 'tree']:
            p.error('--input supports maze, weighted and tree')
        if args.input == '-':
            data = json.load(sys.stdin)
        else:
            with open(args.input, encoding='utf-8') as f:
                data = json.load(f)
    if args.task in ('maze', 'weighted'):
        is_maze = args.task == 'maze'
        start_row, start_col = data.get('start', (0, 0))
        goal_row, goal_col = data.get('goal', (3, 4) if is_maze else (4, 4))
        grid = data.get('grid', MAZE if is_maze else WEIGHTED)
        solve = maze_path if is_maze else weighted_cost
        result = solve((start_row, start_col), (goal_row, goal_col), grid)
    elif args.task == 'tree':
        result = tree_search(tree_from_dict(data) if data else sample_tree(), algorithm=args.algorithm).value
    elif args.task == 'treasure':
        r = treasure_search(matrix(), VALUES, depth=args.depth, algorithm=args.algorithm)
        result = dict(best_target=r.target, best_value=r.value)
    else:
        d = matrix()
        result = dict(maze=maze_path((0, 0), (3, 4), MAZE),
                      weighted=weighted_cost((0, 0), (4, 4), WEIGHTED),
                      tree=tree_search(sample_tree()).value, distance_matrix=d,
                      treasure={str(h): {'best_target': (r := treasure_search(d, VALUES, depth=h)).target,
                                         'best_value': r.value} for h in [2, 4]})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
