"""Generate reproducible sample outputs and comparison measurements (stdlib only)."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from searchlab.pathfinding import astar
from searchlab.games import sample_tree, tree_search, treasure_search, treasure_root_values
from searchlab.examples import MAZE, WEIGHTED, TREASURE_GRID, POSITIONS, VALUES


def distance_matrix():
    return {a: {b: astar(TREASURE_GRID, p, q).cost for b, q in POSITIONS.items()}
            for a, p in POSITIONS.items()}


def measure(fn, repeats, batch):
    """One unmeasured warm-up; each sample is mean microseconds/call in a batch."""
    result = fn()
    raw = []
    for _ in range(repeats):
        start = perf_counter_ns()
        for _ in range(batch):
            fn()
        raw.append((perf_counter_ns() - start) / batch / 1000)
    return result, raw


def generated_map(seed, size=40):
    rng = random.Random(seed)
    grid = [[('X' if rng.random() < .23 else 1) for _ in range(size)]
            for _ in range(size)]
    # Preserve a guaranteed corridor so both heuristics solve the same reachable task.
    for i in range(size):
        grid[0][i] = grid[i][-1] = 1
    return grid


def run(out: Path, repeats=21, batch=20):
    out.mkdir(parents=True, exist_ok=True)
    matrix = distance_matrix()
    measurements, raw_rows = [], []

    def record(group, case, algorithm, order, fn):
        result, raw = measure(fn, repeats, batch)
        row = dict(group=group, case=case, algorithm=algorithm, order=order,
                   median_us=round(statistics.median(raw), 3),
                   min_us=round(min(raw), 3), max_us=round(max(raw), 3))
        if group == 'path':
            row.update(value=result.cost, nodes=result.expanded,
                       generated=result.generated, leaves=0, cutoffs=0)
        else:
            row.update(value=result.value, nodes=result.visited, generated=0,
                       leaves=result.leaves, cutoffs=result.cutoffs)
        measurements.append(row)
        for i, us in enumerate(raw):
            raw_rows.append(dict(group=group, case=case, algorithm=algorithm,
                                 order=order, repeat=i, mean_us=round(us, 6)))
        return result

    paths = {}
    for name, grid, goal, goal_cost in [('maze', MAZE, (3, 4), None),
                                      ('weighted', WEIGHTED, (4, 4), 0)]:
        for heuristic in ['zero', 'manhattan']:
            r = record('path', name, 'astar', heuristic,
                       lambda g=grid, e=goal, h=heuristic, c=goal_cost:
                       astar(g, (0, 0), e, heuristic=h, goal_cost=c))
            paths[f'{name}_{heuristic}'] = asdict(r)

    for seed in range(10):
        grid = generated_map(seed)
        for heuristic in ['zero', 'manhattan']:
            record('path', f'generated_{seed}', 'astar', heuristic,
                   lambda g=grid, h=heuristic: astar(g, (0, 0), (39, 39), heuristic=h))

    for algorithm in ['minimax', 'alphabeta']:
        for order in ['natural', 'reverse', 'ideal']:
            record('tree', 'sample', algorithm, order,
                   lambda a=algorithm, o=order: tree_search(sample_tree(), algorithm=a, order=o))

    for depth in [2, 4]:
        for algorithm in ['minimax', 'alphabeta']:
            for order in ['natural', 'reverse', 'greedy']:
                record('treasure', f'H{depth}', algorithm, order,
                       lambda d=depth, a=algorithm, o=order:
                       treasure_search(matrix, VALUES, depth=d, algorithm=a, order=o))

    # Depth/branching extension: a deterministic larger tree, independent of the map.
    from searchlab.games import TreeNode
    rng = random.Random(20261107)
    def build(depth):
        return TreeNode(value=rng.randint(-20, 20)) if not depth else TreeNode(
            children=[build(depth-1) for _ in range(3)])
    big_tree = build(6)
    for algorithm in ['minimax', 'alphabeta']:
        for order in ['natural', 'reverse', 'ideal']:
            record('tree', 'ternary_depth6', algorithm, order,
                   lambda a=algorithm, o=order: tree_search(big_tree, algorithm=a, order=o))

    examples = dict(paths=paths, distances=matrix,
                    root_values={str(d): treasure_root_values(matrix, VALUES, d) for d in [2, 4]},
                    treasure={str(d): asdict(treasure_search(matrix, VALUES, depth=d)) for d in [2, 4]},
                    tree=asdict(tree_search(sample_tree())))
    (out / 'samples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2)+'\n')
    for name, rows in [('measurements.csv', measurements), ('timings.csv', raw_rows)]:
        with (out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    processor=platform.processor(), seed=20261107, map_seeds=list(range(10)),
                    repeats=repeats, batch=batch, warmup=1,
                    timing='perf_counter_ns; microseconds per call, median of batch means',
                    tree_ideal='Oracle subtree-value ordering: timing includes ordering; node count excludes oracle preprocessing.',
                    astar_tie='f, h, insertion order; expanded includes popped goal, skips stale entries')
    (out / 'environment.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(f'Generated {len(measurements)} measurements and {len(raw_rows)} raw timing samples in {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT/'results')
    p.add_argument('--repeats', type=int, default=21)
    p.add_argument('--batch', type=int, default=20)
    args = p.parse_args()
    if args.repeats < 1 or args.batch < 1:
        p.error('repeats and batch must be positive')
    run(args.out, args.repeats, args.batch)
