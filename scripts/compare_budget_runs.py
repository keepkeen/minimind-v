"""Strict paired comparison with image/document-cluster bootstrap intervals.

Cluster IDs should identify independent scenes/documents; the evaluator defaults
to source-image identities. This is not an official task evaluator or causal proof.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_predictions(path):
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row.get('id')
        if not isinstance(identity, str) or not identity or identity in rows:
            raise ValueError('Prediction IDs must be unique nonempty strings')
        rows[identity] = row
    if not rows:
        raise ValueError('Prediction file is empty')
    return rows


def paired_comparison(baseline, candidate, metric='normalized_em', bootstrap=2000, seed=42):
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    if not baseline or set(baseline) != set(candidate):
        raise ValueError('Paired comparison requires exactly the same nonempty set of IDs')
    pairs, groups = [], {}
    for identity in sorted(baseline):
        left, right = baseline[identity], candidate[identity]
        # Do not silently join old/unrelated runs or image interventions to changed targets.
        for field in ('question', 'answers', 'source_images', 'group_id', 'category'):
            if field not in left or field not in right or left[field] != right[field]:
                raise ValueError(f'{identity}: missing or mismatched {field}; rerun with the current evaluator')
        if not isinstance(left['group_id'], str) or not left['group_id']:
            raise ValueError('group_id must be a nonempty string')
        values = [left.get(metric), right.get(metric)]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f'{identity}: {metric} must be a finite score in [0,1]')
        delta = values[1] - values[0]
        groups.setdefault(left['group_id'], []).append(delta)
        pairs.append((left, right, delta))
    sums = np.array([sum(values) for values in groups.values()], dtype=np.float64)
    counts = np.array([len(values) for values in groups.values()], dtype=np.int64)
    rng, sampled = np.random.default_rng(seed), []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(groups), size=len(groups))
        sampled.append(float(sums[indices].sum() / counts[indices].sum()))
    delta = float(sums.sum() / counts.sum())
    result = {
        'metric': metric, 'samples': len(pairs), 'independent_groups': len(groups),
        'baseline_mean': float(np.mean([left[metric] for left, _, _ in pairs])),
        'candidate_mean': float(np.mean([right[metric] for _, right, _ in pairs])),
        'candidate_minus_baseline': delta,
        'cluster_bootstrap_ci95': [float(v) for v in np.quantile(sampled, [0.025, 0.975])],
        'improved': sum(d > 0 for _, _, d in pairs), 'regressed': sum(d < 0 for _, _, d in pairs),
        'unchanged': sum(d == 0 for _, _, d in pairs), 'bootstrap_replicates': bootstrap, 'seed': seed,
        'estimator': 'example-weighted paired difference; resample entire image/document groups',
        'warnings': ['Intervals describe this dataset, not training-seed variation. '
                     'A changed response or actual-vs-blank gap alone does not prove grounded reasoning.'],
    }
    if len(groups) < 20:
        result['warnings'].append('Fewer than 20 independent groups: interval may be unstable or degenerate.')
    if metric == 'normalized_em':
        correct = [pair for pair in pairs if pair[0][metric] == 1]
        result['baseline_correct_count'] = len(correct)
        result['retention_on_baseline_correct'] = (sum(right[metric] == 1 for _, right, _ in correct) / len(correct)) if correct else None
    result['by_category'] = {}
    for category in sorted({left['category'] for left, _, _ in pairs}):
        subset = [d for left, _, d in pairs if left['category'] == category]
        result['by_category'][category] = {'samples': len(subset), 'candidate_minus_baseline': float(np.mean(subset))}
    result['conditions'] = {'baseline': sorted({left.get('image_condition', 'unknown') for left, _, _ in pairs}),
                            'candidate': sorted({right.get('image_condition', 'unknown') for _, right, _ in pairs})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metric', choices=['normalized_em', 'anls_style'], default='normalized_em')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new filename')
    result = paired_comparison(read_predictions(args.baseline), read_predictions(args.candidate),
                               args.metric, args.bootstrap, args.seed)
    result['source_files'] = {'baseline': str(args.baseline), 'candidate': str(args.candidate)}
    result['run_metadata'] = {}
    for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        summary = path.with_suffix('.summary.json')
        result['run_metadata'][name] = json.loads(summary.read_text()).get('metadata') if summary.exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
