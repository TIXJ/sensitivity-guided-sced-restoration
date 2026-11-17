"""Frozen-policy range diagnostic on a preselected validation subset only."""
import argparse
import sys

import evaluate_revision
from revision_common import CANDIDATES, CANDIDATE_SPLIT, DATASET, RESULTS, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radius', type=float, choices=(40,200,400,600), required=True)
    parser.add_argument('--positive-controls', action='store_true',
                        help='Two screened validation cases to check the unchanged policy/evaluator.')
    args = parser.parse_args()
    data = read(CANDIDATES)
    samples = [s for s in data['samples'] if s.get('is_complete_sced',True)]
    original = read(CANDIDATE_SPLIT)
    selected = []
    retained = {s['sample_id'] for s in read(DATASET)['samples']}
    demands = (16000.,16400.) if args.positive_controls else (16400.,16600.,16800.)
    for demand in demands:
        choices = sorted((i for i in original['val_ids'] if samples[i]['demand_mw']==demand
                          and (not args.positive_controls or samples[i]['sample_id'] in retained)),
                         key=lambda i:samples[i]['sample_id'])
        selected.append(choices[0])
    assert not set(selected).intersection(original['train_ids']+original['test_ids'])
    split = dict(original, val_ids=selected,
                 diagnostic_note='First validation ID at each of 16.4/16.6/16.8 GW; not a new train/test split.')
    suffix = '_positive_controls' if args.positive_controls else ''
    if args.positive_controls:
        split['diagnostic_note']='First screened validation ID at 16.0 and 16.4 GW; positive controls, not pooled with the three stress cases.'
    path = RESULTS/f'optimization_diagnostics/frozen_range_validation_subset{suffix}.json'
    if path.exists():
        assert read(path)==split, 'Preserve the original pilot selection'
    else:
        write(path,split)
    evaluate_revision.CANDIDATE_SPLIT = path
    output = RESULTS/f'optimization_diagnostics/frozen_range_{int(args.radius)}{suffix}.json'
    sys.argv = [str(evaluate_revision.__file__), '--scope','val', '--radius',str(args.radius),
                '--output',str(output)]
    print('FROZEN POLICY DIAGNOSTIC: no retraining, validation subset only; '
          'range changes are not reported as performance of the original policy interface.',flush=True)
    evaluate_revision.main()


if __name__=='__main__':
    main()
