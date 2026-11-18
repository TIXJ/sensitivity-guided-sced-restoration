"""Summarize every original/new small-case seed without test-case pooling."""
from pathlib import Path
import hashlib
import json
import statistics

REV = Path(__file__).resolve().parents[1]
RESULTS = REV / 'experiment_results/smallcase_multiseed_v1'


def archive_scope(publication):
    if publication is None:
        return 'These additions are local; immutable public v1.1.1 predates the added small-system seeds.'
    assert publication['version'] == 'v1.2.0' and publication['fresh_public_clone']
    expected = {f'case{case}_seed{seed}' for case in (39, 118) for seed in (1, 2)}
    assert set(publication['tests']) == expected
    assert all(row['completed'] and row['same_ids_pass_flags_steps_reasons']
               for row in publication['tests'].values())
    return ('The added scripts, selected actors and results are distributed in public v1.2.0. '
            'All four actors were checked through physical inference from a fresh public download; '
            'v1.1.1 is retained unchanged.')


def summarize_case(baseline, new_runs):
    assert baseline['complete'] and len(new_runs) == 2
    baseline_rows = [r for r in baseline['rows'] if r['mode'] == 'policy']
    ids = [r['sample_id'] for r in baseline_rows]
    assert len(ids) == len(set(ids)) and ids
    groups = [baseline_rows]
    for seed, run in zip((1, 2), new_runs):
        assert run['complete'] and run['seed'] == seed and run['contract'] == 'common_smallcase_v2'
        assert run['protocol']['updates'] == 100 and run['protocol']['partition_seed'] == 42
        assert [r['sample_id'] for r in run['rows']] == ids
        assert all(isinstance(r['passed'], bool) for r in run['rows'])
        groups.append(run['rows'])
    counts = [sum(r['passed'] for r in group) for group in groups]
    rates = [100 * n / len(ids) for n in counts]
    steps = [statistics.mean(r['steps'] for r in group) for group in groups]
    return {'training_seeds': [42, 1, 2], 'test_total': len(ids), 'test_sample_ids': ids,
            'counts': counts, 'pass_rates_pct': rates,
            'mean_pass_rate_pct': statistics.mean(rates),
            'sample_sd_percentage_points': statistics.stdev(rates),
            'all_case_mean_steps': steps, 'mean_of_seed_mean_steps': statistics.mean(steps),
            'sample_sd_seed_mean_steps': statistics.stdev(steps),
            'descriptive_seed_mean_runtime_sec': [statistics.mean(r['runtime_sec'] for r in group) for group in groups],
            'statistics_unit': 'training seed; sample SD uses divisor 3-1; identical test cases reused across seeds',
            'timing_scope': 'Descriptive separate measurements, not a new paired speed comparison.'}


def main():
    publication_path = REV / '_audit/smallcase_public_validation.json'
    publication = json.loads(publication_path.read_text(encoding='utf-8')) if publication_path.exists() else None
    summary = {'complete': True, 'cases': {}, 'source_sha256': {},
               'public_archive_scope': archive_scope(publication)}
    for case in (39, 118):
        base = REV / f'experiment_results/evaluation_contract_v2/case{case}_common.json'
        paths = [RESULTS / f'case{case}_seed{seed}/common_test.json' for seed in (1, 2)]
        baseline = json.loads(base.read_text(encoding='utf-8'))
        runs = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
        for seed, run in zip((1, 2), runs):
            receipt_path = RESULTS / f'case{case}_seed{seed}/execution_receipt.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            assert receipt['complete'] and receipt['outer_updates'] == 100
            assert receipt['actor_optimizer_steps'] > 0 and receipt['critic_optimizer_steps'] > 0
            assert receipt['test_passed'] == sum(r['passed'] for r in run['rows'])
            assert receipt['test_total'] == len(run['rows'])
            checkpoint = RESULTS / f'case{case}_seed{seed}/selected_validation_actor.pth'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == run['checkpoint_sha256']
            paths.extend([receipt_path, checkpoint])
        record = summarize_case(baseline, runs)
        assert record['test_total'] == (30 if case == 39 else 20)
        summary['cases'][str(case)] = record
        for path in [base, *paths]:
            summary['source_sha256'][str(path.relative_to(REV))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Small-System Training-Seed Completion', '',
             '| System | Seed 42 | Seed 1 | Seed 2 | Mean pass rate +/- sample SD |',
             '| --- | --- | --- | --- | --- |']
    for case, row in summary['cases'].items():
        counts = [f'{n}/{row["test_total"]}' for n in row['counts']]
        lines.append(f'| IEEE {case}-bus | ' + ' | '.join(counts) + f' | {row["mean_pass_rate_pct"]:.1f}% +/- {row["sample_sd_percentage_points"]:.1f} percentage points |')
    lines += ['', 'All runs use the existing partitions and original system-specific environments. New runs retain the original training budgets and validation-selection rule. Failed cases enter the step and timing summaries. The three rates reuse the same test cases; they are not independent samples of operating conditions.', '',
              'The original seed-42 records are retained without rewriting their measured runtimes. The new runtimes are descriptive and do not alter any of the 300-bus paired timing or cost results.', '',
              summary['public_archive_scope']]
    (RESULTS / 'Smallcase_Seed_Report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
