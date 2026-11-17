"""Audit existing failed-seed histories without replacing or retraining them."""
import math
from revision_common import RESULTS, read, sha, write


def nonfinite(value, prefix=''):
    if isinstance(value, dict):
        return [p for k,v in value.items() for p in nonfinite(v, prefix+'.'+k)]
    if isinstance(value, list):
        return [p for i,v in enumerate(value) for p in nonfinite(v, f'{prefix}[{i}]')]
    return [prefix] if isinstance(value, float) and not math.isfinite(value) else []


def main():
    result = {'scope': 'Audit of existing seed-2 records; no new optimization or model selection.', 'runs': {}}
    for method in ('sac', 'full68', 'ddpg', 'td3'):
        path = RESULTS/f'{method}_seed2/training_results.json'
        data = read(path)
        history, validation = data['history'], data['validation']
        assert [r['update'] for r in history] == list(range(1,101))
        best = max(validation, key=lambda r: (r['passed'], r['avg_reward']))
        metrics = ('avg_voltage_target_fraction','avg_action_execution_mismatch',
                   'active_action_l2','voltage_action_abs','action_abs_max','critic_loss','actor_loss',
                   'critic_updates','actor_updates','avg_final_n1_mw','avg_final_voltage_pu')
        result['runs'][method] = {'source_sha256': sha(path), 'training_updates': len(history),
                                  'all_validation_zero': all(r['passed']==0 for r in validation),
                                  'validation_selected': best,
                                  'nonfinite_recorded_values': nonfinite(data),
                                  'last_update_metrics': {k: history[-1].get(k) for k in metrics},
                                  'missing_historical_metrics': ['per-coordinate saturation fraction', 'complete per-term reward decomposition']}
    result['interpretation'] = (
        'Failed comparators complete the training budget but never accept a validation case. '
        'Their selected validation states have zero linear overload and remaining voltage violations. '
        'The recorded action magnitudes indicate boundary-seeking behavior, not a proven causal mechanism. '
        'Absence of recorded nonfinite values is not a proof that all internal tensors were finite. '
        'Null losses before updates begin are missing metrics, not numerical failures. '
        'Exact saturation fractions and full reward decomposition were not logged and cannot be reconstructed.')
    output = RESULTS/'evaluation_contract_v2/seed2_diagnostic.json'
    if output.exists():
        raise ValueError('Output exists')
    write(output, result)
    print({k: {'all_validation_zero':v['all_validation_zero'], 'selected_update':v['validation_selected']['update'],
               'nonfinite_count':len(v['nonfinite_recorded_values'])} for k,v in result['runs'].items()})


if __name__ == '__main__':
    main()
