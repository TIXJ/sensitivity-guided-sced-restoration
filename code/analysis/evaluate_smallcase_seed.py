"""Physically evaluate a newly trained small-system validation-selected actor."""
import argparse
from pathlib import Path


def seed_checkpoint(case, seed, results):
    if case not in (39, 118) or seed not in (1, 2):
        raise ValueError('Additional small-system runs are cases 39/118 and seeds 1/2.')
    return Path(results) / f'smallcase_multiseed_v1/case{case}_seed{seed}/selected_validation_actor.pth'


def main():
    from revision_common import RESULTS, CODE, sha, write
    from evaluate_smallcases_common import load_case, evaluate
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=(39, 118), type=int, required=True)
    parser.add_argument('--seed', choices=(1, 2), type=int, required=True)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.max_samples < 0 or Path(args.output).exists():
        parser.error('Use a nonnegative sample limit and a new output filename.')
    checkpoint = seed_checkpoint(args.case, args.seed, RESULTS)
    if not checkpoint.is_file():
        parser.error('Missing distributed selected actor: ' + str(checkpoint))
    mod, samples, ids, env, _original_action, dataset, _original_checkpoint = load_case(args.case)
    if args.max_samples:
        ids = ids[:args.max_samples]
    saved = torch.load(checkpoint, map_location=mod.DEVICE, weights_only=False)
    assert saved['seed'] == args.seed
    actor = mod.GaussianActor(env._get_obs_dim(), env.action_dim).to(mod.DEVICE)
    actor.load_state_dict(saved['actor'])
    actor.eval()
    def action(obs):
        raw = actor.select_action(obs, **({'det': True} if args.case == 39 else {'deterministic': True}))
        low, high = env.get_action_bounds()
        return low + (raw + 1.) * .5 * (high - low)
    obs = env.reset(ids[0])
    action(obs)
    payload = {'complete': False, 'contract': 'common_smallcase_v2', 'case': args.case, 'seed': args.seed,
               'selected_update': saved['update'], 'checkpoint_sha256': sha(checkpoint),
               'input_hashes': {str(p): sha(p) for p in (dataset, checkpoint, CODE / f'train_rl{args.case}_v1.py',
                                                        CODE / 'smallcase_evaluation_contract.py', Path(__file__))},
               'rows': []}
    for i, sid in enumerate(ids):
        row = evaluate(env, mod, args.case, sid, 'policy', action)
        row.update(case=args.case, sid=sid, sample_id=str(samples[sid].get('sample_id', samples[sid].get('id', sid))),
                   demand_mw=samples[sid]['demand_mw'])
        payload['rows'].append(row)
        write(args.output, payload)
        print(f'case={args.case} seed={args.seed} {i+1}/{len(ids)} pass={row["passed"]} steps={row["steps"]}', flush=True)
    payload['complete'] = True
    write(args.output, payload)


if __name__ == '__main__':
    main()
