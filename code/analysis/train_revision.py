"""Revision-only SAC action/reward ablations, without editing legacy scripts."""
import argparse
import sys
from revision_common import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['physics', 'full68', 'random', 'modes', 'keys'], default='physics')
    p.add_argument('--dimension', type=int, default=32)
    p.add_argument('--shifts', type=int, default=12)
    p.add_argument('--basis-seed', type=int, default=20260721)
    p.add_argument('--success-scale', type=float, default=1.)
    p.add_argument('--preflight', action='store_true')
    args, forwarded = p.parse_known_args()
    if not 1 <= args.dimension < 68 or not 0 <= args.shifts <= args.dimension:
        p.error('Require 1 <= dimension < 68 and 0 <= shifts <= dimension')
    if '--output-dir' not in forwarded:
        p.error('--output-dir is required')
    out = Path(forwarded[forwarded.index('--output-dir')+1]).resolve()
    if not out.is_relative_to(RESULTS.resolve()):
        p.error('Training outputs must be in experiment_results')
    if (out / 'training_results.json').exists() and not args.preflight:
        p.error('Completed run exists; use a new output directory')
    mod = load_aligned(forwarded + (['--preflight-only'] if args.preflight else []))
    configure_env(mod, args.family, args.dimension, args.shifts, args.basis_seed, args.success_scale)
    write(out / 'revision_config.json', dict(vars(args), argv=forwarded, original_code_sha256=sha(CODE / 'train_sac_n1_active_aligned.py')))
    mod.main()

if __name__ == '__main__':
    main()
