"""Explicit current-result entry points; fresh output folders protect evidence.

Requires the author-held data/model/protocol files in their documented project
layout. This is not a claim of a clean-machine installation/reproduction test.
"""
import argparse
import importlib
from pathlib import Path
from revision_common import RESULTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('historical-prepare','historical-run','historical-analyze',
                                        'spatial-prepare','spatial-run','spatial-analyze'))
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    root=Path(args.output_root).resolve()
    if not root.is_relative_to(RESULTS.resolve()):
        parser.error('Use a fresh subdirectory of experiment_results')
    import remeasure_verified_dual as historical
    import remeasure_spatial_dual as spatial
    historical.DEST=root/'historical';spatial.DEST=root/'spatial'
    family,stage=args.stage.split('-')
    if stage in ('prepare','run'):
        getattr(historical if family=='historical' else spatial,stage)()
    else:
        module=importlib.import_module('analyze_verified_dual' if family=='historical' else 'analyze_spatial_dual')
        module.DEST=historical.DEST if family=='historical' else spatial.DEST
        module.main()


if __name__=='__main__':
    main()
