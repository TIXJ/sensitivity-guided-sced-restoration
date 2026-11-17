"""Reorder the same sequence conditions to balance first-period demand levels."""
import argparse
import copy
from collections import Counter
import reviewer_followup as base
from revision_common import RESULTS,read,write,sha

DEST=RESULTS/'reviewer_sequence_balanced_v2'
OLD=RESULTS/'reviewer_followup_v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'))
    args=parser.parse_args()
    hashes={str(p):sha(p) for p in (__file__,base.__file__,base.short.__file__,base.shared.__file__)}
    if args.stage=='prepare':
        if DEST.exists():raise RuntimeError('Preserve prior measurements')
        DEST.mkdir();protocol=copy.deepcopy(read(OLD/'protocol.json'))
        protocol.update(sequence_runner_hashes=hashes,
            original_sequence_conditions_sha256=sha(OLD/'warm_conditions.json'),
            original_sequence_results_sha256=sha(OLD/'warm.json'),
            sequence_origin='Pattern p starts at original period p modulo 5. Same numeric conditions and costs, cyclic reorder only. First-period demand counts are two at each level. Not another independent batch.')
        write(DEST/'protocol.json',protocol);return
    protocol=read(DEST/'protocol.json');assert protocol['sequence_runner_hashes']==hashes
    assert protocol['original_sequence_conditions_sha256']==sha(OLD/'warm_conditions.json')
    original=base.setup
    def balanced(samples,data):
        expected={r['sample_id']:r for r in read(OLD/'warm_conditions.json')['samples']}
        reordered=[]
        fields=('demand_mw','p_weights','q_weights','sample_cp0','sample_cp1','sample_cp2')
        for pattern in range(10):
            group=[r for r in samples if r['pattern']==pattern];offset=pattern%5
            for period,row in enumerate(group[offset:]+group[:offset]):
                assert all(row[f]==expected[row['sample_id']][f] for f in fields)
                item=copy.deepcopy(row);item['original_period']=row['period'];item['period']=period
                reordered.append(item)
        samples[:]=reordered
        assert set(Counter(r['demand_mw'] for r in samples if r['period']==0).values())=={2}
        write(DEST/'warm_conditions.json',{'samples':samples,'same_numeric_conditions_as':str(OLD/'warm_conditions.json')})
        return original(samples,data)
    base.DEST=DEST;base.setup=balanced
    base.warm(protocol)


if __name__=='__main__':main()
