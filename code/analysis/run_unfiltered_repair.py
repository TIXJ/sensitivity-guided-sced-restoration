"""Remove only the legacy evaluator's test-selection filter, preserving its solver."""
import argparse
import sys
from revision_common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    a=p.parse_args()
    source=CODE/'eval_case300_deterministic_repair.py'
    text=source.read_text(encoding='utf-8')
    old='ids = [idx for idx in scope_ids if idx in selected_ids]'
    if text.count(old)!=1: raise RuntimeError('Legacy evaluator changed; review selection patch')
    text=text.replace(old,'ids = list(scope_ids)')
    text=text.replace('"selection_status": selected_status,','"selection_status": "unfiltered_upstream_test",')
    if Path(a.output).exists(): p.error('Output exists')
    sys.argv=[str(source),'--candidate-dataset',str(CANDIDATES),'--candidate-split',str(CANDIDATE_SPLIT),'--screen-report',str(ROOT/'paper_results/case300_fixed_slack_v1_screen.json'),'--gat-checkpoint',str(GAT),'--scope','test','--output',a.output]
    exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})

if __name__=='__main__': main()
