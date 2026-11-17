"""Resolve archived small-system model dependencies without changing old code."""
import argparse
import importlib.util
import runpy
from revision_common import *

def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--case',type=int,choices=[39,118],required=True)
    a,remaining=p.parse_known_args()
    if a.case==39:
        spec=importlib.util.spec_from_file_location('train_gat39_v1',HERE/'legacy_dependencies/train_gat39_v1.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules['train_gat39_v1']=module
        spec.loader.exec_module(module)
    target=CODE/f'eval_case{a.case}_action_diagnosis.py'
    sys.argv=[str(target),*remaining]
    runpy.run_path(str(target),run_name='__main__')

if __name__=='__main__': main()
