"""Matched single-seed retraining at compatible tighter bus-voltage limits."""
import os
import subprocess
import sys
from revision_common import RESULTS,HERE,DATASET,SPLIT,GAT,CODE,read,write,sha,load_aligned


def main():
    os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))
    out=RESULTS/'tight_sac_094_108_seed42_v1'
    protocol={'seed':42,'updates':100,'bus_voltage_band':[.94,1.08],'script_sha256':sha(__file__),
              'scope':'Original retained training/validation IDs and 32D/40MW action space; no new feasibility screening or reward retuning',
              'hashes':{str(p):sha(p) for p in [DATASET,SPLIT,GAT,CODE/'train_sac_n1_active_aligned.py',CODE/'train_rlv35_complete.py']}}
    protocol_path=RESULTS/'tight_policy_protocol.json'
    if protocol_path.exists(): assert read(protocol_path)==protocol
    else: write(protocol_path,protocol)
    if not (out/'training_results.json').exists():
        if out.exists() and any(out.iterdir()):
            import shutil
            from datetime import datetime
            shutil.copytree(out,RESULTS/('interrupted_tight_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
        mod=load_aligned(['--seed','42','--updates','100','--output-dir',str(out)])
        mod.core.RL_VMIN_PU,mod.core.RL_VMAX_PU=.94,1.08
        write(out/'tight_voltage_config.json',protocol)
        mod.main()
    output=RESULTS/'tight_sac_094_108_seed42_v1_full.json'
    if not output.exists() or not read(output).get('complete'):
        subprocess.run([sys.executable,str(HERE/'evaluate_revision.py'),'--checkpoint',str(out/'best_actor.pth'),
                        '--vmin','.94','--vmax','1.08','--output',str(output)],check=True)


if __name__=='__main__': main()
