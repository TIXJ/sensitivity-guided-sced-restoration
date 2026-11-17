"""Assemble actual modified data, scripts, frozen matrices and settings for review."""
import shutil
import zipfile
import importlib.metadata
from revision_common import *

def main():
    import numpy as np
    import pandapower as pp
    mod=load_aligned()
    data=read(CANDIDATES); samples=[s for s in data['samples'] if s.get('is_complete_sced',True)]
    env=mod.CleanN1Env(samples,read(CANDIDATE_SPLIT)['train_ids'],data)
    env.reset(env.sample_ids[0])
    package=REV/'reproducibility_package_v2'; package.mkdir(exist_ok=True)
    pp.to_json(env._ac_net_template,str(package/'case300_network.json'))
    np.savez_compressed(package/'security_matrices.npz',S=env.dcnet.S,lodf=env.dcnet.lodf,basis=env._build_orthogonal_physics_action_basis(),flow_intercept=env.dcnet.flow_intercept,valid_outages=env.valid_contingency_indices,effective_limit=env.dcnet.effective_limit)
    project=package/'ED_RL'
    project.mkdir(exist_ok=True)
    sources=[CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,ROOT/'paper_results/case300_fixed_slack_v1_screen.json']
    sources += [p for p in ROOT.glob('*.py')]
    sources += [ROOT/'case39_routeb_rescuable_dataset.json',ROOT/'case118_routeb_rescuable_dataset.json']
    sources += [ROOT/'paper_results'/name for name in ('case39_action_diagnosis.json','case118_action_diagnosis.json','case300_fixed_slack_v1_full_test_audit.json')]
    for folder in ('gat39_results','gatv118_results','rl39_routeb_main','rl118_routeb_main'):
        sources += [p for p in (ROOT/folder).glob('*') if p.is_file()]
    for checkpoint in CHECKPOINTS.values():
        sources += [p for p in checkpoint.parent.glob('*') if p.is_file()]
    for source in sources:
        target=project/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    shutil.copytree(CODE,project/'paper_results/code',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(HERE,project/'GAT-SAC-SCED-Framework/first revise/experiments',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(REV/'backup_receipt.json',project/'GAT-SAC-SCED-Framework/first revise/backup_receipt.json')
    for name in ('REVISION_GUIDE.md', 'JOINT_FALLBACK_RESULTS.md', 'DELIVERY_REVIEW_CN.md'):
        if (REV/name).exists():
            shutil.copy2(REV/name,project/'GAT-SAC-SCED-Framework/first revise'/name)
    # Retain immutable measured records outside fresh rerun output directories.
    recorded=project/'GAT-SAC-SCED-Framework/first revise/reported_results'
    excluded={'numba_cache','jacobian_cache','jacobian_cache_v2','scratch','interrupted_runs','resume_backups'}
    for source in RESULTS.rglob('*'):
        relative=source.relative_to(RESULTS)
        if not source.is_file() or any(part in excluded for part in relative.parts): continue
        if source.suffix not in ('.json','.csv','.log') and source.name!='best_actor.pth': continue
        target=recorded/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    manuscript=package/'manuscript'; manuscript.mkdir(exist_ok=True)
    for source in REV.iterdir():
        if source.is_file() and source.suffix in ('.tex','.bib','.bbl','.bst','.sty','.cls'):
            shutil.copy2(source,manuscript/source.name)
    for folder in ('figures','revision_figures'):
        shutil.copytree(REV/folder,manuscript/folder,dirs_exist_ok=True)
    # Mechanical relocation makes the copied legacy Windows-root constants portable.
    for source in project.rglob('*.py'):
        content=source.read_text(encoding='utf-8-sig')
        content=content.replace(str(ROOT).replace('\\','\\\\'),str(project).replace('\\','/'))
        content=content.replace(str(ROOT),str(project).replace('\\','/'))
        source.write_text(content,encoding='utf-8')
    settings={name:getattr(mod.core,name) for name in ('RL_VMIN_PU','RL_VMAX_PU','REF_SOURCE_P_MIN_MW','REF_SOURCE_P_MAX_MW','REF_SOURCE_Q_MIN_MVAR','REF_SOURCE_Q_MAX_MVAR','REF_SOURCE_S_MAX_MVA','GEN_VM_MIN','GEN_VM_MAX','EXT_VM_MIN','EXT_VM_MAX')}
    settings['voltage_band_origin']='Predefined benchmark limits; not claimed to be a universal engineering standard.'
    settings['python']=sys.version
    settings['packaged_project_root']=project.resolve().as_posix()
    settings['packages']={p:importlib.metadata.version(p) for p in ('torch','pandapower','numpy','scipy')}
    write(package/'settings.json',settings)
    shutil.copy2(HERE/'relocate_reproducibility.py',package/'relocate_reproducibility.py')
    (package/'README.txt').write_text('Extract this archive, run python relocate_reproducibility.py, and use the supplied scripts under ED_RL/GAT-SAC-SCED-Framework/first revise/experiments. Package versions are in settings.json. A licensed Gurobi installation is required for optimization baselines. Archived measured records and new trained checkpoints are under first revise/reported_results; their original provenance paths are retained. Fresh reruns write first revise/experiment_results and do not overwrite these records. Use python experiments/run_revision.py STAGE from the revision directory rather than the machine-specific CMD launcher. Manuscript LaTeX sources and figures are under manuscript. The package supports training/evaluation with the supplied candidate and screened datasets; a complete clean-machine rerun has not been performed. This package is local; no external upload has been performed.\n',encoding='utf-8')
    manifest=[{'path':p.relative_to(package).as_posix(),'sha256':sha(p)} for p in package.rglob('*') if p.is_file() and p!=package/'manifest.json']
    write(package/'manifest.json',manifest)
    with zipfile.ZipFile(REV/'Reproducibility_Package_v2.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(package).as_posix())
    print('Local reproducibility package created; nothing uploaded.')

if __name__=='__main__': main()
