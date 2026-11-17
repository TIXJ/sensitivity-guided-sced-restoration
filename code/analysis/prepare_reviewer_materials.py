"""Prepare author-held materials on request; never upload or alter old archives."""
from datetime import datetime
import json
import os
import zipfile
import io
from pathlib import Path
from revision_common import REV,ROOT,CODE,RESULTS,CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,CHECKPOINTS,sha,write
os.environ.setdefault('NUMBA_CACHE_DIR',str(RESULTS/'numba_cache'))


def capture_small_networks(folder):
    import pandapower as pp
    from evaluate_smallcases_common import load_case
    from smallcase_evaluation_contract import verify_state
    original=pp.runpp
    rows=[]
    for case in (39,118):
        mod,samples,ids,env,action,dataset,checkpoint=load_case(case)
        env.reset(ids[0]);captured={}
        def capture(net,*args,**kwargs):
            captured['net']=net
            return original(net,*args,**kwargs)
        pp.runpp=capture
        try:info=verify_state(env,mod,case)
        finally:pp.runpp=original
        assert 'net' in captured,case
        path=folder/f'case{case}_operating_network.json'
        pp.to_json(captured['net'],str(path))
        partitions=mod.split_samples(samples) if case==39 else mod.split_samples_stratified(samples)
        rows.append({'case':case,'sample_id':samples[ids[0]].get('sample_id',samples[ids[0]].get('id')),'dataset':str(dataset),
            'file':path.name,'sha256':sha(path),'scope':'Actual verifier network at the first fixed test ID, not a newly designed test case.',
            'partition_sample_ids':{name:[samples[i].get('sample_id',samples[i].get('id')) for i in part]
                                    for name,part in zip(('train','validation','test'),partitions)},
            'verification':info})
    write(folder/'network_snapshots.json',{'rows':rows})


def main():
    base=REV/'Reproducibility_Package_v2.zip'
    assert base.exists()
    assert json.loads((RESULTS/'reviewer_followup_v1/summary.json').read_text())['complete']
    folder=REV/'private_reviewer_materials'/datetime.now().strftime('%Y%m%d_%H%M%S')
    folder.mkdir(parents=True)
    capture_small_networks(folder)
    target=folder/'Reviewer_Materials_On_Request.zip'
    overrides={}
    sources=[CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,*CHECKPOINTS.values(),
        ROOT/'case39_routeb_rescuable_dataset.json',ROOT/'case118_routeb_rescuable_dataset.json']
    sources+=list(CODE.glob('*.py'))+list((REV/'experiments').rglob('*.py'))
    sources+=[p for p in (RESULTS/'jacobian_cache/none').glob('*.npz')]
    for name in ('gat39_results','gatv118_results','rl39_routeb_main','rl118_routeb_main'):
        sources+=[p for p in (ROOT/name).glob('*') if p.is_file() and p.suffix in ('.pth','.pt','.json')]
    for source in sources:
        assert source.is_file(),source
        overrides['ED_RL/'+source.relative_to(ROOT).as_posix()]=source
    for source in RESULTS.rglob('*'):
        rel=source.relative_to(RESULTS)
        if any(p in {'numba_cache','jacobian_cache','jacobian_cache_v2','interrupted_runs','resume_backups'} for p in rel.parts):continue
        if source.is_file() and (source.suffix in ('.json','.csv','.md','.log') or source.name=='best_actor.pth'):
            overrides['ED_RL/GAT-SAC-SCED-Framework/first revise/reported_results/'+rel.as_posix()]=source
    for source in folder.glob('*.json'):overrides['current_network_snapshots/'+source.name]=source
    for source in REV.iterdir():
        if source.is_file() and (source.name.startswith('revision_') or source.name.startswith('build_') or source.name in {'final_evidence_documents.py','check_revision_artifacts.py'}):
            if source.suffix in ('.py','.json','.tex'):overrides['ED_RL/GAT-SAC-SCED-Framework/first revise/'+source.name]=source
    manifest=[]
    import hashlib
    with zipfile.ZipFile(base) as old,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as new:
        import numpy as np
        model=json.loads(CANDIDATES.read_text(encoding='utf-8'))['linear_security_model']
        with np.load(io.BytesIO(old.read('security_matrices.npz'))) as matrices:
            np.testing.assert_allclose(matrices['S'],model['S'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(matrices['lodf'],model['lodf'],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(matrices['valid_outages'],model['valid_contingency_indices'])
        settings=json.loads(old.read('settings.json'))
        previous=settings['packaged_project_root']
        for name in sorted(set(old.namelist())|set(overrides)):
            if name=='manifest.json' or name.startswith('manuscript/'):continue
            payload=overrides[name].read_bytes() if name in overrides else old.read(name)
            if name.endswith('.py') and name.startswith('ED_RL/') and name in overrides:
                original_name='original_source/'+name.removeprefix('ED_RL/')
                new.writestr(original_name,payload)
                manifest.append({'path':original_name,'sha256':hashlib.sha256(payload).hexdigest(),'current_override':True})
                content=payload.decode('utf-8-sig')
                for prefix in (str(ROOT).replace('\\','\\\\'),str(ROOT),ROOT.as_posix()):content=content.replace(prefix,previous)
                payload=content.encode('utf-8')
            new.writestr(name,payload)
            manifest.append({'path':name,'sha256':hashlib.sha256(payload).hexdigest(),'current_override':name in overrides})
        text=('Author-held materials available upon reasonable request. Nothing has been uploaded.\n'
            'This archive updates the earlier reproduction materials with current datasets, fixed split IDs, primary checkpoints, new seed checkpoints, protocols and per-case results.\n'
            'Revision PDFs, response letter and standalone LaTeX sources are supplied in the separate compact document/code delivery, not duplicated in this data archive.\n'
            'Actual small-system verifier networks are under current_network_snapshots; the IEEE-300 network and fixed security matrices are at the archive root.\n'
            'Run relocate_reproducibility.py after extraction to update legacy Python root paths. Source provenance paths inside historical result JSON are retained unchanged.\n'
            'Original, unrelocated current source is retained under original_source so recorded source hashes remain auditable; ED_RL contains relocation-ready copies.\n'
            'reported_results contains immutable evidence, not fresh output destinations. Some timing commands intentionally refuse overwriting a frozen study; copy archived prerequisites or use a new study directory for reruns.\n'
            'The numerical tests and LaTeX compilation were checked locally. A complete clean-machine retraining/evaluation run has not been performed; compatible packages, GPU support and a Gurobi license are required.\n')
        new.writestr('CURRENT_MATERIALS_README.txt',text)
        new.writestr('manifest_current.json',json.dumps(manifest,indent=2))
        new.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(target) as saved:
        assert saved.testzip() is None
        names=set(saved.namelist())
        assert all('ED_RL/'+p.relative_to(ROOT).as_posix() in names for p in (CANDIDATES,CANDIDATE_SPLIT,GAT,*CHECKPOINTS.values()))
    receipt={'archive':str(target),'size_mib':round(target.stat().st_size/1024**2,2),'sha256':sha(target),
        'publicly_uploaded':False,'clean_machine_rerun':False,'files':len(manifest),
        'network_snapshots':str(folder/'network_snapshots.json')}
    write(REV/'private_reviewer_materials/latest.json',receipt)
    print(receipt)


if __name__=='__main__':main()
