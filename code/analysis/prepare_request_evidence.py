"""Prepare real audit inputs/records locally for reasonable-request sharing."""
from datetime import datetime
import hashlib
from pathlib import Path
import zipfile
from revision_common import REV,ROOT,RESULTS,CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,CHECKPOINTS,read,write,sha


def main():
    receipt=read(REV/'private_reviewer_materials/latest.json')
    base=Path(receipt['archive']);assert sha(base)==receipt['sha256']
    assert read(RESULTS/'spatial_dual_final_v1/results.json')['complete']
    target=base.parent/('Primary_Audit_On_Request_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    files=[CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,CHECKPOINTS['sac'],
           REV/'Reproduce_Primary_Results.md',REV/'Table_Evidence_Manifest.json',REV/'Table_Evidence_Index.md',
           RESULTS/'shared_voltage_speed_v3/test_conditions.json']
    for name in ('dual_verified_once_v2','deployment_accounting_v1','spatial_dual_final_v1'):
        files+=list((RESULTS/name).glob('*.json'))
    manifest={}
    with zipfile.ZipFile(base) as source,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as out:
        for name in ('case300_network.json','security_matrices.npz','settings.json'):
            payload=source.read(name);out.writestr(name,payload)
            manifest[name]=hashlib.sha256(payload).hexdigest()
        for file in sorted(set(files)):
            name='ED_RL/'+file.relative_to(ROOT).as_posix();payload=file.read_bytes()
            out.writestr(name,payload);manifest[name]=hashlib.sha256(payload).hexdigest()
        import json
        out.writestr('AUDIT_MANIFEST.json',json.dumps({'base_archive_sha256':receipt['sha256'],
            'purpose':'Actual inputs and per-case evidence for reviewer audit; not a complete code/dependency distribution.',
            'availability':'Upon reasonable request; not publicly uploaded.','entries':manifest},indent=2))
    with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
    result={'archive':str(target),'sha256':sha(target),'size_mib':round(target.stat().st_size/1024**2,2),
            'files':len(manifest),'publicly_uploaded':False,'complete_clean_machine_rerun':False}
    write(REV/'private_reviewer_materials/latest_primary_audit.json',result)
    print(result)


if __name__=='__main__':
    main()
