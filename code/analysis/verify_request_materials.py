"""Check actual author-held archives, not just material-generation scripts."""
import hashlib
from pathlib import Path
import zipfile
from revision_common import REV,ROOT,CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,CHECKPOINTS,read,write,sha


def inspect_archive(path, expected_sha, manifest_name):
    import json
    path=Path(path)
    assert path.is_file() and sha(path)==expected_sha,path
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None,path
        names=archive.namelist()
        assert len(names)==len(set(names)) and all('\\' not in name for name in names)
        manifest=json.loads(archive.read(manifest_name))
        entries=({r['path']:r['sha256'] for r in manifest}
                 if isinstance(manifest,list) else manifest['entries'])
        for name,digest in entries.items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==digest,(path,name)
        return {'archive':str(path),'sha256':expected_sha,'verified_entries':len(entries),
                'size_mib':round(path.stat().st_size/1024**2,2),'names':names}


def main():
    full=read(REV/'private_reviewer_materials/latest.json')
    audit=read(REV/'private_reviewer_materials/latest_primary_audit.json')
    assert not full['publicly_uploaded'] and not audit['publicly_uploaded']
    base=inspect_archive(full['archive'],full['sha256'],'manifest_current.json')
    addendum=inspect_archive(full['addendum'],full['addendum_sha256'],'manifest.json')
    primary=inspect_archive(audit['archive'],audit['sha256'],'AUDIT_MANIFEST.json')
    for name in ('case300_network.json','security_matrices.npz','settings.json'):
        assert name in primary['names']
    for suffix in ('case39_operating_network.json','case118_operating_network.json'):
        assert any(name.endswith(suffix) for name in base['names'])
    with zipfile.ZipFile(primary['archive']) as archive:
        for source in (CANDIDATES,CANDIDATE_SPLIT,DATASET,SPLIT,GAT,CHECKPOINTS['sac']):
            name='ED_RL/'+source.relative_to(ROOT).as_posix()
            assert name in primary['names']
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha(source)
    for suffix in ('dual_verified_once_v2/dual.json','spatial_dual_final_v1/results.json',
                   'shared_voltage_speed_v3/test_conditions.json'):
        matches=[name for name in primary['names'] if name.endswith(suffix)]
        assert len(matches)==1
        relative=matches[0].split('/first revise/',1)[1]
        with zipfile.ZipFile(primary['archive']) as archive:
            assert hashlib.sha256(archive.read(matches[0])).hexdigest()==sha(REV/relative)
    for name in ('Reproduce_Primary_Results.md','Table_Evidence_Manifest.json','final_evidence_documents.py'):
        matches=[n for n in addendum['names'] if n.startswith('ED_RL/') and n.endswith('/'+name)]
        assert len(matches)==1
    result={'publicly_uploaded':False,'complete_clean_machine_rerun':False,
            'checks':'Archive SHA256, ZIP CRC, every manifest entry, essential network/data/record presence and current primary raw-record hashes.',
            'archives':[{k:v for k,v in x.items() if k!='names'} for x in (base,addendum,primary)]}
    write(REV/'private_reviewer_materials/latest_verification.json',result)
    print(result)


if __name__=='__main__':
    main()
