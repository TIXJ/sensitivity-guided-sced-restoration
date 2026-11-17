"""Add order-controlled evidence without rebuilding or replacing the data archive."""
from datetime import datetime
import hashlib
import json
import zipfile
from revision_common import REV,ROOT,RESULTS,read,write,sha


def main():
    receipt_path=REV/'private_reviewer_materials/latest.json';receipt=read(receipt_path)
    from pathlib import Path
    base=Path(receipt['archive']);assert sha(base)==receipt['sha256']
    folder=RESULTS/'reviewer_sequence_balanced_v2'
    assert read(folder/'combined_summary.json')['complete']
    target=base.parent/('Evidence_Addendum_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    files=list(folder.glob('*.json'))+list((REV/'experiments').rglob('*.py'))
    files+=list((RESULTS/'dual_verified_once_v2').glob('*.json'))
    files+=list((RESULTS/'deployment_accounting_v1').glob('*.json'))
    files+=list((RESULTS/'spatial_dual_final_v1').glob('*.json'))
    files+=[REV/'Table_Evidence_Manifest.json',REV/'Table_Evidence_Index.md']
    files+=[REV/'Reproduce_Primary_Results.md']
    files+=[RESULTS/'final_paired_timing_audit.json',RESULTS/'final_paired_timing_cases.csv']
    files+=list(REV.glob('revision_*.py'))+list(REV.glob('build_*.py'))+[REV/'final_evidence_documents.py']
    entries={}
    with zipfile.ZipFile(base) as archive:previous=json.loads(archive.read('settings.json'))['packaged_project_root']
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(set(files)):
            rel=source.relative_to(ROOT).as_posix();payload=source.read_bytes()
            name='ED_RL/'+rel
            if source.suffix=='.py':
                original_name='original_source/'+rel
                archive.writestr(original_name,payload)
                entries[original_name]=hashlib.sha256(payload).hexdigest()
                content=payload.decode('utf-8-sig')
                for prefix in (str(ROOT).replace('\\','\\\\'),str(ROOT),ROOT.as_posix()):content=content.replace(prefix,previous)
                payload=content.encode('utf-8')
            else:
                name='ED_RL/'+source.relative_to(ROOT).as_posix()
            archive.writestr(name,payload);entries[name]=hashlib.sha256(payload).hexdigest()
        archive.writestr('ADDENDUM_README.txt','Use with the author-held data archive and current compact document/code delivery. This adds final dual-start remeasurement on previously examined spatial profiles, route and signed-exchange accounting for the historical workflows, and retained order-controlled sequence evidence. No new policy training or gate selection is performed. Source hashes refer to unrelocated originals under original_source. Nothing was externally uploaded.\n')
        archive.writestr('manifest.json',json.dumps({'base_archive_sha256':receipt['sha256'],'entries':entries},indent=2))
    with zipfile.ZipFile(target) as archive:assert archive.testzip() is None
    receipt.update(addendum=str(target),addendum_sha256=sha(target))
    write(receipt_path,receipt);print(receipt)


if __name__=='__main__':main()
