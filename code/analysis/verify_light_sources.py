"""Compile the delivered TeX sources in a separate directory and compare text."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import zipfile
import fitz


def main():
    here=Path(__file__).resolve().parents[1]
    receipt=json.loads((here/'latest_light_delivery.json').read_text(encoding='utf-8'))
    target=here/('delivery_smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    target.mkdir()
    with zipfile.ZipFile(receipt['zip']) as archive:
        for item in archive.infolist():
            if item.filename.startswith('03_LaTeX_Source/'):
                archive.extract(item,target)
    folder=target/'03_LaTeX_Source'
    checked={}
    for name in ('Manuscript_Revised_Marked','Manuscript_Revised_Clean',
                 'Response_to_Associate_Editor_and_Reviewers'):
        result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',name+'.tex'],
                              cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (folder/(name+'_smoke.log')).write_bytes(result.stdout)
        assert result.returncode==0,result.stdout[-2000:]
        with fitz.open(folder/(name+'.pdf')) as a, fitz.open(here/(name+'.pdf')) as b:
            assert len(a)==len(b)
            assert all(x.get_text()==y.get_text() for x,y in zip(a,b)),name
            checked[name]=len(a)
    (target/'SMOKE_RESULT.json').write_text(json.dumps({'zip':receipt['zip'],
        'independent_compilation':True,'text_matches_current':True,'pages':checked},indent=2),encoding='utf-8')
    print(target)
    print(checked)


if __name__=='__main__':
    main()
