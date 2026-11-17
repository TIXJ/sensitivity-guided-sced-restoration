"""Refresh figure typography from preserved vector assets and recorded data."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import fitz
from figure_typography import normalize_vector_fonts,plot_contingency_errors

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',
                     (ROOT/'Manuscript_Revised_Clean.tex').read_text(encoding='utf-8'))
    backup=ROOT/'figure_sources/before_times_new_roman'
    for name in paths:
        target=ROOT/name;source=backup/name
        source.parent.mkdir(parents=True,exist_ok=True)
        if not source.exists():shutil.copy2(target,source)
        with fitz.open(target) as doc:
            needs_update=any('Times' not in s['font'] for p in doc
                             for b in p.get_text('dict')['blocks'] if 'lines' in b
                             for line in b['lines'] for s in line['spans'])
        if needs_update:
            temporary=target.with_name(target.stem+'_times_tmp.pdf')
            normalize_vector_fonts(target,temporary)
            temporary.replace(target)
    subprocess.run([r'external/Anaconda\envs\lct-GPU\python.exe','-c',
        "import json;from pathlib import Path;from figure_typography import plot_contingency_errors;"
        "root=Path('..');data=json.loads((root/'experiment_results/ac_contingencies.json').read_text());"
        "plot_contingency_errors(data['rows'],root/'revision_figures/contingency_flow_error.pdf')"],
        cwd=ROOT/'experiments',check=True)
    audit={}
    for name in paths:
        with fitz.open(ROOT/name) as doc:
            fonts=sorted({s['font'] for p in doc for b in p.get_text('dict')['blocks'] if 'lines' in b
                          for line in b['lines'] for s in line['spans']})
            assert fonts and all('Times' in font for font in fonts),(name,fonts)
            audit[name]=fonts
            if (ROOT/name).with_suffix('.png').exists():
                doc[0].get_pixmap(matrix=fitz.Matrix(3,3)).save((ROOT/name).with_suffix('.png'))
    (ROOT/'Figure_Font_Audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
