"""Check the review copy and preserve the official submission artifacts."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import sys
import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '01_BILINGUAL_REVIEW'


def main():
    before = json.loads((OUT / '_translation_work' / 'submission_before.json').read_text(encoding='utf-8'))
    unchanged = all(hashlib.sha256((ROOT / '00_REVISION_UPLOAD' / name).read_bytes()).hexdigest() == digest
                    for name, digest in before.items())
    if '--aligned' not in sys.argv:
        assert unchanged, 'Official files changed; validate the authorized synchronized version with --aligned'
    else:
        for name in ('Manuscript_Revised_Clean.pdf','Manuscript_Revised_Marked.pdf',
                     'Response_to_Associate_Editor_and_Reviewers.pdf','Cover_Letter_Revised.pdf','Highlights_Revised.docx'):
            assert (ROOT / name).read_bytes() == (ROOT / '00_REVISION_UPLOAD' / name).read_bytes(), name
    tex = (OUT / 'Response_Bilingual_EN_ZH.tex').read_text(encoding='utf-8')
    assert r'\externaldocument' not in tex
    log = (OUT / 'Response_Bilingual_EN_ZH.log').read_text(encoding='utf-8', errors='replace')
    assert 'Missing character:' not in log
    assert not re.search(r'(?:Reference|Citation) .* undefined', log)
    assert tex.count(r'\rcomment{') == 43
    assert tex.count(r'\aresponse{') == 43
    assert not re.search(r'回复\s*(?:\\#|#)?\d+[:：]',tex), 'Redundant translated response labels'
    assert 'the paragraph on p.' not in '\n'.join(re.findall(r'\\zh\{([^\n]+)',tex)), 'Untranslated location in Chinese'
    assert '旨在通过连续的反馈驱动校正' in tex, 'Preserve the qualification seeks to'
    assert '不构成与前述各行的配对比较' in tex, 'Preserve the unpaired-row qualification'
    doc = fitz.open(OUT / 'Response_Bilingual_EN_ZH.pdf')
    bounds = []
    empty = []
    for i, page in enumerate(doc):
        assert '??' not in page.get_text(), ('unresolved reference',i+1)
        if not page.get_text().strip(): empty.append(i + 1)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if not page.rect.contains(fitz.Rect(span['bbox'])):
                        bounds.append({'page': i + 1, 'text': span['text']})
    assert not bounds, bounds
    assert not empty, empty
    audit = json.loads((OUT / 'Translation_Audit.json').read_text())
    assert audit['source_sha256'] == hashlib.sha256((ROOT / 'Response_to_Associate_Editor_and_Reviewers.tex').read_bytes()).hexdigest()
    audit.update(pages=len(doc), official_upload_unchanged=unchanged,
                 official_upload_matches_current_sources='--aligned' in sys.argv,
                 missing_glyphs=False, unresolved_references=False,
                 text_outside_page=bounds, blank_pages=empty)
    (OUT / 'Translation_Audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    assets = [OUT / 'Response_Bilingual_EN_ZH.tex', OUT / 'README.txt']
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex):
        assets.append(OUT / path)
    with zipfile.ZipFile(OUT / 'Response_Bilingual_Source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(assets)):
            archive.write(path, path.relative_to(OUT).as_posix())
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
