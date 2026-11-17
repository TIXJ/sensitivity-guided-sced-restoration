"""Create an author-review copy without changing submission sources."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '01_BILINGUAL_REVIEW'
WORK = OUT / '_translation_work'
SOURCE = ROOT / 'Response_to_Associate_Editor_and_Reviewers.tex'


def prepare():
    OUT.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    text = SOURCE.read_text(encoding='utf-8')
    start = text.index(r'\begin{document}') + len(r'\begin{document}')
    end = text.index(r'\end{document}')
    boundaries = [start]
    for token in (r'\section*{Response to Reviewer 1',
                  r'\section*{Response to Reviewer 3',
                  r'\rcomment{Comment \#8:}',
                  r'\section*{Response to Reviewer 6'):
        pos = text.index(token, boundaries[-1] + 1)
        boundaries.append(pos)
    boundaries.append(end)
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:]), 1):
        (WORK / f'part_{i}_english.tex').write_text(text[a:b], encoding='utf-8')
    (WORK / 'source_sha256.txt').write_text(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), encoding='ascii')
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / '00_REVISION_UPLOAD').iterdir() if p.is_file()}
    (WORK / 'submission_before.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def strip_zh(text):
    pattern = re.compile(r'\\zh\{')
    while match := pattern.search(text):
        i = match.end()
        depth = 1
        while depth:
            if text[i] == '\\':
                i += 2
                continue
            if text[i] == '{': depth += 1
            if text[i] == '}': depth -= 1
            i += 1
        text = text[:match.start()] + text[i:]
    return text


def build():
    text = SOURCE.read_text(encoding='utf-8')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (WORK / 'source_sha256.txt').read_text()
    parts = []
    for i in range(1, 6):
        english = (WORK / f'part_{i}_english.tex').read_text(encoding='utf-8')
        bilingual = (WORK / f'part_{i}_bilingual.tex').read_text(encoding='utf-8')
        assert re.sub(r'\s+', '', strip_zh(bilingual)) == re.sub(r'\s+', '', english), f'English changed in part {i}'
        parts.append(bilingual)
    preamble = text.split(r'\begin{document}')[0]
    preamble = re.sub(r'(\\documentclass\[\d+pt\]\{article\})',
                      lambda m: m[0] + '\n' + r'\usepackage[UTF8,fontset=windows]{ctex}' + '\n' + r'\setmainfont{Times New Roman}', preamble, count=1)
    preamble += '\n' + r'\newcommand{\zh}[1]{\par\begingroup\normalfont\color{black}\setlength{\parskip}{0.35em}\noindent #1\par\endgroup\smallskip}' + '\n'
    # Long bilingual comments must be able to flow across pages.
    preamble += '\n' + r'''\renewcommand{\rcomment}[2]{%
      \par\Needspace{5\baselineskip}\vspace{0.8em}%
      \noindent\textbullet\ \uline{\textbf{\textit{#1}}}\par
      {\sloppy\leftskip=1.2em\itshape\uline{#2}\par}\vspace{0.3em}}
'''
    notice = r'\zh{中英对照审阅版。英文内容保持与正式回复信一致，中文仅供作者核对，不作为投稿文件。文中引用的页码均指英文标蓝修改稿，而非本对照版页码。表格和图片保留英文原件，其中文说明紧随其后。}'
    body = ''.join(parts)
    body = body.replace('We thank the Associate Editor and reviewers for their comments.',
                        notice + '\n\nWe thank the Associate Editor and reviewers for their comments.', 1)
    result = preamble + '\n\\begin{document}\n' + body + '\n\\end{document}\n'
    (OUT / 'Response_Bilingual_EN_ZH.tex').write_text(result, encoding='utf-8')
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', result):
        target = OUT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    audit = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             'english_preserved': True, 'comments': result.count(r'\rcomment{'),
             'responses': result.count(r'\aresponse{'), 'chinese_blocks': result.count(r'\zh{')}
    (OUT / 'Translation_Audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(audit)


if __name__ == '__main__':
    import sys
    prepare() if '--prepare' in sys.argv else build()
