"""Validate requested wording, frozen locations and unchanged review records."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import unicodedata

import fitz
from docx import Document

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision_response_layout import brace_argument, read_manuscript_labels, unmark, manuscript_float
from revision_acceptance_scope import TITLE, HIGHLIGHTS


def norm(value):
    value=unicodedata.normalize('NFKD',value).replace('\u00ad','')
    value=re.sub(r'-\s*\n\s*(?=[a-z])','',value)
    return re.sub(r'[^a-z0-9]','',value.lower())


def prose(tex,labels):
    tex=unmark(tex)
    tex=re.sub(r'\\(?:page)?ref\{(?:ms-)?([^}]+)\}',lambda m:labels.get(m[1],('',''))[0],tex)
    tex=re.sub(r'\\cite\{[^}]+\}','',tex)
    tex=re.sub(r'\\(?:begin|end)\{[^}]+\}(?:\[[^]]*\])?','',tex)
    tex=re.sub(r'\\[A-Za-z@]+\*?(?:\[[^]]*\])?','',tex)
    return tex.replace('~',' ').replace('{','').replace('}','').replace('\\','')


def anchor(tex,labels):
    # A quotation can span pages. Its prose opening must occur on its stated page.
    tex=re.split(r'\$|\\\[',tex,1)[0]
    words=prose(tex,labels).split()
    return ' '.join(words[:12])


def main():
    lines=[]
    def out(s):
        print(s)
        lines.append(s)
    marked=(ROOT/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    clean=(ROOT/'Manuscript_Revised_Clean.tex').read_text(encoding='utf-8')
    response=(ROOT/'Response_to_Associate_Editor_and_Reviewers.tex').read_text(encoding='utf-8')
    evidence=json.loads((ROOT/'response_evidence.json').read_text(encoding='utf-8'))
    labels=read_manuscript_labels((ROOT/'Manuscript_Revised_Marked.aux').read_text(encoding='utf-8'))
    output=subprocess.check_output(['pdftotext','-enc','UTF-8',str(ROOT/'Manuscript_Revised_Marked.pdf'),'-'])
    pages=output.decode('utf-8').split('\f')
    if not pages[-1].strip():pages.pop()
    plain_pages=[norm(page) for page in pages]
    out('1. Compilation and PDF references')
    for name in ('Manuscript_Revised_Clean','Manuscript_Revised_Marked','Response_to_Associate_Editor_and_Reviewers'):
        log=(ROOT/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'LaTeX Warning: Reference|Citation .*undefined',log),name
        doc=fitz.open(ROOT/(name+'.pdf'))
        assert all('??' not in page.get_text() for page in doc),name
        out(f'  {name}: PASS ({len(doc)} pages, no undefined references or ??)')
    out('2. Word and character limits')
    abstract=re.search(r'\\begin\{abstract\}(?:\[\])?\s*(.*?)\\end\{abstract\}',marked,re.S)[1]
    assert abstract.strip().startswith(r'\rev{') and '\n\n' not in abstract.strip()
    abstract=unmark(abstract)
    count=len(abstract.split())
    assert 150<=count<=250
    assert 12<=len(TITLE.split())<=15
    out(f'  Abstract: {count} words, one paragraph, PASS')
    out(f'  Title: {len(TITLE.split())} space-delimited words, PASS')
    highlights=[p.text for p in Document(ROOT/'Highlights_Revised.docx').paragraphs if p.text][1:]
    assert highlights==HIGHLIGHTS
    for i,h in enumerate(highlights,1):
        assert len(h)<=85
        out(f'  Highlight {i}: {len(h)} characters, PASS | {h}')
    out('3. Single-key citations')
    for name,text in [('Clean',clean),('Marked',marked),('Response',response)]:
        multi=[x for x in re.findall(r'\\cite\{([^}]+)\}',text) if ',' in x]
        assert not multi,(name,multi)
        out(f'  {name}: PASS (0 multi-key citations)')
    out('4. Frozen labels -> manuscript page -> rendered anchor')
    frozen={}
    for match in re.finditer(r'\\@namedef\{r@ms-([^}]+)\}',response):
        value,_=brace_argument(response,match.end())
        number,end=brace_argument(value,0)
        page,_=brace_argument(value,end)
        frozen[match[1]]=(number,page)
    checks=[]
    for label,(number,page) in frozen.items():
        assert labels[label]==(number,page),(label,labels[label],number,page)
        target=plain_pages[int(page)-1]
        if label.startswith(('tab:','fig:')) or label=='fig_framework':
            kind='table' if label.startswith('tab:') else 'figure'
            _,caption=manuscript_float(marked,label,kind)
            snippet=anchor(caption,labels)
        elif label.startswith('reply:quote_'):
            key=label.removeprefix('reply:quote_')
            snippet=anchor(evidence['verbatim_excerpts'][key],labels)
        elif label.startswith('eq:'):
            snippet='('+number+')'
        else:
            pos=marked.index(r'\label{'+label+'}')
            headings=list(re.finditer(r'\\(?:sub)*section\*?\{',marked[:pos]))
            snippet=prose(brace_argument(marked,headings[-1].end()-1)[0],labels)
        hit=norm(snippet) in target
        checks.append({'label':label,'page':page,'number':number,'anchor':snippet,'hit':hit})
        out(f'  {label} -> {page} -> {"yes" if hit else "NO"} | {snippet}')
    assert all(x['hit'] for x in checks),[x for x in checks if not x['hit']]
    out(f'  PASS: {len(checks)} frozen labels agree with AUX and rendered page anchors.')
    out('  Long quotations are located by their opening on the cited page; continuation pages are not misrepresented as single-page quotations.')
    out('5. Abstract and conclusion scope')
    assert '3.9 to 4.6' in abstract and '44' in abstract
    for phrase in ('without establishing an advantage','is slower on an independent','interval includes zero'):
        assert phrase not in abstract
    conclusion=unmark(marked.split(r'\section{Conclusions}\label{sec5}',1)[1].split(r'\printcredits',1)[0]).strip()
    assert not conclusion.endswith('This acceptance is not full alternating-current contingency security.')
    assert re.search(r'44 of (?:the )?75',conclusion)
    assert '3.208' in marked and '23.640' in marked and '4.704' in marked
    out('  PASS: conditional timing and 44/75 retained; negative full-body evidence retained.')
    out('6. Reviewer gratitude grammar')
    assert 'We thank the Reviewer' not in marked+response
    assert response.count(r'\rcomment{')==response.count(r'\aresponse{')==43
    out('  PASS: 0 instances of We thank the Reviewer; 43 comments and 43 responses.')
    out('7. Clean/marked content identity')
    expected=marked.replace(r'\newcommand{\rev}[1]{\textcolor{blue}{#1}}',r'\newcommand{\rev}[1]{#1}').replace(r'\revisionmarkedtrue',r'\revisionmarkedfalse')
    assert clean==expected
    a=fitz.open(ROOT/'Manuscript_Revised_Clean.pdf');b=fitz.open(ROOT/'Manuscript_Revised_Marked.pdf')
    assert len(a)==len(b) and all(x.get_text()==y.get_text() for x,y in zip(a,b))
    out('  PASS: source differs only in revision colouring; rendered page text identical.')
    backup=ROOT.parent/'first_revise_archive_20261112/acceptance_scope_before_20261117'
    readonly=json.loads((backup/'readonly_hashes.json').read_text(encoding='utf-8-sig'))
    assert all(hashlib.sha256(Path(x['Path']).read_bytes()).hexdigest().upper()==x['Hash'] for x in readonly)
    old=json.loads((backup/'response_evidence.json').read_text(encoding='utf-8'))['comments']
    assert all(v['original_comment_sha256']==old[k]['original_comment_sha256'] for k,v in evidence['comments'].items())
    out('  Extra: read-only reference hashes unchanged; all 43 original comments unchanged.')
    out('CHECK placeholders: '+str(re.findall(r'\[CHECK:[^]]+\]',marked+response)))
    (ROOT/'_audit/acceptance_scope_validation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'_audit/acceptance_scope_locations.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
