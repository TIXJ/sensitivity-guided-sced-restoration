"""Nine checks for the final abstract/conclusion and AE #7 synchronization."""
from pathlib import Path
import contextlib
import io
import json
import re
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision_response_layout import brace_argument,unmark
from revision_acceptance_scope import TITLE
import validate_acceptance_scope


def abstract(text):
    return unmark(re.search(r'\\begin\{abstract\}(?:\[\])?\s*(.*?)\\end\{abstract\}',text,re.S)[1]).strip()


def conclusion(text):
    return unmark(text.split(r'\section{Conclusions}\label{sec5}',1)[1].split(r'\printcredits',1)[0]).strip()


def ae7_excerpt(text):
    block=text.split(r'\rcomment{AE Comment \#7:}',1)[1].split(r'\rcomment{AE Comment \#8:}',1)[0]
    assert 'The complete revised abstract is quoted below.' in block
    match=re.search(r'\\revisedexcerpt\{Abstract, p\.~1\}',block)
    assert match is not None
    return brace_argument(block,match.end())[0]


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        validate_acceptance_scope.main()
    marked=(ROOT/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    clean=(ROOT/'Manuscript_Revised_Clean.tex').read_text(encoding='utf-8')
    response=(ROOT/'Response_to_Associate_Editor_and_Reviewers.tex').read_text(encoding='utf-8')
    bilingual=(ROOT/'01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.tex').read_text(encoding='utf-8')
    a=abstract(marked);c=conclusion(marked)
    log=[]
    def out(value):
        print(value);log.append(value)
    out('1. Counts and paragraph structure')
    assert 150<=len(a.split())<=250 and len(TITLE.split())==12
    assert '\n\n' not in a and '\n\n' not in c
    out(f'  Abstract: {len(a.split())} words, one paragraph')
    out(f'  Conclusion: {len(c.split())} words, one paragraph')
    out('  Title: 12 words; PASS')
    out('2. Required abstract passages')
    required=['acceptance-matched rather than population-wide','accepts 44',
              'a gain not reproduced on the tested spatial profiles','IEEE 39-bus and 118-bus']
    for phrase in required:
        assert phrase in a
        out('  '+phrase+': PASS')
    for value in ('0.788','3.604','3.075','3.9 to 4.6',r'9.8\%',
                  'Over all 75 unfiltered candidates the policy accepts 44.'):
        assert value in a
    out('  All protected numbers and complete 44/75 sentence retained: PASS')
    out('3. Prohibited abstract phrase')
    assert 'conditional on the screened benchmark' not in a
    out('  conditional on the screened benchmark count = 0; PASS')
    out('4. Protected conclusion passages')
    assert '44 of 75' in c and 'The 27/27 figure applies only to the screened subset.' in c
    assert 'a saving that does not extend to the new spatial profiles' in c
    out('  44 of 75, 27/27 and requested spatial-profile qualifier: PASS')
    out('5. AE #7 complete abstract quotation')
    actual=ae7_excerpt(response)
    identical=re.sub(r'\s+','',actual)==re.sub(r'\s+','',a)
    assert identical and actual==a
    assert ae7_excerpt(bilingual)==actual
    out('  IDENTICAL: '+str(identical))
    out(f'  Manuscript / AE #7: {len(a)} / {len(actual)} characters; exact TeX match including percent escape')
    out('  Bilingual English quotation matches official response: PASS')
    out('6. Frozen label -> manuscript page -> rendered-text anchor')
    positions=json.loads((ROOT/'_audit/acceptance_scope_locations.json').read_text(encoding='utf-8'))
    for item in positions:
        assert item['hit']
        out('  '+item['label']+' -> '+item['page']+' -> hit')
    before=ROOT.parent/'first_revise_archive_20261112/abstract_scope_before_20261117'
    label_list=lambda text:re.findall(r'\\@namedef\{r@ms-([^}]+)\}',text)
    assert label_list((before/'Response_to_Associate_Editor_and_Reviewers.tex').read_text(encoding='utf-8'))==label_list(response)
    out(f'  {len(positions)}/{len(positions)} = 100%; label names and order unchanged')
    out('  Page test uses quotation openings/captions on the cited start page; long quotations can continue to following pages.')
    out('7. PDF text and compilation logs')
    pages={}
    for name in ('Manuscript_Revised_Clean','Manuscript_Revised_Marked','Response_to_Associate_Editor_and_Reviewers'):
        doc=fitz.open(ROOT/(name+'.pdf'));pages[name]=len(doc)
        assert all('??' not in page.get_text() for page in doc)
        text=(ROOT/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'LaTeX Warning: Reference|Citation .*undefined',text)
        out('  '+name+': PASS (no ?? or undefined references/citations)')
    out('8. Clean/marked line comparison')
    assert len(clean.splitlines())==len(marked.splitlines())
    differences=[(i,x,y) for i,(x,y) in enumerate(zip(clean.splitlines(),marked.splitlines()),1) if x!=y]
    assert len(differences)==2
    assert {x for _,x,_ in differences}=={r'\newcommand{\rev}[1]{#1}',r'\revisionmarkedfalse'}
    out('  Different lines: '+str(len(differences)))
    for number,x,y in differences:out(f'  Line {number}: {x} | {y}')
    out('  PASS: only revision-colour definition and switch differ')
    out('9. Recompiled PDF page counts')
    pages['Response_Bilingual_EN_ZH']=len(fitz.open(ROOT/'01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.pdf'))
    for name,count in pages.items():out('  '+name+': '+str(count)+' pages')
    # The request permits no other manuscript text, bibliography, or data changes.
    previous=(before/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    expected=unmark(previous).replace(abstract(previous),a).replace(conclusion(previous),c)
    assert expected==unmark(marked),'Unexpected manuscript text change outside abstract/conclusion'
    for name in ('revision_references.bib','GAT_SAC_PTDF_LODF_SCED_Restoration.bib','revision-num.bst'):
        assert (before/name).read_bytes()==(ROOT/name).read_bytes(),name
    bbl=(ROOT/'Manuscript_Revised_Clean.bbl').read_text(encoding='utf-8')
    assert bbl.count(r'\bibitem')==25
    assert 'upon reasonable request' in marked
    out('  Extra: all other manuscript text unchanged; bibliography files unchanged; bibitems = 25')
    assert not re.findall(r'\[CHECK:[^]]+\]',marked+response)
    out('CHECK placeholders: none')
    (ROOT/'_audit/abstract_scope_sync_validation.txt').write_text('\n'.join(log)+'\n',encoding='utf-8')
    (ROOT/'_audit/abstract_scope_sync_results.json').write_text(json.dumps({'passed':9,'abstract_words':len(a.split()),'conclusion_words':len(c.split()),'title_words':12,'ae7_identical':identical,'frozen_labels':len(positions),'clean_marked_difference_lines':2,'bibitems':25,'pages':pages,'check_placeholders':[]},indent=2),encoding='utf-8')


if __name__=='__main__':main()
