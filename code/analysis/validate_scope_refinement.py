"""Run the ten requested checks for the local scope-wording refinement."""
from pathlib import Path
import contextlib
import io
import json
import re
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision_response_layout import unmark, brace_argument
from revision_acceptance_scope import TITLE
import validate_acceptance_scope


def main():
    # Includes read-only hashes, original comments, source quotations, and every
    # frozen reference against AUX and pdftotext page openings/captions.
    with contextlib.redirect_stdout(io.StringIO()):
        validate_acceptance_scope.main()
    marked=(ROOT/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    clean=(ROOT/'Manuscript_Revised_Clean.tex').read_text(encoding='utf-8')
    response=(ROOT/'Response_to_Associate_Editor_and_Reviewers.tex').read_text(encoding='utf-8')
    evidence=json.loads((ROOT/'response_evidence.json').read_text(encoding='utf-8'))
    abstract=re.search(r'\\begin\{abstract\}(?:\[\])?\s*(.*?)\\end\{abstract\}',marked,re.S)[1].strip()
    conclusion=marked.split(r'\section{Conclusions}\label{sec5}',1)[1].split(r'\printcredits',1)[0].strip()
    assert '\n\n' not in abstract and '\n\n' not in conclusion
    abstract=unmark(abstract)
    conclusion=unmark(conclusion)
    output=[]
    def out(text):
        print(text)
        output.append(text)
    out('1. PDF text and compilation logs')
    pages={}
    for name in ('Manuscript_Revised_Clean','Manuscript_Revised_Marked','Response_to_Associate_Editor_and_Reviewers'):
        doc=fitz.open(ROOT/(name+'.pdf'))
        pages[name]=len(doc)
        assert all('??' not in p.get_text() for p in doc)
        log=(ROOT/(name+'.log')).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'LaTeX Warning: Reference|Citation .*undefined',log)
        out('  '+name+': PASS (no ?? or undefined reference/citation)')
    out('2. Word counts and paragraph structure')
    ac=len(abstract.split());cc=len(conclusion.split());tc=len(TITLE.split())
    assert 150<=ac<=250 and 12<=tc<=15
    out(f'  Abstract: {ac} words, one paragraph, PASS')
    out(f'  Conclusion: {cc} words, one paragraph, PASS')
    out(f'  Title: {tc} space-delimited words, PASS')
    out('3. Required abstract claims and prohibited phrases')
    assert all(x in abstract for x in ('0.788','3.9 to 4.6','44','9.8','acceptance-matched'))
    assert all(x not in abstract for x in ('conditional on the screened benchmark','without establishing an advantage','interval includes zero'))
    out('  PASS: 0.788, 3.9 to 4.6, 44, 9.8 and acceptance-matched retained; 0 prohibited phrases')
    out('4. Unfiltered and screened conclusion counts')
    assert '44 of 75' in conclusion and '27/27 figure applies only to the screened subset' in conclusion
    out('  PASS: 44 of 75 and 27/27 figure applies only to the screened subset')
    out('5. Frozen label -> manuscript page -> rendered-text anchor')
    locations=json.loads((ROOT/'_audit/acceptance_scope_locations.json').read_text(encoding='utf-8'))
    for item in locations:
        assert item['hit']
        out('  '+item['label']+' -> '+item['page']+' -> hit')
    before=ROOT.parent/'first_revise_archive_20261112/scope_refinement_before_20261117/Response_to_Associate_Editor_and_Reviewers.tex'
    labels=lambda text:re.findall(r'\\@namedef\{r@ms-([^}]+)\}',text)
    assert labels(before.read_text(encoding='utf-8'))==labels(response)
    out(f'  PASS: {len(locations)}/{len(locations)} locations hit (100%); label names and order unchanged')
    out('  Location check: quotation openings/captions are found on the cited start page; long quotations may continue on subsequent pages.')
    out('6. Reviewer gratitude grammar')
    assert 'We thank the Reviewer' not in response+marked
    out('  PASS: We thank the Reviewer count = 0')
    out('7. Repeated security-cut statistics')
    count=unmark(marked).count('3.19 security-cut rounds')
    assert count==1
    assert '31 unsuccessful candidates each exhaust the 50-step budget' in unmark(marked)
    assert 'The spatial-profile saving interval includes zero' in unmark(marked)
    out('  PASS: 3.19 security-cut rounds count = 1; 31 budget-exhausted failures and spatial zero-including interval retained')
    out('8. Complete AE #3 conclusion quotation and full-text quotation alignment')
    excerpt=evidence['verbatim_excerpts']['conclusion_opening']
    assert excerpt!='Two conclusions follow.' and excerpt.startswith('Together, these findings support')
    assert excerpt in conclusion
    ae3=response.split(r'\rcomment{AE Comment \#3:}',1)[1].split(r'\rcomment{AE Comment \#4:}',1)[0]
    assert excerpt in ae3
    assert evidence['verbatim_excerpts']['abstract']==abstract
    assert evidence['verbatim_excerpts']['conclusion']==conclusion
    out('  PASS: AE #3 quotes the complete Together sentence; AE #7/#10 full quotations match manuscript verbatim')
    out('9. Clean and marked identity')
    expected=marked.replace(r'\newcommand{\rev}[1]{\textcolor{blue}{#1}}',r'\newcommand{\rev}[1]{#1}').replace(r'\revisionmarkedtrue',r'\revisionmarkedfalse')
    assert clean==expected
    a=fitz.open(ROOT/'Manuscript_Revised_Clean.pdf');b=fitz.open(ROOT/'Manuscript_Revised_Marked.pdf')
    assert len(a)==len(b) and all(x.get_text()==y.get_text() for x,y in zip(a,b))
    out('  PASS: only revision-colour definition/switch differ; rendered page text identical')
    out('10. Recompiled PDF page counts')
    for name,count in pages.items():out('  '+name+': '+str(count)+' pages')
    checks=re.findall(r'\[CHECK:[^]]+\]',marked+response)
    assert not checks
    out('CHECK placeholders: none')
    (ROOT/'_audit/scope_refinement_validation.txt').write_text('\n'.join(output)+'\n',encoding='utf-8')
    (ROOT/'_audit/scope_refinement_results.json').write_text(json.dumps({'checks_passed':10,'abstract_words':ac,'conclusion_words':cc,'title_words':tc,'frozen_labels':len(locations),'pages':pages,'check_placeholders':[]},indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
