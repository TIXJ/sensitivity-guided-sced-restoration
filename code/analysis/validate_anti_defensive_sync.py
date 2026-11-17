"""Check the authorized local prose edits against the pre-edit snapshot."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'first_revise_archive_20261112/anti_defensive_before_20261117'
sys.path.insert(0, str(ROOT))
from revision_response_layout import brace_argument, unmark


def read(root, name):
    return (root / name).read_text(encoding='utf-8')


def abstract(text):
    return re.search(r'\\begin\{abstract\}\[\](.*?)\\end\{abstract\}', text, re.S)[1].strip()


def conclusion(text):
    return text.split(r'\section{Conclusions}\label{sec5}', 1)[1].split(r'\printcredits', 1)[0].strip()


def main():
    lines = []
    def check(name, result):
        lines.append(f'{name}: {result}')
        assert result, name

    name = 'Manuscript_Revised_Marked.tex'
    old, new = read(BASE, name), read(ROOT, name)
    edits = [
        (' The latter changes both the rollout rule and the voltage feature, and is not an exact implementation of the original 50-step policy.', ''),
        ('The results do not justify an advantage for the 12/20 mixture; it remains one implementation of the sensitivity-guided representation, with the system-wide basis as a simpler alternative.',
         'The mixed basis is one implementation of the sensitivity-guided representation, and the system-wide basis provides a simpler alternative.'),
        ('; neither historical timing is a distribution-independent speed claim.', '.'),
        (' These timings characterize the two specified restoration procedures, rather than every possible optimization implementation.', ''),
    ]
    expected = old
    for before, after in edits:
        assert expected.count(before) == 1, before
        expected = expected.replace(before, after)
    check('Manuscript limited to the four authorized edits', expected == new)
    check('Abstract unchanged', abstract(old) == abstract(new))
    check('Conclusion unchanged', conclusion(old) == conclusion(new))
    counts = {'abstract': len(unmark(abstract(new)).split()),
              'conclusion': len(re.sub(r'\\label\{[^}]+\}', '', unmark(conclusion(new))).split())}
    lines.append(f'Word counts: {counts}')
    for environment in ('table', 'table*', 'figure', 'figure*'):
        pattern = r'\\begin\{' + re.escape(environment) + r'\}.*?\\end\{' + re.escape(environment) + r'\}'
        check(f'{environment} contents unchanged', re.findall(pattern, old, re.S) == re.findall(pattern, new, re.S))
    for filename in ('GAT_SAC_PTDF_LODF_SCED_Restoration.bib', 'revision_references.bib',
                     'Manuscript_Revised_Marked.bbl', 'Manuscript_Revised_Clean.bbl'):
        check(filename + ' unchanged', (BASE / filename).read_bytes() == (ROOT / filename).read_bytes())
    check('Bibliography remains 25 entries', len(re.findall(r'\\bibitem', read(ROOT, 'Manuscript_Revised_Marked.bbl'))) == 25)
    check('Data availability unchanged', old.split(r'\section*{Data availability}', 1)[-1] == new.split(r'\section*{Data availability}', 1)[-1])
    before = json.loads(read(BASE, 'revision_responses_resolved.json'))
    after = json.loads(read(ROOT, 'revision_responses_resolved.json'))
    changed = [k for k in before if before[k]['response'] != after[k]['response']]
    check('Only AE.11, R1.1, R3.6 and R4.1 response bodies changed', set(changed) == {'AE.11', 'R1.1', 'R3.6', 'R4.1'})
    old_reply, reply = read(BASE, 'Response_to_Associate_Editor_and_Reviewers.tex'), read(ROOT, 'Response_to_Associate_Editor_and_Reviewers.tex')
    def comments(text):
        result = []
        for match in re.finditer(r'\\rcomment\{', text):
            title, end = brace_argument(text, match.end() - 1)
            body, _ = brace_argument(text, end)
            result.append((title, body))
        return result
    check('43 original comments unchanged', comments(old_reply) == comments(reply) and len(comments(reply)) == 43)
    for phrase in ('statistical equivalence of the mixed', 'rather than showing general economic dominance', 'differ only because the procedures differ'):
        check('Removed: ' + phrase, phrase not in reply)
    bilingual = read(ROOT, '01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.tex')
    for phrase in ('statistical equivalence of the mixed', 'rather than showing general economic dominance', 'differ only because the procedures differ',
                   '这些运行不能确立混合构造与纯全系统构造之间的优越性或统计等效性', '不能说明一般性的经济优势', '三种端到端耗时的差别来自流程不同'):
        check('Bilingual removal: ' + phrase, phrase not in bilingual)
    clean = read(ROOT, 'Manuscript_Revised_Clean.tex')
    differences = [(i, a, b) for i, (a, b) in enumerate(zip(clean.splitlines(), new.splitlines()), 1) if a != b]
    check('Clean/marked differ in exactly two colouring lines', len(clean.splitlines()) == len(new.splitlines()) and len(differences) == 2)
    lines.append('Colouring differences: ' + str([x[0] for x in differences]))
    pages = {}
    for basename, directory in [('Manuscript_Revised_Marked', ROOT), ('Manuscript_Revised_Clean', ROOT),
                                ('Response_to_Associate_Editor_and_Reviewers', ROOT), ('Response_Bilingual_EN_ZH', ROOT / '01_BILINGUAL_REVIEW')]:
        with fitz.open(directory / (basename + '.pdf')) as doc:
            pages[basename] = len(doc)
            check(basename + ' no unresolved PDF markers', all('??' not in p.get_text() for p in doc))
        log = read(directory, basename + '.log')
        check(basename + ' no undefined references/citations', not re.search(r'LaTeX Warning: Reference|Citation .*undefined', log))
    check('Current source shares full AE.7 abstract', abstract(new).removeprefix(r'\rev{').removesuffix('}') in reply)
    lines.append('PDF pages: ' + str(pages))
    (ROOT / '_audit/anti_defensive_validation.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / '_audit/anti_defensive_results.json').write_text(json.dumps({'scope_passed': True, 'word_counts': counts, 'pages': pages,
        'changed_reply_keys': changed, 'colouring_difference_lines': [x[0] for x in differences],
        'response_sha256': hashlib.sha256((ROOT / 'Response_to_Associate_Editor_and_Reviewers.tex').read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    (ROOT / '_audit/anti_defensive_manuscript.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='Before', tofile='After')), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
