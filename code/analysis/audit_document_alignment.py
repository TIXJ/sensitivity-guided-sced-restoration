"""Reverse-map revision marks and changed objects to reviewer responses.

Topic links are explicit editorial mappings, not claims of verbatim quotation.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision_response_layout import brace_argument, unmark

TOPICS = {
    'front': ['AE.3', 'AE.5', 'AE.7', 'R1.2', 'R1.4'],
    'Introduction': ['AE.1', 'AE.11', 'AE.12', 'R1.4', 'R1.5', 'R3.7'],
    'Sensitivity-Guided Low-Dimensional Redispatch': ['AE.11', 'R2.10', 'R2.11'],
    'Training Procedure': ['R2.8', 'R2.12'],
    'Evaluation Protocol': ['R3.6'],
    'Experimental Settings': ['AE.9', 'R1.1', 'R2.4', 'R2.7', 'R3.6'],
    'Overall Restoration Performance': ['R1.1', 'R1.2', 'R2.1', 'R2.2', 'R3.2', 'R4.1'],
    'Sensitivity-Guided Active-Power Redispatch Ablation': ['AE.11', 'R1.5', 'R2.10', 'R2.11'],
    'RL Algorithm Comparison': ['R1.3', 'R2.9', 'R2.13', 'R3.3', 'R4.3'],
    'Deployment Time and Economic Comparison': ['R2.6', 'R3.6'],
    'Operating Conditions and Security Boundaries': ['R2.3', 'R2.5', 'R2.7', 'R2.8', 'R2.12', 'R3.4', 'R3.6', 'R4.2'],
    'Additional IEEE 39-bus and 118-bus Experiments': ['R1.5', 'R3.1', 'R3.5'],
    'Conclusions': ['AE.3', 'AE.10', 'R1.2', 'R4.1'],
}


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'\\label\{[^}]+\}', '', unmark(s))).strip()


def main():
    manuscript = (ROOT / 'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    baseline = (ROOT / 'GAT_SAC_PTDF_LODF_SCED_Restoration.tex').read_text(encoding='utf-8')
    evidence = json.loads((ROOT / 'response_evidence.json').read_text(encoding='utf-8'))
    letter = (ROOT / 'Response_to_Associate_Editor_and_Reviewers.tex').read_text(encoding='utf-8')
    aux = (ROOT / 'Manuscript_Revised_Marked.aux').read_text(encoding='utf-8')
    positions = {m[1]: {'number': m[2], 'page': m[3]} for m in re.finditer(
        r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]+)\}', aux)}
    links = {}
    for comment, item in evidence['comments'].items():
        for label in item['location_labels'] + item['reproduced_tables'] + item['reproduced_figures']:
            links.setdefault(label, set()).add(comment)
        for quote in item['excerpt_ids']:
            links.setdefault('reply:quote_' + quote, set()).add(comment)
    old_labels = set(re.findall(r'\\label\{([^}]+)\}', baseline))
    new_labels = set(re.findall(r'\\label\{([^}]+)\}', manuscript))
    added_objects = sorted(x for x in new_labels - old_labels if x.startswith(('tab:', 'fig:', 'eq:', 'sec:')))
    marked_objects = []
    for match in re.finditer(r'\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}', manuscript, re.S):
        labels = re.findall(r'\\label\{([^}]+)\}', match[0])
        if labels and r'\rev{' in match[0]: marked_objects.append(labels[-1])
    objects = []
    for label in sorted(set(added_objects + marked_objects)):
        assert label in links, ('Missing response pointer', label)
        assert '{ms-' + label + '}' in letter
        objects.append({'label': label, **positions[label], 'responses': sorted(links[label]),
                        'kind': 'added' if label in added_objects else 'modified_original'})
    blocks = []
    for i, match in enumerate(re.finditer(r'\\rev\{', manuscript), 1):
        content, end = brace_argument(manuscript, match.end() - 1)
        headings = list(re.finditer(r'\\(?:sub)*section\*?\{([^}]+)\}', manuscript[:match.start()]))
        heading = headings[-1][1] if headings else 'front'
        exact = set()
        fragment = norm(content)
        if len(fragment) >= 40:
            for key, excerpt in evidence['verbatim_excerpts'].items():
                if fragment in norm(excerpt) or norm(excerpt) in fragment:
                    exact.update(links.get('reply:quote_' + key, ()))
        para_start = manuscript.rfind('\n\n', 0, match.start()) + 2
        para_end = manuscript.find('\n\n', end)
        paragraph = manuscript[para_start:para_end if para_end >= 0 else len(manuscript)]
        referenced = set(re.findall(r'\\(?:ref|label)\{([^}]+)\}', paragraph))
        contextual = set().union(*(links.get(x, set()) for x in referenced)) if referenced else set()
        topic = TOPICS.get(heading, [])
        author_correction = heading=='Funding' and fragment=='ZDKJXM20250028'
        assert exact or contextual or topic or author_correction, ('Unmapped revision', heading, fragment[:80])
        blocks.append({'id': i, 'section': heading, 'text': fragment,
                       'verbatim_response_ids': sorted(exact),
                       'object_or_paragraph_response_ids': sorted(contextual),
                       'editorially_assigned_topic_response_ids': topic,
                       'author_requested_administrative_correction': author_correction})
    report = {'scope': 'Text/object traceability, not an independent reproduction of experiments. Topic mappings are not verbatim quotations.',
              'comments': len(evidence['comments']), 'revision_blocks': len(blocks),
              'added_or_marked_objects': len(objects), 'unmapped_objects': [],
              'objects': objects, 'revisions': blocks,
              'manuscript_sha256': hashlib.sha256(manuscript.encode()).hexdigest(),
              'response_sha256': hashlib.sha256(letter.encode()).hexdigest(),
              'additional_change': 'Contribution order is described in AE.11; no new empirical claim.'}
    (ROOT / 'Document_Alignment_Audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = ['# 正文与回复信对齐核查', '',
            '本记录区分逐字摘录、图表/段落定位和主题对应。不要求把正文全部重复写入回复信。', '',
            f'- 原意见及回复：{len(evidence["comments"])}条。',
            f'- 修订标记：{len(blocks)}处（包括细粒度词组）。',
            f'- 新增或标蓝图表/编号对象：{len(objects)}个，均有直接回复定位。',
            '- 本轮更新了条件性计时表述，并从既有记录恢复时间—成本图；原始实验数值未改动。', '',
            '| 正文对象 | 页码 | 对应回复 |', '|---|---|---|']
    rows += [f'| {x["label"]} ({x["number"]}) | {x["page"]} | {", ".join(x["responses"])} |' for x in objects]
    rows += ['', '逐处修订的文本和对应关系见 Document_Alignment_Audit.json。排版、字体和元数据调整不作为新增科学结论逐段引用。']
    (ROOT / 'Document_Alignment_Index.md').write_text('\n'.join(rows), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('comments','revision_blocks','added_or_marked_objects','unmapped_objects')}))


if __name__ == '__main__':
    main()
