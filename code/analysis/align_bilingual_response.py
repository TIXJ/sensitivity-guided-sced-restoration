"""Refresh location pointers while retaining the reviewed Chinese translations."""
import re
from build_bilingual_response import OUT, WORK, SOURCE, prepare, strip_zh, build
import sys
sys.path.insert(0, str(SOURCE.parent))
from revision_response_layout import brace_argument, simple_response_format


def chinese_location(line):
    line = line.replace('These revisions are documented in ', '相应修改见')
    line = re.sub(r'the paragraph on p\.~(\\pageref\{[^}]+\})',r'第~\1~页的对应段落',line)
    line = re.sub(r'Section~(\\ref\{[^}]+\})',r'第~\1~节',line)
    text = line.replace(r'\textbf{Location in the revised manuscript:}', r'\textbf{修订稿中的位置：}')
    text = text.replace(r'\textit{Changes in the manuscript:}', r'\textit{修订稿中的修改位置：}')
    for en, zh in [('the paragraph on p.~','段落所在页：'),('Paragraph on p.~', '段落所在页：'), ('Sections~','节~'), ('Section~','节~'),
                   ('Tables~','表~'),('Table~','表~'),('Figures~','图~'),('Figure~','图~'),
                   ('Equations~','公式~'),('Equation~','公式~')]:
        text = text.replace(en, zh)
    text = re.sub(r'\(p\.~(\\pageref\{[^}]+\})\)', r'（第~\1~页）', text)
    return text.replace(', and ', '，以及').replace(' and ', '和').replace('; ', '，').replace(r'.\par', '。')


def main():
    old = [(WORK / f'part_{i}_english.tex').read_text(encoding='utf-8') for i in range(1,6)]
    bilingual = [(WORK / f'part_{i}_bilingual.tex').read_text(encoding='utf-8') for i in range(1,6)]
    prepare()
    old_sentence = 'The manuscript distinguishes this representation from action masking and from generator-wise policy outputs.'
    addition = ' The contribution list now presents the sensitivity-guided representation first, followed by the correction framework and its implementation.'
    for i, (previous, translated) in enumerate(zip(old,bilingual), 1):
        title_updates = (
            ('Sensitivity-Guided Low-Dimensional Redispatch for Base-Case Alternating-Current Feasibility and Linear Contingency Screening',
             'Sensitivity-Guided Low-Dimensional Redispatch for Base-Case AC Feasibility and Linear N-1 Contingency Screening'),
            ('We have revised the title to distinguish base-case alternating-current feasibility from linear contingency screening, without abbreviations.',
             'We have revised the title to distinguish base-case AC feasibility from linear N-1 contingency screening, retaining the conventional AC abbreviation and N-1 notation.'),
            ('We have revised the title to distinguish base-case alternating-current feasibility from linear contingency screening.',
             'We have revised the title to distinguish base-case AC feasibility from linear N-1 contingency screening.'),
            ('We revised the title to limit the stated scope to base-case alternating-current feasibility and linear contingency screening.',
             'We revised the title to limit the stated scope to base-case AC feasibility and linear N-1 contingency screening.'),
        )
        for before, after in title_updates:
            previous = previous.replace(before,after)
            translated = translated.replace(before,after)
        translated = translated.replace('面向基态交流可行性与线性故障筛查的灵敏度引导低维再调度',
                                        '面向基态AC可行性与线性N-1故障筛查的灵敏度引导低维再调度')
        translated = translated.replace('我们已修改标题，在不使用缩略语的情况下区分基态交流可行性与线性故障筛查。',
            '我们已修改标题，明确区分基态AC可行性与线性N-1故障筛查，并保留通用缩写AC及N-1表示法。')
        translated = translated.replace('我们已修改标题，以区分基态交流可行性与线性故障筛查。',
            '我们已修改标题，以区分基态AC可行性与线性N-1故障筛查。')
        translated = translated.replace('我们修改了标题，将所述范围限定为基态交流可行性与线性故障筛查。',
            '我们修改了标题，将所述范围限定为基态AC可行性与线性N-1故障筛查。')
        translated = translated.replace('我们已修改标题，将所述范围限定为基态交流可行性和线性事故筛查。',
            '我们已修改标题，将所述范围限定为基态AC可行性和线性N-1故障筛查。')
        mapping_note = 'Reviewer numbering follows the supplied compilation: Reviewers 1--4 correspond to original reviewer IDs 1, 3, 6 and 8, respectively.'
        previous = previous.replace(mapping_note, '')
        translated = translated.replace(mapping_note, '')
        translated = translated.replace(r'\zh{审稿人编号遵循所提供的汇编：审稿人1--4分别对应原始审稿人编号1、3、6和8。}', '')
        previous = re.sub(r' \(original reviewer ID \d+\)', '', previous)
        translated = re.sub(r' \(original reviewer ID \d+\)', '', translated)
        translated = re.sub(r'（原审稿人编号\d+）', '', translated)
        previous = simple_response_format(previous)
        translated = simple_response_format(translated)
        current = (WORK / f'part_{i}_english.tex').read_text(encoding='utf-8')
        # Page reflow can change which table references share one page pointer.
        location_pattern = r'These revisions are documented in [^\n]+?\.\\par'
        old_locations = re.findall(location_pattern,translated)
        new_locations = re.findall(location_pattern,current)
        assert len(old_locations)==len(new_locations)
        for before,after in zip(old_locations,new_locations):
            if before==after:
                continue
            at = translated.index(before)
            zh_start = translated.index(r'\zh{',at+len(before))
            zh,zh_end = brace_argument(translated,zh_start+3)
            zh,count = re.subn(r'相应修改见[^。]*。',lambda m:chinese_location(after),zh,count=1)
            assert count==1,('Missing translated location sentence',i,before)
            translated = translated[:zh_start]+r'\zh{'+zh+'}'+translated[zh_end:]
            translated = translated.replace(before,after,1)
        review_addition = ' These additions now form a fourth literature-review topic on safe and physics-informed learning.'
        review_anchor = 'Existing journal-specific dispatch references are retained.'
        if review_addition in current and review_addition not in translated:
            at = translated.index(review_anchor)
            zh_start = translated.index(r'\zh{',at+len(review_anchor))
            zh, zh_end = brace_argument(translated,zh_start+3)
            addition_zh = '新增文献现已归入第四个综述点，即安全学习与物理信息学习。'
            zh = zh.replace('相应修改见',addition_zh+'相应修改见',1) if '相应修改见' in zh else zh+addition_zh
            translated = translated[:zh_start]+r'\zh{'+zh+'}'+translated[zh_end:]
            translated = translated.replace(review_anchor,review_anchor+review_addition,1)
        for quote_id in ('safe_learning','related_correction'):
            heading = r'p.~\pageref{ms-reply:quote_'+quote_id+'}'
            prefix = r'\revisedexcerpt{'+heading+'}'
            if prefix not in previous:
                continue
            old_start = translated.index(prefix)
            _, old_end = brace_argument(translated,old_start+len(prefix))
            old_command = translated[old_start:old_end]
            new_start = current.index(prefix)
            new_excerpt, new_end = brace_argument(current,new_start+len(prefix))
            new_command = current[new_start:new_end]
            if old_command==new_command:
                continue
            citations = re.findall(r'\[(\d+)\]',new_excerpt)
            assert len(citations)==2,(quote_id,citations)
            if quote_id=='safe_learning':
                zh = ('安全强化学习关注策略优化与部署过程中的约束违反问题；Su等['+citations[0]+']按约束处理机制和电力系统应用对这些方法进行梳理。'
                      'Wu等['+citations[1]+']将原始-对偶策略优化与基于先验知识的动作屏蔽相结合，用于实时最优潮流。其动作屏蔽用于识别关键动作维度。')
            else:
                zh = ('Lopez-Garcia和Dominguez-Navarro['+citations[0]+']使用类型化图网络和物理信息损失函数，在不依赖有监督调度标签的情况下求解最优潮流。'
                      'Liu等['+citations[1]+']使用物理信息图卷积网络，将直流解向交流最优潮流解进行修正。'
                      '这些方法侧重约束处理或物理信息预测，而本文考察用于顺序修正的固定平衡再调度表示。')
            at = translated.index(old_command)
            zh_start = translated.index(r'\zh{',at+len(old_command))
            assert not translated[at+len(old_command):zh_start].strip()
            _, zh_end = brace_argument(translated,zh_start+3)
            translation = '修订稿第~'+r'\pageref{ms-reply:quote_'+quote_id+'}'+ '~页的相应文字修改如下：'+zh
            translated = translated[:at]+new_command+'\n\n'+r'\zh{'+translation+'}'+translated[zh_end:]
        old_pointers = re.findall(r'^(?:See the excerpt already reproduced in response|The (?:two|\d+) relevant passages are reproduced in Response|The corresponding revisions? appears? on manuscript)[^\n]*\\par', translated, re.M)
        new_pointers = re.findall(r'^The corresponding revisions? appears? on manuscript [^\n]*\\par', current, re.M)
        if old_pointers:
            assert len(old_pointers)==len(new_pointers)
            for before, after in zip(old_pointers,new_pointers):
                if before==after:
                    continue
                at = translated.index(before)
                zh_start = translated.index(r'\zh{',at+len(before))
                assert not translated[at+len(before):zh_start].strip()
                _, zh_end = brace_argument(translated,zh_start+3)
                pages = re.search(r'on manuscript pp?\.~(.+?) and (?:is|are) quoted',after)[1]
                comment = re.search(r'Comment ([A-Z0-9.]+)\.\\par',after)[1]
                zh = '相应修改见修订稿第~'+pages+'~页，具体文字已在对意见'+comment+'的回复中引用。'
                translated = translated[:at]+after+'\n\n'+r'\zh{'+zh+'}'+translated[zh_end:]
        # Add the requested opening to the English reply and its first translation.
        recipient = 'the Associate Editor' if i == 1 else 'the reviewer'
        thanks = f'We thank {recipient} for this comment. '
        zh_thanks = '感谢副主编提出这一意见。' if i == 1 else '感谢审稿人提出这一意见。'
        for match in reversed(list(re.finditer(r'\\aresponse\{', translated))):
            _, heading_end = brace_argument(translated, match.end()-1)
            body, body_end = brace_argument(translated, heading_end)
            if body.startswith(thanks):
                continue
            zh_start = translated.index(r'\zh{', heading_end + 1)
            assert zh_start < body_end
            translated = translated[:zh_start+4] + zh_thanks + translated[zh_start+4:]
            translated = translated[:heading_end+1] + thanks + translated[heading_end+1:]
        a = re.findall(r'^\\textit\{Changes in the manuscript:\}[^\n]+', previous, re.M)
        b = re.findall(r'^\\textit\{Changes in the manuscript:\}[^\n]+', current, re.M)
        assert len(a) == len(b)
        for before, after in zip(a,b):
            if before == after: continue
            at = translated.index(before)
            zh_start = translated.index(r'\zh{', at + len(before))
            assert not translated[at+len(before):zh_start].strip()
            _, zh_end = brace_argument(translated, zh_start + 3)
            translated = translated[:at] + after + '\n\n\\zh{' + chinese_location(after) + '}' + translated[zh_end:]
        for match in re.finditer(r'\\revisedexcerpt\{', previous):
            _, first_end = brace_argument(previous, match.end()-1)
            _, command_end = brace_argument(previous, first_end)
            command = previous[match.start():command_end]
            if command in current:
                continue
            if command not in translated:
                continue
            at = translated.index(command)
            start = translated.index(r'\zh{', at + len(command))
            assert not translated[at+len(command):start].strip()
            _, end = brace_argument(translated, start+3)
            translated = translated[:at] + translated[end:]
        if i == 1 and old_sentence + addition not in previous:
            assert old_sentence + addition in current
            at = translated.index(old_sentence)
            start = translated.index(r'\zh{', at)
            zh, end = brace_argument(translated, start + 3)
            translated = translated[:start] + r'\zh{' + zh + '贡献列表现已首先介绍灵敏度引导表示，随后介绍校正框架及其实现。}' + translated[end:]
            translated = translated.replace(old_sentence, old_sentence + addition, 1)
        assert re.sub(r'\s+','', strip_zh(translated)) == re.sub(r'\s+','',current), f'Untranslated source change in part {i}'
        (WORK / f'part_{i}_bilingual.tex').write_text(translated,encoding='utf-8')
    build()


if __name__ == '__main__':
    main()
