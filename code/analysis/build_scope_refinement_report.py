"""Publish the narrow wording-change audit without changing research records."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    results=json.loads((ROOT/'_audit/scope_refinement_results.json').read_text(encoding='utf-8'))
    changes=[
        ('Response_to_Associate_Editor_and_Reviewers.tex','Together, these findings','AE#3仅摘录Two conclusions follow.','改为完整Together句，且该句保留在新结论中，避免引文并非正文原话'),
        ('Manuscript_Revised_Marked.tex','The three restoration procedures','连续两段重复3.19轮和493.04条割','仅删除新增段中的重复句，前段数字保留一次'),
        ('Manuscript_Revised_Marked.tex','\\begin{abstract}','This advantage is conditional on the screened benchmark.','改为acceptance-matched rather than population-wide，解释失败样本无达到已接受调度的时间'),
        ('Manuscript_Revised_Marked.tex','whose 27 held-out cases','accepted by all three procedures','明确为accepted by the proposed workflow and by both reference procedures'),
        ('Manuscript_Revised_Marked.tex','The last row of Table','段末重复筛选限制及空间测试限制','删除段末筛选限制，空间测试结果保留于效率比较正文，不重复写在此段末'),
        ('Manuscript_Revised_Marked.tex','Acceptance-matched timing within','筛选定义处没有计时适用范围说明','将范围说明追加到数据集筛选定义段末'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','The 27 screened cases are accepted','Reviewer 1#1与8#1末尾重复让步','补充三种流程接受相同27例、排除接受样本不同造成的计时差异，未筛选总体另行报告'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','The screening defines the evaluated cases','Reviewer 3#2末尾双重让步','正面说明相同接受集使这组流程计时可比，但不把筛选等同于物理可接受性的决定因素'),
        ('Manuscript_Revised_Marked.tex','Two conclusions follow.','旧结论逐项重复摘要数字和空间测试说明','单段强调表示设计、相同接受集的速度收益与检查范围，保留44 of 75与27/27限定'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','\\revisedexcerpt{Abstract, p.~1}','摘要和结论摘录对应旧稿','AE#7/#10及其他逐句引文统一从当前正文重新提取'),
        ('Manuscript_Revised_Marked.tex','Time, scheduled cost and acceptance for','图9题注未说明两个总体','题注分别写明27个筛选样本(a)及60个共同接受历史样本(b)'),
        ('Manuscript_Revised_Marked.tex','Panel (b) averages only','未解释60例图点与75例主表时间不同','补充panel(b)仅取共同接受60例，因此不同于全75例均值，引用实际标签tab:review_followup_runtime'),
        ('experiments/build_runtime_cost_plot.py','Scheduled cost (10^3 cost units','纵轴仅写units','改为cost units per dispatch interval，采用一条共享纵轴标签避免两处长标签重叠'),
        ('01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.tex','Together, these findings','中英版仍为上一轮段落','同步完整摘录、221词摘要、168词结论、筛选范围说明与三条审稿人回复'),
    ]
    rows=[]
    for name,needle,before,after in changes:
        text=(ROOT/name).read_text(encoding='utf-8')
        pos=text.find(needle)
        assert pos>=0,(name,needle)
        rows.append(f'| {name}:{text[:pos].count(chr(10))+1} | `{needle}` | {before} → {after} |')
    lines=['# 本轮局部修订交付记录','',
        '## 改动清单','',
        '所有正文文字修改同时写入干净稿与标蓝稿，新增/改写文字按修订标记显示。回复信、中英版和生成脚本均已同步。没有新增实验、表格或图片；只更新已有图9的单位。','',
        '| 文件/行号 | 定位字符串 | 改前 → 改后 |','|---|---|---|',*rows,'',
        '## 保留内容','',
        '- 27例上0.788 s对3.604 s/3.075 s、3.9–4.6倍与acceptance-matched定位均保留。',
        '- 策略层9.8%增量结论保留于摘要、效率比较正文与回复。',
        '- 31个失败样本耗尽预算、44/75、筛选集范围和空间样本配对区间含零的披露均保留。仅减少重复位置。',
        '- 标题、四条亮点、既有实验数值与43条原始意见未改变。',
        '- 数据仍按合理请求提供，未声称已公开。O1/O2均未执行，未新增DOI或强制补引。','',
        '## 两处证据边界及一处引文一致性处理','',
        '- 未采用“动作表示而非学习算法是限制因素”的因果判断，改为由同种子对照支持的“动作表示是有效设计选择”。',
        '- 未采用“certified fallback”，优化补救仍按正文规定的同一接受检查执行，不声称完整交流N−1认证。',
        '- 附件同时要求AE#3引用Together句并替换整个结论，但给定新结论没有该句。为保证直接引文真实，新结论保留Together句作收尾，AE#3逐字引用。',
        '- Reviewer 3#2未采用“筛选决定哪些样本能被接受”的绝对说法，而说明筛选定义评估集合、相同接受集排除成功样本编号差异。','',
        '## 实测词数与页数','',
        f'- 摘要：{results["abstract_words"]}词，单段。',
        f'- 结论：{results["conclusion_words"]}词，单段。',
        f'- 标题：{results["title_words"]}个空格分隔词。',
        *['- '+name+': '+str(count)+'页。' for name,count in results['pages'].items()],
        '- 中英对照版：53页。','',
        '## CHECK清单','', '无。','',
        '## 十项校验实际输出','',
        '第5项在核对冻结值与AUX一致的基础上，逐条核对pdftotext所提取的起始页文字、标题或题注。长引文可能续至下一页，不将起始页核对冒充整段全在一页。正文/回复引文另做逐字源码比对。','',
        '```text',(ROOT/'_audit/scope_refinement_validation.txt').read_text(encoding='utf-8').rstrip(),'```','',
        '本轮是文稿与引文核对，不代表重新开展物理实验或独立复现实测耗时。']
    (ROOT/'_audit/Scope_Refinement_Changes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote _audit/Scope_Refinement_Changes.md')


if __name__=='__main__':
    main()
