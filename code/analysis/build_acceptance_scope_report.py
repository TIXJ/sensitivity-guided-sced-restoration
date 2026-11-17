"""Write the author-facing change list and complete validation transcript."""
from pathlib import Path
import json
import sys

import fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision_acceptance_scope import TITLE, ABSTRACT


def main():
    changes=[
        ('Manuscript_Revised_Marked.tex','\\title[mode=title]', 'Base-Case AC Feasibility','Base-Case Alternating-Current Feasibility，12词'),
        ('Manuscript_Revised_Marked.tex','\\begin{abstract}', '197词摘要并列不同版本的负面结论','214词单段摘要，区分27例条件计时、44/75总体接受和最终八步流程'),
        ('Manuscript_Revised_Marked.tex','This section asks a different question','未明确桥接两种优化比较','新增共同优化器下策略路由增益的评估问题'),
        ('Manuscript_Revised_Marked.tex','The three restoration procedures','仅直接给出速度倍数','先说明三种流程均接受同一27例，再解释参考流程工作'),
        ('Manuscript_Revised_Marked.tex','On this acceptance-matched set','速度倍数未在句首限定总体','4.57/3.90倍明确限于27例接受集匹配比较，成本增加2.06%保留'),
        ('Manuscript_Revised_Marked.tex','The last row of Table','3.208 s行缺少预算耗尽解释','31例各跑满50步，23.640步为75例均值，4.704步限于27例子集'),
        ('Manuscript_Revised_Marked.tex','runtime_cost_feasibility.pdf','旧图未进入正文','恢复两联图，分别呈现27例与最终60例，禁止跨面板速度推断'),
        ('Manuscript_Revised_Marked.tex','Two conclusions follow.','结论偏重并列限制','单段给出表示贡献、条件性计时收益和部署含义，同时保留44/75与交流安全边界'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','Summary of Revisions','无集中修改概述','增加四项修改概述及当前图表编号'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','Reviewer numbers follow','汇编编号1/2/3/4','平台编号1/3/6/8，重复回复指向同步转换'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','\\aresponse{Response \\#2:}', '母语审阅已完成','保留已完成审阅，不编造证书或补充材料迁移'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','The revised title contains','旧标题与亮点数量','AE.3/5更新标题，AE.6四条亮点，AE.7/10直接引用新摘要与结论'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','The contribution is the fixed','核心表示贡献与速度评估连接较弱','AE.11区分表示家族、12/20实现选择及条件性速度贡献，不声称统计等效'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','We thank the Associate Editor and the reviewers again','无简短收尾','增加保留失败记录及可提供核查材料的收尾'),
        ('Cover_Letter_Revised.tex','The revision separates','一般性返修介绍','区分两套计时协议，仅对27例主张3.9–4.6倍'),
        ('Highlights_Revised.docx','Highlights','三条方法性亮点','按要求替换为四条，字符数74/78/77/77'),
        ('01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.tex','Summary of Revisions','旧英文及中文段落','英文逐字对应正式回复，新增段落、图说明和平台编号同步翻译'),
    ]
    rows=[]
    for name,needle,before,after in changes:
        path=ROOT/name
        line='标题/要点'
        if path.suffix=='.tex':
            source=path.read_text(encoding='utf-8')
            index=source.find(needle)
            assert index>=0,(name,needle)
            line=str(source[:index].count('\n')+1)
        rows.append(f'| {name}:{line} | {before} | {after} |')
    pages={n:len(fitz.open(ROOT/(n+'.pdf'))) for n in
        ('Manuscript_Revised_Clean','Manuscript_Revised_Marked','Response_to_Associate_Editor_and_Reviewers','01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH')}
    report=['# 本轮改动与校验报告','',
        '## 改动清单','',
        '下列正文修改均同步到Clean与Marked及提交目录。新增整段摘要和结论按本轮要求整段标蓝，其他位置沿用细粒度标记。原始43条意见、实验记录、前三点综述、贡献列表及参考文献未改变。','',
        '| 文件及行号/锚点 | 改前 | 改后 |','|---|---|---|',*rows,'',
        '## 未逐字采用的证据性表述','',
        '- 标题实际为12个空格分隔词，不是13词，仍在12–15范围内。',
        '- 作者确认没有语言证书，仅写母语审阅已完成。不声称有两张表移入补充材料。',
        '- 4.704步只对应27例筛选子集，不改写成全部44例已接受样本均值。',
        '- 接受集匹配不消除筛选的选择效应，不能写“不受筛选保留哪些样本影响”。',
        '- 27例相同接受数不意味着其他指标相同，成本差2.06%继续保留。',
        '- 最终八步门控流程与原50步策略不是同一部署配置，不能把两个计时协议写成完全相同的流程。',
        '- 三种子结果不能确立混合与纯全系统构造的统计等效性。空间节时区间含零也不能解释为收益恰为零。',
        '- 后备优化仍受同一有限接受检查约束，不写成具有完整交流N−1认证的“certified fallback”。',
        '- 摘要明确仅有功沿低维方向校正，电压设定值独立控制。','',
        '## Tier B','',
        '| 项目 | 状态 | 依据 |','|---|---|---|',
        '| B1：41例配对计时 | 跳过 | 有逐例记录，但SAC诊断/准备与缓存复用口径和预防优化不同，且无可分离的审计开销计时。未估算扣除，也未新增统计结论。 |',
        '| B2：时间—成本—可行性图 | 完成 | 直接读取原27例逐例时间与成本，及最终两流程共同60例的三次重复。先按case平均，再汇总。没有新物理实验。 |',
        '| B3：强制补引另外两篇 | 跳过 | 库内literature_verification.json已记录其月面无线能源与多能源网络韧性范围，非直接调度恢复对照。保留六篇均审阅、四篇直接相关者已引用的准确回复，不强行增加引用或声称六篇全已引用。 |','',
        '图的完整来源、字段、caseID、哈希与聚合方法见runtime_cost_plot_evidence.json。图中60例均值不替代75例主表的1.912/1.725 s。','',
        '## CHECK清单','', '无。未对可选B1缺失的统一计时协议插入正文占位符，直接跳过该项。','',
        '## 新PDF页数','',*['- '+name+': '+str(count)+'页。' for name,count in pages.items()],'',
        '## 七项校验完整输出','',
        '第4项同时核对AUX、冻结值与pdftotext分页后的原文开头/标题。长引文跨页时核对其起始页，不虚称整段均在一页。正文与引文的逐字源码一致性另由check_revision_artifacts.py核对。','',
        '```text',(ROOT/'_audit/acceptance_scope_validation.txt').read_text(encoding='utf-8').rstrip(),'```','',
        '这些检查属于文稿、引用与既有记录一致性检查，不是新增物理实验或独立复现实测耗时。']
    (ROOT/'_audit/Acceptance_Scope_Changes.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (ROOT/'_audit/acceptance_scope_delivery.json').write_text(json.dumps({'title':TITLE,'abstract_words':len(ABSTRACT.split()),'pages':pages,'check_placeholders':[],'tier_b':{'B1':'skipped_incompatible_timing','B2':'completed_from_saved_records','B3':'skipped_not_directly_relevant'}},indent=2),encoding='utf-8')
    print(json.dumps(pages,ensure_ascii=False))


if __name__=='__main__':
    main()
