"""Synchronize the eight author annotations without retranslating unchanged replies."""
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision_response_layout import brace_argument
from revision_annotation_updates import smooth_response
from build_bilingual_response import WORK, prepare, build
from align_bilingual_response import chinese_location

BASE=ROOT.parent/'first_revise_archive_20261112/anti_defensive_before_20261117/01_BILINGUAL_REVIEW/_translation_work'
if not BASE.exists():BASE=WORK
RESPONSES={
    'AE.5':'感谢副主编提出这一意见。我们已修改标题，明确区分基态AC可行性与线性N-1故障筛查。修订稿第1页的标题引用如下。',
    'R1.1':'感谢审稿人提出这一意见。我们现已在同一组完整且未经筛选的75个测试样本上评估所有比较方法。五个需求水平各包含15例，而筛选子集仅覆盖前三个水平。策略和选择规则均在评估前固定。各方法的通过数分别为SAC-32D 44/75、SAC-68D 30/75、DDPG 43/75、TD3 45/75、基态AC-OPF 0/75、预防性LODF-SCOPF 41/75，以及确定性修复28/75。原27例筛选子集的比较保留为条件性结果。不同训练种子的比较另行报告。',
    'R1.2':'感谢审稿人提出这一意见。我们已在摘要、主要性能表、恢复比较表和结论中补充原始策略的44/75结果，并明确27/27仅适用于筛选子集。对应的修改语句和表格条目引用如下。',
    'R1.4':'感谢审稿人提出这一意见。我们已修改标题，明确区分基态AC可行性与线性N-1故障筛查。摘要与问题表述也采用相同的区分。修改后的标题引用如下。',
}
from revision_reply_review import REPLIES
from build_revision_documents import esc
for _key,(_,translation) in REPLIES.items():
    recipient='副主编' if _key.startswith('AE.') else '审稿人'
    RESPONSES[_key]='感谢'+recipient+'提出这一意见。'+translation


def tokens(text):
    """Separate translatable prose/objects from response-container syntax."""
    at=0
    while at<len(text):
        white=re.match(r'\s+',text[at:])
        if white:
            yield 'literal',white[0]
            at+=len(white[0]);continue
        if text.startswith(r'\aresponse{',at):
            _,end=brace_argument(text,at+len(r'\aresponse'))
            body,stop=brace_argument(text,end)
            yield 'literal',text[at:end+1]
            yield from tokens(body)
            yield 'literal','}'
            at=stop;continue
        for command,arity in ((r'\zh',1),(r'\rcomment',2),(r'\revisedexcerpt',2)):
            if text.startswith(command+'{',at):
                end=at+len(command)
                for _ in range(arity):_,end=brace_argument(text,end)
                yield ('zh' if command==r'\zh' else 'unit'),text[at:end]
                at=end;break
        else:
            environment=next((name for name in ('bluetableblock','quote')
                              if text.startswith(r'\begin{'+name+'}',at)),None)
            if environment:
                close=r'\end{'+environment+'}'
                end=text.index(close,at)+len(close)
            else:
                ends=[x for x in (text.find('\n\n',at),text.find(r'\zh{',at)) if x>=0]
                end=min(ends) if ends else len(text)
            value=text[at:end]
            yield ('literal' if value.strip()==r'\newpage' else 'unit'),value
            at=end


def clean_location(text):
    match=re.search(r'\s*These revisions are documented in [^\n]+?\.\\par\s*$',text)
    return (text[:match.start()],match[0].strip()) if match else (text,'')


def norm(text):
    text=clean_location(text)[0]
    text=smooth_response(text)
    return re.sub(r'\s+','',text).lower()


def quote_translation(unit):
    heading,end=brace_argument(unit,len(r'\revisedexcerpt'))
    value,_=brace_argument(unit,end)
    if heading=='Abstract, p.~1' and 'The policy receives' in value:
        return ('摘要第1页修改如下：随着需求、成本和运行条件变化，电力系统滚动运行需要反复求解安全约束经济调度。'
            '本文利用图注意力网络预测线性安全约束调度，将预测投影到功率平衡和发电机边界内，再利用软演员--评论家算法，沿由线路潮流灵敏度与故障后梯度构造的固定平衡再调度方向校正有功功率，并单独控制电压设定值。'
            '策略每步接收线路开断分布因子反馈，并定期刷新交流潮流反馈。独立校核器结合基态交流检查与线性故障筛查决定是否接受。'
            '在300节点匹配校正基准上，27个留出样本经动作基条件筛选得到，本文流程及两种参考流程均接受这些样本。本文流程平均0.788 s得到一个已接受调度，确定性动作张成空间修复和预防性安全约束优化分别为3.604和3.075 s，相同接受结果下为3.9至4.6倍。'
            '该比较具有相同接受集，而非覆盖整个样本总体。策略无法恢复的候选样本耗尽步数预算，因此不存在可比较的达到已接受调度的时间。在全部75个未筛选候选样本上，策略接受44例。'
            r'与匹配的联合优化器相比，最终八步门控流程在历史工况上节时9.8\%，但该增益未在所测试的空间扰动工况上重现。'
            'IEEE 39节点和118节点系统的试验采用系统特定实现，评估相同流程。'
            '在每个训练种子下，灵敏度引导方向均优于同维度随机方向，支持固定灵敏度引导校正接口。')
    if value.startswith('Over all 75 unfiltered candidates'):
        return '摘要第1页补充：在全部75个未筛选候选样本上，策略接受44例。'
    if value.startswith('Over the unfiltered population'):
        return r'修订稿第~\pageref{ms-sec5}~页的结论说明：在未筛选总体上，策略接受75个候选样本中的44个。27/27仅适用于筛选子集。'
    if value.startswith('Together, these findings'):
        return (r'修订稿第~\pageref{ms-sec5}~页的结论说明：综合而言，这些发现支持在已评估运行范围内采用学习校正，'
            '同时保留按相同接受检查进行的优化补救，并在需要故障后可行性时执行显式交流故障筛查。')
    if value.startswith('Two conclusions follow.'):
        return (r'修订稿第~\pageref{ms-sec5}~页的结论修改如下：可得出两个结论。首先，有功动作表示是纠正调度中有效的设计选择。在每个训练种子下，灵敏度引导方向均优于同维度随机方向。'
            '在接受集匹配的基准上，本文流程平均0.788 s返回调度，而两种恢复参考流程为3.075--3.604 s，在相同27例上具有3.9--4.6倍速度优势。'
            '在未筛选总体上，策略接受75个候选样本中的44个。27/27仅适用于筛选子集。'
            '与匹配的优化器相比，最终八步策略层在历史工况上仍节省约十分之一的端到端时间，但这一节时收益未延伸到新的空间扰动工况。'
            '其次，部署取决于接受结果及检查范围。策略无法恢复的候选样本会耗尽步数预算，而非返回已接受的调度。'
            '显式故障后审计表明，基态可行性与线性筛查不能确立交流故障安全性。'
            '综合而言，这些发现支持在已评估运行范围内采用学习校正，同时保留按相同接受检查进行的优化补救，并在需要故障后可行性时执行显式交流故障筛查。')
    if 'quote_screening' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_screening}~页说明：筛选后的校正集仅保留16.0、16.2和16.4 GW三个需求水平。'
            '筛选采用所提出的动作张成空间，因此局限于该集合的比较属于条件性基准，不能确立该张成空间的无条件优势。'
            '未筛选评估保留原测试编号，包括筛选拒绝的样本。在筛选集内进行的接受集匹配计时具有相同的条件限制，因此不推广至未筛选总体。')
    if heading=='Title, p.~1':
        return '修订稿第1页的标题修改为：面向基态交流可行性与线性N-1故障筛查的灵敏度引导低维再调度。'
    if 'quote_safe_learning' in heading:
        citations=re.findall(r'\[(\d+)\]',value)
        assert len(citations)==4
        a,b,c,d=citations
        return (r'修订稿第~\pageref{ms-reply:quote_safe_learning}~页的第四点综述修改如下：'
            f'安全强化学习关注策略优化与部署过程中的约束违反问题。Su等[{a}]按约束处理机制和电力系统应用对这些方法进行梳理。'
            f'Wu等[{b}]将原始-对偶策略优化与基于先验知识的动作屏蔽相结合，用于实时最优潮流，其动作屏蔽用于识别关键动作维度。'
            f'Lopez-Garcia和Dominguez-Navarro[{c}]使用类型化图网络和物理信息损失函数，在不依赖有监督调度标签的情况下求解最优潮流。'
            f'Liu等[{d}]使用物理信息图卷积网络，将直流解向交流最优潮流解进行修正。'
            '这些方法侧重约束处理或物理信息预测，而本文考察用于顺序修正的固定平衡再调度表示。')
    if 'quote_contributions' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_contributions}~页的三项贡献完整引用如下：'
            r'\begin{enumerate}\item 一种灵敏度引导的平衡再调度子空间，用固定低维坐标替代逐机有功输出，约束策略在恢复前提出的动作，同时单独保留发电机电压控制。'
            r'\item 一种校核器参与的纠正调度框架，旨在通过连续的反馈驱动校正，将快速线性SCED预测调整为满足基态AC-PF可行性与所采用LODF故障判据的运行点。'
            r'\item 一种校正环境实现，将交流运行结果、LODF越限量及与动作对齐的线路潮流特征，与独立于策略的校核协议及面向恢复的优化基线相结合。\end{enumerate}')
    if value.startswith('The reference policy restores 44/75'):
        return '摘要第1页补充：参考策略使75个上游300节点算例中的44个恢复可行，三个训练种子恢复41--45例。'
    if value.startswith('The reference policy restores 44 of the 75'):
        return r'修订稿第~\pageref{ms-sec5}~页的结论补充：参考策略恢复75个未筛选测试样本中的44个。27/27仅适用于筛选子集。'
    if heading=='Abstract, p.~1' and value.startswith('Security-constrained') and 'The policy receives' not in value:
        return ('摘要第1页修改如下：随着需求、成本和运行条件的变化，电力系统滚动运行需要反复求解安全约束经济调度。'
                '本文利用图注意力网络预测线性安全约束调度，将预测投影到功率平衡和发电机边界内，'
                '再利用软演员--评论家算法，沿由线路潮流灵敏度与故障后梯度构造的固定平衡再调度方向调整有功功率，并单独控制电压设定值。')
    if value.startswith('The results show that fixed') and value.count('. ')==0:
        return (r'修订稿第~\pageref{ms-sec5}~页的结论开头为：结果表明，固定的、基于灵敏度的有功功率方向'
                '可与软演员--评论家算法结合，用于校正重复求解的安全约束经济调度预测。')
    if 'quote_tight_retraining' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_tight_retraining}~页补充：一项单独的匹配重训练采用兼容的0.94--1.08 pu范围，'
                '保留原训练与验证编号、种子42、100轮外层迭代及原32维、40 MW动作接口，不重新筛选或调节奖励。'
                '验证集所选策略在75个上游工况中通过0例，计入失败后平均为50.00步。该单次重训练与原固定策略测试分开报告。')
    if 'quote_tight_joint' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_tight_joint}~页补充：兼容0.94--1.08 pu范围下的联合优化测试扩展至原全部75例，采用潮流初始化。'
                '60/75通过，计入失败后的平均端到端时间为1.207 s。无控制边界不兼容工况，15次求解不收敛，最终校核未拒绝任何返回候选点。'
                '这些通过点也满足原更宽电压范围，因此不同优化器配置的通过差异不能解释为原范围下物理不可行。这是更宽控制范围的优化结果，而非重训练策略结果。')
    if 'quote_tight_compatibility' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_tight_compatibility}~页补充：此前0.95--1.05 pu固定策略运行通过0/75，平均2.108 s。'
                '由于保留的电压控制区间与联合模型对全部受控母线施加该范围不兼容，此结果不进入鲁棒性排名。'
                '该结果同时涉及约束兼容性与策略适应性，故保留为单独诊断，不能仅作为策略表现不足的证据。')
    if 'quote_final_runtime_result' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_final_runtime_result}~页补充：表~\ref{ms-tab:review_followup_runtime}~显示历史工况的平均耗时下降9.8\%。'
                r'空间扰动工况的节时区间包含零，尚不能确立稳定加速。图~\ref{ms-fig:final_paired_timing}~展示逐工况配对分布。')
    if 'quote_data_availability' in heading:
        return (r'修订稿第~\pageref{ms-reply:quote_data_availability}~页的数据可用性说明为：'
                '支持本研究发现的数据与实现材料可在合理请求下向通讯作者获取。')
    return None


def main():
    old=[(BASE/f'part_{i}_bilingual.tex').read_text(encoding='utf-8') for i in range(1,6)]
    cache={}
    for part in old:
        from revision_acceptance_scope import TITLE, REVIEWERS
        part=part.replace('Sensitivity-Guided Low-Dimensional Redispatch for Base-Case AC Feasibility and Linear N-1 Contingency Screening',TITLE)
        part=re.sub(r'Response to Reviewer ([1-4])',lambda m:'Response to Reviewer '+REVIEWERS[m[1]],part)
        previous=None
        for kind,value in tokens(part):
            if kind=='unit':previous=value
            elif kind=='zh':
                assert previous is not None
                zh,_=brace_argument(value,len(r'\zh'))
                zh=re.sub(r'相应修改见[^。]*。\s*$','',zh).replace('；','。')
                cache.setdefault(norm(previous),zh)
                previous=None
    prepare()
    import json
    responses=json.loads((ROOT/'revision_responses_resolved.json').read_text(encoding='utf-8'))
    custom={}
    for key,item in responses.items():
        if key not in RESPONSES:continue
        english=item['response'].split('\n\n')
        chinese=RESPONSES[key].split('\n\n')
        assert len(english)==len(chinese),(key,'paragraph alignment')
        custom.update({norm(esc(a)):b for a,b in zip(english,chinese)})
    for i in range(1,6):
        current=(WORK/f'part_{i}_english.tex').read_text(encoding='utf-8')
        result=[]
        for kind,value in tokens(current):
            result.append(value)
            if kind!='unit':continue
            base,location=clean_location(value)
            if value.startswith(r'\revisedexcerpt{'):
                zh=quote_translation(value) or cache.get(norm(value))
            elif value.startswith(r'\begin{quote}') and 'Matched 27-case benchmark' in value:
                zh=('四条亮点分别为：匹配的27例基准中，本文流程耗时0.788 s，两种基线为3.075--3.604 s。'
                    '速度优势以接受结果为条件，失败样本会耗尽预算。'
                    '三个训练种子下，灵敏度引导方向优于同维度随机基。'
                    '交流故障审计表明，线性N-1筛查低估电压越限。')
            elif value.startswith('Reviewer numbers follow'):
                zh='审稿人编号遵循期刊平台，分别为1、3、6和8。'
            elif r'\textbf{Title:}' in value:
                zh='题目：面向基态交流可行性与线性N-1故障筛查的灵敏度引导低维再调度。'
            elif value.startswith(r'\section*{Summary of Revisions}'):
                zh='主要修改概述'
            elif re.match(r'\\section\*\{Response to Reviewer (1|3|6|8)\}',value):
                zh='对审稿人'+re.search(r'Reviewer (\d+)',value)[1]+'的回复'
            elif value.startswith(r'\rev{Four changes structure'):
                zh=('本轮主要包括四项修改。（一）完整未筛选集上的接受比较与条件性筛选子集并列报告。'
                    '（二）四种主要策略配置均采用三个训练种子评估，并保留零通过运行。'
                    '（三）显式故障后交流检查与重新锚定测试量化线性接受与故障后可行性的区别。'
                    r'（四）计时分为原27例接受集匹配比较与最终匹配优化器比较，前者具有3.9--4.6倍优势，后者在历史工况上平均节时9.8\%，但在所测试空间扰动工况上未观察到明确的配对节时。'
                    '标题及接受结论区分基态可行性与线性故障筛查。')
            elif value.startswith(r'\rev{We thank the Associate Editor and the reviewers again'):
                zh=('再次感谢副主编与审稿人的意见，这些意见实质性地改变了稿件的结论和证据。'
                    '所有已报告失败样本均予保留，复现材料列于数据可用性声明中。我们愿意提供有助于评价的其他记录。')
            elif re.match(r'We thank Reviewer (1|3|6|8) for these comments',value):
                number=re.search(r'Reviewer (\d+)',value)[1]
                zh='感谢审稿人'+number+'的意见。以下逐条回应。'
            elif value.startswith('The corresponding revision'):
                pages=re.search(r'on manuscript pp?\.~(.+?) and (?:is|are) quoted',value)[1]
                comment=re.search(r'Comment ([A-Z0-9.]+)',value)[1].rstrip('.')
                zh='相应修改见修订稿第~'+pages+'~页，具体文字已在对意见'+comment+'的回复中引用。'
            elif re.match(r'(?:Table|Figure)~\\ref\{ms-',value) and 'is reproduced in our response' in value:
                label=re.search(r'\\ref\{ms-([^}]+)\}',value)[1]
                comment=re.search(r'Comment ([A-Z0-9.]+)',value)[1].rstrip('.')
                name='表' if value.startswith('Table') else '图'
                zh=name+r'~\ref{ms-'+label+r'}位于修订稿第~\pageref{ms-'+label+'}~页，已在对意见'+comment+'的回复中引用。'
            else:zh=custom.get(norm(base)) or cache.get(norm(base))
            if zh is None and value.startswith(r'\begin{bluetableblock}') and 'runtime_cost_feasibility.pdf' in value:
                zh=('时间、计划成本与接受结果。上图为原27例共同测试样本，下图为最终流程共同接受的60例。'
                    '两组结果分开展示，不作跨面板速度比较，成本不含参考电源交换。')
            if zh is None and value.startswith(r'\begin{bluetableblock}'):
                label=re.search(r'\{ms-(tab:[^}]+)\}',value)[1]
                if label in ('tab:baseline','tab:restoration_baseline_v26'):
                    zh=(r'表~\ref{ms-'+label+r'}（修订稿第~\pageref{ms-'+label+r'}~页）：'
                        +('27个经筛选校正样本上的条件性比较，另设一行报告全部75个上游测试候选样本的结果。' if label=='tab:baseline'
                          else '27个按固定动作基筛选的300节点留出校正样本上的端到端恢复比较。最后一行单独报告固定策略在全部75个上游候选样本上的结果，不构成与前述各行的配对比较。'))
            assert zh is not None,('Missing reviewed translation',i,value[:450])
            if value.startswith(r'\begin{bluetableblock}') and 'ms-tab:baseline}' in value:
                zh=(r'表~\ref{ms-tab:baseline}（修订稿第~\pageref{ms-tab:baseline}~页）：'
                    '27个经筛选校正样本上的条件性比较，另设一行报告全部75个上游测试候选样本的结果。')
            if value.startswith(r'\begin{bluetableblock}') and 'ms-tab:restoration_baseline_v26' in value:
                zh=(r'表~\ref{ms-tab:restoration_baseline_v26}（修订稿第~\pageref{ms-tab:restoration_baseline_v26}~页）：'
                    '27个按固定动作基筛选的300节点留出校正样本上的端到端恢复比较。最后一行单独报告固定策略在全部75个上游候选样本上的结果，不构成与前述各行的配对比较。')
            if 'We now state that recovery can move a target' in base:
                zh=('感谢审稿人提出这一意见。我们现已说明，机组边界激活时，恢复过程可能使目标偏离恢复前的基张成空间。'
                    r'对于列向量正交归一的矩阵$B$，采用$\lVert(I-BB^\top)\Delta P_{\mathrm{post}}\rVert_2$及提出动作与执行修正之间的变化进行诊断，并逐步记录两者。'
                    '低维性针对策略参数化，而不是对执行后调度的无条件子空间保证。参考审计记录1773步，最大空间外分量为1.820e-12 MW，'
                    '无一步超过1e-6 MW。提出到恢复后的最大修正变化为1.176e-05 MW。'
                    '这些轨迹在数值精度范围内保持于该张成空间内，同时仍明确保留边界约束可能导致偏离的一般性说明。')
            zh=re.sub(r'回复\s*(?:\\#|#)?\d+[:：]\s*','',zh)
            zh=re.sub(r'(?<!\\)%',r'\\%',zh)
            assert not re.search(r'[\x00-\x08\x0b-\x1f]',zh),('Control character in translation',i,value[:80])
            if location:
                zh+=chinese_location(location).replace(';','。').replace('；','。')
            result.append('\n\n'+r'\zh{'+zh+'}\n\n')
        (WORK/f'part_{i}_bilingual.tex').write_text(''.join(result),encoding='utf-8')
    build()


if __name__=='__main__':main()
