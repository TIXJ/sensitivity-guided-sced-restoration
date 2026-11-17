"""Write the narrow abstract and AE #7 synchronization delivery report."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    changes=[
        ('Manuscript_Revised_Marked.tex','Against a matched joint optimizer','历史工况节时9.8%后结束','照抄追加a gain not reproduced on the tested spatial profiles'),
        ('Manuscript_Revised_Marked.tex','Tests on the IEEE 39-bus','摘要未提小系统','补入39/118-bus句，随后以训练种子下的表示结论收尾'),
        ('Manuscript_Revised_Marked.tex','Against a matched optimizer the final','历史工况节省约十分之一后结束','照抄追加a saving that does not extend to the new spatial profiles'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','The complete revised abstract is quoted below.','旧摘要全文221词','更新为新摘要全文240词，逐字一致并保留完整引用承诺句'),
        ('Response_to_Associate_Editor_and_Reviewers.tex','\\revisedexcerpt{p.~\\pageref{ms-reply:quote_conclusion}}','旧结论引用','同步新结论全文179词及其逐句摘录'),
        ('01_BILINGUAL_REVIEW/Response_Bilingual_EN_ZH.tex','a gain not reproduced','旧英文与中文','摘要、结论及对应中文同步，数字不变'),
        ('revision_acceptance_scope.py','ABSTRACT =','旧摘要/结论来源','只替换本轮指定的三处英文字符串'),
        ('revision_response_layout.py',"assert actual==quotes['abstract']",'生成后未直接检查AE#7实际引用块','增加AE#7实际引用块与全文完全一致的强制校验'),
        ('experiments/sync_annotated_response.py','该增益未在所测试','旧中文译文','同步空间样本限定、39/118节点句及新结论译文'),
    ]
    rows=[]
    for name,needle,before,after in changes:
        text=(ROOT/name).read_text(encoding='utf-8');pos=text.find(needle)
        assert pos>=0,(name,needle)
        rows.append(f'| {name}:{text[:pos].count(chr(10))+1} | `{needle}` | {before} → {after} |')
    result=json.loads((ROOT/'_audit/abstract_scope_sync_results.json').read_text(encoding='utf-8'))
    lines=['# 摘要与AE#7最后同步记录','',
        '## 改动清单','',
        '正文改动同步至Clean和Marked。已验证：去除修订颜色包装后，除本轮摘要和结论外，正文逐字未变。','',
        '| 文件/行号 | 定位字符串 | 改前 → 改后 |','|---|---|---|',*rows,'',
        '检查开始时，当前AE#7已经完整引用原221词摘要，而非仅三句。本轮将其同步至新240词全文，并新增生成期强制校验及回归测试。','',
        '## 词数与页数','',
        f'- 摘要：{result["abstract_words"]}词，单段。',
        f'- 结论：{result["conclusion_words"]}词，单段。',
        '- 标题：12词。',
        *['- '+name+': '+str(count)+'页。' for name,count in result['pages'].items()],
        '- 未插入CHECK标记。',
        '- 数据可用性、参考文献及正文其他部分未改。bibitems仍为25。O1/O2未执行。','',
        '## 九项校验实际输出','',
        '页码核对结合AUX与PDF起始页中的摘录开头/标题/题注。长引文可跨页，不把起始页命中表述成整段均在一页。AE#7另做全文逐字比对。','',
        '```text',(ROOT/'_audit/abstract_scope_sync_validation.txt').read_text(encoding='utf-8').rstrip(),'```']
    (ROOT/'_audit/Abstract_Scope_Sync_Changes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote _audit/Abstract_Scope_Sync_Changes.md')


if __name__=='__main__':main()
