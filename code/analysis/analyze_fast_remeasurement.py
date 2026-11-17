"""Audit the completed implementation remeasurement without selecting settings."""
from statistics import mean,median
import numpy as np
from revision_common import RESULTS,read,write,sha
from remeasure_independent_fast import SOURCE,DEST,ARMS,CONTROL_FIELDS


def maximum_errors(records,key):
    return {field:max((r.get(key,{}).get(field) or 0 for r in records),default=0.) for field in CONTROL_FIELDS}


def main():
    data=read(DEST/'results.json'); original=read(SOURCE/'results.json')
    assert data['complete'] and len(data['rows'])==300
    old={(r['sample_id'],r['arm']):r for r in original['rows']}
    assert len({(r['sample_id'],r['arm']) for r in data['rows']})==300
    assert {(r['sample_id'],r['arm']) for r in data['rows']}==set(old)
    for path,expected in data['protocol']['hashes'].items(): assert sha(path)==expected,path
    stats={}
    for arm in ARMS:
        rows=[r for r in data['rows'] if r['arm']==arm]
        times=[r['runtime_sec'] for r in rows]
        previous=[old[r['sample_id'],arm] for r in rows]
        stats[arm]={'n':len(rows),'passed':sum(r['passed'] for r in rows),
            'preparation_failed':sum(r['preparation_failed'] for r in rows),
            'optimizer_calls':sum(r['optimizer_called'] for r in rows),
            'mean_sec':mean(times),'median_sec':median(times),'p90_sec':float(np.quantile(times,.9)),
            'feature_construction_mean_sec':sum(c['runtime_sec'] for r in rows for c in r.get('voltage_backend_calls',[]))/100,
            'same_trajectory_zero_feature_cost_mean_sec':sum(r['runtime_sec']-sum(c['runtime_sec'] for c in r.get('voltage_backend_calls',[])) for r in rows)/100,
            'original_mean_sec':mean(r['runtime_sec'] for r in previous),
            'acceptance_changes':[r['sample_id'] for r,p in zip(rows,previous) if r['passed']!=p['passed']],
            'preparation_changes':[r['sample_id'] for r,p in zip(rows,previous) if r['preparation_failed']!=p['preparation_failed']],
            'components_mean_sec':{k:sum(r.get(k,0.) for r in rows)/100 for k in
                ('preparation_sec','prediction_projection_sec','initial_verification_sec','policy_and_gate_sec','optimizer_stage_sec','final_verification_sec')},
            'controls_vs_original_max_abs':maximum_errors(rows,'original_control_max_abs_differences')}
    audits=data['audits']
    assert len(audits)==100-stats['sac_only']['preparation_failed']
    assert len({r['sample_id'] for r in audits})==len(audits)
    primary_lookup={(r['sample_id'],r['arm']):r for r in data['rows']}
    trace_hashes_checked=0
    for sample_id,arm in primary_lookup:
        row=primary_lookup[sample_id,arm]
        if not row['preparation_failed']:
            assert sha(DEST/'traces'/sample_id/(arm+'.npz'))==row['action_trace_sha256']
            trace_hashes_checked+=1
    for audit in audits:
        assert sha(DEST/'traces'/audit['sample_id']/'legacy_audit.npz')==audit['legacy_trace_sha256']
    summary={'complete':True,'source_result_sha256':sha(DEST/'results.json'),'stats':stats,
        'same_original_acceptance':all(not s['acceptance_changes'] for s in stats.values()),
        'same_preparation_outcomes':all(not s['preparation_changes'] for s in stats.values()),
        'legacy_replays':len(audits),'all_replay_acceptance_same':all(r['acceptance_same'] for r in audits),
        'all_replay_steps_same':all(r['steps_same'] for r in audits),
        'all_action_shapes_same':all(r['action_shape_same'] for r in audits),
        'max_action_abs_error':max((r['action_max_abs_error'] or 0 for r in audits),default=0),
        'max_jacobian_abs_error':max((r['jacobian_max_abs_error'] for r in audits),default=0),
        'controls_vs_legacy_max_abs':maximum_errors(audits,'control_max_abs_errors'),
        'legacy_controls_vs_original_max_abs':maximum_errors(audits,'legacy_matches_original_controls'),
        'trace_hashes_checked':trace_hashes_checked+len(audits),
        'backend_sources':{source:sum(c['source']==source for r in data['rows'] for c in r.get('voltage_backend_calls',[]))
            for source in ('computed','legacy_fallback','unavailable_zero','memory','disk')}}
    write(DEST/'summary.json',summary)
    names={'sac_only':'纯SAC','direct_pf':'直接优化（PF初始化）','gated_pf':'门控SAC＋优化（PF初始化）'}
    report=['# 同一100例的加速实现复测','',
        '本轮冻结已在验证集检查过的计算实现，复测原先生成的100例；属于同一数据集的软件实现复核。模型、门控范围、动作预算、奖励、求解器预算和验收条件保持原设置。',
        '', '| 流程 | 通过 | 原均值(s) | 本次均值(s) | 本次中位数(s) | 本次P90(s) | 优化器调用 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s=stats[arm]
        report.append(f'| {names[arm]} | {s["passed"]}/100 | {s["original_mean_sec"]:.3f} | {s["mean_sec"]:.3f} | {s["median_sec"]:.3f} | {s["p90_sec"]:.3f} | {s["optimizer_calls"]} |')
    report+=['','计时包含逐例负荷准备、锚定潮流、首次电压灵敏度构造、策略修正、后备优化及最终校验。三种流程按样本轮换顺序串行运行，首次剩余编译开销计入。每个流程保留全部100例，含准备失败。',
        '',f'通过状态相对原记录是否全部保持：{summary["same_original_acceptance"]}。准备失败是否保持：{summary["same_preparation_outcomes"]}。',
        '',f'另对{len(audits)}个准备成功样本使用原特征缓存进行审计回放；其耗时不进入上述工作流程计时。逐步动作形状一致：{summary["all_action_shapes_same"]}；通过状态一致：{summary["all_replay_acceptance_same"]}；步数一致：{summary["all_replay_steps_same"]}。',
        '',f'灵敏度矩阵最大绝对差异：{summary["max_jacobian_abs_error"]:.8g}；归一化动作最大绝对差异：{summary["max_action_abs_error"]:.8g}；最终有功最大差异：{summary["controls_vs_legacy_max_abs"]["final_dispatch_mw"]:.8g} MW；发电机电压最大差异：{summary["controls_vs_legacy_max_abs"]["final_generator_vm_pu"]:.8g} pu；参考源电压最大差异：{summary["controls_vs_legacy_max_abs"]["final_reference_vm_pu"]:.8g} pu。',
        '',f'缓存/后端来源统计：{summary["backend_sources"]}。已核对{summary["trace_hashes_checked"]}份动作轨迹文件及原模型、数据和代码哈希。', '', '## 解释','']
    gate=stats['gated_pf']; direct=stats['direct_pf']
    improvement=100*(1-gate['mean_sec']/gate['original_mean_sec'])
    report.append(f'门控流程相对原记录的均值减少{improvement:.1f}%。直接优化本次也重新计时，公平比较应使用本次两者的结果，而非把两天的单独计时相除。')
    report.append(f'门控流程的特征构造均值为{gate["feature_construction_mean_sec"]:.3f}秒/例。若仅从记录中理想化扣除这部分、其余轨迹和求解调用保持不变，余下仍为{gate["same_trajectory_zero_feature_cost_mean_sec"]:.3f}秒/例；这只是计时分解，不是另一次实际运行成绩。')
    if gate['mean_sec']>=direct['mean_sec']:
        report.append('加速后门控流程在这组新负荷分布上仍慢于直接优化。因此这轮结果支持计算实现获得改善，但不支持全新工况下普遍快于优化器的结论。失败策略尝试和后备优化的累计开销仍需计入。')
    else:
        report.append('本次门控流程均值低于直接优化。这是同一100例的一次配对复测，仍需结合逐例分布解释，不应写成任意新工况下的保证。')
    (DEST/'REMEASUREMENT_CN.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(summary)


if __name__=='__main__': main()
