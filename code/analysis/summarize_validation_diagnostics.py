"""Check diagnostic contracts and summarize measured results without refitting."""
from statistics import mean
import numpy as np
from revision_common import read, write, sha
from diagnose_runtime_and_tight_validation import DEST, select


def main():
    profile=read(DEST/'validation_profile.json')
    reach=read(DEST/'validation_reachability.json')
    joint=read(DEST/'joint_tight_probe.json')
    assert all(x['complete'] for x in (profile,reach,joint))
    _,samples,ids=select()
    wanted={samples[i]['sample_id'] for i in ids}
    assert len(profile['rows'])==6
    for group in (reach,joint):
        assert len(group['rows'])==3 and {r['sample_id'] for r in group['rows']}==wanted
    cold=[r for r in profile['rows'] if r['mode']=='cold_context']
    warm=[r for r in profile['rows'] if r['mode']=='same_context_memory_hit']
    for a,b in zip(cold,warm):
        assert a['sample_id']==b['sample_id'] and a['passed']==b['passed']
        for key in ('final_dispatch_mw','final_generator_vm_pu','final_reference_vm_pu'):
            assert np.allclose(a[key],b[key],atol=1e-9,rtol=0), (a['sample_id'],key)
        assert a['calls']['runpp_in_jacobian']==70
        assert b['calls'].get('runpp_in_jacobian',0)==0
    for group in (reach,joint):
        assert all(r['norm_mw']<40 and r['offspan_mw']<1e-8 for r in group['rows'])
    for path,expected in read(DEST/'protocol.json')['hashes'].items():
        assert sha(path)==expected,path
    summary={'complete':True,'old_experiments_and_checkpoints_unchanged':True,
        'validation_ids':sorted(wanted),
        'cold_total_mean_sec':mean(r['runtime_sec'] for r in cold),
        'warm_total_mean_sec':mean(r['runtime_sec'] for r in warm),
        'cold_jacobian_mean_sec':mean(r['measured_inclusive_sec']['jacobian'] for r in cold),
        'actor_forward_mean_sec':sum(r['measured_inclusive_sec']['actor'] for r in profile['rows'])/sum(r['calls']['actor'] for r in profile['rows']),
        'original_band_witnesses':sum(r['voltage_probes']['original']['passed'] for r in reach['rows']),
        'tight_fixed_P_witnesses':sum(r['voltage_probes']['tight']['passed'] for r in reach['rows']),
        'tight_joint_witnesses':sum(r['passed'] for r in joint['rows']),
        'infeasibility_certified':False,'warm_cold_controls_identical':True}
    write(DEST/'summary.json',summary)
    lines=['# 2026-10-18 验证集诊断', '',
        '本轮未训练模型、未调整奖励或学习率，未修改论文和已有测试结果。', '',
        '## 1. 耗时原因', '',
        '原独立100例日志：直接优化均值1.096秒，门控流程3.533秒。门控的策略初始化/修正/验收段均值2.598秒；优化器调用由91次减至73次，只节省约0.161秒/例的优化阶段耗时。', '',
        f'三个预先选定的验证样本：首次完整策略流程均值{summary["cold_total_mean_sec"]:.3f}秒，其中电压灵敏度构造均值{summary["cold_jacobian_mean_sec"]:.3f}秒，每次调用70次潮流。Actor单次前向均值{1000*summary["actor_forward_mean_sec"]:.3f}毫秒。', '',
        f'完全相同样本/上下文命中内存缓存后，完整策略流程均值{summary["warm_total_mean_sec"]:.3f}秒；冷、热运行的最终控制及通过状态一致。该结果是带计时插桩的验证诊断，不替换独立100例首次运行结果。嵌套计时不可相加。', '',
        '不同负荷分布、网络状态或参考调度不能仅凭总负荷相同就复用缓存。后续应优先验证灵敏度计算加速与严格缓存键，而不是调学习率或把准备时间排除出端到端计时。', '',
        '## 2. 紧电压条件', '',
        '样本来自原筛选验证集：每个16.0/16.2/16.4 GW水平取编号最小的样本，事先固定，无测试样本。', '',
        '| 验证样本 | 最小线性安全调整(MW) | 原电压通过 | 紧电压固定P通过 | 联合搜索调整(MW) | 联合搜索最大电压越界(pu) |',
        '|---|---:|---|---|---:|---:|']
    for r,j in zip(reach['rows'],joint['rows']):
        lines.append(f'| {r["sample_id"]} | {r["norm_mw"]:.3f} | {r["voltage_probes"]["original"]["passed"]} | {r["voltage_probes"]["tight"]["passed"]} | {j["norm_mw"]:.3f} | {j["final_verification"]["v_violation_max_pu"]:.8f} |')
    lines += ['', '所有搜索保留原32维有功基、40 MW范围、电压设备界限及累计界限。三个样本原电压范围均找到通过点；0.94–1.08 pu范围下，固定P修电压和一次有预算限制的联合搜索均未找到严格通过点。', '',
        '联合搜索均因函数评估预算结束，并未用满40 MW。有功N-1检查通过但仍有电压残差。因此结果既不能证明动作空间无解，也不能证明失败只来自SAC训练。下一步若继续研究紧电压，应先加强验证集上的约束可行性求解，不宜盲目重训或扩大动作范围。', '',
        '## 3. 复核', '',
        '已核对原实验/模型哈希、样本划分、冷/热控制一致性和动作范数/子空间误差。原先提交包未覆盖改写。本目录JSON保留逐样本计时、控制量及校验输出。']
    (DEST/'DIAGNOSIS_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary)


if __name__=='__main__': main()
