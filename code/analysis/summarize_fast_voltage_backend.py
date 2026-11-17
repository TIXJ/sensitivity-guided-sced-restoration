"""Summarize validated speed changes and assemble a small implementation bundle."""
import ast
import hashlib
import importlib.metadata
from pathlib import Path
from statistics import mean,median
import zipfile
from revision_common import RESULTS,REV,CODE,read,write,sha


def kernel_hash(path):
    text=Path(path).read_text(encoding='utf-8')
    tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='finite_voltage_jacobian')
    return hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()


def main():
    folder=RESULTS/'fast_voltage_validation_20261018'
    runs={name:read(folder/name/'results.json') for name in ('numba_v1','numba_all25_v1','recycle_all25_v2')}
    assert all(p['complete'] for p in runs.values())
    summaries={}
    for name,p in runs.items():
        by_arm={a:[r for r in p['rows'] if r['arm']==a] for a in ('legacy','fast')}
        stats={a:{'count':len(rows),'passed':sum(r['passed'] for r in rows),
            'mean_sec':mean(r['runtime_sec'] for r in rows),'median_sec':median(r['runtime_sec'] for r in rows)} for a,rows in by_arm.items()}
        stats['reduction_pct']=100*(1-stats['fast']['mean_sec']/stats['legacy']['mean_sec'])
        stats['max_errors']={k:max(r[k] for r in p['comparisons']) for k in
            ('jacobian_max_abs_error','observation_max_abs_error','action_max_abs_error')}
        assert stats['max_errors']['action_max_abs_error']<1e-6
        summaries[name]=stats
    archived=REV/'original_git_snapshot/fast_voltage_recycled_20261018/reset_voltage_jacobian_fast.py'
    same=kernel_hash(archived)==kernel_hash(CODE/'reset_voltage_jacobian_fast.py')
    assert same,'Revalidate changed numerical kernel'
    summary={'complete':True,'evaluations':summaries,'numerical_kernel_matches_full_validation':same,
        'runtime_versions':{p:importlib.metadata.version(p) for p in ('pandapower','numpy','pandas','numba','torch')},
        'deployed_backend_sha256':sha(CODE/'reset_voltage_jacobian_fast.py'),
        'scope':'Validation-only paired implementation tests; historical test results and manuscript timings are not replaced.'}
    write(folder/'summary.json',summary)
    s=summaries['recycle_all25_v2']; n=summaries['numba_all25_v1']
    report=f'''# 首次电压灵敏度计算优化

本轮保留有限差分步长、69个扰动方向、无功限值和最终验收条件，优化Pandapower的计算路径与缓存管理。

| 完整25例验证集，首次完整策略流程 | 原实现均值(s) | 加速实现均值(s) | 耗时减少 |
|---|---:|---:|---:|
| 仅启用Numba | {n['legacy']['mean_sec']:.3f} | {n['fast']['mean_sec']:.3f} | {n['reduction_pct']:.1f}% |
| Numba及公开recycle接口 | {s['legacy']['mean_sec']:.3f} | {s['fast']['mean_sec']:.3f} | {s['reduction_pct']:.1f}% |

最终版本两边均为{s['legacy']['passed']}/25通过。灵敏度矩阵最大绝对误差{s['max_errors']['jacobian_max_abs_error']:.3e}，状态最大绝对误差{s['max_errors']['observation_max_abs_error']:.3e}，策略动作最大绝对差异{s['max_errors']['action_max_abs_error']:.3e}。步数、最终有功和电压控制、通过状态逐例核对一致。

每个样本从空缓存开始，串行交替执行两种实现，计时包含reset、灵敏度、策略修正和最终验收；驻留模型初始化在计时前完成。首例剩余Numba编译开销保留在均值中。三个固定验证样本另有三次配对重复。以上是验证集的软件实现检查，不替换原独立100例的计时结果。

缓存键包含配置后的网络输入表、负荷空间分布、无功、初始发电调度、电压设定、限值、步长及Pandapower版本。一次reset内维持原来的固定初始灵敏度；重新reset时检查完整工况。磁盘文件校验上下文和矩阵摘要，损坏文件重新计算。加速求解失败时回到原数值路径。

独立入口：`experiments/evaluate_revision_fast.py`，默认验证集，要求指定新的输出路径。原训练脚本、检查点、历史缓存、论文及交付包没有覆盖修改。

在项目根目录运行一次验证集评估：

```powershell
& 'external/Anaconda\\envs\\lct-GPU\\python.exe' '.\\GAT-SAC-SCED-Framework\\first revise\\experiments\\evaluate_revision_fast.py' --scope val --output '.\\GAT-SAC-SCED-Framework\\first revise\\experiment_results\\fast_sac_validation_new.json'
```

结果中的`stress_end_to_end_sec`包含逐样本准备与最终校验。`voltage_backend`记录实现哈希和缓存来源。此优化仍需计算69个有限差分扰动，不能据此宣称对全新空间负荷工况已经快于直接优化。
'''
    (folder/'IMPLEMENTATION_CN.md').write_text(report,encoding='utf-8')
    paths=[CODE/'reset_voltage_jacobian_fast.py',REV/'experiments/evaluate_revision_fast.py',
        REV/'experiments/validate_fast_voltage_backend.py',REV/'experiments/test_fast_voltage_cache.py',
        REV/'experiments/summarize_fast_voltage_backend.py',folder/'summary.json',folder/'IMPLEMENTATION_CN.md']
    target=REV/'deliveries/Fast_Voltage_Backend_20261018.zip'
    if target.exists(): raise RuntimeError('Preserve previous delivery')
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as package:
        for path in paths:
            name=path.relative_to(REV.parents[1]).as_posix()
            package.write(path,name)
    print(summary)
    print('DELIVERY',target)


if __name__=='__main__': main()
