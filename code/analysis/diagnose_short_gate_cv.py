"""Development-only cross-validation of a cheap invocation predictor."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold,cross_val_predict
from revision_common import RESULTS,read,write,sha


def vector(sample,initial):
    values=[sample['demand_mw']/16000,initial['max_gap_pu'],initial['sum_gap_pu']]
    for field in ('p_weights','q_weights','sample_cp1','sample_cp2'):
        v=np.asarray(sample[field]);v=v/np.mean(v)
        values.extend(np.mean(part) for part in np.array_split(v,6))
        values.extend((float(np.min(v)),float(np.max(v)),float(np.std(v))))
    return np.array(values)


def main():
    source=RESULTS/'implicit_short_policy_v1'
    data=read(source/'val.json');assert data['complete']
    conditions={r['sample_id']:r for r in read(source/'val_conditions.json')['samples']}
    arms={a:{r['sample_id']:r for r in data['rows'] if r['arm']==a} for a in ('direct','short4','short8')}
    ids=[sid for sid,r in arms['direct'].items() if not r['preparation_failed'] and conditions[sid]['demand_mw']<=16400]
    x=np.array([vector(conditions[sid],arms['direct'][sid]['initial_features']) for sid in ids])
    rows=[]
    for cap in (4,8):
        y=np.array([(arms['direct'][sid]['optimizer_sec'] if arms['short'+str(cap)][sid]['policy_gate_passed'] else 0.)
                    -arms['short'+str(cap)][sid]['policy_sec'] for sid in ids])
        for depth in (1,2):
            for leaf in (4,6,8):
                model=DecisionTreeRegressor(max_depth=depth,min_samples_leaf=leaf,random_state=42)
                prediction=cross_val_predict(model,x,y,cv=KFold(6,shuffle=True,random_state=42))
                for threshold in (0.,.05,.1):
                    use=prediction>threshold
                    rows.append({'cap':cap,'depth':depth,'min_leaf':leaf,'threshold_sec':threshold,
                        'invoked':int(use.sum()),'out_of_fold_benefit_mean_all60_sec':float(np.sum(y[use])/60)})
    out={'complete':True,'development_source_sha256':sha(source/'val.json'),'script_sha256':sha(__file__),
         'n':len(ids),'candidates':rows,'best':max(rows,key=lambda r:r['out_of_fold_benefit_mean_all60_sec'])}
    write(RESULTS/'short_gate_cv.json',out);print(out['best'])


if __name__=='__main__':main()
