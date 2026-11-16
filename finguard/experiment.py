"""Run locked temporal evaluation and paired graph ablations."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier
from .data import fetch, load_ibm, load_raw_amlsim
from .features import build, TX, BEHAVIOR, GRAPH, ALL, rule_score, graph_score

METRICS=['precision','recall','pr_auc_ap','fpr']

def metrics(y,s,t):
    p=s>=t; tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'precision':float(precision_score(y,p,zero_division=0)), 'recall':float(recall_score(y,p,zero_division=0)),
        'pr_auc_ap':float(average_precision_score(y,s)), 'fpr':float(fp/(tn+fp)) if tn+fp else 0.,
        'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn), 'alerts':int(p.sum())}

def threshold(y,s,budget=.05):
    """Maximize validation recall at <=5% empirical validation FPR; conservative ties."""
    candidates=np.r_[np.unique(s),np.nextafter(max(s),np.inf)]
    neg=np.sort(s[y==0]); pos=np.sort(s[y==1])
    fpr=(len(neg)-np.searchsorted(neg,candidates,'left'))/len(neg)
    recall=(len(pos)-np.searchsorted(pos,candidates,'left'))/len(pos)
    ok=np.flatnonzero(fpr<=budget)
    best=ok[recall[ok]==max(recall[ok])]
    return float(candidates[best[-1]])

def block_indices(steps,seed=73,n=300):
    # Non-overlapping 5-step blocks preserve some within-time dependence.
    rng=np.random.default_rng(seed)
    groups=[np.flatnonzero((steps-steps.min())//5==k) for k in np.unique((steps-steps.min())//5)]
    return [np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]) for _ in range(n)]

def evaluate(root='.',raw=None):
    root=Path(root); out=root/'reports'; out.mkdir(exist_ok=True)
    if raw:
        d=load_raw_amlsim(raw)
        manifest={'source':str(raw),'sha256':hashlib.sha256(Path(raw).read_bytes()).hexdigest(), 'label_definition':'Raw simulator isSAR per transaction', 'rows':len(d)}
    else:
        d,manifest=load_ibm(fetch(root/'data/raw'))
    d,x=build(d)
    # Temporal cut points by unique steps, no row-wise random split.
    times=np.sort(d.step.unique()); n=len(times)
    a,b,c,e=[int(n*q) for q in [.54,.60,.74,.80]]
    train=d.step<times[a]; val=(d.step>=times[b])&(d.step<times[c]); test=d.step>=times[e]
    for mask in [train,val,test]:
        if d.loc[mask,'label'].nunique()!=2: raise ValueError('Each split must contain both classes')
    masks={'train':train,'validation':val,'test':test}
    manifest['splits']={k:{'n':int(v.sum()),'min_step':int(d.loc[v,'step'].min()),'max_step':int(d.loc[v,'step'].max()),'positive_rate':float(d.loc[v,'label'].mean())} for k,v in masks.items()}
    manifest['excluded_gap_rows']=int((~(train|val|test)).sum())
    manifest['environment']={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','networkx','fastapi','streamlit']}
    y=d.label.to_numpy(); tr=train.to_numpy(); va=val.to_numpy(); te=test.to_numpy()
    scores={}; models={}
    log=x.log_amount.to_numpy(); med=np.median(log[tr]); mad=np.median(np.abs(log[tr]-med))
    scores['Robust amount']=np.abs(log-med)/max(mad,1e-8)
    scores['Rules']=rule_score(x); scores['Graph motif score']=graph_score(x)
    iso=IsolationForest(n_estimators=150,random_state=42,n_jobs=2).fit(x.loc[tr,TX+BEHAVIOR])
    scores['Isolation Forest']=-iso.score_samples(x[TX+BEHAVIOR])
    lr=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42))
    lr.fit(x.loc[tr,TX+BEHAVIOR],y[tr]); scores['Logistic regression']=lr.predict_proba(x[TX+BEHAVIOR])[:,1]
    for name,cols in [('XGB transaction',TX+BEHAVIOR),('XGB graph only',GRAPH),('XGB full',ALL),('XGB no cycles',[c for c in ALL if c not in ['reverse_edge','closes_triangle']])]:
        model=XGBClassifier(n_estimators=160,max_depth=3,learning_rate=.06,subsample=.85,colsample_bytree=1.,
            reg_lambda=5.,min_child_weight=10,random_state=42,n_jobs=2,tree_method='hist',
            scale_pos_weight=float((y[tr]==0).sum()/(y[tr]==1).sum()))
        model.fit(x.loc[tr,cols],y[tr]); scores[name]=model.predict_proba(x[cols])[:,1]; models[name]=model
    idx=block_indices(d.loc[te,'step'].to_numpy()); results=[]; thresholds={}; cis={}
    for name,s in scores.items():
        t=threshold(y[va],s[va]); thresholds[name]=t
        m=metrics(y[te],s[te],t)
        boot=[metrics(y[te][i],s[te][i],t) for i in idx if len(np.unique(y[te][i]))==2]
        ci={k:[float(v) for v in np.quantile([r[k] for r in boot],[.025,.975])] for k in METRICS}
        cis[name]=ci; results.append({'model':name,'threshold':t,**m})
    results=pd.DataFrame(results); results.to_csv(out/'metrics.csv',index=False)
    deltas={}
    for other in ['XGB transaction','XGB no cycles','XGB graph only']:
        delta=[]
        for i in idx:
            yy=y[te][i]
            if len(np.unique(yy))==2:
                delta.append(average_precision_score(yy,scores['XGB full'][te][i])-average_precision_score(yy,scores[other][te][i]))
        deltas[other]={'ap_delta_full_minus_comparator':float(average_precision_score(y[te],scores['XGB full'][te])-average_precision_score(y[te],scores[other][te])), 'paired_block_ci95':np.quantile(delta,[.025,.975]).tolist()}
    # Fixed hyperparameters, repeat three initialization seeds for the central ablation.
    seed_rows=[]
    for seed in [7,19,42]:
        for name,cols in [('XGB transaction',TX+BEHAVIOR),('XGB full',ALL)]:
            model=models[name] if seed==42 else XGBClassifier(**{**models[name].get_params(),'random_state':seed})
            if seed!=42: model.fit(x.loc[tr,cols],y[tr])
            s=model.predict_proba(x[cols])[:,1]; t=threshold(y[va],s[va])
            seed_rows.append({'seed':seed,'model':name,**metrics(y[te],s[te],t)})
    pd.DataFrame(seed_rows).to_csv(out/'seed_sensitivity.csv',index=False)
    (out/'confidence_intervals.json').write_text(json.dumps(cis,indent=2))
    (out/'ablation.json').write_text(json.dumps(deltas,indent=2))
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Native XGBoost serialization, never unpickle untrusted models.
    model_dir=root/'models'; model_dir.mkdir(exist_ok=True)
    models['XGB full'].save_model(model_dir/'xgb_full.json')
    metadata={'features':ALL,'threshold':thresholds['XGB full'],'window':30,'model':'XGB full',
        'label_definition':manifest['label_definition'],'version':'0.1.0','data_sha256':manifest.get('files_sha256',{})}
    (model_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
    pred=d.loc[te].copy(); pred['score']=scores['XGB full'][te]; pred['flagged']=pred.score>=thresholds['XGB full']
    pred.to_csv(out/'test_predictions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for name,s in scores.items():
        p,r,_=precision_recall_curve(y[te],s[te]); ax.plot(r,p,label=name,alpha=.85)
    ax.axhline(y[te].mean(),color='gray',linestyle='--',label='Test prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='FinGuard | IBM account-associated transfer benchmark')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'precision_recall.png',dpi=160); plt.close(fig)
    print(results.to_string(index=False)); print(json.dumps(deltas,indent=2))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--amlsim-raw')
    args=p.parse_args(); evaluate(args.root,args.amlsim_raw)
