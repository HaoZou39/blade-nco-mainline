from __future__ import annotations
import json,math,time
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from audit_common import *

def train_arm(arm,cal,p0):
    x=cal[FEATURES].to_numpy(float); h=cal.H0.to_numpy(float); px=p0[FEATURES].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(h).all() or not np.isfinite(px).all(): raise RuntimeError(f"nonfinite GP data {arm}")
    transform=e1.robust_transform(h); y=np.log1p(h); scaled=(y-transform["center"])/transform["scale"]; membership=member_sha(cal.instance_hash.tolist()); formulas=[]; preds=[]; rows=[]
    for seed in SEEDS:
        sd=OUT/"models"/arm/f"seed_{seed}"; fp=sd/"formula.json"; mp=sd/"AUDIT_SEED_COMPLETE.json"; expected={"arm":arm,"seed":seed,"membership_sha256":membership,"gp_config_sha256":cfg_sha(),"m0_sha256":sha(M0)}; source="TRAINED_THIS_AUDIT"
        if fp.is_file() and mp.is_file():
            meta=json.loads(mp.read_text(encoding="utf-8"))
            if not all(meta.get(k)==v for k,v in expected.items()): raise RuntimeError(f"stale model cache {sd}")
            formula=json.loads(fp.read_text(encoding="utf-8")); runtime=float(meta["training_wall_seconds"]); source="REUSED_AUDIT_CACHE"
        else:
            sd.mkdir(parents=True,exist_ok=True); print(f"[{arm}][seed={seed}] GP4 train",flush=True); t=time.perf_counter(); log,hall=e1.train_seed(x,scaled,FEATURES,GP_CFG,seed); formula,candidates=e1.select_sperf(hall,x,y,transform); runtime=time.perf_counter()-t
            log.to_parquet(sd/"training_log.parquet",index=False); write_json(sd/"hof_candidates.json",candidates); write_json(fp,formula); write_json(mp,{**expected,"training_wall_seconds":runtime,"formula_sha256":sha(fp),"completed_utc":now()}); print(f"[{arm}][seed={seed}] complete {runtime:.2f}s",flush=True)
        pred=e1.formula_prediction(formula,px,transform)
        if not np.isfinite(pred).all() or np.ptp(pred)<=1e-15: raise RuntimeError(f"catastrophic formula failure {arm} seed {seed}")
        formulas.append(formula); preds.append(pred); rows.append({"arm":arm,"gp_seed":seed,"train_n":len(cal),"formula":formula.get("formula",""),"canonical_formula":formula["canonical_formula"],"formula_nodes":int(formula["nodes"]),"formula_depth":int(formula.get("depth",0)),"used_features":"|".join(formula.get("used_features",[])),"formula_complexity":int(formula["nodes"]),"training_runtime_seconds":runtime,"training_runtime_source":source,"archive_fit_spearman":float(formula["archive_fit_spearman"]),"archive_fit_mae":float(formula["archive_fit_mae"]),"p0_seed_spearman":float(spearmanr(p0.H0,pred).statistic),"p0_prediction_finite":True,"p0_prediction_variance":float(np.var(pred))})
    ensemble=np.median(np.stack(preds),axis=0)
    if not np.isfinite(ensemble).all() or np.ptp(ensemble)<=1e-15: raise RuntimeError(f"catastrophic ensemble failure {arm}")
    return formulas,ensemble,transform,rows

def metrics(actual,pred,hashes):
    actual=np.asarray(actual,float); pred=np.asarray(pred,float); hs=np.asarray(hashes,str)
    if len(actual)!=len(pred) or not np.isfinite(pred).all() or np.ptp(pred)<=1e-15: raise RuntimeError("metric alignment/prediction failure")
    order=np.lexsort((hs,-pred)); base=float(np.mean(actual)); r={"P0_Spearman":float(spearmanr(actual,pred).statistic),"prediction_mean":float(np.mean(pred)),"prediction_std":float(np.std(pred)),"prediction_min":float(np.min(pred)),"prediction_q25":float(np.quantile(pred,.25)),"prediction_median":float(np.median(pred)),"prediction_q75":float(np.quantile(pred,.75)),"prediction_max":float(np.max(pred)),"full_panel_mean_H0":base}
    for pct in (5,10,20):
        n=max(1,int(math.ceil(len(actual)*pct/100))); mean=float(np.mean(actual[order[:n]])); r[f"Top{pct}_n"]=n; r[f"Top{pct}_mean_H0"]=mean; r[f"ENR{pct}"]=mean/base
    return r

def structure(arm,f):
    counts=f.niche_id.value_counts().to_numpy(float); p=counts/counts.sum(); entropy=float(-np.sum(p*np.log(p)))
    return {"arm":arm,"calibration_n":len(f),"represented_niches":f.niche_id.nunique(),"effective_niche_number":float(np.exp(entropy)),"niche_entropy_nats":entropy,"normalized_niche_entropy":float(entropy/np.log(len(counts))) if len(counts)>1 else 0.,"operator_coverage":f.operator_id.nunique(),"operators":"|".join(sorted(f.operator_id.astype(str).unique())),"H0_mean":float(f.H0.mean()),"H0_std":float(f.H0.std(ddof=1)),"H0_min":float(f.H0.min()),"H0_q05":float(f.H0.quantile(.05)),"H0_q25":float(f.H0.quantile(.25)),"H0_median":float(f.H0.median()),"H0_q75":float(f.H0.quantile(.75)),"H0_q95":float(f.H0.quantile(.95)),"H0_max":float(f.H0.max())}

def full_o0(qd,old_f,old_t,rnd_f,rnd_t):
    x=qd[FEATURES].to_numpy(float); op=np.median(np.stack([e1.formula_prediction(f,x,old_t) for f in old_f]),axis=0); rp=np.median(np.stack([e1.formula_prediction(f,x,rnd_t) for f in rnd_f]),axis=0); hs=qd.instance_hash.astype(str).to_numpy(); oo=np.lexsort((hs,-op)); ro=np.lexsort((hs,-rp)); ot=set(hs[oo[:10000]]); rt=set(hs[ro[:10000]])
    labels=pd.read_parquet(HIST_HARD,columns=["instance_hash","H0"]); labels["instance_hash"]=labels.instance_hash.astype(str); lm=labels[labels.instance_hash.isin(set(hs))].drop_duplicates("instance_hash").set_index("instance_hash").H0.to_dict()
    def realized(s):
        v=[float(lm[h]) for h in s if h in lm]; return len(v),float(np.mean(v)) if v else None
    on,ov=realized(ot); rn,rv=realized(rt); result={"top10k_overlap_n":len(ot&rt),"top10k_jaccard":len(ot&rt)/len(ot|rt),"OLD_CENTROID435_top10k_niche_coverage":qd[qd.instance_hash.isin(ot)].niche_id.nunique(),"RANDOM500_top10k_niche_coverage":qd[qd.instance_hash.isin(rt)].niche_id.nunique(),"OLD_CENTROID435_prediction_mean":float(np.mean(op)),"OLD_CENTROID435_prediction_std":float(np.std(op)),"RANDOM500_prediction_mean":float(np.mean(rp)),"RANDOM500_prediction_std":float(np.std(rp)),"OLD_CENTROID435_exact_labeled_top10k_n":on,"OLD_CENTROID435_realized_mean_H0_cached_subset":ov,"RANDOM500_exact_labeled_top10k_n":rn,"RANDOM500_realized_mean_H0_cached_subset":rv,"new_concorde_calls_for_optional_replay":0}
    pd.DataFrame([result]).to_csv(OUT/"optional_full_o0_replay.csv",index=False); return result

def figures(p0,summary,old,rnd):
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,arm,color in zip(axes,["OLD_CENTROID435","RANDOM500"],["#4c78a8","#f58518"]):
        pred=p0[f"prediction_{arm}"].to_numpy(float); order=np.lexsort((p0.instance_hash.astype(str).to_numpy(),-pred)); x=np.arange(1,len(p0)+1)/len(p0); y=p0.H0.to_numpy(float)[order]; ax.scatter(x,y,s=5,alpha=.22,color=color); ax.plot(x,pd.Series(y).rolling(max(15,len(y)//35),center=True,min_periods=1).mean(),color=color,lw=2); ax.axvline(.1,color="black",ls="--",lw=1); ax.set_title(arm); ax.set_xlabel("Predicted rank fraction")
    axes[0].set_ylabel("True H0"); fig.suptitle("Shared P0: true hardness by predicted rank"); fig.tight_layout(); fig.savefig(OUT/"fig01_p0_true_vs_predicted_rank.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); pos=np.arange(3); w=.34
    for off,arm,color in [(-w/2,"OLD_CENTROID435","#4c78a8"),(w/2,"RANDOM500","#f58518")]:
        r=summary[summary.arm.eq(arm)].iloc[0]; ax.bar(pos+off,[r.ENR5,r.ENR10,r.ENR20],w,label=arm,color=color)
    ax.set_xticks(pos,["ENR5","ENR10 (PRIMARY)","ENR20"]); ax.set_ylabel("Enrichment ratio"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"fig02_enrichment_comparison.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); a=old.niche_id.value_counts().sort_values(ascending=False).to_numpy(); b=rnd.niche_id.value_counts().sort_values(ascending=False).to_numpy(); ax.plot(np.arange(1,len(a)+1),a,label="OLD_CENTROID435"); ax.plot(np.arange(1,len(b)+1),b,label="RANDOM500"); ax.set_xlabel("Represented niche rank"); ax.set_ylabel("Members"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"fig03_calibration_niche_distribution.png",dpi=180); plt.close(fig)
