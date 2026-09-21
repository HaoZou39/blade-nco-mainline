"""Train GP0/HGB controls and evaluate the sealed Cycle-0 panel."""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from features.qd4 import QD4_FEATURE_NAMES
from audit.gp_modeling import train_seed, tree_from_dict, safe_scaled_prediction, tree_nodes

AUDIT=ROOT/"audits/cycle0_initialization_validation_v1"; MASTER_SEED=2026
Q4=list(QD4_FEATURE_NAMES); GP20=json.loads((AUDIT/"00_protocol/gp20_features.json").read_text())["features"]
HGB_CFG=dict(learning_rate=.05,max_iter=500,max_leaf_nodes=31,max_depth=None,min_samples_leaf=20,l2_regularization=1e-4,early_stopping=False,random_state=2026)
GP_CFG={"population_size":2000,"generations":60,"initial_depth_min":2,"initial_depth_max":4,"maximum_depth":6,"maximum_nodes":25,"maximum_unique_features":10,"maximum_constants":25,"tournament_size":7,"elitism_count":20,"crossover_probability":.65,"subtree_mutation_probability":.15,"point_mutation_probability":.10,"hoist_mutation_probability":.05,"reproduction_probability":.05,"parsimony_coefficient":5e-4,"hall_of_fame_size":100,"prediction_invalid_threshold":1e6,"fitness_prediction_clip":20.,"maximum_validation_clip_fraction":.01,"huber_delta":1.}

def _json_default(value):
 if isinstance(value,np.generic): return value.item()
 if isinstance(value,np.ndarray): return value.tolist()
 raise TypeError(type(value).__name__)
def js(path:Path,value:Any): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,default=_json_default)+"\n",encoding="utf-8")
def sha(path:Path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def robust(y):
 center=float(np.median(y)); iqr=float(np.subtract(*np.percentile(y,[75,25]))); mad=float(np.median(np.abs(y-center))); sd=float(np.std(y,ddof=1))
 method,scale=next((n,v) for n,v in [("IQR",iqr),("1.4826_MAD",1.4826*mad),("STD",sd),("ONE",1.)] if np.isfinite(v) and v>1e-12)
 return {"fit_split":"TRAIN_ONLY","formula":"loss_log = log1p(loss_raw)","center_train_median_loss_log":center,"scale_train_loss_log":scale,"scale_method":method}
def inv(v,t): return np.asarray(v)*t["scale_train_loss_log"]+t["center_train_median_loss_log"]
def predict_record(record,x,t):
 scaled,clip,valid=safe_scaled_prediction(tree_from_dict(record["tree"]),x,GP_CFG)
 if not valid or clip>GP_CFG["maximum_validation_clip_fraction"]: raise RuntimeError("selected formula invalid")
 return inv(scaled,t)
def pred_metrics(y,p):
 rho=float(spearmanr(y,p).statistic) if np.ptp(p)>1e-15 else 0.; pear=float(pearsonr(y,p).statistic) if np.ptp(p)>1e-15 else 0.
 return {"spearman":rho,"pearson":pear,"R2":float(r2_score(y,p)),"MAE":float(mean_absolute_error(y,p)),
  "huber_loss":float(np.mean(np.where(np.abs(y-p)<=1,.5*(y-p)**2,np.abs(y-p)-.5))),"prediction_SD":float(np.std(p,ddof=1)),
  "target_SD":float(np.std(y,ddof=1)),"prediction_target_SD_ratio":float(np.std(p,ddof=1)/np.std(y,ddof=1))}
def tail_metrics(y_raw,y,p):
 out={}
 hashes=np.arange(len(y))
 for pct in (10,20):
  n=max(1,int(math.ceil(len(y)*pct/100))); pi=np.lexsort((hashes,-p))[:n]; ti=set(np.lexsort((hashes,-y))[:n])
  out[f"top{pct}_capture"]=len(set(pi)&ti)/n; out[f"top{pct}_enrichment"]=float(np.mean(y_raw[pi])/np.mean(y_raw))
 return out

def rscript():
 return os.environ.get("BLADE_RSCRIPT") or shutil.which("Rscript") or r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe"
def extract_gp20(hashes,coords):
 out=AUDIT/"07_hgb_controls/gp20_features.parquet"
 if out.is_file():
  d=pd.read_parquet(out)
  if d.instance_hash.astype(str).tolist()!=hashes: raise RuntimeError("cached GP20 alignment")
  return d
 scaler=json.loads((ROOT/"assets/qd/reference_cloud/formal_raw_scaler.json").read_text()); params={x["feature_name"]:(x["center_median"],x["scale"]) for x in scaler["parameters"]}
 frames=[]
 for start in range(0,len(hashes),250):
  stop=min(start+250,len(hashes))
  with tempfile.TemporaryDirectory(prefix="cycle0_gp20_") as td:
   td=Path(td); inp=td/"xy.csv"; raw=td/"raw.csv"; schema=td/"schema.csv"
   with inp.open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f,lineterminator="\n"); w.writerow(["state_id","node_idx","x","y"])
    for h,xy in zip(hashes[start:stop],coords[start:stop]):
     for i,(x,y) in enumerate(xy): w.writerow([h,i,repr(float(x)),repr(float(y))])
   proc=subprocess.run([rscript(),str(ROOT/"src/features/extract_salesperson_raw_features.R"),str(inp),str(raw),str(schema)],capture_output=True,text=True,encoding="utf-8",errors="replace")
   if proc.returncode or not raw.is_file(): raise RuntimeError(proc.stderr[-1000:])
   d=pd.read_csv(raw); d=d.rename(columns={"state_id":"instance_hash"})
   if d.instance_hash.astype(str).tolist()!=hashes[start:stop] or d.get("feature_error",pd.Series([np.nan]*len(d))).notna().any(): raise RuntimeError("salesperson GP20 failure")
   frame=pd.DataFrame({"instance_hash":d.instance_hash.astype(str)})
   for name in GP20: frame[name]=np.arcsinh((d[name].to_numpy(float)-params[name][0])/params[name][1])
   frames.append(frame); print(f"GP20_FEATURE_PROGRESS={stop}/{len(hashes)}",flush=True)
 result=pd.concat(frames,ignore_index=True); result.to_parquet(out,index=False); return result

def selected_hof(hall,x,y,t):
 rows=[]
 for rec in hall:
  try:
   p=predict_record(rec,x,t); eligible=np.isfinite(p).all()
  except Exception: eligible=False; p=np.zeros(len(y))
  rho=float(spearmanr(y,p).statistic) if eligible and np.ptp(p)>1e-15 else 0.; mae=float(np.mean(np.abs(y-p))) if eligible else math.inf
  rows.append({**rec,"validation_eligible":eligible,"validation_spearman":rho,"validation_mae":mae})
 usable=[r for r in rows if r["validation_eligible"]]; best=max(r["validation_spearman"] for r in usable)
 finalists=[r for r in usable if r["validation_spearman"]>=best-1e-6]
 return min(finalists,key=lambda r:(r["nodes"],r["validation_mae"],r["canonical_formula"])),rows

def load_data():
 labels=pd.read_parquet(AUDIT/"03_oracle_labels/exact_hardness.parquet"); bundle=np.load(AUDIT/"03_oracle_labels/selected_coordinates.npz")
 hashes=bundle["instance_hash"].astype(str).tolist(); coords=bundle["coordinates"]
 if labels.instance_hash.astype(str).tolist()!=hashes: raise RuntimeError("label alignment")
 order=pd.read_parquet(AUDIT/"02_gp0_calibration_design/structural_probe_order.parquet")
 frozen_qd=pd.read_parquet(ROOT/"audits/uniform_rooted_qd_bootstrap_v1/05_offspring_qd_assignments.parquet",columns=["instance_hash",*Q4])
 dev=pd.read_parquet(AUDIT/"02_gp0_calibration_design/cycle0_gp_dev.parquet")
 base=(labels[["instance_hash","H0","H0_log1p"]]
       .merge(order[["instance_hash","niche_id","structural_rank","centroid_distance",*[f'normalized_{x}' for x in Q4]]],on="instance_hash",validate="one_to_one")
       .merge(frozen_qd,on="instance_hash",validate="one_to_one"))
 return base,dev,hashes,coords

def train_models():
 base,dev,hashes,coords=load_data(); gp20=extract_gp20(hashes,coords)
 base=base.merge(gp20[["instance_hash",*[x for x in GP20 if x not in Q4]]],on="instance_hash",validate="one_to_one")
 devh=dev.instance_hash.astype(str).tolist(); probe=base.set_index("instance_hash").loc[devh].reset_index(); predictions=[]
 for depth,outname in [(1,"04_gp4_d1"),(2,"05_gp4_d2"),(5,"06_gp4_d5")]:
  train=base[base.structural_rank.le(depth)].copy(); x=train[Q4].to_numpy(float); y=train.H0_log1p.to_numpy(float); xp=probe[Q4].to_numpy(float); t=robust(y); seed_preds=[]; formulas=[]
  for seed in range(2026,2031):
   sd=AUDIT/outname/f"seed_{seed}"; sd.mkdir(parents=True,exist_ok=True); formula_path=sd/"formula.json"
   if formula_path.is_file() and (sd/"training_log.parquet").is_file() and (sd/"hof_candidates.json").is_file():
    chosen=json.loads(formula_path.read_text())["selected"]; print(f"GP_D{depth}_SEED_{seed}=REUSED",flush=True)
   else:
    print(f"GP_D{depth}_SEED_{seed}=START",flush=True); log,hall=train_seed(x,(y-t['center_train_median_loss_log'])/t['scale_train_loss_log'],Q4,GP_CFG,seed)
    chosen,candidates=selected_hof(hall,x,y,t); log.to_parquet(sd/"training_log.parquet",index=False)
    js(formula_path,{"seed":seed,"selector":"SPERF","spearman_tie_epsilon":1e-6,"transform":t,"selected":chosen,"all_predictions_finite":True}); js(sd/"hof_candidates.json",candidates)
   p=predict_record(chosen,xp,t); seed_preds.append(p); formulas.append({"seed":seed,"formula":chosen["formula"],"canonical_formula":chosen["canonical_formula"],"nodes":chosen["nodes"]})
  ensemble=np.median(np.vstack(seed_preds),axis=0); predictions.append(pd.DataFrame({"instance_hash":devh,"model":f"GP4_D{depth}","prediction":ensemble}))
  js(AUDIT/outname/"ensemble.json",{"model":f"GP0_D{depth}","train_n":len(train),"seeds":list(range(2026,2031)),"operational_output":"median","formulas":formulas,"finite":bool(np.isfinite(ensemble).all()),"prediction_sd":float(np.std(ensemble,ddof=1)),"constant":bool(np.ptp(ensemble)<=1e-15)})
  for features,label in [(Q4,"HGB4"),(GP20,"HGB20")]:
   model=HistGradientBoostingRegressor(**HGB_CFG).fit(train[features].to_numpy(float),y); p=model.predict(probe[features].to_numpy(float)); predictions.append(pd.DataFrame({"instance_hash":devh,"model":f"{label}_D{depth}","prediction":p}))
   path=AUDIT/"07_hgb_controls"/f"{label.lower()}_d{depth}.joblib"; joblib.dump({"model":model,"features":features},path,compress=3)
 allpred=pd.concat(predictions,ignore_index=True); allpred.to_parquet(AUDIT/"08_prediction_evaluation/all_predictions.parquet",index=False)
 evaluate(base,dev,allpred)

def pair_accuracy(y,p):
 i,j=np.triu_indices(len(y),1); truth=np.sign(y[i]-y[j]); guess=np.sign(p[i]-p[j]); return float(np.mean(np.where(guess==0,.5,guess==truth)))
def evaluate(base,dev,pred):
 devh=dev.instance_hash.astype(str).tolist(); truth=base.set_index('instance_hash').loc[devh].reset_index(); y=truth.H0_log1p.to_numpy(float); raw=truth.H0.to_numpy(float)
 global_rows=[]; tail_rows=[]; niche_rows=[]
 for model,g in pred.groupby('model',sort=True):
  p=g.set_index('instance_hash').loc[devh].prediction.to_numpy(float); gm=pred_metrics(y,p); global_rows.append({"model":model,"train_n":len(base[base.structural_rank.le(int(model.rsplit('D',1)[1]))]),**gm}); tail_rows.append({"model":model,**tail_metrics(raw,y,p)})
  tmp=truth[["instance_hash","niche_id","H0","H0_log1p"]].copy(); tmp['prediction']=p
  for niche,nf in tmp.groupby('niche_id'):
   if len(nf)<5: continue
   yy=nf.H0_log1p.to_numpy(float); pp=nf.prediction.to_numpy(float)
   niche_rows.append({"model":model,"niche_id":int(niche),"n":len(nf),"spearman":float(spearmanr(yy,pp).statistic) if np.ptp(pp)>1e-15 else 0.,"kendall_tau":float(kendalltau(yy,pp).statistic) if np.ptp(pp)>1e-15 else 0.,"pairwise_accuracy":pair_accuracy(yy,pp)})
 pd.DataFrame(global_rows).to_csv(AUDIT/"08_prediction_evaluation/global_metrics.csv",index=False); pd.DataFrame(tail_rows).to_csv(AUDIT/"08_prediction_evaluation/tail_metrics.csv",index=False)
 per=pd.DataFrame(niche_rows); per.to_parquet(AUDIT/"09_within_niche_ranking/per_niche_metrics.parquet",index=False); summaries=[]
 for model,g in per.groupby('model'):
  row={"model":model,"eligible_niches":len(g)}
  for col in ('spearman','kendall_tau','pairwise_accuracy'):
   v=g[col].to_numpy(float); w=g.n.to_numpy(float); row.update({f"mean_{col}":float(v.mean()),f"median_{col}":float(np.median(v)),f"P25_{col}":float(np.quantile(v,.25)),f"P75_{col}":float(np.quantile(v,.75)),f"fraction_positive_{col}":float(np.mean(v>0)),f"fraction_gt_0.2_{col}":float(np.mean(v>.2)),f"fraction_gt_0.4_{col}":float(np.mean(v>.4)),f"candidate_weighted_{col}":float(np.average(v,weights=w))})
  summaries.append(row)
 pd.DataFrame(summaries).to_csv(AUDIT/"09_within_niche_ranking/summary.csv",index=False)
 acquisition_and_bootstrap(truth,pred,per)

def acquisition_and_bootstrap(truth,pred,per):
 models=[f"GP4_D{x}" for x in (1,2,5)]; truth=truth.copy(); rng=np.random.default_rng(2026)
 predmap={m:pred[pred.model.eq(m)].set_index('instance_hash').prediction for m in models}; rows={1:[],2:[],5:[]}; niche_detail=[]
 for niche,g in truth.groupby('niche_id',sort=True):
  if len(g)<5: continue
  hashes=g.instance_hash.astype(str).to_numpy(); y=g.H0.to_numpy(float); yh=g.H0_log1p.to_numpy(float); nearest=int(np.argmin(g.centroid_distance.to_numpy(float)))
  for r in (1,2,5):
   random_idx=np.sort(rng.choice(len(g),size=min(r,len(g)),replace=False)); random_mean=float(y[random_idx].mean()); structural_mean=float(y[np.argsort(g.centroid_distance.to_numpy(float),kind='mergesort')[:r]].mean())
   rows[r].extend([{"niche_id":int(niche),"model":"RANDOM","selected_true_H_mean":random_mean},{"niche_id":int(niche),"model":"STRUCTURAL_CENTROID","selected_true_H_mean":structural_mean}])
   for m in models:
    p=predmap[m].loc[hashes].to_numpy(float); idx=np.lexsort((hashes,-p))[:r]; mean=float(y[idx].mean()); pct=float(np.mean([(np.sum(y<=y[i])-.5)/len(y) for i in idx])); hit=float(int(np.argmax(y)) in idx)
    rows[r].append({"niche_id":int(niche),"model":m,"selected_true_H_mean":mean,"mean_percentile":pct,"true_best_hit":hit,"random_enrichment":mean/random_mean})
    niche_detail.append({"niche_id":int(niche),"model":m,"r":r,"selected_H":mean,"random_H":random_mean})
 for r,items in rows.items():
  f=pd.DataFrame(items); summary=[]
  for m,g in f.groupby('model'):
   v=g.selected_true_H_mean; summary.append({"model":m,"eligible_niches":len(g),"mean_true_H":float(v.mean()),"median_true_H":float(v.median()),"P75_true_H":float(v.quantile(.75)),"P90_true_H":float(v.quantile(.9)),"mean_percentile":float(g.mean_percentile.mean()) if 'mean_percentile' in g else math.nan,"median_percentile":float(g.mean_percentile.median()) if 'mean_percentile' in g else math.nan,"true_best_hit_rate":float(g.true_best_hit.mean()) if 'true_best_hit' in g else math.nan,"random_enrichment":float(g.random_enrichment.mean()) if 'random_enrichment' in g else math.nan})
  pd.DataFrame(summary).to_csv(AUDIT/f"10_acquisition_evaluation/top{r}.csv",index=False)
 detail=pd.DataFrame(niche_detail); detail.to_parquet(AUDIT/"10_acquisition_evaluation/per_niche_acquisition.parquet",index=False)
 bootstrap(truth,pred,per,detail); decisions_and_figures()

def bootstrap(truth,pred,per,detail):
 niches=np.sort(truth.niche_id.unique()); k=len(niches); rng=np.random.default_rng(2026); models=[f'GP4_D{x}' for x in (1,2,5)]
 niche_to_pos={int(n):i for i,n in enumerate(niches)}; truth=truth.reset_index(drop=True); y=truth.H0_log1p.to_numpy(float)
 row_by_pos=[np.flatnonzero(truth.niche_id.to_numpy(int)==n) for n in niches]
 predictions={m:pred[pred.model.eq(m)].set_index('instance_hash').loc[truth.instance_hash].prediction.to_numpy(float) for m in models}
 within_rho={m:np.full(k,np.nan) for m in models}; within_pair={m:np.full(k,np.nan) for m in models}
 acq={(m,r):np.full(k,np.nan) for m in models for r in (1,2,5)}; random_h={r:np.full(k,np.nan) for r in (1,2,5)}
 for row in per.itertuples(index=False):
  if row.model in models: pos=niche_to_pos[int(row.niche_id)]; within_rho[row.model][pos]=row.spearman; within_pair[row.model][pos]=row.pairwise_accuracy
 for row in detail.itertuples(index=False):
  pos=niche_to_pos[int(row.niche_id)]; acq[(row.model,int(row.r))][pos]=row.selected_H; random_h[int(row.r)][pos]=row.random_H
 def metrics(pos,model):
  row_idx=np.concatenate([row_by_pos[int(i)] for i in pos]); rho=float(spearmanr(y[row_idx],predictions[model][row_idx]).statistic)
  return {"global_spearman":rho,"median_within_niche_spearman":float(np.nanmedian(within_rho[model][pos])),"pairwise_ranking_accuracy":float(np.nanmean(within_pair[model][pos])),**{f"top{r}_true_H":float(np.nanmean(acq[(model,r)][pos])) for r in (1,2,5)}}
 original=np.arange(k); point={m:metrics(original,m) for m in models}
 boot={(d,k):[] for d in (1,2) for k in next(iter(point.values()))}
 random_boot={(d,r):[] for d in (1,2,5) for r in (1,2,5)}
 for b in range(10000):
  sample=rng.integers(0,k,size=k); ref=metrics(sample,'GP4_D5')
  for d in (1,2):
   cur=metrics(sample,f'GP4_D{d}')
   for metric_name in cur: boot[(d,metric_name)].append(cur[metric_name]-ref[metric_name])
  for d in (1,2,5):
   for r in (1,2,5):
    random_boot[(d,r)].append(float(np.nanmean(acq[(f'GP4_D{d}',r)][sample]-random_h[r][sample])))
  if (b+1)%1000==0: print(f"NICHE_BOOTSTRAP={b+1}/10000",flush=True)
 rows=[]
 for (d,k),vals in boot.items(): rows.append({"comparison":f"D{d}-D5","metric":k,"point_difference":point[f'GP4_D{d}'][k]-point['GP4_D5'][k],"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),"replicates":10000,"seed":2026})
 for (d,r),vals in random_boot.items(): rows.append({"comparison":f"D{d}-RANDOM","metric":f"top{r}_true_H","point_difference":float(np.mean(vals)),"ci_low":float(np.quantile(vals,.025)),"ci_high":float(np.quantile(vals,.975)),"replicates":10000,"seed":2026})
 pd.DataFrame(rows).to_csv(AUDIT/"10_acquisition_evaluation/bootstrap.csv",index=False)

def decisions_and_figures():
 boot=pd.read_csv(AUDIT/"10_acquisition_evaluation/bootstrap.csv"); acq={r:pd.read_csv(AUDIT/f"10_acquisition_evaluation/top{r}.csv").set_index('model') for r in (1,2,5)}; saturation=[]
 for d in (1,2,5):
  ratios={r:float(acq[r].loc[f'GP4_D{d}','mean_true_H']/acq[r].loc['GP4_D5','mean_true_H']) for r in (1,2,5)}; gates=[]
  for r in (1,2,5):
   if d==5: gates.append(True); continue
   ci=float(boot[(boot.comparison==f'D{d}-D5')&(boot.metric==f'top{r}_true_H')].iloc[0].ci_low); gain=float(acq[r].loc['GP4_D5','mean_true_H']-acq[r].loc['RANDOM','mean_true_H']); gates.append(ci>=-.05*gain)
  ens=json.loads((AUDIT/{1:'04_gp4_d1',2:'05_gp4_d2',5:'06_gp4_d5'}[d]/'ensemble.json').read_text()); stable=ens['finite'] and not ens['constant'] and len(ens['formulas'])==5
  b1=boot[(boot.comparison==f'D{d}-RANDOM')&(boot.metric=='top1_true_H')].iloc[0]
  d5_gain=float(acq[1].loc['GP4_D5','mean_true_H']-acq[1].loc['RANDOM','mean_true_H'])
  condition_c=bool(b1.point_difference>0 and (b1.ci_low>0 or b1.ci_low>=-.05*max(d5_gain,0.)))
  saturated=all(v>=.95 for v in ratios.values()) and all(gates)
  saturation.append({"depth":d,**{f"top{r}_ratio_to_D5":ratios[r] for r in (1,2,5)},"bootstrap_gate":all(gates),"stable":stable,"condition_C":condition_c,"saturated":saturated,"qualifies":saturated and stable and condition_c})
 sat=pd.DataFrame(saturation); choice=next((int(r.depth) for r in sat.itertuples() if r.qualifies),None); classification={1:'B1',2:'B2',5:'B3',None:'B4'}[choice]
 label={'B1':'ONE_PROBE_PER_REACHED_NICHE_SUFFICIENT','B2':'TWO_PROBES_PER_REACHED_NICHE_REQUIRED','B3':'FIVE_PROBES_PER_REACHED_NICHE_REQUIRED','B4':'GP0_ACQUISITION_SIGNAL_INSUFFICIENT'}[classification]
 js(AUDIT/"11_decision/phase_B_final.json",{"classification":classification,"label":label,"minimum_saturated_probes_per_niche":choice or 'NONE',"saturation":saturation,"final_protocol_modified":False})
 js(AUDIT/"11_decision/final_recommendation.json",{"phase_A":json.loads((AUDIT/'11_decision/phase_A_final.json').read_text()),"phase_B":json.loads((AUDIT/'11_decision/phase_B_final.json').read_text()),"human_review_required":True,"final_protocol_modified":False})
 make_figures()

def make_figures():
 gm=pd.read_csv(AUDIT/"08_prediction_evaluation/global_metrics.csv"); wm=pd.read_csv(AUDIT/"09_within_niche_ranking/summary.csv")
 gp=gm[gm.model.str.startswith('GP4')].copy(); gp['depth']=gp.model.str.extract(r'D(\d+)').astype(int)
 for num,col,title in [(4,'spearman','Global Spearman'),(5,'median_spearman','Median within-niche Spearman'),(6,'mean_pairwise_accuracy','Mean within-niche pairwise accuracy')]:
  src=gp if num==4 else wm[wm.model.str.startswith('GP4')].assign(depth=lambda x:x.model.str.extract(r'D(\d+)').astype(int)); fig,ax=plt.subplots(figsize=(6,4)); ax.plot(src.depth,src[col],marker='o'); ax.set(xticks=[1,2,5],xlabel='probes/niche',ylabel=title); fig.tight_layout(); fig.savefig(AUDIT/f"12_figures/{num:02d}_gp0_{'global_spearman' if num==4 else ('within_niche_ranking' if num==5 else 'pairwise_accuracy')}_vs_probe_depth.png",dpi=160); plt.close(fig)
 acqs={r:pd.read_csv(AUDIT/f"10_acquisition_evaluation/top{r}.csv") for r in (1,2,5)}
 fig,ax=plt.subplots(figsize=(6,4)); x=[1,2,5]; ax.plot(x,[float(acqs[1].set_index('model').loc[f'GP4_D{d}','mean_true_H']) for d in x],marker='o'); ax.set(xticks=x,xlabel='probes/niche',ylabel='Top1 acquired true H'); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/07_gp0_top1_acquisition_vs_probe_depth.png",dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));
 for r in (1,2,5): ax.plot(x,[float(acqs[r].set_index('model').loc[f'GP4_D{d}','mean_true_H']) for d in x],marker='o',label=f'Top{r}')
 ax.legend(); ax.set(xticks=x,xlabel='probes/niche',ylabel='acquired true H'); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/08_gp0_multilayer_acquisition_vs_probe_depth.png",dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4)); width=.22
 for j,r in enumerate((1,2,5)): ax.bar(np.arange(3)+(j-1)*width,[float(acqs[r].set_index('model').loc[f'GP4_D{d}','random_enrichment']) for d in x],width,label=f'Top{r}')
 ax.set(xticks=np.arange(3),xticklabels=['D1','D2','D5'],ylabel='enrichment over matched random'); ax.legend(); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/09_gp0_acquisition_enrichment_vs_random.png",dpi=160); plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4)); ax.bar(gm.model,gm.spearman); ax.tick_params(axis='x',rotation=50); ax.set_ylabel('dev Spearman'); fig.tight_layout(); fig.savefig(AUDIT/"12_figures/10_gp4_vs_hgb4_vs_hgb20.png",dpi=160); plt.close(fig)

if __name__=='__main__': train_models()
