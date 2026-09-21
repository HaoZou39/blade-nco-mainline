from __future__ import annotations
import gc,json,os,time
import numpy as np,pandas as pd,torch
from audit_common import *
from bq.checkpoint import load_canonical_checkpoint
from bq.training_hardness import teacher_forced_training_hardness,protocol_manifest
from mutation import apply_operator
from oracle.concorde import canonical_oracle_ledger,resolve_backend,solve_batch
from qd.coordinates import load_centroids
from qd.roots import recover_uniform_roots

def reconstruct_arms():
    qd=pd.read_parquet(O0_QD); qd["instance_hash"]=qd.instance_hash.astype(str)
    if len(qd)!=O0_N or qd.instance_hash.nunique()!=O0_N: raise RuntimeError("O0 identity failure")
    centers=load_centroids(); rows=[]
    if centers.shape!=(1000,4): raise RuntimeError(f"centroid shape {centers.shape}")
    for niche,f in qd.groupby("niche_id",sort=True):
        x=f[NORM].to_numpy(float); hs=f.instance_hash.to_numpy(str); d=np.linalg.norm(x-centers[int(niche)],axis=1)
        tied=np.flatnonzero(np.isclose(d,d.min(),rtol=0,atol=1e-15)); k=int(tied[np.argmin(hs[tied])])
        rows.append({"niche_id":int(niche),"instance_hash":str(hs[k]),"centroid_distance":float(d[k])})
    old=pd.DataFrame(rows).sort_values("niche_id").reset_index(drop=True)
    hist=json.loads(OLD_JSON.read_text(encoding="utf-8")); hh=[str(x["instance_hash"]) for x in hist["members"]]; hn=[int(x["niche_id"]) for x in hist["members"]]
    if old.instance_hash.tolist()!=hh or old.niche_id.tolist()!=hn or member_sha(hh)!=hist["ordered_membership_sha256"]: raise RuntimeError("Old435 reconstruction mismatch")
    old=old.merge(qd,on=["niche_id","instance_hash"],validate="one_to_one"); old.insert(0,"sample_rank",np.arange(1,len(old)+1)); old.insert(0,"arm","OLD_CENTROID435")
    stable=qd.sort_values("instance_hash",kind="mergesort").reset_index(drop=True); idx=np.random.default_rng(SAMPLE_SEED).permutation(len(stable))[:RANDOM_N]
    rnd=stable.iloc[idx].copy().reset_index(drop=True); rnd.insert(0,"stable_pool_index",idx); rnd.insert(0,"sample_rank",np.arange(1,RANDOM_N+1)); rnd.insert(0,"arm","RANDOM500")
    if len(old)!=OLD_N or old.niche_id.nunique()!=OLD_N or len(rnd)!=RANDOM_N or rnd.instance_hash.nunique()!=RANDOM_N: raise RuntimeError("arm size failure")
    info={"o0_n":len(qd),"o0_unique_hashes":qd.instance_hash.nunique(),"reached_niches":qd.niche_id.nunique(),"old_n":len(old),"old_niches":old.niche_id.nunique(),"old_exact":True,"old_membership_sha256":member_sha(old.instance_hash.tolist()),"random_n":len(rnd),"random_unique":rnd.instance_hash.nunique(),"random_seed":SAMPLE_SEED,"random_draw_count":1,"random_algorithm":"instance_hash ascending + default_rng(2026).permutation[:500]","random_membership_sha256":member_sha(rnd.instance_hash.tolist()),"stable_pool_sha256":member_sha(stable.instance_hash.tolist())}
    return qd,old,rnd,info

def replay_random_coordinates(rnd):
    m=pd.read_parquet(O0_MANIFEST); u=m[m.proposal_status.eq("accepted")&m.valid_after_repair.astype(bool)&~m.duplicate_status.astype(bool)].copy(); u["child_hash"]=u.child_hash.astype(str)
    if len(u)!=O0_N or u.child_hash.duplicated().any(): raise RuntimeError("usable O0 mismatch")
    rows=u.set_index("child_hash",drop=False).loc[rnd.instance_hash.tolist()].reset_index(drop=True); roots=recover_uniform_roots(); coords=[]
    for i,r in enumerate(rows.itertuples(index=False),1):
        z=apply_operator(str(r.operator_id),roots.coordinates[int(r.root_index)],np.random.default_rng(int(r.child_seed)))
        if z.final_output_hash!=str(r.child_hash): raise RuntimeError(f"coordinate replay mismatch {r.child_hash}")
        coords.append(z.final_coordinates)
        if i%100==0: print(f"RANDOM500_COORDINATE_REPLAY={i}/500",flush=True)
    result=np.stack(coords).astype(np.float64)
    if result.shape!=(500,100,2): raise RuntimeError(str(result.shape))
    return result

def tour_cache(wanted):
    paths=[OUT/"oracle/random500_ledger.parquet",HIST_TOURS,RUN/"cycles/C00/07_oracle/gp0_probes/ledger.parquet",RUN/"cycles/C00/07_oracle/selected/ledger.parquet"]
    rows={}; sources={}; ws=set(wanted)
    for p in paths:
        if not p.is_file(): continue
        f=canonical_oracle_ledger(pd.read_parquet(p)); f["instance_hash"]=f.instance_hash.astype(str)
        for r in f[f.instance_hash.isin(ws)].itertuples(index=False):
            d=r._asdict(); h=str(d["instance_hash"])
            if h in rows and str(rows[h]["tour_hash"])!=str(d["tour_hash"]): raise RuntimeError(f"tour conflict {h}")
            if h not in rows and bool(d["verified"]) and d["status"]=="solved": rows[h]=d; sources[h]=str(p)
    ordered=[rows[h] for h in wanted if h in rows]
    return (canonical_oracle_ledger(pd.DataFrame(ordered)) if ordered else None),sources

def label_random(rnd,coords):
    od=OUT/"oracle"; od.mkdir(parents=True,exist_ok=True); hashes=rnd.instance_hash.tolist(); np.savez_compressed(od/"random500_coordinates.npz",instance_hash=np.asarray(hashes),coordinates=coords)
    cached,sources=tour_cache(hashes); hits=0 if cached is None else len(cached); backend=resolve_backend(ROOT/"third_party/concorde/concorde")
    t=time.perf_counter(); ledger=solve_batch(coords,hashes,od/"concorde_work",backend,workers=8,existing=cached); concorde_s=time.perf_counter()-t; ledger.to_parquet(od/"random500_ledger.parquet",index=False)
    if ledger.instance_hash.astype(str).tolist()!=hashes or not ledger.verified.astype(bool).all(): raise RuntimeError("Oracle ledger failure")
    hp=od/"random500_exact_hardness.parquet"; mp=od/"random500_hardness_meta.json"; membership=member_sha(hashes); m0sha=sha(M0); hardness_s=0.; reused=False
    if hp.is_file() and mp.is_file():
        meta=json.loads(mp.read_text(encoding="utf-8")); labeled=pd.read_parquet(hp)
        if meta.get("m0_sha256")!=m0sha or meta.get("membership_sha256")!=membership or labeled.instance_hash.astype(str).tolist()!=hashes: raise RuntimeError("stale hardness cache")
        reused=True
    else:
        tours=np.asarray([[int(x) for x in str(v).split()] for v in ledger.expert_tour],dtype=np.int64); torch.set_num_threads(max(1,min(16,os.cpu_count() or 1))); device=torch.device("cpu"); model,meta=load_canonical_checkpoint(M0,device)
        t=time.perf_counter(); h=teacher_forced_training_hardness(model,coords,tours,device,batch_size=128); hardness_s=time.perf_counter()-t; labeled=ledger.copy()
        for k,v in h.items(): labeled[k]=v
        labeled["H0"]=labeled.loss_raw; labeled["H0_log1p"]=labeled.loss_log; labeled.to_parquet(hp,index=False)
        write_json(mp,{"m0_sha256":m0sha,"membership_sha256":membership,"checkpoint_meta":meta,"protocol":protocol_manifest(),"device":"cpu","parameter_updates":0,"wall_seconds":hardness_s}); del model; gc.collect()
    hist=pd.read_parquet(HIST_HARD); hist["instance_hash"]=hist.instance_hash.astype(str); overlap=labeled.merge(hist[["instance_hash","tour_hash","H0"]],on="instance_hash",suffixes=("","_hist"))
    if len(overlap) and not np.array_equal(overlap.tour_hash.astype(str),overlap.tour_hash_hist.astype(str)): raise RuntimeError("historical tour mismatch")
    info={"cache_hits":hits,"new_concorde_calls":RANDOM_N-hits,"failures":0,"concorde_wall_seconds":concorde_s,"hardness_refreshes":0 if reused else RANDOM_N,"hardness_wall_seconds":hardness_s,"hardness_cache_reused":reused,"historical_h0_overlap_n":len(overlap),"historical_h0_max_abs_difference":float(np.max(np.abs(overlap.H0-overlap.H0_hist))) if len(overlap) else None,"backend":backend,"cache_source_counts":pd.Series(list(sources.values())).value_counts().to_dict()}
    return labeled,info

def attach_hardness_and_p0(old,rnd,rnd_h):
    hist=pd.read_parquet(HIST_HARD); hist["instance_hash"]=hist.instance_hash.astype(str)
    if hist.instance_hash.duplicated().any(): raise RuntimeError("duplicate historical H0")
    old=old.merge(hist[["instance_hash","tour_hash","H0","H0_log1p"]],on="instance_hash",how="left",validate="one_to_one")
    rnd_h=rnd_h.copy(); rnd_h["instance_hash"]=rnd_h.instance_hash.astype(str); rnd=rnd.merge(rnd_h[["instance_hash","tour_hash","H0","H0_log1p"]],on="instance_hash",how="left",validate="one_to_one")
    if old.H0.isna().any() or rnd.H0.isna().any(): raise RuntimeError("calibration H0 incomplete")
    p0=pd.read_csv(P0_CSV); p0["instance_hash"]=p0.instance_hash.astype(str)
    if len(p0)!=P0_N or p0.instance_hash.duplicated().any(): raise RuntimeError("P0 identity failure")
    p0=p0.merge(hist[["instance_hash","H0","tour_hash"]],on="instance_hash",how="left",suffixes=("","_label"),validate="one_to_one")
    if p0.H0.isna().any() or not np.array_equal(p0.tour_hash.astype(str),p0.tour_hash_label.astype(str)): raise RuntimeError("P0 label mismatch")
    om=p0.instance_hash.isin(set(old.instance_hash)); rm=p0.instance_hash.isin(set(rnd.instance_hash)); shared=p0[~(om|rm)].copy().reset_index(drop=True)
    if set(shared.instance_hash)&(set(old.instance_hash)|set(rnd.instance_hash)): raise RuntimeError("P0 leakage")
    leak={"raw_p0_n":len(p0),"raw_p0_membership_sha256":member_sha(p0.instance_hash.tolist()),"excluded_old_overlap_n":int(om.sum()),"excluded_random_overlap_n":int(rm.sum()),"excluded_union_n":int((om|rm).sum()),"final_shared_p0_n":len(shared),"final_shared_p0_membership_sha256":member_sha(shared.instance_hash.tolist()),"same_panel_for_both_arms":True,"remaining_overlap_n":0}
    (OUT/"05_leakage_audit.md").write_text(f"# Leakage audit\n\n- Raw canonical E1/E2 P0: **{len(p0)}**\n- Old overlap excluded: **{int(om.sum())}**\n- Random overlap excluded: **{int(rm.sum())}**\n- Unique exclusions: **{int((om|rm).sum())}**\n- Final shared panel: **{len(shared)}**\n- Remaining overlap: **0**\n\nMembership SHA-256: `{leak['final_shared_p0_membership_sha256']}`.\n",encoding="utf-8")
    return old,rnd,shared,leak
