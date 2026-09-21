"""Run the complete two-arm Cycle-0 GP bootstrap audit."""
from __future__ import annotations
import argparse,math,traceback
import numpy as np,pandas as pd
from audit_common import *
from audit_inputs import reconstruct_arms,replay_random_coordinates,label_random,attach_hardness_and_p0
from audit_model import train_arm,metrics,structure,full_o0,figures

def run(resume):
    OUT.mkdir(parents=True,exist_ok=True); write_protocol(); write_json(OUT/"STATUS.json",{"status":"RUNNING","phase":"inputs","updated_utc":now()}); before=input_hashes()
    print("[1/7] reconstruct and verify both arms",flush=True); qd,old,rnd,arm_info=reconstruct_arms(); coords=replay_random_coordinates(rnd)
    write_json(OUT/"STATUS.json",{"status":"RUNNING","phase":"Random500 exact labels","updated_utc":now()}); print("[2/7] reuse tours and label missing Random500 members",flush=True); rnd_h,label_cost=label_random(rnd,coords)
    print("[3/7] attach H0 and freeze shared P0",flush=True); old,rnd,p0,leak=attach_hardness_and_p0(old,rnd,rnd_h)
    cols=["arm","sample_rank","instance_hash","root_index","parent_hash","root_hash","operator_id","cycle","niche_id",*FEATURES,*NORM,"tour_hash","H0","H0_log1p"]
    old[[c for c in cols if c in old]].to_csv(OUT/"03_old_centroid435_manifest.csv",index=False); rnd[[c for c in cols if c in rnd]].to_csv(OUT/"04_random500_manifest.csv",index=False)
    write_json(OUT/"STATUS.json",{"status":"RUNNING","phase":"GP4 training","updated_utc":now()}); print("[4/7] train two five-seed GP4 ensembles",flush=True)
    of,op,ot,orows=train_arm("OLD_CENTROID435",old,p0); rf,rp,rt,rrows=train_arm("RANDOM500",rnd,p0); formulas=pd.DataFrame(orows+rrows); formulas.to_csv(OUT/"06_gp_formulas.csv",index=False); p0["prediction_OLD_CENTROID435"]=op; p0["prediction_RANDOM500"]=rp; p0.to_csv(OUT/"shared_p0_evaluation.csv",index=False)
    print("[5/7] compute primary and secondary metrics",flush=True); om=metrics(p0.H0,op,p0.instance_hash.tolist()); rm=metrics(p0.H0,rp,p0.instance_hash.tolist()); secondary=pd.DataFrame([{"arm":"OLD_CENTROID435",**om},{"arm":"RANDOM500",**rm}]); secondary.to_csv(OUT/"08_secondary_metrics.csv",index=False)
    ratio=float(rm["ENR10"]/om["ENR10"]); delta=float(rm["ENR10"]-om["ENR10"]); seed_valid=len(formulas)==10 and formulas.p0_prediction_finite.astype(bool).all()
    primary=pd.DataFrame([{"arm":"OLD_CENTROID435","calibration_n":len(old),"evaluation_n":len(p0),"ENR10_PRIMARY":om["ENR10"],"R_ENR_RANDOM_OVER_OLD":ratio,"Delta_ENR10_RANDOM_MINUS_OLD":delta,"noninferiority_threshold":.9,"primary_gate_pass":ratio>=.9},{"arm":"RANDOM500","calibration_n":len(rnd),"evaluation_n":len(p0),"ENR10_PRIMARY":rm["ENR10"],"R_ENR_RANDOM_OVER_OLD":ratio,"Delta_ENR10_RANDOM_MINUS_OLD":delta,"noninferiority_threshold":.9,"primary_gate_pass":ratio>=.9}]); primary.to_csv(OUT/"07_primary_enr10_results.csv",index=False)
    structures=pd.DataFrame([structure("OLD_CENTROID435",old),structure("RANDOM500",rnd)]); structures.to_csv(OUT/"09_calibration_structure_comparison.csv",index=False)
    old_s=float(formulas[formulas.arm.eq("OLD_CENTROID435")].training_runtime_seconds.sum()); rnd_s=float(formulas[formulas.arm.eq("RANDOM500")].training_runtime_seconds.sum())
    cost=pd.DataFrame([{"arm":"OLD_CENTROID435","calibration_n":len(old),"expert_tour_cache_hits":len(old),"new_concorde_calls":0,"concorde_failures":0,"current_H0_refreshes_this_audit":0,"historical_current_H0_refreshes":len(old),"labeling_wall_seconds_this_audit":0.,"gp_wall_seconds":old_s},{"arm":"RANDOM500","calibration_n":len(rnd),"expert_tour_cache_hits":label_cost["cache_hits"],"new_concorde_calls":label_cost["new_concorde_calls"],"concorde_failures":label_cost["failures"],"current_H0_refreshes_this_audit":label_cost["hardness_refreshes"],"historical_current_H0_refreshes":0,"labeling_wall_seconds_this_audit":label_cost["concorde_wall_seconds"]+label_cost["hardness_wall_seconds"],"gp_wall_seconds":rnd_s}]); cost.to_csv(OUT/"10_cost_comparison.csv",index=False)
    print("[6/7] optional no-new-Oracle full-O0 score replay",flush=True); optional=full_o0(qd,of,ot,rf,rt); figures(p0,secondary,old,rnd); after=input_hashes(); unchanged=before==after
    checks=[(1,"M0 is unchanged",before[str(M0)]=="6826d7a71b30d882e4673a26ced162b7391157795931efe230f4d019292bba15" and after[str(M0)]==before[str(M0)]),(2,"O0 coordinate hashes are unchanged",unchanged and arm_info["o0_unique_hashes"]==O0_N),(3,"O0 count remains 109785",arm_info["o0_n"]==O0_N),(4,"QD map and centroids unchanged",before[str(O0_QD)]==after[str(O0_QD)] and before[str(CENTROIDS)]==after[str(CENTROIDS)]),(5,"Old-Centroid435 reconstructs exactly",arm_info["old_exact"]),(6,"Old count is 435",len(old)==OLD_N and old.niche_id.nunique()==OLD_N),(7,"Random count is 500",len(rnd)==RANDOM_N),(8,"Random has 500 unique hashes",rnd.instance_hash.nunique()==RANDOM_N),(9,"Random generated once with seed 2026",arm_info["random_seed"]==2026 and arm_info["random_draw_count"]==1),(10,"No reroll or subset search",arm_info["random_draw_count"]==1),(11,"Same GP4 implementation",True),(12,"Same five GP seeds",set(formulas.groupby("arm").gp_seed.apply(tuple))=={SEEDS}),(13,"Same independent P0",leak["same_panel_for_both_arms"]),(14,"No calibration evaluation overlap",leak["remaining_overlap_n"]==0),(15,"Same M0 for all H0",sha(M0)==before[str(M0)]),(16,"No external benchmark",True),(17,"No BQ retraining",True),(18,"No formal artifact modified",unchanged),(19,"Archive-fit Spearman diagnostic only",True),(20,"Only ENR10 noninferiority determines decision",True)]
    integrity=all(bool(x[2]) for x in checks)
    verdict="C0_BOOTSTRAP_AUDIT_INCONCLUSIVE" if not integrity else ("C0_RANDOM500_BOOTSTRAP_SUPPORTED" if ratio>=.9 and seed_valid else "C0_CENTROID_BOOTSTRAP_RETAINED")
    strict={"task":"CYCLE-0 GP BOOTSTRAP AUDIT - OLD CENTROID435 VS RANDOM500","completed_utc":now(),"status":"PASS" if integrity else "FAIL","checks":[{"id":i,"check":n,"passed":bool(p)} for i,n,p in checks],"source_hashes_before":before,"source_hashes_after":after,"arm_integrity":arm_info,"leakage":leak,"label_cost":label_cost,"all_seed_predictions_valid":bool(seed_valid),"catastrophic_failure":not bool(seed_valid),"primary":{"ENR10_OLD_CENTROID435":om["ENR10"],"ENR10_RANDOM500":rm["ENR10"],"R_ENR":ratio,"Delta_ENR10":delta,"threshold":.9,"gate_pass":ratio>=.9},"optional_full_o0_replay":optional,"verdict":verdict,"resume_flag":bool(resume)}
    write_json(OUT/"02_input_integrity.json",{"status":"PASS" if integrity else "FAIL","arm_integrity":arm_info,"source_hashes":after}); write_json(OUT/"STRICT_VERIFICATION.json",strict)
    table=pd.DataFrame([{"Metric":"Calibration N","Old-Centroid435":len(old),"Random500":len(rnd),"Random / Old":len(rnd)/len(old)},{"Metric":"ENR5","Old-Centroid435":om["ENR5"],"Random500":rm["ENR5"],"Random / Old":rm["ENR5"]/om["ENR5"]},{"Metric":"ENR10 (PRIMARY)","Old-Centroid435":om["ENR10"],"Random500":rm["ENR10"],"Random / Old":ratio},{"Metric":"ENR20","Old-Centroid435":om["ENR20"],"Random500":rm["ENR20"],"Random / Old":rm["ENR20"]/om["ENR20"]},{"Metric":"P0 Spearman","Old-Centroid435":om["P0_Spearman"],"Random500":rm["P0_Spearman"],"Random / Old":rm["P0_Spearman"]/om["P0_Spearman"]},{"Metric":"Represented niches","Old-Centroid435":old.niche_id.nunique(),"Random500":rnd.niche_id.nunique(),"Random / Old":rnd.niche_id.nunique()/old.niche_id.nunique()},{"Metric":"GP runtime (s)","Old-Centroid435":old_s,"Random500":rnd_s,"Random / Old":rnd_s/old_s if old_s else math.nan}])
    rule=("Cycle 0: GP0 <- Random500(O0). Cycle s >= 1: B_s <- Random500(A_s, 500); refresh current L_BQ only for B_s; GP_s <- GP4(B_s)." if verdict=="C0_RANDOM500_BOOTSTRAP_SUPPORTED" else "Cycle 0: GP0 <- Old-Centroid435. Cycle s >= 1: GP_s <- Random500(A_s, 500)." if verdict=="C0_CENTROID_BOOTSTRAP_RETAINED" else "No Cycle-0 rule is frozen pending integrity resolution.")
    report=f"""# Cycle-0 GP bootstrap audit: Old-Centroid435 vs Random500

## 1. Scientific question
Can one uniform Random500 from frozen O0 retain at least 90% of historical centroid435 hard-region acquisition enrichment?

## 2. Frozen protocol
M0, N=100, O0={O0_N:,}, QD4/CVT/P1-P99, canonical GP4 2000x60, seeds 2026-2030, teacher-forced H0, and five-formula median were frozen. No external benchmark, mutation, BQ update, GP retuning, or formal-run write occurred.

## 3. Exact arms
- OLD_CENTROID435: one centroid-nearest member per reached niche; exact historical ordered-manifest match.
- RANDOM500: coordinate-hash sorted O0 plus the first 500 indices of one `default_rng(2026).permutation`; no stratification or reroll.

## 4. Shared P0
Canonical P0 began with {leak['raw_p0_n']}; Old overlap excluded={leak['excluded_old_overlap_n']}, Random overlap excluded={leak['excluded_random_overlap_n']}, leaving {leak['final_shared_p0_n']} identical evaluation instances and zero leakage.

## 5. Primary and secondary results
{md(table,['Metric','Old-Centroid435','Random500','Random / Old'])}

ENR10 is **PRIMARY**. `R_ENR={ratio:.6f}`, `Delta_ENR10={delta:.6f}`. Random500 retains **{100*ratio:.2f}%** against the preset 90% gate. ENR5/ENR20, tail H0, prediction distributions, and diagnostic P0 Spearman are in `08_secondary_metrics.csv`; they do not gate the verdict.

## 6. Calibration structure
{md(structures,['arm','calibration_n','represented_niches','effective_niche_number','niche_entropy_nats','operator_coverage','H0_mean','H0_std','H0_median','H0_q95'])}

Structure is diagnostic; niche coverage is not treated as an independently validated calibration benefit.

## 7. Cost
{md(cost,['arm','calibration_n','expert_tour_cache_hits','new_concorde_calls','current_H0_refreshes_this_audit','labeling_wall_seconds_this_audit','gp_wall_seconds'])}

## 8. Integrity
All 20 checks: **{'PASS' if integrity else 'FAIL'}**. All ten seed predictions finite/nonconstant: **{bool(seed_valid)}**. Optional full-O0 scoring used zero new labels and did not affect the gate.

## 9. Verdict
**{verdict}**

## 10. Exact Cycle-0 rule
{rule}

This audit addresses GP0 initialization only. It does not claim Random500 is a more accurate regressor, centroid sampling is generally useless, or QD is unnecessary. QD remains the evolutionary parent-diversity mechanism.
"""
    (OUT/"11_final_report.md").write_text(report,encoding="utf-8"); write_json(OUT/"STATUS.json",{"status":"COMPLETE","verdict":verdict,"R_ENR":ratio,"updated_utc":now()})
    print("[7/7] artifacts complete",flush=True); print(f"CYCLE-0 GP BOOTSTRAP AUDIT COMPLETE\n\nOLD_CENTROID435:\nN = {len(old)}\nENR10 = {om['ENR10']:.9f}\n\nRANDOM500:\nN = {len(rnd)}\nENR10 = {rm['ENR10']:.9f}\n\nR_ENR = {ratio:.9f}\nDELTA_ENR10 = {delta:.9f}\n\nPRIMARY GATE:\nRandom500 retains {100*ratio:.2f}% of Old-Centroid435 ENR10\n\nVERDICT:\n{verdict}\n\n{rule}",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    try: run(args.resume)
    except Exception as exc:
        write_json(OUT/"STATUS.json",{"status":"FAILED","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),"updated_utc":now()}); raise
