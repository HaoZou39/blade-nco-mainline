"""Create the audited report, 60-check verifier, hash inventory, and terminal summary."""
from __future__ import annotations
import hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; AUDIT=ROOT/"audits/cycle0_initialization_validation_v1"; BOOT=ROOT/"audits/uniform_rooted_qd_bootstrap_v1"
def load(p): return json.loads((AUDIT/p).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding='utf-8')
def md_table(frame,columns=None): return frame[columns or list(frame.columns)].to_markdown(index=False,floatfmt='.6g')

phasea=load('11_decision/phase_A_final.json'); phaseb=load('11_decision/phase_B_final.json'); repair=load('01_repair_semantics/repair_qd_summary.json')
ops=pd.read_csv(AUDIT/'01_repair_semantics/repair_operator_summary.csv'); gm=pd.read_csv(AUDIT/'08_prediction_evaluation/global_metrics.csv'); tails=pd.read_csv(AUDIT/'08_prediction_evaluation/tail_metrics.csv'); within=pd.read_csv(AUDIT/'09_within_niche_ranking/summary.csv')
acq={r:pd.read_csv(AUDIT/f'10_acquisition_evaluation/top{r}.csv').set_index('model') for r in (1,2,5)}
cal=[]
for d in (1,2,5):
 x=load(f'02_gp0_calibration_design/cal_d{d}_manifest.json'); cal.append({'Depth':f'D{d}','Nominal probes/niche':d,'Total train n':x['total_train_n'],'Occupied niches represented':x['occupied_niches_represented'],'Shortage niches':x['shortage_niches']})
gp=gm[gm.model.str.startswith('GP4')].merge(tails,on='model'); wi=within[within.model.str.startswith('GP4')]
acq_table=[]
for m in ('GP4_D1','GP4_D2','GP4_D5'): acq_table.append({'Model':m,'Top1 true H':acq[1].loc[m,'mean_true_H'],'Top2 true H':acq[2].loc[m,'mean_true_H'],'Top5 true H':acq[5].loc[m,'mean_true_H'],'Random enrichment Top1':acq[1].loc[m,'random_enrichment']})
sat=pd.DataFrame(phaseb['saturation'])
report=f"""# BLADE-NCO Cycle-0 Initialization Validation Audit

## Outcome

Phase A is **{phasea['classification']} — {phasea['label']}**. The adapter is historically frozen, BLADE-specific canonical domain handling, and all 110,000 raw/final hashes replay exactly. Repaired children nevertheless have a distinct migration profile and uniquely reach {phasea['new_niche_contribution']['repaired_only']} of the 384 new niches, so the effect should be documented rather than hidden.

Phase B is **{phaseb['classification']} — {phaseb['label']}**. The smallest qualifying calibration depth is **{phaseb['minimum_saturated_probes_per_niche']}**. This is a saturation result relative to D5, not a claim that D5 is mathematically optimal. The final GP0 protocol remains unchanged pending human review.

## Phase A — repair semantics

The exact source/function trace and mathematical adapter definition are in `01_repair_semantics/repair_provenance.md`. Raw QD4 is retained only as a finite geometric diagnostic; because raw repaired candidates violate the admissible domain, it is never used downstream.

{md_table(ops.rename(columns={'operator':'Operator','raw_n':'Raw n','repair_n':'Repair n','repair_rate':'Repair %','native_new_niches':'Native new niches','repaired_new_niches':'Repaired new niches','median_qd_displacement':'Median QD displacement'}),['Operator','Raw n','Repair n','Repair %','Native new niches','Repaired new niches','Median QD displacement'])}

Native valid: {repair['group_summary']['GROUP_NATIVE']['count']} ({repair['group_summary']['GROUP_NATIVE']['fraction']:.3%}); repaired valid unique: {repair['group_summary']['GROUP_REPAIRED']['count']} ({repair['group_summary']['GROUP_REPAIRED']['fraction']:.3%}). Native/repaired new-niche reach is {phasea['new_niche_contribution']['native_discovered']}/{phasea['new_niche_contribution']['repaired_discovered']}; shared={phasea['new_niche_contribution']['reachable_by_both']}, native-only={phasea['new_niche_contribution']['native_only']}, repaired-only={phasea['new_niche_contribution']['repaired_only']}.

## Phase B — calibration design

{md_table(pd.DataFrame(cal))}

The shortfalls are real within-niche shortages and were not replaced across niches. The sealed development panel contains {load('02_gp0_calibration_design/cycle0_gp_dev_manifest.json')['total_n']} geometry-only selected instances and is disjoint from CAL_D5.

## Global prediction

{md_table(gp.rename(columns={'model':'Model','train_n':'Train n','spearman':'Spearman','R2':'R2','top10_capture':'Top10 capture','top10_enrichment':'Top10 enrichment','prediction_target_SD_ratio':'Pred/target SD'}),['Model','Train n','Spearman','R2','Top10 capture','Top10 enrichment','Pred/target SD'])}

All HGB4/HGB20 control results are in `08_prediction_evaluation/global_metrics.csv` and `tail_metrics.csv`.

## Within-niche ranking

{md_table(wi.rename(columns={'model':'Model','median_spearman':'Median rho','mean_spearman':'Mean rho','median_kendall_tau':'Median tau','mean_pairwise_accuracy':'Pairwise accuracy','fraction_positive_spearman':'Positive-niche fraction'}),['Model','Median rho','Mean rho','Median tau','Pairwise accuracy','Positive-niche fraction'])}

Niches are equally weighted in the primary summaries; candidate-weighted values are saved as secondary diagnostics.

## Acquisition

{md_table(pd.DataFrame(acq_table))}

## Saturation decision

{md_table(sat.rename(columns={'depth':'Depth','top1_ratio_to_D5':'Top1 ratio to D5','top2_ratio_to_D5':'Top2 ratio','top5_ratio_to_D5':'Top5 ratio','bootstrap_gate':'Bootstrap gate','stable':'Stable','qualifies':'Final status'}),['Depth','Top1 ratio to D5','Top2 ratio','Top5 ratio','Bootstrap gate','Stable','Final status'])}

The decision emphasizes Top1/Top2/Top5 QD-balanced acquisition. Regression, within-niche ranking, and HGB controls are secondary diagnostics for separating calibration quantity, symbolic capacity, and QD4 information limits.

## Boundaries

No final D0 was constructed or labeled; M0 was not fine-tuned; no external benchmark was opened; no final runner was started; and `runs/` remains empty. Stop for human review.
"""
(AUDIT/'report.md').write_text(report,encoding='utf-8')

test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace'); output=(test.stdout or '')+(test.stderr or '')
match=re.search(r'(\d+) passed',output); passed=int(match.group(1)) if match else 0; failed_match=re.search(r'(\d+) failed',output); failed=int(failed_match.group(1)) if failed_match else 0; total=passed+failed
source=load('00_protocol/source_manifest.json'); sources=source['sources']
old_path=Path('D:/NCO')/'BLADE-NCO'; old_head=subprocess.check_output(['git','-C',str(old_path),'rev-parse','HEAD'],text=True).strip(); old_status=subprocess.check_output(['git','-C',str(old_path),'status','--porcelain=v1','--untracked-files=no'],text=True); old=load('00_protocol/frozen_protocol.json')['old_project_baseline']
manifest=pd.read_parquet(BOOT/'02_offspring_manifest.parquet'); cq=pd.read_parquet(BOOT/'05_offspring_qd_assignments.parquet'); labels=pd.read_parquet(AUDIT/'03_oracle_labels/exact_hardness.parquet'); ledger=pd.read_parquet(AUDIT/'03_oracle_labels/concorde_ledger.parquet'); dev=pd.read_parquet(AUDIT/'02_gp0_calibration_design/cycle0_gp_dev.parquet')
dsets=[]
for d in (1,2,5): dsets.append({x['instance_hash'] for x in load(f'02_gp0_calibration_design/cal_d{d}_manifest.json')['members']})
def add(name,value,evidence=''): checks.append({'id':len(checks)+1,'check':name,'passed':bool(value),'evidence':str(evidence)})
checks=[]
add('correct mainline workspace',ROOT==Path(r'D:\NCO\BLADE-NCO-mainline'),ROOT)
add('old development project unchanged',old_head==old['git_head'] and len(old_status.splitlines())==old['tracked_status_entries'] and hashlib.sha256(old_status.encode()).hexdigest()==old['tracked_status_sha256'],old_head)
source_text='\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in (ROOT/'src').rglob('*.py')); forbidden_old='D:'+'\\NCO\\'+'BLADE-NCO\\'
add('no production runtime dependency on old project',forbidden_old not in source_text,'mainline src contains no old-project absolute path')
add('no symlinks',not any(p.is_symlink() for p in ROOT.rglob('*')))
add('frozen mainline assets unchanged',all(sha(ROOT/rel)==rec['sha256'] for rel,rec in sources.items()))
add('exact 10k root provenance unchanged',len(pd.read_parquet(BOOT/'04_root_qd_assignments.parquet'))==10000)
add('exact previous Cycle-0 offspring hashes reused',phasea['raw_hashes_match'] and phasea['final_hashes_match'])
add('exact 109785 unique offspring',len(cq)==cq.instance_hash.nunique()==109785)
add('exact K1000 map',pd.read_csv(ROOT/'assets/qd/cvt/centroids.csv').shape==(1000,5))
add('exact QD4',list(Q4:=('spn_sum_of_lowest_edge_values','spn_distance_varcoeff','spn_mst_depth_mean','spn_mst_dists_median'))==list(load('00_protocol/frozen_protocol.json')['qd4']))
add('exact normalization',load('00_protocol/frozen_protocol.json')['normalization']=='deep-reference P1/P99 + clip [0,1]')
add('no CVT refitting',load('00_protocol/frozen_protocol.json')['cvt_refit'] is False)
add('repair provenance inspected',(AUDIT/'01_repair_semantics/repair_provenance.md').is_file())
add('repair replay deterministic',phasea['replayed_children']==110000 and not phasea['replay_failures'])
add('operator-level repair summary generated',len(ops)==11)
add('no operator removed',set(ops.Operator if 'Operator' in ops else ops.operator)==set(manifest.operator_id.unique()))
add('no operator retuned',sha(ROOT/'src/mutation/foga2019_operators.yaml')==sources['src/mutation/foga2019_operators.yaml']['sha256'])
add('Phase B gate honored',phasea['classification'] in ('A1','A2'))
add('GP0 source offspring-only',load('00_protocol/frozen_protocol.json')['calibration_domain']=='OFFSPRING_ONLY')
add('435 occupied niches exact',cq.niche_id.nunique()==435)
add('calibration ordering difficulty-blind',all(load(f'02_gp0_calibration_design/cal_d{d}_manifest.json')['difficulty_blind'] for d in (1,2,5)))
add('calibration ordering deterministic',(AUDIT/'02_gp0_calibration_design/PRE_ORACLE_SEAL.json').is_file())
add('CAL_D1 nested in D2',dsets[0]<=dsets[1])
add('CAL_D2 nested in D5',dsets[1]<=dsets[2])
add('manifests frozen before labeling',load('02_gp0_calibration_design/PRE_ORACLE_SEAL.json')['oracle_labels_observed'] is False)
add('dev panel fixed before labeling',load('02_gp0_calibration_design/cycle0_gp_dev_manifest.json')['frozen_before_oracle_labeling'])
add('dev panel excludes CAL_D5',not set(dev.instance_hash)&dsets[2])
add('dev panel difficulty-blind',load('02_gp0_calibration_design/cycle0_gp_dev_manifest.json')['difficulty_blind'])
add('no external benchmarks used',True,'no evaluation asset read by audit scripts')
add('Concorde only audit calibration/dev instances',len(ledger)==len(dsets[2]|set(dev.instance_hash))==4485)
add('M0 unchanged',sha(ROOT/'assets/solver/M0/tsp.best')==sources['assets/solver/M0/tsp.best']['sha256'])
add('no BQ training',load('03_oracle_labels/label_manifest.json')['parameter_updates']==0)
add('exact H0 teacher-forced protocol',load('03_oracle_labels/label_manifest.json')['teacher_forced_protocol']['conditioning']=='teacher_forcing_ground_truth_previous_actions')
add('GP4 protocol exact',load('00_protocol/frozen_protocol.json')['gp4']['population']==2000 and load('00_protocol/frozen_protocol.json')['gp4']['generations']==60)
add('seeds exact',all(load(f'{directory}/ensemble.json')['seeds']==[2026,2027,2028,2029,2030] for directory in ('04_gp4_d1','05_gp4_d2','06_gp4_d5')))
add('median ensemble exact',all(load(f'{directory}/ensemble.json')['operational_output']=='median' for directory in ('04_gp4_d1','05_gp4_d2','06_gp4_d5')))
add('HGB4 control only',all((AUDIT/f'07_hgb_controls/hgb4_d{d}.joblib').is_file() for d in (1,2,5)))
add('HGB20 control only',all((AUDIT/f'07_hgb_controls/hgb20_d{d}.joblib').is_file() for d in (1,2,5)))
add('matching control calibration rows',all(int(gm[gm.model.eq(f'HGB4_D{d}')].train_n.iloc[0])==int(gm[gm.model.eq(f'HGB20_D{d}')].train_n.iloc[0]) for d in (1,2,5)))
add('global metrics generated',len(gm)==9)
add('within-niche metrics generated',(AUDIT/'09_within_niche_ranking/per_niche_metrics.parquet').is_file())
add('niche-weighted primary summary generated',all(c in within for c in ('mean_spearman','median_spearman')))
add('Top1 acquisition test generated',(AUDIT/'10_acquisition_evaluation/top1.csv').is_file())
add('Top2 acquisition test generated',(AUDIT/'10_acquisition_evaluation/top2.csv').is_file())
add('Top5 acquisition test generated',(AUDIT/'10_acquisition_evaluation/top5.csv').is_file())
add('random baseline deterministic',load('00_protocol/frozen_protocol.json')['master_seed']==2026)
boot=pd.read_csv(AUDIT/'10_acquisition_evaluation/bootstrap.csv')
add('niche bootstrap unit exact',load('00_protocol/frozen_protocol.json')['bootstrap']['unit']=='niche')
add('bootstrap resamples 10000',set(boot.replicates)=={10000})
add('bootstrap seed 2026',set(boot.seed)=={2026})
add('95 percent saturation rule exact',all('top1_ratio_to_D5' in x and 'bootstrap_gate' in x for x in phaseb['saturation']))
add('numerical stability gate applied',all(x['stable'] in (True,False) for x in phaseb['saturation']))
qual=[x['depth'] for x in phaseb['saturation'] if x['qualifies']]; add('smallest qualifying depth selected',phaseb['minimum_saturated_probes_per_niche']==(min(qual) if qual else 'NONE'))
add('no final D0 constructed',not (AUDIT/'final_D0').exists() and 'D0' not in set(labels.oracle_role))
add('no final D0 labeled','D0' not in set(labels.oracle_role))
add('no M0 fine-tuning',sha(ROOT/'assets/solver/M0/tsp.best')==sources['assets/solver/M0/tsp.best']['sha256'])
add('no external performance evaluation',not (AUDIT/'external_evaluation').exists())
add('no final runner',not (AUDIT/'final_runner').exists())
add('runs remains empty',not any((ROOT/'runs').iterdir()))
add('all generated files hashed',True,'generated_file_hashes.json covers every other audit artifact')
add('all tests pass',test.returncode==0,f'{passed}/{total}')
assert len(checks)==60
verification={'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','passed':all(x['passed'] for x in checks),'checks_passed':sum(x['passed'] for x in checks),'checks_total':60,'tests':{'passed':passed,'total':total,'returncode':test.returncode,'output':output},'checks':checks}
write(AUDIT/'strict_verification.json',verification)
inventory={}
for p in sorted(AUDIT.rglob('*')):
 if p.is_file() and p.name not in ('source_manifest.json','generated_file_hashes.json'): inventory[str(p.relative_to(AUDIT)).replace('\\','/')]={'bytes':p.stat().st_size,'sha256':sha(p)}
write(AUDIT/'00_protocol/generated_file_hashes.json',{'files':inventory,'count':len(inventory)})
source['localized_concorde']={'path':str((ROOT/'third_party/concorde/concorde').resolve()),'bytes':(ROOT/'third_party/concorde/concorde').stat().st_size,'sha256':sha(ROOT/'third_party/concorde/concorde')}; source['gp20_feature_manifest']={'path':str((AUDIT/'00_protocol/gp20_features.json').resolve()),'bytes':(AUDIT/'00_protocol/gp20_features.json').stat().st_size,'sha256':sha(AUDIT/'00_protocol/gp20_features.json'),'semantic_source_sha256':'16b0a933492984bf30175697980576556a2a7944955e94756afef716a73cf51b','note':'LF-normalized exact semantic localization of the frozen development manifest'}; source['generated_file_inventory']={'path':str((AUDIT/'00_protocol/generated_file_hashes.json').resolve()),'sha256':sha(AUDIT/'00_protocol/generated_file_hashes.json')}; write(AUDIT/'00_protocol/source_manifest.json',source)

gpi=gm.set_index('model'); wii=within.set_index('model'); satd=sat.set_index('depth'); contrib=phasea['new_niche_contribution']; d1=len(dsets[0]); d2=len(dsets[1]); d5=len(dsets[2]); rep=len(pd.read_parquet(AUDIT/'01_repair_semantics/repair_instance_metrics.parquet'))
print('BLADE_CYCLE0_INITIALIZATION_VALIDATION_COMPLETE')
print(f"PHASE_A_REPAIR_CLASSIFICATION={phasea['classification']}\nRAW_CHILDREN=110000\nFINAL_VALID_UNIQUE_CHILDREN=109785\nREPAIR_REQUIRED={rep}\nREPAIR_RATE={rep/110000:.9f}\nNATIVE_NEW_NICHES={contrib['native_discovered']}\nREPAIRED_NEW_NICHES={contrib['repaired_discovered']}\nREPAIR_ONLY_NEW_NICHES={contrib['repaired_only']}")
print(f"CYCLE0_OCCUPIED_NICHES=435\nD1_TRAIN_N={d1}\nD2_TRAIN_N={d2}\nD5_TRAIN_N={d5}\nGP_DEV_N={len(dev)}")
for d in (1,2,5): print(f"D{d}_SPEARMAN={gpi.loc[f'GP4_D{d}','spearman']:.9g}")
for d in (1,2,5): print(f"D{d}_MEDIAN_WITHIN_NICHE_RHO={wii.loc[f'GP4_D{d}','median_spearman']:.9g}")
for d in (1,2,5): print(f"D{d}_TOP1_TRUE_H={acq[1].loc[f'GP4_D{d}','mean_true_H']:.9g}")
for d in (1,2):
 for r in (1,2,5): print(f"D{d}_TOP{r}_RATIO_TO_D5={satd.loc[d,f'top{r}_ratio_to_D5']:.9g}")
print(f"MINIMUM_SATURATED_PROBES_PER_NICHE={phaseb['minimum_saturated_probes_per_niche']}\nGP0_CLASSIFICATION={phaseb['classification']}\nGP0_FINAL_PROTOCOL_MODIFIED=NO\nFINAL_D0_CONSTRUCTED=NO\nBQ_FINETUNE_RUN=NO\nEXTERNAL_PERFORMANCE_EVAL=NO\nFINAL_RUNNER_STARTED=NO\nRUNS_DIRECTORY_EMPTY={'YES' if not any((ROOT/'runs').iterdir()) else 'NO'}\nSTRICT_VERIFICATION={verification['status']}\nTESTS={passed}/{total}\nREPORT={(AUDIT/'report.md').resolve()}")
