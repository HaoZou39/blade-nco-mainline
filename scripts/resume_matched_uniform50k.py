#!/usr/bin/env python3
"""Independent Uniform50k runner; imports the frozen Base train/eval unchanged.

Writes only inside the baseline experiment's resume_20261022 directory.
EvoReal stays DATA_PENDING until its original training pool is identified.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import numpy as np
import pandas as pd
import torch
import run_blade_core_ablation_5c_noanchor_foga11base_v1 as canonical
from evaluation.external_v2 import coordinate_hash, coordinate_set_hash, verify_evoreal_sources
from final_run.stage import atomic_json, sha256_file, utc_now, verify_stage
from oracle.concorde import resolve_backend, solve_batch, integer_distance_matrix, tour_cost_int, tour_hash

RUN_ID = 'BLADE_BASE_UNIFORM_EVOREAL_5C_BENCHMARK_V1'
PRIOR = ROOT / 'runs' / RUN_ID
OUT = PRIOR / 'resume_20261022'
HIST = canonical.RUN
ARM = 'B0_UNIFORM50K'
GEN_SEED = 2026102200
SHUFFLE_SEED = 2026102201


def read(path):
    return json.loads(Path(path).read_text())


def freeze_json(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'frozen record mismatch: {path}')
    else:
        atomic_json(path, value)


class Status:
    def __init__(self):
        self.value = dict(run_id=RUN_ID, pid=os.getpid(), host=socket.gethostname(),
                          GPU_STATUS='READY', UNIFORM50K='RUNNING', EVOREAL50K='DATA_PENDING',
                          EVOREAL_DATA_PATH=None, EVOREAL_PROVENANCE='PARTIAL',
                          CANONICAL_M0='VERIFIED', EXISTING_BASE_ABLATION='UNCHANGED',
                          phase='initializing', completion_label_issued=False)

    def update(self, **values):
        self.value.update(values)
        self.value['timestamp_utc'] = utc_now()
        atomic_json(OUT / 'status.json', self.value)
        with (OUT / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(self.value) + '\n')


def historical_identity():
    previous = read(PRIOR / '12_integrity_verification.json')
    paths = {row['path']: row['sha256'] for row in previous['existing_checkpoints']}
    paths.update(previous['historical_inputs'])
    paths.update({str(p): sha256_file(p) for p in PRIOR.iterdir() if p.is_file()})
    for path, expected in paths.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f'historical artifact changed: {path}')
    freeze_json(OUT / 'preserved_artifacts.json', paths)
    return paths


def assert_preserved():
    for path, expected in read(OUT / 'preserved_artifacts.json').items():
        if sha256_file(path) != expected:
            raise RuntimeError(f'protected artifact changed: {path}')


def preflight():
    historical_identity()
    frozen = read(HIST / 'protocol/frozen_experiment_config.json')
    assert frozen['training'] == canonical.CONFIG['training']
    assert canonical.HARD_N == 10000 and canonical.EPOCHS == 256
    assert canonical.LOGICAL_BATCH == 256 and canonical.STEPS_PER_CYCLE == 10240
    assert canonical.LR == 3.75e-5
    backend = resolve_backend()
    assert torch.cuda.is_available(), 'CUDA required; launch outside the device-isolated sandbox'
    devices = []
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        assert float(torch.ones(8, device=f'cuda:{i}').sum()) == 8
        model, meta = canonical.load_canonical_checkpoint(canonical.M0, torch.device(f'cuda:{i}'))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        devices.append(dict(index=i, name=prop.name, bytes=prop.total_memory, M0=meta))
        del model
    probe = np.random.default_rng(0).random((256, 100, 2))
    step = canonical.production_optimizer_audit(probe)
    assert step['status'] == 'PASS'
    torch.cuda.empty_cache()
    sources = [Path(canonical.__file__), ROOT / 'src/bq/supervised.py',
               ROOT / 'src/bq/protocol.py', ROOT / 'src/bq/checkpoint.py',
               ROOT / 'src/bq/training_hardness.py', ROOT / 'src/bq/tsp_metric.py',
               Path(canonical.evaluator.__file__), ROOT / 'src/oracle/concorde.py']
    freeze_json(OUT / 'frozen_implementation.json', {str(p): sha256_file(p) for p in sources})
    freeze_json(OUT / 'training_protocol.json', frozen['training'])
    result = dict(status='PASS', timestamp_utc=utc_now(), devices=devices,
                  gpu_inventory=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.total', '--format=csv'], text=True),
                  torch_version=torch.__version__, backend=backend, production_optimizer_audit=step,
                  GPU_ROOT_CAUSE='Execution sandbox masks NVIDIA device nodes; host CUDA works without driver changes.',
                  evoreal_source_hashes=verify_evoreal_sources())
    atomic_json(OUT / f'preflight_{os.getpid()}.json', result)
    return result


def frozen_tests():
    ordered, pointsets = set(), set()
    audit = []
    for name in canonical.BENCHMARKS:
        payload = canonical.evaluator.load_fixed_references(name)
        if 'coordinates' in payload:
            arrays = payload['coordinates']
            actual = hashlib.sha256(np.ascontiguousarray(arrays, dtype='<f4').tobytes()).hexdigest()
            assert actual == canonical.evaluator.FROZEN_TENSOR_SHA256[name], name
        else:
            arrays = [row[1] for row in payload['records']]
        for xy in arrays:
            variants = [xy, np.asarray(xy, dtype=np.float32).astype(np.float64)]
            if name == 'OriginalTSPLIB70':
                variants.append(canonical.evaluator.aspect_preserving_normalize(xy))
                variants.append(np.asarray(variants[-1], dtype=np.float32).astype(np.float64))
            for variant in variants:
                ordered.add(coordinate_hash(variant))
                pointsets.add(coordinate_set_hash(variant))
        audit.append(dict(dataset=name, count=len(arrays), frozen_tensor_sha256=payload['dataset_sha256'],
                          membership_sha256=canonical.digest_values(payload['hashes'])))
    freeze_json(OUT / 'frozen_evaluation_membership.json', audit)
    return ordered, pointsets


def prepare_uniform(status):
    started = time.perf_counter()
    status.update(phase='freeze_uniform50k_and_check_eight_set_overlap')
    forbidden, forbidden_sets = frozen_tests()
    data = OUT / 'data'
    data.mkdir(exist_ok=True)
    path = data / 'uniform50k.npy'
    manifest_path = OUT / '02_uniform50k_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        assert sha256_file(path) == manifest['dataset_sha256']
        coordinates = np.load(path, allow_pickle=False)
    else:
        assert not path.exists(), 'unmanifested dataset requires inspection'
        rng = np.random.Generator(np.random.PCG64(GEN_SEED))
        rows, seen, seen_sets = [], set(), set()
        rejected = 0
        while len(rows) < 50000:
            xy = rng.uniform(0., 1., size=(100, 2))
            h, hs = coordinate_hash(xy), coordinate_set_hash(xy)
            f32 = xy.astype(np.float32).astype(np.float64)
            if h in seen or hs in seen_sets or h in forbidden or hs in forbidden_sets or coordinate_hash(f32) in forbidden or coordinate_set_hash(f32) in forbidden_sets:
                rejected += 1
                continue
            rows.append(xy)
            seen.add(h)
            seen_sets.add(hs)
        coordinates = np.stack(rows)
        order = np.random.Generator(np.random.PCG64(SHUFFLE_SEED)).permutation(50000)
        coordinates = coordinates[order]
        np.save(path, coordinates, allow_pickle=False)
        np.save(data / 'shuffle_permutation.npy', order, allow_pickle=False)
        manifest = dict(status='FROZEN', path=str(path), dataset_sha256=sha256_file(path),
                        valid_unique_count=50000, shape=list(coordinates.shape), dtype=str(coordinates.dtype),
                        distribution='iid Uniform([0,1]^2); no normalization', rng='numpy.PCG64',
                        numpy_version=np.__version__, generation_seed=GEN_SEED, shuffle_seed=SHUFFLE_SEED,
                        shuffle_count=1, rejected=rejected, train_test_overlap=0,
                        shuffle_sha256=sha256_file(data / 'shuffle_permutation.npy'),
                        generation_wall_s=time.perf_counter()-started, cycles=[])
        for c in range(5):
            hashes = [coordinate_hash(x) for x in coordinates[c*10000:(c+1)*10000]]
            membership = data / f'D{c}_membership.txt'
            membership.write_text('\n'.join(hashes)+'\n')
            manifest['cycles'].append(dict(cycle=c, count=10000, start=c*10000, end=(c+1)*10000,
                                          membership_sha256=sha256_file(membership)))
        freeze_json(manifest_path, manifest)
    assert coordinates.shape == (50000,100,2) and np.isfinite(coordinates).all()
    assert coordinates.min() >= 0 and coordinates.max() <= 1
    hashes = [coordinate_hash(x) for x in coordinates]
    sets = {coordinate_set_hash(x) for x in coordinates}
    assert len(set(hashes)) == len(sets) == 50000
    assert not (set(hashes) & forbidden) and not (sets & forbidden_sets)
    for c in range(5):
        assert canonical.digest_values(hashes[c*10000:(c+1)*10000]) == manifest['cycles'][c]['membership_sha256']
    return coordinates, hashes


def labels(coordinates, hashes, backend, status):
    started = time.perf_counter()
    directory = OUT / 'oracle'
    directory.mkdir(exist_ok=True)
    ledgers = []
    # Persist every 500 labels so interruption cannot lose a whole 50k solve.
    for begin in range(0, 50000, 500):
        path = directory / f'labels_{begin:05d}.parquet'
        cert = path.with_suffix('.json')
        status.update(phase='concorde_labels', labeled_instances=begin, total_instances=50000)
        if cert.exists():
            assert sha256_file(path) == read(cert)['sha256']
            ledger = pd.read_parquet(path)
        else:
            assert not path.exists(), 'uncertified label shard requires inspection'
            ledger = solve_batch(coordinates[begin:begin+500], hashes[begin:begin+500],
                                 directory / 'work', backend, workers=8)
            ledger.to_parquet(path, index=False)
            freeze_json(cert, dict(sha256=sha256_file(path), count=len(ledger)))
        assert ledger.instance_hash.tolist() == hashes[begin:begin+500]
        assert ledger.verified.all() and ledger.status.eq('solved').all()
        assert ledger.binary_hash.eq(backend['binary_hash']).all()
        ledgers.append(ledger)
    ledger = pd.concat(ledgers, ignore_index=True)
    tours = np.stack([np.fromstring(s, sep=' ', dtype=np.int64) for s in ledger.expert_tour])
    assert tours.shape == (50000,100)
    assert np.array_equal(np.sort(tours, axis=1), np.broadcast_to(np.arange(100), tours.shape))
    for i, tour in enumerate(tours):
        assert tour_cost_int(integer_distance_matrix(coordinates[i]), tour) == ledger.iloc[i].optimal_cost_int
        assert tour_hash(tour.tolist()) == ledger.iloc[i].tour_hash
    record = dict(status='PASS', count=50000, valid_hamiltonian_tours=50000,
                  recomputed_integer_costs_match=50000, binary_sha256=backend['binary_hash'],
                  protocol='canonical oracle.concorde.solve_batch; EXPLICIT FULL_MATRIX rint Euclidean * 1000000',
                  label_wall_s=time.perf_counter()-started, solver_sum_s=float(ledger.runtime_sec.sum()),
                  workers=8, shards={p.name: sha256_file(p) for p in directory.glob('labels_*.parquet')})
    atomic_json(OUT / '05_expert_tour_integrity.json', record)
    return tours


def reports(status):
    historical = pd.read_csv(PRIOR / '07_all_eight_gap_trajectory.csv')
    frames = [historical]
    for p in sorted((OUT / ARM / 'evaluations').glob('M*_eight.csv')):
        frame = pd.read_csv(p)
        frame['arm'] = ARM
        frame['model_index'] = frame.model.str[1:].astype(int)
        frames.append(frame)
    trajectory = pd.concat(frames, ignore_index=True)
    trajectory.to_csv(OUT / '07_all_eight_gap_trajectory.csv', index=False)
    table = pd.read_csv(PRIOR / '08_M5_six_method_comparison.csv')
    current = trajectory[(trajectory.arm == ARM) & (trajectory.model == 'M5')]
    mask = table.method.eq('BQ + Uniform50k')
    table.loc[mask, 'status'] = status.value['UNIFORM50K']
    if len(current) == 8:
        for row in current.itertuples():
            table.loc[mask, row.dataset] = row.mean_gap_percent
        table.loc[mask, 'GAP8'] = current.mean_gap_percent.mean()
        table.loc[mask, 'arm'] = ARM
    table.loc[table.method.eq('BQ + EvoReal-generated50k'), 'status'] = 'DATA_PENDING'
    table.to_csv(OUT / '08_M5_six_method_comparison.csv', index=False)
    costs = []
    budgets = []
    for p in sorted((OUT / ARM / 'cycles').glob('C*/COMPLETE.json')):
        record = read(p)
        budgets.append(dict(arm=ARM, cycle=p.parent.name, **record['training']))
        costs.append(dict(arm=ARM, cycle=p.parent.name, training_s=record['training']['training_loop_wall_s'],
                          evaluation_s=record['evaluation_s']))
    pd.concat([pd.read_csv(PRIOR / '06_training_budget_comparison.csv'), pd.DataFrame(budgets)], ignore_index=True).to_csv(OUT / '06_training_budget_comparison.csv', index=False)
    pd.concat([pd.read_csv(PRIOR / '10_data_generation_and_training_costs.csv'), pd.DataFrame(costs)], ignore_index=True).to_csv(OUT / '10_data_generation_and_training_costs.csv', index=False)
    assert_preserved()
    atomic_json(OUT / '12_integrity_verification.json', dict(status='INCOMPLETE_B1_DATA_PENDING',
                uniform_status=status.value['UNIFORM50K'], existing_artifacts_unchanged=True,
                completion_label_issued=False, canonical_M0_sha256=canonical.M0_SHA))
    conclusion = 'Q1–Q4 remain pending new M5 results.'
    if len(current) == 8:
        b0 = float(current.mean_gap_percent.mean())
        full = float(table.loc[table.method.eq('Full'), 'GAP8'].iloc[0])
        random = float(table.loc[table.method.eq('Random'), 'GAP8'].iloc[0])
        conclusion = f'M5 GAP8: Full {full:.6f}%, Random {random:.6f}%, Uniform50k {b0:.6f}%. Lower is better. Q2 remains pending EvoReal data.'
    (OUT / '11_final_baseline_report.md').write_text(
        '# Baseline resume status — combined experiment incomplete\n\n'
        + '\n'.join(f'- {key}: {status.value[key]}' for key in ('GPU_STATUS','UNIFORM50K','EVOREAL50K','CANONICAL_M0','EXISTING_BASE_ABLATION'))
        + '\n\nGPU root cause: the execution sandbox hides NVIDIA device nodes; CUDA works on the host.\n\n'
        + conclusion + '\n\nThe original partial audit files remain unchanged one directory above. '
        'The canonical train and evaluate functions are imported without edits; only output locations and experiment identity are redirected in this process. '
        'Each epoch retains the final 16-instance batch, matching the reference (39 × 256 + 16, 40 updates). '
        'M0 results are shared from the verified canonical evaluation. Stop at M5; no extra seeds.\n\n'
        'EvoReal-generated50k is a generator-data baseline, not the complete two-stage method. '
        'Original70 and EvoReal synthetic sets are historical development/related-distribution benchmarks. '
        'Generator design validation overlap remains unresolved; no independent real-world generalization claim is made.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = Status()
        try:
            pre = preflight()
            if args.preflight_only:
                print(json.dumps(pre, indent=2))
                return
            status.update(phase='preflight_passed', physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
            coordinates, hashes = prepare_uniform(status)
            tours = labels(coordinates, hashes, pre['backend'], status)
            # Redirect module globals in this independent process; no historical files are changed.
            canonical.RUN = OUT
            canonical.RUN_ID = RUN_ID
            device = canonical.base._device()
            evaldir = OUT / ARM / 'evaluations'
            evaldir.mkdir(parents=True, exist_ok=True)
            m0path = HIST / 'protocol/common_initialization/M0_external_benchmark_results.csv'
            m0 = pd.read_csv(m0path)
            assert len(m0) == 8 and set(m0.dataset) == set(canonical.BENCHMARKS)
            assert m0.model.eq('M0').all() and m0.checkpoint_sha256.eq(canonical.M0_SHA).all()
            assert m0.failed_tours.eq(0).all()
            freeze_json(OUT / 'M0_evaluation_reuse.json', dict(path=str(m0path), sha256=sha256_file(m0path), M0_sha256=canonical.M0_SHA))
            reports(status)
            for cycle in range(5):
                status.update(phase='training', cycle=cycle, labeled_instances=50000)
                frozen_tests()
                for p, expected in read(OUT / 'frozen_implementation.json').items():
                    assert sha256_file(p) == expected, p
                src = canonical.model_path(ARM, cycle)
                model, meta = canonical.load_canonical_checkpoint(src, device, expected_sha256=canonical.M0_SHA if cycle == 0 else None)
                sl = slice(cycle*10000, (cycle+1)*10000)
                model, meta, timing = canonical.train(ARM, cycle, model, meta['sha256'], coordinates[sl], hashes[sl], tours[sl], device, status, True)
                ckpt = canonical.model_path(ARM, cycle+1)
                frame, elapsed = canonical.evaluate(model, ckpt, f'M{cycle+1}', evaldir / f'M{cycle+1}_eight.csv', status, ARM)
                assert len(frame) == 8 and frame.failed_tours.eq(0).all()
                atomic_json(OUT / ARM / 'cycles' / f'C{cycle:02d}' / 'COMPLETE.json',
                            dict(training=timing, evaluation_s=elapsed, checkpoint_sha256=meta['sha256'], timestamp_utc=utc_now()))
                del model
                torch.cuda.empty_cache()
                reports(status)
            status.update(UNIFORM50K='COMPLETE', phase='B0_M5_complete_B1_DATA_PENDING')
            reports(status)
        except BaseException as exc:
            status.update(UNIFORM50K='BLOCKED', phase='failed', error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
