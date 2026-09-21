# E4.5 Corrected Union Full Sanity

## Architecture

Five isolated cycles C00-C04 started from frozen M0. Every cycle selected hard training data by global GP Top10k from the unique parent+offspring union and selected evolutionary parents by frozen-centroid QD coverage plus global GP fill from the same union.

## Integrity and mechanism

All five cycles completed with 10,000 unique T and P instances, fixed Random500 GP calibration, unchanged QD/FOGA assets, 10k hard + 10k Uniform BQ adaptation, LR 3.75e-5, 256 epochs and fixed E256 checkpoints. No age cap, penalty, offspring quota, early stopping or benchmark-driven decision was used.

## Survival and age

- T old-parent fractions: [0.0006, 0.0913, 0.102, 0.0649, 0.1101]
- P old-parent fractions: [0.0008, 0.0937, 0.1286, 0.1119, 0.1558]
- Median T/P survival: 0.091300 / 0.111900
- Final age0/age1/age2/age3+: 0.844200 / 0.110400 / 0.022400 / 0.023000
- Final mean/max age: 0.230200 / 4

## Renewal, mutation, hardness and repair

- Median P turnover: 0.888100
- Median cross-cycle T Jaccard: 0.050787
- Median parent-beats-all / child-wins: 0.002200 / 0.994200
- Median best-child delta: 0.00351084
- Pre-adaptation mean H: [0.078179, 0.089796, 0.096822, 0.098124, 0.109553]
- Repair mean deltas: [-0.029036, -0.024941, -0.03055, -0.026613, -0.025771]

## Diversity and timing

- P N_eff: [95.794, 38.48, 70.622, 100.41, 51.498]
- P occupied niches: [435, 818, 992, 1000, 1000]
- Total five-cycle wall-clock: 21438.717s
- Mean cycle wall-clock: 4287.743s
- Mean union overhead: 4.314s (0.101%)

## Passive external evaluation

M0-M5 were evaluated on all frozen eight datasets. N=100 OOD aggregate mean-gap trajectory: [2.870867, 2.453668, 2.748085, 2.539334, 2.022308, 1.637378]. Results were not used for stopping or checkpoint selection.

## Historical comparison

`15_historical_offspring_only_diagnostic.csv` is explicitly a diverged-trajectory diagnostic and is not a controlled causal replay or success gate.

## Sanity questions

- **Q1.** Yes. Both T_s and P_s contain exactly 10,000 unique instances in every cycle.
- **Q2.** Yes. Median old-parent survival is 0.0913 in T and 0.1119 in P.
- **Q3.** Observed T survival trajectory: [0.0006, 0.0913, 0.102, 0.0649, 0.1101]; P: [0.0008, 0.0937, 0.1286, 0.1119, 0.1558].
- **Q4.** Final age0=0.8442, age3+=0.0230, max age=4; pathological lockup=False.
- **Q5.** Yes. Median fraction with at least one better child is 0.9942.
- **Q6.** Yes. P occupied niches are [435, 818, 992, 1000, 1000].
- **Q7.** BQ repair mean deltas are [-0.029036, -0.024941, -0.03055, -0.026613, -0.025771] (negative means repair).
- **Q8.** Pre-adaptation hard-set means are [0.078179, 0.089796, 0.096822, 0.098124, 0.109553]; frontier deltas are diagnostic only.
- **Q9.** Mean union overhead is 4.314s (0.101% of cycle wall-clock).
- **Q10.** No blocking implementation or integrity failure was observed.

## Verdict

**CORRECTED_UNION_FULL_SANITY_PASS**

## Next recommendation

Use this corrected union architecture as the common base for the next Random / GP-only / Full ablation.

No later ablation was launched.
