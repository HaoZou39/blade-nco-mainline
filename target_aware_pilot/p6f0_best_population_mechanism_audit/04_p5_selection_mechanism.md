# P5 selection mechanism

The historical rule uses exactly `spn_distance_varcoeff` then `spn_fraction_shorter_mean_distance`. TSPLIB48 aspect-normalized features define median/IQR robust coordinates; strict means nearest-reference distance within the frozen leave-one-reference radius **and** inside the target box. The selector allocates 3×3 target-cell quotas, performs 12×12 microbin round-robin, orders deterministically by rank/distance/microbin/hash, applies lineage caps with controlled doubling, and fills shortages from the frontier. No HGB score or hardness threshold enters mutation, survivor, archive retention, or final acquisition.

Requested survivor 70/30 was not achieved: Original G1 retained 2,149 target + 7,851 frontier and G2 4,813 + 5,187 because strict supply was short. The final training set is a separate quota acquisition, not a fixed 70/30 role sampler. Final strict-cell fractions are Original: 41.2%, S1: 40.3%, S2: 40.0%.

Strict/frontier true and predicted M4 CE are **missing historical evidence**. Structure, lineage, FFT and NN descriptive statistics are available in `04_strict_frontier_comparison.csv`; they cannot establish training value.
