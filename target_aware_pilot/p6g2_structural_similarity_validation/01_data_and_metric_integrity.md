# P6-G2 Data and Metric Integrity

This is a read-only validation. No FOGA, EvoReal generation, Concorde, BQ/HGB/GP inference or training was run. TSPLIB22 was not loaded.

## Frozen inputs

| pool | n | unique_n | membership_sha256 |
|---|---|---|---|
| P03 | 10000 | 10000 | b1e2082c621f3fcd8a90511eda87817baad6935b81eed4d8db91ad7225333590 |
| FOGA_RAW | 959453 | 959453 | 1f48d09de9d56337fb34b905da6251240d320bb0a8053d77a233e77e3b7c9907 |
| EvoReal50k | 50000 | 50000 | 16f01f952f4ba846c538e69927c7832e70b2c9054e5b14669a242cef3e50d702 |

- TSPLIB48: 48 unique development references; the predefined 75–150 control contains 17 references.
- FFT contrast: frozen `FFT_APPROX + NN-ratio`, center `[18.580921394460262, 0.542585079454774]`, IQR scale `[31.211828974880028, 0.2815404231115694]`, radius `0.177057522`.
- Formal F40: frozen order and existing P6-G1 values. Robust distance scaling was fit on an equal synthetic-only 30k sample (10k P03 + 10k FOGA union + 10k EvoReal), seed 20261029; no TSPLIB outcomes or HGB importance entered weighting. Effective dimensions: **40/40**; zero-scale dimensions: `[]`. Native missing values are pairwise-excluded without imputation, requiring at least 30 dimensions.

## Independent geometry diagnostic (fixed before results)

Coordinates are centered and divided by maximum radial norm, preserving aspect ratio and providing translation/uniform-scale invariance. Candidate orientation is searched over fixed 5° rotations and reflection. Metrics are: multiscale 4×4/8×8/16×16 occupancy JS with 0.5 pseudocount, radial-distance W1, pairwise-distance W1, self-median-normalized kNN-distance W1 for k=1/3/5, NN-CV difference, and empty-cell fraction difference. The reported composite is the unweighted mean of bounded occupancy, pairwise, radial, transformed kNN and transformed NN-CV components. It is an explanatory diagnostic, not a learned selector, fitness, or new coverage threshold.

Different node counts still affect empirical resolution and occupancy sparsity; therefore all 48 and the 75–150 subset are reported separately. Pool-size extreme-neighbor effects are separated from 30× matched 10k comparisons.
