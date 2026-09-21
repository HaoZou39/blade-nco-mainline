# FFT / NN-ratio calibration report

## Official implementation search

Repository: `https://github.com/HenryZhu1029/EvoReal`
Pinned commit: `919616337105e894e06463734ec1dd71f6451309`

A recursive file listing of the official EvoReal repository contains generator, trainer, and TSPLIB split files, but **no FFT energy or NN-ratio feature implementation** used for Figure 4 / Figure 5 / modularization.

Author validation coordinates **are** public:

- `EvoReal_Main/EvoReal_main/TSP/LEHD/data/tsplib_typeS1.txt` (19 instances)
- `EvoReal_Main/EvoReal_main/TSP/LEHD/data/tsplib_typeS2.txt` (12 instances)
- `EvoReal_Main/EvoReal_main/TSP/LEHD/data/tsplib_typeS3.txt` (17 instances)

These 48 instances match the paper's 48-problem TSPLIB validation split.

Status: `EXACT_FFT_SOURCE_CODE_UNAVAILABLE`
FFT statistic class: `PAPER_DEFINITION_REIMPLEMENTATION`

## Selected FFT statistic

1. Algorithm 1 normalization (shared `max_diff`, aspect ratio preserved).
2. 2-D histogram / density map: 32x32 raw counts on [[0.0, 1.0], [0.0, 1.0]].
3. 2-D FFT with `fftshift`.
4. DC bin set to 0.
5. Main frequency region: annulus `1 < r <= 4` pixels from DC.
6. Aggregation: mean of `|FFT|` over that region.

This is **not** total spectral energy and **not** the previous Parseval statistic (~16284).

## Calibration on author S1 / non-S1

- Author S1 FFT > 35: 0.789 (median=63.609)
- Author non-S1 FFT <= 35: 1.000 (median=13.681)
- Direction S1 > non-S1: True
- Threshold 35 inside observed range: True
- Author S2 NN-ratio < 0.5: 1.000 (median=0.292)
- Author S3 NN-ratio >= 0.5: 1.000 (median=0.562)
- FFT calibration accepted: True

Author NN-ratio recovers the published 0.5 split **exactly** on S2 vs S3.

## Uniform TSP-100 sanity check (Figure 5 comparator)

- n=500
- FFT median=8.866, share>35=0.000
- NN-ratio median=0.553, share>=0.5=0.840

Uniform points are expected to be non-S1 (FFT mostly <= 35) and near the NN-ratio=0.5 boundary.

## Algorithm 1 unit test

Tested 12 frozen instances. `max(ptp_x, ptp_y) == 1` and original aspect ratio preserved: True.

## Calibration verdict: `CALIBRATED_ON_AUTHOR_TSPLIB_VALIDATION`

