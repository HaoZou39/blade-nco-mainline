# Population FFT–NN Comparison

Generator provenance is available for all 50k EvoReal instances and is kept separate from approximate feature classification. B/C seeds are never pooled for inferential claims. Marginal statistics include quantiles rather than only means.

| dataset | feature | n | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|---|---|
| TSPLIB48 | fft_energy | 48 | 9.133478 | 11.781026 | 18.580921 | 42.992855 | 108.954930 |
| TSPLIB48 | nn_ratio | 48 | 0.247204 | 0.413479 | 0.542585 | 0.695020 | 1.307712 |
| B_Original | fft_energy | 10000 | 9.663548 | 13.266427 | 19.964879 | 25.922979 | 30.522947 |
| B_Original | nn_ratio | 10000 | 0.532194 | 0.660114 | 0.900470 | 1.065285 | 1.199624 |
| C_Original | fft_energy | 10000 | 9.668908 | 13.261935 | 20.429623 | 26.786053 | 31.588803 |
| C_Original | nn_ratio | 10000 | 0.531202 | 0.666880 | 0.905859 | 1.067667 | 1.204876 |
| B_S1 | fft_energy | 10000 | 9.811238 | 13.315539 | 19.968242 | 25.807666 | 30.325645 |
| B_S1 | nn_ratio | 10000 | 0.533967 | 0.667534 | 0.896431 | 1.066333 | 1.199128 |
| C_S1 | fft_energy | 10000 | 9.782658 | 13.260799 | 20.453186 | 26.914753 | 31.475771 |
| C_S1 | nn_ratio | 10000 | 0.531425 | 0.671178 | 0.902850 | 1.071772 | 1.205300 |
| B_S2 | fft_energy | 10000 | 9.805226 | 13.367060 | 19.963888 | 26.016976 | 30.371211 |
| B_S2 | nn_ratio | 10000 | 0.530553 | 0.665615 | 0.904535 | 1.065179 | 1.201282 |
| C_S2 | fft_energy | 10000 | 9.798549 | 13.404986 | 20.296526 | 26.837389 | 31.396051 |
| C_S2 | nn_ratio | 10000 | 0.526467 | 0.666801 | 0.909670 | 1.068005 | 1.203826 |
| EvoReal50k | fft_energy | 50000 | 7.011478 | 9.913060 | 14.610017 | 29.208117 | 42.918656 |
| EvoReal50k | nn_ratio | 50000 | 0.224614 | 0.460135 | 0.632889 | 0.842493 | 1.634418 |
