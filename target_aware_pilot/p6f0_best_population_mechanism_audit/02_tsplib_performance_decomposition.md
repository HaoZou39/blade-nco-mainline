# TSPLIB performance decomposition

P5-E1 gap: Original 1.9180%, S1 2.1038%, S2 2.4142%; mean **2.1453%**. B mean is **2.2022%**. P5-B mean difference is **-0.0569 pp** (negative favors P5), seed wins **2/3**, hierarchical bootstrap 95% CI **[-0.3290, +0.2661] pp**.

Original P5-E1 improves TSPLIB48 by **+0.1455 pp** and TSPLIB22 by **+0.6342 pp**. S1 also improves both (+0.1354, +0.0648); S2 regresses TSPLIB48 (-0.3722) while still improving TSPLIB22 (+0.1848). Thus the good seeds are not development48-only, while S2's overall failure is driven by its development48 regression rather than held-out22.

For Original, the top-5 improvement contribution ratio is **1.105** and the mean gap after removing the five largest gains is **1.8143%**. There are **15/70** identities improved in all three seeds and **11/70** improved in Original+S1 but regressed in S2. Full tails, instance IDs and node bins are in the CSV tables; no benchmark case was removed from the official endpoint.
