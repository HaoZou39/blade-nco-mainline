# Survivor Selection Integrity

All 12 stages pass union membership, 10,000-unique survivor, previous-P1-as-G2-parent, final-P2 identity, coordinate-hash, and provenance checks.
E0 samples uniformly without replacement from a hash-sorted union with the frozen seed. E1 uses the unmodified P6-A float32 squared-distance matrix and exact HiGHS dual-simplex TU formulation; every E1 solve is optimal, integral, and satisfies all 48 quotas.
Parent records win parent/child hash collisions. No greedy fallback occurred.

Integrity rows: 12; final populations: 6; status: **PASS**.
