# P03 Population Integrity

- Source: `runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.npz` and `runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/P_next.parquet`.
- Actual population: **10,000** unique N=100 instances; 1,163 lineages.
- Ordered membership SHA256: `558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190`.
- Bundle SHA256: `8ad76b90c08997bba33da1be6bf3dbfa8c28ee05bbb4a5fe4d1a9ecdb699dd62`; metadata SHA256: `f6f6a342940fcafd5bf00fd5c9c2467a88e864396f7b59d1f5d9ea07b3086297`.
- All coordinate hashes reproduce IDs; finite/in-bounds/nondegenerate fractions are 1.000/1.000/1.000.
- The ordered IDs exactly equal formal `C04/04_qd/parent_qd.parquet`, and the membership SHA exactly equals C04 FOGA runtime's recorded parent SHA. P03 is therefore the formal population after C03 selection and before C04 mutation.
- P03 is not reconstructed from C04 Hard10k and not taken from the P2.0/P2.1 expanded pool.

## M4 training history

C00-C03 `T_selected` are treated as the actual M4 training history. Their union is 36,328 unique instances. C04 Hard10k is deliberately excluded from that history because it trained M5, not M4.

| Cycle | Rows | Unique | New vs previous | Membership SHA256 | Path |
|---|---:|---:|---:|---|---|
| C00 | 10,000 | 10,000 | 10,000 | `2762e3843315bc11bce8b4bc91e9da965bb55f720ba52468a134db9809f0798f` | `/home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C00/06_selection/T_selected.parquet` |
| C01 | 10,000 | 10,000 | 8,923 | `c745d14f72a714fcf7b0b4fd77420041aa12731f551a3a7fdbf25e050cb2c7da` | `/home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C01/06_selection/T_selected.parquet` |
| C02 | 10,000 | 10,000 | 8,704 | `9e7a9ecc301ceb2bcd8f0438f1f4fda0e4d2b18aaec97a0034c639fdd7e01583` | `/home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C02/06_selection/T_selected.parquet` |
| C03 | 10,000 | 10,000 | 8,701 | `558ea278823a39275704d71e8e999efe58a56217fb23c7555cabe035c986d190` | `/home/haozou/blade-nco-mainline/runs/BLADE_CORE_ABLATION_5C_NOANCHOR_FOGA11BASE_V1/A0_FULL_BASE/cycles/C03/06_selection/T_selected.parquet` |
