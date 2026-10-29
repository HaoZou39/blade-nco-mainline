# E0 versus E1 Structure Comparison

| Stage | Arm | Strict | SWD | 3×3 JS | Effective cells | Largest cell | Coverage | Effective lineages | Max lineage | FFT median | NN-CV median | Auxiliary JS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded_archive | E0 | 10.05% | 0.536298 | 0.445236 | 4.460 | 40.91% | 45/48 | 646.3 | 0.82% | 25.0002 | 1.0155 | 0.519709 |
| bounded_archive | E1 | 13.73% | 0.407849 | 0.335486 | 5.552 | 27.34% | 45/48 | 645.4 | 0.84% | 23.7743 | 1.0034 | 0.520802 |
| provisional_structure10k | E0 | 30.16% | 0.366734 | 0.210295 | 6.307 | 22.96% | 45/48 | 563.7 | 1.20% | 23.5447 | 1.0016 | 0.493910 |
| provisional_structure10k | E1 | 41.20% | 0.239921 | 0.069735 | 5.828 | 26.83% | 45/48 | 517.3 | 1.28% | 21.0648 | 0.9616 | 0.516381 |


- Candidate-space effect: before final acquisition, E1's bounded archive is closer in the frozen two-feature geometry (SWD/JS 0.407849/0.335486) than E0 (0.536298/0.445236), and contains +1,104 more strict candidates.
- Final-acquisition effect: applying exactly the same selector improves both arms, but E1 remains closer (SWD 0.239921 vs 0.366734; JS 0.069735 vs 0.210295) and supplies +1,104 more strict instances.
- Both cover 45/48 reference neighborhoods. E0 has higher effective lineage diversity (563.7 vs 517.3) and slightly better auxiliary FFT/NN-CV JS (0.493910 vs 0.516381); E1 is therefore not declared universally more diverse from 3×3 JS alone.
- E0 adjusted quotas `[1451, 136, 0, 1671, 1543, 50, 2296, 1670, 1183]`; E1 adjusted quotas `[2683, 289, 0, 599, 2475, 92, 1224, 599, 2039]`. Both record explicit capacity redistribution and zero final shortfall.
- E0/E1 final membership overlap is 3,406/10,000. No post-gap reselection is permitted.
