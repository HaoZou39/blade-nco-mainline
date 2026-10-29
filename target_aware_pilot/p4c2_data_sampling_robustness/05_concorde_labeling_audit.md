# Concorde Labeling Audit

Canonical P4-B2 integer-distance Concorde convention was reused. BQ hardness was not computed.

| Dataset | Verified | Cache hits | New calls | Failures | Wall (s) | Label bundle SHA256 |
|---|---:|---:|---:|---:|---:|---|
| S1-R0 | 10,000 | 6,638 | 3,362 | 0 | 140.1 | `b391352fb4f3b1deb49529b64a6c0723ffdb197cd3a8d55e35120d8c88164452` |
| S1-R2 | 10,000 | 6,534 | 3,466 | 0 | 146.5 | `17469c56e8ef2b55997fd4360c9aae0193e48eb3a551fe1298ece5492866a2ea` |
| S2-R0 | 10,000 | 8,275 | 1,725 | 0 | 74.7 | `1cf94625d252e868122322e8b973e333eccd41b18cf1ff8e752c5f713e0d0e9e` |
| S2-R2 | 10,000 | 7,741 | 2,259 | 0 | 97.4 | `053138392954d047e322a1e95ff70b28eecfaf268d5c285d5dc61ab00768cca1` |

- Total logical labels: 40,000; physical new calls: 10,812; cache reuses: 29,188.
- All tours are 0-based Hamiltonian permutations; integer objectives and implicit closure were independently recomputed.

Status: **PASS**.
