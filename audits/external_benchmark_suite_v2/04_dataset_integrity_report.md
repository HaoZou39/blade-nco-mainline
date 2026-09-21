# Dataset integrity

All new sets were regenerated twice from the frozen derived seed and required bitwise equality. No repair, filtering, mutation, augmentation, or performance-based selection was applied.

- EvoReal_S1_100: shape=[1000, 100, 2], tensor_sha256=28c8e4c0b8f57f840e6b444e45a21c795993205faae4e27ffc9dcae1df55448f, duplicate-whole=0, duplicate-node diagnostic=0, all recorded overlaps=0.
- EvoReal_S2_100: shape=[1000, 100, 2], tensor_sha256=f393b1be940e09ff3f27c5818fb8b721dd240891d0cda22fedefc10e28c7256c, duplicate-whole=0, duplicate-node diagnostic=1, all recorded overlaps=0.
- EvoReal_S3_100: shape=[1000, 100, 2], tensor_sha256=d1f74c229e6efc609845f3570e58fa199158b5feaafc526792c2b3ee9dbab48f, duplicate-whole=0, duplicate-node diagnostic=2, all recorded overlaps=0.
- GaussianMixture500: shape=[1000, 500, 2], tensor_sha256=818dc8de29128231a725d2d9e3fdfb1f439c7d9054fe8c31124ed4fc9d1accd2, duplicate-whole=0, duplicate-node diagnostic=0, all recorded overlaps=0.

- OriginalTSPLIB70: 71 frozen source/destination hashes checked; tree manifest hash=4d6752d3aee6a91b6a65186a1058790f13b00d91de31729a5781dfb5d360cbd7.
- S2/S3 duplicate-node diagnostics are exact official-source output and were not repaired.
