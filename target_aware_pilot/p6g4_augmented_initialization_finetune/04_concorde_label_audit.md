# Concorde Labeling Audit

| status | replicate | arm | verified | cache_reuse_n | missing_before_global_solve_n | failures | membership_sha256 | labeled_bundle_sha256 | all_tours_legal | all_costs_reproduced | concorde_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PASS | Original | A | 10000 | 1444 | 8556 | 0 | e8536477ba2dc2fb91096ad5ca433ad0ff450ba00ea0cb1c4372842efb203105 | 60fec807aada42e21fe0d1b69c870e7bba3ac88a709c1f58d1e7edacbb83bcbc | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |
| PASS | Original | B | 10000 | 1375 | 8625 | 0 | baeb76e7541592d5d94de40b7869581ba2233ec985b6dbd4579034e75983d8f6 | f13fb4f1961edc9b35f7e05be71f782d1ea12e4aa022e5d25f78bd5186a930d2 | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |
| PASS | S1 | A | 10000 | 507 | 9493 | 0 | 1b797b3c49ebc2de9d7631c080a79f5dd1b0a7a4da049dcc25390e47e53848bc | 33fc8ded6fa9d80804bd7e94c2a81c2479fd7e8c3bacd8766cdf82b306981c3a | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |
| PASS | S1 | B | 10000 | 485 | 9515 | 0 | 7c65130b7048c324f311c35078a22cfc0821b08b1afeab818e0dfa41eb16bddd | 9bbb85d9f7d307de923d09cde34cb89cc2a38e5cfc622282811e4488b906d1a8 | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |
| PASS | S2 | A | 10000 | 529 | 9471 | 0 | e6ee9f13a1a3ccd851fdc0ef8af6cd3a16e10886ff2d0651445fab2fed7f684d | 8610d6e7cd9b7721a16a8f977586c3efd90b91721f8ca51ca43b1b0130ed12a1 | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |
| PASS | S2 | B | 10000 | 518 | 9482 | 0 | f76517e16f109f275e2a37d809cf3b5ab36fd82663a26311ace61493afaeb4f7 | 2cea28a486acb0e4dca8a23feaa7d503eff8cdea10f1e9ea0631263f9f76a8e1 | True | True | 38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0 |

六组共60,000 memberships；global unique=33,596；复用=2,675；实际新Concorde调用=30,921；失败=0。
标注wall-clock=1290.556s；canonical Concorde SHA256=`38a647d3f04196231c725e90d23c2c883c0a23899f9bab8a257ae6f777d1dec0`。
全部tour均为合法0-based TSP100 Hamiltonian permutation，整数目标与训练闭环约定已由正式validator复核；没有替换实例。
