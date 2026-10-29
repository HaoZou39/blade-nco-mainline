# Population Evolution Causal Audit

The persisted artifacts confirm the intended recurrent population path for all Original/S1/S2 replicates and both arms:

`P03(10k) → mutate current P0 (20k attempts) → unique parent+child union → select P1(10k) → mutate current P1 (20k attempts) → unique parent+child union → select P2/G2(10k)`.

- Each G1 and G2 offspring-diagnostics table has exactly 20,000 attempts in E0 and E1.
- Each selection JSON records exactly 10,000 survivors.
- Every G2 mutation attempt's parent hash belongs to that arm's persisted G1 population; G1 therefore actually becomes the G2 parent population.
- Every G2-born survivor's direct parent is in G1; every G1-born survivor's direct parent is in formal P03.
- The six G2 populations are fully persisted as ordered parquet manifests plus coordinate NPZ bundles.

The E0/E1 mutation budget is matched within every replicate. Mutation seeds/namespaces differ by replicate but are paired between arms according to the frozen P5 protocol; the selection rule is the arm-level difference.

Crucial causal clarification: survivor selection **did execute**, and E1 G1 survivors changed G2's parents. The old M5 did not train on G2 survivors because P2.2 separately retained eligible offspring in a bounded Archive30k and then acquired Structure10k from that archive. Thus the final survivor population was off the old training-data causal path; it was not absent.

Artifact-level causal/provenance status: **PASS**.

Cross-check: the prior P5 survivor audit contains 12 generation/arm rows and was used only as an existing consistency record; P6 independently re-read the saved G1/G2 artifacts.
