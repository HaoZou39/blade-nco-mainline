
BLADE PARALLEL MECHANISM ANALYSIS COMPLETE

SNAPSHOT:
    cycles analyzed = C00-C04
    latest solver in snapshot = M5

A1 — QD DELAYED VALUE:
    QD-rescued parents = 3922
    +1 cycle T-entry = 0.1222
    +2 cycle T-entry = 0.0747
    +3 cycle T-entry = 0.0371
    new-niche descendant rate = 0.1475
    verdict = QD_DELAYED_LINEAGE_VALUE_OBSERVED

A2 — MUTATION:
    strongest T-enriched operator(s) = cluster_mutation, implosion_mutation, compression_mutation
    strongest niche-expansion operator(s) = cluster_mutation, axis_projection_mutation, expansion_mutation
    operator roles differentiated = mixed
    verdict = OPERATOR_CONTRIBUTION_MIXED

A3 — TRAINING VALUE:
    hardness vs next OOD5 association = -0.2000
    N_eff vs next OOD5 association = -0.9000
    turnover vs next OOD5 association = -0.1000
    key counterexample cycles = ['C00', 'C04']
    verdict = TRAINING_VALUE_NOT_EXPLAINED_BY_HARDNESS_ALONE

A4 — UNION SURVIVOR VALUE:
    productive survivor fraction = 0.8840
    inert survivor fraction = 0.3721
    age>=2 future T-entry = 0.0287
    verdict = SURVIVOR_MEMORY_PRODUCTIVE

SYNTHESIS:
    strongest supported mechanism = mutation generates structured variation; training value is not hardness-only by default
    weakest / unresolved mechanism = causal QD vs GP-only and Random ablations remain unrun

RECOMMENDED NEXT TRAINING ABLATION:
    RANDOM / MUTATION-ONLY BASELINE

E4.6 M15 available at analysis completion:
    NO

FINAL STATUS:

    BLADE_PARALLEL_MECHANISM_ANALYSIS_COMPLETE
