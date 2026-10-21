# Stage 2.1 — Independent Mathematical Corrections

**Base**: `d7bf79d76f66a8f706cdc9a45ac07c48accdf576` (Stage 2)
**Scope**: theory-only formal clarification; no new experiments, Oracle labels, BQ/HGB training, mutation runs, or mainline algorithm changes.
**Outcome**: the bridge-product proof survives with a corrected filtration and a causal target rule. Applicability and novelty remain OPEN.

## M1. Correct the retention information set

The actual selector is deterministic conditional on a completely scored candidate pool. Writing
`P(R_l | F_l^score)=1_(R_l)` is correct; a strictly fractional probability conditioned on that full sigma-algebra cannot be a pathwise guarantee.

For each cycle `t=k+l`, let `F_t^pre` contain the entering BQ checkpoint, parent population, archive, and other realized history, **excluding unresolved candidate-generation randomness**. Define `G_l` as presence of the next bridge in the candidate union and `R_l` as retaining that bridge in the selected population. On each admissible pre-batch history:
- `P(G_l | F_t^pre) >= a_l`.
- `P(R_l | F_t^pre, G_l) >= rho_l`, on histories with nonzero probability of `G_l`.

This yields `P(G_l ∩ R_l | F_t^pre) >= a_l rho_l` by ordered conditioning, without independence. The second probability averages over still-unresolved candidate-batch outcomes (and any other named randomness); it is *not* `P(R_l | F_t^score,G_l)`.

A **different, stronger** condition is a pathwise deterministic certificate: every admissible scored candidate union containing a bridge retains a bridge by the exact per-niche elite/global-fill rule. This gives `rho_l=1`; it is not established for actual BLADE.

## M2. Remove look-ahead for moving solver targets

Each next bridge set `B_(l+1)` must be selected measurably **before** the cycle-`k+l` batch intended to reach it. After being selected, its identity stays unchanged through selection into the next parent population.

The last set `B_L` may use the entering checkpoint `q_(k+L-1)`, which is already available before the final batch, but no later checkpoint or the realized final offspring. Require, on admissible histories,
`B_L ⊆ {x: h_(k+L-1)^BQ(x) >= κ}` almost surely.

This is a **causal adapted-target model**; it does not assume a fixed future model trajectory. If all bridge sets are instead frozen at the initial cycle, terminal inclusion must hold uniformly over the allowed future solver trajectories. Neither model supplies positive operator reachability constants by itself.

## M3. State hash compatibility exactly

First-occurrence deduplication is by instance hash. For the exact occupied-niche union equality it suffices that equal-hash instances share the same niche label in the realized pre-deduplication pool.

For the operator-specific bridge theorem, this weaker condition is **not sufficient**: two same-hash, same-niche candidates may have different membership in the bridge set. The accepted result remains represented after dedup only if equal hashes preserve bridge-set membership. Injectivity of the hash on the realized distinct phenotypes is a sufficient (stronger) mathematical condition. Collision resistance alone is not a proof of injectivity.

## M4. Separate true-hardness presence from novelty

The original `D_k^gen` and `D_k^acq` mean the true-hard region is represented in the candidate union or selected population. These events may occur with no new valid offspring. The updated `definitions.tex` distinguishes:
- **New valid hard offspring:** a valid, newly generated phenotype not in previous candidate pools that exceeds current true BQ hardness threshold;
- **New acquisition:** a true-hard phenotype absent from previous acquisition pools that is selected now;
- **Fresh hard offspring acquired now:** intersection of these sets with current selected offspring;
- **Hardness reactivation:** an old phenotype crosses the threshold under a changed BQ checkpoint, without instance novelty;
- **New QD niche occupancy:** novelty in the fixed structural partition, which can saturate while new within-niche phenotypes continue appearing.

The conditional bridge-chain theorem remains a **presence** guarantee. It must not be cited as a new-hard-instance discovery guarantee without adding the appropriate exclusion/certification assumptions.

## Revised theorem, proof and status

With causal `B_0,...,B_L`, an initial retained bridge, a conditional generation bound `a_l`, and pre-generation-conditioned retention bound `rho_l`, the Stage 2 bound remains:
```text
P(final generation-level bridge hit) >= (Π_{l=0}^{L-1} a_l) (Π_{l=0}^{L-2} rho_l)
P(final acquisition-level bridge hit) >= Π_{l=0}^{L-1} (a_l rho_l).
```
The revised proof applies the tower property to the event `H_l ∩ G_l ∩ R_l`, where `H_l` is known before that cycle's batch. No independence is required.

**Mathematical validity:** PROVED under the corrected, explicit conditions.
**S6-AE applicability:** PARTIAL/OPEN. Deterministic scheduling, rejection behavior, per-niche elite/global fill, and `P=T` are verified. Positive operator transition constants and retention certificates have not been established.
**Scientific novelty:** NOT CLAIMED for the standard conditional product identity, and OPEN for a full BLADE-specific tracking/separation theorem.
**Comparison with HAC:** UNRESOLVED. No equal-compute empirical or theoretical superiority is inferred here.

## Independent review checks / limitations

- Hash-niche compatibility preserves the occupied-niche invariant; a stronger bridge-compatibility condition is now called out.
- The progressive bridge-loss counterexample remains valid, because current-hardness score elite selection can eliminate useful stepping stones even under perfect prediction.
- An entire frozen seeded run does not estimate nonzero conditional probabilities; a scientific ensemble and corresponding conditioning must be declared before numerical bounds are asserted.
- Cost comparison requires explicit accounting and cannot equate FOGA attempts to HAC gradient steps.
- The analysis is **source-level review**. The updated `main.tex` has **not** been recompiled as part of this correction branch, because the isolated execution environment could not obtain repository files through GitHub network access. The existing committed `theory/main.pdf` is the Stage 2 PDF, **not** the updated Stage 2.1 typeset output. Compile the new LaTeX in Prism or a LaTeX-enabled checkout before merge.

## Next decision (do not perform automatically)

After the LaTeX compilation and independent review pass, either:
1. pursue solver-relative per-niche hardness-tracking regret with prediction error and solver drift explicitly modeled; or
2. pursue a non-vacuous bridge-retention certificate with actual operator kernels and alignment assumptions.

Do not start Stage 3, run solver experiments, or claim a BLADE-versus-HAC advantage on the strength of this Stage 2.1 repair.
