# Stage 1.5 method alignment

This document maps the verified current HGB40 + Elite-QD1000 (`P=T`) S6-AE path to the corrected mathematical variables. Historical GP4 is descriptive context only and is not the current-method definition.

## Cycle-level map

| Actual S6-AE operation | Mathematical object | Verified semantics | Status / caveat |
|---|---|---|---|
| BQ checkpoint entering cycle C`k` | `q_k` | M0 enters C00; the prior cycle’s E256 checkpoint enters later cycles. | VERIFIED |
| Fixed parent bundle | ordered `P_k = (X_(k,1),…,X_(k,10000))` | C00 uses the frozen U0 bundle; later cycles load the previous `P_next.npz`. | VERIFIED |
| FOGA mutation batch | `Z_(k,i,o) ~ K~_(k,o)(X_(k,i),· | S_k)` on `X ∪ {∂}` | Deterministic nested schedule: every one of 10,000 parents receives every one of 11 operators once. Rejected/invalid outcomes are `∂` and are not resampled. | VERIFIED; this is not random parent/operator selection. |
| Operator/repair randomness | coordinate `ξ_(k,i,o)` on `Ω` | Seed is derived from cycle, parent hash, operator, and frozen master material; sampled parameters and repair follow that seed. | VERIFIED as implementation mechanism; the theorem’s repeated-run ensemble is OPEN. |
| Accepted offspring | `O_k = FirstUnique({Z_(k,i,o) ∈ X})` | Invalid/rejected outputs contribute no child; accepted outputs are deduplicated by child hash in schedule order; neither case triggers redraw. | VERIFIED |
| Candidate pool | `C_k = unique(P_k ∪ O_k)` | C00 reuses the exact corrected frozen union; later cycles concatenate parents then children and keep first hash occurrence. | VERIFIED |
| Current-hardness calibration archive sample | `B_k ⊂ A_k` with labels under `q_k` | Branch-local archive; all rows if ≤10,000, otherwise canonical-sort then seeded sample; current LBQ is refreshed under the entering solver. | VERIFIED |
| HGB40 fit and score | `ĥ_k : X → R` | Five seeded HistGradientBoostingRegressors, frozen ordered F40, target `log1p(LBQ)`, median prediction; no operator feature. | VERIFIED; predictive retention guarantees are OPEN. |
| QD niche assignment | `g(x) ∈ {1,…,1000}` | Frozen QD4 descriptors, frozen normalization, nearest CVT centroid. | VERIFIED operationally; niche-only Markov/lumpability is OPEN. |
| Per-niche quality elite | `Q_k` | Maximum current HGB prediction within every occupied niche; ascending instance hash breaks score ties. `|Q_k|` equals occupied niches, not 10,000. | VERIFIED |
| Global HGB fill | `F_k` | Descending HGB then ascending hash among candidates not already in `Q_k`, filling to 10,000. | VERIFIED |
| Acquisition set | `T_k = Q_k ∪ F_k` | Exactly 10,000 unique candidates. It is frozen before exact labeling. | VERIFIED |
| Next parent population | `P_(k+1)` | Exact membership and order copy of `T_k`; `P=T`. | VERIFIED |
| Occupied-niche set | `S(P_k) = {g(x): x ∈ P_k}` | Parent-inclusive union plus the complete `Q_k` floor and `P=T` imply `S(P_(k+1)) = S(C_k) ⊇ S(P_k)`. | PROVED and VERIFIED; does not imply hash or bridge survival. |
| True current BQ hardness | `h_k^BQ(x)` | Current LBQ under entering checkpoint `q_k`; exact labels are obtained for frozen `T_k`, not every member of `C_k`. | PARTIAL: the theorem’s threshold `κ` is OPEN. |
| Branch archive update | `A_(k+1)` | Newly labeled selected instances are merged into the branch-local archive after selection. | VERIFIED |
| BQ adaptation | `q_k → q_(k+1)` | Hard-only x1 training on `T_k`, 256 epochs, seeded per-epoch permutation and suffix sampling, then E256 checkpoint. | VERIFIED; training randomness is separate from acquisition scheduling. |

## Event map

| Quantity | Definition | What is and is not observed |
|---|---|---|
| True current hardness | `h_k^BQ(x) = h(x; θ_k, q_k)` | Mathematically defined for an instance once the metric is fixed; exactly evaluated in S6-AE only for selected/acquired instances and calibration items. |
| HGB40 prediction | `ĥ_k(x)` | Available for the candidate union and used for ranking; it is not true hardness. |
| Generation-level discovery | `D_k^gen = {∃x∈C_k: h_k^BQ(x)≥κ}` | A latent candidate-union event, covering retained parents and accepted children, unless all candidates are exactly evaluated. HGB threshold crossing is not a substitute. |
| Acquisition-level discovery | `D_k^acq = {∃x∈T_k: h_k^BQ(x)≥κ}` | Operationally verifiable after `T_k` is frozen and labeled. It additionally requires HGB/QD retention. |
| Post-adaptation hardness | `h_(k+1)^BQ(x)` | Uses the adapted checkpoint and can differ from `h_k^BQ(x)` for the same instance. |

## Probability-space map

The model uses `(Ω, F, P)` only after declaring what varies. Candidate sources of randomness are:

1. the initial-root ensemble, if U0 is treated as random rather than conditioned on its frozen file;
2. stochastic FOGA parameters and repair driven by content-derived seeds;
3. seeded archive-calibration sampling when the archive exceeds 10,000;
4. seeded HGB estimator behavior;
5. BQ training permutations and suffix sampling;
6. explicit randomness in a comparator;
7. declared numerical/runtime nondeterminism, if it is intentionally modeled.

In the completed S6-AE run, U0, source files, seed derivations, tie-breaks, and runtime identity are frozen. Conditional on all of them, the recorded execution is one trajectory. The theory must not infer sampling probabilities for parent or operator choice from that trajectory, because those choices are scheduled deterministically.

## Hazard mapping

The generic Stage 1 factorization `r γ α` remains valid only for a policy that genuinely samples a parent with probability at least `γ`. It is not the S6-AE mechanism. For S6-AE, a candidate mechanistic premise would instead be either:

- a direct history-uniform bound on `P(D_k^gen | no earlier hit)`, followed by a separate retention bound from `D_k^gen` to `D_k^acq`; or
- `p_M ≥ r α_batch`, where `r` lower-bounds availability of a useful parent and `α_batch` lower-bounds the chance that its fixed 11-operator batch generates the declared true-hard hit under a named RNG ensemble.

Neither bound is established in Stage 1.5. Equal current-hardness prediction accuracy is insufficient to establish future reachability retention.

## OPEN implementation-to-theory gaps

- scientific threshold `κ` and exact operational definition of a “discovery”;
- a repeated-run probability law suitable for inference beyond the frozen seed;
- measurable/support bounds for each full mutation-and-repair kernel;
- a generation-to-acquisition retention bound for HGB40 + Elite-QD1000;
- strong or distribution-dependent lumpability of any reduced niche state;
- compute-matched fixed-generator/HAC definitions and hazard bounds.

## Stage 2.1 independent review: information and discovery semantics

- The mutation/generation opportunity uses the **pre-generation** history as its conditioning information. Subsequent realization of the candidate batch and all selection scores makes the Elite-QD membership test deterministic (0/1).
- The exact HGB/QD rank test supplies a **pathwise retention certificate** only when it is proved on every admissible scored union. A fractional `rho_l` instead refers to a lower bound over unresolved candidate/estimator randomness conditional on the pre-generation history **and** the generated-bridge event.
- Each next bridge set is chosen before the batch intended to reach it. The terminal set may use the final cycle's **entering** checkpoint but no future checkpoint or post-generation outcome; an upfront fixed terminal set needs a stronger all-trajectories inclusion premise.
- Hash first-occurrence deduplication needs equal-hash **niche compatibility** for exact occupied-niche set union. Bridge-set propagation additionally needs equal-hash **bridge-membership compatibility**, with collision-free realized phenotypes as a sufficient condition.
- `D_k^gen` and `D_k^acq` describe true-hardness **presence** (possibly inherited), not discovery of new instances. The corrected LaTeX now separately defines `D_k^{new,gen}`, `D_k^{new,acq}`, hardness reactivation, and novel niche occupancy.
- These are mathematical modeling clarifications, **not** new empirical validations or changes to S6-AE.
