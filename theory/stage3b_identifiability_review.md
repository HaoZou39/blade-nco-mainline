# Stage 3B — Observable guarantees, selective labels, and identifiability

This review continues from the frozen Stage 3A decision
`C — TRACKING_ASSUMPTIONS_NOT_JUSTIFIED`. It does not alter any Stage 1–3A
definition, proof, counterexample, or decision. Stage 3A is frozen at commit
`0ca42f81` on branch `theory/stage3a-hardness-tracking`; Stage 3B is developed
on `theory/stage3b-identifiability` and is not merged automatically.

No new mutation, evolution, Concorde, true-LBQ evaluation, HGB training, BQ
adaptation, or HAC comparison was run. Numerical entries below are read-only
joins and sums over existing frozen artifacts.

## 1. Exact deterministic selection result

Fix cycle `k`. Define a strict total order on `C_k` by:

1. descending `hgb_predicted_log1p_hardness`;
2. ascending `instance_hash` for score ties.

For any finite set `A`, let `Top_r(A)` be the first `r` members in this order.
With `m=10,000`, `K=1,000`, and `q_k=|Q_k|`, the verified implementation is:

```text
G_k = Top_m(C_k)
T_k = Q_k ∪ Top_(m-q_k)(C_k \ Q_k)
P_(k+1) = T_k
```

Here `Q_k` is one maximum-HGB candidate per occupied niche, and `G_k` is the
diagnostic pure-global HGB Top10k frozen before exact labeling.

The exact set identity is:

```text
T_k \ G_k = Q_k \ G_k.
```

Consequently:

```text
|T_k \ G_k| = |G_k \ T_k| ≤ |Q_k| ≤ K.
```

Thus only niche elites outside pure Global Top10k consume protected slots.
Elites already in `G_k` have zero membership opportunity cost.

Define:

```text
Delta_QD(k)
  = sum_(x in G_k) hat_f_k(x) - sum_(x in T_k) hat_f_k(x).
```

Then:

```text
Delta_QD(k)
  = sum_(x in G_k \ T_k) hat_f_k(x)
    - sum_(x in Q_k \ G_k) hat_f_k(x)
  ≥ 0.
```

The value is exactly computable from frozen predictions and membership files.
This is a deterministic surrogate-score opportunity cost. It is not a
true-LBQ regret, reachability guarantee, or training-value guarantee.

Statuses:

| Axis | Status |
|---|---|
| Mathematical validity | **PROVED** |
| S6-AE applicability | **VERIFIED** |
| Scientific novelty | **STANDARD constrained top-k exchange** |

## 2. Proof and edge cases

There are `m - |G_k ∩ Q_k| ≥ m - q_k` members of `G_k \ Q_k`. Every such
member precedes every member outside `G_k`. The global fill therefore draws
all `m-q_k` entries from `G_k \ Q_k`; it cannot introduce an outside-`G_k`
candidate. Hence all members of `T_k \ G_k` come from `Q_k \ G_k`, and every
member of `Q_k \ G_k` is forced into `T_k` by definition. Equal cardinality
of `T_k` and `G_k` gives equality of the two directed-difference sizes.

The score inequality follows because pure Global Top10k maximizes the score
sum among all 10,000-member subsets. It may be an equality when boundary
scores tie. The hash tie-break makes the selected identity set unique even
when predicted scores are equal.

Required assumptions:

- `C_k` contains at least `m` unique candidates.
- The score/hash rule is a strict total order.
- `Q_k ⊆ C_k` and `|Q_k| ≤ m`.
- `T_k` is exactly `Q_k` plus the first `m-|Q_k|` candidates outside `Q_k`.

All hold on the inspected S6-AE path. No truth labels are required for this
result.

Read-only replay of the ten frozen cycles gives:

| Cycle | `|C_k|` | `|Q_k|` | `|T\G|` | `|T∩G|` | `Delta_QD` |
|---|---:|---:|---:|---:|---:|
| C00 | 119,781 | 435 | 41 | 9,959 | 0.574 |
| C01 | 119,700 | 831 | 463 | 9,537 | 4.415 |
| C02 | 119,507 | 991 | 711 | 9,289 | 9.928 |
| C03 | 119,487 | 1,000 | 800 | 9,200 | 14.322 |
| C04 | 119,531 | 1,000 | 838 | 9,162 | 22.946 |
| C05 | 119,547 | 1,000 | 848 | 9,152 | 27.599 |
| C06 | 119,667 | 1,000 | 853 | 9,147 | 41.746 |
| C07 | 119,661 | 1,000 | 873 | 9,127 | 75.619 |
| C08 | 119,700 | 1,000 | 873 | 9,127 | 70.111 |
| C09 | 119,664 | 1,000 | 876 | 9,124 | 83.503 |

`Delta_QD` is a sum on the HGB log-target scale. Its per-member version is
`Delta_QD / 10,000`, matching the stored `G_mean - T_mean` prediction metric.

Coverage reaches all 1,000 niches at C03. Thereafter `|Q_k|` is constant, but
the number of genuinely forced elites and their predicted-score cost continue
to change. Coverage saturation therefore does not saturate within-niche
selection pressure or its opportunity cost.

## 3. Observable versus unobservable quantities

Let:

```text
f_k(x) = log(1 + LBQ(M_k, x))
```

and let `V_k ⊆ C_k` contain candidates with verified true hardness under the
same entering checkpoint `M_k`. In the inspected artifacts:

```text
V_k = T_k ∪ (B_k ∩ C_k),
```

where `B_k` is the current-checkpoint archive calibration table. A label from
an earlier checkpoint is not a current label.

Four information classes are used:

1. **Deterministically observable** — identities, descriptors, niches,
   predictions, ranks, `Q_k`, `T_k`, `G_k`, overlaps, and `Delta_QD`.
2. **Verified true-hardness** — `f_k(x)` for `x ∈ V_k`, with checkpoint and
   hash alignment.
3. **Latent** — quantities depending on any unobserved `f_k(x)` in
   `C_k \ V_k`.
4. **Hypothetical** — quantities requiring a new sampling design, model,
   range restriction, future transition law, or causal training assumption.

Exactly identifiable now:

- current-hardness distribution within `T_k`;
- number and fraction of `T_k` above a predeclared threshold;
- measured current hardness of every acquired niche elite;
- residuals on `T_k` or `V_k`, explicitly conditional on that subset;
- `T_k`/`G_k` overlap and symmetric-difference size;
- exact surrogate-score cost `Delta_QD`;
- observed acquisition-level hard-instance counts.

Not identified now:

- complete-pool HGB residual distribution or uniform error;
- maximum current hardness in a niche containing unlabeled candidates;
- true regret of its selected niche elite;
- complete-pool hard-tail recall;
- complete true-hardness mean/sum comparison between `T_k` and `G_k`;
- historical-best hardness re-evaluated under the current checkpoint;
- cross-checkpoint generalization or solver drift;
- future structural reachability;
- downstream BQ training value.

The calibration sample is randomized within the accumulated archive, not
within the current candidate union. Its intersection with `C_k` is therefore
not a documented probability sample of current unselected candidates.

## 4. Selective-label indistinguishability construction

Define the observable record `O_k` to contain all deterministic quantities
and all labels on `V_k`. A target is point identified only if every completion
of labels on `C_k \ V_k` consistent with `O_k` gives the same target value.

Take an unverified candidate `z ∈ C_k \ V_k`. Construct two worlds with:

- identical candidate identities and coordinates;
- identical descriptors and niches;
- identical HGB predictions and hash order;
- identical `Q_k`, `T_k`, and `G_k`;
- identical labels for every candidate in `V_k`;
- different `f_k(z)`.

For niche maximum and elite regret, let `z` share a niche with selected elite
`e`, fix `f_k(e)=1`, and keep all other niche values at most one in world W0.
Set `f_k(z)=0` in W0 and `f_k(z)=H>1` in WH. The selected set is identical,
but elite regret changes from zero to `H-1`. Without a finite hardness range,
the difference is unbounded.

For top-r hard-tail recall, take `r` selected candidates with fixed verified
value one and `r` unverified nonselected candidates. Assign the latter zero in
W0 and `H>1` in WH. The true top-r set is entirely selected in W0 and entirely
unselected in WH, so recall changes from one to zero.

For Elite-QD versus pure Global Top10k, choose an unverified displaced member
`d ∈ (G_k \ T_k) \ V_k`. Values on `T_k \ G_k` are fixed because all actual
acquisitions are labeled. Setting `f_k(d)=0` or `H` leaves all observables
unchanged but can reverse the sign of the true `G_k`–`T_k` score difference.

This proves non-identifiability; it does **not** prove that realized S6-AE
selection is poor.

## 5. Minimal additional information required

### Known finite hardness range

Suppose a justified finite range `L ≤ f_k(x) ≤ U` holds on `C_k`. Define:

```text
A = T_k \ G_k
D = G_k \ T_k
D_V = D ∩ V_k
D_U = D \ V_k.
```

Since `A ⊆ T_k ⊆ V_k`, the true sum difference obeys the sharp interval:

```text
sum_(D_V) f + |D_U| L - sum_A f
  ≤ sum_G f - sum_T f
  ≤ sum_(D_V) f + |D_U| U - sum_A f.
```

The bounds are sharp because every unobserved displaced label may equal an
endpoint. Nonnegative LBQ supplies `L=0`, but the inspected artifacts provide
no finite scientifically justified `U`; point identification still fails.

### Exact paired verification of T versus G

No full-pool evaluation is needed. Every member of `T_k \ G_k` is already
labeled. Evaluating only the still-unlabeled members of `G_k \ T_k` under the
same `M_k` makes the true `T_k`–`G_k` sum/mean comparison exact. The exchange
identity caps this request at 1,000 labels per cycle and often fewer.

### Finite-pool aggregate estimation

For candidate-pool means, tail mass, or recall, use a probability sample from
unselected candidates with recorded positive inclusion probabilities. A
practical design would stratify by frozen niche and HGB-score bin. Horvitz–
Thompson or stratified estimators then have a design-based interpretation.
Deterministic zero-probability regions cannot be repaired by inverse weighting.

### Maxima and elite regret

Exact maxima require complete verification of the target niche, a valid local
finite-pool error certificate covering every contender, or an explicit tail
model. A small random sample does not identify an exact maximum.

### Cross-cycle and training claims

Use same-checkpoint paired evaluation for cross-cycle comparisons. A causal
training intervention is required for downstream training value. Neither can
be inferred from acquisition hardness alone.

## 6. S6-AE implementation applicability

The inspected implementation constructs:

```text
quality = one max-HGB/hash elite per occupied niche
pure_g  = first 10,000 candidates in global HGB/hash order
fill    = first 10,000-|quality| candidates outside quality
T       = quality concatenated with fill
P       = exact copy of T
```

It explicitly checks:

- one quality elite per occupied niche;
- unique `T` of size 10,000;
- exact `P=T` identity;
- every quality elite in `T`;
- equal forced-addition and displaced-global counts;
- selection frozen before Oracle labeling.

All ten frozen cycles satisfy the set theorem. Available current-checkpoint
candidate labels and missing displaced labels are:

| Cycle | `|V_k|` | Additional labeled outside `T_k` | `|G\T|` | Unlabeled `G\T` |
|---|---:|---:|---:|---:|
| C00 | 10,409 | 409 | 41 | 41 |
| C01 | 18,530 | 8,530 | 463 | 421 |
| C02 | 14,306 | 4,306 | 711 | 662 |
| C03 | 12,899 | 2,899 | 800 | 765 |
| C04 | 12,231 | 2,231 | 838 | 803 |
| C05 | 11,767 | 1,767 | 848 | 820 |
| C06 | 11,521 | 1,521 | 853 | 821 |
| C07 | 11,358 | 1,358 | 873 | 849 |
| C08 | 11,143 | 1,143 | 873 | 860 |
| C09 | 10,977 | 977 | 876 | 862 |

Every cycle therefore lacks at least one displaced true label; the exact true
`G_k`–`T_k` comparison is not identified.

The file `training_exact_hardness_cost.csv` does not change this conclusion.
Its Global row has only `|G_k ∩ T_k|` naturally labeled members—9,959 at C00
and 9,124 at C09—while its Elite-QD row has all 10,000. The recorded mean
difference is descriptive across unequal, selection-dependent subsets, not an
exact full-set selector comparison.

## 7. Literature novelty assessment

### QD selection opportunity-cost theory

The exact membership identity and nonnegative predicted-score cost are an
elementary protected-set/top-k exchange. QD and MAP-Elites already formalize
coverage by retaining high-quality representatives in behavioral cells. The
present result is useful for auditing S6-AE but is not a novel QD theorem.

### Selective-label information limits

The two-world construction is a finite, BLADE-specific presentation of a
standard selective-label and partial-feedback obstruction. Existing work
already explains why outcomes observed only after policy-dependent selection
do not identify counterfactual performance without additional structure.
Partial identification under bounded outcomes and propensity-based learning
from logged bandit feedback are likewise established topics.

### Mechanism-specific verification certificates

The useful implementation-specific observation is that an exact current-
checkpoint `T_k` versus `G_k` comparison requires only labels on the missing
symmetric difference, at most `K=1,000`, rather than all roughly 120,000
candidates. This follows directly from the elementary exchange identity. It
is a practical audit certificate, not presently a publication-level theorem.

Surrogate-assisted evolutionary optimization also has a mature literature on
managing which candidates receive expensive exact evaluation. BLADE notation
does not by itself create novelty.

Overall novelty status: **STANDARD supporting mathematics; no publication-
core theorem established**.

## 8. Do existing frozen artifacts suffice?

They suffice for:

- exact reconstruction and verification of `C_k`, `Q_k`, `T_k`, and `G_k`;
- exact selected/global overlap and exchange counts;
- exact predicted opportunity cost;
- selected-population current-hardness summaries;
- residual summaries on explicitly verified subsets;
- counting how many displaced Global candidates remain unlabeled.

They do not suffice for:

- exact true-hardness comparison of `T_k` and `G_k`;
- candidate-pool niche maxima or selected-elite true regret;
- complete-pool hard-tail recall;
- full-pool HGB error certification;
- current-checkpoint historical-best tracking;
- future reachability or downstream solver-training value.

For those targets, the status is **NOT IDENTIFIABLE FROM INSPECTED ARTIFACTS**.

## 9. Minimal validation protocol if separately authorized

The smallest protocol depends on the estimand.

For exact Elite-QD versus Global Top10k current-hardness comparison:

1. Freeze and hash `C_k`, predictions, niches, `Q_k`, `T_k`, and `G_k` before
   any additional truth evaluation.
2. Fix the entering checkpoint `M_k` and the log1p-LBQ estimand.
3. Reuse verified current labels only when checkpoint and instance hash match.
4. Evaluate every still-unlabeled member of `G_k \ T_k`; do not evaluate the
   rest of the union.
5. Compare sums and means by cancelling `G_k ∩ T_k` and pairing
   `G_k \ T_k` against `T_k \ G_k`.
6. Report the label count and confirm it is at most `K`.

For full-pool aggregate claims, additionally predeclare a probability sample
from `C_k \ (T_k ∪ G_k)`, with positive recorded inclusion probabilities and
niche/score strata. Report design-based uncertainty and nonresponse. For exact
niche maxima, verify the whole niche or a fully certified contender set.

This protocol is proposed only because current artifacts are insufficient. It
was not executed in Stage 3B.

## 10. Final GO/NO-GO recommendation

Do not pursue another positive global tracking theorem from the current
selected-only evidence. The deterministic selector audit is complete and the
remaining key true-hardness claims are not point identified. Further positive
theory should wait for a separately approved, same-checkpoint verification
design—most economically, the at-most-1,000 missing displaced Global members
for an exact `T_k`–`G_k` comparison.

The theoretical material is still valuable as a correctness and information-
limit appendix, but current frozen labels cannot support the desired full-pool
claims.

**B — IDENTIFIABILITY_LIMIT_CONFIRMED**
