# Stage 3A — Solver-relative hardness tracking review

**Scope:** theory-only extension of the merged Stage 2.1 baseline. No mutation,
evolution, Concorde, true-LBQ evaluation, HGB retraining, BQ adaptation, or
algorithm change was performed.

## 1. Research question and motivation

The question is whether the verified HGB40 + Elite-QD1000 (`P=T`) mechanism
can retain a high-current-hardness representative in each fixed QD niche while
the entering BQ solver changes from `M_k` to `M_(k+1)`.

This is narrower than future structural reachability and narrower than
downstream training value. A candidate can be the hardest currently available
member of its niche yet be a poor mutation stepping stone; training on a hard
candidate can also fail to improve an external benchmark. Stage 3A does not
identify these quantities.

## 2. Precise definitions

For the solver `M_k` entering cycle `k`, define the true current target on the
HGB-compatible scale by

`f_k(x) = log(1 + L_BQ(M_k,x))`.

Let `fhat_k(x)` be the median HGB40 prediction. For every occupied candidate
niche `j`, let

`E_(k,j) = C_k ∩ N_j`

and let `e_(k,j)` be the maximum-`fhat_k` member of `E_(k,j)`, with ascending
instance hash breaking prediction ties. The verified selector gives

`e_(k,j) ∈ Q_k ⊆ T_k = P_(k+1)`.

For the latent finite historical candidate ledger

`U_k = union_(t=0)^k C_t`,

define, whenever niche `j` has been discovered,

`V_(k,j) = max_(x in U_k ∩ N_j) f_k(x)`

and

`R_(k,j) = V_(k,j) - f_k(e_(k,j))`.

`V_(k,j)` is a hindsight mathematical benchmark. S6-AE does not evaluate
current true LBQ for every member of `C_k` or relabel every discarded member
of `U_k` under every later checkpoint.

## 3. One-step per-niche quality result

Assume a finite-pool prediction error bound

`|fhat_k(x) - f_k(x)| <= epsilon_(k,j)` for every `x in E_(k,j)`.

Then

`f_k(e_(k,j)) >= max_(x in E_(k,j)) f_k(x) - 2 epsilon_(k,j)`.

Because `P_k ⊆ C_k`, if the parent niche is nonempty,

`f_k(e_(k,j)) >= max_(x in P_k ∩ N_j) f_k(x) - 2 epsilon_(k,j)`.

### Proof

Choose a true maximizer `x*` in the finite nonempty candidate niche. Prediction
optimality and the two error inequalities give

`f_k(e) >= fhat_k(e)-epsilon >= fhat_k(x*)-epsilon >= f_k(x*)-2epsilon`.

The hash tie-break only makes the prediction maximizer single-valued; every
tied prediction maximizer satisfies the same inequality.

### Edge conditions

- Empty candidate niches have neither an elite nor a maximum.
- The parent-relative result requires `P_k ∩ N_j` to be nonempty.
- Hash deduplication must preserve the phenotype, niche, and score identity
  used in the statement on the realized pool.
- The additive bound is on `log(1+LBQ)`. On the raw scale it becomes a
  multiplicative-type bound, not the same additive `2 epsilon` statement.

### Status

- **Mathematical validity:** PROVED under the stated error bound.
- **S6-AE applicability:** PARTIAL. Selection and scale match, but complete-pool
  true LBQ is unavailable, so the uniform error premise is not verified.
- **Scientific novelty:** STANDARD approximate-argmax inequality.

## 4. Solver-drift tracking result

Assume additionally that

`|f_(k+1)(x) - f_k(x)| <= delta_(k,j)`

on `E_(k,j)`. Then

`f_(k+1)(e_(k,j)) >= max_(x in E_(k,j)) f_(k+1)(x)
                         - 2 epsilon_(k,j) - 2 delta_(k,j)`.

The requested parent-relative consequence is

`f_(k+1)(e_(k,j)) >= max_(x in P_k ∩ N_j) f_(k+1)(x)
                         - 2 epsilon_(k,j) - 2 delta_(k,j)`.

Only the smaller set `(P_k ∩ N_j) union {e_(k,j)}` needs the drift bound for
this parent-relative form.

### Time alignment

- `e_(k,j)` is selected with `M_k` and `fhat_k`.
- It is retained in `P_(k+1)` because it belongs to the QD floor and `P=T`.
- BQ adaptation then creates `M_(k+1)`.
- The inequality under `M_(k+1)` is an ex post consequence of the drift
  assumption. Cycle `k` does not use future scores or labels.

The result compares only with instances already available in cycle `k`. It
does not cover new `C_(k+1)` offspring, guarantee the same hash survives the
next selection, or say anything about mutation reachability or training value.

### Status

- **Mathematical validity:** PROVED under local prediction and drift bounds.
- **S6-AE applicability:** PARTIAL/OPEN. Timing is verified; the numerical
  error and drift certificates are not.
- **Scientific novelty:** STANDARD stability consequence of approximate
  maximization.

## 5. Multicycle tracking regret

Define the historical-unavailability penalty

`A_(k,j) = [max_(x in U_(k-1) ∩ N_j) f_k(x)
             - max_(x in C_k ∩ N_j) f_k(x)]_+`,

with value zero when the old historical set is empty. Directly,

`R_(k,j) <= 2 epsilon_(k,j) + A_(k,j)`.

This is the cleanest non-accumulating decomposition: prediction error controls
selection among currently available candidates, while `A_(k,j)` measures the
price of discarded historical candidates becoming valuable again.

For a recurrence, suppose niche `j` existed at cycle `k-1`, the previous elite
is available through `e_(k-1,j) in P_k subset C_k`, prediction error is bounded
on the current candidate niche, and solver drift is bounded on **every**
historical candidate in `U_(k-1) ∩ N_j`. Then

`R_(k,j) <= R_(k-1,j) + 2 epsilon_(k,j) + 2 delta_(k-1,j)`.

If `tau_j` is the first discovery cycle, then

`R_(tau_j,j) <= 2 epsilon_(tau_j,j)`

and

`R_(k,j) <= 2 sum_(s=tau_j)^k epsilon_(s,j)
             + 2 sum_(s=tau_j)^(k-1) delta_(s,j)`.

### Why the recurrence is valid

The previous elite is still in the next candidate union. Historical drift
bounds the current hardness of the old hindsight maximizer relative to its
old hardness and also bounds the previous elite in the opposite direction.
This costs `2 delta`. Current approximate maximization costs `2 epsilon`.
Newly generated candidates already belong to `C_k` and therefore add no
separate historical-unavailability term.

### Why it is weak

- Drift must be bounded over the entire historical niche ledger, including
  discarded and unlabelled candidates.
- Constant per-cycle `epsilon` and `delta` give linear cumulative regret.
- No bounded steady-state result follows without summable variation, a
  uniform bound on `A_(k,j)`, or reintroduction of historical candidates.
- S6-AE retains the previous elite, not the complete `U_k` archive.

The linear accumulation is real. In one niche with capacity one and zero
solver drift, introduce candidates with true scores `f(x_t)=-2t epsilon`.
Give `x_(t-1)` and `x_t` the same predicted score using errors of magnitude
`epsilon`, and let the hash rule choose `x_t`. Historical regret is exactly
`2t epsilon`.

### Status

- **Mathematical validity:** PROVED under full historical-set drift.
- **S6-AE applicability:** OPEN.
- **Scientific novelty:** STANDARD telescoping/variation-style bound.

## 6. Counterexamples and failure boundaries

### A. Perfect HGB prediction, severe solver drift

In one niche let `f_k(a)=1`, `f_k(b)=0`, with perfect prediction, so `a` is
selected. After adaptation let `f_(k+1)(a)=0` and `f_(k+1)(b)=H`. The
post-update parent-relative gap is the arbitrary value `H`. Current prediction
accuracy cannot replace a drift bound.

### B. Small solver drift, inaccurate HGB ranking

Set drift to zero, `f(a)=H`, `f(b)=0`, but reverse their predicted order. The
selected gap is arbitrary `H`. Drift control cannot replace candidate-set
prediction/ranking control.

### C. Full niche coverage, deteriorating reachability

The existing progressive bridge-loss construction keeps all niches occupied
while higher-current-hardness decoys replace every future-useful bridge.
Perfect current prediction does not prevent this failure.

### D. Low tracking regret, no downstream BQ improvement

Let every selected elite be the exact current maximum, so tracking regret is
zero. An adaptation map may nevertheless leave the solver unchanged or worsen
loss on a separate evaluation instance. Hardness tracking contains no theorem
about gradient alignment, interference, or generalization.

### E. Reset/stationary generation performs better

Let the inherited population contain only a state whose mutations cannot
reach the current hard target. A reset distribution that samples the target
with probability one wins in one equal-cost opportunity despite preserving no
historical elite.

## 7. Implementation applicability

Verified in S6-AE:

- `fhat_k` is a five-seed median HGB40 prediction trained on `log1p(LBQ)`.
- Every occupied candidate niche contributes its maximum-current-HGB member.
- Prediction ties use ascending instance hash.
- Parent–Child Union makes every previous parent available before selection.
- The complete QD floor is contained in `T_k=P_(k+1)`.
- `T_k` is frozen and exactly labelled before BQ adaptation creates `M_(k+1)`.

Not verified:

- a uniform HGB error bound on every member of each candidate niche;
- a drift bound even on the selected population, much less the entire latent
  historical candidate ledger;
- operational storage/re-evaluation of every `U_k` member;
- any implication from current hardness to future reachability;
- any implication from low tracking regret to BQ training benefit.

Actual S6-AE obtains exact current LBQ for selected instances and calibration
items, not the complete candidate union. Existing finite traces cannot certify
the universal or history-uniform premises required by the theorems.

## 8. Novelty assessment

The `2 epsilon` result is the elementary approximate-argmax inequality. The
drift extension and cumulative variation terms are standard in spirit relative
to moving-optimum and dynamic-regret analysis. Elite preservation in fixed
feature cells is standard QD machinery.

The BLADE-specific fact used by the recurrence is only that Parent–Child Union
and `P=T` reintroduce the immediately previous niche elite into the next
candidate union. This yields a clean one-step memory term but not a novel
long-run result.

A potentially publishable contribution would require at least one of:

1. a non-vacuous, implementation-supported HGB error certificate on evolving
   full candidate pools;
2. a solver-update stability bound tied to the actual BQ adaptation map;
3. a structural condition connecting current-hardness elites to FOGA-11
   future reachability; or
4. a theorem connecting selected hardness/diversity to downstream BQ training
   value under explicit optimization assumptions.

None is established here. No BLADE-over-HAC claim follows.

## 9. Remaining mathematical assumptions

- Measurable, hash-compatible candidate/niche identities on every compared
  finite pool.
- Complete-pool per-niche prediction error bounds on the correct log scale.
- Local cross-checkpoint drift bounds on the exact comparison set.
- Full historical-set drift for the cumulative recurrence.
- A bound on historical unavailability, or summable error/drift, for any
  non-accumulating long-run statement.
- Separate reachability and training-value alignment assumptions.
- A declared probability ensemble if the bounds are to become probabilistic
  rather than pathwise conditional statements.

## 10. Recommendation and stopping decision

Do not proceed automatically to Stage 3B. The next theory stage would be
justified only after deciding which missing certificate can be supported by
the implementation without new unapproved experiments. The most direct
candidate is a finite-pool error/drift audit using already existing labels,
but selected-only labels cannot establish the required complete-pool bound.

The mathematics is internally consistent and the requested recurrence is
valid under explicit assumptions. However, the assumptions needed to make it
an implementation-relevant long-run tracking theorem are not currently
supported, and the resulting inequalities are standard.

**C — TRACKING_ASSUMPTIONS_NOT_JUSTIFIED**
