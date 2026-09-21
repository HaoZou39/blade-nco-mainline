# A0 interpretation cases

Observed cycle ENR_GP10 = [0.9856, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE]
Observed cycle ENR_ORACLE10 = [2.3462, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE]
median ETA10 = -0.0107
surrogate verdict = SURROGATE_HEADROOM_INCONCLUSIVE
QD verdict = QD_SURROGATE_LIMITATION_NOT_DIRECTLY_ESTABLISHED

- CASE 1 not the observed C00 pattern
- CASE 2 relevant
- CASE 3 not observed
- CASE 4 not estimable on TIER1
- CASE 5 not estimable on TIER1
- CASE 6 mixed: A3 says training value is not hardness-only, but unbiased QD capture is not estimable

## CASE 1
Oracle enrichment >> GP enrichment > Random.
Meaningful PANEL-ORACLE hardness headroom exists; GP has value but is the likely acquisition bottleneck.
A0 does not equate that bottleneck with downstream generalization headroom.

## CASE 2
Oracle enrichment >> GP enrichment ≈ Random.
The surrogate is close to random and strongly bottlenecks hardness exploitation on the independent panel.

## CASE 3
GP enrichment ≈ Oracle enrichment >> Random.
The surrogate is not the primary bottleneck; lack of downstream QD advantage must be explained elsewhere.

## CASE 4
QD descendants contain Oracle-hard cases, but GP capture is low.
Strongest evidence that the surrogate masks QD value. Requires TIER1 QD-domain exact-H.

## CASE 5
QD descendants do not contain additional hard cases.
QD may provide structural diversity without mediation through solver-relative hardness.

## CASE 6
GP captures Oracle-hard QD descendants, but A3/downstream improvement remains weak.
Bottleneck lies after acquisition; hardness is not sufficient for training value.

## Mandatory distinction
A0 evaluates surrogate -> current hardness acquisition.
It does not establish hardness -> downstream generalization.
Better Oracle-headroom capture would not necessarily improve the external gap.
