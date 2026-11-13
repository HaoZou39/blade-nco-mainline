# D3-CVT-256 working map

This note records the frozen mapping used by later experiments. It is a working behavior map, not an optimal descriptor and not an optimal partition.

## Descriptor

D3 keeps three salesperson features, in this order:

1. `spn_mst_dists_varcoeff`
2. `spn_distance_median`
3. `spn_angles_mean`

Stage 1 selected D3 as the parsimonious working descriptor. D4 and D5 remain sensitivity alternatives. The choice was not reopened for this freeze. Stage 2A stayed `DECISION_C`: the same three features do not separate every geometric family better than a random partition. A niche is therefore a region of scaled D3, not a unique geometric class of TSP instances.

## Reference cloud

The centers were fit on the Stage 2A R_small cloud: 1,500 independent Uniform100 roots, each observed at mutation depths 0 through 10, giving 16,500 states. The mutations are frozen Unified-12. The cloud describes the short-chain behavior reached without a hardness or QD objective. It does not represent every TSP100 instance. The 33,000-state extended cloud was a sensitivity fit and is not this map.

## Partition

Each state is scaled by the frozen transform `asinh((x - median) / IQR)`. Medians and IQRs are the calibration values stored in this package. They are not reestimated on new instances.

The 256 centers are the `KMeans(n_clusters=256, n_init=1, random_state=20261108)` fit on those 16,500 scaled vectors. The centers follow the empirical density of the short-chain cloud. They are not a uniform centroidal Voronoi tessellation of a bounding box. K=256 is the working resolution chosen for later pilots. It is not a theoretical optimum, and this freeze does not compare other values of K.

## Assignment

A new TSP100 coordinate array is passed through the packaged R extractor, scaled with the same medians and IQRs, and assigned by Euclidean distance in scaled D3. The niche id is the index of the nearest center. If two centers are equally near, the smaller index wins. An instance outside the reference support still receives that nearest niche. The support label is only a diagnostic: normal at or below the Stage 2B-D E_SHORT depth-10 P95, warning through the P99, and severe beyond the P99.

## What this map does not say

Depth-1000 Unified-12 states did not cross the pre-registered support-failure line, but replacing the short-chain reference with a long-chain reference remained inconclusive. This package does not claim that QD increases hardness, that QD improves BQ generalization, or that D3 removes structural aliasing.
