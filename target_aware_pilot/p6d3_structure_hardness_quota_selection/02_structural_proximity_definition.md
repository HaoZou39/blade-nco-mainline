# Structural Proximity Definition

`d_struct(X)=min_j FrozenTop5Cost(phi(X),phi(TSPLIB48_j))`, using the exact P6-A transformed five-dimensional representation and squared Euclidean cost. Lower is closer. NA semantics, feature order, scaling and float precision are inherited from the frozen P6-A artifacts; QD centroid distance and raw F40 are not used. This instance score is distinct from population exact OT.
