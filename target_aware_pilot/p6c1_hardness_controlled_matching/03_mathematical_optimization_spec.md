# Exact Optimization Specification

Variables `a_ij` cover every union candidate and 48 references. Candidate rows have capacity ≤1; reference rows equal the frozen 208/209 quotas. C1 additionally has ten equality rows for frozen E0 hardness-decile quotas. C0 omits these ten rows. The total 10k constraint is implied by reference mass.

This matrix is equivalent to source→hardness-bin→candidate→reference→sink min-cost flow and is totally unimodular. We solve its LP relaxation with HiGHS dual simplex and require zero fractional variables, 10k unique candidates, exact reconstruction of both quota families, solver optimal status, and objective equality. No greedy repair or weighted hardness term is used.
