# Agent progress

The BGP implementation now uses symbolic multiplicative stationarization
before perturbation. Focused regression coverage passes for multiplicative
trends, expectations, IRFs, moments, derivatives, and equation rebuilds.

Higher-order moment calculations now reconstruct hidden growth-factor
variables in internal solver order before combining them with perturbation
moments. The implementation was committed and pushed to the feature branch.
`tasks/bgp_math.md` contains the mathematical explanation of the
stationarization, growth restrictions, expectations, IRFs, covariances, and
differences from the former additive approach.
