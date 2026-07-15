# Agent progress

The BGP implementation now uses symbolic multiplicative stationarization
before perturbation. Focused regression coverage passes for multiplicative
trends, expectations, IRFs, moments, derivatives, and equation rebuilds.

Higher-order moment calculations now reconstruct hidden growth-factor
variables in internal solver order before combining them with perturbation
moments. The remaining work is final validation, diff review, commit, and push.
