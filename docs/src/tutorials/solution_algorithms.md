# Solution Algorithms

This page provides a detailed explanation of the perturbation solution algorithms implemented in `MacroModelling.jl` for solving DSGE models at first, second, and third order. The methods are based on the approaches described in [villemot2011solving](@citet), [andreasen2018pruning](@citet), and [levintal2017fifth](@citet).

## Overview

The package uses **perturbation methods** to approximate the solution of DSGE models around the non-stochastic steady state. The solution takes the form of policy functions that express current variables as functions of state variables (lagged endogenous variables and shocks). Higher-order approximations capture nonlinearities and provide more accurate solutions, especially when:
- The model is highly nonlinear
- Shocks are large
- Risk and uncertainty effects matter

The general form of the model is:
```math
\mathbb{E}_t[f(y_{t+1}, y_t, y_{t-1}, u_t)] = 0
```
where ``y_t`` are endogenous variables and ``u_t`` are exogenous shocks.

## First-Order Perturbation Solution

### Problem Formulation

The first-order approximation yields a linear policy function:
```math
y_t = \bar{y} + S_1 \begin{bmatrix} y_{t-1} - \bar{y} \\ u_t \end{bmatrix}
```
where ``\bar{y}`` is the steady state and ``S_1`` is the first-order solution matrix.

### Derivation

Taking the first-order Taylor expansion of the model equations around the steady state:
```math
\nabla_+ \mathbb{E}_t[y_{t+1} - \bar{y}] + \nabla_0 (y_t - \bar{y}) + \nabla_- (y_{t-1} - \bar{y}) + \nabla_e u_t = 0
```

where:
- ``\nabla_+`` is the Jacobian with respect to future variables ``y_{t+1}``
- ``\nabla_0`` is the Jacobian with respect to current variables ``y_t``
- ``\nabla_-`` is the Jacobian with respect to past variables ``y_{t-1}``
- ``\nabla_e`` is the Jacobian with respect to exogenous shocks ``u_t``

### Algorithm

The solution matrix ``S_1 = [S_1^y \quad S_1^u]`` where ``S_1^y`` relates to lagged endogenous variables and ``S_1^u`` relates to shocks.

**Step 1: Solve Quadratic Matrix Equation (QME)**

For the purely dynamic variables (those appearing with leads and lags), we solve:
```math
\tilde{A}_+ X^2 + \tilde{A}_0 X + \tilde{A}_- = 0
```

where ``X`` represents the relationship between current and past dynamic variables. The solution to this equation gives us how forward-looking variables respond to state variables.

The package implements several algorithms for solving the QME:
- **Schur decomposition** (default): Most reliable and fast
- **Doubling algorithm**: Very precise, good for ill-conditioned problems
- **Linear time iteration**: Iterative method
- **Quadratic iteration**: Alternative iterative method

**Step 2: Construct Transformation Matrices**

After preprocessing the first-order derivatives using QR decomposition to improve numerical stability:
```math
Q^\top \nabla_0 = \begin{bmatrix} \bar{A}_0^u \\ A_0 \end{bmatrix}
```

and similarly for ``A_+``, ``A_-``, where ``Q`` is orthogonal.

**Step 3: Solve for Static Variables**

Variables that only appear in the current period (static variables) are solved using:
```math
S_1^{static} = -(\bar{A}_0^u)^{-1}(A_+^u D L + \tilde{A}_0^u X + A_-^u)
```

where ``D`` and ``L`` map dynamic variables appropriately.

**Step 4: Solve for Shock Response**

The response to shocks is obtained by solving:
```math
(\nabla_0 + \nabla_+ S_1^y) S_1^u = -\nabla_e
```

This is solved using LU decomposition for efficiency.

### Matrix Construction

The final first-order solution matrix is:
```math
S_1 = \begin{bmatrix} S_1^y & S_1^u \end{bmatrix}
```

where:
- ``S_1^y \in \mathbb{R}^{n \times n_-}`` maps past endogenous variables to current variables
- ``S_1^u \in \mathbb{R}^{n \times n_e}`` maps current shocks to current variables
- ``n`` is the number of endogenous variables
- ``n_-`` is the number of state variables
- ``n_e`` is the number of shocks

## Second-Order Perturbation Solution

### Problem Formulation

The second-order approximation adds quadratic terms:
```math
y_t = \bar{y} + S_1 \begin{bmatrix} y_{t-1} - \bar{y} \\ u_t \end{bmatrix} + \frac{1}{2} S_2 \begin{bmatrix} y_{t-1} - \bar{y} \\ u_t \end{bmatrix}^{\otimes 2}
```

where ``S_2`` is the second-order solution tensor (represented as a matrix), and ``\otimes 2`` denotes the Kronecker product of a vector with itself.

### Derivation

The second-order Taylor expansion includes Hessian terms. After substituting the first-order solution and collecting second-order terms, we obtain a generalized Sylvester equation.

### Algorithm

**Step 1: Construct Auxiliary Matrices**

Define the augmented state vector:
```math
s_t = \begin{bmatrix} y_{t-1} - \bar{y} \\ \sigma \\ u_t \end{bmatrix}
```

where ``\sigma`` represents the perturbation parameter (typically set to 1).

Create matrices for the first-order solution in augmented form:
```math
S_1^{aug} = \begin{bmatrix} S_1^y & 0 & S_1^u \end{bmatrix}
```

**Step 2: Setup Sylvester Equation Components**

The second-order solution satisfies:
```math
A S_2 B + C = S_2
```

where:

- ``A = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_+ \in \mathbb{R}^{n \times n}``

- ``B = U_2^\top \mathrm{kron}(s_{t-1}^{aug}, s_{t-1}^{aug}) C_2`` is the Kronecker product of augmented states, with:
  - ``U_2`` a permutation matrix for duplication removal
  - ``C_2`` a compression matrix for symmetry

- ``C = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_2 \mathrm{kron}(s_t^{aug}, s_t^{aug})`` contains the second-order derivatives

The term ``\nabla_2`` represents the Hessian of the model equations, which is a sparse matrix due to most variables not appearing in products.

**Step 3: Solve Generalized Sylvester Equation**

The equation ``A S_2 B + C = S_2`` can be rewritten as:
```math
(I - A \otimes B^\top) \mathrm{vec}(S_2) = \mathrm{vec}(C)
```

The package implements several solution methods:
- **Iterative methods** (default for large systems)
- **Direct methods** using eigendecomposition
- **Doubling algorithm** for enhanced precision

**Step 4: Post-processing**

The raw solution is sparse due to the structure of ``U_2`` and ``C_2``. The final matrix is stored in an appropriate format (sparse or dense) based on its density.

### Matrix Construction

The second-order solution matrix is:
```math
S_2 \in \mathbb{R}^{n \times \frac{(n_- + 1 + n_e)(n_- + 1 + n_e + 1)}{2}}
```

The size accounts for:
- Symmetry in second derivatives (avoiding duplicate terms like ``x_i x_j`` and ``x_j x_i``)
- The perturbation parameter ``\sigma``
- All combinations of state variables and shocks

## Third-Order Perturbation Solution

### Problem Formulation

The third-order approximation includes cubic terms:
```math
y_t = \bar{y} + S_1 s_t + \frac{1}{2} S_2 s_t^{\otimes 2} + \frac{1}{6} S_3 s_t^{\otimes 3}
```

where ``s_t`` is the augmented state vector (as in second order) and ``S_3`` is the third-order solution tensor.

### Derivation

The third-order solution requires computing third-order derivatives (the Hessian tensor) of the model equations. The solution again takes the form of a generalized Sylvester equation, but now involving third-order terms.

### Algorithm

**Step 1: Construct Third-Order Auxiliary Matrices**

Similar to second order, but now including terms involving ``S_2``:
```math
s_2^{aug} = S_2 \mathrm{kron}(s_{t-1}^{aug}, s_{t-1}^{aug})
```

**Step 2: Setup Sylvester Equation**

The third-order solution satisfies:
```math
A S_3 B + C = S_3
```

where:

- ``A = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_+`` (same as second order)

- ``B`` involves third-order Kronecker products with special structure:
```math
B = U_3^\top \left[ \mathrm{kron}(s_{t-1}^{aug}, \sigma) + P_1^L \mathrm{kron}(s_{t-1}^{aug}, \sigma) P_1^R + P_2^L \mathrm{kron}(s_{t-1}^{aug}, \sigma) P_2^R + \mathrm{kron}^3(s_{t-1}^{aug}) \right] C_3
```
where:
  - ``U_3`` removes duplicates in third-order terms
  - ``P_1^L, P_1^R, P_2^L, P_2^R`` are permutation matrices handling different orderings
  - ``\mathrm{kron}^3`` is the third Kronecker power
  - ``C_3`` is a compression matrix

- ``C`` includes third derivatives and second-order solution terms:
```math
C = (\nabla_0 + \nabla_+ S_1^y)^{-1} \left[ \nabla_3 \mathrm{kron}^3(s_t^{aug}) + \nabla_2 \mathrm{kron}(s_t^{aug}, s_2^{aug}) \right]
```

The key challenge is efficiently computing ``\mathrm{kron}^3(s_{t-1}^{aug})`` and the interaction between third derivatives ``\nabla_3`` and the existing second-order solution.

**Step 3: Solve Third-Order Sylvester Equation**

Same methods as second order, but with larger matrices. Sparsity and efficient Kronecker product implementations are crucial for performance.

**Step 4: Sparse Storage**

The third-order solution is extremely sparse. The package uses specialized compression to store only non-zero elements.

### Matrix Construction

The third-order solution matrix is:
```math
S_3 \in \mathbb{R}^{n \times \frac{(n_- + 1 + n_e)(n_- + 1 + n_e + 1)(n_- + 1 + n_e + 2)}{6}}
```

The size accounts for:
- Three-way symmetry in third derivatives
- All combinations of three state variables/shocks
- The perturbation parameter

## Pruning

For second and third-order solutions, the package implements **pruning** following [andreasen2018pruning](@citet). Pruning prevents explosive behavior in simulations by carefully tracking different orders of approximation separately.

The pruned state-space representation for third-order is:
```math
\begin{aligned}
y_t^{(1)} &= \bar{y} + S_1 s_t \\
y_t^{(2)} &= \bar{y} + S_1 s_t + \frac{1}{2} S_2 (s_t^{(1)})^{\otimes 2} \\
y_t^{(3)} &= \bar{y} + S_1 s_t + \frac{1}{2} S_2 (s_t^{(1)})^{\otimes 2} + S_2 s_t^{(1)} \otimes (s_t^{(2)} - s_t^{(1)}) + \frac{1}{6} S_3 (s_t^{(1)})^{\otimes 3}
\end{aligned}
```

where ``s_t^{(k)}`` are state vectors at order ``k``.

## Computational Considerations

### Automatic Differentiation

The package uses symbolic differentiation to compute ``\nabla_1, \nabla_2, \nabla_3`` exactly. This is done:
1. Parse model equations symbolically
2. Compute derivatives symbolically
3. Generate efficient evaluation code

### Sparsity Exploitation

- Second and third derivative tensors are extremely sparse
- Compressed Sparse Column (CSC) format for storage
- Specialized Kronecker product implementations that exploit sparsity
- Drop tolerance for removing near-zero elements

### Numerical Stability

- QR decomposition for preprocessing
- LU factorization with pivoting
- Iterative refinement in equation solvers
- Multiple solver algorithms with fallback options
- Tolerance checking at each step

### Caching

To improve performance for repeated solutions (e.g., in estimation):
- Preallocated workspace matrices
- Cached factorizations when parameters change little
- Warm-starting iterative solvers with previous solutions

## Example: Checking Solution Quality

After obtaining a solution, you can verify its accuracy:

```julia
using MacroModelling

@model RBC begin
    1/c[0] = (β/c[1]) * (α * exp(z[1]) * k[0]^(α-1) + (1-δ))
    c[0] + k[0] = (1-δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + σ * eps_z[x]
end

@parameters RBC begin
    α = 0.5
    β = 0.95
    δ = 0.02
    ρ = 0.9
    σ = 0.01
end

# Get different order solutions
get_solution(RBC)  # First order (default)
get_solution(RBC, order = 2)  # Second order
get_solution(RBC, order = 3)  # Third order
```

The solution accuracy can be assessed by:
- Checking residuals of the policy function
- Comparing impulse responses across orders
- Examining Euler equation errors in simulations

## References

The implementation closely follows:

- **First order**: [villemot2011solving](@citet) provides the theoretical foundation for the QME approach
- **Second/Third order**: [levintal2017fifth](@citet) extends the method to higher orders using generalized Sylvester equations
- **Pruning**: [andreasen2018pruning](@citet) develops the pruned state-space system preventing explosive simulations

Additional implementation details are inspired by Dynare (see [dynare](@citet)).
