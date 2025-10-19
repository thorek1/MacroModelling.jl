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

## Tensor Representation and Flattening

Higher-order perturbation solutions naturally involve tensors (multidimensional arrays). However, the algorithms work with these tensors in a flattened 2D matrix representation for computational efficiency. This section explains the mapping between tensor and matrix representations.

### Tensor Form of Higher-Order Solutions

In their natural form, higher-order perturbation solutions are represented as tensors:

**Second-order**: The solution is a 3-tensor ``\mathcal{S}_2 \in \mathbb{R}^{n \times n_s \times n_s}`` where:
```math
y_{t,i} - \bar{y}_i = \sum_{j=1}^{n_s} \sum_{k=1}^{n_s} \mathcal{S}_{2,ijk} \, s_{t,j} \, s_{t,k}
```
for variable ``i`` and state variables ``s_t = [y_{t-1} - \bar{y}, \sigma, u_t]^\top`` of dimension ``n_s = n_- + 1 + n_e``.

**Third-order**: The solution is a 4-tensor ``\mathcal{S}_3 \in \mathbb{R}^{n \times n_s \times n_s \times n_s}`` where:
```math
y_{t,i} - \bar{y}_i = \sum_{j=1}^{n_s} \sum_{k=1}^{n_s} \sum_{\ell=1}^{n_s} \mathcal{S}_{3,ijk\ell} \, s_{t,j} \, s_{t,k} \, s_{t,\ell}
```

### Symmetry and Compression

Due to the symmetry of partial derivatives, many tensor entries are equal:
- **Second-order**: ``\mathcal{S}_{2,ijk} = \mathcal{S}_{2,ikj}`` (order of ``j`` and ``k`` doesn't matter)
- **Third-order**: ``\mathcal{S}_{3,ijk\ell}`` is symmetric in indices ``j, k, \ell`` (6 equivalent orderings)

This symmetry allows significant **compression** - we only need to store unique entries.

### Flattening to 2D Matrices

The implementation works with **flattened** 2D matrices rather than full tensors:

#### Second-Order Flattening

The 3-tensor ``\mathcal{S}_2`` is flattened into a matrix ``S_2 \in \mathbb{R}^{n \times m_2}`` where ``m_2 = \frac{n_s(n_s+1)}{2}`` (only unique entries due to symmetry).

**Mapping**: The Kronecker product ``s_t \otimes s_t`` creates a vector of length ``n_s^2``:
```math
s_t \otimes s_t = \begin{bmatrix} s_{t,1} s_{t,1} \\ s_{t,1} s_{t,2} \\ \vdots \\ s_{t,1} s_{t,n_s} \\ s_{t,2} s_{t,1} \\ \vdots \\ s_{t,n_s} s_{t,n_s} \end{bmatrix}
```

The **compression matrix** ``C_2 \in \mathbb{R}^{n_s^2 \times m_2}`` selects the unique symmetric entries:
- It maps the full Kronecker product (``n_s^2`` entries) to compressed form (``m_2`` unique entries)
- Column ``k`` of ``C_2`` has a single 1 at the position of the ``k``-th unique product term
- For example, if ``s_t = [s_1, s_2]^\top`` then ``C_2`` maps ``[s_1 s_1, s_1 s_2, s_2 s_1, s_2 s_2]^\top`` to ``[s_1 s_1, s_1 s_2, s_2 s_2]^\top``

The **uncompression matrix** ``U_2 = C_2^\top D`` where ``D`` duplicates entries to restore full form:
```math
U_2 \in \mathbb{R}^{m_2 \times n_s^2}
```
- It expands the compressed solution back to full Kronecker product form
- Ensures ``s_t \otimes s_t = U_2 \cdot C_2^\top (s_t \otimes s_t)``

**Usage in algorithm**:
```julia
# Compress: full Kronecker product → unique entries
compressed_kron = C_2' * kron(s_t, s_t)  # m_2 × 1

# Uncompress: solution in compressed form → full form for multiplication
full_form = U_2 * S_2_compressed  # Apply solution
```

The second-order solution satisfies ``A S_2 B + C = S_2`` where ``B`` operates in compressed space.

#### Third-Order Flattening  

The 4-tensor ``\mathcal{S}_3`` is flattened into ``S_3 \in \mathbb{R}^{n \times m_3}`` where ``m_3 = \frac{n_s(n_s+1)(n_s+2)}{6}`` (unique entries with three-way symmetry).

The Kronecker cube ``s_t \otimes s_t \otimes s_t`` has ``n_s^3`` entries, but only ``m_3`` are unique.

**Compression and uncompression matrices**:
- ``C_3 \in \mathbb{R}^{n_s^3 \times m_3}``: maps full cube to unique entries
- ``U_3 \in \mathbb{R}^{m_3 \times n_s^3}``: expands back to full form

The construction is analogous to second-order but handles three-way symmetry.

### Permutation Matrices

Permutation matrices handle different orderings of tensor indices. They are crucial when combining terms that involve products in different orders.

#### Second-Order Permutations

For second-order, the main permutation concern is swapping the two state arguments. The symmetry ``\mathcal{S}_{2,ijk} = \mathcal{S}_{2,ikj}`` is handled by ``C_2`` and ``U_2``.

#### Third-Order Permutations

Third-order involves multiple permutation matrices due to three indices:

**Matrix ``\mathbf{P}``**: Combines different orderings in derivative calculations. It represents the sum:
```math
\mathbf{P} = P_{(1,2,3)} + P_{(1,3,2)} + P_{(3,1,2)}
```
where ``P_{(\pi)}`` permutes a flattened tensor according to permutation ``\pi``. This handles terms like:
```math
\frac{\partial^3 f}{\partial x_i \partial x_j \partial x_k} \text{ with all orderings of } i,j,k
```

**Left and right permutations** (``P_{1\ell}, P_{1r}``, etc.): Handle specific reorderings needed when computing products:
- ``P_{1\ell}``: Left-multiplication permutation swapping first two indices ``(i,j,k) \to (j,i,k)``
- ``P_{1r}``: Right-multiplication permutation (transpose of ``P_{1\ell}``)
- ``P_{1\ell̂}, P_{2\ell̂}``: Permutations for derivative space (full dimension ``\bar{n}``)
- ``P_{1\ell̄}, P_{2\ell̄}``: Permutations for compressed state space
- ``P_{1r̃}, P_{2r̃}``: Right-multiplication versions for compressed space

These enable efficient computation of terms like:
```math
\nabla_3 \mathrm{kron}(s_t, s_t, s_t) + \nabla_3 P_{1\ell̂} \mathrm{kron}(s_t, s_t, s_t) P_{1r̃}
```
which combines the derivative tensor with different argument orderings.

**Matrix ``\mathbf{SP}``**: A special sparse permutation that identifies which state variables actually appear in third-order terms (exploiting sparsity in the derivative tensor).

### Example: Second-Order Matrix Construction

Consider a simple case with ``n_s = 3`` state variables ``s_t = [s_1, s_2, s_3]^\top``.

**Full Kronecker product** (9 entries):
```math
s_t \otimes s_t = [s_1 s_1, s_1 s_2, s_1 s_3, s_2 s_1, s_2 s_2, s_2 s_3, s_3 s_1, s_3 s_2, s_3 s_3]^\top
```

**Compressed form** (6 unique entries due to symmetry):
```math
\text{compressed} = [s_1 s_1, s_1 s_2, s_1 s_3, s_2 s_2, s_2 s_3, s_3 s_3]^\top
```

**Compression matrix** ``C_2`` (9 × 6):
```math
C_2 = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
```

Notice rows 2 and 4 are identical (both map to ``s_1 s_2``), as are rows 3 and 7 (``s_1 s_3``), and rows 6 and 8 (``s_2 s_3``).

The algorithm computes ``S_2`` in the compressed 6-column space and uses ``U_2`` when applying it to full Kronecker products.

### Why This Matters

This flattening approach provides significant benefits:

1. **Memory efficiency**: Storing ``\frac{n_s(n_s+1)}{2}`` instead of ``n_s^2`` entries for second-order (factor of ~2 savings)
2. **Computational efficiency**: Operating on compressed matrices is faster
3. **Numerical stability**: Compressed form naturally enforces symmetry constraints
4. **Sparse operations**: Real models are sparse; flattened form enables efficient sparse linear algebra

The compression/permutation matrices are precomputed once when the model is parsed and reused throughout the solution process.

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

where ``S_2`` is stored in **compressed form** (see the Tensor Representation section above):

- ``A = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_+ \in \mathbb{R}^{n \times n}``

- ``B = U_2^\top \mathrm{kron}(s_{t-1}^{aug}, s_{t-1}^{aug}) C_2 \in \mathbb{R}^{m_2 \times m_2}`` operates in compressed space:
  - ``C_2 \in \mathbb{R}^{n_s^2 \times m_2}``: compression matrix selecting unique symmetric entries
  - ``U_2 \in \mathbb{R}^{m_2 \times n_s^2}``: uncompression matrix expanding back to full Kronecker form
  - The product ``U_2^\top \mathrm{kron}(s, s) C_2`` represents the state evolution in compressed coordinates

- ``C = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_2 \mathrm{kron}(s_t^{aug}, s_t^{aug})`` contains the second-order derivatives term:
  - ``\nabla_2`` is the Hessian of model equations (sparse due to limited cross-products)
  - Applied to the Kronecker product of current states
  - Then pre-multiplied by the inverted Jacobian

The compression ensures we solve for only ``m_2 = \frac{n_s(n_s+1)}{2}`` unique entries instead of ``n_s^2``.

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

where ``S_3`` is stored in **compressed form** with ``m_3 = \frac{n_s(n_s+1)(n_s+2)}{6}`` columns:

- ``A = (\nabla_0 + \nabla_+ S_1^y)^{-1} \nabla_+`` (same as second order)

- ``B \in \mathbb{R}^{m_3 \times m_3}`` involves third-order Kronecker products with permutations:
```math
B = U_3^\top \left[ \mathrm{kron}(s_{t-1}^{aug}, \sigma) + P_{1\ell̄} \mathrm{kron}(s_{t-1}^{aug}, \sigma) P_{1r̃} + P_{2\ell̄} \mathrm{kron}(s_{t-1}^{aug}, \sigma) P_{2r̃} + \mathrm{kron}^3(s_{t-1}^{aug}) \right] C_3
```
where (see Tensor Representation section):
  - ``C_3 \in \mathbb{R}^{n_s^3 \times m_3}``: compression matrix selecting unique entries with three-way symmetry
  - ``U_3 \in \mathbb{R}^{m_3 \times n_s^3}``: uncompression matrix expanding to full Kronecker cube
  - ``P_{1\ell̄}, P_{2\ell̄}``: left permutation matrices reordering the three indices in compressed space
  - ``P_{1r̃}, P_{2r̃}``: corresponding right permutation matrices (transposes)
  - ``\mathrm{kron}^3(s) = s \otimes s \otimes s``: third Kronecker power
  - The multiple permutation terms handle different orderings: ``(i,j,k,\sigma)``, ``(i,k,j,\sigma)``, ``(k,i,j,\sigma)``

- ``C`` includes third derivatives and interactions with the second-order solution:
```math
C = (\nabla_0 + \nabla_+ S_1^y)^{-1} \left[ \nabla_3 \left(\mathrm{kron}^3(s_t^{aug}) + P_{1\ell̂} \mathrm{kron}^3(s_t^{aug}) P_{1r̃} + \cdots \right) + \nabla_2 \mathrm{kron}(s_t^{aug}, s_2^{aug}) \right]
```
where:
  - ``\nabla_3``: third-order derivative tensor (extremely sparse, applied in flattened form)
  - ``P_{1\ell̂}, P_{2\ell̂}``: permutations in full derivative space dimension ``\bar{n}``
  - The second term couples second-order solution ``S_2`` with first-order states

The key computational challenges:
1. Efficiently computing ``\mathrm{kron}^3(s_{t-1}^{aug})`` in compressed form
2. Applying multiple permutations to handle all derivative orderings
3. Maintaining sparsity throughout (``\nabla_3`` is extremely sparse)

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
    z[0] = ρᶻ * z[-1] + σᶻ * ϵᶻ[x]
end

@parameters RBC begin
    α = 0.5
    β = 0.95
    δ = 0.02
    ρᶻ = 0.9
    σᶻ = 0.01
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
