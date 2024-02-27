from sympy import MatrixSymbol, Matrix, diff#, ArrayTensorProduct, PermuteDims
from sympy.tensor.array import permutedims
from sympy.tensor.array.expressions import ArrayTensorProduct
from sympy import init_printing
from sympy.combinatorics import Permutation
init_printing(use_unicode=True)

sol_d = MatrixSymbol('sol_d',9,3)
e1 = MatrixSymbol('e1',3,9)
e2 = MatrixSymbol('e2',3,9)
nab = MatrixSymbol('nab',9,17)
colA = MatrixSymbol('colA',17,3)
colB = MatrixSymbol('colB',17,9)

# x = nab * colA * e1 * sol_d * e2 * sol_d * e2 * e2.T

x = (nab * colB) * (sol_d * e2) * e2.T

diff(x,sol_d)
[0,1,2,3]
ArrayTensorProduct(e1.T*cs.T*nab.T, e2*sol_d*e2*e2.T).as_explicit()
permutedims(ArrayTensorProduct(e1.T*cs.T*nab.T, e2*sol_d*e2*e2.T), Permutation(3)(1, 2)))
ArrayAdd(PermuteDims(ArrayTensorProduct(e1.T*cs.T*nab.T, e2*sol_d*e2*e2.T), (3)(1 2)), PermuteDims(ArrayTensorProduct(e2.T*sol_d.T*e1.T*cs.T*nab.T, e2*e2.T), (3)(1 2)))


    expand = @ignore_derivatives [ℒ.diagm(ones(T.nVars))[T.future_not_past_and_mixed_idx,:], ℒ.diagm(ones(T.nVars))[T.past_not_future_and_mixed_idx,:]] 

    colselect = ℒ.diagm(ones(size(∇₁,2)))[:,1:T.nFuture_not_past_and_mixed]

    A = ∇₁ * colselect * expand[1]
    # B = ∇₁[:,T.nFuture_not_past_and_mixed .+ range(1,T.nVars)]
    # C = ∇₁[:,T.nFuture_not_past_and_mixed + T.nVars .+ range(1,T.nPast_not_future_and_mixed)] * expand[2]

    sol_buf = sol_d * expand[2]

    sol_buf2 = sol_d * expand[2] * sol_d * expand[2]

    # err1 = A * sol_buf2 + B * sol_buf + C

    err1 = ∇₁ * colselect * expand[1] * sol_d * expand[2] * sol_d * expand[2] * expand[2]'  # + B * sol_buf + C
    
    
