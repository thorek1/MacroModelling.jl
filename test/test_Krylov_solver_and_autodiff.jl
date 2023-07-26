using SparseArrays
import LinearAlgebra as ℒ
import LinearOperators
import Krylov
import ForwardDiff as ℱ
import RecursiveFactorization as RF

using MacroModelling
include("models/FS2000.jl")


SSS(m)

using FiniteDifferences
SSS(m,derivatives = false)[1]
pars = copy(m.parameter_values)
fin_grad = FiniteDifferences.grad(central_fdm(4,1),x->SSS(m,derivatives = false, parameters = x)[10],pars)[1]
SSS(m, parameters = pars)[10,2:end]

SSS(m, parameters = pars)
get_solution(m)

include("models/RBC_CME.jl")


get_solution(m,algorithm = :pruned_second_order)

parameters = m.parameter_values
𝓂 = m
verbose = true

SS_and_pars, solution_error = 𝓂.SS_solve_func(parameters, 𝓂, verbose)
    
∇₁ = calculate_jacobian(parameters, SS_and_pars, 𝓂)

𝑺₁, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)

∇₂ = calculate_hessian(parameters, SS_and_pars, 𝓂)


M₂ = 𝓂.solution.perturbation.second_order_auxilliary_matrices
T = 𝓂.timings
tol = eps()



# Indices and number of variables
i₊ = T.future_not_past_and_mixed_idx;
i₋ = T.past_not_future_and_mixed_idx;

n₋ = T.nPast_not_future_and_mixed
n₊ = T.nFuture_not_past_and_mixed
nₑ = T.nExo
n  = T.nVars
nₑ₋ = n₋ + 1 + nₑ

# 1st order solution
𝐒₁ = @views [𝑺₁[:,1:n₋] zeros(n) 𝑺₁[:,n₋+1:end]] |> sparse
droptol!(𝐒₁,tol)

𝐒₁₋╱𝟏ₑ = @views [𝐒₁[i₋,:]; zeros(nₑ + 1, n₋) spdiagm(ones(nₑ + 1))[1,:] zeros(nₑ + 1, nₑ)];

⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋ = @views [(𝐒₁ * 𝐒₁₋╱𝟏ₑ)[i₊,:]
                            𝐒₁
                            spdiagm(ones(nₑ₋))[[range(1,n₋)...,n₋ + 1 .+ range(1,nₑ)...],:]];

𝐒₁₊╱𝟎 = @views [𝐒₁[i₊,:]
                zeros(n₋ + n + nₑ, nₑ₋)];


∇₁₊𝐒₁➕∇₁₀ = @views -∇₁[:,1:n₊] * 𝐒₁[i₊,1:n₋] * ℒ.diagm(ones(n))[i₋,:] - ∇₁[:,range(1,n) .+ n₊]

spinv = sparse(inv(∇₁₊𝐒₁➕∇₁₀))
droptol!(spinv,tol)

∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹ = - ∇₂ * sparse(ℒ.kron(⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋, ⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋) + ℒ.kron(𝐒₁₊╱𝟎, 𝐒₁₊╱𝟎) * M₂.𝛔) * M₂.𝐂₂ 

X = spinv * ∇₂⎸k⎸𝐒₁𝐒₁₋╱𝟏ₑ⎹╱𝐒₁╱𝟏ₑ₋➕𝛔k𝐒₁₊╱𝟎⎹
droptol!(X,tol)

∇₁₊ = @views sparse(∇₁[:,1:n₊] * spdiagm(ones(n))[i₊,:])

B = spinv * ∇₁₊
droptol!(B,tol)

C = (M₂.𝐔₂ * ℒ.kron(𝐒₁₋╱𝟏ₑ, 𝐒₁₋╱𝟏ₑ) + M₂.𝐔₂ * M₂.𝛔) * M₂.𝐂₂
droptol!(C,tol)


concat_sparse = [vec(B) 
                vec(C) 
                vec(X)]

dims_sparse = [size(B) 
                size(C) 
                size(X)]


function sylvester_equation_solver(concat_sparse_vec::AbstractArray{Float64}; dims::Vector{Tuple{Int,Int}})
    lenA = dims[1][1] * dims[1][2]
    lenB = dims[2][1] * dims[2][2]

    A = sparse(reshape(concat_sparse_vec[1 : lenA],dims[1]))
    B = sparse(reshape(concat_sparse_vec[lenA .+ (1 : lenB)],dims[2]))
    X = sparse(reshape(concat_sparse_vec[lenA + lenB + 1 : end],dims[3]))

    function sylvester!(sol,𝐱)
        𝐗 = sparse(reshape(𝐱, size(X)))
        sol .= vec(𝐗 - A * 𝐗 * B)
        return sol
    end

    sylvester = LinearOperators.LinearOperator(Float64, length(X), length(X), false, false, sylvester!)

    x, info = Krylov.bicgstab(sylvester, sparsevec(collect(-X)), atol = tol)

    if !info.solved
        x, info = Krylov.gmres(sylvester, sparsevec(collect(-X)), atol = tol)
    end

    x = reshape(x,size(X))
    # droptol!(x,tol)
end


x = sylvester_equation_solver(concat_sparse,dims = dims_sparse)

# collect(X + x - B * x * C)
# xx = x - B * x * C
# collect(xx)
# collect(x)
# collect(X)

B_1 = findnz(B)[1] |> unique
B_2 = findnz(B)[2] |> unique

union(B_1,B_2)

function sylvester_equation_solver_conditions(concat_sparse_vec, x; dims::Vector{Tuple{Int,Int}})
    lenA = dims[1][1] * dims[1][2]
    lenB = dims[2][1] * dims[2][2]

    A = sparse(reshape(concat_sparse_vec[1 : lenA],dims[1]))
    B = sparse(reshape(concat_sparse_vec[lenA .+ (1 : lenB)],dims[2]))
    X = sparse(reshape(concat_sparse_vec[lenA + lenB + 1 : end],dims[3]))

    collect(X + x - A * x * B)
end

sylvester_equation_solver_conditions(concat_sparse,sylvester_equation_solver(concat_sparse,dims = dims_sparse),dims = dims_sparse)|>collect


function sylvester_equation_solver(concat_sparse_vec::AbstractArray{ℱ.Dual{Z,S,N}}; dims::Vector{Tuple{Int,Int}},tol::AbstractFloat = 1e-10) where {Z,S,N}
    # unpack: AoS -> SoA
    concat_sparse_vec_values = ℱ.value.(concat_sparse_vec)

    lenA = dims[1][1] * dims[1][2]
    lenB = dims[2][1] * dims[2][2]

    A = (reshape(concat_sparse_vec_values[1 : lenA],dims[1]))
    B = (reshape(concat_sparse_vec_values[lenA .+ (1 : lenB)],dims[2]))
    # X = sparse(reshape(concat_sparse_vec_values[lenA + lenB + 1 : end],dims[3]))


    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ℱ.partials, hcat, concat_sparse_vec)'

    # get f(vs)
    val = sylvester_equation_solver(concat_sparse_vec_values, dims = dims)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    b = ℱ.jacobian(x -> sylvester_equation_solver_conditions(x, val, dims = dims), concat_sparse_vec_values)
    a = ℱ.jacobian(x -> sylvester_equation_solver_conditions(concat_sparse_vec_values, x, dims = dims), val)
    # println(A)
    # println(size(A))
    # b = hcat(ℒ.kron(-x * B, ℒ.I(size(A,1)))', ℒ.kron(ℒ.I(size(B,1)), A * x), ℒ.I(length(X)))
    # a = reshape(permutedims(reshape(ℒ.I - ℒ.kron(A, B) ,size(B,1), size(A,1), size(A,1), size(B,1)), [2, 3, 4, 1]), size(A,1) * size(B,1), size(A,1) * size(B,1))

    Â = RF.lu(a, check = false)

    if !ℒ.issuccess(Â)
        Â = ℒ.svd(a)
    end
    
    jvp = -(Â \ b) * ps

    # lm = LinearMap{Float64}(x -> A * reshape(x, size(B)), length(B))

    # jvp = - sparse(reshape(ℐ.gmres(lm, sparsevec(B)), size(B))) * ps
    # jvp *= -ps

    # pack: SoA -> AoS
    return reshape(map(val, eachrow(jvp)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end,size(val))
end



using ForwardDiff

jaco = ForwardDiff.jacobian(x->sylvester_equation_solver(x,dims = dims_sparse),collect(concat_sparse))|>sparse

jaco |> collect
jaco[:,1:length(B)]|> sparse
jaco[:,length(B) .+ (1 : length(C))]|> sparse
jaco[:,length(B) + length(C) + 1 : end]#|> sparse

B|>collect
C|>collect
X|>collect
x|>collect

reduce(hcat,size(B),size(C))

using Zygote
jacoAA = ForwardDiff.jacobian(x->sylvester_equation_solver_conditions(concat_sparse,x,dims = dims_sparse),x)

jacoZ_AA = Zygote.jacobian(x->sylvester_equation_solver_conditions(collect(concat_sparse),x,dims = dims_sparse),collect(x))[1]
collect(jacoZ_AA)

result = reshape(permutedims(reshape(CC,size(C,1),size(B,1),size(B,1),size(C,1)), [2, 3, 4, 1]),size(B,1)*size(C,1),size(B,1)*size(C,1))




# the last part is ℒ.I(length(X))

jacoZ_B = Zygote.jacobian(y->sylvester_equation_solver_conditions(y, collect(x), dims = dims_sparse), collect(concat_sparse))[1]|>sparse

jacoZ_BB = Zygote.jacobian(y->sylvester_equation_solver_conditions([y; vec(collect(C)); vec(collect(X))], collect(x), dims = dims_sparse), collect(vec(B)))[1]
jacoZ_CC = Zygote.jacobian(y->sylvester_equation_solver_conditions([vec(collect(B)); y;vec(collect(X))], collect(x), dims = dims_sparse), collect(vec(C)))[1]
jacoZ_XX = Zygote.jacobian(y->sylvester_equation_solver_conditions([vec(collect(B));vec(collect(C));y], collect(x), dims = dims_sparse), collect(vec(X)))[1]



sparse(ℒ.kron(-x*C, II)')
tot = sparse(ℒ.kron(-spdiagm(B2) * x*C ,ℒ.I(size(B,1)))')

tot * spdiagm(kron(B2,B1))


jacoZ_BB |> sparse

II  = spzeros(size(B))

[II[i,i] = 1 for i in B_2]
B_1 = findnz(B)[1] |> unique
B_2 = findnz(B)[2] |> unique
B1 = sparsevec(B_1,1,size(B,1))
B2 = sparsevec(B_2,1,size(B,1))
kron(B1,B2)
kron(I(size(C,1)), B*x)|>collect




deltaaa = collect(kron(x*C,I(size(B,1)))') + jacoZ_BB
# AAA = Zygote.jacobian(y-> reshape(y,size(A)) * X * B, vec(A))[1]

# kron(x*C,I(n1))' == AAA


# BBB = Zygote.jacobian(y-> A * X * reshape(y,size(B)), vec(B))[1]

# kron(I(n2),A*X) == BBB



BBB = Zygote.jacobian(y-> reshape(y,size(B)) * x * C, collect(vec(B)))[1]


kron(B,x)
kron(C,x)



sparse(jacoZ_BB)
sparse(jacoZ_CC)


sum(jacoZ_BB[:,length(B) + length(C) + 1 : end]- ℒ.I)

xyz = sparse(jacoZ_BB[:,1:length(C)])
findnz(xyz)[3]|>sort|>unique
XjacoZ_BB[:,1:length(B)] |> sparse
kron(collect(x),B)
XYZ = - collect(x) * collect(C)

findnz(sparse(XYZ))[3]|>sort|>unique
-pinv(jacoZ_AA) * vec(ℒ.I - x * C)


kron(C',x)
kron(B',x)

kron(ℒ.I(size(B,2)), x * C)


Â = RF.lu(jacoZ_AA, check = false)


Â = ℒ.svd(jacoZ_AA)
-pinv(jacoZ_AA) * jacoZ_BB
collect(jacoZ_AA)



# Assume A, B, I are already defined

CC = ℒ.I - kron(B, C)  # Equivalent to ArrayTensorProduct(-A.T, B)

for i in permutations(1:4)

result = reshape(permutedims(reshape(CC,size(C,1),size(B,1),size(B,1),size(C,1)),i),size(B,1)*size(C,1),size(B,1)*size(C,1))
if result == jacoZ_AA
    println(i)
end
end
# DD = ℒ.I(size(B,1) * size(C,1))  # Equivalent to ArrayTensorProduct(I, I)

# Reshape and permute dimensions
CCC = permutedims(reshape(CC', size(B,2), size(B,1), size(C,2), size(C,1)), (1,3,4,2));
# DD = permutedims(reshape(DD, size(B,2), size(B,1), size(C,2), size(C,1)), (2,1,4,3));
# reshape(DD,(size(B,1) * size(C,1),size(B,1) * size(C,1)))'
# Array addition
result = reshape(CCC,(size(B,1) * size(C,1),size(B,1) * size(C,1)))'



# Perform array addition operation
# spdiagm(ℒ.diag(kron(B', C)) .== 0)
# aaa =  reshape((spdiagm(ℒ.diag(kron(B', C)) .!= 0) - kron(B', C)),(size(C,1),size(B,1),size(C,2),size(B,2)));
aaa =  reshape(- kron(B', C),(size(C,1),size(B,1),size(C,2),size(B,2)));
bbb =  reshape(ℒ.I(size(B,1) * size(C,1)) ,(size(C,1),size(B,1),size(C,2),size(B,2)));
# aaa =  reshape(kron(B', C),(size(C,1),size(B,1),size(C,2),size(B,2)));
aa = reshape(permutedims(aaa+bbb,(4,3,2,1)),(size(B,1)*size(C,1), size(B,1)*size(C,1)))

AA
aaaa = sparse(aa - AA)
droptol!(aaaa,eps(Float32))

aa =  collect(ℒ.I - kron(B, C'))

spB = collect(B)
spC = collect(C)
Ib = collect(ℒ.I(size(B,1)))
Ic = collect(ℒ.I(size(C,1)))
using TensorOperations
@tensor AAA[i,j,k,l] :=  - spB[i,j]*spC[k,l] ;
# Assume I is the identity matrix
@tensor DD[i,j,k,l] := Ib[i,j] * Ic[k,l];  # Equivalent to ArrayTensorProduct(I, I)

# AAA + DD;

# Permute dimensions
result = permutedims(AAA + DD, (2,1,4,3));
# DD = permutedims(DD, (4,3,2,1));

# Array addition
# result = AAA + DD;

reshape(result,(size(B,1) * size(C,1), size(B,1) * size(C,1)))
# AA
collect(jacoZ_AA)


# B C



# start from scratch

n1 = 3
n2 = 4
A = rand(n1,n1)|>collect
B = rand(n2,n2)|>collect
x = rand(n1,n2)|>collect
X = rand(n1,n2)|>collect
using Zygote
using LinearAlgebra

zsp = Zygote.jacobian(x -> X + x - A * x * B,x)[1]#|>sparse
# 12×12 Matrix{Float64}:
#   1.0          0.0         0.0       -0.010578  -0.512005  …   0.0        0.0        0.0         0.0        0.0
#   0.0          1.0         0.0        0.0       -0.405139      0.0        0.0        0.0         0.0        0.0
#   0.0          0.0         1.0       -0.313406  -0.475476      0.0        0.0        0.0         0.0        0.0
#  -0.00872096  -0.422118    0.0        1.0        0.0           0.0        0.0       -0.0104956  -0.508015   0.0
#   0.0         -0.334012   -0.410547   0.0        1.0           0.0        0.0        0.0        -0.401982  -0.49409
#  -0.258384    -0.392001    0.0        0.0        0.0       …   0.0        0.0       -0.310964   -0.471771   0.0
#   0.0          0.0         0.0        0.0        0.0          -0.520492   0.0       -0.0139415  -0.674807   0.0
#   0.0          0.0         0.0        0.0        0.0           0.588146  -0.506225   0.0        -0.53396   -0.65631
#   0.0          0.0         0.0        0.0        0.0          -0.483357   1.0       -0.41306    -0.626663   0.0
#  -0.00228955  -0.11082     0.0        0.0        0.0          -0.498453   0.0        1.0         0.0        0.0
#   0.0         -0.0876896  -0.107782   0.0        0.0       …  -0.394415  -0.48479    0.0         1.0        0.0
#  -0.0678347   -0.102914    0.0        0.0        0.0          -0.462891   0.0        0.0         0.0        1.0

kron(A,B)
linnnz = reshape(permutedims(reshape((I - kron(A',B)),n1,n2,n1,n2),(1,3,2,4)),n1*n2,n1*n2)'
linnnz = reshape(permutedims(reshape(I- kron(A,B),n1,n2,n1,n2),(2,1,4,3)),n1*n2,n1*n2)#|>sparse
linnnz = reshape(permutedims(reshape(I- kron(A,B),n1,n1,n2,n2),(2,1,4,3)),n1*n2,n1*n2)#|>sparse

findnz(sparse(zsp).==1)[3]|>sort
(findnz(linnnz)[3]|>sort)==(findnz(sparse(zsp))[3]|>sort)

using LinearAlgebra
using TensorOperations
using Combinatorics
# Assume A, B, I are already defined
@tensor C[i,j,k,l] := -A[i,j]*B[k,l]  # Equivalent to ArrayTensorProduct(-A.T, B)


Ia = collect(ℒ.I(size(A,1)))
Ib = collect(ℒ.I(size(B,1)))

# Assume I is the identity matrix
@tensor D[i,j,k,l] := Ia[i,j]*Ib[k,l]  # Equivalent to ArrayTensorProduct(I, I)

for i in permutations(1:4)
# Permute dimensions
CC = permutedims(C, [1, 4, 2, 3]);
DD = permutedims(D, [1, 4, 2, 3]);

# Array addition
result = reshape(CC + DD,n1*n2,n1*n2)
if result==zsp
    println(i)
end
end



for i in permutations(1:4)
result = reshape(permutedims(reshape((I - kron(A,B)),n2,n1,n1,n2),i),n1*n2,n1*n2)
if result==zsp
    println(i)
end
end

result = reshape(permutedims(reshape((I - kron(A,B)),n2,n1,n1,n2),[2, 3, 4, 1]),n1*n2,n1*n2)




# start Again

n1 = 3
n2 = 4
A = rand(n1,n1)|>collect
B = rand(n2,n2)|>collect
x = rand(n1,n2)|>collect
X = rand(n1,n2)|>collect
using Zygote
using LinearAlgebra
AAA = Zygote.jacobian(y-> reshape(y,size(A)) * X * B, vec(A))[1]

kron(X*B,I(n1))' == AAA


BBB = Zygote.jacobian(y-> A * X * reshape(y,size(B)), vec(B))[1]

kron(I(n2),A*X) == BBB
