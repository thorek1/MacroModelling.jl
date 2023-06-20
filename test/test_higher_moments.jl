using MacroModelling
import IterativeSolvers as ℐ
using LinearAlgebra, LinearMaps

function intersecting_elements(arr1::Union{Symbol,Vector}, arr2::Union{Symbol,Vector})
    common = []
    
    if arr1 isa Symbol
        arr1 = [arr1]
    end
    if arr2 isa Symbol
        arr2 = [arr2]
    else
        arr2 = copy(arr2)
    end

    for i in arr1
        if i in arr2
            push!(common, i)
            deleteat!(arr2, findfirst(==(i), arr2))  # remove the matched element from arr2
        end
    end
    return common
end

function position_in_symmetric_matrix(position::Int, length::Int)
    # Check if the vector length is a triangular number
    n = round(Int, (-1 + sqrt(1 + 8*length)) / 2)

    @assert n*(n+1)/2 == length "The length of the input vector is not valid to form a square symmetric matrix."

    @assert position >= 1 && position <= length "Invalid position in the vector."

    # Initialize the matrix position variables
    row = 1
    col = 1

    # Iterate until we reach the desired position
    for i in 1:length
        if i == position
            break
        end
        if col == n
            row += 1
            col = row
        else
            col += 1
        end
    end

    # If it's a diagonal element, return one position
    if row == col
        return (row, col)
    # If it's an off-diagonal element, return two positions
    else
        return (row, col), (col, row)
    end
end



function position_in_full_vector(position::Int, length::Int)
    # Check if the vector length is a triangular number
    n = round(Int, (-1 + sqrt(1 + 8*length)) / 2)

    @assert n*(n+1)/2 == length "The length of the input vector is not valid to form a square symmetric matrix."

    @assert position >= 1 && position <= length "Invalid position in the vector."

    # Initialize the matrix position variables
    row = 1
    col = 1

    # Iterate until we reach the desired position
    for i in 1:length
        if i == position
            break
        end
        if col == n
            row += 1
            col = row
        else
            col += 1
        end
    end

    # Calculate the corresponding position(s) in the vector
    vector_position = Int(n * (row - 1) + col)

    if row == col
        return vector_position
    else
        return vector_position, Int(n * (col - 1) + row)
    end
end

# position_in_full_vector(6,6)

# position_in_symmetric_matrix(2,6)

# AA = rand(4,4)
# AA = Symmetric(AA)
# vecAA = upper_triangle(AA)
# vec(AA)
# position_in_full_vector(1,6)

function upper_triangle(mat::AbstractArray{T,2}) where T
    @assert size(mat, 1) == size(mat, 2) "The input matrix must be square"

    upper_elems = T[]
    for i in 1:size(mat, 1)
        for j in i:size(mat, 2)
            push!(upper_elems, mat[i, j])
        end
    end
    return upper_elems
end


function upper_triangle(mat::AbstractArray{T,3}; alt::Bool = false, triple::Bool = false) where T
    @assert size(mat, 1) == size(mat, 2) "The input matrix must be square"

    upper_elems = T[]
    if alt
        for i in 1:size(mat, 1)
            for j in 1:size(mat, 2)
                for k in j:size(mat, 3)
                    push!(upper_elems, mat[i, j, k])
                end
            end
        end
    elseif triple
        for i in 1:size(mat, 1)
            for j in i:size(mat, 2)
                for k in j:size(mat, 3)
                    push!(upper_elems, mat[i, j, k])
                end
            end
        end
    else
        for j in 1:size(mat, 2)
            for k in j:size(mat, 3)
                for i in 1:size(mat, 1)
                    push!(upper_elems, mat[i, j, k])
                end
            end
        end
    end
    return upper_elems
end

function expand_mat(A::Matrix, nu::Int)
    n = size(A, 1)
    B = spzeros(Float64, n*nu, n*nu)
    for i in 1:n
        for j in 1:n
            B[((i-1)*nu+1):i*nu, ((j-1)*nu+1):j*nu] = A[i, j] * I(nu)
        end
    end
    return B
end


# Recursive function to generate all integer vectors of length n that sum up to L1
function allVL1(n::Int, L1::Int)
    # Base case: if n is 1, return L1
    if n == 1
        return [L1]
    else
        # Recursive case: generate all possible vectors for smaller values of n and L1
        v = []
        for i in 0:L1
            for vec in allVL1(n-1, L1-i)
                push!(v, [i; vec])
            end
        end
        return v
    end
end

function duplication(p)
    a = sparse(tril(ones(p,p)))

    j = 1

    for k in 1:p
        for i in 1:p
            if a[i,k]== 1
                a[i,k] = j
                j +=1
            end
        end
    end

    a = a + transpose(tril(a,-1))

    j = Int.(vec(a))

    mm = Int(p*(p+1)/2)
    
    DP = zeros(p*p,mm)

    for r in 1:size(DP,1)
        DP[r, j[r]] = 1
    end

    DPinv = (DP'*DP)\DP'

    return DP, DPinv
end


function triplication(p)
    TP = zeros(Int,p^3, Int(p*(p+1)*(p+2)/6))
    # TPinv = zeros(Int(p*(p+1)*(p+2)/6), Int(p*(p+1)*(p+2)/6))

    for k=1:p
        for j=k:p
            for i=j:p
                idx = unique([[i, j, k],
                            [i, k, j],
                            [j, k, i],
                            [j, i, k],
                            [k, j, i],
                            [k, i, j]])
                for r in idx        
                    ii = r[1]
                    jj = r[2]
                    kk = r[3]

                    n = ii + (jj-1)*p + (kk-1)*p^2
                    mm = Int(i+(j-1)*p + 1/2*(k-1)*p^2 - 1/2*j*(j-1) + 1/6*k*(k-1)*(k-2) - 1/2*(k-1)^2*p)
                    
                    TP[n,mm] = 1

                    # if i==j && j==k
                    #     TPinv[m,n] = 1
                    # elseif i>j && j==k
                    #     TPinv[m,n] = 1/3
                    # elseif i==j && j>k
                    #     TPinv[m,n] = 1/3
                    # elseif i>j && j>k
                    #     TPinv[m,n] = 1/6
                    # end

                end
                n=n+1
            end
        end
    end

    return TP
end
# triplication(2)
# end



# translate_mod_file("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/AnSchorfheide_Gaussian.mod")
# include("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/AnSchorfheide_Gaussian.jl")

include("AnSchorfheide_Gaussian.jl")
m = AnSchorfheide_Gaussian


varobs = [:YGR, :INFL, :INT]
T = m.timings
states = m.timings.past_not_future_and_mixed

nx = T.nPast_not_future_and_mixed
nu = T.nExo
ny = T.nVars


id1_xf       = 1:nx
id2_xs       = id1_xf[end]    .+ (1:nx)
id3_xf_xf    = id2_xs[end]    .+ (1:nx^2)
id4_xrd      = id3_xf_xf[end] .+ (1:nx)
id5_xf_xs    = id4_xrd[end]   .+ (1:nx^2)
id6_xf_xf_xf = id5_xf_xs[end] .+ (1:nx^3)
id1_u        = 1:nu
id2_u_u      = id1_u[end]       .+ (1:nu^2)
id3_xf_u     = id2_u_u[end]     .+ (1:nx*nu)
id4_u_xf     = id3_xf_u[end]    .+ (1:nx*nu)
id5_xs_u     = id4_u_xf[end]    .+ (1:nx*nu)   
id6_u_xs     = id5_xs_u[end]    .+ (1:nx*nu)  
id7_xf_xf_u  = id6_u_xs[end]    .+ (1:nx^2*nu)
id8_xf_u_xf  = id7_xf_xf_u[end] .+ (1:nx^2*nu)
id9_u_xf_xf  = id8_xf_u_xf[end] .+ (1:nx^2*nu)    
id10_xf_u_u  = id9_u_xf_xf[end] .+ (1:nx*nu^2)
id11_u_xf_u  = id10_xf_u_u[end] .+ (1:nx*nu^2)
id12_u_u_xf  = id11_u_xf_u[end] .+ (1:nx*nu^2)
id13_u_u_u   = id12_u_u_xf[end] .+ (1:nu^3)  


model_order = [:YGR,:INFL,:INT,:y, :R,:g,:z,:c,:dy,:p,:e_z, :e_g, :e_r]


declaration_order = [:c, :dy, :p, :y, :R, :g, :z, :YGR, :INFL, :INT]
indexin(declaration_order,m.var)

sol = get_solution(m,m.parameter_values, algorithm = :second_order)

hx = sol[2][indexin(intersect(model_order,states),m.var),indexin(intersect(model_order,states),states)]
hu = sol[2][indexin(intersect(model_order,states),m.var),((T.nPast_not_future_and_mixed + 1):end)[indexin(intersect(model_order,m.exo),m.exo)]] 
gx = sol[2][indexin(intersect(model_order,m.var),m.var),indexin(intersect(model_order,states),states)]
gu = sol[2][indexin(intersect(model_order,m.var),m.var),((T.nPast_not_future_and_mixed + 1):end)[indexin(intersect(model_order,m.exo),m.exo)]]

second_order_helper = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^2,4)
second_order_axis = vcat(T.past_not_future_and_mixed,:Volatility,T.exo)
k = 1
for i in second_order_axis
    for j in second_order_axis
        second_order_helper[k,:] = [j,i,k,string(i)*string(j)]
        k += 1
    end
end



second_order_helper_ordered = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^2,4)
second_order_axis_ordered = vcat(intersect(model_order,T.past_not_future_and_mixed),:Volatility,intersect(model_order,T.exo))
k = 1
for i in second_order_axis_ordered
    for j in second_order_axis_ordered
        second_order_helper_ordered[k,:] = [i,j,k,string(i)*string(j)]
        k += 1
    end
end



Hxx = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Huu = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Hxu = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
hss = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]

Gxx = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Guu = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Gxu = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
gss = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]


M2u = vec(I(T.nExo))


# first order
A = hx
B = hu
C = gx
D = gu

c = zeros(T.nPast_not_future_and_mixed)
d = zeros(T.nVars)

ybar = sol[1][indexin(intersect(model_order,m.var),m.var)]

Fxi = I(m.timings.nExo)

## First-order moments, ie expectation of variables
IminA = I-A
Ez   = IminA\c
Ey   = ybar + C*Ez+d; # recall y = yss + C*z + d


## Compute Zero-Lag Cumulants of innovations, states and controls
nz = size(A,1);

BFxi = B*Fxi;
DFxi = D*Fxi

CkronC = kron(C,C)
BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)

CC = BFxi*Fxi*BFxi'
# B * B'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + DFxi * Fxi * DFxi'

DP, DPinv = duplication(nx)

E_XF2min = DPinv * vec(C2z0)
E_XF1 = zeros(nx)



# Second order solution
DP, DPinv = duplication(nu)

ximin = vcat(DPinv * vec(I(nu)), E_XF1, E_XF2min)

nz = 2 * nx + nx^2
nxi = nu + nu^2 + 2 * nx * nu
nximin = nu + Int(nu * (nu + 1) / 2) + nu * nx


nx2= nx*(nx+1)/2; nx3=nx2*(nx+2)/3; nx4=nx3*(nx+3)/4;     
nu2 = nu*(nu+1)/2; 

# # Symbolic variables for shocks
# u = sym('u',[nu 1])  # Create symbolic variables for epsilon
# xf = sym('xf',[nx 1])  # Symbolic variables for first-order terms xf    
# E_uu   = sym('E_uu_',[nu2 1]) # unique elements for second-order product moments of epsi

# u = ones(nu)
# xf
# # Create minimal xi_t vector    
# u_u = kron(u,u)
# xf_u = kron(xf,u)




vvv = reverse(allVL1(nximin,2))
vvv[21]

hx_hx = kron(hx,hx)
hx_hu = kron(hx,hu)
hu_hx = kron(hu,hx)
hu_hu = kron(hu,hu)


# get Fxi
nu2 = Int(nu * (nu+1) / 2);
nxi = nu + nu^2 + 2*nx*nu;
nximin = nu + nu2 + nu*nx;

col1_u       = 1:nu;
col2_u_u     = col1_u[end]   .+ (1:nu2);
col3_xf_u    = col2_u_u[end] .+ (1:nu*nx);

row1_u       = 1:nu;
row2_u_u     = row1_u[end]    .+ (1:nu^2);    
row3_xf_u    = row2_u_u[end]  .+ (1:nu*nx);
row4_u_xf    = row3_xf_u[end] .+ (1:nx*nu);



DPu = DP
K_u_x = reshape(kron(vec(I(nu)), I(nx)), nu*nx, nu*nx)


Iu = I(nu); 
Iux = I(nu*nx);
Fxi = zeros(nxi,nximin)


Fxi[row1_u,col1_u] = Iu; 
Fxi[row2_u_u,col2_u_u] = DPu; 
Fxi[row3_xf_u,col3_xf_u] = Iux; 
Fxi[row4_u_xf,col3_xf_u] = K_u_x; 




A = zeros(nz,nz);
B = zeros(nz,nxi);
C = zeros(ny,nz);
D = zeros(ny,nxi);
c = zeros(nz,1);
d = zeros(ny,1);

A[id1_xf,id1_xf] = hx
A[id2_xs,id2_xs] = hx
A[id2_xs,id3_xf_xf] = 0.5*Hxx
A[id3_xf_xf,id3_xf_xf] = hx_hx
    
B[id1_xf,id1_u] = hu;
B[id2_xs,id2_u_u] = 1/2*Huu;
B[id2_xs,id3_xf_u] = Hxu;
B[id3_xf_xf,id2_u_u] = hu_hu;
B[id3_xf_xf,id3_xf_u] = hx_hu;
B[id3_xf_xf,id4_u_xf] = hu_hx;

C[1:ny,id1_xf] = gx;
C[1:ny,id2_xs] = gx;
C[1:ny,id3_xf_xf] = 1/2*Gxx;

D[1:ny,id1_u] = gu;
D[1:ny,id2_u_u] = 1/2*Guu;
D[1:ny,id3_xf_u] = Gxu;

c[id2_xs,1] = 1/2*hss + 1/2*Huu*M2u; 
c[id3_xf_xf,1] =hu_hu*M2u;

d[1:ny,1] = 1/2*gss + 1/2*Guu*M2u;

## First-order moments, ie expectation of variables
IminA = I-A;
Ez   = IminA\c;
Ey   = ybar + C*Ez+d; # recall y = yss + C*z + d
    

## Compute Zero-Lag Cumulants of innovations, states and controls
# GAMMA2XI = spzeros(nximin,nximin)
# GAMMA2XI[1:(nu + size(DP,2)),1:(nu + size(DP,2))] = I(nu + size(DP,2))
# GAMMA2XI[nu .+ (1:size(DP,2)),nu .+ (1:size(DP,2))] += diagm(DPinv * vec(I(nu)))
# GAMMA2XI[nu + size(DP,2) + 1 : end,nu + size(DP,2) + 1 : end] = expand_mat(C2z0,nu)

# matt = GAMMA2XI[nu .+ (1:size(DP,2)),nu .+ (1:size(DP,2))]
# findnz(kron(matt,kron(matt,matt)))

# nz = size(A,1);

# BFxi = B*Fxi
# DFxi = D*Fxi

# CkronC = kron(C,C)
# BFxikronBFxi= kron(BFxi,BFxi)
# DFxikronDFxi= kron(DFxi,DFxi)

# CC = BFxi *  GAMMA2XI  * BFxi'

# # B' * (pinv(Fxi)' * GAMMA2XI' * pinv(Fxi))' * B
# lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

# C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

# C2y0 = C * C2z0 * C' + DFxi * GAMMA2XI * DFxi'


# diag(C2y0)


# GAMMAMax = sparse(pinv(B) * CC' * pinv(B'))
# droptol!(GAMMAMax,1e-6)
# CC = B *  GAMMA2XI  * B'

# # B' * (pinv(Fxi)' * GAMMA2XI' * pinv(Fxi))' * B
# lm = LinearMap{Float64}(x -> - B * reshape(x,size(B,2),size(B,2)) * B' , size(B,2)^2)

# C2z00 = sparse(reshape(ℐ.gmres(lm, vec(-CC)), (size(B,2),size(B,2))))
# droptol!(C2z00, eps())


# B * GAMMAMax * B'
# B * C2z00 * B'
# B * (pinv(Fxi)' * GAMMA2XI * pinv(Fxi)) * B'
# idx = zeros(nu,nu)




# # go from ξ̃ to ξ using F 
# ξ = vcat(DP'*vec(I(nu)), zeros(nx) , DP'*vec(C2z0[1:nu,1:nu]))

# F = [I(nu*nx)]

# # M₂ = E[ξ ⨂ ξ]
# # ξ = [μ, μ ⨂ μ, μ ⨂ x, x ⨂ μ]

# M₂ = spzeros(nxi,nxi)
# M₂[1:nu,1:nu] = I(nu)
# M₂[nu.+(1:nu^2),nu.+(1:nu^2)] = diagm(vec(ones(nu,nu)+I(nu)))
# M₂[nu+nu^2 .+ (1:2*nu*nx),nu+nu^2 .+ (1:2*nu*nx)] = diagm(vcat(vec(C2z0[1:nu,1:nu]),vec(C2z0[1:nu,1:nu])))


####  Γ₂
# nu = 2
# nx = 2
# write a loop to fill Γ₂
# size of input vector
Γ₂ = spzeros(Int(nu + nu*(nu+1)/2 + nx*nu), Int(nu + nu*(nu+1)/2 + nx*nu))

Ε = fill(:ϵᵢₖ,nu,nu)
Ε[diagind(Ε)] .= :ϵ²

inputs = vcat(fill(:ϵ, nu), upper_triangle(Ε), fill(:ϵx, Int(nx * (nx + 1) / 2)))

n_shocks = Int(nu + nu * (nu + 1) / 2)

for (i¹,s¹) in enumerate(inputs)
    for (i²,s²) in enumerate(inputs)
        if i¹ == i² #s¹ == s² && 
            if s² == :ϵ
                Γ₂[i¹,i²] = 1 # Variance of ϵ
            end

            if s² == :ϵ²
                Γ₂[i¹,i²] = 2 # Variance of ϵ²
            end

            if s² == :ϵᵢₖ
                Γ₂[i¹,i²] = 1
            end

            if i¹ > n_shocks
                positions = position_in_symmetric_matrix(i² - n_shocks, Int(nx*(nx+1)/2))

                if positions isa Tuple{Int,Int}
                    pos = positions
                    for iᵉ in 1:nu
                        Γ₂[n_shocks + (pos[1] - 1) * nu + iᵉ, n_shocks + (pos[2] - 1) * nu + iᵉ] = C2z0[pos...] # Covariance of x
                    end
                else
                    for pos in positions
                        for iᵉ in 1:nu
                            Γ₂[n_shocks + (pos[1] - 1) * nu + iᵉ, n_shocks + (pos[2] - 1) * nu + iᵉ] = C2z0[pos...] # Covariance of x
                        end
                    end
                end
            end

        end
    end
end




BFxi = B*Fxi
DFxi = D*Fxi

CC = BFxi *  Γ₂  * BFxi'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + DFxi * Γ₂ * DFxi'





####  Γ₃
# write a loop to fill Γ₂
# size of input vector
n_entries = Int(nu + nu*(nu+1)/2 + nx*nu)
Γ₃ = zeros(n_entries, n_entries, n_entries)

Ε = reshape([(:ϵ, (i,k)) for k in 1:nu for i in 1:nu],nu,nu)

K = reshape([(:x, (k,i)) for k in 1:nx for i in 1:nx],nx,nx)

inputs = vcat([(:ϵ, i) for i in 1:nu], upper_triangle(Ε), vec(K))

n_shocks = Int(nu + nu * (nu + 1) / 2)

for (i¹,s¹) in enumerate(inputs)
    for (i²,s²) in enumerate(inputs)
        for (i³,s³) in enumerate(inputs)
            indices = Set()
            indices_x1 = Set()
            indices_x2 = Set()

            n_x = 0
            n_ϵ2 = 0
            n_same_indices_within_x = 0
            n_same_indices_within_ϵ = 0

            if s¹[1] == :x
                push!(indices_x1,s¹[2][1])
                push!(indices_x2,s¹[2][2])

                if s¹[2][1] == s¹[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s¹[2] isa Tuple
                    if s¹[2][1] == s¹[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            if s²[1] == :x
                push!(indices_x1,s²[2][1])
                push!(indices_x2,s²[2][2])

                if s²[2][1] == s²[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s²[2] isa Tuple
                    if s²[2][1] == s²[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            if s³[1] == :x
                push!(indices_x1,s³[2][1])
                push!(indices_x2,s³[2][2])

                if s³[2][1] == s³[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s³[2] isa Tuple
                    if s³[2][1] == s³[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            n_same_indices_within = n_same_indices_within_ϵ + n_same_indices_within_x

            n_same_indices_across = s¹[2] == s²[2] || s¹[2] == s³[2] || s³[2] == s²[2]

            for k in s¹[2]
                push!(indices,k)
            end
            for k in s²[2]
                push!(indices,k)
            end
            for k in s³[2]
                push!(indices,k)
            end

            if s¹[1] == s²[1] && s¹[1] == s³[1] && s¹[1] == :ϵ
                if (i¹ == i² || i¹ == i³ || i² == i³) && !(i¹ == i² && i¹ == i³)
                    if indices |> length == 1 && n_ϵ2 < 2#  || n_same_indices_across == 2)
                        Γ₃[i¹,i²,i³] = 2
                    end

                    if n_ϵ2 == 3 && n_same_indices_across == true && n_same_indices_within == 1
                        Γ₃[i¹,i²,i³] = 2
                    end
                end

                if i¹ == i² && i¹ == i³
                    if s¹[2] isa Tuple
                        if s¹[2][1] == s¹[2][2]
                            Γ₃[i¹,i²,i³] = 8 # Variance of ϵ²
                        end
                    end
                end

                if n_ϵ2 == 1 && n_same_indices_across == false && n_same_indices_within == 0 && indices |> length == 2
                    Γ₃[i¹,i²,i³] = 1
                end
            end

            if n_x == 2 && n_same_indices_within_ϵ == 1 && s¹[2][2] == s²[2][2] && s²[2][2] == s³[2][2] #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
                idxs = collect(indices_x1)

                if length(idxs) == 1
                    Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[1]]
                else
                    Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[2]]
                end
            end

            if n_x == 2 && n_ϵ2 == 1 && n_same_indices_within_ϵ == 0 && length(collect(indices_x2)) == 2 #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
                idxs = collect(indices_x1)

                if length(idxs) == 1
                    Γ₃[i¹,i²,i³] = C2z0[idxs[1],idxs[1]]
                else
                    Γ₃[i¹,i²,i³] = C2z0[idxs[1],idxs[2]]
                end
            end
        end
    end
end

Γ₃

Γ₃xi = reshape(Γ₃,n_entries^2,n_entries)



BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)


BFxi = B*Fxi
DFxi = D*Fxi

CkronC = kron(C,C)
BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)

CC = BFxikronBFxi *  Γ₃xi  * BFxi'
AA = kron(A,A)
lm = LinearMap{Float64}(x -> AA * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C3z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
reshape(C3z0,8,8,8)

C3y0 = CkronC * C3z0 * C' + DFxikronDFxi * Γ₃xi * DFxi'
reshape(C3y0,5,5,5)


# make the loop return a matrix with the shock related entries and two matrices you can use to multiply with C2z0 instead of sorting the entries one by one. akin to a permutation matrix
# sparsify matrices
# check if SpeedMapping helps with the sylvester equations


####  Γ₄  
# create inputs from C2z0

U = spzeros(nx^3,nx^3)

for i=1:nx^2
    for k=1:nx
        U[(i-1)*nx+k,(k-1)*nx^2+i] = 1       
    end
end

P = kron(I(nx),U)


using Statistics, LinearAlgebra, StatsBase

# check distributional properties by simulating
shocks = randn(2,100000)
sim = get_irf(m, shocks = shocks, periods = 0, levels = true, algorithm = :pruned_second_order, initial_state = collect(get_SS(m, derivatives=false)))

[mean(i) for i in eachrow(sim[:,:,1])]
[Statistics.var(i) for i in eachrow(sim[:,:,1])]
[skewness(i) for i in eachrow(sim[:,:,1])]
[kurtosis(i) for i in eachrow(sim[:,:,1])]

(diag(C2y0))[[2,4,5,1,3]]

([reshape(C3y0,m.timings.nVars,m.timings.nVars,m.timings.nVars)[i,i,i] for i in 1:m.timings.nVars] ./ diag(C2y0).^(3/2))[[2,4,5,1,3]]


sim_lin = get_irf(m, shocks = shocks, periods = 0, levels = true, initial_state = collect(get_SS(m, derivatives=false)))

[mean(i) for i in eachrow(sim_lin[:,:,1])]
[Statistics.var(i) for i in eachrow(sim_lin[:,:,1])]
[skewness(i) for i in eachrow(sim_lin[:,:,1])]
[kurtosis(i) for i in eachrow(sim_lin[:,:,1])]

[mean(i) for i in eachrow(sim_lin[:,:,1])]


Statistics.var(sim, dims = 2)
diag(C2y0)
StatsBase.skewness(sim[1,:])
diag(reshape(C3y0,5,5,5))


[StatsBase.skewness(i) for i in eachrow(sim[:,:,1])]
[reshape(C3y0,5,5,5)[i,i,i] for i in 1:5]
[StatsBase.skewness(i) for i in eachrow(sim[:,:,1])]
[kurtosis(i) for i in eachrow(sim[:,:,1])]





# transition to third order pruned solution

varobs = [:YGR, :INFL, :INT]
T = m.timings
states = m.timings.past_not_future_and_mixed

nx = T.nPast_not_future_and_mixed
nu = T.nExo
ny = T.nVars


id1_xf       = 1:nx
id2_xs       = id1_xf[end]    .+ (1:nx)
id3_xf_xf    = id2_xs[end]    .+ (1:nx^2)
id4_xrd      = id3_xf_xf[end] .+ (1:nx)
id5_xf_xs    = id4_xrd[end]   .+ (1:nx^2)
id6_xf_xf_xf = id5_xf_xs[end] .+ (1:nx^3)
id1_u        = 1:nu
id2_u_u      = id1_u[end]       .+ (1:nu^2)
id3_xf_u     = id2_u_u[end]     .+ (1:nx*nu)
id4_u_xf     = id3_xf_u[end]    .+ (1:nx*nu)
id5_xs_u     = id4_u_xf[end]    .+ (1:nx*nu)   
id6_u_xs     = id5_xs_u[end]    .+ (1:nx*nu)  
id7_xf_xf_u  = id6_u_xs[end]    .+ (1:nx^2*nu)
id8_xf_u_xf  = id7_xf_xf_u[end] .+ (1:nx^2*nu)
id9_u_xf_xf  = id8_xf_u_xf[end] .+ (1:nx^2*nu)    
id10_xf_u_u  = id9_u_xf_xf[end] .+ (1:nx*nu^2)
id11_u_xf_u  = id10_xf_u_u[end] .+ (1:nx*nu^2)
id12_u_u_xf  = id11_u_xf_u[end] .+ (1:nx*nu^2)
id13_u_u_u   = id12_u_u_xf[end] .+ (1:nu^3)  


model_order = [:YGR,:INFL,:INT,:y, :R,:g,:z,:c,:dy,:p,:e_z, :e_g, :e_r]


declaration_order = [:c, :dy, :p, :y, :R, :g, :z, :YGR, :INFL, :INT]
indexin(declaration_order,m.var)

sol = get_solution(m,m.parameter_values, algorithm = :third_order)

hx = sol[2][indexin(intersect(model_order,states),m.var),indexin(intersect(model_order,states),states)]
hu = sol[2][indexin(intersect(model_order,states),m.var),((T.nPast_not_future_and_mixed + 1):end)[indexin(intersect(model_order,m.exo),m.exo)]] 
gx = sol[2][indexin(intersect(model_order,m.var),m.var),indexin(intersect(model_order,states),states)]
gu = sol[2][indexin(intersect(model_order,m.var),m.var),((T.nPast_not_future_and_mixed + 1):end)[indexin(intersect(model_order,m.exo),m.exo)]]

second_order_helper = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^2,4)
second_order_axis = vcat(T.past_not_future_and_mixed,:Volatility,T.exo)
k = 1
for i in second_order_axis
    for j in second_order_axis
        second_order_helper[k,:] = [j,i,k,string(i)*string(j)]
        k += 1
    end
end



second_order_helper_ordered = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^2,4)
second_order_axis_ordered = vcat(intersect(model_order,T.past_not_future_and_mixed),:Volatility,intersect(model_order,T.exo))
k = 1
for i in second_order_axis_ordered
    for j in second_order_axis_ordered
        second_order_helper_ordered[k,:] = [i,j,k,string(i)*string(j)]
        k += 1
    end
end



Hxx = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Huu = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Hxu = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
hss = sol[3][indexin(intersect(model_order,states),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]

Gxx = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Guu = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Gxu = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
gss = sol[3][indexin(intersect(model_order,m.var),m.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]



third_order_helper = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^3,5)
third_order_axis = vcat(T.past_not_future_and_mixed,:Volatility,T.exo)
k = 1
for i in third_order_axis
    for j in third_order_axis
        for l in third_order_axis
            third_order_helper[k,:] = [j,i,l,k,string(i)*string(j)*string(l)]
            k += 1
        end
    end
end


third_order_helper_ordered = Matrix(undef,(T.nPast_not_future_and_mixed+1+T.nExo)^3,5)
third_order_axis_ordered = vcat(intersect(model_order,T.past_not_future_and_mixed),:Volatility,intersect(model_order,T.exo))
k = 1
for i in third_order_axis_ordered
    for j in third_order_axis_ordered
        for l in third_order_axis_ordered
            third_order_helper_ordered[k,:] = [i,j,l,k,string(i)*string(j)*string(l)]
            k += 1
        end
    end
end



Hxxx = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (states,),5],third_order_helper[:,5]),4]]
Hxxu = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Hxuu = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Huuu = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Hxss = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Huss = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Hsss = sol[4][indexin(intersect(model_order,states),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .== :Volatility .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]


Gxxx = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (states,),5],third_order_helper[:,5]),4]]
Gxxu = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Gxuu = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Guuu = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Gxss = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Guss = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Gsss = sol[4][indexin(intersect(model_order,m.var),m.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .== :Volatility .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]


nz = 3*nx + 2*nx^2 +nx^3;
nxi = nu+nu^2+2*nx*nu+2*nx*nu+3*nx^2*nu+3*nu^2*nx+nu^3;    
nu2 = nu*(nu+1)/2; nx2 = nx*(nx+1)/2; nu3 = nu2*(nu+2)/3;
nximin = nu + nu2 + 2*nu*nx + nu*nx2 + nu2*nx + nu3;

M2u = vec(I(T.nExo))
M3u = zeros(nu^3)

hx_hx = kron(hx,hx); hu_hu = kron(hu,hu); hx_hu=kron(hx,hu); hu_hx = kron(hu,hx);
A = spzeros(nz,nz);
B = spzeros(nz,nxi);
C = spzeros(ny,nz);
D = spzeros(ny,nxi);
c = spzeros(nz,1);
d = spzeros(ny,1);

A[id1_xf,id1_xf] = hx;
A[id2_xs,id2_xs] = hx;
A[id2_xs,id3_xf_xf] = 1/2*Hxx;
A[id3_xf_xf,id3_xf_xf] = hx_hx;
A[id4_xrd,id1_xf] = 3/6*Hxss;
A[id4_xrd,id4_xrd] = hx;
A[id4_xrd,id5_xf_xs] = Hxx;
A[id4_xrd,id6_xf_xf_xf] = 1/6*Hxxx;
A[id5_xf_xs,id1_xf] = kron(hx,1/2*hss);
A[id5_xf_xs,id5_xf_xs] = hx_hx;
A[id5_xf_xs,id6_xf_xf_xf] = kron(hx,1/2*Hxx);
A[id6_xf_xf_xf,id6_xf_xf_xf] = kron(hx,hx_hx);

B[id1_xf,id1_u] = hu;    
B[id2_xs,id2_u_u] = 1/2*Huu;
B[id2_xs,id3_xf_u] = Hxu;    
B[id3_xf_xf,id2_u_u] = hu_hu;
B[id3_xf_xf,id3_xf_u] = hx_hu;
B[id3_xf_xf,id4_u_xf] = hu_hx;    
B[id4_xrd,id1_u] = 3/6*Huss;
B[id4_xrd,id5_xs_u] = Hxu;
B[id4_xrd,id7_xf_xf_u] = 3/6*Hxxu;
B[id4_xrd,id10_xf_u_u] = 3/6*Hxuu;
B[id4_xrd,id13_u_u_u] =  1/6*Huuu;    
B[id5_xf_xs,id1_u] = kron(hu,1/2*hss);
B[id5_xf_xs,id6_u_xs] =  hu_hx;
B[id5_xf_xs,id7_xf_xf_u] = kron(hx,Hxu);
B[id5_xf_xs,id9_u_xf_xf] = kron(hu,1/2*Hxx);
B[id5_xf_xs,id10_xf_u_u] = kron(hx,1/2*Huu);
B[id5_xf_xs,id11_u_xf_u] = kron(hu,Hxu);
B[id5_xf_xs,id13_u_u_u] = kron(hu,1/2*Huu);    
B[id6_xf_xf_xf,id7_xf_xf_u] =  kron(hx_hx,hu);
B[id6_xf_xf_xf,id8_xf_u_xf] = kron(hx,hu_hx);
B[id6_xf_xf_xf,id9_u_xf_xf] = kron(hu,hx_hx);
B[id6_xf_xf_xf,id10_xf_u_u] = kron(hx_hu,hu);
B[id6_xf_xf_xf,id11_u_xf_u] = kron(hu,hx_hu);
B[id6_xf_xf_xf,id12_u_u_xf] = kron(hu_hu,hx);
B[id6_xf_xf_xf,id13_u_u_u]  = kron(hu,hu_hu);         

C[1:ny,id1_xf] = gx+.5*Gxss;
C[1:ny,id2_xs] = gx;
C[1:ny,id3_xf_xf] = 0.5*Gxx;
C[1:ny,id4_xrd] = gx;
C[1:ny,id5_xf_xs] = Gxx;
C[1:ny,id6_xf_xf_xf] = 1/6*Gxxx;

D[1:ny,id1_u] = gu+.5*Guss;
D[1:ny,id2_u_u] = 0.5*Guu;
D[1:ny,id3_xf_u] = Gxu;
D[1:ny,id5_xs_u] = Gxu;    
D[1:ny,id7_xf_xf_u] = 1/2*Gxxu;
D[1:ny,id10_xf_u_u] = 1/2*Gxuu;
D[1:ny,id13_u_u_u] = 1/6*Guuu;

c[id2_xs,1] = 1/2*hss + 1/2*Huu*M2u;
c[id3_xf_xf,1] = hu_hu*M2u; 
c[id4_xrd,1] = 1/6*Huuu*M3u + 1/6*Hsss; 
c[id5_xf_xs,1] =  kron(hu,1/2*Huu)*M3u; 
c[id6_xf_xf_xf,1] = kron(hu_hu,hu)*M3u;

d[1:ny,1] = 0.5*gss + 0.5*Guu*M2u + 1/6*Guuu*M3u + 1/6*Gsss;


## First-order moments, ie expectation of variables
IminA = I-A;
Ez   = collect(IminA)\c;
Ey   = ybar + C*Ez+d; # recall y = yss + C*z + d



## Second-order moments
####  Γ₂
# nu = 2
# nx = 2
# write a loop to fill Γ₂
# size of input vector
n_entries = Int(nu + nu*(nu+1)/2 + nx*nu)
nz = 3*nx + 2*nx^2 +nx^3
nxi = nu + nu^2 + 2*nx*nu + 2*nx*nu + 3*nx^2*nu + 3*nu^2*nx + nu^3
nu2 = nu*(nu+1)/2 |> Int
nx2 = nx*(nx+1)/2 |> Int
nu3 = nu2*(nu+2)/3 |> Int
nximin = nu + nu2 + 2*nu*nx + nu*nx2 + nu2*nx + nu3
# nxi = nu + nu^2 + 3*nx*nu + 3*nu*nx^2 + 3*nu^2*nx + nu^3

col1_u       = 1:nu
col2_u_u     = col1_u[end]       .+ (1:nu2)
col3_xf_u    = col2_u_u[end]     .+ (1:nu*nx)
col4_xs_u    = col3_xf_u[end]    .+ (1:nu*nx)
col5_xf_xf_u = col4_xs_u[end]    .+ (1:nu*nx2)
col6_xf_u_u  = col5_xf_xf_u[end] .+ (1:nu2*nx)
col7_u_u_u   = col6_xf_u_u[end]  .+ (1:nu3)

row1_u       = 1:nu
row2_u_u     = row1_u[end]       .+ (1:nu^2)
row3_xf_u    = row2_u_u[end]     .+ (1:nx*nu)
row4_u_xf    = row3_xf_u[end]    .+ (1:nx*nu)
row5_xs_u    = row4_u_xf[end]    .+ (1:nx*nu)
row6_u_xs    = row5_xs_u[end]    .+ (1:nx*nu)
row7_xf_xf_u = row6_u_xs[end]    .+ (1:nu*nx^2)
row8_xf_u_xf = row7_xf_xf_u[end] .+ (1:nu*nx^2)
row9_u_xf_xf = row8_xf_u_xf[end] .+ (1:nu*nx^2)
row10_xf_u_u = row9_u_xf_xf[end] .+ (1:nx*nu^2)
row11_u_xf_u = row10_xf_u_u[end] .+ (1:nx*nu^2)
row12_u_u_xf = row11_u_xf_u[end] .+ (1:nx*nu^2)
row13_u_u_u  = row12_u_u_xf[end] .+ (1:nu^3)

DPx, DPxinv = duplication(nx)
DPu, DPuinv = duplication(nu)
TPu = triplication(nu)

K_u_x  = reshape(kron(vec(I(nu)), I(nx)), nu*nx, nu*nx)
K_u_xx = reshape(kron(vec(I(nu)), I(nx^2)), nu*nx^2, nu*nx^2)
K_u_xu = reshape(kron(vec(I(nu)), I(nu*nx)), nu^2*nx, nu^2*nx)
K_ux_x = reshape(kron(vec(I(nu*nx)), I(nx)), nu*nx^2, nu*nx^2)
K_uu_x = reshape(kron(vec(I(nu^2)), I(nx)), nu^2*nx, nu^2*nx)



# if sparseflag
#     Ix = speye(nx);
#     Iu = speye(nu); 
#     Iux = speye(nu*nx);
#     Fxi = spalloc(nxi,nximin,nu+nu^2+4*nu*nx+3*nx^2*nu+3*nx*nu^2+nu^3);
# else
Ix = I(nx)
Iu = I(nu)
Iux = I(nu*nx)
Fxi = spzeros(Bool,nxi,nximin)
# end
DPx_Iu = kron(DPx,Iu)
Ix_DPu = kron(Ix,DPu)

Fxi[row1_u,col1_u] = Iu
Fxi[row2_u_u,col2_u_u] = DPu
Fxi[row3_xf_u,col3_xf_u] = Iux
Fxi[row4_u_xf,col3_xf_u] = K_u_x
Fxi[row5_xs_u,col4_xs_u] = Iux
Fxi[row6_u_xs,col4_xs_u] = K_u_x
Fxi[row7_xf_xf_u,col5_xf_xf_u] = DPx_Iu
Fxi[row8_xf_u_xf,col5_xf_xf_u] = K_ux_x*DPx_Iu
Fxi[row9_u_xf_xf,col5_xf_xf_u] = K_u_xx*DPx_Iu
Fxi[row10_xf_u_u,col6_xf_u_u] = Ix_DPu
Fxi[row11_u_xf_u,col6_xf_u_u] = K_u_xu*Ix_DPu
Fxi[row12_u_u_xf,col6_xf_u_u] = K_uu_x*Ix_DPu
Fxi[row13_u_u_u,col7_u_u_u] = TPu




Γ₂ = spzeros(nximin, nximin)


u_u = reshape([[i, k] for k in m.timings.exo for i in m.timings.exo],nu,nu)

xf_u = reshape([[i, (k,1)] for k in m.timings.past_not_future_and_mixed for i in m.timings.exo],nx,nu)
# u_xf = reshape([(i, (k,1)) for i in m.timings.exo for k in m.timings.past_not_future_and_mixed],nu,nx)

xs_u = reshape([[i, (k,2)] for k in m.timings.past_not_future_and_mixed for i in m.timings.exo],nx,nu)
# u_xs = reshape([(i, (k,2)) for i in m.timings.exo for k in m.timings.past_not_future_and_mixed],nu,nx)

xf_xf_u = reshape([[i, (k,1), (j,1)] for j in m.timings.past_not_future_and_mixed for k in m.timings.past_not_future_and_mixed for i in m.timings.exo],nx, nx, nu)
# xf_u_xf = reshape([(i, (k,1), (j,1)) for j in m.timings.past_not_future_and_mixed for i in m.timings.exo for k in m.timings.past_not_future_and_mixed],nx, nu, nx)
# u_xf_xf = reshape([(i, (k,1), (j,1)) for i in m.timings.exo for j in m.timings.past_not_future_and_mixed for k in m.timings.past_not_future_and_mixed],nu, nx, nx)

xf_u_u = reshape([[i, k, (j,1)] for k in m.timings.exo for i in m.timings.exo for j in m.timings.past_not_future_and_mixed],nx, nu, nu)
# u_xf_u = reshape([(i, k, (j,1)) for k in m.timings.exo for j in m.timings.past_not_future_and_mixed for i in m.timings.exo],nx, nu, nu)
# u_u_xf = reshape([(i, k, (j,1)) for k in m.timings.exo for i in m.timings.exo for j in m.timings.past_not_future_and_mixed],nx, nu, nu)

u_u_u = reshape([[i, k, j] for k in m.timings.exo for i in m.timings.exo for j in m.timings.exo],nu, nu, nu)


inputs = vcat(m.timings.exo, upper_triangle(u_u), vec(xf_u), vec(xs_u), upper_triangle(xf_xf_u), upper_triangle(xf_u_u, alt = true), upper_triangle(u_u_u, triple = true))


# ximin = [u; upper_triangle(u_u);   xf_u;   xs_u;   kron(DPxinv,I_nu)*xf_xf_u;   kron(I_nx,DPuinv)*xf_u_u;   TPuinv*u_u_u-E_uuu]

# K = reshape([(:ϵx, (k,i)) for k in 1:nx for i in 1:nx],nx,nx)

# inputs = vcat([(:ϵ, i) for i in 1:nu], upper_triangle(u_u), vec(K))

n_shocks = Int(nu + nu * (nu + 1) / 2)

for (i¹,s¹) in enumerate(inputs)
    for (i²,s²) in enumerate(inputs)
        # for (i³,s³) in enumerate(inputs)
            combo = [(s¹ isa Symbol ? [s¹] : s¹)... , (s² isa Symbol ? [s²] : s²)...]
            combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])

            intrsct = intersecting_elements(s¹,s²)
            intrsct_cnts = Dict([element => count(==(element),intrsct) for element in unique(intrsct)])

            if any([k ∈ m.timings.exo && v == 1 for (k,v) in combo_cnts])
                continue
            elseif all([k ∈ m.timings.exo && v == 2 for (k,v) in combo_cnts]) && 
                all([k ∈ m.timings.exo && v == 1 for (k,v) in intrsct_cnts]) && 
                length(intrsct_cnts) > 0

                Γ₂[i¹,i²] = 1

            elseif all([k ∈ m.timings.exo && v == 4 for (k,v) in combo_cnts]) && 
                all([k ∈ m.timings.exo && v == 2 for (k,v) in intrsct_cnts])

                Γ₂[i¹,i²] = 2

            elseif length(setdiff(keys(combo_cnts),m.timings.exo)) == 0 && 
                length(intrsct_cnts) > 0 && 
                all([intrsct_cnts[i] > 0 for i in collect(intersect(keys(combo_cnts),keys(intrsct_cnts)))]) && 
                any([combo_cnts[i] == 4 for i in collect(intersect(keys(combo_cnts),keys(intrsct_cnts)))])

                Γ₂[i¹,i²] = 3

            elseif all([k ∈ m.timings.exo && v == 6 for (k,v) in combo_cnts]) && 
                all([k ∈ m.timings.exo && v == 3 for (k,v) in intrsct_cnts])

                Γ₂[i¹,i²] = 15
                
            elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && 
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 2 && 
                all([i[2] == 1 for i in keys(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))]) && 
                all([k ∈ m.timings.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)])

                if all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
                    indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    Γ₂[i¹,i²] = C2z0[idxs[1], idxs[2]]
                elseif all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
                    indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    Γ₂[i¹,i²] = 3 * C2z0[idxs[1], idxs[2]]
                end

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 2 && # non shocks have max two entries
                all([i[2] == 2 for i in keys(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))]) && # non shocks are all double entries
                all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

                vars = setdiff(keys(combo_cnts), m.timings.exo)
                indices_mat = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in vars] .+ m.timings.nPast_not_future_and_mixed

                idxs = length(indices_mat) == 1 ? [indices_mat[1],indices_mat[1]] : indices_mat
                    
                indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in vars]

                idxs2 = length(indices) == 1 ? [indices[1],indices[1]] : indices
            
                Γ₂[i¹,i²] = C2z0[idxs[1], idxs[2]] + Ey[idxs2[1]] * Ey[idxs2[2]]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
                length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2, combo_cnts)) == 1 &&
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) > 2 && # non shocks have more than two entries
                all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

                indices_second_mean = [indexin([i[1][1]], intersect(model_order,m.var))[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && u == 1 && j[2] == 2, combo_cnts)][1]

                indices_first_variance = [indexin([i[1][1]], m.timings.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1, combo_cnts)]
                
                indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

                indices_first = (indices_first_variance[1] - 1) * m.timings.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * m.timings.nPast_not_future_and_mixed

                indices_second = [indexin([i[1][1]], m.timings.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && u == 1 && j[2] == 2, combo_cnts)][1] + m.timings.nPast_not_future_and_mixed

                Γ₂[i¹,i²] = C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 4 && # non shocks have four entries
                all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

                vars1 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s¹)]
                vars2 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s²)]
                
                if vars1 == vars2
                    Γ₂[i¹,i²] = 
                    C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]
                else
                    Γ₂[i¹,i²] = 
                    C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
                    C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]]
                end

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && 
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2, combo_cnts))) == 1 &&  
                sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1, combo_cnts))) == 0 &&  
                all([k ∈ m.timings.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)])

                indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in setdiff(keys(combo_cnts), m.timings.exo)][1]
                
                if all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
                    Γ₂[i¹,i²] = 3 * Ey[indices]
                elseif all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
                    Γ₂[i¹,i²] = Ey[indices]
                end
            end
    end
end

Γ₂


BFxi = B*Fxi
DFxi = D*Fxi

CC = BFxi *  Γ₂  * BFxi'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + DFxi * Γ₂ * DFxi'




# Γ₂[1:10,1:10]
# Γ₂[1:2,20:end]
# Γ₂[20:end,1:2]
# Γ₂[26:end,26:end]
# findnz(Γ₂)
# Γ₂xi = reshape(Γ₂, n_entries^2, n_entries)
# intrsct = findall(in(inputs[3] isa  Symbol ? [inputs[3]] : inputs[3]),inputs[26])
# (inputs[10]...,inputs[10]...)
# findall(in(b),a)

# s¹ = inputs[10]
# s² = inputs[1]
# # C2z0[2,2]
# combo = [(s¹ isa Symbol ? [s¹] : s¹)... , (s² isa Symbol ? [s²] : s²)...]
# combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])

# intrsct = intersecting_elements(s¹,s²)
# intrsct_cnts = Dict([element => count(==(element),intrsct) for element in unique(intrsct)])

# C2z0[3:4,3:4] + Ey[2:3]*Ey[2:3]'

# vars1 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s¹)]
# vars2 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s²)]

# C2z0[vars1[1], vars2[1]]^2 + C2z0[vars1[1], vars2[2]]^2 + C2z0[vars1[2], vars2[2]]^2

# C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]]


# C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] 
# C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]
# C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]


# var_combos = vcat.(vars[1:end-1], vars[2:end])

# sum([C2z0[i[1],i[2]]^2 for i in var_combos])

# kron(C2z0[1:2,1:2],C2z0[1:2,1:2])

# kron(C2z0[1:2,1:2],C2z0[1:2,1:2])[[2,3,5]]|>sum

# vcat(s¹,s²)

# kron(C2z0[1:2,1:2],C2z0[1:2,1:2])[[4,6,7]]|>sum


# vars

# C2z0[1,1] * C2z0[2,2] + C2z0[1,2] * C2z0[1,2] + C2z0[1,2] * C2z0[2,1]


# indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]

# length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 
# all([i[2] == 2 for i in keys(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))])
# sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 2
# all([k ∈ m.timings.exo && v == 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)])
# all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])


# all([k ∈ m.timings.exo && v == 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) && all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])


# intrsct_cnts[only(intersect(keys(combo_cnts),keys(intrsct_cnts)))]

# intrsct_cnts|>length
# intersecting_elements(collect(inputs[5]),collect(inputs[29]))
# # define your arrays
# array1 = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
# array2 = [2, 2, 4, 4, 5, 5, 5, 5]

# # get common elements
# common = intersecting_elements(array1, array2)

# # print result
# println(common)








# BFxi = B*Fxi
# DFxi = D*Fxi

# CC = BFxi *  Γ₂  * BFxi'

# lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

# C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

# C2y0 = C * C2z0 * C' + DFxi * Γ₂ * DFxi'




## Third-order moments
####  Γ₃

Γ₃ = zeros(nximin, nximin, nximin);

# covars
s¹ = inputs[1]
s² = inputs[3]
s³ = inputs[10]

s¹ = inputs[2]
s² = inputs[5]
s³ = inputs[11]

s¹ = inputs[28]
s² = inputs[5]
s³ = inputs[10]

s¹ = inputs[27]
s² = inputs[5]
s³ = inputs[11]

s¹ = inputs[1]
s² = inputs[4]
s³ = inputs[11]

s¹ = inputs[1]
s² = inputs[14]
s³ = inputs[3]

s¹ = inputs[1]
s² = inputs[16]
s³ = inputs[3]

s¹ = inputs[1]
s² = inputs[15]
s³ = inputs[4]

s¹ = inputs[1]
s² = inputs[17]
s³ = inputs[4]

s¹ = inputs[1]
s² = inputs[25]
s³ = inputs[6]
# no
s¹ = inputs[1]
s² = inputs[11]
s³ = inputs[4]

s¹ = inputs[1]
s² = inputs[5]
s³ = inputs[10]


s¹ = inputs[26]
s² = inputs[5]
s³ = inputs[10]

s¹ = inputs[1]
s² = inputs[16]
s³ = inputs[1]

s¹ = inputs[1]
s² = inputs[14]
s³ = inputs[5]


# s¹ = inputs[1]
# s² = inputs[14]
# s³ = inputs[3]

s¹ = inputs[1]
s² = inputs[17]
s³ = inputs[4]

s¹ = inputs[1]
s² = inputs[6]
s³ = inputs[25]

s¹ = inputs[1]
s² = inputs[5]
s³ = inputs[16]

combo = [(s¹ isa Symbol ? [s¹] : s¹)... , (s² isa Symbol ? [s²] : s²)..., (s³ isa Symbol ? [s³] : s³)...]
combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])

intrsct = []
push!(intrsct, intersecting_elements(s¹,s²))
push!(intrsct, intersecting_elements(s¹,s³))
push!(intrsct, intersecting_elements(s²,s³))

intrsct_unique_cnts = Dict([(element,v) => count(==(element),i) for (v,i) in enumerate(intrsct) for element in unique(i)])
intrsct_cnts_raw = [element => count(==(element),i) for (v,i) in enumerate(intrsct) for element in unique(i)]
intrsct_cnts = Dict(intrsct_cnts_raw)
intrsct_cnts_type = Dict([(element[1],element[2]) => count(==(element),intrsct_cnts_raw) for element in unique(intrsct_cnts_raw)])

S = []
push!(S, s¹ isa Symbol ? [s¹] : s¹)
push!(S, s² isa Symbol ? [s²] : s²)
push!(S, s³ isa Symbol ? [s³] : s³)

s = Dict([(element,v) => count(==(element),i) for (v,i) in enumerate(S) for element in unique(i)])

any([i ∈ keys(intrsct_cnts_type) for i in keys(filter(((j,u),) -> j[1] ∈ m.timings.exo && j[2] == 2 && u == 2, s))])
any([i ∈ keys(filter(((j,u),) -> j[1] ∈ m.timings.exo && sum(u) == 2 && u == 2, s)) for i in keys(filter(((j,u),) -> j[1] ∈ m.timings.exo, intrsct_cnts_type))])

# filter(i -> length(i) == 2 && all([ii ∈ m.timings.exo for ii in i]), S)[1] |> unique
length(S[collect(keys(filter(((j,u),) -> j[1] ∈ m.timings.exo && sum(u) == 2 && u == 2, s)))[1][2]]) > 2


shock_indices = [1:nu + nu2..., nximin - nu3 + 1:nximin...]

for (i¹,s¹) in enumerate(inputs)
    for (i²,s²) in enumerate(inputs)
        for (i³,s³) in enumerate(inputs)

            combo = [(s¹ isa Symbol ? [s¹] : s¹)... , (s² isa Symbol ? [s²] : s²)..., (s³ isa Symbol ? [s³] : s³)...]
            combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])
            
            intrsct = []
            push!(intrsct, intersecting_elements(s¹,s²))
            push!(intrsct, intersecting_elements(s¹,s³))
            push!(intrsct, intersecting_elements(s²,s³))

            intrsct_unique_cnts = Dict([(element,v) => count(==(element),i) for (v,i) in enumerate(intrsct) for element in unique(i)])
            intrsct_cnts_raw = [element => count(==(element),i) for (v,i) in enumerate(intrsct) for element in unique(i)]
            intrsct_cnts = Dict(intrsct_cnts_raw)
            intrsct_cnts_type = Dict([(element[1],element[2]) => count(==(element),intrsct_cnts_raw) for element in unique(intrsct_cnts_raw)])

            S = []
            push!(S, s¹ isa Symbol ? [s¹] : s¹)
            push!(S, s² isa Symbol ? [s²] : s²)
            push!(S, s³ isa Symbol ? [s³] : s³)

            s = Dict([(element,v) => count(==(element),i) for (v,i) in enumerate(S) for element in unique(i)])

            if any([k ∈ m.timings.exo && v == 1 for (k,v) in combo_cnts])
                continue
            elseif i¹ ∈ shock_indices && i² ∈ shock_indices && i³ ∈ shock_indices
                if (length(filter(((j,u),) -> u == 2, s)) == 2  || length(filter(((j,u),) -> (u == 3 && j[2] == 1) || (u == 2 && j[2] >= 1) || (u == 1 && j[2] >= 2), intrsct_cnts_type)) > 0) && 
                    (length(s) <= 4 || (i¹ == i² || i¹ == i³ || i² == i³) && !(i¹ == i² && i¹ == i³)) && 
                    !all(values(s) .== 2) &&
                    sum(values(combo_cnts)) <= 6 &&
                    length(filter(((j,u),) -> j ∈ m.timings.exo && u == 4, combo_cnts)) > 0 &&
                    sum([i[2] for i in s]) % 2 == 0

                    Γ₃[i¹,i²,i³] = 2

                elseif all([k ∈ m.timings.exo && v == 2 for (k,v) in combo_cnts]) &&
                length(intrsct_cnts_type) == 2

                    Γ₃[i¹,i²,i³] = 1

                elseif i¹ == i² && i¹ == i³ && sum(values(combo_cnts)) == 6

                    Γ₃[i¹,i²,i³] = 8 # Variance of ϵ²

                elseif (length(filter(((j,u),) -> u == 2, s)) == 2  || length(filter(((j,u),) -> (u == 3 && j[2] == 1) || (u == 2 && j[2] >= 1) || (u == 1 && j[2] >= 2), intrsct_cnts_type)) > 0 || length(filter(((j,u),) -> (u == 1 && j[2] == 1), intrsct_cnts_type)) >= 2) && 
                    # (length(s) <= 4 || (i¹ == i² || i¹ == i³ || i² == i³) && !(i¹ == i² && i¹ == i³)) && 
                    (any(values(s) .== 3) || any(values(intrsct_cnts_type) .== 3)) &&
                    sum(values(combo_cnts)) <= 6 &&
                    length(filter(((j,u),) -> j ∈ m.timings.exo && u == 4, combo_cnts)) > 0 &&
                    sum([i[2] for i in s]) % 2 == 0

                    Γ₃[i¹,i²,i³] = 3

                elseif any(values(combo_cnts) .== 6) && length(intrsct_cnts_type) == 2 && !(i¹ == i² && i¹ == i³)

                    Γ₃[i¹,i²,i³] = 12 # Variance of ϵ²

                elseif all(values(combo_cnts) .== 4) && any(values(s) .== 2) && !(all(values(intrsct_cnts_type) .== 3))

                    Γ₃[i¹,i²,i³] = 6

                elseif all(values(combo_cnts) .== 4) && (any(values(s) .== 3) || all(values(intrsct_cnts_type) .== 3))

                    Γ₃[i¹,i²,i³] = 9

                elseif sort(collect(values(combo_cnts))) == [2,6] && length(s) > 3
                # elseif all([k ∈ m.timings.exo && v == 6 for (k,v) in combo_cnts]) && 
                #     all([ii[1] ∈ m.timings.exo && v == 3 for ii in intrsct_cnts])
                    Γ₃[i¹,i²,i³] = 15

                elseif all(values(combo_cnts) .== 8)

                    Γ₃[i¹,i²,i³] = 90

                end


            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u == 2, combo_cnts)) == 2 && 
                length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2 && u == 1, combo_cnts)) == 1 &&
                length(combo_cnts) == 3 &&
                length(intrsct_cnts_type) == 2

                indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in setdiff(keys(combo_cnts), m.timings.exo)][1]

                Γ₃[i¹,i²,i³] = Ey[indices]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u ∈ [2,4], combo_cnts)) == 2 && 
                length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2 && u == 1, combo_cnts)) == 1 &&
                length(combo_cnts) == 3 &&
                sum(values(intrsct_unique_cnts)) == 3

                indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in setdiff(keys(combo_cnts), m.timings.exo)][1]

                Γ₃[i¹,i²,i³] = 2 * Ey[indices]#C2z0[idxs[1], idxs[2]]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u == 4, combo_cnts)) == 1 && 
                length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2 && u == 1, combo_cnts)) == 1 &&
                length(combo_cnts) == 2
                # sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 2 && 
                # all([i[2] == 1 for i in keys(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))]) && 
                # all([k ∈ m.timings.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)])

                # if all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
                    # indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                    # idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in setdiff(keys(combo_cnts), m.timings.exo)][1]

                    Γ₃[i¹,i²,i³] = 2 * Ey[indices]#C2z0[idxs[1], idxs[2]]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u ∈ [2,4], combo_cnts)) == 2 && 
                ((length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 2, combo_cnts)) == 1 &&
                length(combo_cnts) == 3) || 
                (length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 1, combo_cnts)) == 2 &&
                length(combo_cnts) == 4)) &&
                sum(values(intrsct_unique_cnts)) == 3

                indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                
                Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1], idxs[2]]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u == 4, combo_cnts)) == 1 && 
                ((length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 2, combo_cnts)) == 1 &&
                length(combo_cnts) == 2) || 
                (length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 1, combo_cnts)) == 2 &&
                    length(combo_cnts) == 3))

                indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                
                Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1], idxs[2]]

            elseif length(filter(((j,u),) -> j ∈ m.timings.exo && u == 2, combo_cnts)) == 2 && 
                ((length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 2, combo_cnts)) == 1 &&
                length(combo_cnts) == 3 && length(intrsct_cnts_type) == 2 && length(s) == 5)  || 
                (length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1 && u == 1, combo_cnts)) == 2 &&
                length(combo_cnts) == 4 && ((length(intrsct_cnts_type) == 1 && length(s) == 4) || (length(intrsct_cnts_type) == 2 && length(s) == 6))))# &&
                # ((length(intrsct_cnts_type) == 2 && length(s) == 6) ||
                # (length(intrsct_cnts_type) == 1 && length(s) == 5))# && 
                # filter(((j,u),) -> j[1] ∈ m.timings.exo && sum(u) == 2 && u == 2, s) == 0))
                # length(S[collect(keys(filter(((j,u),) -> j[1] ∈ m.timings.exo && sum(u) == 2 && u == 2, s)))[1][2]]) != 2))# && !(i¹ ∈ nu.+(1:nu2) || i² ∈ nu.+(1:nu2) || i³ ∈ nu.+(1:nu2)))

                indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
                idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                
                Γ₃[i¹,i²,i³] = C2z0[idxs[1], idxs[2]]

            #     elseif all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
            #         indices = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in setdiff(keys(combo_cnts), m.timings.exo)]
    
            #         idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
            #         Γ₃[i¹,i²,i³] = 3 * C2z0[idxs[1], idxs[2]]
                # end

            # elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
            #     sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 2 && # non shocks have max two entries
            #     all([i[2] == 2 for i in keys(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))]) && # non shocks are all double entries
            #     all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

            #     vars = setdiff(keys(combo_cnts), m.timings.exo)
            #     indices_mat = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in vars] .+ m.timings.nPast_not_future_and_mixed

            #     idxs = length(indices_mat) == 1 ? [indices_mat[1],indices_mat[1]] : indices_mat
                    
            #     indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in vars]

            #     idxs2 = length(indices) == 1 ? [indices[1],indices[1]] : indices
            
            #     Γ₃[i¹,i²,i³] = C2z0[idxs[1], idxs[2]] + Ey[idxs2[1]] * Ey[idxs2[2]]

            # elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
            #     length(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2, combo_cnts)) == 1 &&
            #     sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) > 2 && # non shocks have more than two entries
            #     all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

            #     indices_second_mean = [indexin([i[1][1]], intersect(model_order,m.var))[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && u == 1 && j[2] == 2, combo_cnts)][1]

            #     indices_first_variance = [indexin([i[1][1]], m.timings.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1, combo_cnts)]
                
            #     indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

            #     indices_first = (indices_first_variance[1] - 1) * m.timings.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * m.timings.nPast_not_future_and_mixed

            #     indices_second = [indexin([i[1][1]], m.timings.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ m.timings.exo) && u == 1 && j[2] == 2, combo_cnts)][1] + m.timings.nPast_not_future_and_mixed

            #     Γ₃[i¹,i²,i³] = C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean]

            # elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && # at least one shock 
            #     sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo), combo_cnts))) == 4 && # non shocks have four entries
            #     all([k ∈ m.timings.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)]) # all shocks appear in both entries

            #     vars1 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s¹)]
            #     vars2 = [indexin([i[1]], m.timings.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ m.timings.exo), s²)]
                
            #     if vars1 == vars2
            #         Γ₃[i¹,i²,i³] = 
            #         C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
            #         C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
            #         C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]
            #     else
            #         Γ₃[i¹,i²,i³] = 
            #         C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
            #         C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
            #         C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]]
            #     end

            # elseif length(filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)) > 0 && 
            #     sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 2, combo_cnts))) == 1 &&  
            #     sum(values(filter(((j,u),) -> !(j ∈ m.timings.exo) && j[2] == 1, combo_cnts))) == 0 &&  
            #     all([k ∈ m.timings.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ m.timings.exo, intrsct_cnts)])

            #     indices = [indexin([i[1]], intersect(model_order,m.var))[1] for i in setdiff(keys(combo_cnts), m.timings.exo)][1]
                
            #     if all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
            #         Γ₃[i¹,i²,i³] = 3 * Ey[indices]
            #     elseif all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ m.timings.exo, combo_cnts)])
            #         Γ₃[i¹,i²,i³] = Ey[indices]
            #     end
            end


            # indices = Set()
            # indices_x1 = Set()
            # indices_x2 = Set()

            # n_x = 0
            # n_ϵ2 = 0
            # n_same_indices_within_x = 0
            # n_same_indices_within_ϵ = 0

            # if s¹[1] == :x
            #     push!(indices_x1,s¹[2][1])
            #     push!(indices_x2,s¹[2][2])

            #     if s¹[2][1] == s¹[2][2]
            #         n_same_indices_within_x += 1
            #     end
            #     n_x += 1
            # else
            #     if s¹[2] isa Tuple
            #         if s¹[2][1] == s¹[2][2]
            #             n_same_indices_within_ϵ += 1
            #         end
            #         n_ϵ2 += 1
            #     end
            # end

            # if s²[1] == :x
            #     push!(indices_x1,s²[2][1])
            #     push!(indices_x2,s²[2][2])

            #     if s²[2][1] == s²[2][2]
            #         n_same_indices_within_x += 1
            #     end
            #     n_x += 1
            # else
            #     if s²[2] isa Tuple
            #         if s²[2][1] == s²[2][2]
            #             n_same_indices_within_ϵ += 1
            #         end
            #         n_ϵ2 += 1
            #     end
            # end

            # if s³[1] == :x
            #     push!(indices_x1,s³[2][1])
            #     push!(indices_x2,s³[2][2])

            #     if s³[2][1] == s³[2][2]
            #         n_same_indices_within_x += 1
            #     end
            #     n_x += 1
            # else
            #     if s³[2] isa Tuple
            #         if s³[2][1] == s³[2][2]
            #             n_same_indices_within_ϵ += 1
            #         end
            #         n_ϵ2 += 1
            #     end
            # end

            # n_same_indices_within = n_same_indices_within_ϵ + n_same_indices_within_x

            # n_same_indices_across = s¹[2] == s²[2] || s¹[2] == s³[2] || s³[2] == s²[2]

            # for k in s¹[2]
            #     push!(indices,k)
            # end
            # for k in s²[2]
            #     push!(indices,k)
            # end
            # for k in s³[2]
            #     push!(indices,k)
            # end

            # if n_x == 2 && n_same_indices_within_ϵ == 1 && s¹[2][2] == s²[2][2] && s²[2][2] == s³[2][2] #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
            #     idxs = collect(indices_x1)

            #     if length(idxs) == 1
            #         Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[1]]
            #     else
            #         Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[2]]
            #     end
            # end

            # if n_x == 2 && n_ϵ2 == 1 && n_same_indices_within_ϵ == 0 && length(collect(indices_x2)) == 2 #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
            #     idxs = collect(indices_x1)

            #     if length(idxs) == 1
            #         Γ₃[i¹,i²,i³] = C2z0[idxs[1],idxs[1]]
            #     else
            #         Γ₃[i¹,i²,i³] = C2z0[idxs[1],idxs[2]]
            #     end
            # end
        end
    end
end

Γ₃

Γ₃[:,:,1]#[16,5]
gamma3xi[:,:,1]#[17,4]
GAMMA3Xi = gamma3["GAMMA3XI"]
Γ₃xi = reshape(Γ₃,length(inputs)^2,length(inputs))



BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)


BFxi = B*Fxi
DFxi = D*Fxi

CkronC = kron(C,C)
BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)

CC = BFxikronBFxi *  Γ₃xi  * BFxi'
AA = kron(A,A)
lm = LinearMap{Float64}(x -> AA * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C3z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
reshape(C3z0,8,8,8)

C3y0 = CkronC * C3z0 * C' + DFxikronDFxi * Γ₃xi * DFxi'
reshape(C3y0,5,5,5)








# write a loop to fill Γ₄
# size of input vector

n_entries = Int(nu + nu*(nu+1)/2 + nx*nu)
Γ₄ = zeros(n_entries, n_entries, n_entries)

Ε = reshape([(:ϵ, (i,k)) for k in 1:nu for i in 1:nu],nu,nu)

K = reshape([(:x, (k,i)) for k in 1:nx for i in 1:nx],nx,nx)

inputs = vcat([(:ϵ, i) for i in 1:nu], upper_triangle(Ε), vec(K))

n_shocks = Int(nu + nu * (nu + 1) / 2)

for (i¹,s¹) in enumerate(inputs)
    for (i²,s²) in enumerate(inputs)
        for (i³,s³) in enumerate(inputs)
            indices = Set()
            indices_x1 = Set()
            indices_x2 = Set()

            n_x = 0
            n_ϵ2 = 0
            n_same_indices_within_x = 0
            n_same_indices_within_ϵ = 0

            if s¹[1] == :x
                push!(indices_x1,s¹[2][1])
                push!(indices_x2,s¹[2][2])

                if s¹[2][1] == s¹[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s¹[2] isa Tuple
                    if s¹[2][1] == s¹[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            if s²[1] == :x
                push!(indices_x1,s²[2][1])
                push!(indices_x2,s²[2][2])

                if s²[2][1] == s²[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s²[2] isa Tuple
                    if s²[2][1] == s²[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            if s³[1] == :x
                push!(indices_x1,s³[2][1])
                push!(indices_x2,s³[2][2])

                if s³[2][1] == s³[2][2]
                    n_same_indices_within_x += 1
                end
                n_x += 1
            else
                if s³[2] isa Tuple
                    if s³[2][1] == s³[2][2]
                        n_same_indices_within_ϵ += 1
                    end
                    n_ϵ2 += 1
                end
            end

            n_same_indices_within = n_same_indices_within_ϵ + n_same_indices_within_x

            n_same_indices_across = s¹[2] == s²[2] || s¹[2] == s³[2] || s³[2] == s²[2]

            for k in s¹[2]
                push!(indices,k)
            end
            for k in s²[2]
                push!(indices,k)
            end
            for k in s³[2]
                push!(indices,k)
            end

            if s¹[1] == s²[1] && s¹[1] == s³[1] && s¹[1] == :ϵ
                if (i¹ == i² || i¹ == i³ || i² == i³) && !(i¹ == i² && i¹ == i³)
                    if indices |> length == 1 && n_ϵ2 < 2#  || n_same_indices_across == 2)
                        Γ₄[i¹,i²,i³] = 2
                    end

                    if n_ϵ2 == 3 && n_same_indices_across == true && n_same_indices_within == 1
                        Γ₄[i¹,i²,i³] = 2
                    end
                end

                if i¹ == i² && i¹ == i³
                    if s¹[2] isa Tuple
                        if s¹[2][1] == s¹[2][2]
                            Γ₄[i¹,i²,i³] = 8 # Variance of ϵ²
                        end
                    end
                end

                if n_ϵ2 == 1 && n_same_indices_across == false && n_same_indices_within == 0 && indices |> length == 2
                    Γ₄[i¹,i²,i³] = 1
                end
            end

            if n_x == 2 && n_same_indices_within_ϵ == 1 && s¹[2][2] == s²[2][2] && s²[2][2] == s³[2][2] #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
                idxs = collect(indices_x1)

                if length(idxs) == 1
                    Γ₄[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[1]]
                else
                    Γ₄[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[2]]
                end
            end

            if n_x == 2 && n_ϵ2 == 1 && n_same_indices_within_ϵ == 0 && length(collect(indices_x2)) == 2 #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
                idxs = collect(indices_x1)

                if length(idxs) == 1
                    Γ₄[i¹,i²,i³] = C2z0[idxs[1],idxs[1]]
                else
                    Γ₄[i¹,i²,i³] = C2z0[idxs[1],idxs[2]]
                end
            end
        end
    end
end

Γ₄

Γ₄xi = reshape(Γ₄,n_entries^2,n_entries)



BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)


BFxi = B*Fxi
DFxi = D*Fxi

CkronC = kron(C,C)
BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)

CC = BFxikronBFxi *  Γ₄xi  * BFxi'
AA = kron(A,A)
lm = LinearMap{Float64}(x -> AA * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C3z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
reshape(C3z0,8,8,8)

C3y0 = CkronC * C3z0 * C' + DFxikronDFxi * Γ₄xi * DFxi'
reshape(C3y0,5,5,5)









function upper_triangle_vector_index_to_matrix_index(idx::Int, len::Int)
    # Determine the size of the matrix
    n = Int((-1 + sqrt(1 + 8*len)) / 2)
    
    # Calculate the row and column indices
    row = Int(ceil((sqrt(8*idx + 1) - 1) / 2))
    col = idx - (row*(row - 1)) ÷ 2

    return (row, col)
end

function upper_triangle_vector_index_to_matrix_index(idx::Int, len::Int)
    # Determine the size of the matrix
    n = Int((-1 + sqrt(1 + 8*len)) / 2)

    # Calculate the row and column indices
    row = n - Int(ceil(sqrt(2*(n+1)*(n+1) - 8*(len - idx))))
    col = idx + row*(row-1) ÷ 2 - ((n*(n+1)) ÷ 2 - len)

    if row == col
        # Diagonal element, only appears once
        return [(row, col)]
    else
        # Off-diagonal element, appears twice
        return [(row, col), (col, row)]
    end
end



upper_triangle_vector_index_to_matrix_index(1,6)



function vector_to_symmetric_matrix(vec::Array{Int, 1})
    # Check if the vector length is a triangular number
    n = round(Int, (-1 + sqrt(1 + 8*length(vec))) / 2)

    @assert n*(n+1)/2 == length(vec) "The length of the input vector is not valid to form a square symmetric matrix."

    # Initialize a square matrix with zeros
    mat = zeros(Int, n, n)

    # Fill the matrix's upper triangle and mirror it to the lower triangle
    idx = 1
    for i in 1:n
        for j in i:n
            mat[i, j] = vec[idx]
            mat[j, i] = vec[idx]
            idx += 1
        end
    end
    return mat
end

vector_to_symmetric_matrix([1,2,3,4,5,6])


function vec_to_mat_pos(pos::Int, vec_len::Int)
    # Check if the vector length is a triangular number
    n = round(Int, (-1 + sqrt(1 + 8*vec_len)) / 2)

    @assert n*(n+1)/2 == vec_len "The length of the input vector is not valid to form a square symmetric matrix."

    @assert pos >= 1 && pos <= vec_len "Invalid position in the vector."

    # Find the corresponding position in the symmetric matrix
    # i = 0
    # while pos > (i*(i+1))/2
    #     i += 1
    # end
    # j = pos - Int((i*(i-1))/2)
    i = 1
    while pos > i
        pos -= i
        i += 1
    end
    j = pos

    if i == j
        return (i, j)
    else
        return (i, j), (j, i)
    end
end

vec_to_mat_pos(3,6)
ones(3,3)
GAMMA2XI

position_in_symmetric_matrix(5,6)

position_in_symmetric_matrix(10,10)



nx
x¹ = 1:nx
x² = 1:nx
ϵ¹ = 1:nu
ϵ² = 1:nu


filler = fill(0.0,nu*nx, nu*nx)
for (ix1, x1) in enumerate(x¹)
    for (ix2, x2) in enumerate(x²)
        for (ie, e) in enumerate(ϵ¹)
            filler[nx * (ix2 - 1) + ie, nx * (ix1 - 1) + ie] = C2z0[ix1,ix2]
        end
    end
end



translate_mod_file("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/RBCmodel.mod")
include("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/RBCmodel.jl")

get_SS(RBCmodel)
get_SSS(RBCmodel)

get_solution(RBCmodel)
get_solution(RBCmodel, algorithm = :second_order)


shocks = [-0.981766231206793	0.00566920780391355	-0.267932340906166	-0.545427805362502	1.25853326534101	0.424036915280029	-0.204214677344615	0.994818547445083	-0.0798824440178837	-0.934560734112974	1.28670504067155	0.421802419436837	-0.743660346405064	-0.862780623456242	-1.09065208887269	1.83304107380247	-1.28689389412790	-0.310768858770842	0.974108126967603	-1.38740865322850	-0.836604458917015	-1.35214515200421	2.02759728776116	-0.137302885673647	-0.903074835815232	1.48259088418515	-0.310032509481618	0.584990246466085	-1.56031081285004	-1.65275382641708	-0.698239086847836	0.112953728888711	-2.03342017086565	-1.61233985927637	1.13658176915241	0.163246352986328	-0.155381203509501	-1.40810204595777	-1.51871555031922	0.386292142725089	-0.000773133691575285	0.469407282431870	0.257616874137028	-0.357291726338660	-0.0671284313002403	-0.509377822890645	-0.572608000483035	-0.147906717692361	0.659169421154452	1.43522102848992	-0.152034207304474	0.251941858386604	-0.497417461124781	0.116905664818320	0.275289277178508	-0.755203709697797	2.22957146785763	0.555154719294527	0.652305796615919	1.00826877453041	0.146105572979838	-0.256634416499596	0.133895908994531	-0.126483349212664	-0.633927959755159	0.907133860407389	-0.273679953571960	1.82388873695224	0.860301403454271	-1.39072648787288	0.921571185239675	-0.573433531133032	-1.12163606150109	0.558870707471904	0.134167317144201	-0.305535778447510	-0.128003909185354	0.304803563180243	-1.08644647965890	0.211174776626958	0.105030564096587	1.34013194086943	-0.419193084207268	-0.282889207566104	-0.360991736007167	1.64440633681238	1.40117561319074	0.679065261692241	-0.765370248561438	-1.34234842716183	-0.239447249386274	-0.776283223795091	-0.531575414835315	0.917380050169770	1.57200338272837	-0.513998768224665	1.92573206372285	0.232005688808544	-0.242143109543622	1.23883093120441	-1.41532573969461	-0.179337523151752	-0.212365055270431	0.182272817349738	-0.602318698167148	-0.346523666443487	-1.54898756197352	0.389286456722984	0.161679629361318	-1.14563809627829	0.110726561125987	1.74312708735490	-0.887866046193286	-0.962490134419171	0.416635224936179	-0.129945288421254	0.117346639135514	0.512960562736274	-1.27700773178666	-0.490825567754596	0.882622869078784	-0.139145597436045	-0.415451951244163	-1.77358666213416	-0.178423793176077	-0.907607641186415	1.87307000038037	1.28218845843930	-1.60422910386494	0.719010838189493	-1.52603594928721	-1.37999259538368	-0.194977580291328	-0.280710794639170	-1.05795243272254	-0.654838055078413	-0.665961993947025	-0.461071768356961	0.854564020717438	0.332509817976761	-0.177234083072455	-0.308713439112466	0.938118717765595	-0.757221425355940	-0.712448709127880	-0.563549511044288	-1.43123656129064	0.161744938618198	-0.672951954188959	-0.458499980329041	0.0244046128938637	-0.496640568315743	1.35406598347984	0.293763425795126	-0.705633466968328	1.40625157150124	1.32340621373365	0.899330414722574	-1.18252513081990	-0.322950964416424	-0.910438470306844	0.660778342774986	0.0485028676109636	-0.165850941059446	-1.51443608925401	-0.340555985222154	1.31578358944924	1.19027768103090	-0.320448799898888	0.347066142665661	0.630265145604789	-1.69090679243806	-0.203763777026184	-0.475958946118186	0.319954456472243	-1.88628755451303	-1.04787873026814	-1.18056308587166	0.334985468756267	-0.938139597493430	-0.273470738075715	-0.507811885426022	-0.292412361280691	-0.999995084440302	-0.512842841073832	-1.31612461222777	-0.955944745178966	-0.0178114913692724	-1.06804573709090	0.582593873815166	-1.23000668719641	-0.748689390673097	-1.77403803935419	1.74101125991652	2.12359286746926	0.207779551382047	0.702860190972616	0.584273398968520	0.135762636569675	0.906139667552781	-0.396190496031138	-0.109470660048003	0.803032724736956	0.859892536345077	-0.219175830930152	-1.94608025063852	-0.346152377168754	0.0522105176963491	-0.0731303043116516	1.81949647225938	1.02601550900064	0.224966377714619	0.151333802965623	-0.344659002315051	0.216561028027892	0.229581344854598	0.0179606386497292	0.376375447680896	0.765183891120639	0.626063790690268	0.290707695454633	0.699655512610052	-0.268989052976038	0.329870635701514	1.00789036932820	0.0311923442567386	1.17906051815900	-1.58892212129123	-0.294108547449947	-0.392791063044009	1.13570856818270	-0.0767492345399025	0.620193707410215	-1.71171295121418	0.147439194506687	-0.668634181122350	-0.991652780349161	-0.516484808780462	-0.201191397131899	-0.697552710181397	-0.499725915949662	-0.938177053836373	0.313621378032044	0.515318272363608	0.372115785456450	0.225539916791242	-0.754554621729607	-1.17185828416390	0.414564160827272	1.59040164925735]

irfs = get_irf(RBCmodel, shocks = shocks, periods = 0, levels = true, algorithm = :pruned_second_order, initial_state = collect(get_SS(RBCmodel, derivatives=false)))
irfs = get_irf(RBCmodel, shocks = shocks, periods = 0, levels = true)

using Statistics, LinearAlgebra

mean(irfs, dims = 2)
sqrt.(var(irfs, dims = 2))
[skewness(i) for i in eachrow(irfs[:,:,1])]
[kurtosis(i) for i in eachrow(irfs[:,:,1])]




state_update, pruning = MacroModelling.parse_algorithm_to_state_update(:pruned_second_order, RBCmodel)
Y = zeros(RBCmodel.timings.nVars,size(shocks,2)+1)
initial_state = zero(collect(get_SS(RBCmodel, derivatives=false)))
shock_history = shocks
periods = size(shocks,2)

Y[:,2], pruned_state = state_update(initial_state, shock_history[:,1], initial_state)

for t in 2:periods
    Y[:,t+1], pruned_state = state_update(Y[:,t], shock_history[:,t],pruned_state)
end
Y .+= collect(get_SS(RBCmodel, derivatives=false))

# change reference stady state in get_irf
kron(mean(Y, dims = 2),mean(Y, dims = 2)')
mean(Y, dims = 2)[[2,3,6]]
(Y * Y' / (periods+1))[[2,3,6],[2,3,6]]

(Y * Y' / (periods+1))

(Y * Y' / (periods+1)) - kron(mean(Y, dims = 2),mean(Y, dims = 2)')

third_moment = zeros(RBCmodel.timings.nVars,RBCmodel.timings.nVars,RBCmodel.timings.nVars);
for (i,v) in enumerate(eachrow(Y))
    third_moment[:,i,:] = Y * diagm(v) * Y' / (periods+1)
end
third_moment[[2,3,6],[2,3,6],[2,3,6]]

fourth_moment = zeros(RBCmodel.timings.nVars,RBCmodel.timings.nVars,RBCmodel.timings.nVars,RBCmodel.timings.nVars);
for (h,u) in enumerate(eachrow(Y))
    for (i,v) in enumerate(eachrow(Y))
        fourth_moment[:,h,i,:] = Y * diagm(u) * diagm(v) * Y' / (periods+1)
    end
end
fourth_moment[[2,3,6],[2,3,6],[2,3,6],[2,3,6]]


[Statistics.std(i) for i in eachrow(Y)][[2,3,6]]
[Statistics.var(i) for i in eachrow(Y)][[2,3,6]]

std(Y, dims = 2)[[2,3,6]]
[skewness(i) for i in eachrow(irfs[:,:,1])]
[kurtosis(i) for i in eachrow(irfs[:,:,1])]



# calc theoretical moments
sol = get_solution(RBCmodel)
sol2 = get_solution(RBCmodel, algorithm = :second_order)
# reshape(permutedims(sol2([:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎],RBCmodel.timings.past_not_future_and_mixed,[:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎]),[2,1,3]),RBCmodel.timings.nPast_not_future_and_mixed,RBCmodel.timings.nPast_not_future_and_mixed^2)

Hxx = reshape(permutedims(sol2([:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎],RBCmodel.timings.past_not_future_and_mixed,[:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎]),[2,1,3]),RBCmodel.timings.nPast_not_future_and_mixed,RBCmodel.timings.nPast_not_future_and_mixed^2)
Gxx = reshape(permutedims(sol2([:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎],setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed),[:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎]),[2,1,3]),RBCmodel.timings.nVars - RBCmodel.timings.nPast_not_future_and_mixed,RBCmodel.timings.nPast_not_future_and_mixed^2)

Huu = sol2(:u_a₍ₓ₎,RBCmodel.timings.past_not_future_and_mixed,:u_a₍ₓ₎)|>collect
Guu = sol2(:u_a₍ₓ₎,setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed),:u_a₍ₓ₎)|>collect
Hxu = sol2(:u_a₍ₓ₎,RBCmodel.timings.past_not_future_and_mixed,[:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎])|>collect
Gxu = sol2(:u_a₍ₓ₎,setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed),[:a₍₋₁₎,:c₍₋₁₎,:k₍₋₁₎])|>collect
hss = sol2(:Volatility,RBCmodel.timings.past_not_future_and_mixed,:Volatility)|>collect
gss = sol2(:Volatility,setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed),:Volatility)|>collect

hx = sol[2:end-1,:](:,RBCmodel.timings.past_not_future_and_mixed)|>collect
gx = sol[2:end-1,:](:,setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed))|>collect
hu = sol(:u_a₍ₓ₎,RBCmodel.timings.past_not_future_and_mixed)|>collect
gu = sol(:u_a₍ₓ₎,setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed))|>collect

# AA = kron(A,A')
# CC = kron(C,C')
# RBCmodel.timings.past_not_future_and_mixed
# T.future_not_past_and_mixed
# A = @views 𝑺₁[T.past_not_future_and_mixed_idx,1:T.nPast_not_future_and_mixed] * ℒ.diagm(ones(length(subset_indices)))[indexin(T.past_not_future_and_mixed_idx,subset_indices),:]
# C = @views 𝑺₁[subset_indices,T.nPast_not_future_and_mixed+1:end]

# CC = C * C'

# lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

# # reshape(ℐ.bicgstabl(lm, vec(-CC)), size(CC))
# reshape(ℐ.gmres(lm, vec(-CC)), size(CC))


A = ([hx zero(hx) zero(Hxx)
            zero(hx) hx Hxx/2
            zeros(size(hx)[1]^2,2*size(hx)[1]) kron(hx,hx)])

B = sparse([hu zero(Huu) zero(Hxu) zero(Hxu)
    zero(hu) Huu/2 Hxu zero(Hxu)
    zeros(size(hu,1)^2,size(hu,2)) kron(hu,hu) kron(hx,hu) kron(hu,hx)])

C = [gx' gx' Gxx/2]

D = [gu Guu/2 Gxu]

c = [zero(hss)
(hss + Huu)/2
kron(hu,hu)]

d = (gss + Guu) / 2


Ez = (I - A) \ c

Ey = get_SS(RBCmodel, derivatives = false)(setdiff(RBCmodel.timings.var,RBCmodel.timings.past_not_future_and_mixed)) + C * Ez + d

get_SSS(RBCmodel, algorithm = :pruned_second_order)