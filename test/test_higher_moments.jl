using MacroModelling
import IterativeSolvers as ℐ
using LinearAlgebra, LinearMaps



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

function upper_triangle(mat::AbstractMatrix{T}) where T
    @assert size(mat, 1) == size(mat, 2) "The input matrix must be square"

    upper_elems = T[]
    for i in 1:size(mat, 1)
        for j in i:size(mat, 2)
            push!(upper_elems, mat[i, j])
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

    m = Int(p*(p+1)/2)
    
    DP = zeros(p*p,m)

    for r in 1:size(DP,1)
        DP[r, j[r]] = 1
    end

    DPinv = (DP'*DP)\DP'

    return DP, DPinv
end

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
GAMMA2XI = spzeros(nximin,nximin)
GAMMA2XI[1:(nu + size(DP,2)),1:(nu + size(DP,2))] = I(nu + size(DP,2))
GAMMA2XI[nu .+ (1:size(DP,2)),nu .+ (1:size(DP,2))] += diagm(DPinv * vec(I(nu)))
GAMMA2XI[nu + size(DP,2) + 1 : end,nu + size(DP,2) + 1 : end] = expand_mat(C2z0,nu)

matt = GAMMA2XI[nu .+ (1:size(DP,2)),nu .+ (1:size(DP,2))]
findnz(kron(matt,kron(matt,matt)))

nz = size(A,1);

BFxi = B*Fxi
DFxi = D*Fxi

CkronC = kron(C,C)
BFxikronBFxi= kron(BFxi,BFxi)
DFxikronDFxi= kron(DFxi,DFxi)

CC = BFxi *  GAMMA2XI  * BFxi'

# B' * (pinv(Fxi)' * GAMMA2XI' * pinv(Fxi))' * B
lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + DFxi * GAMMA2XI * DFxi'


diag(C2y0)


GAMMAMax = sparse(pinv(B) * CC' * pinv(B'))
droptol!(GAMMAMax,1e-6)
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
nu = 2
nx = 2
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



####  Γ₃
nu = 2
nx = 2
# write a loop to fill Γ₂
# size of input vector
Γ₃ = zeros(Int(nu + nu*(nu+1)/2 + nx*nu), Int(nu + nu*(nu+1)/2 + nx*nu), Int(nu + nu*(nu+1)/2 + nx*nu))

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

            n_same_indices_acros = s¹[2] == s²[2] || s¹[2] == s³[2] || s³[2] == s²[2]

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
                    if indices |> length == 1 && n_ϵ2 < 2#  || n_same_indices_acros == 2)
                        Γ₃[i¹,i²,i³] = 2
                    end

                    if n_ϵ2 == 3 && n_same_indices_acros == true && n_same_indices_within == 1
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

                if n_ϵ2 == 1 && n_same_indices_acros == false && n_same_indices_within == 0 && indices |> length == 2
                    Γ₃[i¹,i²,i³] = 1
                end
            end

            if n_x == 2 && n_same_indices_within_ϵ == 1 && s¹[2][2] == s²[2][2] && s²[2][2] == s³[2][2] #exactly one is epsilon with matching indices, there is one more with matching indices, the last index is common across the two x and epsilon
                # println(indices_x1)
                # println([i¹,i²,i³])
                idxs = collect(indices_x1)
                # println(idxs)

                if length(idxs) == 1
                    Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[1]]
                else
                    Γ₃[i¹,i²,i³] = 2 * C2z0[idxs[1],idxs[2]]
                end
                # Γ₃[i¹,i²,i³] = 1
            end
                # if s² == :ϵ²
                #     Γ₃[i¹,i²] = 2 # Variance of ϵ²
                # end

                # if s² == :ϵᵢₖ
                #     Γ₃[i¹,i²] = 1
                # end

                # if i¹ > n_shocks
                #     positions = position_in_symmetric_matrix(i² - n_shocks, Int(nx*(nx+1)/2))

                #     if positions isa Tuple{Int,Int}
                #         pos = positions
                #         for iᵉ in 1:nu
                #             Γ₃[n_shocks + (pos[1] - 1) * nu + iᵉ, n_shocks + (pos[2] - 1) * nu + iᵉ] = C2z0[pos...] # Covariance of x
                #         end
                #     else
                #         for pos in positions
                #             for iᵉ in 1:nu
                #                 Γ₃[n_shocks + (pos[1] - 1) * nu + iᵉ, n_shocks + (pos[2] - 1) * nu + iᵉ] = C2z0[pos...] # Covariance of x
                #             end
                #         end
                #     end
                # end
            # end
        end
    end
end

Γ₃

inputs

2*C2z0[1,1]^1


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