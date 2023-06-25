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



# translate_mod_file("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/AnSchorfheide_Gaussian.mod")
# include("/Users/thorekockerols/Downloads/ReplicationDSGEHOS-main/AnSchorfheide_Gaussian.jl")

include("AnSchorfheide_Gaussian3.jl")
m = AnSchorfheide_Gaussian


T = m.timings
states = T.past_not_future_and_mixed

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



sol = get_solution(m,m.parameter_values, algorithm = :third_order)

hx = sol[2][indexin(states,T.var),1:T.nPast_not_future_and_mixed]
hu = sol[2][indexin(states,T.var),((T.nPast_not_future_and_mixed + 1):end)] 
gx = sol[2][indexin(T.var,T.var),1:T.nPast_not_future_and_mixed]
gu = sol[2][indexin(T.var,T.var),((T.nPast_not_future_and_mixed + 1):end)]


# first order
A = hx
B = hu
C = gx
D = gu

c = zeros(T.nPast_not_future_and_mixed)
d = zeros(T.nVars)

ybar = sol[1][indexin(T.var,T.var)]

## First-order moments, ie expectation of variables
IminA = I - A
Ez   = IminA \ c
Ey   = ybar + C * Ez + d # recall y = yss + C*z + d


## Compute Zero-Lag Cumulants of innovations, states and controls
nz = size(A,1);

CC = B*B'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + D * D'



# Second order solution
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
second_order_axis_ordered = vcat(T.past_not_future_and_mixed,:Volatility,T.exo)
k = 1
for i in second_order_axis_ordered
    for j in second_order_axis_ordered
        second_order_helper_ordered[k,:] = [i,j,k,string(i)*string(j)]
        k += 1
    end
end



Hxx = sol[3][indexin(states,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Huu = sol[3][indexin(states,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Hxu = sol[3][indexin(states,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
hss = sol[3][indexin(states,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]

Gxx = sol[3][indexin(T.var,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (states,),4],second_order_helper[:,4]),3]]
Guu = sol[3][indexin(T.var,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (T.exo,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
Gxu = sol[3][indexin(T.var,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .∈ (states,) .&& second_order_helper_ordered[:,2] .∈ (T.exo,),4],second_order_helper[:,4]),3]]
gss = sol[3][indexin(T.var,T.var),second_order_helper[indexin(second_order_helper_ordered[second_order_helper_ordered[:,1] .== :Volatility .&& second_order_helper_ordered[:,2] .== :Volatility,4],second_order_helper[:,4]),3]]


M2u = vec(I(T.nExo))




hx_hx = kron(hx,hx)
hx_hu = kron(hx,hu)
hu_hx = kron(hu,hx)
hu_hu = kron(hu,hu)


# get Fxi
nu2 = Int(nu * (nu+1) / 2);
nxi = nu + nu^2 + 2*nx*nu;
nximin = nu + nu2 + nu*nx;
nz = 2 * nx + nx^2

col1_u       = 1:nu;
col2_u_u     = col1_u[end]   .+ (1:nu2);
col3_xf_u    = col2_u_u[end] .+ (1:nu*nx);

row1_u       = 1:nu;
row2_u_u     = row1_u[end]    .+ (1:nu^2);    
row3_xf_u    = row2_u_u[end]  .+ (1:nu*nx);
row4_u_xf    = row3_xf_u[end] .+ (1:nx*nu);



DPu, DPinv = duplication(nu)
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
Ey   = ybar + C*Ez+d # recall y = yss + C*z + d


####  Γ₂
# write a loop to fill Γ₂
# size of input vector


function build_Γ₂(T, C2z0::AbstractMatrix, Ey::Vector)

    nu = T.nExo

    nx = T.nPast_not_future_and_mixed
        
    nx2 = nx*(nx+1)/2 |> Int

    nu2 = nu*(nu+1)/2 |> Int
    
    u_u = reshape([[(i, i == k ? 1 : 0), (k, i == k ? 1 : 0)] for k in T.exo for i in T.exo],nu,nu)

    xf_u = reshape([[i, (k,1)] for k in T.past_not_future_and_mixed for i in T.exo],nx,nu)

    if size(C2z0) == (nu,nu) 
        nximin = nu + nu2 + nu*nx

        inputs = vcat(T.exo, upper_triangle(u_u), vec(xf_u))
        
        Γ = spzeros(nximin, nximin);

        shock_indices = [1:nu + nu2...]

    else
        nu3 = nu2*(nu+2)/3 |> Int

        nximin = nu + nu2 + 2*nu*nx + nu*nx2 + nu2*nx + nu3
        
        xs_u = reshape([[i, (k,2)] for k in T.past_not_future_and_mixed for i in T.exo],nx,nu)

        xf_xf_u = reshape([[i, (k,1), (j,1)] for j in T.past_not_future_and_mixed for k in T.past_not_future_and_mixed for i in T.exo],nx, nx, nu)

        xf_u_u = reshape([[i, k, (j,1)] for k in T.exo for i in T.exo for j in T.past_not_future_and_mixed],nx, nu, nu)

        u_u_u = reshape([[i, k, j] for k in T.exo for i in T.exo for j in T.exo],nu, nu, nu)

        inputs = vcat(T.exo, upper_triangle(u_u), vec(xf_u), vec(xs_u), upper_triangle(xf_xf_u), upper_triangle(xf_u_u, alt = true), upper_triangle(u_u_u, triple = true))
        
        Γ = zeros(nximin, nximin);

        shock_indices = [1:nu + nu2..., nximin - nu3 + 1:nximin...]

    end

    for (i¹,s¹) in enumerate(inputs)
        for (i²,s²) in enumerate(inputs)
            terms1 = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in (s¹ isa Symbol ? [s¹] : s¹)]
            terms2 = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in (s² isa Symbol ? [s²] : s²)]

            intrsct = intersecting_elements(terms1,terms2)

            intrsct_cnts = Dict([element => count(==(element),intrsct) for element in unique(intrsct)])
            
            ϵ²_terms = filter(u -> all([(v isa Symbol ? 0 : v[1]) ∈ T.exo for v in u]) && u[1][2] == 1,[(s¹ isa Symbol ? [s¹] : s¹), (s² isa Symbol ? [s²] : s²)])

            terms = filter(u -> !(all([(v isa Symbol ? 0 : v[1]) ∈ T.exo for v in u]) && u[1][2] == 1),[(s¹ isa Symbol ? [s¹] : s¹), (s² isa Symbol ? [s²] : s²)])

            if length(ϵ²_terms) > 0
                ϵ²_combo = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in reduce(vcat,ϵ²_terms)]
                ϵ²_combo_cnts = Dict([element => count(==(element),ϵ²_combo) for element in unique(ϵ²_combo)])
            else
                ϵ²_combo = []
                ϵ²_combo_cnts = []
            end

            if length(terms) > 0
                combo = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in reduce(vcat,terms)]
                combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])
            else
                combo = []
                combo_cnts = []
            end

            combined_combo = vcat(combo,ϵ²_combo)
            combined_combo_cnts = Dict([element => count(==(element),combined_combo) for element in unique(combined_combo)])

            if any([k ∈ T.exo && v == 1 for (k,v) in combined_combo_cnts])
                continue
            elseif all([k ∈ T.exo && v == 2 for (k,v) in combined_combo_cnts]) && 
                all([k ∈ T.exo && v == 1 for (k,v) in intrsct_cnts]) && 
                length(intrsct_cnts) > 0

                Γ[i¹,i²] = 1

            elseif all([k ∈ T.exo && v == 4 for (k,v) in combined_combo_cnts]) && 
                all([k ∈ T.exo && v == 2 for (k,v) in intrsct_cnts])

                Γ[i¹,i²] = 2

            elseif length(setdiff(keys(combined_combo_cnts),T.exo)) == 0 && 
                length(intrsct_cnts) > 0 && 
                all([intrsct_cnts[i] > 0 for i in collect(intersect(keys(combined_combo_cnts),keys(intrsct_cnts)))]) && 
                any([combined_combo_cnts[i] == 4 for i in collect(intersect(keys(combined_combo_cnts),keys(intrsct_cnts)))])

                Γ[i¹,i²] = 3

            elseif all([k ∈ T.exo && v == 6 for (k,v) in combined_combo_cnts]) && 
                all([k ∈ T.exo && v == 3 for (k,v) in intrsct_cnts])

                Γ[i¹,i²] = 15
                
            elseif length(filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)) > 0 && 
                sum(values(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))) == 2 && 
                all([i[2] == 1 for i in keys(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))]) && 
                all([k ∈ T.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ T.exo, intrsct_cnts)])

                if all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)])
                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
    
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    Γ[i¹,i²] = C2z0[idxs[1], idxs[2]]
                elseif all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)])
                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
    
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    Γ[i¹,i²] = 3 * C2z0[idxs[1], idxs[2]]
                end

            elseif length(filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)) > 0 && # at least one shock 
                sum(values(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))) == 2 && # non shocks have max two entries
                all([i[2] == 2 for i in keys(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))]) && # non shocks are all double entries
                all([k ∈ T.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ T.exo, intrsct_cnts)]) # all shocks appear in both entries

                vars = setdiff(keys(combined_combo_cnts), T.exo)
                indices_mat = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in vars] .+ T.nPast_not_future_and_mixed

                idxs = length(indices_mat) == 1 ? [indices_mat[1],indices_mat[1]] : indices_mat
                    
                indices = [indexin([i[1]], T.var)[1] for i in vars]

                idxs2 = length(indices) == 1 ? [indices[1],indices[1]] : indices
            
                Γ[i¹,i²] = C2z0[idxs[1], idxs[2]] + Ey[idxs2[1]] * Ey[idxs2[2]]

            elseif length(filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)) > 0 && # at least one shock 
                length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2, combined_combo_cnts)) == 1 &&
                sum(values(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))) > 2 && # non shocks have more than two entries
                all([k ∈ T.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ T.exo, intrsct_cnts)]) # all shocks appear in both entries

                indices_second_mean = [indexin([i[1][1]], T.var)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combined_combo_cnts)][1]

                indices_first_variance = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts)]
                
                indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

                indices_first = (indices_first_variance[1] - 1) * T.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * T.nPast_not_future_and_mixed

                indices_second = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combined_combo_cnts)][1] + T.nPast_not_future_and_mixed

                Γ[i¹,i²] = C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean]

            elseif length(filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)) > 0 && # at least one shock 
                sum(values(filter(((j,u),) -> !(j ∈ T.exo), combined_combo_cnts))) == 4 && # non shocks have four entries
                all([k ∈ T.exo && v >= 1 for (k,v) in filter(((j,u),) -> j ∈ T.exo, intrsct_cnts)]) # all shocks appear in both entries

                vars1 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo), s¹)]
                vars2 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo), s²)]
                
                if vars1 == vars2
                    Γ[i¹,i²] = 
                    C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]
                else
                    Γ[i¹,i²] = 
                    C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
                    C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
                    C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]]
                end

            elseif length(filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)) > 0 && 
                sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2, combined_combo_cnts))) == 1 &&  
                sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 0 &&  
                all([k ∈ T.exo && v >= 1 for (k,v) in  filter(((j,u),) -> j ∈ T.exo, intrsct_cnts)])

                indices = [indexin([i[1]], T.var)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)][1]
                
                if all([v == 4 for (k,v) in filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)])
                    Γ[i¹,i²] = 3 * Ey[indices]
                elseif all([v == 2 for (k,v) in filter(((j,u),) -> j ∈ T.exo, combined_combo_cnts)])
                    Γ[i¹,i²] = Ey[indices]
                end
            end
        end
    end

    return Γ
end


build_Γ₂(T, C2z0::AbstractMatrix) = build_Γ₂(T, C2z0::AbstractMatrix, [0])

Γ₂ = build_Γ₂(T, C2z0)

CC = B * Fxi * Γ₂ * (B * Fxi)'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + D * Fxi * Γ₂ * (D * Fxi)'

diag(C2y0)



####  Γ₃
# write a loop to fill Γ₂
# size of input vector

function build_Γ₃(T, C2z0::AbstractMatrix, Ey::Vector)
    nx = T.nPast_not_future_and_mixed
    
    nx2 = nx*(nx+1)/2 |> Int

    nu = T.nExo

    nu2 = nu*(nu+1)/2 |> Int

    u_u = reshape([[(i, i == k ? 1 : 0), (k, i == k ? 1 : 0)] for k in T.exo for i in T.exo],nu,nu)

    xf_u = reshape([[i, (k,1)] for k in T.past_not_future_and_mixed for i in T.exo],nx,nu)

    if size(C2z0) == (2 * nx + nx^2, 2 * nx + nx^2)
        nximin = nu + nu2 + nu*nx 

        inputs = vcat(T.exo, upper_triangle(u_u), vec(xf_u))

        shock_indices = [1:nu + nu2...]
    else
        nu3 = nu2*(nu+2)/3 |> Int
    
        nximin = nu + nu2 + 2*nu*nx + nu*nx2 + nu2*nx + nu3
    
        xs_u = reshape([[i, (k,2)] for k in T.past_not_future_and_mixed for i in T.exo],nx,nu)
    
        xf_xf_u = reshape([[i, (k,1), (j,1)] for j in T.past_not_future_and_mixed for k in T.past_not_future_and_mixed for i in T.exo],nx, nx, nu)
    
        xf_u_u = reshape([[i, k, (j,1)] for k in T.exo for i in T.exo for j in T.past_not_future_and_mixed],nx, nu, nu)
    
        u_u_u = reshape([[i, k, j] for k in T.exo for i in T.exo for j in T.exo],nu, nu, nu)
    
        inputs = vcat(T.exo, upper_triangle(u_u), vec(xf_u), vec(xs_u), upper_triangle(xf_xf_u), upper_triangle(xf_u_u, alt = true), upper_triangle(u_u_u, triple = true))

        shock_indices = [1:nu + nu2..., nximin - nu3 + 1:nximin...]
    end
    
    # Γ₃ = zeros(nximin, nximin, nximin);
    Γ₃ = spzeros(nximin^2, nximin);

    for (i¹,s¹) in enumerate(inputs)
        for (i²,s²) in enumerate(inputs)
            for (i³,s³) in enumerate(inputs)

                ϵ²_terms = filter(u -> all([(v isa Symbol ? 0 : v[1]) ∈ T.exo for v in u]) && u[1][2] == 1,[(s¹ isa Symbol ? [s¹] : s¹), (s² isa Symbol ? [s²] : s²), (s³ isa Symbol ? [s³] : s³)])

                terms = filter(u -> !(all([(v isa Symbol ? 0 : v[1]) ∈ T.exo for v in u]) && u[1][2] == 1),[(s¹ isa Symbol ? [s¹] : s¹), (s² isa Symbol ? [s²] : s²), (s³ isa Symbol ? [s³] : s³)])

                if length(ϵ²_terms) > 0
                    ϵ²_combo = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in reduce(vcat,ϵ²_terms)]
                    ϵ²_combo_cnts = Dict([element => count(==(element),ϵ²_combo) for element in unique(ϵ²_combo)])
                else
                    ϵ²_combo = []
                    ϵ²_combo_cnts = []
                end

                if length(terms) > 0
                    combo = [c isa Symbol ? c : c[1] ∈ T.exo ? c[1] : c for c in reduce(vcat,terms)]
                    combo_cnts = Dict([element => count(==(element),combo) for element in unique(combo)])
                else
                    combo = []
                    combo_cnts = []
                end

                combined_combo = vcat(combo,ϵ²_combo)
                combined_combo_cnts = Dict([element => count(==(element),combined_combo) for element in unique(combined_combo)])

                intersect_ϵ² = intersecting_elements(combo,ϵ²_combo)
                intersect_ϵ²_cnts = Dict([element => count(==(element),intersect_ϵ²) for element in unique(intersect_ϵ²)])

                if any([k ∈ T.exo && v == 1 for (k,v) in combined_combo_cnts])
                    continue
                elseif i¹ ∈ shock_indices && i² ∈ shock_indices && i³ ∈ shock_indices
                    if length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 &&
                        length(filter(((j,u),) -> j ∈ T.exo && u % 2 != 0, combined_combo_cnts)) == 0 &&
                        length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 1 &&
                        length(intersect_ϵ²_cnts) > 0

                        Γ₃[(i¹-1)*nximin+i²,i³] = 2

                    elseif all([k ∈ T.exo && v == 2 for (k,v) in combined_combo_cnts]) &&
                        length(ϵ²_combo_cnts) == 0

                        Γ₃[(i¹-1)*nximin+i²,i³] = 1

                    elseif i¹ == i² && i¹ == i³ && all(values(combined_combo_cnts) .== 6)

                        Γ₃[(i¹-1)*nximin+i²,i³] = 8

                    elseif length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 &&
                        length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1  &&
                        length(ϵ²_combo_cnts) == 0

                        Γ₃[(i¹-1)*nximin+i²,i³] = 3

                    elseif any(values(combined_combo_cnts) .== 6) && 
                        length(intersect_ϵ²_cnts) > 0 && 
                        !(i¹ == i² && i¹ == i³)

                        Γ₃[(i¹-1)*nximin+i²,i³] = 12 # Variance of ϵ²

                    elseif all(values(combined_combo_cnts) .== 4) && any(values(combo_cnts) .== 2) && !(all(values(intersect_ϵ²_cnts) .== 3))

                        Γ₃[(i¹-1)*nximin+i²,i³] = 6

                    elseif all(values(combined_combo_cnts) .== 4)

                        Γ₃[(i¹-1)*nximin+i²,i³] = 9

                    elseif length(filter(((j,u),) -> j ∈ T.exo && u == 6, combined_combo_cnts)) == 1 &&
                        length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1  &&
                        length(ϵ²_combo_cnts) == 0
                        
                        Γ₃[(i¹-1)*nximin+i²,i³] = 15

                    elseif all(values(combined_combo_cnts) .== 8)

                        Γ₃[(i¹-1)*nximin+i²,i³] = 90

                    end

                elseif length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2 && 
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1 &&
                    length(combined_combo_cnts) == 3 &&
                    length(ϵ²_combo_cnts) == 0

                    indices = [indexin([i[1]], T.var)[1] for i in setdiff(keys(combo_cnts), T.exo)][1]

                    Γ₃[(i¹-1)*nximin+i²,i³] = Ey[indices]

                elseif length(filter(((j,u),) -> j ∈ T.exo && u ∈ [2,4], combined_combo_cnts)) == 2 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 0 &&
                    (length(combined_combo_cnts) == 2 || 
                    (length(combined_combo_cnts) == 3 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1)) && 
                    length(intersect_ϵ²_cnts) == 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1

                    indices = [indexin([i[1]], T.var)[1] for i in setdiff(keys(combo_cnts), T.exo)][1]

                    Γ₃[(i¹-1)*nximin+i²,i³] = 3 * Ey[indices]

                elseif length(filter(((j,u),) -> j ∈ T.exo && u ∈ [4,6], combined_combo_cnts)) == 1 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 1 &&
                    (length(combined_combo_cnts) == 2 || 
                    (length(combined_combo_cnts) == 3 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1)) && 
                    length(intersect_ϵ²_cnts) > 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1

                    indices = [indexin([i[1]], T.var)[1] for i in setdiff(keys(combo_cnts), T.exo)][1]

                    if length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1
        
                        Γ₃[(i¹-1)*nximin+i²,i³] = 2 * Ey[indices]

                    elseif length(filter(((j,u),) -> j ∈ T.exo && u == 6, combined_combo_cnts)) == 1

                        Γ₃[(i¹-1)*nximin+i²,i³] = 12 * Ey[indices]
                    end

                elseif length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 1 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 &&
                    (length(combined_combo_cnts) == 2 || 
                    (length(combined_combo_cnts) == 3 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1)) && 
                    length(intersect_ϵ²_cnts) > 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 2, combined_combo_cnts)) == 1

                    vars = setdiff(keys(combo_cnts), T.exo)
                    indices_mat = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in vars] .+ T.nPast_not_future_and_mixed

                    idxs = length(indices_mat) == 1 ? [indices_mat[1],indices_mat[1]] : indices_mat
                        
                    indices = [indexin([i[1]], T.var)[1] for i in vars]

                    idxs2 = length(indices) == 1 ? [indices[1],indices[1]] : indices
                
                    Γ₃[(i¹-1)*nximin+i²,i³] = 2 * (C2z0[idxs[1], idxs[2]] + Ey[idxs2[1]] * Ey[idxs2[2]])

                elseif length(ϵ²_combo_cnts) == 0 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2 && 
                    (length(combined_combo_cnts) == 2 || 
                    (length(combined_combo_cnts) == 3 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2)) && 
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 2, combined_combo_cnts)) == 1

                    vars = setdiff(keys(combo_cnts), T.exo)
                    indices_mat = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in vars] .+ T.nPast_not_future_and_mixed

                    idxs = length(indices_mat) == 1 ? [indices_mat[1],indices_mat[1]] : indices_mat
                        
                    indices = [indexin([i[1]], T.var)[1] for i in vars]

                    idxs2 = length(indices) == 1 ? [indices[1],indices[1]] : indices
                
                    Γ₃[(i¹-1)*nximin+i²,i³] = C2z0[idxs[1], idxs[2]] + Ey[idxs2[1]] * Ey[idxs2[2]]

                elseif (sum(values(combined_combo_cnts)) == 6 ||
                    (sum(values(combined_combo_cnts)) == 8 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 1)) &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 2

                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
        
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices

                    if length(ϵ²_combo_cnts) == 1 &&
                        length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 1 &&
                        length(intersect_ϵ²_cnts) > 0
                    
                        Γ₃[(i¹-1)*nximin+i²,i³] = 2 * C2z0[idxs[1], idxs[2]]

                    elseif length(ϵ²_combo_cnts) == 0

                        Γ₃[(i¹-1)*nximin+i²,i³] = 3 * C2z0[idxs[1], idxs[2]]

                    end

                elseif length(ϵ²_combo_cnts) == 0  &&
                    sum(values(combined_combo_cnts)) == 6 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 2

                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
        
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices
                    
                    Γ₃[(i¹-1)*nximin+i²,i³] = C2z0[idxs[1], idxs[2]]

                elseif length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 1 &&
                    (sum(values(combined_combo_cnts)) == 7) && 
                    length(intersect_ϵ²_cnts) > 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1 &&
                    (length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 2, combined_combo_cnts)) == 1 ||
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 1, combined_combo_cnts)) == 2)

                    indices_second_mean = [indexin([i[1][1]], T.var)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1]

                    indices_first_variance = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combo_cnts)]
                    
                    indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

                    indices_first = (indices_first_variance[1] - 1) * T.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * T.nPast_not_future_and_mixed

                    indices_second = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1] + T.nPast_not_future_and_mixed

                    Γ₃[(i¹-1)*nximin+i²,i³] = 2 * (C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean])

                elseif length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 0 &&
                    (sum(values(combined_combo_cnts)) == 7) && 
                    length(intersect_ϵ²_cnts) == 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1 &&
                    (length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 2, combined_combo_cnts)) == 1 ||
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 1, combined_combo_cnts)) == 2)

                    indices_second_mean = [indexin([i[1][1]], T.var)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1]

                    indices_first_variance = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combo_cnts)]
                    
                    indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

                    indices_first = (indices_first_variance[1] - 1) * T.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * T.nPast_not_future_and_mixed

                    indices_second = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1] + T.nPast_not_future_and_mixed

                    Γ₃[(i¹-1)*nximin+i²,i³] = C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean]

                elseif length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 && 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, ϵ²_combo_cnts)) == 0 &&
                    (sum(values(combined_combo_cnts)) == 7) && 
                    length(intersect_ϵ²_cnts) == 0 &&
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 2 && u == 1, combined_combo_cnts)) == 1 &&
                    (length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 2, combined_combo_cnts)) == 1 ||
                    length(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1 && u == 1, combined_combo_cnts)) == 2)

                    indices_second_mean = [indexin([i[1][1]], T.var)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1]

                    indices_first_variance = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combo_cnts)]
                    
                    indices_first_variance = length(indices_first_variance) == 1 ? [indices_first_variance[1], indices_first_variance[1]] : indices_first_variance

                    indices_first = (indices_first_variance[1] - 1) * T.nPast_not_future_and_mixed + indices_first_variance[2] + 2 * T.nPast_not_future_and_mixed

                    indices_second = [indexin([i[1][1]], T.past_not_future_and_mixed)[1] for i in filter(((j,u),) -> !(j ∈ T.exo) && u == 1 && j[2] == 2, combo_cnts)][1] + T.nPast_not_future_and_mixed

                    Γ₃[(i¹-1)*nximin+i²,i³] = 3 * (C2z0[indices_second, indices_first] + C2z0[indices_first_variance[1], indices_first_variance[2]] * Ey[indices_second_mean])

                elseif length(ϵ²_combo_cnts) == 1  &&
                    sum(values(combined_combo_cnts)) == 8 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 6, combined_combo_cnts)) == 1 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 2

                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
        
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices

                    Γ₃[(i¹-1)*nximin+i²,i³] = 12 * C2z0[idxs[1], idxs[2]]

                elseif length(ϵ²_combo_cnts) == 0  &&
                    sum(values(combined_combo_cnts)) == 8 &&
                    length(filter(((j,u),) -> j ∈ T.exo && u == 6, combined_combo_cnts)) == 1 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 2

                    indices = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in setdiff(keys(combined_combo_cnts), T.exo)]
        
                    idxs = length(indices) == 1 ? [indices[1],indices[1]] : indices

                    Γ₃[(i¹-1)*nximin+i²,i³] = 15 * C2z0[idxs[1], idxs[2]]

                elseif length(intersect_ϵ²_cnts) == 1 && # at least one shock 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 4

                    vars1 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s¹)]
                    vars2 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s²)]
                    vars3 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s³)]
                    
                    if length(vars1) == 0
                        vars1 = vars3
                    end

                    if length(vars2) == 0
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars2) == 1
                        vars1 = vcat(vars1,vars2)
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars3) == 1
                        vars1 = vcat(vars1,vars3)
                    end

                    if length(vars2) == 1 && length(vars3) == 1
                        vars2 = vcat(vars2,vars3)
                    end

                    sort!(vars1)
                    sort!(vars2)

                    if vars1 == vars2
                        Γ₃[(i¹-1)*nximin+i²,i³] = 2 * (
                            C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]])
                    else
                        Γ₃[(i¹-1)*nximin+i²,i³] = 2 * (
                            C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
                            C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]])
                    end

                elseif length(ϵ²_combo_cnts) == 0 && # at least one shock 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 2, combined_combo_cnts)) == 2 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 4

                    vars1 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s¹)]
                    vars2 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s²)]
                    vars3 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s³)]
                    
                    if length(vars1) == 0
                        vars1 = vars3
                    end

                    if length(vars2) == 0
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars2) == 1
                        vars1 = vcat(vars1,vars2)
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars3) == 1
                        vars1 = vcat(vars1,vars3)
                    end

                    if length(vars2) == 1 && length(vars3) == 1
                        vars2 = vcat(vars2,vars3)
                    end

                    sort!(vars1)
                    sort!(vars2)

                    if vars1 == vars2
                        Γ₃[(i¹-1)*nximin+i²,i³] = 
                            C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]]
                    else
                        Γ₃[(i¹-1)*nximin+i²,i³] = 
                            C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
                            C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]]
                    end

                elseif length(ϵ²_combo_cnts) == 0 && # at least one shock 
                    length(filter(((j,u),) -> j ∈ T.exo && u == 4, combined_combo_cnts)) == 1 && 
                    sum(values(filter(((j,u),) -> !(j ∈ T.exo) && j[2] == 1, combined_combo_cnts))) == 4

                    vars1 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s¹)]
                    vars2 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s²)]
                    vars3 = [indexin([i[1]], T.past_not_future_and_mixed)[1] for i in filter(j -> !(j ∈ T.exo) && !(j[1] ∈ T.exo), s³)]
                    
                    if length(vars1) == 0
                        vars1 = vars3
                    end

                    if length(vars2) == 0
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars2) == 1
                        vars1 = vcat(vars1,vars2)
                        vars2 = vars3
                    end

                    if length(vars1) == 1 && length(vars3) == 1
                        vars1 = vcat(vars1,vars3)
                    end

                    if length(vars2) == 1 && length(vars3) == 1
                        vars2 = vcat(vars2,vars3)
                    end

                    sort!(vars1)
                    sort!(vars2)

                    if vars1 == vars2
                        Γ₃[(i¹-1)*nximin+i²,i³] = 3 * (
                            C2z0[vars1[1], vars1[1]] * C2z0[vars2[2], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars1[2], vars2[1]])
                    else
                        Γ₃[(i¹-1)*nximin+i²,i³] =  3 * (
                            C2z0[vars1[1], vars1[2]] * C2z0[vars2[1], vars2[2]] + 
                            C2z0[vars1[1], vars2[2]] * C2z0[vars2[1], vars1[2]] + 
                            C2z0[vars2[1], vars1[2]] * C2z0[vars1[1], vars2[2]])
                    end

                end

            end
        end
    end

    return Γ₃
end

Γ₃ = build_Γ₃(T,C2z0,vec(Ey))


CC = kron(B * Fxi, B * Fxi) *  Γ₃  * (B * Fxi)'
AA = kron(A,A)

lm = LinearMap{Float64}(x -> AA * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C3z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
reshape(C3z0, nz, nz, nz)

C3y0 = kron(C,C) * C3z0 * C' + kron(D * Fxi, D * Fxi) * Γ₃ * (D * Fxi)'
reshape(C3y0, T.nVars, T.nVars, T.nVars)

[C3y0[(i-1) * size(C3y0,2) + i,i] for i in 1:size(C3y0,2)]


collect(C3y0) ./ repeat(abs.(C2y0).^(3/2), size(C3y0,2))
[C3y0[(i-1) * size(C3y0,2) + i,i] / C2y0[i,i]^(3/2) for i in 1:size(C3y0,2)]



# transition to third order pruned solution
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
third_order_axis_ordered = vcat(T.past_not_future_and_mixed,:Volatility,T.exo)
k = 1
for i in third_order_axis_ordered
    for j in third_order_axis_ordered
        for l in third_order_axis_ordered
            third_order_helper_ordered[k,:] = [i,j,l,k,string(i)*string(j)*string(l)]
            k += 1
        end
    end
end



Hxxx = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (states,),5],third_order_helper[:,5]),4]]
Hxxu = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Hxuu = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Huuu = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Hxss = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Huss = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Hsss = sol[4][indexin(states,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .== :Volatility .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]


Gxxx = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (states,),5],third_order_helper[:,5]),4]]
Gxxu = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (states,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Gxuu = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Guuu = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .∈ (T.exo,) .&& third_order_helper_ordered[:,3] .∈ (T.exo,),5],third_order_helper[:,5]),4]]
Gxss = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (states,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Guss = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .∈ (T.exo,) .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]
Gsss = sol[4][indexin(T.var,T.var),third_order_helper_ordered[indexin(third_order_helper_ordered[third_order_helper_ordered[:,1] .== :Volatility .&& third_order_helper_ordered[:,2] .== :Volatility .&& third_order_helper_ordered[:,3] .== :Volatility,5],third_order_helper[:,5]),4]]


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
IminA = I - A;
Ez   = collect(IminA) \ c;
Ey   = ybar + C * Ez + d # recall y = yss + C*z + d



## Second-order moments
####  Γ₂
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


Γ₂ = build_Γ₂(T,C2z0,vec(Ey))

CC = B * Fxi *  Γ₂  * (B * Fxi)'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C2z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))

C2y0 = C * C2z0 * C' + D * Fxi * Γ₂ * (D * Fxi)'

diag(C2y0)



## Third-order moments
####  Γ₃
Γ₃ = build_Γ₃(T, C2z0, vec(Ey))

# using BenchmarkTools
# @benchmark build_Γ₃(m, C2z0, vec(Ey))

CC = kron(B * Fxi, B * Fxi) * Γ₃ * (B * Fxi)'
AA = kron(A,A)
lm = LinearMap{Float64}(x -> AA * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

C3z0 = reshape(ℐ.gmres(lm, vec(-CC)), size(CC))
reshape(C3z0,size(C3z0,2),size(C3z0,2),size(C3z0,2))

C3y0 = kron(C,C) * C3z0 * C' + kron(D * Fxi, D * Fxi) * Γ₃ * (D * Fxi)'
reshape(C3y0,size(C3y0,2),size(C3y0,2),size(C3y0,2))

[C3y0[(i-1) * size(C3y0,2) + i,i] for i in 1:size(C3y0,2)]

collect(C3y0) ./ repeat(abs.(C2y0).^(3/2), size(C3y0,2))
[C3y0[(i-1) * size(C3y0,2) + i,i] / C2y0[i,i]^(3/2) for i in 1:size(C3y0,2)]



###########

# make the loop return a matrix with the shock related entries and two matrices you can use to multiply with C2z0 instead of sorting the entries one by one. akin to a permutation matrix
# sparsify matrices
# check if SpeedMapping helps with the sylvester equations



# test against simulated moments
# using Statistics, LinearAlgebra, StatsBase

# # check distributional properties by simulating
# shocks = randn(2,100000)
# shocks .-= mean(shocks,dims=2)
# sim = get_irf(m, shocks = shocks, periods = 0, levels = true, algorithm = :pruned_second_order, initial_state = collect(get_SS(m, derivatives=false)))

# [mean(i) for i in eachrow(sim[:,:,1])]
# [Statistics.var(i) for i in eachrow(sim[:,:,1])]
# [skewness(i) for i in eachrow(sim[:,:,1])]
# [kurtosis(i) for i in eachrow(sim[:,:,1])]

# (diag(C2y0))[[2,4,5,1,3]]

# ([reshape(C3y0,T.nVars,T.nVars,T.nVars)[i,i,i] for i in 1:T.nVars] ./ diag(C2y0).^(3/2))[[2,4,5,1,3]]


# sim_lin = get_irf(m, shocks = shocks, periods = 0, levels = true, initial_state = collect(get_SS(m, derivatives=false)))

# [mean(i) for i in eachrow(sim_lin[:,:,1])]
# [Statistics.var(i) for i in eachrow(sim_lin[:,:,1])]
# [skewness(i) for i in eachrow(sim_lin[:,:,1])]
# [kurtosis(i) for i in eachrow(sim_lin[:,:,1])]

# [mean(i) for i in eachrow(sim_lin[:,:,1])]


# Statistics.var(sim, dims = 2)
# diag(C2y0)
# StatsBase.skewness(sim[1,:])
# diag(reshape(C3y0,5,5,5))


# [StatsBase.skewness(i) for i in eachrow(sim[:,:,1])]
# [reshape(C3y0,5,5,5)[i,i,i] for i in 1:5]
# [StatsBase.skewness(i) for i in eachrow(sim[:,:,1])]
# [kurtosis(i) for i in eachrow(sim[:,:,1])]


#############





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