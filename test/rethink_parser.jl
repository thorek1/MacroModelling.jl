using Revise
using MacroModelling
import MacroModelling: get_NSSS_and_parameters
using BenchmarkTools

include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")
get_solution(m)
get_std(m, verbose = true)
get_irf(m)


include("../models/Smets_Wouters_2007.jl")
Smets_Wouters_2007.jacobian[1]|>sparse
get_solution(Smets_Wouters_2007,verbose = true)
get_std(Smets_Wouters_2007, verbose = true)


include("../models/NAWM_EAUS_2008.jl")
get_solution(NAWM_EAUS_2008,verbose = true)
get_std(Smets_Wouters_2007, verbose = true)

Smets_Wouters_2007
𝓂 = m
𝓂 = Smets_Wouters_2007
𝓂 = NAWM_EAUS_2008


ne = 230
np = 404
nx = 154
calc! = 𝓂.jacobian[1]

e = zeros(Symbolics.Num,ne)
Symbolics.@variables p[1:nx], v[1:np]

calc!(e,v,p)

Symbolics.sparsejacobian(e,v)

function take_nth_order_derivatives(f!::Function, 
                                    nx::Int, 
                                    np::Int, 
                                    nϵ::Int;
                                    max_perturbation_order::Int = 1)
    Symbolics.@variables 𝒳[1:nx], 𝒫[1:np]

    ϵˢ = zeros(Symbolics.Num, nϵ)

    f!(ϵˢ, 𝒳, 𝒫)

    if max_perturbation_order >= 2 
        second_order_idxs = [nx * (i-1) + k for i in 1:nx for k in 1:i]
        if max_perturbation_order == 3
            third_order_idxs = [nx^2 * (i-1) + nx * (k-1) + l for i in 1:nx for k in 1:i for l in 1:k]
        end
    end

    first_order  = Symbolics.Num[]
    second_order = Symbolics.Num[]
    third_order  = Symbolics.Num[]
    row1 = Int[]
    row2 = Int[]
    row3 = Int[]
    column1 = Int[]
    column2 = Int[]
    column3 = Int[]

    for (c1, var1) in enumerate(𝒳)
        for (r, eq) in enumerate(ϵˢ)
            if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
                deriv_first = Symbolics.derivative(eq, var1)
                
                push!(first_order, deriv_first)
                push!(row1, r)
                push!(column1, c1)

                if max_perturbation_order >= 2 
                    for (c2, var2) in enumerate(𝒳)
                        if (((c1 - 1) * nx + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                            deriv_second = Symbolics.derivative(deriv_first, var2)
                            
                            push!(second_order, deriv_second)
                            push!(row2, r)
                            push!(column2, Int.(indexin([(c1 - 1) * nx + c2], second_order_idxs))...)

                            if max_perturbation_order == 3
                                for (c3, var3) in enumerate(𝒳)
                                    if (((c1 - 1) * nx^2 + (c2 - 1) * nx + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                        deriv_third = Symbolics.derivative(deriv_second, var3)
    
                                        push!(third_order, deriv_third)
                                        push!(row3, r)
                                        push!(column3, Int.(indexin([(c1 - 1) * nx^2 + (c2 - 1) * nx + c3], third_order_idxs))...)
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return (first_order, row1, column1), (second_order, row2, column2), (third_order, row3, column3)
end


function take_nth_order_derivatives_direct(f!::Function, 
                                    nx::Int, 
                                    np::Int, 
                                    nϵ::Int;
                                    max_perturbation_order::Int = 1)
    Symbolics.@variables 𝒳[1:nx], 𝒫[1:np]

    ϵˢ = zeros(Symbolics.Num, nϵ)

    f!(ϵˢ, 𝒳, 𝒫)

    sp_first_order = Symbolics.sparsejacobian(ϵˢ,𝒳)

    if max_perturbation_order >= 2
        sp_second_order = Symbolics.sparsejacobian(sp_first_order.nzval,𝒳)
        if max_perturbation_order == 3
            sp_third_order = Symbolics.sparsejacobian(sp_second_order.nzval,𝒳)
            return sp_first_order, sp_second_order, sp_third_order
        else
            return sp_first_order, sp_second_order
        end
    else
        return sp_first_order
    end
end



function take_nth_order_derivatives_sparse(f!::Function, 
                                          nx::Int, 
                                          np::Int, 
                                          nϵ::Int;
                                          max_perturbation_order::Int = 1)
    Symbolics.@variables 𝒳[1:nx], 𝒫[1:np]

    ϵˢ = zeros(Symbolics.Num, nϵ)

    # Evaluate the function symbolically
    f!(ϵˢ, 𝒳, 𝒫)

    # --- First Order ---
    sp_first_order = Symbolics.sparsejacobian(ϵˢ, 𝒳)
    
    # Extract non-zero values and their (row, column) indices for the first order
    vals1 = sp_first_order.nzval
    rows1 = Int[]
    cols1 = Int[] # This will store the uncompressed column index for the first order

    # Build a map from the linear index in nzval to the (row, col) index in the sparse matrix
    # This is needed to trace back for higher orders
    nzval1_to_idx1 = Dict{Int, Tuple{Int, Int}}()
    k1 = 1 # linear index counter for nzval of sp_first_order
    for j = 1:size(sp_first_order, 2) # Iterate through columns
        for i_ptr = sp_first_order.colptr[j]:sp_first_order.colptr[j+1]-1
            i = sp_first_order.rowval[i_ptr] # original row index
            push!(rows1, i)
            push!(cols1, j) # uncompressed column index
            nzval1_to_idx1[k1] = (i, j)
            k1 += 1
        end
    end

    result1 = sparse(rows1, cols1, vals1, nϵ, nx)
    result2 = sparse(Int[], Int[], Symbolics.Num[], nϵ, nx * (nx + 1) / 2) # Initialize as empty
    result3 = sparse(Int[], Int[], Symbolics.Num[], nϵ, nx * (nx + 1)* (nx + 2) / 6) # Initialize as empty

    if max_perturbation_order >= 2
        # --- Second Order ---
        # The sparse Jacobian of the first order non-zeros gives d^2 f_i / (dx_j dx_k)
        # where the row index corresponds to the k-th non-zero of the first Jacobian (f_i w.r.t x_j)
        # and the column index corresponds to x_k.
        sp_second_order = Symbolics.sparsejacobian(sp_first_order.nzval, 𝒳)

        vals2 = Symbolics.Num[]
        rows2 = Int[]
        cols2 = Int[] # This will store the compressed column index

        # Build a map from the linear index in nzval of sp_second_order to the (row, col) index in sp_second_order
        nzval2_to_idx2 = Dict{Int, Tuple{Int, Int}}()
        k2 = 1 # linear index counter for nzval of sp_second_order
        for j2 = 1:size(sp_second_order, 2) # Iterate through columns (corresponds to x_k)
            for i_ptr2 = sp_second_order.colptr[j2]:sp_second_order.colptr[j2+1]-1
                i2 = sp_second_order.rowval[i_ptr2] # row index in sp_second_order (corresponds to the r2-th non-zero of sp_first_order)
                val = sp_second_order.nzval[i_ptr2]

                # Trace back: i2 is the linear index of a non-zero in sp_first_order.nzval
                # nzval1_to_idx1[i2] gives (original_row_f, original_col_x1) from first derivative
                original_row_f, original_col_x1 = nzval1_to_idx1[i2] # f_i w.r.t x_j -> (i, j)

                original_col_x2 = j2 # w.r.t x_k -> k

                # We have the derivative d^2 f_i / (dx_j dx_k) with indices (original_row_f, original_col_x1, original_col_x2)
                # Original code compresses based on the variable indices (c1, c2) where c2 <= c1
                c1 = original_col_x1
                c2 = original_col_x2

                # Check symmetry condition for compression (c2 <= c1)
                # The original code's second_order_idxs are generated for i=1:nx, k=1:i,
                # mapping (i, k) to nx*(i-1)+k. It then checks if (c1, c2) corresponds to such an (i,k) pair.
                # This corresponds to checking if c2 <= c1 and mapping it to the index for (c1, c2)
                # in the lower triangle.
                if c2 <= c1
                    # Calculate the compressed column index: nx*(c1-1) + c2
                    compressed_col_idx = nx * (c1 - 1) + c2

                    push!(vals2, val)
                    push!(rows2, original_row_f)
                    push!(cols2, compressed_col_idx)
                end
                
                nzval2_to_idx2[k2] = (i2, j2) # Store mapping for tracing back third order
                k2 += 1
            end
        end
        println(maximum(cols2))
        result2 = sparse(rows2, cols2, vals2, nϵ, nx * (nx + 1) / 2) 

        if max_perturbation_order == 3
            # --- Third Order ---
            # The sparse Jacobian of the second order non-zeros gives d^3 f_i / (dx_j dx_k dx_l)
            # where the row index corresponds to the m-th non-zero of the second Jacobian (d^2 f_i / (dx_j dx_k))
            # and the column index corresponds to x_l.
            sp_third_order = Symbolics.sparsejacobian(sp_second_order.nzval, 𝒳)

            vals3 = Symbolics.Num[]
            rows3 = Int[]
            cols3 = Int[] # This will store the compressed column index

            k3 = 1 # linear index counter for nzval of sp_third_order
            for j3 = 1:size(sp_third_order, 2) # Iterate through columns (corresponds to x_l)
                for i_ptr3 = sp_third_order.colptr[j3]:sp_third_order.colptr[j3+1]-1
                    i3 = sp_third_order.rowval[i_ptr3] # row index in sp_third_order (corresponds to the r3-th non-zero of sp_second_order)
                    val = sp_third_order.nzval[i_ptr3]

                    # Trace back: i3 is the linear index of a non-zero in sp_second_order.nzval
                    # nzval2_to_idx2[i3] gives (r2, c2) from second derivative spmatrix
                    r2_sp2, c2_sp2 = nzval2_to_idx2[i3] # (linear_idx_in_nzval1, original_col_x2_spmatrix)

                    # r2_sp2 is the linear index of a non-zero in sp_first_order.nzval
                    # nzval1_to_idx1[r2_sp2] gives (original_row_f, original_col_x1) from first derivative
                    original_row_f, original_col_x1 = nzval1_to_idx1[r2_sp2] # f_i w.r.t x_j -> (i, j)

                    # c2_sp2 is the column index in sp_second_order, which corresponds to the variable x_k
                    original_col_x2 = c2_sp2 # w.r.t x_k -> k

                    # j3 is the column index in sp_third_order, which corresponds to the variable x_l
                    original_col_x3 = j3 # w.r.t x_l -> l

                    # We have the derivative d^3 f_i / (dx_j dx_k dx_l) with indices (original_row_f, original_col_x1, original_col_x2, original_col_x3)
                    # Original code compresses based on the variable indices (c1, c2, c3) where c3 <= c2 <= c1
                    c1 = original_col_x1
                    c2 = original_col_x2
                    c3 = original_col_x3

                    # Check symmetry condition for compression (c3 <= c2 <= c1)
                    # The original code's third_order_idxs are generated for i=1:nx, k=1:i, l=1:k,
                    # mapping (i, k, l) to nx^2*(i-1) + nx*(k-1) + l. It checks if (c1, c2, c3)
                    # corresponds to such an (i, k, l) triplet. This corresponds to checking
                    # if c3 <= c2 <= c1 and mapping it to the index for (c1, c2, c3)
                    # in the structure generated by iterating (c1 from 1..nx, c2 from 1..c1, c3 from 1..c2).
                    # This seems inconsistent with the formula generation (l <= k <= i).
                    # Let's re-read the original code: third_order_idxs = [nx^2 * (i-1) + nx * (k-1) + l for i in 1:nx for k in 1:i for l in 1:k].
                    # It maps (i, k, l) with l <= k <= i. It checks if the flat index of (c1, c2, c3)
                    # is in this list. This implies the compression keeps (c1, c2, c3) if c3 <= c2 <= c1
                    # and the index is the one calculated by the generator with (i,k,l) = (c1, c2, c3).
                    # So the compressed index for (c1, c2, c3) with c3 <= c2 <= c1 is nx^2*(c1-1) + nx*(c2-1) + c3.
                    if c3 <= c2 <= c1
                         # Calculate the compressed column index: nx^2*(c1-1) + nx*(c2-1) + c3
                        compressed_col_idx = nx^2 * (c1 - 1) + nx * (c2 - 1) + c3

                        push!(vals3, val)
                        push!(rows3, original_row_f)
                        push!(cols3, compressed_col_idx)
                    end

                    k3 += 1
                end
            end
            result3 = sparse(rows3, cols3, vals3, nϵ, nx * (nx + 1)* (nx + 2) / 6)

        end
    end

    return result1, result2, result3
end


import SparseArrays
function take_nth_order_derivatives_sparse_compressed(f!::Function,
                                                     nx::Int,
                                                     np::Int,
                                                     nϵ::Int;
                                                     max_perturbation_order::Int = 1)
    Symbolics.@variables 𝒳[1:nx], 𝒫[1:np]

    ϵˢ = zeros(Symbolics.Num, nϵ)

    # Evaluate the function symbolically
    f!(ϵˢ, 𝒳, 𝒫)

    # --- Pre-generate compressed index lists and their mapping to compressed column index ---
    # These lists define the order and mapping for the compressed sparse arrays
    second_order_idxs_list = Int[]
    flat_to_compressed_idx_2 = Dict{Int, Int}() # Maps the flat uncompressed index (e.g., nx*(c1-1)+c2) to its position in the compressed list
    num_compressed_cols2 = 0

    if max_perturbation_order >= 2
        compressed_col_counter2 = 1
        # Corresponds to pairs (i, k) where k <= i in the original list generation
        for i in 1:nx # Corresponds to c1 in the original derivative notation d^2/dx_c1 dx_c2
            for k in 1:i # Corresponds to c2 in the original derivative notation d^2/dx_c1 dx_c2
                # The flat index calculated for the pair (i, k) in the list generation
                flat_idx = nx * (i - 1) + k
                push!(second_order_idxs_list, flat_idx)
                flat_to_compressed_idx_2[flat_idx] = compressed_col_counter2
                compressed_col_counter2 += 1
            end
        end
        num_compressed_cols2 = length(second_order_idxs_list)
    end

    third_order_idxs_list = Int[]
    flat_to_compressed_idx_3 = Dict{Int, Int}() # Maps the flat uncompressed index (e.g., nx^2*(c1-1)+nx*(c2-1)+c3) to its position in the compressed list
    num_compressed_cols3 = 0

    if max_perturbation_order == 3
        compressed_col_counter3 = 1
        # Corresponds to triplets (i, k, l) where l <= k <= i in the original list generation
        for i in 1:nx # Corresponds to c1 in d^3/dx_c1 dx_c2 dx_c3
            for k in 1:i # Corresponds to c2 in d^3/dx_c1 dx_c2 dx_c3
                for l in 1:k # Corresponds to c3 in d^3/dx_c1 dx_c2 dx_c3
                    # The flat index calculated for the triplet (i, k, l) in the list generation
                    flat_idx = nx^2 * (i - 1) + nx * (k - 1) + l
                    push!(third_order_idxs_list, flat_idx)
                    flat_to_compressed_idx_3[flat_idx] = compressed_col_counter3
                    compressed_col_counter3 += 1
                end
            end
        end
        num_compressed_cols3 = length(third_order_idxs_list)
    end


    # --- First Order ---
    # This is a standard Jacobian (nϵ x nx), not compressed in the same way
    sp_first_order = Symbolics.sparsejacobian(ϵˢ, 𝒳)

    # Build a map from the linear index in nzval of sp_first_order to the (row, col) index in the sparse matrix
    # This is needed to trace back for higher orders
    nzval1_to_idx1 = Dict{Int, Tuple{Int, Int}}()
    k1 = 1 # linear index counter for nzval of sp_first_order
    for j = 1:size(sp_first_order, 2) # Iterate through columns
        for i_ptr = sp_first_order.colptr[j]:sp_first_order.colptr[j+1]-1
            i = sp_first_order.rowval[i_ptr] # original row index
            nzval1_to_idx1[k1] = (i, j)
            k1 += 1
        end
    end

    result1 = sp_first_order
    result2 = spzeros(Symbolics.Num, 0, 0) # Placeholder for empty sparse matrix
    result3 = spzeros(Symbolics.Num, 0, 0) # Placeholder for empty sparse matrix


    if max_perturbation_order >= 2
        # --- Second Order ---
        # The sparse Jacobian of the first order non-zeros gives d^2 f_i / (dx_j dx_k)
        # where the row index corresponds to the r2-th non-zero of the first Jacobian (f_i w.r.t x_j)
        # and the column index corresponds to x_k.
        sp_second_order_flat = Symbolics.sparsejacobian(sp_first_order.nzval, 𝒳)

        # Collect (row, compressed_col, value) triplets for the compressed sparse matrix
        sparse_rows2 = Int[]
        sparse_cols2 = Int[]
        sparse_vals2 = Symbolics.Num[]

        # Iterate through the non-zero entries of the flattened second derivative matrix
        k_sp2 = 1 # linear index in sp_second_order_flat.nzval
        for col_sp2 = 1:size(sp_second_order_flat, 2) # column index in sp_second_order_flat (corresponds to dx_k)
            for i_ptr_sp2 = sp_second_order_flat.colptr[col_sp2]:sp_second_order_flat.colptr[col_sp2+1]-1
                row_sp2 = sp_second_order_flat.rowval[i_ptr_sp2] # row index in sp_second_order_flat (corresponds to the row_sp2-th nzval of J1)
                val = sp_second_order_flat.nzval[i_ptr_sp2]

                # Trace back: row_sp2 is the linear index of a non-zero in sp_first_order.nzval
                # nzval1_to_idx1[row_sp2] gives (original_row_f, original_col_x1) from first derivative
                original_row_f, original_col_x1 = nzval1_to_idx1[row_sp2] # f_i w.r.t x_j -> (i, j)

                original_col_x2 = col_sp2 # w.r.t x_k -> k

                # We have the derivative d^2 f_i / (dx_j dx_k) with indices (original_row_f, original_col_x1, original_col_x2)
                # We need to map this to the compressed column index based on (c1, c2) -> (original_col_x1, original_col_x2)
                # The original code's list is generated for (i, k) with k <= i, and it checks ((c1 - 1) * nx + c2) ∈ second_order_idxs
                # This means c1 corresponds to the first variable index in the pair, c2 to the second.
                # So we need the pair (original_col_x1, original_col_x2) to correspond to (i, k) in the generator, requiring original_col_x2 <= original_col_x1.
                # The flat index used for lookup is nx * (original_col_x1 - 1) + original_col_x2.
                flat_idx_to_check = nx * (original_col_x1 - 1) + original_col_x2

                # Check if this flat index is in the list generated by the original code's logic
                if haskey(flat_to_compressed_idx_2, flat_idx_to_check)
                     # Get the compressed column index (position in the sorted list)
                     compressed_col_idx = flat_to_compressed_idx_2[flat_idx_to_check]

                     push!(sparse_rows2, original_row_f)
                     push!(sparse_cols2, compressed_col_idx)
                     push!(sparse_vals2, val)
                end

                k_sp2 += 1 # Increment linear index counter for sp_second_order_flat.nzval
            end
        end
        # Construct the second order compressed sparse matrix
        result2 = SparseArrays.sparse!(sparse_rows2, sparse_cols2, sparse_vals2, nϵ, num_compressed_cols2)


        if max_perturbation_order == 3
            # --- Third Order ---
            # We need a map from the linear index in nzval of sp_second_order_flat to the (row, col) index in sp_second_order_flat
            # This is needed to trace back for third order
            nzval2_to_idx2 = Dict{Int, Tuple{Int, Int}}()
            k2 = 1 # linear index counter for nzval of sp_second_order_flat
            for j2 = 1:size(sp_second_order_flat, 2) # Iterate through columns
                 for i_ptr2 = sp_second_order_flat.colptr[j2]:sp_second_order_flat.colptr[j2+1]-1
                    i2 = sp_second_order_flat.rowval[i_ptr2] # original row index in sp_second_order_flat
                    nzval2_to_idx2[k2] = (i2, j2)
                    k2 += 1
                end
            end


            sp_third_order_flat = Symbolics.sparsejacobian(sp_second_order_flat.nzval, 𝒳)

            # Collect (row, compressed_col, value) triplets for the compressed sparse matrix
            sparse_rows3 = Int[]
            sparse_cols3 = Int[]
            sparse_vals3 = Symbolics.Num[]

            # Iterate through the non-zero entries of the flattened third derivative matrix
            k_sp3 = 1 # linear index in sp_third_order_flat.nzval
            for col_sp3 = 1:size(sp_third_order_flat, 2) # column index in sp_third_order_flat (corresponds to dx_l)
                for i_ptr_sp3 = sp_third_order_flat.colptr[col_sp3]:sp_third_order_flat.colptr[col_sp3+1]-1
                    row_sp3 = sp_third_order_flat.rowval[i_ptr_sp3] # row index in sp_third_order_flat (corresponds to the row_sp3-th nzval of J2)
                    val = sp_third_order_flat.nzval[i_ptr_sp3]

                    # Trace back: row_sp3 is the linear index of a non-zero in sp_second_order_flat.nzval
                    # nzval2_to_idx2[row_sp3] gives (r2_sp2, c2_sp2) from sp_second_order_flat
                    r2_sp2, c2_sp2 = nzval2_to_idx2[row_sp3] # (linear_idx_in_nzval1, original_col_x2_spmatrix)

                    # Trace back further: r2_sp2 is the linear index of a non-zero in sp_first_order.nzval
                    # nzval1_to_idx1[r2_sp2] gives (original_row_f, original_col_x1) from first derivative
                    original_row_f, original_col_x1 = nzval1_to_idx1[r2_sp2] # f_i w.r.t x_j -> (i, j)

                    # c2_sp2 is the column index in sp_second_order_flat, corresponds to dx_k
                    original_col_x2 = c2_sp2

                    # col_sp3 is the column index in sp_third_order_flat, corresponds to dx_l
                    original_col_x3 = col_sp3

                    # We have the derivative d^3 f_i / (dx_j dx_k dx_l) with indices (original_row_f, original_col_x1, original_col_x2, original_col_x3)
                    # We need to map this to the compressed column index based on (c1, c2, c3) -> (original_col_x1, original_col_x2, original_col_x3)
                    # The original code's list is generated for (i, k, l) with l <= k <= i, and it checks ((c1-1)*nx^2 + (c2-1)*nx + c3) ∈ third_order_idxs
                    # This means c1=i, c2=k, c3=l. Requires original_col_x3 <= original_col_x2 <= original_col_x1.
                    # The flat index used for lookup is nx^2 * (original_col_x1 - 1) + nx * (original_col_x2 - 1) + original_col_x3.
                    flat_idx_to_check = nx^2 * (original_col_x1 - 1) + nx * (original_col_x2 - 1) + original_col_x3


                    # Check if this flat index is in the list generated by the original code's logic
                    if haskey(flat_to_compressed_idx_3, flat_idx_to_check)
                         # Get the compressed column index (position in the sorted list)
                         compressed_col_idx = flat_to_compressed_idx_3[flat_idx_to_check]

                         push!(sparse_rows3, original_row_f)
                         push!(sparse_cols3, compressed_col_idx)
                         push!(sparse_vals3, val)
                    end

                    k_sp3 += 1 # Increment linear index counter for sp_third_order_flat.nzval
                end
            end
             # Construct the third order compressed sparse matrix
            result3 = SparseArrays.sparse!(sparse_rows3, sparse_cols3, sparse_vals3, nϵ, num_compressed_cols3)
        end
    end

    return result1, result2, result3
end

nx*(nx+1)/2
nx*(nx+1)/2
nx*(nx+1)*(nx+2)/6
302621
first_order = take_nth_order_derivatives_sparse(calc!, nx, np, ne; max_perturbation_order = 1)
first_order_sp = take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 2)
first_order = take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 3)
first_order_sp[2]
first_order[3]
spjac



using Symbolics, SparseArrays

function take_nth_order_derivatives_sparse_compressed(
    f!::Function,
    nx::Int,
    np::Int,
    nϵ::Int;
    max_perturbation_order::Int = 1
)
    Symbolics.@variables 𝒳[1:nx] 𝒫[1:np]
    ϵˢ = zeros(Symbolics.Num, nϵ)
    f!(ϵˢ, 𝒳, 𝒫)

    # --- 1st order ---
    sp1 = Symbolics.sparsejacobian(ϵˢ, 𝒳)       # nϵ × nx
    # map linear index in sp1.nzval → (row, var₁)
    nz1_to_rc = Dict{Int,Tuple{Int,Int}}()
    cnt1 = 1
    for j in 1:size(sp1,2), ptr in sp1.colptr[j]:(sp1.colptr[j+1]-1)
        nz1_to_rc[cnt1] = (sp1.rowval[ptr], j)
        cnt1 += 1
    end

    # prepare defaults
    result1 = sp1
    result2 = spzeros(Symbolics.Num, nϵ, 0)
    result3 = spzeros(Symbolics.Num, nϵ, 0)

    # --- 2nd order (compressed Hessian) ---
    if max_perturbation_order ≥ 2
        sp2_flat = Symbolics.sparsejacobian(sp1.nzval, 𝒳)

        rows2 = Int[]
        cols2 = Int[]
        vals2 = Symbolics.Num[]

        for k2 in 1:size(sp2_flat,2)
            for ptr2 in sp2_flat.colptr[k2]:(sp2_flat.colptr[k2+1]-1)
                r2  = sp2_flat.rowval[ptr2]
                val = sp2_flat.nzval[ptr2]
                i, j = nz1_to_rc[r2]
                k    = k2

                # keep only k ≤ j and inline c₂ = ((j−1)*j)÷2 + k
                if k ≤ j
                    c2 = ((j - 1) * j) ÷ 2 + k
                    push!(rows2, i)
                    push!(cols2, c2)
                    push!(vals2, val)
                end
            end
        end

        ncols2  = (nx * (nx + 1)) ÷ 2
        result2 = SparseArrays.sparse!(rows2, cols2, vals2, nϵ, ncols2)
    end

    # --- 3rd order (compressed third‑derivative) ---
    if max_perturbation_order == 3
        # map linear index in sp2_flat.nzval → (r2_index, var₂)
        nz2_to_rck = Dict{Int,Tuple{Int,Int}}()
        cnt2 = 1
        for k2 in 1:size(sp2_flat,2), ptr2 in sp2_flat.colptr[k2]:(sp2_flat.colptr[k2+1]-1)
            nz2_to_rck[cnt2] = (sp2_flat.rowval[ptr2], k2)
            cnt2 += 1
        end

        sp3_flat = Symbolics.sparsejacobian(sp2_flat.nzval, 𝒳)

        rows3 = Int[]
        cols3 = Int[]
        vals3 = Symbolics.Num[]

        for k3 in 1:size(sp3_flat,2)
            for ptr3 in sp3_flat.colptr[k3]:(sp3_flat.colptr[k3+1]-1)
                r3  = sp3_flat.rowval[ptr3]
                val = sp3_flat.nzval[ptr3]

                r2, k2 = nz2_to_rck[r3]
                i, j    = nz1_to_rc[r2]
                ℓ       = k3

                # keep only ℓ ≤ k₂ ≤ j and inline
                # c₃ = ((j−1)*j*(j+1))÷6 + ((k₂−1)*k₂)÷2 + ℓ
                if ℓ ≤ k2 ≤ j
                    c3 = ((j - 1) * j * (j + 1)) ÷ 6 +
                         ((k2 - 1) * k2) ÷ 2 +
                         ℓ
                    push!(rows3, i)
                    push!(cols3, c3)
                    push!(vals3, val)
                end
            end
        end

        ncols3  = (nx * (nx + 1) * (nx + 2)) ÷ 6
        result3 = SparseArrays.sparse!(rows3, cols3, vals3, nϵ, ncols3)
    end

    return result1, result2, result3
end



function take_nth_order_derivatives_general(
    f!::Function,
    nx::Int,
    np::Int,
    nϵ::Int;
    max_perturbation_order::Int = 1,
    output_compressed::Bool = true # New argument
)
    if max_perturbation_order < 1
        throw(ArgumentError("max_perturbation_order must be at least 1"))
    end

    Symbolics.@variables 𝒳[1:nx] 𝒫[1:np]
    ϵˢ = zeros(Symbolics.Num, nϵ)

    # Evaluate the function symbolically
    f!(ϵˢ, 𝒳, 𝒫)

    results = [] # To store sparse matrices for each order

    # --- Order 1 ---
    # This is the base case: the standard sparse Jacobian (nϵ x nx)
    # The first order is always output in its standard uncompressed form (nϵ x nx)
    sp_curr = Symbolics.sparsejacobian(ϵˢ, 𝒳)
    push!(results, sp_curr)

    if max_perturbation_order == 1
        return Tuple(results)
    end

    # --- Prepare for higher orders (Order 2 to max_perturbation_order) ---
    # We need to maintain a map from the linear index in the previous Jacobian's
    # nzval to the original equation row and the tuple of variable indices.
    # Map structure: Dict{Int, Tuple{Int, Tuple{Vararg{Int}}}}
    # Key: linear index in the previous nzval vector (1 to count of non-zeros)
    # Value: (original_equation_row, (v_1, v_2, ..., v_{n-1}))

    # Initialize map for Order 1 (mapping linear index in sp_curr.nzval -> (row, (v1,)))
    nz_to_indices_prev = Dict{Int, Tuple{Int, Tuple{Int}}}()
    k_lin = 1 # Linear index counter for sp_curr.nzval
    # Iterate through the non-zeros of the current sparse matrix (Order 1)
    for j = 1:size(sp_curr, 2) # Column index in sp_curr (corresponds to v1)
        for ptr = sp_curr.colptr[j]:(sp_curr.colptr[j+1]-1)
            row = sp_curr.rowval[ptr] # Original equation row index
            nz_to_indices_prev[k_lin] = (row, (j,)) # Store (equation row, (v1,))
            k_lin += 1
        end
    end

    nzvals_prev = sp_curr.nzval # nzvals from Order 1

    # --- Iterate for orders n = 2, 3, ..., max_perturbation_order ---
    for n = 2:max_perturbation_order

        # Compute the Jacobian of the previous level's non-zero values w.r.t. 𝒳
        # This gives a flat matrix where rows correspond to non-zeros from order n-1
        # and columns correspond to the n-th variable we differentiate by (x_vn).
        sp_flat_curr = Symbolics.sparsejacobian(nzvals_prev, 𝒳)

        # Build the nz_to_indices map for the *current* level (order n)
        # Map: linear index in sp_flat_curr.nzval -> (original_row_f, (v_1, ..., v_n))
        nz_to_indices_curr = Dict{Int, Tuple{Int, Tuple{Vararg{Int}}}}()
        k_lin_curr = 1 # linear index counter for nzval of sp_flat_curr
        # Iterate through the non-zeros of the current flat Jacobian
        for col_curr = 1:size(sp_flat_curr, 2) # Column index in sp_flat_curr (corresponds to v_n)
            for ptr_curr = sp_flat_curr.colptr[col_curr]:(sp_flat_curr.colptr[col_curr+1]-1)
                row_curr = sp_flat_curr.rowval[ptr_curr] # Row index in sp_flat_curr (corresponds to the row_curr-th nzval of previous level)

                # Get previous indices info from the map of order n-1
                prev_info = nz_to_indices_prev[row_curr]
                orig_row_f = prev_info[1] # Original equation row
                vars_prev = prev_info[2] # Tuple of variables from previous order (v_1, ..., v_{n-1})

                # Append the current variable index (v_n)
                vars_curr = (vars_prev..., col_curr) # Full tuple (v_1, ..., v_n)

                # Store info for the current level's non-zero
                nz_to_indices_curr[k_lin_curr] = (orig_row_f, vars_curr)
                k_lin_curr += 1
            end
        end

        if output_compressed
            # --- Construct the COMPRESSED sparse matrix for order n ---
            sparse_rows_n = Int[]
            sparse_cols_n = Int[] # This will store the compressed column index
            sparse_vals_n = Symbolics.Num[]

            # Calculate the total number of compressed columns for order n
            # This is the number of tuples (v_n, ..., v_1) such that 1 <= v_n <= ... <= v_1 <= nx
            # which is given by the combination with repetition formula: (nx + n - 1) choose n
            ncols_n_compressed = binomial(nx + n - 1, n)

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr
            for col_flat_curr = 1:size(sp_flat_curr, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr.colptr[col_flat_curr]:(sp_flat_curr.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr
                    val = sp_flat_curr.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Check the compression rule: v_n <= v_{n-1} <= ... <= v_1
                    # Iterate from the second to last variable (v_{n-1}) down to the first (v_1)
                    is_compressed = true
                    for k_rule = 1:(n-1)
                         # We check v_{n-k_rule+1} <= v_{n-k_rule}
                         # Example n=3: k_rule=1 -> check v3 <= v2; k_rule=2 -> check v2 <= v1
                        if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                            is_compressed = false
                            break
                        end
                    end

                    if is_compressed
                        # Calculate the compressed column index c_n for the tuple (v_1, ..., v_n)
                        # using the derived formula: c_n = sum_{k=1}^{n-1} binomial(v_k + n - k - 1, n - k + 1) + v_n
                        compressed_col_idx = 0
                        for k_formula = 1:(n-1)
                             # The variable index is var_indices_full[k_formula] (v_k_formula)
                             # The order of the inner binomial is n - k_formula + 1
                             # The upper index is var_indices_full[k_formula] + n - k_formula - 1
                             term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                             compressed_col_idx += term
                        end
                        # Add the last term: v_n (var_indices_full[n])
                        compressed_col_idx += var_indices_full[n]

                        push!(sparse_rows_n, orig_row_f)
                        push!(sparse_cols_n, compressed_col_idx)
                        push!(sparse_vals_n, val)
                    end

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr.nzval
                end
            end

            # Construct the compressed sparse matrix for order n
            # Dimensions are nϵ rows by the number of unique compressed combinations
            sparse_matrix_n = SparseArrays.sparse!(sparse_rows_n, sparse_cols_n, sparse_vals_n, nϵ, Int(ncols_n_compressed))
            push!(results, sparse_matrix_n)

        else # output_compressed == false
            # --- Construct the UNCOMPRESSED sparse matrix for order n ---
            # Output matrix is nϵ rows x nx^n columns
            sparse_rows_n_uncomp = Int[]
            sparse_cols_n_uncomp = Int[] # Uncompressed column index (1 to nx^n)
            sparse_vals_n_uncomp = Symbolics.Num[]

            # Total number of uncompressed columns
            ncols_n_uncompressed = BigInt(nx)^n # Use BigInt for the power calculation, cast to Int for sparse dimensions later

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr
            for col_flat_curr = 1:size(sp_flat_curr, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr.colptr[col_flat_curr]:(sp_flat_curr.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr
                    val = sp_flat_curr.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Calculate the UNCOMPRESSED column index for the tuple (v_1, ..., v_n)
                    # This maps the tuple (v1, ..., vn) to a unique index from 1 to nx^n
                    # Formula: 1 + (v1-1)*nx^(n-1) + (v2-1)*nx^(n-2) + ... + (vn-1)*nx^0
                    uncompressed_col_idx = 1 # 1-based
                    power_of_nx = BigInt(nx)^(n-1) # Start with nx^(n-1) for v1 term
                    for i = 1:n
                        uncompressed_col_idx += (var_indices_full[i] - 1) * power_of_nx
                        if i < n # Avoid nx^-1
                            power_of_nx = div(power_of_nx, nx) # Integer division
                        end
                    end

                    push!(sparse_rows_n_uncomp, orig_row_f)
                    push!(sparse_cols_n_uncomp, Int(uncompressed_col_idx)) # Cast to Int
                    push!(sparse_vals_n_uncomp, val)

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr.nzval
                end
            end

            # Construct the uncompressed sparse matrix for order n
            # Dimensions are nϵ rows by nx^n columns
            sparse_matrix_n_uncomp = SparseArrays.sparse!(sparse_rows_n_uncomp, sparse_cols_n_uncomp, sparse_vals_n_uncomp, nϵ, Int(ncols_n_uncompressed))
            push!(results, sparse_matrix_n_uncomp)

        end # End of if output_compressed / else

        # Prepare for the next iteration (order n+1)
        nzvals_prev = sp_flat_curr.nzval # The nzvals for the next step are the current ones
        nz_to_indices_prev = nz_to_indices_curr # The map for the next step is the current map

    end # End of loop for orders n = 2 to max_perturbation_order

    return Tuple(results) # Return results as a tuple of sparse matrices
end


function take_nth_order_derivatives_with_params(
    f!::Function,
    nx::Int,
    np::Int,
    nϵ::Int;
    max_perturbation_order::Int = 1,
    output_compressed::Bool = true # Controls compression for X derivatives (order >= 2)
)
    if max_perturbation_order < 1
        throw(ArgumentError("max_perturbation_order must be at least 1"))
    end
    if np < 0
         throw(ArgumentError("np must be non-negative"))
    end

    Symbolics.@variables 𝒳[1:nx] 𝒫[1:np]
    ϵˢ = zeros(Symbolics.Num, nϵ)

    # Evaluate the function symbolically
    f!(ϵˢ, 𝒳, 𝒫)

    results = [] # To store pairs of sparse matrices (X_matrix, P_matrix) for each order

    # --- Order 1 ---
    # Compute the 1st order derivative with respect to X (Jacobian)
    spX_order_1 = Symbolics.sparsejacobian(ϵˢ, 𝒳) # nϵ x nx

    # Compute the derivative of the non-zeros of the 1st X-derivative w.r.t. P
    # This is an intermediate step. The final P matrix will be built from this.
    spP_of_flatX_nzval_order_1 = Symbolics.sparsejacobian(spX_order_1.nzval, 𝒫) # nnz(spX_order_1) x np

    # Determine dimensions for the Order 1 P matrix
    X_nrows_1 = nϵ
    X_ncols_1 = nx
    P_nrows_1 = X_nrows_1 * X_ncols_1
    P_ncols_1 = np

    # Build the Order 1 P matrix (dimensions nϵ*nx x np)
    sparse_rows_1_P = Int[] # Row index in the flattened space of spX_order_1
    sparse_cols_1_P = Int[] # Column index for parameters (1 to np)
    sparse_vals_1_P = Symbolics.Num[]

    # Map linear index in spX_order_1.nzval to its (row, col) in spX_order_1
    nz_lin_to_rc_1 = Dict{Int, Tuple{Int, Int}}()
    k_lin = 1
    for j = 1:size(spX_order_1, 2) # col
        for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
             r = spX_order_1.rowval[ptr] # row
             nz_lin_to_rc_1[k_lin] = (r, j)
             k_lin += 1
        end
    end


    # Iterate through the non-zero entries of spP_of_flatX_nzval_order_1
    k_temp_P = 1 # linear index counter for nzval
    for p_col = 1:size(spP_of_flatX_nzval_order_1, 2) # Parameter index
        for i_ptr_temp_P = spP_of_flatX_nzval_order_1.colptr[p_col]:(spP_of_flatX_nzval_order_1.colptr[p_col+1]-1)
            temp_row = spP_of_flatX_nzval_order_1.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval (corresponds to temp_row-th nzval of spX_order_1)
            p_val = spP_of_flatX_nzval_order_1.nzval[i_ptr_temp_P] # Derivative value w.r.t. parameter

            # Get the (row, col) in spX_order_1 corresponding to this derivative
            r_X1, c_X1 = nz_lin_to_rc_1[temp_row]

            # Calculate the row index in spP_order_1 (flattened index of spX_order_1)
            P_row_idx = (r_X1 - 1) * X_ncols_1 + c_X1
            P_col_idx = p_col # Parameter column index

            push!(sparse_rows_1_P, P_row_idx)
            push!(sparse_cols_1_P, P_col_idx)
            push!(sparse_vals_1_P, p_val)

            k_temp_P += 1
        end
    end

    spP_order_1 = sparse(sparse_rows_1_P, sparse_cols_1_P, sparse_vals_1_P, P_nrows_1, P_ncols_1)


    # Store the pair for order 1
    push!(results, (spX_order_1, spP_order_1))

    if max_perturbation_order == 1
        return Tuple(results)
    end

    # --- Prepare for higher orders (Order 2 to max_perturbation_order) ---
    # Initialize map for Order 1: linear index in spX_order_1.nzval -> (row, (v1,))
    # This map is needed to trace indices for Order 2
    # We already built nz_lin_to_rc_1 above, reuse it and wrap the variable index in a Tuple
    nz_to_indices_prev = Dict{Int, Tuple{Int, Tuple{Int}}}()
     k_lin = 1
     for j = 1:size(spX_order_1, 2)
         for ptr = spX_order_1.colptr[j]:(spX_order_1.colptr[j+1]-1)
             r = spX_order_1.rowval[ptr]
             nz_to_indices_prev[k_lin] = (r, (j,)) # Store (equation row, (v1,))
             k_lin += 1
         end
     end

    nzvals_prev = spX_order_1.nzval # nzvals from Order 1 X-matrix

    # --- Iterate for orders n = 2, 3, ..., max_perturbation_order ---
    for n = 2:max_perturbation_order

        # Compute the Jacobian of the previous level's nzval w.r.t. 𝒳
        # This gives a flat matrix where rows correspond to non-zeros from order n-1 X-matrix
        # and columns correspond to the n-th variable we differentiate by (x_vn).
        sp_flat_curr_X = Symbolics.sparsejacobian(nzvals_prev, 𝒳) # nnz(spX_order_(n-1)) x nx

        # Build the nz_to_indices map for the *current* level (order n)
        # Map: linear index in sp_flat_curr_X.nzval -> (original_row_f, (v_1, ..., v_n))
        nz_to_indices_curr = Dict{Int, Tuple{Int, Tuple{Vararg{Int}}}}()
        k_lin_curr = 1 # linear index counter for nzval of sp_flat_curr_X
        # Iterate through the non-zeros of the current flat Jacobian
        for col_curr = 1:size(sp_flat_curr_X, 2) # Column index in sp_flat_curr_X (corresponds to v_n)
            for ptr_curr = sp_flat_curr_X.colptr[col_curr]:(sp_flat_curr_X.colptr[col_curr+1]-1)
                row_curr = sp_flat_curr_X.rowval[ptr_curr] # Row index in sp_flat_curr_X (corresponds to the row_curr-th nzval of previous level)

                # Get previous indices info from the map of order n-1
                prev_info = nz_to_indices_prev[row_curr]
                orig_row_f = prev_info[1] # Original equation row
                vars_prev = prev_info[2] # Tuple of variables from previous order (v_1, ..., v_{n-1})

                # Append the current variable index (v_n)
                vars_curr = (vars_prev..., col_curr) # Full tuple (v_1, ..., v_n)

                # Store info for the current level's non-zero
                nz_to_indices_curr[k_lin_curr] = (orig_row_f, vars_curr)
                k_lin_curr += 1
            end
        end

        # --- Construct the X-derivative sparse matrix for order n (compressed or uncompressed) ---
        local spX_order_n # Declare variable to hold the resulting X matrix
        local X_ncols_n # Number of columns in the resulting spX_order_n matrix

        if output_compressed
            # COMPRESSED output: nϵ x binomial(nx + n - 1, n)
            sparse_rows_n = Int[]
            sparse_cols_n = Int[] # This will store the compressed column index
            sparse_vals_n = Symbolics.Num[]

            # Calculate the total number of compressed columns for order n
            X_ncols_n = Int(binomial(nx + n - 1, n))

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
            for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                    val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Check the compression rule: v_n <= v_{n-1} <= ... <= v_1
                    is_compressed = true
                    for k_rule = 1:(n-1)
                         # Check v_{n-k_rule+1} <= v_{n-k_rule}
                        if var_indices_full[n-k_rule+1] > var_indices_full[n-k_rule]
                            is_compressed = false
                            break
                        end
                    end

                    if is_compressed
                        # Calculate the compressed column index c_n for the tuple (v_1, ..., v_n)
                        # using the derived formula: c_n = sum_{k=1}^{n-1} binomial(v_k + n - k - 1, n - k + 1) + v_n
                        compressed_col_idx = 0
                        for k_formula = 1:(n-1)
                             term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                             compressed_col_idx += term
                        end
                        # Add the last term: v_n (var_indices_full[n])
                        compressed_col_idx += var_indices_full[n]

                        push!(sparse_rows_n, orig_row_f)
                        push!(sparse_cols_n, compressed_col_idx)
                        push!(sparse_vals_n, val)
                    end

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                end
            end
            # Construct the compressed sparse matrix for order n
            spX_order_n = sparse(sparse_rows_n, sparse_cols_n, sparse_vals_n, nϵ, X_ncols_n)

        else # output_compressed == false
            # UNCOMPRESSED output: nϵ x nx^n
            sparse_rows_n_uncomp = Int[]
            sparse_cols_n_uncomp = Int[] # Uncompressed column index (1 to nx^n)
            sparse_vals_n_uncomp = Symbolics.Num[]

            # Total number of uncompressed columns
            X_ncols_n = Int(BigInt(nx)^n) # Use BigInt for the power calculation, cast to Int

            # Iterate through the non-zero entries of the current flat Jacobian (sp_flat_curr_X)
            k_flat_curr = 1 # linear index counter for nzval of sp_flat_curr_X
            for col_flat_curr = 1:size(sp_flat_curr_X, 2) # This corresponds to the n-th variable (v_n)
                for i_ptr_flat_curr = sp_flat_curr_X.colptr[col_flat_curr]:(sp_flat_curr_X.colptr[col_flat_curr+1]-1)
                    # row_flat_curr = sp_flat_curr_X.rowval[i_ptr_flat_curr] # Row index in sp_flat_curr_X
                    val = sp_flat_curr_X.nzval[i_ptr_flat_curr] # The derivative value

                    # Get the full info for this non-zero from the map
                    # The linear index in sp_flat_curr_X.nzval is k_flat_curr
                    orig_row_f, var_indices_full = nz_to_indices_curr[k_flat_curr] # (v_1, ..., v_n)

                    # Calculate the UNCOMPRESSED column index for the tuple (v_1, ..., v_n)
                    # This maps the tuple (v1, ..., vn) to a unique index from 1 to nx^n
                    # Formula: 1 + (v1-1)*nx^(n-1) + (v2-1)*nx^(n-2) + ... + (vn-1)*nx^0
                    uncompressed_col_idx = 1 # 1-based
                    power_of_nx = BigInt(nx)^(n-1) # Start with nx^(n-1) for v1 term
                    for i = 1:n
                        uncompressed_col_col_idx_term = (var_indices_full[i] - 1) * power_of_nx
                        # Check for overflow before adding
                        # if (uncompressed_col_idx > 0 && uncompressed_col_col_idx_term > 0 && uncompressed_col_idx + uncompressed_col_col_idx_term <= uncompressed_col_idx) ||
                        #    (uncompressed_col_idx < 0 && uncompressed_col_col_idx_term < 0 && uncompressed_col_idx + uncompressed_col_col_idx_term >= uncompressed_col_idx)
                        #    error("Integer overflow calculating uncompressed column index")
                        # end
                        uncompressed_col_idx += uncompressed_col_col_idx_term

                        if i < n # Avoid nx^-1
                            power_of_nx = div(power_of_nx, nx) # Integer division
                        end
                    end

                    push!(sparse_rows_n_uncomp, orig_row_f)
                    push!(sparse_cols_n_uncomp, Int(uncompressed_col_idx)) # Cast to Int
                    push!(sparse_vals_n_uncomp, val)

                    k_flat_curr += 1 # Increment linear index counter for sp_flat_curr_X.nzval
                end
            end
            # Construct the uncompressed sparse matrix for order n
            spX_order_n = sparse(sparse_rows_n_uncomp, sparse_cols_n_uncomp, sparse_vals_n_uncomp, nϵ, X_ncols_n)

        end # End of if output_compressed / else


        # --- Compute the P-derivative sparse matrix for order n ---
        # This is the Jacobian of the nzval of the intermediate flat X-Jacobian (sp_flat_curr_X) w.r.t. 𝒫.
        # sp_flat_curr_X.nzval contains expressions for d^n f_i / (dx_v1 ... dx_vn) for all
        # non-zero such values that were propagated from the previous step.
        spP_of_flatX_nzval_curr = Symbolics.sparsejacobian(sp_flat_curr_X.nzval, 𝒫) # nnz(sp_flat_curr_X) x np

        # Determine the desired dimensions of spP_order_n
        # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
        P_nrows_n = nϵ * X_ncols_n
        P_ncols_n = np

        sparse_rows_n_P = Int[] # Row index in the flattened space of spX_order_n (1 to P_nrows_n)
        sparse_cols_n_P = Int[] # Column index for parameters (1 to np)
        sparse_vals_n_P = Symbolics.Num[]

        # Iterate through the non-zero entries of spP_of_flatX_nzval_curr
        # Its rows correspond to the non-zeros in sp_flat_curr_X
        k_temp_P = 1 # linear index counter for nzval of spP_of_flatX_nzval_curr
        for p_col = 1:size(spP_of_flatX_nzval_curr, 2) # Column index in spP_of_flatX_nzval_curr (corresponds to parameter index)
            for i_ptr_temp_P = spP_of_flatX_nzval_curr.colptr[p_col]:(spP_of_flatX_nzval_curr.colptr[p_col+1]-1)
                temp_row = spP_of_flatX_nzval_curr.rowval[i_ptr_temp_P] # Row index in spP_of_flatX_nzval_curr (corresponds to the temp_row-th nzval of sp_flat_curr_X)
                p_val = spP_of_flatX_nzval_curr.nzval[i_ptr_temp_P] # The derivative w.r.t. parameter value

                # Get the full info for the X-derivative term that this P-derivative is from
                # temp_row is the linear index in sp_flat_curr_X.nzval
                # This corresponds to the derivative d^n f_orig_row_f / (dx_v1 ... dx_vn)
                orig_row_f, var_indices_full = nz_to_indices_curr[temp_row] # (v_1, ..., v_n)

                # We need to find the column index (X_col_idx) this term corresponds to
                # in the final spX_order_n matrix (which might be compressed or uncompressed)
                local X_col_idx # Column index in the final spX_order_n matrix (1 to X_ncols_n)

                if output_compressed
                    # Calculate the compressed column index
                    compressed_col_idx = 0
                    for k_formula = 1:(n-1)
                         term = binomial(var_indices_full[k_formula] + n - k_formula - 1, n - k_formula + 1)
                         compressed_col_idx += term
                    end
                    compressed_col_idx += var_indices_full[n]
                    X_col_idx = compressed_col_idx # The column in spX_order_n is the compressed one

                else # output_compressed == false
                    # Calculate the uncompressed column index
                    uncompressed_col_idx = 1
                    power_of_nx = BigInt(nx)^(n-1)
                    for i = 1:n
                        uncompressed_col_idx += (var_indices_full[i] - 1) * power_of_nx
                        if i < n
                            power_of_nx = div(power_of_nx, nx)
                        end
                    end
                    X_col_idx = Int(uncompressed_col_idx) # The column in spX_order_n is the uncompressed one
                end

                # Calculate the row index in spP_order_n
                # This maps the (orig_row_f, X_col_idx) pair in spX_order_n's grid to a linear index
                # Formula: (row_in_X - 1) * num_cols_in_X + col_in_X
                P_row_idx = (orig_row_f - 1) * X_ncols_n + X_col_idx

                # The column index in spP_order_n is the parameter index
                P_col_idx = p_col

                push!(sparse_rows_n_P, P_row_idx)
                push!(sparse_cols_n_P, P_col_idx)
                push!(sparse_vals_n_P, p_val)

                k_temp_P += 1 # Increment linear index counter for spP_of_flatX_nzval_curr.nzval
            end
        end

        # Construct the P-derivative sparse matrix for order n
        # Dimensions are (rows of spX_order_n * cols of spX_order_n) x np
        spP_order_n = sparse(sparse_rows_n_P, sparse_cols_n_P, sparse_vals_n_P, P_nrows_n, P_ncols_n)

        # Store the pair (X-matrix, P-matrix) for order n
        push!(results, (spX_order_n, spP_order_n))


        # Prepare for the next iteration (order n+1)
        # The nzvals for the next X-Jacobian step are the nzvals of the current flat X-Jacobian
        nzvals_prev = sp_flat_curr_X.nzval
        # The map for the next step should provide info for order n derivatives
        nz_to_indices_prev = nz_to_indices_curr

    end # End of loop for orders n = 2 to max_perturbation_order

    return Tuple(results) # Return results as a tuple of (X_matrix, P_matrix) pairs
end

using BenchmarkTools
calc! = Smets_Wouters_2007.jacobian[1]
ne = 230
nx = 404
np = 154

nth_order = take_nth_order_derivatives_with_params(calc!, nx, np, ne; max_perturbation_order = 1);
ss = similar(nth_order[1][1], Float64)
ss.nzval |> sum
nth_order = take_nth_order_derivatives_general(calc!, nx, np, ne; max_perturbation_order = 4, output_compressed = true)
nth_order[4]
@benchmark first_order = take_nth_order_derivatives(calc!, nx, np, ne; max_perturbation_order = 1)
@benchmark first_order = take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 1)
@benchmark first_order = take_nth_order_derivatives_general(calc!, nx, np, ne; max_perturbation_order = 1)


@benchmark first_order = take_nth_order_derivatives(calc!, nx, np, ne; max_perturbation_order = 2)
@benchmark first_order = take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 2)
@benchmark first_order = take_nth_order_derivatives_general(calc!, nx, np, ne; max_perturbation_order = 2)

@benchmark first_order = take_nth_order_derivatives(calc!, nx, np, ne; max_perturbation_order = 3)
@benchmark first_order = take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 3)
@benchmark first_order = take_nth_order_derivatives_sparse(calc!, nx, np, ne; max_perturbation_order = 3)
@benchmark first_order = take_nth_order_derivatives_general(calc!, nx, np, ne; max_perturbation_order = 3)

first_order[2][1]
first_order[1][2]
first_order[1]
first_order[2][1]
first_order[3]

Symbolics.@variables 𝒳[1:nx] 𝒫[1:np]

func = Symbolics.build_function(first_order[1], 𝒳, 𝒫,cse = true, expression = Val(false))

func2 = Symbolics.build_function(first_order[2], 𝒳, 𝒫,cse = true, expression = Val(false));
func3 = Symbolics.build_function(first_order[3], 𝒳, 𝒫,cse = true);#, expression = Val(false))
func3[2]
spp2 = sparse(first_order[2][2], first_order[2][3], first_order[2][1], ne, nx*(nx+1)/2)
spp2.colptr[231]
first_order_sp[2].colptr[231]

spp2.rowval[231]
first_order_sp[2].rowval[231]

Symbolics.simplify(spp2.nzval[231])
Symbolics.simplify(first_order_sp[2].nzval[231])
spp2.nzval[231] + first_order_sp[2].nzval[231]

anyzeros = Symbolics.simplify.(spp2.nzval)

anyzeros[295]
first_order_sp[2].nzval[294]

spp2 - first_order_sp[2]
@profview for i in 1:5 take_nth_order_derivatives(calc!, nx, np, ne; max_perturbation_order = 1) end

@profview take_nth_order_derivatives_sparse_compressed(calc!, nx, np, ne; max_perturbation_order = 3)
@profview take_nth_order_derivatives_general(calc!, nx, np, ne; max_perturbation_order = 3)

# Polyester.@batch for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
# for rc1 in 0:length(vars_X) * length(eqs_sub) - 1
for (c1, var1) in enumerate(𝒳)
    for (r, eq) in enumerate(ϵˢ)
    # r, c1 = divrem(rc1, length(vars_X)) .+ 1
    # var1 = vars_X[c1]
    # eq = eqs_sub[r]
        if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq, var1)
            
            push!(first_order, deriv_first)
            push!(row1, r)
            # push!(row1, r...)
            push!(column1, c1)
            if max_perturbation_order >= 2 
                for (c2, var2) in enumerate(𝒳)
                    if (((c1 - 1) * length(vars) + c2) ∈ second_order_idxs) && (Symbol(var2) ∈ Symbol.(Symbolics.get_variables(deriv_first)))
                        deriv_second = Symbolics.derivative(deriv_first, var2)
                        
                        push!(second_order, deriv_second)
                        push!(row2, r)
                        push!(column2, Int.(indexin([(c1 - 1) * length(vars) + c2], second_order_idxs))...)
                        if max_perturbation_order == 3
                            for (c3, var3) in enumerate(𝒳)
                                if (((c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3) ∈ third_order_idxs) && (Symbol(var3) ∈ Symbol.(Symbolics.get_variables(deriv_second)))
                                    deriv_third = Symbolics.derivative(deriv_second,var3)

                                    push!(third_order, deriv_third)
                                    push!(row3, r)
                                    push!(column3, Int.(indexin([(c1 - 1) * length(vars)^2 + (c2 - 1) * length(vars) + c3], third_order_idxs))...)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
    

import DifferentiationInterface as 𝒟
using SparseConnectivityTracer
using SparseMatrixColorings
backend = 𝒟.AutoSymbolics()
    
backend = 𝒟.AutoSparse(
    𝒟.AutoSymbolics();  # any object from ADTypes
    
)

ϵ = zeros(16)

prep = 𝒟.prepare_jacobian(calc!, ϵ, backend, ones(24), 𝒟.Constant(zeros(12)));
sparsity_pattern(prep)
𝒟.similar(sparsity_pattern(prep), eltype(ϵ))

spjac = Symbolics.sparsejacobian(e ,v)
jac = Symbolics.jacobian(e ,v)

spjacfunc = Symbolics.build_function(spjac, v, p, cse = true, skipzeros = true, expression = false)

jacfunc = Symbolics.build_function(jac, v, p, cse = true, skipzeros = true, expression = false)


spjacfunc[2](spjac,v,(p))

sphess = Symbolics.sparsejacobian((spjac.nzval) ,v)

hess = Symbolics.jacobian(vec(jac),v)

spthird = Symbolics.sparsejacobian((sphess.nzval),v)

spthird = Symbolics.sparsejacobian(vec(hess),v)


spjac.nzval

jacfunc[2](jac,(v),zero(p))
using LinearAlgebra
a = randn(3,3)

A = Symmetric((triu((a+a')/2)))
A.data.nzval
# get_std(m)

# aaa = get_std(m, derivatives = false, algorithm = :pruned_second_order)

aaaa = get_std(m, derivatives = false, algorithm = :pruned_second_order)

aaaa = get_std(m, derivatives = true, algorithm = :pruned_second_order)

# 𝓂.hessian[3].jac_exe# (zeros(24),zeros(28))


aaaa = get_std(m, derivatives = false, algorithm = :pruned_third_order)


aaaa = get_std(m, derivatives = true, algorithm = :pruned_third_order)



isapprox(aaa,aaaa)

isapprox(reshape(m.hessian[2],size(m.hessian[4])),m.hessian[4])

import MacroModelling: reshape_sparse_matrix, reshape_sparse_matrix_as_dense
# reshape_sparse_matrix(m.hessian[2])
reshape_sparse_matrix_as_dense(m.hessian[2])
sparse(reshape(m.hessian[2],size(m.hessian[4])))

isapprox(findnz(m.hessian[4])[2], findnz(m.model_hessian[2] * m.solution.perturbation.second_order_auxilliary_matrices.𝐔∇₂)[2]
)


using SparseArrays

function compute_mapping(A::SparseMatrixCSC)
    # A is assumed to be of size (n*m) x m.
    rows, m = size(A)
    @assert rows % m == 0 "The number of rows must be a multiple of the number of columns."
    n = div(rows, m)
    
    # Total number of nonzeros.
    nz = nonzeros(A)
    N = length(nz)
    
    # Reconstruct column indices from A.colptr.
    col_indices = Vector{Int}(undef, N)
    for j in 1:m
        for k in A.colptr[j]:(A.colptr[j+1]-1)
            col_indices[k] = j
        end
    end
    
    # Row indices are directly available.
    I = A.rowval
    J = col_indices

    # Compute new indices for the reshaped matrix B of size n x (m*m).
    # For each nonzero at position (r, c) in A:
    #   new row:  newI = floor((r-1)/m) + 1
    #   intra-block row: r_mod = r - (newI-1)*m
    #   new column: newJ = (c-1)*m + r_mod
    newI = floor.((I .- 1) ./ m) .+ 1
    newJ = (J .- 1) .* m .+ (I .- (newI .- 1) .* m)
    
    # Compute the sorted order permutation for CSC storage (sorted by newJ then newI).
    sorted_order = sortperm(1:N, by = i -> (newJ[i], newI[i]))
    
    # Build the mapping vector: for each nonzero in A (its internal order), mapping[k] is its
    # position in the nonzero vector of the reshaped matrix.
    mapping = Vector{Int}(undef, N)
    for rank in 1:N
        mapping[sorted_order[rank]] = rank
    end
    
    return mapping
end


# Example usage:
n = 3; m = 4
A = spzeros(n*m, m)
A[2, 1] = 10
A[5, 2] = 20
A[11, 3] = 30
A[12, 4] = 40

mapping = compute_mapping(m.hessian[2])

A = nonzeros(m.hessian[2])
A .= 1:36
aa = copy(sparse(reshape(m.hessian[2],16,24^2)))
AA = nonzeros(aa) 

A .- AA

nonzeros(aa) .- (1:36)
nonzeros(aa)[mapping]
println("Mapping vector from A's nonzeros to reshaped B's nonzeros:")
println(mapping)


SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix

import Zygote
import ForwardDiff
import FiniteDifferences
import DifferentiationInterface as 𝒟
import SparseMatrixColorings: GreedyColoringAlgorithm, sparsity_pattern
import SparseConnectivityTracer: TracerSparsityDetector

vcat(𝓂.parameter_values, 𝓂.solution.non_stochastic_steady_state[(end - length(𝓂.calibration_equations)+1):end], 𝓂.solution.non_stochastic_steady_state[1:(end - length(𝓂.calibration_equations))])

jac_deriv_zyg = Zygote.jacobian(x->calculate_jacobian(x, SS_and_pars, 𝓂), 𝓂.parameter_values)[1] # |> Matrix
jac_deriv_for = ForwardDiff.jacobian(x->calculate_jacobian(x, SS_and_pars, 𝓂), 𝓂.parameter_values) # |> Matrix

jac_deriv_zyg = Zygote.gradient(x->sum(abs2,calculate_jacobian(x, SS_and_pars, 𝓂)), 𝓂.parameter_values)[1] # |> Matrix
jac_deriv_for = ForwardDiff.gradient(x->sum(abs2,calculate_jacobian(x, SS_and_pars, 𝓂)), 𝓂.parameter_values) # |> Matrix
jac_deriv_fin = FiniteDifferences.grad(FiniteDifferences.central_fdm(3,1), x->sum(abs2,calculate_jacobian(x, SS_and_pars, 𝓂)), 𝓂.parameter_values)[1] # |> Matrix


jac_deriv_zyg - jac_deriv_for

m.jacobian_SS_and_pars_vars[3]

parameters = 𝓂.parameter_values

backend = 𝒟.AutoSparse(
    𝒟.AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector = TracerSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm(),
)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

∂ = 𝒟.Constant(vcat(SS_and_pars[vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)], shocks_ss))
C = vcat(parameters, 𝓂.solution.non_stochastic_steady_state[(end - length(𝓂.calibration_equations)+1):end], 𝓂.solution.non_stochastic_steady_state[1:(end - length(𝓂.calibration_equations))]) # [dyn_ss_idx])

𝒟.jacobian!(𝓂.jacobian_SS_and_pars_vars[1], 𝓂.jacobian_SS_and_pars_vars[2], 𝓂.jacobian_SS_and_pars_vars[3], backend, C, ∂)

𝓂.jacobian_SS_and_pars_vars[1](C, vcat(SS_and_pars[vcat(dyn_var_future_idx, dyn_var_present_idx, dyn_var_past_idx)], shocks_ss))

SS(m)

get_solution(m)
get_solution(m)
# 9*x-7*i>3*(3*x-7*u)
# -7*i>3*(3*x-7*u) - 9*x
# 7*i> - 3*(3*x-7*u) + 9*x

# i > - (3*(3*x-7*u) - 9*x) / 7
# i > - 3 / 7 * ((3*x-7*u) - 3*x)
# i > - 3 / 7 * (3*x-7*u - 3*x)
# i > - 3 / 7 * (-7*u )
# i >  3 * u
# i <3 * u

𝓂.model_jacobian_SS_and_pars_vars[2]
𝓂.jacobian_SS_and_pars_vars[2]'

# (:Pibar)   1.00083          0.0       0.332779      0.0       0.00111065   0.0         0.0             0.0       0.0    -0.332779     0.0
include("../models/NAWM_EAUS_2008.jl")

𝓂 = NAWM_EAUS_2008




SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)


parameters = 𝓂.parameter_values
StSt = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]

par = vcat(parameters, calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

# X = [SS[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; SS[dyn_ss_idx]; par; shocks_ss]

deriv_vars = vcat(StSt[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],shocks_ss)
SS_and_pars = vcat(par, StSt)#[dyn_ss_idx])

C = 𝒟.Constant(SS_and_pars)

backend = 𝒟.AutoFastDifferentiation()

𝒟.jacobian!(𝓂.jacobian[1], 𝓂.jacobian[2], 𝓂.jacobian[3], 𝓂.jacobian[4], backend, deriv_vars, C)



∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
@benchmark calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

∇₁s = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
@benchmark calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

isapprox(∇₁s,∇₁)

@model Tmp begin
    ## Resource constraint
    ynetm[0] * 1 + c[0] * (-c_yc/(ynetmcons)) + sk[0] * ( -sk_yc/(ynetmcons) ) + sc[0] * (-sc_yc/(ynetmcons) ) + hc[0] * (- actualhc_yc/(ynetmcons) ) + zc[0] * ( -ac_zc/(1-ac_zc)*actualhc_yc/(ynetmcons) ) + ac[0] * (inv(zc_ac-1)*actualhc_yc/(ynetmcons) ) + hk[0] * (- actualhk_yc/(ynetmcons) ) + zk[0] * ( -ak_zk/(1-ak_zk)*actualhk_yc/(ynetmcons) ) + ak[0] * (inv(zk_ak-1)*actualhk_yc/(ynetmcons) ) = 0
    ## Euler equation
    c[0] * (1) + c[1] * (-1) + r[0] * (1) = 0
    ## Aggregate production function FIXME [0], [-1] or SS (muc and nc)
    yc[0] * (1) + chiz[0] * (-1/(1-bb)) + k[-1] * (-al) + l[0] * (  al) + ly[0] * (-1) + nc[0] * (-(muc[0]-1)/(1-bb)) + muc[0] * ((bb+muc[0]*log(nc[0]))/(1-bb)) + u[0] * (-al) + ac[0] * (-bb/(1-bb)*(theta-1)) = 0
    ## demand of capital
    yc[0] * (1) + kl[0] * (-1) + l[0] * ( 1) + ly[0] * (-1) + muc[0] * (-1) + d[0] * (- (1/(1+del/d_pk))) + pk[0] * (- (del/(d_pk+del))) + u[0] * (-edu*(1/(d_pk/del+1))) = 0
    ## capacity choice
    yc[0] * (1) + u[0] * (-(1+edp)) + muc[0] * (-1) + kl[0] * (-1) + l[0] * ( 1) + ly[0] * (-1) + pk[0] * (-1) = 0
    ## labor demand in final output
    yc[0] * (1) + muc[0] * (-1) + ly[0] * (-1) + w[0] * (-1) = 0
    ## production of new investment goods FIXME [0], [-1] or SS (muk and nk)
    yk[0] * (1) + chi[0]  * (-1/(1-bb)) + kl[0] * (- al) + l[0] * (al-1/(1-lc_l)) + u[0] * (- al) + ly[0] * (lc_l/(1-lc_l)) + nk[0] * (-(muk[0]-1)/(1-bb)) + muk[0] * ((bb+muk[0]*log(nk[0]))/(1-bb)) + pk[0] * (-bb/(1-bb)) + ak[0] * (-bb/(1-bb)*(th-1)) = 0
    ## Real wage
    w[0] * (1) + l[0] * (-fi) + c[0] * (-1) + muw[0] * (-1) = 0
    ## Profits embodied
    prk[0] * (1) + yk[0] * (-1) + pk[0] * (-1) + muk[0] * (1) = 0
    ## Profits disembodied
    prc[0] * (1) + yc[0] * (-1) + muc[0] * (1) = 0
    ## Value of an adopted innovation for embodied
    vk[0] * (1) + ak[0] * (-(1-prk_vk)) + prk[0] * (-prk_vk) + vk[1] * (-(1-prk_vk)) + ak[1] * ((1-prk_vk)) + r[0] * ((1-prk_vk)) = 0
    ## Value of an adopted innovation for disembodied
    vc[0] * (1) + ac[0] * (-(1-prc_vc)) + prc[0] * (-prc_vc) + vc[1] * (-(1-prc_vc)) + ac[1] * ((1-prc_vc)) + r[0] * ((1-prc_vc)) = 0
    ## Capital accumulation
    k[0] * ( 1) + u[0] * ( edu*del/(1+gk)) + k[-1] * (-jcof) + yk[0] * (-(1-jcof)) = 0
    ## Law of motion for embodied productivity
    zk[0] * ( 1) + zk[-1] * (-1) + sk[-1] * (-rho*(gzk+ok)/(1+gzk)) + sf[-1] * ( rho*(gzk+ok)/(1+gzk)) + chik[-1] * (-(gzk+ok)/(1+gzk)) = 0
    ## Law of motion for disembodied productivity
    zc[0] * ( 1) + zc[-1] * (-1) + sc[-1] * (-rho*(gzc+oc)/(1+gzc)) + sf[-1] * ( rho*(gzc+oc)/(1+gzc)) = 0
    ## Free entry for embodied
    sk[0] * ( 1-rho) + zk[0] * (-1) + sf[0] * ( rho) + jk[1] * (-1) + zk[1] * (1) + r[0] * (1) = 0
    ## Free entry for disembodied
    sc[0] * (1-rho) + zc[0] * (-1) + sf[0] * (rho) + jc[1] * (-1) + zc[1] * (1) + r[0] * (1) = 0
    ## Bellman for not adopted disemb innovation
    jc[0] * (-1) + hc[0] * (-(hc_jc+phic*elc*lamc/R*rz*(1-zc_ac*vc_jc))) + r[0] * (-(1+hc_jc)) + zc[0] * ( phic*rz*((1-lamc)+lamc*zc_ac*vc_jc)/R) + ac[1] * (-phic*lamc*rz*zc_ac*vc_jc/R) + vc[1] * ( phic*lamc*rz*zc_ac*vc_jc/R) + sf[0] * (-phic*elc*lamc*rz/R*(zc_ac*vc_jc-1)) + zc[1] * (-phic*rz*(1-lamc)/R) + jc[1] * ( phic*rz*(1-lamc)/R) = 0
    ## law of motion for adopted disembodied innvo
    ac[0] * (1) + ac[-1] * (-phic*(1-lamc)/(1+gzc)) + hc[-1] * (-elc*lamc*((phic/(1+gzc))*zc_ac-phic/(1+gzc))) + sf[-1] * (elc*lamc*((phic/(1+gzc))*zc_ac-phic/(1+gzc))) + zc[-1] * (-(1-phic*(1-lamc)/(1+gzc))) = 0
    ## optimal investment in adoption of disemb innov
    zc[0] * (1) + sf[0] * (-(1+ellc)) + r[0] * (-1) + hc[0] * (ellc) + vc[1] * (1/(1-jc_vc*ac_zc)) + ac[1] * (-1/(1-jc_vc*ac_zc)) + jc[1] * (-1/(vc_jc*zc_ac-1)) + zc[1] * (1/(vc_jc*zc_ac-1))
    ## Bellman for not adopted emb innovation
    jk[0] * (-1) + hk[0] * (-(hk_jk+(1-ok)*elk*lamk/R*ra*(1-zk_ak*vk_jk))) + r[0] * (-(1+hk_jk)) + zk[0] * (phik*ra*((1-lamk)+lamk*zk_ak*vk_jk)/R) + ak[1] * (-phik*lamk*ra*zk_ak*vk_jk/R) + vk[1] * (phik*lamk*ra*zk_ak*vk_jk/R) + sf[0] * (- phik*elk*lamk*ra/R*(zk_ak*vk_jk-1)) + zk[1] * (-phik*ra*(1-lamk)/R) + jk[1] * (phik*ra*(1-lamk)/R) = 0
    ## law of motion for adopted embodied innvo
    ak[0] * (1) + ak[-1] * (-phik*(1-lamk)/(1+gzk)) + hk[-1] * (-elk*lamk*((phik/(1+gzk))*zk_ak-phik/(1+gzk))) + sf[-1] * (elk*lamk*((phik/(1+gzk))*zk_ak-phik/(1+gzk))) + zk[-1] * (-(1-phik*(1-lamk)/(1+gzk))) = 0
    ## optimal investment in adoption of emb innov
    zk[0] * (1) + sf[0] * (-(1+ellk)) + r[0] * (-1) + hk[0] * (ellk) + vk[1] * (1/(1-jk_vk*ak_zk)) + ak[1] * (-1/(1-jk_vk*ak_zk)) + jk[1] * (-1/(vk_jk*zk_ak-1)) + zk[1] * (1/(vk_jk*zk_ak-1)) = 0
    ## Arbitrage
    pk[0] * (1) + r[0] * (1) + d[1] * (- (R-1-gpk)/R) + pk[1] * (-(1+gpk)/R) = 0
    ## entry into final goods sector
    muc[0] * (1) + yc[0] * (mucof) + sf[0] * (-mucof) + nc[0] * (-mucof) = 0
    ## m
    muc[0] * (1) + nc[0] * (-etamuc) = 0
    ## entry into capital goods sector
    muk[0] * (1) + yk[0] * (mukcof) + pk[0] * (mukcof) + sf[0] * (-mukcof) + nk[0] * (-mukcof) = 0
    ## mk
    muk[0] * (1) + nk[0] * (-etamuk) = 0
    ## equivalence between klzero and jlag
    kl[0] * (1) + k[-1] * (-1) = 0
    ## Definition of output net of total overhead costs
    ynet[0] * (1) + yc[0] * (-1/(1-oc_yc)) + nc[0] * (occ_yc/(1-oc_yc)) + nk[0] * (ock_yc/(1-oc_yc)) + sf[0] * (oc_yc/(1-oc_yc)) = 0
    ## definition of scaling factor
    sf[0] * (1) + kl[0] * (-1) + ak[0] * (-bb*(1-th)) + ac[0] * (bb*(1-theta)) = 0
    ## definition of ynetm
    ynetm[0] * (1) + ynet[0] * (- 1/(1-mc_yc*inv(ynet_yc)-mk_yc*inv(ynet_yc))) + yc[0] * (mc_yc/ynetmcons) + muc[0] * (-mc_yc/ynetmcons) + pk[0] * (mk_yc/ynetmcons) + yk[0] * (mk_yc/ynetmcons) + muk[0] * (-mk_yc/ynetmcons) = 0
    ## Definition of total value added
    yT[0] * (1) + ynetm[0] * (-ynetmcons/(ynetmcons+pkyk_yc)) + pk[0] * (-pkyk_yc/(ynetmcons+pkyk_yc)) + yk[0] * (-pkyk_yc/(ynetmcons+pkyk_yc)) = 0
    ## labor demand in capital goods production
    yk[0] * (1) + pk[0] * (1) + muk[0] * (-1) + w[0] * (-1) + l[0] * (- 1/(1-lc_l)) + ly[0] * (lc_l/(1-lc_l)) = 0
    # ## embodied productivity shock process
    chi[0] = ρᵡ * chi[-1] + σᵡ* eps_chi[x] = 0
    # ## Labor augmenting technology shock process
    chiz[0] = ρᶻᵪ * chiz[-1] + σᶻᵪ * eps_chi_z[x] = 0
    # ## Wage markup shock process
    muw[0] = muw[-1] * ρᵐʷ + σᵐʷ * eps_muw[x] = 0
    # ## Wage markup shock process
    chik[0] = ρᵏᵪ * chik[-1] + σᵏᵪ * eps_chi_k[x] = 0

end

@parameters Tmp begin
    bet    = 0.95 	     ## discount factor
    del    = 0.08 	     ## depreciation rate
    fi     = 1          ## labor supply curvature
    al     = 1/3        ## k share
    g_y    = 0.2*0.7    ## ss g/y ratio
    th     = 1/0.6      ## elasticity of substitution intermediate good sector
    rho    = 0.9        ## parameter embodied technology
    eta    = 0.0
    theta  = 1/0.6
    # muw[ss]    = 1.2        ## ss wage markup
    muw_ss    = 1.2        ## ss wage markup
    # muc[ss]    = 1.1
    muc_ss = 1.1
    # nc[ss]     = 1
    nc_ss = 1
    # nk[ss]     = 1
    nk_ss     = 1
    dmuc   = -muc_ss
    etamuc = dmuc*nc_ss/muc_ss
    boc    = (muc_ss-1)/muc_ss
    # muk[ss]    = 1.2
    muk_ss    = 1.2
    etamuk = etamuc
    lamk   = 0.1
    lamc   = 0.1
    elk    = 0.9
    elc    = 0.9
    ellk   = elk-1
    ellc   = elc-1
    o  = 0.03
    oz = 0.03
    oc = 0.03
    ok = 0.03
    phic   = 1-oc
    phik   = 1-ok
    bb     = 0.5 ## intermediate share in final output
    ## Nonstochastic steady state
    gpk    = -0.026
    gy     =  0.024
    gk     =  gy - gpk
    gzc    = (gy-al*gk)/bb*(1-bb)/(theta-1)
    gzk    = (gpk-gzc*bb*(theta-1))/(bb*(1-th))
    gtfp   =  gy-al*gk+gzk*(al*bb*(th-1))/(1-al*(1-bb))
    measbls = (0.014-gy+al*gk)/(gzk*(al*bb*(th-1))/(1-al*(1-bb)))
    gv     = gy
    gvz    = gy
    R      = (1+gy)/bet
    d_pk   = R-(1+gpk)                   ## definition of R
    yc_pkkc = muc_ss/(al*(1-bb))*(d_pk+del) ## foc for k
    yk_kk   = muk_ss/(al*(1-bb))*(d_pk+del) ## new capital to capital in capital production sector
    yk_k   = (gk+del)/(1+gk)             ## new capital to capital ratio
    kk_k   = yk_k/yk_kk                  ## share of capital in capital production.
    kc_k   = 1-kk_k
    kk_kc  = kk_k/kc_k
    lk_lc  = kk_kc
    lk_l   = lk_lc/(lk_lc+1)
    lc_l   = 1-lk_l
    pkyk_yc= kk_kc*muk_ss/muc_ss
    mk_yc  = bb*1/th*pkyk_yc/muk_ss
    mc_yc  = bb*1/theta/muc_ss
    pkk_yc = inv(yc_pkkc)/kc_k
    pik_yc = pkk_yc*muc_ss/muk_ss              ## value of total capital stock removing fluctuations in relative price of capital due to markup variations
    prk_yc   = pkyk_yc*(1-1/th)*bb/muk_ss
    prc_yc  = (1-1/theta)*bb/muc_ss
    prk_vk   = 1-(1+gv)*phik/((1+gzk)*R) ## bellman for va
    prc_vc = 1-(1+gvz)*phic/((1+gzc)*R) ## bellman for vz
    yc_vk    = prk_vk*inv(prk_yc)
    yc_vc   = prc_vc*inv(prc_yc)
    zk_ak   = ((gzk+ok)/(lamk*phik)+1)
    zc_ac   = ((gzc+oc)/(lamc*phic)+1)
    ac_zc   = inv(zc_ac)
    ak_zk   = inv(zk_ak)
    ra     = (1+gy)/(1+gzk)
    rz     = (1+gy)/(1+gzc)
    jk_yc    = inv(1-elk*phik*lamk*ra/R-(1-lamk)*phik*ra/R)*(1-elk)*phik*lamk*ra*zk_ak/R*inv(yc_vk) ## zk * jk /yc bellman for not adopted innov
    jc_yc   = inv(1/phic-elc*lamc*rz/R-(1-lamc)*rz/R)*(1-elc)*lamc*rz*zc_ac/R*inv(yc_vc) ## zc*jc/yc bellman for not adopted innov
    hk_yc    = phik*elk*lamk*ra/R*(inv(yc_vk)*zk_ak-jk_yc) ## zk *hk/yc
    hc_yc    = phic*elc*lamc*rz/R*(inv(yc_vc)*zc_ac-jc_yc) ## zc *hc/yc
    sk_yc    = jk_yc*(gzk+o)*(1+gv)*inv((1+gzk)*R) ## from free entry cond't
    sc_yc   = jc_yc*(gzc+oz)*(1+gvz)*inv((1+gzc)*R)
    hc_jc  = hc_yc/jc_yc
    hk_jk  = hk_yc/jk_yc
    vc_jc  = inv(yc_vc)/jc_yc
    vk_jk  = inv(yc_vk)/jk_yc
    jc_vc=inv(vc_jc)
    jk_vk=inv(vk_jk)
    bock   = boc*pkyk_yc*(muk_ss-1)*muc_ss/(muk_ss*(muc_ss-1))
    occ_yc=boc*pik_yc
    ock_yc=bock*pik_yc
    oc_yc=occ_yc+ock_yc
    c_yc = 1-oc_yc-g_y-mc_yc-mk_yc-sk_yc-sc_yc-((phic/(1+gzc))^2-inv(zc_ac))*hc_yc-((phik/(1+gzk))^2-inv(zk_ak))*hk_yc
    pi_yc=(muc_ss-1)/muc_ss-oc_yc
    # u[ss]=.8
    u_ss=.8
    edu=al*(1-bb)*yc_pkkc/(muc_ss*del) ## from foc wrt utilization, edu = elasticity of depreciation with respect to capacity
    edup=0 ## partial of edu wrt u
    edp=1/3##(edu)-1+edup/(edu*u); ## elasticity of del' (i.e. elasticity of delta prima)
    actualhk_yc=hk_yc*(1-ak_zk) ## total expenses in adoption of capital specific innovations
    actualhc_yc=hc_yc*(1-ac_zc) ## total expenses in adoption of consumption specific innovations
    inv_Y=pkyk_yc/(pkyk_yc+1-mc_yc-mk_yc-occ_yc-ock_yc) ## investment output ratio
    Y_yc=pkyk_yc/inv_Y
    ynet_yc=1-oc_yc
    ## Coefficients for the log-linearization
    qcof = (1-del)*(1+gpk)/R
    jcof = (1-del)/(1+gk)
    vcof = (1+gy)/((1+gzk)*R)
    vzcof= (1+gy)/((1+gzc)*R)
    mucof= muc_ss-1
    mukcof=muk_ss-1
    ycof=(ynet_yc-mc_yc-mk_yc+pkyk_yc-actualhk_yc-actualhc_yc-sk_yc-sc_yc)^(-1)
    ynetmcons=1-oc_yc-mc_yc-mk_yc ## fraction of ynetm in y
    # NOTE Shock to Embodied Technology
    ρᵡ = (0.7)^4   # autoregressive component
    σᵡ = 0.01    # standard deviation
    # Disembodied Technology Shock
    ρᶻᵪ = (0.7)^4
    σᶻᵪ = 0.01
    # Wage markup shock
    ρᵐʷ   = 0.60 # (rhow)
    σᵐʷ = 0.01
    # Chik shock
    ρᵏᵪ = 0.8
    σᵏᵪ = 0.01
end

𝓂 = Tmp




import MacroModelling: match_pattern, get_symbols, normcdf, normpdf, norminvcdf, norminv, qnorm, normlogpdf, normpdf, normcdf, pnorm, dnorm, erfc, erfcinv, solve_quadratic_matrix_equation, get_NSSS_and_parameters
import Symbolics



future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎$")))
present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))
past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎$")))
shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎$")))
ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎$")))

sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))


vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                dyn_present_list[indexin(sort(present),present)],
                dyn_past_list[indexin(sort(past),past)],
                dyn_exo_list[indexin(sort(exo),exo)])

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

final_indices = vcat(𝓂.parameters, SS_and_pars_names_lead_lag)


# Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

# # overwrite SymPyCall names
# input_args = vcat(future_varss,
#                     present_varss,
#                     past_varss,
#                     ss_varss,
#                     𝓂.parameters,
#                     𝓂.calibration_equations_parameters,
#                     shock_varss)

# eval(:(Symbolics.@variables $(input_args...)))

# Symbolics.@variables 𝔛[1:length(input_args)]

# calib_eq_no_vars = reduce(union, get_symbols.(𝓂.calibration_equations_no_var), init = []) |> collect

# eval(:(Symbolics.@variables $((vcat(SS_and_pars_names_lead_lag, calib_eq_no_vars))...)))

# vars = eval(:(Symbolics.@variables $(vars_raw...)))

# eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

# input_X = Pair{Symbolics.Num, Symbolics.Num}[]
# input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]

# for (v,input) in enumerate(input_args)
#     push!(input_X, eval(input) => eval(𝔛[v]))

#     if input ∈ shock_varss
#         push!(input_X_no_time, eval(𝔛[v]) => 0)
#     else
#         input_no_time = Symbol(replace(string(input), r"₍₁₎$"=>"", r"₍₀₎$"=>"" , r"₍₋₁₎$"=>"", r"₍ₛₛ₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

#         vv = indexin([input_no_time], final_indices)
        
#         if vv[1] isa Int
#             push!(input_X_no_time, eval(𝔛[v]) => eval(𝔛[vv[1]]))
#         end
#     end
# end

# vars_X = map(x -> Symbolics.substitute(x, input_X), vars)

# eqs
# sort_order = calculate_kahn_topological_sort_order(𝓂.calibration_equations_no_var)

# sort_order = sorted_indices

# all_vars = union(future, present, past)
# sort!(all_vars)


pars_and_SS = Expr[]
for (i, p) in enumerate(vcat(𝓂.parameters, 𝓂.calibration_equations_parameters))
    push!(pars_and_SS, :($p = parameters_and_SS[$i]))
end

nn = length(pars_and_SS)

for (i, p) in enumerate(dyn_ss_list[indexin(sort(stst),stst)])
    push!(pars_and_SS, :($p = parameters_and_SS[$(i + nn)]))
end



deriv_vars = Expr[]
# for (k, m) in enumerate(["₍₁₎", "₍₀₎", "₍₋₁₎"])
    for (i, u) in enumerate(vars_raw)
        # push!(deriv_vars, :($(Symbol(string(u) * m)) = variables[$(i + (k-1) * length(vars_raw))]))
        push!(deriv_vars, :($u = variables[$i]))
    end
# end

# for (i, u) in enumerate(dyn_exo_list)
#     push!(deriv_vars, :($u = variables[$(i + 3 * length(vars_raw))]))
# end


eeqqss = Expr[]
for (i, u) in enumerate(𝓂.dyn_equations)
    push!(eeqqss, :(ϵ[$i] = $u))
end



dyn_eqs = :(function model_dynamics!(ϵ, variables, parameters_and_SS)
    @inbounds begin
        $(pars_and_SS...)
        $(𝓂.calibration_equations_no_var...)
        $(deriv_vars...)
        $(eeqqss...)
    end
    return nothing # [$(𝓂.dyn_equations...)]
end)


eval(dyn_eqs)



using DifferentiationInterface
using BenchmarkTools
using Symbolics, ForwardDiff, Zygote, Enzyme, FastDifferentiation, Mooncake, SparseMatrixColorings, SparseConnectivityTracer


stst = get_irf(𝓂, shocks = :none, variables = :all, levels = true, periods = 1) |> collect
stst_and_calib_pars = SS(𝓂, derivatives = false) |> collect

# stst = stst_and_calib_pars[1:end-length(𝓂.calibration_equations_parameters)]

calib_pars = stst_and_calib_pars[end-length(𝓂.calibration_equations_parameters)+1:end]

jac = zeros(length(𝓂.dyn_equations), length(deriv_vars));

ϵ = zeros(length(𝓂.dyn_equations))

SS_and_pars, (iters, tol_reached) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)

STST = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]

par = vcat(𝓂.parameter_values, calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

X = [STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; STST[dyn_ss_idx]; par; shocks_ss]

deriv_vars = vcat(STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],shocks_ss)
SS_and_pars = vcat(par, STST) # [dyn_ss_idx])

# @benchmark model_dynamics!(ϵ, 
                            # vcat(STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],shocks_ss), 
                            # vcat(par, STST[dyn_ss_idx]))




backend = AutoFastDifferentiation()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, zero(deriv_vars), Constant(zero(SS_and_pars))); # 3.3s

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prephess = prepare_jacobian(prep.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

@time prep3rd = prepare_jacobian(prephess.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

jac_buffer = similar(sparsity_pattern(prephess), eltype(stst))

backend = AutoSparse(
    # AutoForwardDiff();  # any object from ADTypes
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prephess = prepare_jacobian(prep.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

@time prep3rd = prepare_jacobian(prephess.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

jacobian!(jac_deriv!, jac, jac_buffer, prephess, backend, SS_and_pars, C)



prep.jac_exe(deriv_vars, SS_and_pars)

jac_deriv(SS_and_pars, deriv_vars) = 𝓂.jacobian[4].jac_exe(deriv_vars, SS_and_pars)

jac2 = zeros(length(𝓂.dyn_equations) * length(deriv_vars), length(SS_and_pars))

C = Constant(deriv_vars)

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep2 = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.
jac_buffer = similar(sparsity_pattern(prep2), eltype(stst))


backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep2 = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.


prep.jac_exe(deriv_vars, SS_and_pars)

prep.jac_exe!(jac, deriv_vars, SS_and_pars)

jac_deriv(SS_and_pars, derivvars) = prep.jac_exe(derivvars, SS_and_pars)

jac_deriv!(jacc, SS_and_pars, derivvars) = prep.jac_exe!(jacc, derivvars, SS_and_pars)

C = Constant(deriv_vars)

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv!, jac, backend, SS_and_pars, C); # 3.

jac_buffer = similar(sparsity_pattern(prepjac), eltype(deriv_vars))

backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv!, jac, backend, SS_and_pars, C); # 3.

jacobian!(jac_deriv!, jac, jac_buffer, prepjac, backend, SS_and_pars, C)

@benchmark jacobian!(jac_deriv!, jac, jac_buffer, prepjac, backend, SS_and_pars, C)



backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.

jacobian!(jac_deriv, jac_buffer, prepjac, backend, SS_and_pars, C)

@benchmark jacobian!(jac_deriv, jac_buffer, prepjac, backend, SS_and_pars, C)


DifferentiationInterface.jacobian(jac_deriv, prep2, backend, SS_and_pars, Constant(deriv_vars))

# @benchmark model_dynamics(vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), vcat(𝓂.parameter_values, calib_pars, stst))

prepare_jacobian(x->x, AutoForwardDiff(), [0])

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
# DifferentiationInterface.jacobian!(model_dynamics, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))
jac_buffer = similar(sparsity_pattern(prep), eltype(stst))

# prep

backend = AutoSparse(
    # AutoForwardDiff();  # any object from ADTypes
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoFastDifferentiation()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, zero(deriv_vars), Constant(zero(SS_and_pars))); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars))


𝓂.jacobian[1]
𝓂.jacobian[3]
jacobian!(𝓂.jacobian... ,backend, deriv_vars, Constant(SS_and_pars))
𝓂.jacobian[3]
C = Constant(SS_and_pars)
@benchmark jacobian!(𝓂.jacobian... ,backend, deriv_vars, Constant(SS_and_pars))
@benchmark jacobian!(𝓂.jacobian[1], 𝓂.jacobian[2], 𝓂.jacobian[3], 𝓂.jacobian[4], backend, deriv_vars, C)


backend = AutoSparse(
    AutoSymbolics();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoForwardDiff()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoForwardDiff()

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # 1.3s
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))



backend = AutoMooncake(; config=nothing)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))



backend = AutoMooncake(; config=nothing)

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # crashes
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))


backend = AutoZygote()

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # 1.3s
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))


backend = AutoEnzyme() # (; mode=pushforward, function_annotation=Nothing)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 0.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # forever
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))



# Reorder the calibration equations accordingly so that for each equation,
# all unknowns on its right-hand side have been defined by an earlier equation.
sorted_calibration_equations_no_var = 𝓂.calibration_equations_no_var[sorted_indices]





eqs_sub = Symbolics.Num[]
for subst in eqs
    for _ in calib_eqs
        for calib_eq in calib_eqs
            subst = Symbolics.substitute(subst, calib_eq)
        end
    end
    # subst = Symbolics.fixpoint_sub(subst, calib_eqs)
    subst = Symbolics.substitute(subst, input_X)
    push!(eqs_sub, subst)
end

if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    end
end

first_order = Symbolics.Num[]
second_order = Symbolics.Num[]
third_order = Symbolics.Num[]
row1 = Int[]
row2 = Int[]
row3 = Int[]
column1 = Int[]
column2 = Int[]
column3 = Int[]

for (c1, var1) in enumerate(vars_X)
    for (r, eq) in enumerate(eqs_sub)
        if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq, var1)
        end
    end
end



## SVD tests (doesnt work)


using LinearAlgebra
SSVVDD = jac_buffer[1:230,1:230] |> collect |> svd
cutoff = 1-1e-7
n_cutoff = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,SSVVDD.S[1:i]) / sum(abs2,SSVVDD.S) > cutoff ? break : n_cutoff += 1
end
(sum(abs2,SSVVDD.S) - sum(abs2,SSVVDD.S[1:30])) / sum(abs2,SSVVDD.S)

SSVVDD.S[31:90]
backend = AutoSymbolics()


n_cutoff = 120
# SSVVDD.U[:,1:n_cutoff]
# SSVVDD.V[:,1:n_cutoff]

A = jac_buffer[1:230,1:230] |> collect
B = jac_buffer[1:230,231:460] |> collect
C = jac_buffer[1:230,461:690] |> collect

# Compute the singular value decomposition of A.
U, S, V = svd(B)
# Determine effective rank r: count singular values above tol.
cutoff = 1-1e-7
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end

# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
# S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V

# In our quadratic equation, we assume that the highest‐order term
# (multiplying X^2) is dominated by A.
# Represent X in the reduced space as:
#    X ≈ U_r * X_tilde * V_r'
#
# To derive a reduced quadratic equation, substitute:
#    X_tilde ≈ the unknown (r×r) matrix,
# and (assuming that V_r' * U_r ≈ I) approximate:
#    X^2 ≈ U_r * (X_tilde^2) * V_r'.
#
# Project the matrices:
A_r = U_r' * A * U_r      # (r x r)
B_r = U_r' * B * V_r      # (r x r)
C_r = V_r' * C * V_r      # (r x r)

X̃, iter, reached_tol = solve_quadratic_matrix_equation(A_r, B_r, C_r, Val(:doubling), 𝓂.timings)#, initial_guess = randn(size(A_r)))

reached_tol
X̃

get_solution(𝓂)

# Define the function for the reduced quadratic equation:
# F( X_tilde ) = A_r * X_tilde^2 + B_r * X_tilde + C_r
norm(A_r * (X̃ * X̃) + B_r * X̃ + C_r) / max(norm(X̃), norm(A_r))


n = size(A,1)

U, S, V = svd(A)


r = sum(S .> 1e-12)

r = 10
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V


# S = Diagonal(S_vec)  # S is diagonal with singular values
# Define Q = V' * U (which is orthogonal)
Q_r = V_r' * U_r

# Create an arbitrary matrix X in the full space
# X = randn(n, n)
# Express X in the transformed coordinates: let Y = U' * X * V, so that X = U * Y * V'
Y = U_r' * XX * V_r


U_r' / Y / (V_r) - XX

# Compute the original residual of the quadratic equation:
R_full = A * XX^2 + B * XX + C

U_r' * A * XX^2 * V_r   +    U_r' * B * XX * V_r    +    U_r' * C * V_r

U_r' \ V_r' * XX * U_r * V_r
# In the transformed space, note that:
#   X^2 = U * Y * (V' * U) * Y * V' = U * Y * Q * Y * V'
# Thus, the transformed (projected) equation is:
# R_proj = S * (Q * Y * Q * Y) + (U' * B * U) * Y + (U' * C * V)
R_proj = S_r * Q_r * Y * Q_r * Y + (U_r' * B * U_r) * Y + (U_r' * C * V_r)

norm(R_full)
norm(R_proj)

U, S, V = svd(XX)

# Determine effective rank r: count singular values above tol.
cutoff = 1-1e-12
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end

# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V

U_r * S_r * V_r' - XX
U * Diagonal(S) * V' - XX

norm(A * (U_r * S_r * V_r' * U_r * S_r * V_r') + B * U_r * S_r * V_r' + C)

A * (XX * XX) + B * XX + C


U, S, V = svd(A)

# Determine effective rank r: count singular values above tol.
cutoff = 1-eps()
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end
r = sum(S .> 1e-12)
# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V
U_r * V_r'
norm(U_r * S_r * V_r' - A)
norm(U_r * S_r * V_r' * (XX * XX) + B * XX + C)
norm(U_r * S_r * V_r' * (XX * XX) + B * XX + C)
norm(A * (XX * XX) + B * XX + C)


X = U_r * X̃ * V_r'
X = U_r * X̃ * V_r'
X - XX




A = SSVVDD.V[:,1:n_cutoff]' * jac_buffer[1:230,1:230] * SSVVDD.U[:,1:n_cutoff]
B = SSVVDD.V[:,1:n_cutoff]' * jac_buffer[1:230,231:460] * SSVVDD.V[:,1:n_cutoff]
C = SSVVDD.U[:,1:n_cutoff]' * jac_buffer[1:230,461:690] * SSVVDD.V[:,1:n_cutoff]


X̃, iter, reached_tol = solve_quadratic_matrix_equation(A,B,C, Val(:doubling), 𝓂.timings)
reached_tol
X

X = SSVVDD.V[:,1:n_cutoff] * X̃ * SSVVDD.U[:,1:n_cutoff]'

XX, iter, reached_tol = solve_quadratic_matrix_equation(jac_buffer[1:230,1:230],jac_buffer[1:230,231:460],jac_buffer[1:230,461:690], Val(:doubling), 𝓂.timings)
reached_tol

norm(X-XX)/max(norm(X),norm(XX)) 
