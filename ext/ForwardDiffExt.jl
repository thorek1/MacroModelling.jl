module ForwardDiffExt

# ForwardDiff Dual number specializations for forward-mode automatic differentiation
#
# Strategy for each function:
#   1. Extract Float64 values from Dual numbers using ℱ.value.(...)
#   2. Compute the function result on Float64 values
#   3. Compute partials using implicit differentiation or chain rule
#   4. Reconstruct Dual numbers by combining values and partials

import MacroModelling
import MacroModelling:
    # Types
    ℳ, constants, workspaces, caches, CalculationOptions,
    higher_order_workspace, sylvester_workspace, lyapunov_workspace,
    SolverTolerances, AdTolerances,
    # Constructors / factories
    Higher_order_workspace, merge_calculation_options,
    # Functions being specialized
    sparse_preallocated!, solve_stochastic_steady_state_newton,
    get_NSSS_and_parameters, calculate_first_order_solution,
    solve_quadratic_matrix_equation, solve_sylvester_equation,
    solve_lyapunov_equation, calculate_loglikelihood, primal,
    # Internal helpers
    initialise_constants!, ensure_computational_constants!,
    ensure_model_structure_constants!, ensure_first_order_constants!,
    ensure_first_order_workspace_buffers!, ensure_sylvester_krylov_buffers!,
    ensure_sylvester_doubling_buffers!, ensure_qme_doubling_workspace!,
    ensure_lyapunov_workspace!, evaluate_custom_steady_state_function,
    solve_nsss_wrapper, update_ss_counter!, factorize_lu!, solve_lu_left!,
    get_initial_covariance, find_shocks, normalize_presample_periods,
    compressed_kron²!, compressed_kron²_power!, compressed_kron³!,
    compressed_kron³_power!, compressed_kron², compressed_kron³,
    compressed_kron²_power, compressed_kron³_power,
    compressed_pair_hessian!, compressed_triple_hessian!,
    ensure_sss_kron_buffers!,
    # Constants
    DEFAULT_SOLVER_PARAMETERS, DEFAULT_QME_ALGORITHM

import ForwardDiff
const ℱ = ForwardDiff

import LinearAlgebra as ℒ
import LinearSolve as 𝒮
import SparseArrays: SparseMatrixCSC, SparseVector, AbstractSparseMatrix, sparse, sparsevec

# ── Extend primal() for ForwardDiff.Dual ──
MacroModelling.primal(x::ℱ.Dual) = ℱ.value(x)


# ── sparse_preallocated! ──

function MacroModelling.sparse_preallocated!(Ŝ::Matrix{ℱ.Dual{Z,S,N}}; ℂ::higher_order_workspace = Higher_order_workspace()) where {Z,S,N}
    sparse(Ŝ)
end


# ── solve_stochastic_steady_state_newton (2nd order) ──

function MacroModelling.solve_stochastic_steady_state_newton(::Val{:second_order}, 
                                              𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                              x::Vector{ℱ.Dual{Z,S,N}},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}

    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    x̂ = ℱ.value.(x)
    
    # Get cached computational constants
    constants = initialise_constants!(𝓂)
    so = constants.second_order
    cc = ensure_computational_constants!(constants)
    ℂ = 𝓂.workspaces.second_order
    T = constants.post_model_macro
    I_nPast = T.I_nPast

    nPast = length(x̂)
    n_state_aug = nPast + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    # Pre-sliced by the caller (see the primal method in
    # `steady_state/stochastic_steady_state.jl`).
    A = 𝐒₁̂
    B = 𝐒₂̂
    B̂ = B
 
    # Allocate or reuse workspace for partials and SSS kron buffers.
    # NOTE: when this overload is called from a higher-level ForwardDiff path,
    # `ℂ` may have been mutated to a `Dual`-typed workspace by the upstream
    # perturbation solver. Since the SSS Newton iter here is intentionally
    # carried out on the primal (`S`) values only, we allocate fresh `S`-typed
    # local buffers whenever the cached ones are not `S`-typed.
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=false)
    if size(ℂ.∂x_second_order) != (nPast, N) || eltype(ℂ.∂x_second_order) !== S
        ℂ.∂x_second_order = zeros(S, nPast, N)
    else
        fill!(ℂ.∂x_second_order, zero(S))
    end
    ∂x̄ = ℂ.∂x_second_order
    n_aug = nPast + 1
    n_aug2 = n_aug * (n_aug + 1) ÷ 2
    if eltype(ℂ.x_aug_buf) === S && length(ℂ.kron_x_aug_xx) == n_aug2 && size(ℂ.kron_x_aug_I) == (n_aug2, nPast)
        x_aug = ℂ.x_aug_buf
        kron_x_aug = ℂ.kron_x_aug_xx
        kron_x_aug_I = ℂ.kron_x_aug_I
    else
        x_aug = zeros(S, n_aug)
        kron_x_aug = zeros(S, n_aug2)
        kron_x_aug_I = zeros(S, n_aug2, nPast)
    end
    state_identity = @view cc.I_state_vol[:, 1:nPast]
    x_aug[end] = one(S)

    max_iters = 100
    for i in 1:max_iters
        copyto!(x_aug, 1, x̂, 1, nPast)
        compressed_kron²!(kron_x_aug_I, x_aug, state_identity)
        ∂x = (A + B * kron_x_aug_I - I_nPast)

        compressed_kron²_power!(kron_x_aug, x_aug)
        Δx = A * x̂ + B̂ * kron_x_aug / 2 - x̂
        ∂x_lu = ℒ.lu(∂x, check = false)
        ℒ.issuccess(∂x_lu) || break
        Δx = ∂x_lu \ Δx

        if i > 5 && isapprox(A * x̂ + B̂ * kron_x_aug / 2, x̂, rtol = tol)
            break
        end
        
        ℒ.axpy!(-1, Δx, x̂)
    end

    copyto!(x_aug, 1, x̂, 1, nPast)
    compressed_kron²_power!(kron_x_aug, x_aug)
    compressed_kron²!(kron_x_aug_I, x_aug, state_identity)
    solved = isapprox(A * x̂ + B̂ * kron_x_aug / 2, x̂, rtol = tol)

    if solved
        TMP = A + B * kron_x_aug_I - I_nPast
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)

            ∂A = ∂𝐒₁
            ∂B̂ = ∂𝐒₂

            tmp = ∂A * x̂ + ∂B̂ * kron_x_aug / 2

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end


# ── solve_stochastic_steady_state_newton (3rd order) ──

function MacroModelling.solve_stochastic_steady_state_newton(::Val{:third_order}, 
                                              𝐒₁::Matrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₂::AbstractSparseMatrix{ℱ.Dual{Z,S,N}}, 
                                              𝐒₃::AbstractSparseMatrix{ℱ.Dual{Z,S,N}},
                                              x::Vector{ℱ.Dual{Z,S,N}},
                                              𝓂::ℳ;
                                              tol::AbstractFloat = 1e-14)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    𝐒₁̂ = ℱ.value.(𝐒₁)
    𝐒₂̂ = ℱ.value.(𝐒₂)
    𝐒₃̂ = ℱ.value.(𝐒₃)
    x̂ = ℱ.value.(x)
    
    # Get cached computational constants
    so = ensure_computational_constants!(𝓂.constants)
    T = 𝓂.constants.post_model_macro
    ℂ = 𝓂.workspaces.third_order
    I_nPast = T.I_nPast

    nPast = length(x̂)
    n_state_aug = nPast + 1
    n_state_pair = n_state_aug * (n_state_aug + 1) ÷ 2
    n_state_triple = n_state_aug * (n_state_aug + 1) * (n_state_aug + 2) ÷ 6
    # Pre-sliced by the caller, as at second order.
    A = 𝐒₁̂
    B = 𝐒₂̂
    B̂ = B
    C = 𝐒₃̂
    Ĉ = C

    # Allocate or reuse workspace for partials and SSS kron buffers.
    # See note in the `:second_order` overload above — fall back to fresh
    # `S`-typed local buffers when the cached workspace got mutated to a
    # `Dual`-typed one upstream.
    ensure_sss_kron_buffers!(ℂ, nPast; third_order=true)
    if size(ℂ.∂x_third_order) != (nPast, N) || eltype(ℂ.∂x_third_order) !== S
        ℂ.∂x_third_order = zeros(S, nPast, N)
    else
        fill!(ℂ.∂x_third_order, zero(S))
    end
    ∂x̄ = ℂ.∂x_third_order
    n_aug = nPast + 1
    n_aug2 = n_aug * (n_aug + 1) ÷ 2
    n_aug3 = n_aug * (n_aug + 1) * (n_aug + 2) ÷ 6
    if eltype(ℂ.x_aug_buf) === S && length(ℂ.kron_x_aug_xx) == n_aug2 &&
       length(ℂ.kron_x_aug_x_kron) == n_aug3 &&
       size(ℂ.kron_x_aug_I) == (n_aug2, nPast) &&
       size(ℂ.kron_x_kron_I) == (n_aug3, nPast)
        x_aug = ℂ.x_aug_buf
        kron_x_aug = ℂ.kron_x_aug_xx
        kron_x_kron = ℂ.kron_x_aug_x_kron
        kron_x_aug_I = ℂ.kron_x_aug_I
        kron_x_kron_I = ℂ.kron_x_kron_I
    else
        x_aug = zeros(S, n_aug)
        kron_x_aug = zeros(S, n_aug2)
        kron_x_kron = zeros(S, n_aug3)
        kron_x_aug_I = zeros(S, n_aug2, nPast)
        kron_x_kron_I = zeros(S, n_aug3, nPast)
    end
    state_identity = @view so.I_state_vol[:, 1:nPast]
    x_aug[end] = one(S)

    max_iters = 100
    for i in 1:max_iters
        copyto!(x_aug, 1, x̂, 1, nPast)
        compressed_kron²_power!(kron_x_aug, x_aug)
        compressed_kron³_power!(kron_x_kron, x_aug)
        compressed_kron²!(kron_x_aug_I, x_aug, state_identity)
        compressed_kron³!(kron_x_kron_I, x_aug, x_aug, state_identity)
        ∂x = (A + B * kron_x_aug_I + C * kron_x_kron_I / 2 - I_nPast)

        Δx = A * x̂ + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6 - x̂
        ∂x_lu = ℒ.lu(∂x, check = false)
        ℒ.issuccess(∂x_lu) || break
        Δx = ∂x_lu \ Δx

        if i > 5 && isapprox(A * x̂ + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x̂, rtol = tol)
            break
        end
        
        ℒ.axpy!(-1, Δx, x̂)
    end

    copyto!(x_aug, 1, x̂, 1, nPast)
    compressed_kron²_power!(kron_x_aug, x_aug)
    compressed_kron³_power!(kron_x_kron, x_aug)
    compressed_kron²!(kron_x_aug_I, x_aug, @view(so.I_state_vol[:, 1:nPast]))
    compressed_kron³!(kron_x_kron_I, x_aug, x_aug, @view(so.I_state_vol[:, 1:nPast]))
    solved = isapprox(A * x̂ + B̂ * kron_x_aug / 2 + Ĉ * kron_x_kron / 6, x̂, rtol = tol)
    
    if solved
        TMP = A + B * kron_x_aug_I + C * kron_x_kron_I / 2 - I_nPast
        for i in 1:N
            ∂𝐒₁ = ℱ.partials.(𝐒₁, i)
            ∂𝐒₂ = ℱ.partials.(𝐒₂, i)
            ∂𝐒₃ = ℱ.partials.(𝐒₃, i)

            ∂A = ∂𝐒₁
            ∂B̂ = ∂𝐒₂
            ∂Ĉ = ∂𝐒₃

            tmp = ∂A * x̂ + ∂B̂ * kron_x_aug / 2 + ∂Ĉ * kron_x_kron / 6

            ∂x̄[:,i] = -TMP \ tmp
        end
    end
    
    return reshape(map(x̂, eachrow(∂x̄)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(x̂)), solved
end


# ── separate_values_and_partials_from_sparsevec_dual (internal helper) ──

function separate_values_and_partials_from_sparsevec_dual(V::SparseVector{ℱ.Dual{Z,S,N}}; tol::AbstractFloat = eps()) where {Z,S,N}
    nrows = length(V)
    ncols = length(V.nzval[1].partials)

    rows = Int[]
    cols = Int[]

    prtls = Float64[]

    for (i,v) in enumerate(V.nzind)
        for (k,w) in enumerate(V.nzval[i].partials)
            if abs(w) > tol
                push!(rows,v)
                push!(cols,k)
                push!(prtls,w)
            end
        end
    end

    vvals = sparsevec(V.nzind,[i.value for i in V.nzval],nrows)
    ps = sparse(rows,cols,prtls,nrows,ncols)

    return vvals, ps
end


# ── get_NSSS_and_parameters ──

function MacroModelling.get_NSSS_and_parameters(𝓂::ℳ, 
                                parameter_values_dual::Vector{ℱ.Dual{Z,S,N}}; 
                                opts::CalculationOptions = merge_calculation_options(),
                                cold_start::Bool = false,
                                estimation::Bool = false,
                                caching::Bool = true)::Tuple{Vector{ℱ.Dual{Z,S,N}}, Tuple{S, Int}} where {Z, S <: AbstractFloat, N}
    parameter_values = ℱ.value.(parameter_values_dual)
    ensure_model_structure_constants!(𝓂.constants, 𝓂.equations.calibration_parameters)
    ms = 𝓂.constants.post_complete_parameters
    T = 𝓂.constants.post_model_macro
    qme_ws = 𝓂.workspaces.first_order

    if 𝓂.functions.NSSS_custom isa Function
        vars_in_ss_equations = ms.vars_in_ss_equations
        expected_length = length(vars_in_ss_equations) + length(𝓂.equations.calibration_parameters)

        SS_and_pars_tmp = evaluate_custom_steady_state_function(
            𝓂,
            parameter_values,
            expected_length,
            length(𝓂.constants.post_complete_parameters.parameters),
        )

        residual = 𝓂.workspaces.nsss_solver.check_residual
        fill!(residual, 0.0)
        
        𝓂.functions.NSSS_check(residual, parameter_values, SS_and_pars_tmp)
        
        solution_error = ℒ.norm(residual)

        iters = 0

        X = ms.custom_ss_expand_matrix
        SS_and_pars = X * SS_and_pars_tmp
    else
        fastest_idx = 𝓂.constants.post_complete_parameters.nsss_fastest_solver_parameter_idx
        preferred_solver_parameter_idx = fastest_idx < 1 || fastest_idx > length(DEFAULT_SOLVER_PARAMETERS) ? 1 : fastest_idx
        SS_and_pars, (solution_error, iters) = solve_nsss_wrapper(parameter_values, 𝓂, opts.tol, opts.verbose, cold_start, DEFAULT_SOLVER_PARAMETERS, preferred_solver_parameter_idx = preferred_solver_parameter_idx)
    end
    
    # Allocate or reuse workspace for partials
    if size(qme_ws.∂SS_and_pars) != (length(SS_and_pars), N)
        qme_ws.∂SS_and_pars = zeros(S, length(SS_and_pars), N)
    else
        fill!(qme_ws.∂SS_and_pars, zero(S))
    end
    ∂SS_and_pars = qme_ws.∂SS_and_pars

    if solution_error > opts.tol.nsss.acceptance_tol || isnan(solution_error)
        if opts.verbose println("Failed to find NSSS") end

        # Update failed counter
        update_ss_counter!(𝓂.counters, false, estimation = estimation)

        solution_error = S(10.0)
    else
        # Update success counter
        update_ss_counter!(𝓂.counters, true, estimation = estimation)

        custom_ss_expand_matrix = ms.custom_ss_expand_matrix
        

        ∂ = parameter_values
        C = SS_and_pars[ms.SS_and_pars_no_exo_idx]

        if eltype(𝓂.caches.NSSS_∂equations_∂parameters) != eltype(parameter_values)
            if 𝓂.caches.NSSS_∂equations_∂parameters isa SparseMatrixCSC
                jac_cache = similar(𝓂.caches.NSSS_∂equations_∂parameters, eltype(parameter_values))
                jac_cache.nzval .= 0
            else
                jac_cache = zeros(eltype(parameter_values), size(𝓂.caches.NSSS_∂equations_∂parameters))
            end
        else
            jac_cache = 𝓂.caches.NSSS_∂equations_∂parameters
            if jac_cache isa SparseMatrixCSC
                jac_cache.nzval .= 0
            else
                fill!(jac_cache, zero(eltype(jac_cache)))
            end
        end

        𝓂.functions.NSSS_∂equations_∂parameters(jac_cache, ∂, C)

        ∂SS_equations_∂parameters = jac_cache

        
        if eltype(𝓂.caches.NSSS_∂equations_∂SS_and_pars) != eltype(parameter_values)
            if 𝓂.caches.NSSS_∂equations_∂SS_and_pars isa SparseMatrixCSC
                jac_cache = similar(𝓂.caches.NSSS_∂equations_∂SS_and_pars, eltype(SS_and_pars))
                jac_cache.nzval .= 0
            else
                jac_cache = zeros(eltype(SS_and_pars), size(𝓂.caches.NSSS_∂equations_∂SS_and_pars))
            end
        else
            jac_cache = 𝓂.caches.NSSS_∂equations_∂SS_and_pars
            if jac_cache isa SparseMatrixCSC
                jac_cache.nzval .= 0
            else
                fill!(jac_cache, zero(eltype(jac_cache)))
            end
        end

        𝓂.functions.NSSS_∂equations_∂SS_and_pars(jac_cache, ∂, C)

        ∂SS_equations_∂SS_and_pars = jac_cache

        if ∂SS_equations_∂SS_and_pars isa SparseMatrixCSC
            rhs_n_rows = size(∂SS_equations_∂SS_and_pars, 1)
            rhs_n_cols = size(∂SS_equations_∂parameters, 2)

            if length(qme_ws.nsss_sparse_rhs) != rhs_n_rows
                qme_ws.nsss_sparse_rhs = zeros(eltype(SS_and_pars), rhs_n_rows)
            end

            if size(qme_ws.nsss_jvp_rhs) != (rhs_n_rows, rhs_n_cols)
                qme_ws.nsss_jvp_rhs = zeros(eltype(SS_and_pars), rhs_n_rows, rhs_n_cols)
            end

            if size(qme_ws.nsss_sparse_lu_buffer.A) != (rhs_n_rows, rhs_n_rows)
                sparse_prob = 𝒮.LinearProblem(∂SS_equations_∂SS_and_pars, qme_ws.nsss_sparse_rhs)
                qme_ws.nsss_sparse_lu_buffer = 𝒮.init(sparse_prob,
                                                      𝒮.LUFactorization(),
                                                      verbose = isdefined(𝒮, :LinearVerbosity) ? 𝒮.LinearVerbosity(𝒮.SciMLLogging.Minimal()) : false)
            else
                qme_ws.nsss_sparse_lu_buffer.A = ∂SS_equations_∂SS_and_pars
            end

            sparse_solved = true
            for j in 1:rhs_n_cols
                @views copyto!(qme_ws.nsss_sparse_rhs, ∂SS_equations_∂parameters[:, j])
                qme_ws.nsss_sparse_lu_buffer.b = qme_ws.nsss_sparse_rhs
                sparse_sol = 𝒮.solve!(qme_ws.nsss_sparse_lu_buffer)

                if sparse_sol.retcode != 𝒮.SciMLBase.ReturnCode.Default && !𝒮.SciMLBase.successful_retcode(sparse_sol.retcode)
                    sparse_solved = false
                    break
                end

                @views copyto!(qme_ws.nsss_jvp_rhs[:, j], qme_ws.nsss_sparse_lu_buffer.u)
            end

            if !sparse_solved
                if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
                solution_error = S(10.0)
            else
                ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)
                jvp_no_exo = custom_ss_expand_matrix * qme_ws.nsss_jvp_rhs
                for i in 1:N
                    parameter_values_partials = ℱ.partials.(parameter_values_dual, i)
                    @view(∂SS_and_pars[:,i]) .= jvp_no_exo * parameter_values_partials
                end
            end
        else
            qme_ws.fast_lu_ws_nsss, qme_ws.fast_lu_dims_nsss, solved_nsss, nsss_lu = factorize_lu!(Val(:FastLapack), ∂SS_equations_∂SS_and_pars,
                                                                                                     qme_ws.fast_lu_ws_nsss,
                                                                                                     qme_ws.fast_lu_dims_nsss)

            if !solved_nsss
                if opts.verbose println("Failed to calculate implicit derivative of NSSS") end
                solution_error = S(10.0)
            else
                rhs_dense = ∂SS_equations_∂parameters isa Matrix ? ∂SS_equations_∂parameters : Matrix(∂SS_equations_∂parameters)

                if size(qme_ws.nsss_jvp_rhs) != size(rhs_dense)
                    qme_ws.nsss_jvp_rhs = zeros(eltype(rhs_dense), size(rhs_dense))
                end
                copyto!(qme_ws.nsss_jvp_rhs, rhs_dense)

                solve_lu_left!(∂SS_equations_∂SS_and_pars,
                               qme_ws.nsss_jvp_rhs,
                               qme_ws.fast_lu_ws_nsss,
                               nsss_lu)

                ℒ.rmul!(qme_ws.nsss_jvp_rhs, -1)
                jvp_no_exo = custom_ss_expand_matrix * qme_ws.nsss_jvp_rhs
                for i in 1:N
                    parameter_values_partials = ℱ.partials.(parameter_values_dual, i)
                    @view(∂SS_and_pars[:,i]) .= jvp_no_exo * parameter_values_partials
                end
            end
        end
    end
    
    # Cache write: store NSSS result and stamp (using Float64 values)
    if caching
        cache_ss = 𝓂.caches.non_stochastic_steady_state
        if length(cache_ss) != length(SS_and_pars)
            resize!(cache_ss, length(SS_and_pars))
        end
        copyto!(cache_ss, SS_and_pars)
        solved = !(solution_error > opts.tol.nsss.acceptance_tol)
        if solved
            𝓂.caches.valid_for.non_stochastic_steady_state = Float64.(parameter_values)
        else
            𝓂.caches.valid_for.non_stochastic_steady_state = Float64[]
        end
    end

    return reshape(map(SS_and_pars, eachrow(∂SS_and_pars)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(SS_and_pars)), (solution_error, iters)
end


# ── calculate_first_order_solution ──

function MacroModelling.calculate_first_order_solution(∇₁::Matrix{ℱ.Dual{Z,S,N}},
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        opts::CalculationOptions = merge_calculation_options(),
                                        use_fastlapack_lu::Bool = true,
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                        parameter_values::AbstractVector{<:Real} = Float64[],
                                        caching::Bool = true)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Matrix{Float64}, Bool} where {Z,S,N}
    T = constants.post_model_macro
    ensure_first_order_constants!(constants)
    idx_constants = constants.post_complete_parameters
    qme_ws = workspaces.first_order
    sylv_ws = workspaces.sylvester_1st_order
    ensure_first_order_workspace_buffers!(qme_ws, T, length(idx_constants.dyn_index), length(idx_constants.comb))
    ensure_sylvester_krylov_buffers!(qme_ws.sylvester, T.nVars, T.nVars)
    ensure_sylvester_doubling_buffers!(qme_ws.sylvester, T.nVars, T.nVars)

    if size(qme_ws.p_tmp) != size(∇₁)
        qme_ws.p_tmp = zeros(S, size(∇₁, 1), size(∇₁, 2))
    end
    ∇̂₁ = qme_ws.p_tmp
    @inbounds for j in axes(∇₁, 2), i in axes(∇₁, 1)
        ∇̂₁[i, j] = ℱ.value(∇₁[i, j])
    end

    expand_future = idx_constants.expand_future
    expand_past = idx_constants.expand_past

    A = qme_ws.𝐀₀
    B = qme_ws.∇₀
    X = qme_ws.sylvester.tmp
    AXB = qme_ws.sylvester.𝐗
    AA = qme_ws.sylvester.𝐂
    X² = qme_ws.sylvester.𝐀
    dA = qme_ws.sylvester.𝐀¹
    dB = qme_ws.sylvester.𝐁
    dC = qme_ws.sylvester.𝐁¹
    CC = qme_ws.sylvester.𝐂_dbl
    tmp = qme_ws.sylvester.𝐂¹
    B_sylv = qme_ws.sylvester.𝐂B

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(∇̂₁), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(∇̂₁)} ? initial_guess : Matrix{eltype(∇̂₁)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    𝐒₁, qme_sol, solved = calculate_first_order_solution(∇̂₁, constants, workspaces, cache; opts = opts, initial_guess = initial_guess_value, caching = caching)

    if !solved 
        return ∇₁, qme_sol, false
    end

    ℒ.mul!(A, @view(∇̂₁[:,1:T.nFuture_not_past_and_mixed]), expand_future)
    copyto!(B, @view(∇̂₁[:,idx_constants.nabla_zero_cols]))

    ℒ.mul!(X, @view(𝐒₁[:,1:end-T.nExo]), expand_past)

    copyto!(AXB, B)
    ℒ.mul!(AXB, A, X, 1, 1)

    qme_ws.fast_lu_ws_nabla0, qme_ws.fast_lu_dims_nabla0, solved_AXB, AXBfact = factorize_lu!((use_fastlapack_lu ? Val(:FastLapack) : Val(:Julia)), AXB,
                                                                                                 qme_ws.fast_lu_ws_nabla0,
                                                                                                 qme_ws.fast_lu_dims_nabla0)

    if !solved_AXB
        return ∇₁, qme_sol, false
    end

    copyto!(AA, A)
    solve_lu_left!(AXB, AA, qme_ws.fast_lu_ws_nabla0, AXBfact;
                   use_fastlapack_lu = use_fastlapack_lu)

    ℒ.mul!(X², X, X)

    # Allocate or reuse workspace for partials
    if size(qme_ws.X̃_first_order) != (length(𝐒₁[:,1:end-T.nExo]), N)
        qme_ws.X̃_first_order = zeros(length(𝐒₁[:,1:end-T.nExo]), N)
    else
        fill!(qme_ws.X̃_first_order, zero(eltype(qme_ws.X̃_first_order)))
    end
    X̃ = qme_ws.X̃_first_order

    p = ∇̂₁

    copyto!(B_sylv, X)
    ℒ.rmul!(B_sylv, -1)

    initial_guess = zeros(eltype(X), size(X, 1), size(X, 2))

    prev_capture = sylv_ws.pow_capture
    sylv_ws.pow_iters = 0
    sylv_ws.pow_capture = true
    sylv_ws.pow_transposed = false
    sylv_cache_captured = false

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        p .= ℱ.partials.(∇₁, i)

        ℒ.mul!(dA, @view(p[:,1:T.nFuture_not_past_and_mixed]), expand_future)
        copyto!(dB, @view(p[:,idx_constants.nabla_zero_cols]))
        ℒ.mul!(dC, @view(p[:,idx_constants.nabla_minus_cols]), expand_past)

        copyto!(CC, dC)
        ℒ.mul!(tmp, dA, X²)
        CC .+= tmp
        ℒ.mul!(tmp, dB, X)
        CC .+= tmp

        solve_lu_left!(AXB, CC, qme_ws.fast_lu_ws_nabla0, AXBfact;
                       use_fastlapack_lu = use_fastlapack_lu)

        if ℒ.norm(CC) < eps() continue end

        ℒ.rmul!(CC, -1)

        dX, slvd = solve_sylvester_equation(AA, B_sylv, CC, sylv_ws,
                                                initial_guess = initial_guess,
                                                sylvester_algorithm = opts.sylvester_algorithm²,
                                                preconditioner = opts.sylvester_preconditioner,
                                                tol = opts.tol.first_order.ad.sylvester,
                                                verbose = opts.verbose)
    
        if !sylv_cache_captured
            sylv_ws.pow_capture = false  # captured A^(2^k) on first solve; reuse for subsequent
            sylv_cache_captured = true
        end

        if !slvd
            fill!(view(X̃, :, i), NaN)
            solved = false
            continue
        end

        # initial_guess = dX

        @views copyto!(X̃[:,i],dX[:,T.past_not_future_and_mixed_idx])
    end
    sylv_ws.pow_capture = prev_capture
    sylv_ws.pow_iters = 0

    x = reshape(map(𝐒₁[:,1:end-T.nExo], eachrow(X̃)) do v, p
            ℱ.Dual{Z}(v, p...) # Z is the tag
        end, size(𝐒₁[:,1:end-T.nExo]))

    Jm = expand_past
    
    ∇₊ = ∇₁[:,1:T.nFuture_not_past_and_mixed] * expand_future
    ∇₀ = ∇₁[:,idx_constants.nabla_zero_cols]
    ∇ₑ = ∇₁[:,idx_constants.nabla_e_start:end]

    B = -((∇₊ * x * Jm + ∇₀) \ ∇ₑ)

    S₁ = hcat(x, B)

    S₁_value = ℱ.value.(S₁)
    S₁_existing = cache.first_order_solution_matrix
    if S₁_existing isa Matrix{S} && size(S₁_existing) == size(S₁_value)
        copyto!(S₁_existing, S₁_value)
    else
        cache.first_order_solution_matrix = S₁_value
    end

    if !isempty(parameter_values)
        cache.valid_for.first_order_solution = Float64.(MacroModelling.primal.(parameter_values))
    end

    return S₁, qme_sol, solved
end


# ── solve_quadratic_matrix_equation ──

function MacroModelling.solve_quadratic_matrix_equation(A::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        B::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        C::AbstractMatrix{ℱ.Dual{Z,S,N}}, 
                                        constants::constants,
                                        workspaces::workspaces,
                                        cache::caches;
                                        initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                        tol::AdTolerances = AdTolerances(), 
                                        quadratic_matrix_equation_algorithm::Symbol = DEFAULT_QME_ALGORITHM,
                                        verbose::Bool = false,
                                        caching::Bool = true) where {Z,S,N}
    T = constants.post_model_macro
    # unpack: AoS -> SoA
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(Â), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(Â)} ? initial_guess : Matrix{eltype(Â)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    qme_ws = ensure_qme_doubling_workspace!(workspaces,
                                            T.nVars - T.nPresent_only)

    X, solved = solve_quadratic_matrix_equation(Â, B̂, Ĉ,
                                                constants,
                                                workspaces,
                                                cache;
                                                tol = tol.qme,
                                                initial_guess = initial_guess_value,
                                                quadratic_matrix_equation_algorithm = quadratic_matrix_equation_algorithm,
                                                verbose = verbose,
                                                caching = caching)

    AXB = Â * X + B̂
    
    AXBfact = ℒ.lu(AXB, check = false)

    if !ℒ.issuccess(AXBfact)
        AXBfact = ℒ.svd(AXB)
    end

    invAXB = inv(AXBfact)

    AA = invAXB * Â

    X² = X * X

    # Allocate or reuse workspace for partials
    if size(qme_ws.X̃) != (length(X), N)
        qme_ws.X̃ = zeros(length(X), N)
    else
        fill!(qme_ws.X̃, zero(eltype(qme_ws.X̃)))
    end
    X̃ = qme_ws.X̃

    sws = qme_ws.sylvester
    prev_capture = sws.pow_capture
    sws.pow_iters = 0
    sws.pow_capture = true
    sws.pow_transposed = false
    qme_sylv_cache_captured = false

    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        dA = ℱ.partials.(A, i)
        dB = ℱ.partials.(B, i)
        dC = ℱ.partials.(C, i)
    
        CC = invAXB * (dA * X² + dB * X + dC)

        if ℒ.norm(CC) < eps() continue end
    
        dX, slvd = solve_sylvester_equation(AA, -X, -CC, qme_ws.sylvester,
                            sylvester_algorithm = :doubling,
                            tol = tol.sylvester)

        if !qme_sylv_cache_captured
            sws.pow_capture = false  # captured A^(2^k) on first solve; reuse for subsequent
            qme_sylv_cache_captured = true
        end

        solved = Bool(solved) && Bool(slvd)

        if !slvd
            fill!(view(X̃, :, i), NaN)
            continue
        end

        X̃[:,i] = vec(dX)
    end
    sws.pow_capture = prev_capture
    sws.pow_iters = 0
    
    return reshape(map(X, eachrow(X̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(X)), solved
end


# ── solve_sylvester_equation ──

function MacroModelling.solve_sylvester_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    B::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    𝕊ℂ::sylvester_workspace;
                                    initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                    sylvester_algorithm::Symbol = :doubling,
                                    preconditioner::Symbol = :ilu,
                                    tol::SolverTolerances = SolverTolerances(),
                                    verbose::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # Extract Float64 values from Dual numbers
    Â = ℱ.value.(A)
    B̂ = ℱ.value.(B)
    Ĉ = ℱ.value.(C)

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(Â), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(Â)} ? initial_guess : Matrix{eltype(Â)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    # Capture A^(2^k), B^(2^k) sequence from primal so the partial-loop solves can replay them.
    prev_capture = 𝕊ℂ.pow_capture
    𝕊ℂ.pow_iters = 0
    𝕊ℂ.pow_capture = true
    𝕊ℂ.pow_transposed = false

    P̂, solved = solve_sylvester_equation(Â, B̂, Ĉ, 𝕊ℂ,
                                        sylvester_algorithm = sylvester_algorithm, 
                                        preconditioner = preconditioner,
                                        tol = tol, 
                                        verbose = verbose, 
                                        initial_guess = initial_guess_value)

    𝕊ℂ.pow_capture = false

    if size(𝕊ℂ.P) != size(P̂)
        𝕊ℂ.P = zeros(eltype(P̂), size(P̂)...)
    end
    copyto!(𝕊ℂ.P, P̂)
    P̂_stable = 𝕊ℂ.P

    # Allocate or reuse workspaces for temporary copies
    if size(𝕊ℂ.Ã_fd) != size(Â)
        𝕊ℂ.Ã_fd = copy(Â)
    else
        copyto!(𝕊ℂ.Ã_fd, Â)
    end
    Ã = 𝕊ℂ.Ã_fd
    
    if size(𝕊ℂ.B̃_fd) != size(B̂)
        𝕊ℂ.B̃_fd = copy(B̂)
    else
        copyto!(𝕊ℂ.B̃_fd, B̂)
    end
    B̃ = 𝕊ℂ.B̃_fd
    
    if size(𝕊ℂ.C̃_fd) != size(Ĉ)
        𝕊ℂ.C̃_fd = copy(Ĉ)
    else
        copyto!(𝕊ℂ.C̃_fd, Ĉ)
    end
    C̃ = 𝕊ℂ.C̃_fd
    
    # Allocate or reuse workspace for partials
    if size(𝕊ℂ.P̃) != (length(P̂), N)
        𝕊ℂ.P̃ = zeros(S, length(P̂), N)
    else
        fill!(𝕊ℂ.P̃, zero(S))
    end
    P̃ = 𝕊ℂ.P̃
    
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        B̃ .= ℱ.partials.(B, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂_stable * B̂ + Â * P̂_stable * B̃ + C̃
        
        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_sylvester_equation(Â, B̂, X, 𝕊ℂ,
                                            sylvester_algorithm = sylvester_algorithm, 
                                            preconditioner = preconditioner,
                                            tol = tol, 
                                            verbose = verbose)

        solved = solved && slvd

        if !slvd
            fill!(view(P̃, :, i), NaN)
            continue
        end

        P̃[:,i] = vec(P)
    end
    𝕊ℂ.pow_capture = prev_capture
    𝕊ℂ.pow_iters = 0
    
    return reshape(map(P̂_stable, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂_stable)), solved
end


# ── solve_lyapunov_equation ──

function MacroModelling.solve_lyapunov_equation(  A::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    C::AbstractMatrix{ℱ.Dual{Z,S,N}},
                                    workspace::lyapunov_workspace;
                                    initial_guess::AbstractMatrix{<:Real} = zeros(0,0),
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::SolverTolerances = SolverTolerances(atol = 1e-14,
                                                                                rtol = 1e-14,
                                                                              initial_guess_acceptance_tol = 1e-12,
                                                                              acceptance_tol = 1e-12),
                                                                        verbose::Bool = false,
                                                                        has_unit_roots::Bool = false)::Tuple{Matrix{ℱ.Dual{Z,S,N}}, Bool} where {Z,S,N}
    # Extract Float64 values from Dual numbers
    Â = ℱ.value.(A)
    Ĉ = ℱ.value.(C)

    initial_guess_value = if length(initial_guess) == 0
        zeros(eltype(Â), 0, 0)
    elseif eltype(initial_guess) <: AbstractFloat
        initial_guess isa Matrix{eltype(Â)} ? initial_guess : Matrix{eltype(Â)}(initial_guess)
    else
        ℱ.value.(initial_guess)
    end

    # Capture A^(2^k) sequence from primal so the partial-loop solves can replay them.
    prev_capture = workspace.pow_capture
    workspace.pow_iters = 0
    workspace.pow_capture = true
    workspace.pow_transposed = false

    P̂, solved = solve_lyapunov_equation(Â, Ĉ, workspace;
                                        lyapunov_algorithm = lyapunov_algorithm,
                                        initial_guess = initial_guess_value,
                                        tol = tol,
                                        verbose = verbose,
                                        has_unit_roots = has_unit_roots)

    workspace.pow_capture = false

    if size(workspace.P) != size(P̂)
        workspace.P = zeros(eltype(P̂), size(P̂)...)
    end
    copyto!(workspace.P, P̂)
    P̂_stable = workspace.P

    # Allocate or reuse workspaces for temporary copies
    if size(workspace.Ã_fd) != size(Â)
        workspace.Ã_fd = copy(Â)
    else
        copyto!(workspace.Ã_fd, Â)
    end
    Ã = workspace.Ã_fd
    
    if size(workspace.C̃_fd) != size(Ĉ)
        workspace.C̃_fd = copy(Ĉ)
    else
        copyto!(workspace.C̃_fd, Ĉ)
    end
    C̃ = workspace.C̃_fd
    
    # Allocate or reuse workspace for partials
    if size(workspace.P̃) != (length(P̂), N)
        workspace.P̃ = zeros(length(P̂), N)
    else
        fill!(workspace.P̃, zero(eltype(workspace.P̃)))
    end
    P̃ = workspace.P̃
    
    # https://arxiv.org/abs/2011.11430  
    for i in 1:N
        Ã .= ℱ.partials.(A, i)
        C̃ .= ℱ.partials.(C, i)

        X = Ã * P̂_stable * Â' + Â * P̂_stable * Ã' + C̃

        if ℒ.norm(X) < eps() continue end

        P, slvd = solve_lyapunov_equation(Â, X, workspace;
                lyapunov_algorithm = lyapunov_algorithm,
                tol = tol,
                verbose = verbose,
                has_unit_roots = has_unit_roots)
        
        solved = solved && slvd

        if !slvd
            fill!(view(P̃, :, i), NaN)
            continue
        end

        P̃[:,i] = vec(P)
    end
    workspace.pow_capture = prev_capture
    workspace.pow_iters = 0
    
    return reshape(map(P̂_stable, eachrow(P̃)) do v, p
        ℱ.Dual{Z}(v, p...) # Z is the tag
    end, size(P̂_stable)), solved
end


# ── calculate_loglikelihood (Kalman filter for Dual numbers) ──

function MacroModelling.calculate_loglikelihood(::Val{:kalman},
                                ::Val,
                                observables_index::Vector{Int},
                                𝐒::Union{Matrix{ℱ.Dual{Z,S,N}},Vector{AbstractMatrix{ℱ.Dual{Z,S,N}}}},
                                data_in_deviations::Matrix{R},
                                constants::constants,
                                state,
                                workspaces::workspaces;
                                warmup_iterations::Int = 0,
                                presample_periods::Int = 0,
                                initial_covariance::Symbol = :theoretical,
                                filter_algorithm::Symbol = :LagrangeNewton,
                                lyapunov_algorithm::Symbol = :doubling,
                                on_failure_loglikelihood::U = -Inf,
                                measurement_error::Union{Nothing,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
                                opts::CalculationOptions = merge_calculation_options())::ℱ.Dual{Z,S,N} where {Z,S,N,R <: Real, U <: AbstractFloat}
    presample_periods = normalize_presample_periods(presample_periods, size(data_in_deviations, 2))
                                                
    T = constants.post_model_macro
    idx_constants = constants.post_complete_parameters
    lyap_ws = ensure_lyapunov_workspace!(workspaces, T.nVars, :first_order)
    kalman_ws = workspaces.kalman

    observables_and_states = sort(union(T.past_not_future_and_mixed_idx, observables_index))
    observables_sorted = sort(observables_index)
    I_nVars = idx_constants.diag_nVars

    A = @views 𝐒[observables_and_states,1:T.nPast_not_future_and_mixed] * I_nVars[T.past_not_future_and_mixed_idx, observables_and_states]
    B = @views 𝐒[observables_and_states,T.nPast_not_future_and_mixed+1:end]

    C = @views I_nVars[observables_sorted, observables_and_states]
    𝐁 = B * B'

    P = get_initial_covariance(Val(initial_covariance), A, 𝐁, lyap_ws, opts = opts)

    if !(eltype(P) <: ℱ.Dual)
        dual_zero = zero(A[1])
        P_float = P
        P = similar(A, size(P_float, 1), size(P_float, 2))
        @inbounds for i in eachindex(P)
            P[i] = dual_zero + S(P_float[i])
        end
    end

    # Initial mean: honour `state[1]` (set by the get_loglikelihood
    # `initial_state` override). Pre-edit this was unconditionally zero, which
    # silently dropped ForwardDiff Duals supplied via `initial_state`.
    DT = eltype(A)
    u_raw = state[1][observables_and_states]
    u = u_raw isa Vector{DT} ? copy(u_raw) : convert(Vector{DT}, u_raw)
    z = C * u
    loglik = zero(eltype(A))

    # Pre-allocate Dual-typed loop buffers
    ns = size(A, 1)
    no = size(C, 1)
    v = zeros(DT, no)
    CP = zeros(DT, no, ns)
    F_buf = zeros(DT, no, no)
    PCt = zeros(DT, ns, no)
    K = zeros(DT, ns, no)
    KC = zeros(DT, ns, ns)
    PmKCP = zeros(DT, ns, ns)
    AP = zeros(DT, ns, ns)
    Kv = zeros(DT, ns)
    uKv = zeros(DT, ns)
    w = zeros(DT, no)

    for t in 1:size(data_in_deviations, 2)
        if !all(isfinite.(z))
            if opts.verbose println("KF not finite at step $t") end
            return on_failure_loglikelihood
        end

        @views v .= data_in_deviations[:, t] .- z
        ℒ.mul!(CP, C, P)
        ℒ.mul!(F_buf, CP, C')

        # Add the measurement-error covariance H: F = C P C' + H (a vector of
        # per-observable variances, or a full covariance matrix).
        if measurement_error !== nothing
            if measurement_error isa AbstractMatrix
                @inbounds for j in 1:no, i in 1:no
                    F_buf[i, j] += measurement_error[i, j]
                end
            else
                @inbounds for i in 1:no
                    F_buf[i, i] += measurement_error[i]
                end
            end
        end

        luF = ℒ.lu(F_buf, check = false)
        if !ℒ.issuccess(luF)
            if opts.verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        Fdet = ℒ.det(luF)
        if Fdet < eps(Float64)
            if opts.verbose println("KF factorisation failed step $t") end
            return on_failure_loglikelihood
        end

        if t > presample_periods
            ℒ.ldiv!(w, luF, v)
            loglik += log(Fdet) + ℒ.dot(v, w)
        end

        invF = inv(luF)
        ℒ.mul!(PCt, P, C')
        ℒ.mul!(K, PCt, invF)

        # P = A * (P - K * C * P) * A' + 𝐁
        ℒ.mul!(KC, K, C)
        ℒ.mul!(PmKCP, KC, P)
        ℒ.axpby!(1, P, -1, PmKCP)
        ℒ.mul!(AP, A, PmKCP)
        ℒ.mul!(P, AP, A')
        ℒ.axpy!(1, 𝐁, P)

        # u = A * (u + K * v)
        ℒ.mul!(Kv, K, v)
        copyto!(uKv, u)
        ℒ.axpy!(1, Kv, uKv)
        ℒ.mul!(u, A, uKv)
        ℒ.mul!(z, C, u)
    end

    return -(loglik + ((size(data_in_deviations, 2) - presample_periods) * size(data_in_deviations, 1)) * log(2π)) / 2
end


# ── find_shocks (LagrangeNewton, 2nd order) for Dual numbers ──
# Iterative solvers diverge with Dual numbers due to generic LU vs LAPACK
# numerical differences. Solve with Float64 primals, then compute partials
# via the implicit function theorem.

function MacroModelling.find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{ℱ.Dual{Z,V,N}},
                    kron_buffer::Vector{ℱ.Dual{Z,V,N}},
                    kron_buffer2::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    𝐒ⁱ²ᵉ::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    shock_independent::Vector{ℱ.Dual{Z,V,N}};
                    kwargs...) where {Z,V,N}

    # Extract Float64 primals
    ig_f   = ℱ.value.(initial_guess)
    kb_f   = ℱ.value.(kron_buffer)
    kb2_f  = ℱ.value.(kron_buffer2)
    Si_f   = ℱ.value.(𝐒ⁱ)
    Si2e_f = ℱ.value.(𝐒ⁱ²ᵉ)
    si_f   = ℱ.value.(shock_independent)

    # Solve the primal LagrangeNewton on Float64.
    x_f, matched = find_shocks(Val(:LagrangeNewton),
        ig_f, kb_f, kb2_f, J, Si_f, Si2e_f, si_f; kwargs...)

    if !matched
        return ℱ.Dual{Z,V,N}.(x_f), false
    end

    # Propagate partials through the linearised KKT system at the optimum.
    # Implicit differentiation through the linearised KKT block.
    # Build  fXλp = [A tmp'; -tmp 0]  once, factor it, and solve for each
    # parameter direction. RHS is differentiation of the KKT residual:
    #   g_x = tmp'·λ - 2x   →  d g_x = (d_Si + 2·d_Si2e·kron(I,x))' · λ
    #   g_λ = si - Si·x - Si2e·kron(x,x)
    # 𝐒ⁱ²ᵉ lives in the compressed shock-pair basis, so the shock kron terms
    # must be compressed too (mirrors the primal find_shocks).
    n_x = length(x_f)
    n_obs = size(Si_f, 1)
    kIx = compressed_kron²(x_f, J)
    tmp = Si_f + 2 * Si2e_f * kIx
    λ = tmp' \ (2 .* x_f)
    A_mat = zeros(V, n_x, n_x)
    compressed_pair_hessian!(A_mat, 2 .* (Si2e_f' * λ))
    # The KKT block's -2I: a loop over the diagonal, not `2 .* Matrix(I(n_x))`,
    # which materialised an n_x x n_x dense identity and a second temporary for
    # no reason. Nothing about `V` required it — this works for `Dual` too.
    @inbounds for i in 1:n_x
        A_mat[i, i] -= 2
    end
    kxx = compressed_kron²_power(x_f)

    fXλp = [A_mat   tmp';
            -tmp    zeros(V, n_obs, n_obs)]
    fXλp_lu = ℒ.lu(fXλp, check = false)
    if !ℒ.issuccess(fXλp_lu)
        return ℱ.Dual{Z,V,N}.(x_f), false
    end

    partials_matrix = zeros(V, n_x, N)

    for k in 1:N
        d_si   = V[ℱ.partials(shock_independent[i])[k] for i in eachindex(shock_independent)]
        d_Si   = V[ℱ.partials(𝐒ⁱ[i])[k]                for i in eachindex(𝐒ⁱ)]
        d_Si2e = V[ℱ.partials(𝐒ⁱ²ᵉ[i])[k]              for i in eachindex(𝐒ⁱ²ᵉ)]

        d_Si_mat   = reshape(d_Si,   size(Si_f))
        d_Si2e_mat = reshape(d_Si2e, size(Si2e_f))

        dtmp = d_Si_mat + 2 * d_Si2e_mat * kIx
        d_g_x = dtmp' * λ
        d_g_λ = d_si - d_Si_mat * x_f - d_Si2e_mat * kxx

        sol = fXλp_lu \ vcat(-d_g_x, -d_g_λ)
        partials_matrix[:, k] = sol[1:n_x]
    end

    x_dual = Vector{ℱ.Dual{Z,V,N}}(undef, n_x)
    for i in 1:n_x
        x_dual[i] = ℱ.Dual{Z,V,N}(x_f[i],
            ℱ.Partials{N,V}(NTuple{N,V}(partials_matrix[i, k] for k in 1:N)))
    end

    return x_dual, matched
end


# ── find_shocks (LagrangeNewton, 3rd order) for Dual numbers ──
# Same implicit-differentiation strategy as the 2nd-order variant.
# Residual: g(x) = si - Si*x - Si2e*kron(x,x) - Si3e*kron(x,kron(x,x)) = 0
# Jacobian: Si + 2*Si2e*kron(I,x) + 3*Si3e*kron(I,kron(x,x))

function MacroModelling.find_shocks(::Val{:LagrangeNewton},
                    initial_guess::Vector{ℱ.Dual{Z,V,N}},
                    kron_buffer::Vector{ℱ.Dual{Z,V,N}},
                    kron_buffer²::Vector{ℱ.Dual{Z,V,N}},
                    kron_buffer2::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    kron_buffer3::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    kron_buffer4::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    J::ℒ.Diagonal{Bool, Vector{Bool}},
                    𝐒ⁱ::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    𝐒ⁱ²ᵉ::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    𝐒ⁱ³ᵉ::AbstractMatrix{ℱ.Dual{Z,V,N}},
                    shock_independent::Vector{ℱ.Dual{Z,V,N}};
                    kwargs...) where {Z,V,N}

    # Extract Float64 primals
    ig_f   = ℱ.value.(initial_guess)
    kb_f   = ℱ.value.(kron_buffer)
    kb²_f  = ℱ.value.(kron_buffer²)
    kb2_f  = ℱ.value.(kron_buffer2)
    kb3_f  = ℱ.value.(kron_buffer3)
    kb4_f  = ℱ.value.(kron_buffer4)
    Si_f   = ℱ.value.(𝐒ⁱ)
    Si2e_f = ℱ.value.(𝐒ⁱ²ᵉ)
    Si3e_f = ℱ.value.(𝐒ⁱ³ᵉ)
    si_f   = ℱ.value.(shock_independent)

    x_f, matched = find_shocks(Val(:LagrangeNewton),
        ig_f, kb_f, kb²_f, kb2_f, kb3_f, kb4_f, J, Si_f, Si2e_f, Si3e_f, si_f; kwargs...)

    if !matched
        return ℱ.Dual{Z,V,N}.(x_f), false
    end

    # Implicit differentiation through the linearised KKT block.
    # fXλp = [A tmp'; -tmp 0] with
    #   A = reshape((2·Si2e + 6·Si3e·kron(I,kIx))'·λ, n_x, n_x) - 2I
    #   tmp = Si + 2·Si2e·kron(I,x) + 3·Si3e·kron(I,kron(x,x))
    # 𝐒ⁱ²ᵉ/𝐒ⁱ³ᵉ live in the compressed shock pair/triple bases, so the shock
    # kron terms must be compressed too (mirrors the primal find_shocks).
    n_x = length(x_f)
    n_obs = size(Si_f, 1)
    kxx  = compressed_kron²_power(x_f)
    kxxx = compressed_kron³_power(x_f)
    kIx  = compressed_kron²(x_f, J)
    kIxx = compressed_kron³(x_f, x_f, J)

    tmp = Si_f + 2 * Si2e_f * kIx + 3 * Si3e_f * kIxx
    λ = tmp' \ (2 .* x_f)
    A_mat = zeros(V, n_x, n_x)
    compressed_pair_hessian!(A_mat, 2 .* (Si2e_f' * λ))
    compressed_triple_hessian!(A_mat, 6 .* (Si3e_f' * λ), x_f)
    # The KKT block's -2I: a loop over the diagonal, not `2 .* Matrix(I(n_x))`,
    # which materialised an n_x x n_x dense identity and a second temporary for
    # no reason. Nothing about `V` required it — this works for `Dual` too.
    @inbounds for i in 1:n_x
        A_mat[i, i] -= 2
    end

    fXλp = [A_mat   tmp';
            -tmp    zeros(V, n_obs, n_obs)]
    fXλp_lu = ℒ.lu(fXλp, check = false)
    if !ℒ.issuccess(fXλp_lu)
        return ℱ.Dual{Z,V,N}.(x_f), false
    end

    partials_matrix = zeros(V, n_x, N)

    for k in 1:N
        d_si   = V[ℱ.partials(shock_independent[i])[k] for i in eachindex(shock_independent)]
        d_Si   = V[ℱ.partials(𝐒ⁱ[i])[k]                for i in eachindex(𝐒ⁱ)]
        d_Si2e = V[ℱ.partials(𝐒ⁱ²ᵉ[i])[k]              for i in eachindex(𝐒ⁱ²ᵉ)]
        d_Si3e = V[ℱ.partials(𝐒ⁱ³ᵉ[i])[k]              for i in eachindex(𝐒ⁱ³ᵉ)]

        d_Si_mat   = reshape(d_Si,   size(Si_f))
        d_Si2e_mat = reshape(d_Si2e, size(Si2e_f))
        d_Si3e_mat = reshape(d_Si3e, size(Si3e_f))

        dtmp = d_Si_mat + 2 * d_Si2e_mat * kIx + 3 * d_Si3e_mat * kIxx
        d_g_x = dtmp' * λ
        d_g_λ = d_si - d_Si_mat * x_f - d_Si2e_mat * kxx - d_Si3e_mat * kxxx

        sol = fXλp_lu \ vcat(-d_g_x, -d_g_λ)
        partials_matrix[:, k] = sol[1:n_x]
    end

    x_dual = Vector{ℱ.Dual{Z,V,N}}(undef, n_x)
    for i in 1:n_x
        x_dual[i] = ℱ.Dual{Z,V,N}(x_f[i],
            ℱ.Partials{N,V}(NTuple{N,V}(partials_matrix[i, k] for k in 1:N)))
    end

    return x_dual, matched
end


end # module ForwardDiffExt
