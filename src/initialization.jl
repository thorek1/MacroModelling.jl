# =============================================================================
# WORKSPACE AND CACHE MANAGEMENT FUNCTIONS
# =============================================================================
# Struct constructors are defined in structures.jl
# This file contains ensure_*, update_*, and other management functions for
# workspaces, caches, constants, and calculation options
# =============================================================================

"""
    ensure_lyapunov_doubling_buffers!(ws::lyapunov_workspace{T}) where T

Ensure the doubling algorithm buffers are allocated in the workspace.
"""
function ensure_lyapunov_doubling_buffers!(ws::lyapunov_workspace{T}) where T
    n = ws.n
    if size(ws.𝐂, 1) != n
        ws.𝐂 = zeros(T, n, n)
    end
    if size(ws.𝐂¹, 1) != n
        ws.𝐂¹ = zeros(T, n, n)
    end
    if size(ws.𝐀, 1) != n
        ws.𝐀 = zeros(T, n, n)
    end
    if size(ws.𝐂A, 1) != n
        ws.𝐂A = zeros(T, n, n)
    end
    if size(ws.𝐀², 1) != n
        ws.𝐀² = zeros(T, n, n)
    end
    return ws
end

"""
    ensure_lyapunov_krylov_buffers!(ws::lyapunov_workspace{T}) where T

Ensure the Krylov method buffers are allocated in the workspace.
"""
function ensure_lyapunov_krylov_buffers!(ws::lyapunov_workspace{T}) where T
    n = ws.n
    if size(ws.tmp̄, 1) != n
        ws.tmp̄ = zeros(T, n, n)
    end
    if size(ws.𝐗, 1) != n
        ws.𝐗 = zeros(T, n, n)
    end
    if length(ws.b) != n * n
        ws.b = zeros(T, n * n)
    end
    return ws
end

"""
    ensure_lyapunov_bicgstab_solver!(ws::lyapunov_workspace{T}) where T

Ensure the bicgstab solver workspace is allocated.
"""
function ensure_lyapunov_bicgstab_solver!(ws::lyapunov_workspace{T}) where T
    ensure_lyapunov_krylov_buffers!(ws)
    n = ws.n
    if length(ws.bicgstab_workspace.x) != n * n && n > 0
        ws.bicgstab_workspace = Krylov.BicgstabWorkspace(n * n, n * n, Vector{T})
    end
    return ws
end

"""
    ensure_lyapunov_gmres_solver!(ws::lyapunov_workspace{T}) where T

Ensure the gmres solver workspace is allocated.
"""
function ensure_lyapunov_gmres_solver!(ws::lyapunov_workspace{T}) where T
    ensure_lyapunov_krylov_buffers!(ws)
    n = ws.n
    if length(ws.gmres_workspace.x) != n * n && n > 0
        ws.gmres_workspace = Krylov.GmresWorkspace(n * n, n * n, Vector{T}; memory = 20)
    end
    return ws
end

"""
    ensure_sylvester_doubling_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T

Ensure the doubling algorithm buffers are allocated in the workspace.
`n` is the row dimension (size of A), `m` is the column dimension (size of B).
"""
function ensure_sylvester_doubling_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T
    # Update stored dimensions
    ws.n = n
    ws.m = m
    
    # A-related buffers (n×n)
    if size(ws.𝐀, 1) != n || size(ws.𝐀, 2) != n
        ws.𝐀 = zeros(T, n, n)
    end
    if size(ws.𝐀¹, 1) != n || size(ws.𝐀¹, 2) != n
        ws.𝐀¹ = zeros(T, n, n)
    end
    
    # B-related buffers (m×m)
    if size(ws.𝐁, 1) != m || size(ws.𝐁, 2) != m
        ws.𝐁 = zeros(T, m, m)
    end
    if size(ws.𝐁¹, 1) != m || size(ws.𝐁¹, 2) != m
        ws.𝐁¹ = zeros(T, m, m)
    end
    
    # C-related buffers (n×m)
    if size(ws.𝐂_dbl, 1) != n || size(ws.𝐂_dbl, 2) != m
        ws.𝐂_dbl = zeros(T, n, m)
    end
    if size(ws.𝐂¹, 1) != n || size(ws.𝐂¹, 2) != m
        ws.𝐂¹ = zeros(T, n, m)
    end
    if size(ws.𝐂B, 1) != n || size(ws.𝐂B, 2) != m
        ws.𝐂B = zeros(T, n, m)
    end
    
    return ws
end

"""
    ensure_sylvester_krylov_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T

Ensure the Krylov method buffers are allocated in the workspace.
"""
function ensure_sylvester_krylov_buffers!(ws::sylvester_workspace{T}, n::Int, m::Int) where T
    ws.n = n
    ws.m = m
    
    # All are n×m matrices
    if size(ws.tmp, 1) != n || size(ws.tmp, 2) != m
        ws.tmp = zeros(T, n, m)
    end
    if size(ws.𝐗, 1) != n || size(ws.𝐗, 2) != m
        ws.𝐗 = zeros(T, n, m)
    end
    if size(ws.𝐂, 1) != n || size(ws.𝐂, 2) != m
        ws.𝐂 = zeros(T, n, m)
    end
    
    return ws
end

"""
    ensure_find_shocks_buffers!(ws::find_shocks_workspace{T}, n_exo::Int; third_order::Bool = false) where T

Ensure the find_shocks workspaces are allocated for the given number of shocks.
Only allocates 3rd order buffers if third_order=true.
Buffer sizes: kron_buffer (n_exo^2), kron_buffer2 (n_exo^2 × n_exo), 
              kron_buffer² (n_exo^3), kron_buffer3 (n_exo^3 × n_exo), kron_buffer4 (n_exo^3 × n_exo^2)
"""
function ensure_find_shocks_buffers!(ws::find_shocks_workspace{T}, n_exo::Int; third_order::Bool = false) where T
    ws.n_exo = n_exo
    
    n_exo² = n_exo^2
    n_exo³ = n_exo^3
    
    # 2nd order buffers (always needed)
    if length(ws.kron_buffer) != n_exo²
        ws.kron_buffer = zeros(T, n_exo²)
    end
    if size(ws.kron_buffer2, 1) != n_exo² || size(ws.kron_buffer2, 2) != n_exo
        ws.kron_buffer2 = zeros(T, n_exo², n_exo)
    end
    
    # 3rd order buffers (only if needed)
    if third_order
        if length(ws.kron_buffer²) != n_exo³
            ws.kron_buffer² = zeros(T, n_exo³)
        end
        if size(ws.kron_buffer3, 1) != n_exo³ || size(ws.kron_buffer3, 2) != n_exo
            ws.kron_buffer3 = zeros(T, n_exo³, n_exo)
        end
        if size(ws.kron_buffer4, 1) != n_exo³ || size(ws.kron_buffer4, 2) != n_exo²
            ws.kron_buffer4 = zeros(T, n_exo³, n_exo²)
        end
    end
    
    return ws
end


"""
    ensure_inversion_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_past::Int; third_order::Bool = false) where T

Ensure the inversion workspaces are allocated for the given dimensions.
Only allocates 3rd order buffers if third_order=true.
"""
function ensure_inversion_buffers!(ws::inversion_workspace{T}, n_exo::Int, n_past::Int; third_order::Bool = false) where T
    ws.n_exo = n_exo
    ws.n_past = n_past
    
    n_exo² = n_exo^2
    n_exo³ = n_exo^3
    n_state_vol = n_past + 1
    n_aug = n_past + 1 + n_exo
    
    # Shock-related kron buffers (2nd order)
    if length(ws.kron_buffer) != n_exo²
        ws.kron_buffer = zeros(T, n_exo²)
    end
    if size(ws.kron_buffer2, 1) != n_exo² || size(ws.kron_buffer2, 2) != n_exo
        ws.kron_buffer2 = zeros(T, n_exo², n_exo)
    end
    
    # Shock-related kron buffers (3rd order)
    if third_order
        if length(ws.kron_buffer²) != n_exo³
            ws.kron_buffer² = zeros(T, n_exo³)
        end
        if size(ws.kron_buffer3, 1) != n_exo³ || size(ws.kron_buffer3, 2) != n_exo
            ws.kron_buffer3 = zeros(T, n_exo³, n_exo)
        end
        if size(ws.kron_buffer4, 1) != n_exo³ || size(ws.kron_buffer4, 2) != n_exo²
            ws.kron_buffer4 = zeros(T, n_exo³, n_exo²)
        end
    end
    
    # State-related kron buffers
    # kron_buffer_state holds kron(J, state_vol) where J is (n_exo, n_exo) and state_vol is (n_state_vol,)
    if size(ws.kron_buffer_state, 1) != n_exo * n_state_vol || size(ws.kron_buffer_state, 2) != n_exo
        ws.kron_buffer_state = zeros(T, n_exo * n_state_vol, n_exo)
    end
    if length(ws.kronstate_vol) != n_state_vol^2
        ws.kronstate_vol = zeros(T, n_state_vol^2)
    end
    if length(ws.kronaug_state) != n_aug^2
        ws.kronaug_state = zeros(T, n_aug^2)
    end
    if third_order
        if length(ws.kron_kron_aug_state) != n_aug^3
            ws.kron_kron_aug_state = zeros(T, n_aug^3)
        end
    end
    
    # State vectors
    if length(ws.state_vol) != n_state_vol
        ws.state_vol = zeros(T, n_state_vol)
    end
    if length(ws.aug_state₁) != n_aug
        ws.aug_state₁ = zeros(T, n_aug)
    end
    if length(ws.aug_state₂) != n_aug
        ws.aug_state₂ = zeros(T, n_aug)
    end
    
    return ws
end


"""
    ensure_kalman_buffers!(ws::kalman_workspace{T}, n_obs::Int, n_states::Int) where T

Ensure the Kalman workspaces are allocated for the given dimensions.
"""
function ensure_kalman_buffers!(ws::kalman_workspace{T}, n_obs::Int, n_states::Int) where T
    # Check if dimensions changed
    if ws.n_obs == n_obs && ws.n_states == n_states
        return ws
    end
    
    ws.n_obs = n_obs
    ws.n_states = n_states
    
    # State and observation vectors
    if length(ws.u) != n_states
        ws.u = zeros(T, n_states)
    end
    if length(ws.z) != n_obs
        ws.z = zeros(T, n_obs)
    end
    if length(ws.ztmp) != n_obs
        ws.ztmp = zeros(T, n_obs)
    end
    if length(ws.utmp) != n_states
        ws.utmp = zeros(T, n_states)
    end
    
    # Matrix buffers
    if size(ws.Ctmp, 1) != n_obs || size(ws.Ctmp, 2) != n_states
        ws.Ctmp = zeros(T, n_obs, n_states)
    end
    if size(ws.F, 1) != n_obs || size(ws.F, 2) != n_obs
        ws.F = zeros(T, n_obs, n_obs)
    end
    if size(ws.K, 1) != n_states || size(ws.K, 2) != n_obs
        ws.K = zeros(T, n_states, n_obs)
    end
    if size(ws.tmp, 1) != n_states || size(ws.tmp, 2) != n_states
        ws.tmp = zeros(T, n_states, n_states)
    end
    if size(ws.Ptmp, 1) != n_states || size(ws.Ptmp, 2) != n_states
        ws.Ptmp = zeros(T, n_states, n_states)
    end
    
    return ws
end


function _axis_has_string(axis)
    axis === nothing && return false
    T = eltype(axis)
    if T === String
        return true
    elseif T === Symbol
        return false
    elseif T <: Union{Symbol, String}
        return !isempty(axis) && any(x -> x isa String, axis)
    end
    return false
end

function _choose_axis_type(var_axis, calib_axis, exo_axis_plain, exo_axis_with_subscript, full_NSSS_display)
    return (_axis_has_string(var_axis) ||
            _axis_has_string(calib_axis) ||
            _axis_has_string(exo_axis_plain) ||
            _axis_has_string(exo_axis_with_subscript) ||
            _axis_has_string(full_NSSS_display)) ? String : Symbol
end

function _convert_axis(axis, ::Type{S}) where {S <: Union{Symbol, String}}
    axis === nothing && return Vector{S}()
    return S === String ? string.(axis) : Symbol.(axis)
end

function update_post_parameters_macro(p::post_parameters_macro; kwargs...)
    return post_parameters_macro(
        get(kwargs, :parameters_as_function_of_parameters, p.parameters_as_function_of_parameters),
        get(kwargs, :precompile, p.precompile),
        get(kwargs, :simplify, p.simplify),
        get(kwargs, :symbolic, p.symbolic),
        get(kwargs, :guess, p.guess),
        get(kwargs, :ss_calib_list, p.ss_calib_list),
        get(kwargs, :par_calib_list, p.par_calib_list),
        get(kwargs, :bounds, p.bounds),
        get(kwargs, :ss_solver_parameters_algorithm, p.ss_solver_parameters_algorithm),
        get(kwargs, :ss_solver_parameters_maxtime, p.ss_solver_parameters_maxtime),
    )
end

function update_post_complete_parameters(p::post_complete_parameters; kwargs...)
    var_axis_in = get(kwargs, :var_axis, p.var_axis)
    calib_axis_in = get(kwargs, :calib_axis, p.calib_axis)
    exo_axis_plain_in = get(kwargs, :exo_axis_plain, p.exo_axis_plain)
    exo_axis_with_subscript_in = get(kwargs, :exo_axis_with_subscript, p.exo_axis_with_subscript)
    full_NSSS_display_in = get(kwargs, :full_NSSS_display, p.full_NSSS_display)
    S = _choose_axis_type(var_axis_in, calib_axis_in, exo_axis_plain_in, exo_axis_with_subscript_in, full_NSSS_display_in)
    var_axis = _convert_axis(var_axis_in, S)
    calib_axis = _convert_axis(calib_axis_in, S)
    exo_axis_plain = _convert_axis(exo_axis_plain_in, S)
    exo_axis_with_subscript = _convert_axis(exo_axis_with_subscript_in, S)
    full_NSSS_display = _convert_axis(full_NSSS_display_in, S)
    return post_complete_parameters{S}(
        get(kwargs, :parameters, p.parameters),
        get(kwargs, :missing_parameters, p.missing_parameters),
        get(kwargs, :dyn_var_future_idx, p.dyn_var_future_idx),
        get(kwargs, :dyn_var_present_idx, p.dyn_var_present_idx),
        get(kwargs, :dyn_var_past_idx, p.dyn_var_past_idx),
        get(kwargs, :dyn_ss_idx, p.dyn_ss_idx),
        # get(kwargs, :shocks_ss, p.shocks_ss),
        get(kwargs, :diag_nVars, p.diag_nVars),
        var_axis,
        calib_axis,
        exo_axis_plain,
        exo_axis_with_subscript,
        # get(kwargs, :var_has_curly, p.var_has_curly),
        # get(kwargs, :exo_has_curly, p.exo_has_curly),
        get(kwargs, :SS_and_pars_names, p.SS_and_pars_names),
        # get(kwargs, :all_variables, p.all_variables),
        # get(kwargs, :NSSS_labels, p.NSSS_labels),
        # get(kwargs, :aux_indices, p.aux_indices),
        # get(kwargs, :processed_all_variables, p.processed_all_variables),
        full_NSSS_display,
        get(kwargs, :steady_state_expand_matrix, p.steady_state_expand_matrix),
        get(kwargs, :custom_ss_expand_matrix, p.custom_ss_expand_matrix),
        get(kwargs, :vars_in_ss_equations, p.vars_in_ss_equations),
        get(kwargs, :vars_in_ss_equations_with_aux, p.vars_in_ss_equations_with_aux),
        get(kwargs, :SS_and_pars_names_lead_lag, p.SS_and_pars_names_lead_lag),
        # get(kwargs, :SS_and_pars_names_no_exo, p.SS_and_pars_names_no_exo),
        get(kwargs, :SS_and_pars_no_exo_idx, p.SS_and_pars_no_exo_idx),
        get(kwargs, :vars_idx_excluding_aux_obc, p.vars_idx_excluding_aux_obc),
        get(kwargs, :vars_idx_excluding_obc, p.vars_idx_excluding_obc),
        get(kwargs, :initialized, p.initialized),
        get(kwargs, :dyn_index, p.dyn_index),
        get(kwargs, :reverse_dynamic_order, p.reverse_dynamic_order),
        get(kwargs, :comb, p.comb),
        get(kwargs, :future_not_past_and_mixed_in_comb, p.future_not_past_and_mixed_in_comb),
        get(kwargs, :past_not_future_and_mixed_in_comb, p.past_not_future_and_mixed_in_comb),
        get(kwargs, :Ir, p.Ir),
        get(kwargs, :nabla_zero_cols, p.nabla_zero_cols),
        get(kwargs, :nabla_minus_cols, p.nabla_minus_cols),
        get(kwargs, :nabla_e_start, p.nabla_e_start),
        get(kwargs, :expand_future, p.expand_future),
        get(kwargs, :expand_past, p.expand_past),
    )
end

# Initialize all commonly used constants at once (call at entry points)
# This reduces repeated ensure_*! calls throughout the codebase
function initialise_constants!(𝓂)
    ensure_computational_constants!(𝓂)
    ensure_name_display_constants!(𝓂)
    ensure_first_order_constants!(𝓂)
    return 𝓂.constants
end

function ensure_name_display_constants!(𝓂)
    constants = 𝓂.constants
    # Use model from constants
    T = constants.post_model_macro
    
    if isempty(constants.post_complete_parameters.var_axis)
        var_has_curly = any(x -> contains(string(x), "◖"), T.var)
        if var_has_curly
            var_decomposed = decompose_name.(T.var)
            var_axis = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in var_decomposed]
        else
            var_axis = T.var
        end

        if var_has_curly
            calib_axis = replace.(string.(𝓂.equations.calibration_parameters), "◖" => "{", "◗" => "}")
        else
            calib_axis = 𝓂.equations.calibration_parameters
        end

        exo_has_curly = any(x -> contains(string(x), "◖"), T.exo)
        if exo_has_curly
            exo_decomposed = decompose_name.(T.exo)
            exo_axis_plain = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in exo_decomposed]
            exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), exo_axis_plain)
        else
            exo_axis_plain = T.exo
            exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), T.exo)
        end

        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            var_axis = var_axis,
            calib_axis = calib_axis,
            exo_axis_plain = exo_axis_plain,
            exo_axis_with_subscript = exo_axis_with_subscript,
            var_has_curly = var_has_curly,
            exo_has_curly = exo_has_curly,
        )
    end

    return constants.post_complete_parameters
end


function set_up_name_display_cache(T::post_model_macro, calibration_equations_parameters)
    var_has_curly = any(x -> contains(string(x), "◖"), T.var)
    if var_has_curly
        var_decomposed = decompose_name.(T.var)
        var_axis = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in var_decomposed]
    else
        var_axis = T.var
    end

    if var_has_curly
        calib_axis = replace.(string.(calibration_equations_parameters), "◖" => "{", "◗" => "}")
    else
        calib_axis = calibration_equations_parameters
    end

    exo_has_curly = any(x -> contains(string(x), "◖"), T.exo)
    if exo_has_curly
        exo_decomposed = decompose_name.(T.exo)
        exo_axis_plain = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in exo_decomposed]
        exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), exo_axis_plain)
    else
        exo_axis_plain = T.exo
        exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), T.exo)
    end

    return (
        var_axis = var_axis,
        calib_axis = calib_axis,
        exo_axis_plain = exo_axis_plain,
        exo_axis_with_subscript = exo_axis_with_subscript,
        var_has_curly = var_has_curly,
        exo_has_curly = exo_has_curly,
    )
end


function ensure_computational_constants!(𝓂)
    constants = 𝓂.constants
    so = constants.second_order
    if isempty(so.s_in_s⁺)
        # Use timings from constants if available, otherwise from model
        T = constants.post_model_macro
        nᵉ = T.nExo
        nˢ = T.nPast_not_future_and_mixed

        s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ + 1), zeros(Bool, nᵉ)))
        s_in_s = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)

        kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
        kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)

        e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_e_s = ℒ.kron(e_in_s⁺, s_in_s⁺)

        # Compute sparse index patterns for filter operations
        shockvar_idxs = sparse(ℒ.kron(e_in_s⁺, s_in_s⁺)).nzind
        shock_idxs = sparse(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)).nzind
        shock_idxs2 = sparse(ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)).nzind
        shock²_idxs = sparse(ℒ.kron(e_in_s⁺, e_in_s⁺)).nzind
        var_vol²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind

        so.s_in_s⁺ = s_in_s⁺
        so.s_in_s = s_in_s
        so.kron_s⁺_s⁺ = kron_s⁺_s⁺
        so.kron_s⁺_s = kron_s⁺_s
        so.kron_s⁺_s⁺_s⁺ = kron_s⁺_s⁺_s⁺
        so.kron_s_s⁺_s⁺ = kron_s_s⁺_s⁺
        so.e_in_s⁺ = e_in_s⁺
        so.v_in_s⁺ = v_in_s⁺
        so.kron_s_s = kron_s_s
        so.kron_e_e = kron_e_e
        so.kron_v_v = kron_v_v
        so.kron_e_s = kron_e_s
        so.shockvar_idxs = shockvar_idxs
        so.shock_idxs = shock_idxs
        so.shock_idxs2 = shock_idxs2
        so.shock²_idxs = shock²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    return constants.second_order
end

function ensure_computational_constants!(constants::constants)
    so = constants.second_order
    if isempty(so.s_in_s⁺)
        # Use timings from constants
        T = constants.post_model_macro
        nᵉ = T.nExo
        nˢ = T.nPast_not_future_and_mixed

        s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ + 1), zeros(Bool, nᵉ)))
        s_in_s = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)

        kron_s⁺_s⁺_s⁺ = ℒ.kron(s_in_s⁺, kron_s⁺_s⁺)
        kron_s_s⁺_s⁺ = ℒ.kron(kron_s⁺_s⁺, s_in_s)

        e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_e_s = ℒ.kron(e_in_s⁺, s_in_s⁺)

        # Compute sparse index patterns for filter operations
        shockvar_idxs = sparse(ℒ.kron(e_in_s⁺, s_in_s⁺)).nzind
        shock_idxs = sparse(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)).nzind
        shock_idxs2 = sparse(ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)).nzind
        shock²_idxs = sparse(ℒ.kron(e_in_s⁺, e_in_s⁺)).nzind
        var_vol²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind

        so.s_in_s⁺ = s_in_s⁺
        so.s_in_s = s_in_s
        so.kron_s⁺_s⁺ = kron_s⁺_s⁺
        so.kron_s⁺_s = kron_s⁺_s
        so.kron_s⁺_s⁺_s⁺ = kron_s⁺_s⁺_s⁺
        so.kron_s_s⁺_s⁺ = kron_s_s⁺_s⁺
        so.e_in_s⁺ = e_in_s⁺
        so.v_in_s⁺ = v_in_s⁺
        so.kron_s_s = kron_s_s
        so.kron_e_e = kron_e_e
        so.kron_v_v = kron_v_v
        so.kron_e_s = kron_e_s
        so.shockvar_idxs = shockvar_idxs
        so.shock_idxs = shock_idxs
        so.shock_idxs2 = shock_idxs2
        so.shock²_idxs = shock²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    return constants.second_order
end

function ensure_conditional_forecast_constants!(𝓂; third_order::Bool = false)
    constants = 𝓂.constants
    so = ensure_computational_constants!(𝓂)

    if isempty(so.var²_idxs)
        s_in_s⁺ = so.s_in_s
        e_in_s⁺ = so.e_in_s⁺

        shock_idxs = so.shock_idxs
        shock²_idxs = so.shock²_idxs
        shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)
        var_vol²_idxs = so.var_vol²_idxs
        var²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind
        so.var²_idxs = var²_idxs
        so.shockvar²_idxs = shockvar²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    if third_order
        to = constants.third_order
        if isempty(to.var_vol³_idxs)
            sv_in_s⁺ = so.s_in_s⁺
            e_in_s⁺ = so.e_in_s⁺
            ones_e = zero(e_in_s⁺) .+ 1

            var_vol³_idxs = sparse(ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺))).nzind
            shock_idxs2 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, ones_e), ones_e)).nzind
            shock_idxs3 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), ones_e)).nzind
            shock³_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar1_idxs = sparse(ℒ.kron(ones_e, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar2_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(ones_e, e_in_s⁺))).nzind
            shockvar3_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, ones_e))).nzind
            shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
            shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)

            to.var_vol³_idxs = var_vol³_idxs
            to.shock_idxs2 = shock_idxs2
            to.shock_idxs3 = shock_idxs3
            to.shock³_idxs = shock³_idxs
            to.shockvar1_idxs = shockvar1_idxs
            to.shockvar2_idxs = shockvar2_idxs
            to.shockvar3_idxs = shockvar3_idxs
            to.shockvar³2_idxs = shockvar³2_idxs
            to.shockvar³_idxs = shockvar³_idxs
        end
    end

    return so
end

function ensure_conditional_forecast_constants!(constants::constants; third_order::Bool = false)
    so = ensure_computational_constants!(constants)

    if isempty(so.var²_idxs)
        s_in_s⁺ = so.s_in_s
        e_in_s⁺ = so.e_in_s⁺

        shock_idxs = so.shock_idxs
        shock²_idxs = so.shock²_idxs
        shockvar²_idxs = setdiff(shock_idxs, shock²_idxs)
        var_vol²_idxs = so.var_vol²_idxs
        var²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind
        so.var²_idxs = var²_idxs
        so.shockvar²_idxs = shockvar²_idxs
        so.var_vol²_idxs = var_vol²_idxs
    end

    if third_order
        to = constants.third_order
        if isempty(to.var_vol³_idxs)
            sv_in_s⁺ = so.s_in_s⁺
            e_in_s⁺ = so.e_in_s⁺
            ones_e = zero(e_in_s⁺) .+ 1

            var_vol³_idxs = sparse(ℒ.kron(sv_in_s⁺, ℒ.kron(sv_in_s⁺, sv_in_s⁺))).nzind
            shock_idxs2 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, ones_e), ones_e)).nzind
            shock_idxs3 = sparse(ℒ.kron(ℒ.kron(e_in_s⁺, e_in_s⁺), ones_e)).nzind
            shock³_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar1_idxs = sparse(ℒ.kron(ones_e, ℒ.kron(e_in_s⁺, e_in_s⁺))).nzind
            shockvar2_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(ones_e, e_in_s⁺))).nzind
            shockvar3_idxs = sparse(ℒ.kron(e_in_s⁺, ℒ.kron(e_in_s⁺, ones_e))).nzind
            shockvar³2_idxs = setdiff(shock_idxs2, shock³_idxs, shockvar1_idxs, shockvar2_idxs, shockvar3_idxs)
            shockvar³_idxs = setdiff(shock_idxs3, shock³_idxs)

            to.var_vol³_idxs = var_vol³_idxs
            to.shock_idxs2 = shock_idxs2
            to.shock_idxs3 = shock_idxs3
            to.shock³_idxs = shock³_idxs
            to.shockvar1_idxs = shockvar1_idxs
            to.shockvar2_idxs = shockvar2_idxs
            to.shockvar3_idxs = shockvar3_idxs
            to.shockvar³2_idxs = shockvar³2_idxs
            to.shockvar³_idxs = shockvar³_idxs
        end
    end

    return so
end

function build_first_order_index_cache(T, I_nVars)
    dyn_index = T.nPresent_only + 1:T.nVars

    reverse_dynamic_order_tmp = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    if any(isnothing.(reverse_dynamic_order_tmp))
        reverse_dynamic_order = Int[]
    else
        reverse_dynamic_order = Int.(reverse_dynamic_order_tmp)
    end
    
    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb_tmp = indexin(T.future_not_past_and_mixed_idx, comb)
    
    if any(isnothing.(future_not_past_and_mixed_in_comb_tmp))
        future_not_past_and_mixed_in_comb = Int[]
    else
        future_not_past_and_mixed_in_comb = Int.(future_not_past_and_mixed_in_comb_tmp)
    end

    past_not_future_and_mixed_in_comb_tmp = indexin(T.past_not_future_and_mixed_idx, comb)
    
    if any(isnothing.(past_not_future_and_mixed_in_comb_tmp))
        past_not_future_and_mixed_in_comb = Int[]
    else
        past_not_future_and_mixed_in_comb = Int.(past_not_future_and_mixed_in_comb_tmp)
    end

    Ir = ℒ.I(length(comb))

    nabla_zero_cols = (T.nFuture_not_past_and_mixed + 1):(T.nFuture_not_past_and_mixed + T.nVars)
    nabla_minus_cols = (T.nFuture_not_past_and_mixed + T.nVars + 1):(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed)
    nabla_e_start = T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1

    expand_future = I_nVars[T.future_not_past_and_mixed_idx,:]
    expand_past = I_nVars[T.past_not_future_and_mixed_idx,:]

    return (
        initialized = true,
        dyn_index = dyn_index,
        reverse_dynamic_order = reverse_dynamic_order,
        comb = comb,
        future_not_past_and_mixed_in_comb = future_not_past_and_mixed_in_comb,
        past_not_future_and_mixed_in_comb = past_not_future_and_mixed_in_comb,
        Ir = Ir,
        nabla_zero_cols = nabla_zero_cols,
        nabla_minus_cols = nabla_minus_cols,
        nabla_e_start = nabla_e_start,
        expand_future = expand_future,
        expand_past = expand_past,
    )
end

function ensure_first_order_constants!(𝓂)
    constants = 𝓂.constants
    if !constants.post_complete_parameters.initialized
        # Use timings from constants if available, otherwise from model
        T = constants.post_model_macro
        diag_nVars = constants.post_complete_parameters.diag_nVars
        if size(diag_nVars, 1) == 0
            diag_nVars = ℒ.I(T.nVars)
        end
        cache = build_first_order_index_cache(T, diag_nVars)
        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            diag_nVars = diag_nVars,
            initialized = cache.initialized,
            dyn_index = cache.dyn_index,
            reverse_dynamic_order = cache.reverse_dynamic_order,
            comb = cache.comb,
            future_not_past_and_mixed_in_comb = cache.future_not_past_and_mixed_in_comb,
            past_not_future_and_mixed_in_comb = cache.past_not_future_and_mixed_in_comb,
            Ir = cache.Ir,
            nabla_zero_cols = cache.nabla_zero_cols,
            nabla_minus_cols = cache.nabla_minus_cols,
            nabla_e_start = cache.nabla_e_start,
            expand_future = cache.expand_future,
            expand_past = cache.expand_past,
        )
    end
    return constants.post_complete_parameters
end

function ensure_first_order_constants!(constants::constants)
    if !constants.post_complete_parameters.initialized
        # Use timings from constants if available
        T = constants.post_model_macro
        diag_nVars = constants.post_complete_parameters.diag_nVars
        if size(diag_nVars, 1) == 0
            diag_nVars = ℒ.I(T.nVars)
        end
        cache = build_first_order_index_cache(T, diag_nVars)
        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            diag_nVars = diag_nVars,
            initialized = cache.initialized,
            dyn_index = cache.dyn_index,
            reverse_dynamic_order = cache.reverse_dynamic_order,
            comb = cache.comb,
            future_not_past_and_mixed_in_comb = cache.future_not_past_and_mixed_in_comb,
            past_not_future_and_mixed_in_comb = cache.past_not_future_and_mixed_in_comb,
            Ir = cache.Ir,
            nabla_zero_cols = cache.nabla_zero_cols,
            nabla_minus_cols = cache.nabla_minus_cols,
            nabla_e_start = cache.nabla_e_start,
            expand_future = cache.expand_future,
            expand_past = cache.expand_past,
        )
    end
    return constants.post_complete_parameters
end


"""
    ensure_qme_workspace!(𝓂)
    ensure_qme_workspace!(workspaces, n)

Ensure the QME (quadratic matrix equation) workspace is properly sized for the model.
The workspace dimension is `n = nVars - nPresent_only` (the size of the QME matrices).
If the workspace is the wrong size, it will be reallocated.
"""
function ensure_qme_workspace!(𝓂)
    T = 𝓂.constants.post_model_macro
    n = T.nVars - T.nPresent_only
    nPast = T.nPast_not_future_and_mixed
    return ensure_qme_workspace!(𝓂.workspaces, n, nPast)
end

function ensure_qme_workspace!(workspaces::workspaces, n::Int, nPast::Int = 0)
    ws = workspaces.qme
    # Check if workspace needs to be resized (either n or nPast changed)
    if size(ws.E, 1) != n || size(ws.I_nPast, 1) != nPast
        workspaces.qme = Qme_workspace(n, nPast = nPast)
    end
    return workspaces.qme
end

"""
    ensure_sylvester_1st_order_workspace!(𝓂)
    ensure_sylvester_1st_order_workspace!(workspaces)

Return the first-order sylvester workspace from the model or workspaces.
The workspace is lazily sized by the sylvester solver when needed.
"""
function ensure_sylvester_1st_order_workspace!(𝓂)
    return 𝓂.workspaces.sylvester_1st_order
end

function ensure_sylvester_1st_order_workspace!(workspaces::workspaces)
    return workspaces.sylvester_1st_order
end


"""
    ensure_lyapunov_workspace!(workspaces, n, order::Symbol)

Ensure the Lyapunov workspace for the specified moment order is properly sized.
`n` is the dimension of the square matrices.
`order` should be `:first_order`, `:second_order`, or `:third_order`.
If the workspace is the wrong size, it will be reallocated.
Note: buffers are still lazily allocated when algorithms are actually used.
"""
function ensure_lyapunov_workspace!(workspaces::workspaces, n::Int, order::Symbol)
    if order == :first_order
        ws = workspaces.lyapunov_1st_order
        if ws.n != n
            workspaces.lyapunov_1st_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_1st_order
    elseif order == :second_order
        ws = workspaces.lyapunov_2nd_order
        if ws.n != n
            workspaces.lyapunov_2nd_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_2nd_order
    elseif order == :third_order
        ws = workspaces.lyapunov_3rd_order
        if ws.n != n
            workspaces.lyapunov_3rd_order = Lyapunov_workspace(n)
        end
        return workspaces.lyapunov_3rd_order
    else
        error("Invalid order: $order. Must be :first_order, :second_order, or :third_order")
    end
end

"""
    ensure_lyapunov_workspace_1st_order!(𝓂)

Ensure the first-order Lyapunov workspace is properly sized for the model.
The dimension is `nVars` (size of the covariance matrix).
"""
function ensure_lyapunov_workspace_1st_order!(𝓂)
    T = 𝓂.constants.post_model_macro
    n = T.nVars
    return ensure_lyapunov_workspace!(𝓂.workspaces, n, :first_order)
end


"""
    ensure_inversion_workspace!(𝓂; third_order::Bool = false)

Ensure the inversion filter workspace is properly sized for the model.
Dimensions are based on nExo (number of shocks) and nPast_not_future_and_mixed.
"""
function ensure_inversion_workspace!(𝓂; third_order::Bool = false)
    T = 𝓂.constants.post_model_macro
    n_exo = T.nExo
    n_past = T.nPast_not_future_and_mixed
    ensure_inversion_buffers!(𝓂.workspaces.inversion, n_exo, n_past; third_order = third_order)
    return 𝓂.workspaces.inversion
end


"""
    ensure_kalman_workspace!(𝓂)

Ensure the Kalman filter workspace is available. Returns the workspace for use.
Actual buffer resizing happens lazily in ensure_kalman_buffers! when dimensions are known.
"""
function ensure_kalman_workspace!(𝓂)
    return 𝓂.workspaces.kalman
end


function create_selector_matrix(target::Vector{Symbol}, source::Vector{Symbol})
    selector = spzeros(Float64, length(target), length(source))
    idx = indexin(target, source)
    for (i, j) in enumerate(idx)
        if !isnothing(j)
            selector[i, j] = 1.0
        end
    end
    return selector
end

function ensure_model_structure_constants!(constants::constants, calibration_parameters::Vector{Symbol})
    T = constants.post_model_macro
    if isempty(constants.post_complete_parameters.SS_and_pars_names)
        SS_and_pars_names = vcat(
            Symbol.(replace.(string.(sort(union(T.var, T.exo_past, T.exo_future))),
                    r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),
            calibration_parameters,
        )

        all_variables = Symbol.(sort(union(T.var, T.aux, T.exo_present)))

        NSSS_labels = Symbol.(vcat(sort(union(T.exo_present, T.var)), calibration_parameters))

        aux_indices = Int.(indexin(T.aux, all_variables))
        processed_all_variables = copy(all_variables)
        processed_all_variables[aux_indices] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), T.aux)

        full_NSSS = copy(processed_all_variables)
        if any(x -> contains(string(x), "◖"), full_NSSS)
            full_NSSS_decomposed = decompose_name.(full_NSSS)
            full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
        end
        # full_NSSS_display = Vector{Union{Symbol, String}}(full_NSSS)
        full_NSSS_display = copy(full_NSSS)

        steady_state_expand_matrix = create_selector_matrix(processed_all_variables, NSSS_labels)

        vars_in_ss_equations = T.vars_in_ss_equations_no_aux
        vars_in_ss_equations_with_aux = T.vars_in_ss_equations
        extended_SS_and_pars = vcat(map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), T.var), calibration_parameters)
        custom_ss_expand_matrix = create_selector_matrix(extended_SS_and_pars, vcat(vars_in_ss_equations, calibration_parameters))

        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(T.var, T.exo_past, T.exo_future)))), calibration_parameters)
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(T.var, T.exo_past, T.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), calibration_parameters)
        SS_and_pars_no_exo_idx = Int.(indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag))

        vars_non_obc = T.var[.!contains.(string.(T.var), "ᵒᵇᶜ")]
        vars_idx_excluding_aux_obc = Int.(indexin(setdiff(vars_non_obc, union(T.aux, T.exo_present)), all_variables))
        vars_idx_excluding_obc = Int.(indexin(vars_non_obc, all_variables))

        constants.post_complete_parameters = update_post_complete_parameters(
            constants.post_complete_parameters;
            SS_and_pars_names = SS_and_pars_names,
            # all_variables = all_variables,
            # NSSS_labels = NSSS_labels,
            # aux_indices = aux_indices,
            # processed_all_variables = processed_all_variables,
            full_NSSS_display = full_NSSS_display,
            steady_state_expand_matrix = steady_state_expand_matrix,
            custom_ss_expand_matrix = custom_ss_expand_matrix,
            vars_in_ss_equations = vars_in_ss_equations,
            vars_in_ss_equations_with_aux = vars_in_ss_equations_with_aux,
            SS_and_pars_names_lead_lag = SS_and_pars_names_lead_lag,
            # SS_and_pars_names_no_exo = SS_and_pars_names_no_exo,
            SS_and_pars_no_exo_idx = SS_and_pars_no_exo_idx,
            vars_idx_excluding_aux_obc = vars_idx_excluding_aux_obc,
            vars_idx_excluding_obc = vars_idx_excluding_obc,
        )
    end

    return constants.post_complete_parameters
end

function compute_e4(nᵉ::Int)
    if nᵉ == 0
        return Float64[]
    end
    E_e4 = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4)
    quadrup = multiplicate(nᵉ, 4)
    comb4 = reduce(vcat, generateSumVectors(nᵉ, 4))
    comb4 = comb4 isa Int64 ? reshape([comb4], 1, 1) : comb4
    for j = 1:size(comb4, 1)
        E_e4[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb4[j, :])
    end
    return quadrup * E_e4
end

function compute_e6(nᵉ::Int)
    if nᵉ == 0
        return Float64[]
    end
    E_e6 = zeros(nᵉ * (nᵉ + 1)÷2 * (nᵉ + 2)÷3 * (nᵉ + 3)÷4 * (nᵉ + 4)÷5 * (nᵉ + 5)÷6)
    sextup = multiplicate(nᵉ, 6)
    comb6 = reduce(vcat, generateSumVectors(nᵉ, 6))
    comb6 = comb6 isa Int64 ? reshape([comb6], 1, 1) : comb6
    for j = 1:size(comb6, 1)
        E_e6[j] = product_moments(ℒ.I(nᵉ), 1:nᵉ, comb6[j, :])
    end
    return sextup * E_e6
end

function ensure_moments_constants!(constants::constants)
    so = ensure_computational_constants!(constants)
    to = constants.third_order
    # Use timings from constants
    T = constants.post_model_macro
    nᵉ = T.nExo
    
    if isempty(so.kron_states)
        so.kron_states = ℒ.kron(so.s_in_s, so.s_in_s)
    end
    if isempty(so.kron_s_e)
        so.kron_s_e = ℒ.kron(so.s_in_s, so.e_in_s⁺)
    end
    if size(so.I_plus_s_s, 1) == 0
        nˢ = T.nPast_not_future_and_mixed
        so.I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
    end
    if isempty(so.e4)
        so.e4 = compute_e4(nᵉ)
    end
    # Cache vec(I(nᵉ)) - used repeatedly in moments calculations
    if isempty(so.vec_Iₑ)
        so.vec_Iₑ = vec(collect(Float64, ℒ.I(nᵉ)))
    end
    # Cache reshaped e4 matrices - used repeatedly in moments calculations
    if isempty(so.e4_nᵉ²_nᵉ²)
        so.e4_nᵉ²_nᵉ² = reshape(so.e4, nᵉ^2, nᵉ^2)
    end
    if isempty(so.e4_nᵉ_nᵉ³)
        so.e4_nᵉ_nᵉ³ = reshape(so.e4, nᵉ, nᵉ^3)
    end
    # Cache e4 minus outer product of vec(I) - used in Γ₂ and Γ₃ matrices
    if isempty(so.e4_minus_vecIₑ_outer)
        so.e4_minus_vecIₑ_outer = so.e4_nᵉ²_nᵉ² - so.vec_Iₑ * so.vec_Iₑ'
    end
    if isempty(to.e6)
        to.e6 = compute_e6(nᵉ)
    end
    if isempty(to.kron_e_v)
        to.kron_e_v = ℒ.kron(so.e_in_s⁺, so.v_in_s⁺)
    end
    # Cache reshaped e6 matrix - used in third order moments calculations
    if isempty(to.e6_nᵉ³_nᵉ³)
        to.e6_nᵉ³_nᵉ³ = reshape(to.e6, nᵉ^3, nᵉ^3)
    end
    return so
end

function ensure_moments_substate_indices!(𝓂, nˢ::Int)
    constants = 𝓂.constants
    to = constants.third_order
    if !haskey(to.substate_indices, nˢ)
        # Use timings from constants if available, otherwise from model
        T = constants.post_model_macro
        nᵉ = T.nExo
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ * nˢ)), nˢ * nᵉ^2, nˢ * nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ * nˢ^2, nᵉ * nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))
        to.substate_indices[nˢ] = moments_substate_indices(I_plus_s_s, e_es, e_ss, ss_s, s_s)
    end
    return to.substate_indices[nˢ]
end

function ensure_moments_dependency_kron_indices!(𝓂, dependencies::Vector{Symbol}, s_in_s⁺::BitVector)
    constants = 𝓂.constants
    to = constants.third_order
    key = Tuple(dependencies)
    if !haskey(to.dependency_kron_indices, key)
        so = ensure_computational_constants!(𝓂)
        to.dependency_kron_indices[key] = moments_dependency_kron_indices(
            ℒ.kron(s_in_s⁺, s_in_s⁺),
            ℒ.kron(s_in_s⁺, so.e_in_s⁺),
            ℒ.kron(s_in_s⁺, so.v_in_s⁺),
        )
    end
    return to.dependency_kron_indices[key]
end


