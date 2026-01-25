
function Second_order_cache()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    empty_sparse_float = spzeros(Float64, 0, 0)
    return second_order(
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        BitVector(),
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        BitVector(),
        empty_sparse_float,
        Float64[],
    )
end

function Third_order_cache()
    empty_sparse_int = SparseMatrixCSC{Int, Int64}(ℒ.I, 0, 0)
    return third_order(
        empty_sparse_int,
        empty_sparse_int,
        Dict{Vector{Int}, Int}(),
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        empty_sparse_int,
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Int[],
        Float64[],
        BitVector(),
        Dict{Int, moments_substate_indices}(),
        Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_indices}(),
    )
end


function Krylov_workspace(;S::Type = Float64)
    krylov_workspace(  GmresWorkspace(0,0,Vector{S}),
                    DqgmresWorkspace(0,0,Vector{S}),
                    BicgstabWorkspace(0,0,Vector{S}))
end

function Sylvester_workspace(;S::Type = Float64)
    sylvester_workspace(   zeros(S,0,0),
                        zeros(S,0,0),
                        zeros(S,0,0),
                        Krylov_workspace(S = S))
end

function Higher_order_workspace(;T::Type = Float64, S::Type = Float64)
    higher_order_workspace(spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        spzeros(T,0,0),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        (Int[], Int[], T[], Int[], Int[], Int[], T[]),
                        zeros(T,0,0),
                        Sylvester_workspace(S = S))
end

function Workspaces(;T::Type = Float64, S::Type = Float64)
    workspaces(Higher_order_workspace(T = T, S = S),
                Higher_order_workspace(T = T, S = S),
                Float64[])
end

function Constants(model_struct; T::Type = Float64, S::Type = Float64)
    constants( model_struct,
            post_parameters_macro(
                Symbol[],
                false,
                true,
                Dict{Symbol, Float64}(),
                Set{Symbol}[],
                Set{Symbol}[],
                # Set{Symbol}[],
                # Set{Symbol}[],
                Dict{Symbol,Tuple{Float64,Float64}}()
                ),
            post_complete_parameters{Symbol}(
                Symbol[],
                Symbol[],
                Int[],
                Int[],
                Int[],
                Int[],
                # Int[],
                ℒ.I(0),
                Symbol[],
                Symbol[],
                Symbol[],
                Symbol[],
                # false,
                # false,
                Symbol[],
                # Symbol[],
                # Symbol[],
                # Int[],
                # Symbol[],
                Symbol[],
                spzeros(Float64, 0, 0),
                spzeros(Float64, 0, 0),
                Symbol[],
                Symbol[],
                Symbol[],
                # Symbol[],
                Int[],
                Int[],
                Int[],
                false,
                1:0,
                Int[],
                Int[],
                Int[],
                Int[],
                ℒ.I(0),
                1:0,
                1:0,
                1,
                zeros(Bool, 0, 0),
                zeros(Bool, 0, 0)),
            Second_order_cache(),
            Third_order_cache())
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
# This reduces repeated ensure_*_cache! calls throughout the codebase
function initialise_constants!(𝓂)
    ensure_computational_constants_cache!(𝓂)
    ensure_name_display_cache!(𝓂)
    ensure_first_order_index_cache!(𝓂)
    return 𝓂.constants
end

function ensure_name_display_cache!(𝓂)
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


function ensure_computational_constants_cache!(𝓂)
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

function ensure_computational_constants_cache!(constants::constants)
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

function ensure_conditional_forecast_index_cache!(𝓂; third_order::Bool = false)
    constants = 𝓂.constants
    so = ensure_computational_constants_cache!(𝓂)

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

function ensure_conditional_forecast_index_cache!(constants::constants; third_order::Bool = false)
    so = ensure_computational_constants_cache!(constants)

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

function ensure_first_order_index_cache!(𝓂)
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

function ensure_first_order_index_cache!(constants::constants)
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

function ensure_model_structure_cache!(𝓂)
    constants = 𝓂.constants
    if isempty(constants.post_complete_parameters.SS_and_pars_names)
        SS_and_pars_names = vcat(
            Symbol.(replace.(string.(sort(union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future))),
                    r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),
            𝓂.equations.calibration_parameters,
        )

        all_variables = Symbol.(sort(union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.aux, 𝓂.constants.post_model_macro.exo_present)))

        NSSS_labels = Symbol.(vcat(sort(union(𝓂.constants.post_model_macro.exo_present, 𝓂.constants.post_model_macro.var)), 𝓂.equations.calibration_parameters))

        aux_indices = Int.(indexin(𝓂.constants.post_model_macro.aux, all_variables))
        processed_all_variables = copy(all_variables)
        processed_all_variables[aux_indices] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.constants.post_model_macro.aux)

        full_NSSS = copy(processed_all_variables)
        if any(x -> contains(string(x), "◖"), full_NSSS)
            full_NSSS_decomposed = decompose_name.(full_NSSS)
            full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
        end
        # full_NSSS_display = Vector{Union{Symbol, String}}(full_NSSS)
        full_NSSS_display = copy(full_NSSS)

        steady_state_expand_matrix = create_selector_matrix(processed_all_variables, NSSS_labels)

        vars_in_ss_equations = 𝓂.constants.post_model_macro.vars_in_ss_equations_no_aux
        vars_in_ss_equations_with_aux = 𝓂.constants.post_model_macro.vars_in_ss_equations
        extended_SS_and_pars = vcat(map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.constants.post_model_macro.var), 𝓂.equations.calibration_parameters)
        custom_ss_expand_matrix = create_selector_matrix(extended_SS_and_pars, vcat(vars_in_ss_equations, 𝓂.equations.calibration_parameters))

        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future)))), 𝓂.equations.calibration_parameters)
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.constants.post_model_macro.var, 𝓂.constants.post_model_macro.exo_past, 𝓂.constants.post_model_macro.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.equations.calibration_parameters)
        SS_and_pars_no_exo_idx = Int.(indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag))

        vars_non_obc = 𝓂.constants.post_model_macro.var[.!contains.(string.(𝓂.constants.post_model_macro.var), "ᵒᵇᶜ")]
        vars_idx_excluding_aux_obc = Int.(indexin(setdiff(vars_non_obc, union(𝓂.constants.post_model_macro.aux, 𝓂.constants.post_model_macro.exo_present)), all_variables))
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

function ensure_moments_cache!(𝓂)
    constants = 𝓂.constants
    so = ensure_computational_constants_cache!(𝓂)
    to = constants.third_order
    # Use timings from constants if available, otherwise from model
    T = constants.post_model_macro
    
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
        so.e4 = compute_e4(T.nExo)
    end
    if isempty(to.e6)
        to.e6 = compute_e6(T.nExo)
    end
    if isempty(to.kron_e_v)
        to.kron_e_v = ℒ.kron(so.e_in_s⁺, so.v_in_s⁺)
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
        so = ensure_computational_constants_cache!(𝓂)
        to.dependency_kron_indices[key] = moments_dependency_kron_indices(
            ℒ.kron(s_in_s⁺, s_in_s⁺),
            ℒ.kron(s_in_s⁺, so.e_in_s⁺),
            ℒ.kron(s_in_s⁺, so.v_in_s⁺),
        )
    end
    return to.dependency_kron_indices[key]
end


struct Tolerances
    NSSS_acceptance_tol::AbstractFloat
    NSSS_xtol::AbstractFloat
    NSSS_ftol::AbstractFloat
    NSSS_rel_xtol::AbstractFloat

    qme_tol::AbstractFloat
    qme_acceptance_tol::AbstractFloat

    sylvester_tol::AbstractFloat
    sylvester_acceptance_tol::AbstractFloat

    lyapunov_tol::AbstractFloat
    lyapunov_acceptance_tol::AbstractFloat

    droptol::AbstractFloat

    dependencies_tol::AbstractFloat
end

struct CalculationOptions
    quadratic_matrix_equation_algorithm::Symbol
    
    sylvester_algorithm²::Symbol
    sylvester_algorithm³::Symbol
    
    lyapunov_algorithm::Symbol
    
    tol::Tolerances
    verbose::Bool
end

@stable default_mode = "disable" begin
"""
$(SIGNATURES)
Function to manually define tolerances for the solvers of various problems: non-stochastic steady state solver (NSSS), Sylvester equations, Lyapunov equation, and quadratic matrix equation (qme).

# Keyword Arguments
- `NSSS_acceptance_tol` [Default: `1e-12`, Type: `AbstractFloat`]: Acceptance tolerance for non-stochastic steady state solver.
- `NSSS_xtol` [Default: `1e-12`, Type: `AbstractFloat`]: Absolute tolerance for solver steps for non-stochastic steady state solver.
- `NSSS_ftol` [Default: `1e-14`, Type: `AbstractFloat`]: Absolute tolerance for solver function values for non-stochastic steady state solver.
- `NSSS_rel_xtol` [Default: `eps()`, Type: `AbstractFloat`]: Relative tolerance for solver steps for non-stochastic steady state solver.

- `qme_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for quadratic matrix equation solver.
- `qme_acceptance_tol` [Default: `1e-8`, Type: `AbstractFloat`]: Acceptance tolerance for quadratic matrix equation solver.

- `sylvester_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for Sylvester equation solver.
- `sylvester_acceptance_tol` [Default: `1e-10`, Type: `AbstractFloat`]: Acceptance tolerance for Sylvester equation solver.

- `lyapunov_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for Lyapunov equation solver.
- `lyapunov_acceptance_tol` [Default: `1e-12`, Type: `AbstractFloat`]: Acceptance tolerance for Lyapunov equation solver.

- `droptol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance below which matrix entries are considered 0.

- `dependencies_tol` [Default: `1e-12`, Type: `AbstractFloat`]: tolerance for the effect of a variable on the variable of interest when isolating part of the system for calculating covariance related statistics
"""
function Tolerances(;NSSS_acceptance_tol::AbstractFloat = 1e-12,
                    NSSS_xtol::AbstractFloat = 1e-12,
                    NSSS_ftol::AbstractFloat = 1e-14,
                    NSSS_rel_xtol::AbstractFloat = eps(),
                    
                    qme_tol::AbstractFloat = 1e-14,
                    qme_acceptance_tol::AbstractFloat = 1e-8,

                    sylvester_tol::AbstractFloat = 1e-14,
                    sylvester_acceptance_tol::AbstractFloat = 1e-10,

                    lyapunov_tol::AbstractFloat = 1e-14,
                    lyapunov_acceptance_tol::AbstractFloat = 1e-12,

                    droptol::AbstractFloat = 1e-14,

                    dependencies_tol::AbstractFloat = 1e-12)
    
    return Tolerances(NSSS_acceptance_tol,
                        NSSS_xtol,
                        NSSS_ftol,
                        NSSS_rel_xtol, 
                        qme_tol,
                        qme_acceptance_tol,
                        sylvester_tol,
                        sylvester_acceptance_tol,
                        lyapunov_tol,
                        lyapunov_acceptance_tol,
                        droptol,
                        dependencies_tol)
end


function merge_calculation_options(;quadratic_matrix_equation_algorithm::Symbol = :schur,
                                    sylvester_algorithm²::Symbol = :doubling,
                                    sylvester_algorithm³::Symbol = :bicgstab,
                                    lyapunov_algorithm::Symbol = :doubling,
                                    tol::Tolerances = Tolerances(),
                                    verbose::Bool = false)
                                    
    return CalculationOptions(quadratic_matrix_equation_algorithm, 
                                sylvester_algorithm², 
                                sylvester_algorithm³, 
                                lyapunov_algorithm, 
                                tol, 
                                verbose)
end

end # dispatch_doctor
