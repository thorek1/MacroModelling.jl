mutable struct krylov_caches{G <: AbstractFloat}
    gmres::GmresWorkspace{G,G,Vector{G}}
    dqgmres::DqgmresWorkspace{G,G,Vector{G}}
    bicgstab::BicgstabWorkspace{G,G,Vector{G}}
end

mutable struct sylvester_caches{G <: AbstractFloat}
    tmp::Matrix{G}
    𝐗::Matrix{G}
    𝐂::Matrix{G}
    krylov_caches::krylov_caches{G}
end

mutable struct higher_order_caches{F <: Real, G <: AbstractFloat}
    tmpkron0::SparseMatrixCSC{F, Int}
    tmpkron1::SparseMatrixCSC{F, Int}
    tmpkron11::SparseMatrixCSC{F, Int}
    tmpkron12::SparseMatrixCSC{F, Int}
    tmpkron2::SparseMatrixCSC{F, Int}
    tmpkron22::SparseMatrixCSC{F, Int}
    tmp_sparse_prealloc1::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc2::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc3::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc4::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc5::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    tmp_sparse_prealloc6::Tuple{Vector{Int}, Vector{Int}, Vector{F}, Vector{Int}, Vector{Int}, Vector{Int}, Vector{F}}
    Ŝ::Matrix{F}
    sylvester_caches::sylvester_caches{G}
end

# Cache for model-constant display names
# Stores precomputed variable and shock names that depend only on model structure
struct name_display_cache
    # Processed variable names (with curly brackets formatted)
    var_axis::Vector
    # Processed shock names (with curly brackets formatted, WITHOUT ₍ₓ₎ suffix)
    exo_axis_plain::Vector
    # Processed shock names (with curly brackets formatted and WITH ₍ₓ₎ suffix)
    exo_axis_with_subscript::Vector
    # Flag indicating if variables contain curly brackets
    var_has_curly::Bool
    # Flag indicating if shocks contain curly brackets
    exo_has_curly::Bool
end

# Cache for model structure information
# Stores precomputed lists and labels that depend only on model structure
struct model_structure_cache
    # SS_and_pars_names: processed variable names + calibration parameters
    SS_and_pars_names::Vector{Symbol}
    # all_variables: sorted union of var, aux, exo_present
    all_variables::Vector{Symbol}
    # NSSS_labels: sorted union of exo_present, var + calibration parameters
    NSSS_labels::Vector{Symbol}
    # aux_indices_in_all_variables: indexin(𝓂.aux, all_variables)
    aux_indices::Vector{Int}
    # processed_all_variables: all_variables with aux names cleaned
    processed_all_variables::Vector{Symbol}
    # full_NSSS with aux cleaned and curly bracket display formatting
    full_NSSS_display::Vector{Union{Symbol, String}}
    # Selector from NSSS order to full steady-state vector (processed variables)
    steady_state_expand_matrix::SparseMatrixCSC{Float64, Int}
    # Selector from custom steady-state order to var + calibrated parameters
    custom_ss_expand_matrix::SparseMatrixCSC{Float64, Int}
    # vars_in_ss_equations: cached for custom steady state mapping
    vars_in_ss_equations::Vector{Symbol}
    # SS_and_pars_names in lead/lag form
    SS_and_pars_names_lead_lag::Vector{Symbol}
    # SS_and_pars_names without exo and lead/lag markers
    SS_and_pars_names_no_exo::Vector{Symbol}
    # Cached indices for SS_and_pars_names_no_exo in lead/lag list
    SS_and_pars_no_exo_idx::Vector{Int}
    # Cached variable indices excluding auxiliary and OBC variables
    vars_idx_excluding_aux_obc::Vector{Int}
    # Cached variable indices excluding OBC variables
    vars_idx_excluding_obc::Vector{Int}
end

# Cache for model-constant computational objects
# Stores precomputed BitVectors, kronecker products, and size information
struct computational_constants_cache
    # BitVector for state selection: [ones(nPast+1), zeros(nExo)]
    s_in_s⁺::BitVector
    # BitVector for state selection: [ones(nPast), zeros(nExo+1)]  
    s_in_s::BitVector
    # Kronecker product: kron(s_in_s⁺, s_in_s⁺)
    kron_s⁺_s⁺::BitVector
    # Kronecker product: kron(s_in_s⁺, s_in_s)
    kron_s⁺_s::BitVector
    # Size for identity matrix of past states (store size, not the matrix itself)
    nPast::Int
    # Additional BitVectors for moments calculations
    e_in_s⁺::BitVector  # [zeros(nPast+1), ones(nExo)]
    v_in_s⁺::BitVector  # [zeros(nPast), 1, zeros(nExo)]
    # Diagonal matrix for state selection
    diag_nVars::ℒ.Diagonal{Float64, Vector{Float64}}
    # Additional kron products for moments and filter calculations
    kron_s_s::BitVector  # kron(s_in_s⁺, s_in_s⁺) - same as kron_s⁺_s⁺ but used with s_in_s naming
    kron_e_e::BitVector  # kron(e_in_s⁺, e_in_s⁺)
    kron_v_v::BitVector  # kron(v_in_s⁺, v_in_s⁺)
    kron_s_e::BitVector  # kron(s_in_s⁺, e_in_s⁺)
    kron_e_s::BitVector  # kron(e_in_s⁺, s_in_s⁺)
    # Sparse index patterns for filter operations (from kron().nzind)
    shockvar_idxs::Vector{Int}  # kron(e_in_s⁺, s_in_s⁺) |> sparse |> .nzind
    shock_idxs::Vector{Int}     # kron(e_in_s⁺, ones) |> sparse |> .nzind
    shock_idxs2::Vector{Int}    # kron(ones, e_in_s⁺) |> sparse |> .nzind
    shock²_idxs::Vector{Int}    # kron(e_in_s⁺, e_in_s⁺) |> sparse |> .nzind
    var_vol²_idxs::Vector{Int}  # kron(s_in_s⁺, s_in_s⁺) |> sparse |> .nzind
end

struct moments_substate_cache
    I_plus_s_s::SparseMatrixCSC{Float64, Int}
    e_es::SparseMatrixCSC{Float64, Int}
    e_ss::SparseMatrixCSC{Float64, Int}
    ss_s::SparseMatrixCSC{Float64, Int}
    s_s::SparseMatrixCSC{Float64, Int}
end

struct moments_dependency_kron_cache
    kron_s_s::BitVector
    kron_s_e::BitVector
    kron_s_v::BitVector
end

mutable struct moments_cache
    kron_states::BitVector
    kron_s_e::BitVector
    I_plus_s_s::SparseMatrixCSC{Float64, Int}
    e4::Vector{Float64}
    e6::Vector{Float64}
    kron_e_v::BitVector
    substate_cache::Dict{Int, moments_substate_cache}
    dependency_kron_cache::Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_cache}
end

struct first_order_index_cache{I, M}
    initialized::Bool
    dyn_index::UnitRange{Int}
    reverse_dynamic_order::Vector{Union{Nothing, Int}}
    comb::Vector{Int}
    future_not_past_and_mixed_in_comb::Vector{Union{Nothing, Int}}
    past_not_future_and_mixed_in_comb::Vector{Union{Nothing, Int}}
    Ir::I
    nabla_zero_cols::UnitRange{Int}
    nabla_minus_cols::UnitRange{Int}
    nabla_e_start::Int
    expand_future::M
    expand_past::M
end

mutable struct caches#{F <: Real, G <: AbstractFloat}
    # Timings information (model structure - constant for a given model)
    timings::Union{Nothing, timings}
    
    # Cache structures
    second_order_caches::higher_order_caches#{F, G}
    third_order_caches::higher_order_caches#{F, G}
    name_display_cache::name_display_cache
    model_structure_cache::model_structure_cache
    computational_constants::computational_constants_cache
    moments_cache::moments_cache
    first_order_index_cache::first_order_index_cache
    custom_steady_state_buffer::Vector{Float64}
end

function First_order_index_cache()
    empty_range = 1:0
    empty_union_vec = Vector{Union{Nothing, Int}}()
    empty_int_vec = Int[]
    empty_matrix = zeros(0,0)
    return first_order_index_cache(false,
                                    empty_range,
                                    empty_union_vec,
                                    empty_int_vec,
                                    empty_union_vec,
                                    empty_union_vec,
                                    ℒ.I(0),
                                    empty_range,
                                    empty_range,
                                    1,
                                    empty_matrix,
                                    empty_matrix)
end

function Moments_cache()
    empty_sparse = spzeros(Float64, 0, 0)
    return moments_cache(BitVector(),
                        BitVector(),
                        empty_sparse,
                        Float64[],
                        Float64[],
                        BitVector(),
                        Dict{Int, moments_substate_cache}(),
                        Dict{Tuple{Vararg{Symbol}}, moments_dependency_kron_cache}())
end


function Krylov_caches(;S::Type = Float64)
    krylov_caches(  GmresWorkspace(0,0,Vector{S}),
                    DqgmresWorkspace(0,0,Vector{S}),
                    BicgstabWorkspace(0,0,Vector{S}))
end

function Sylvester_caches(;S::Type = Float64)
    sylvester_caches(   zeros(S,0,0),
                        zeros(S,0,0),
                        zeros(S,0,0),
                        Krylov_caches(S = S))
end

function Higher_order_caches(;T::Type = Float64, S::Type = Float64)
    higher_order_caches(spzeros(T,0,0),
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
                        Sylvester_caches(S = S))
end

function Caches(;T::Type = Float64, S::Type = Float64)
    caches( nothing,  # timings will be set later from model
            Higher_order_caches(T = T, S = S),
            Higher_order_caches(T = T, S = S),
            name_display_cache([], [], [], false, false),
            model_structure_cache(Symbol[], Symbol[], Symbol[], Int[], Symbol[],
                                Union{Symbol,String}[], spzeros(Float64, 0, 0), spzeros(Float64, 0, 0),
                                Symbol[], Symbol[], Symbol[], Int[], Int[], Int[]),
            computational_constants_cache(BitVector(), BitVector(), BitVector(), BitVector(), 0, 
                                         BitVector(), BitVector(), ℒ.Diagonal(Float64[]),
                                         BitVector(), BitVector(), BitVector(), BitVector(), BitVector(),
                                         Int[], Int[], Int[], Int[], Int[]),
            Moments_cache(),
            First_order_index_cache(),
            Float64[])
end

# Initialize timings in the cache (should be called once after model creation)
function set_timings!(cache::caches, T::timings)
    cache.timings = T
    return cache
end

function ensure_name_display_cache!(𝓂)
    cache = 𝓂.caches
    ndc = cache.name_display_cache
    # Use timings from cache if available, otherwise from model
    T = isnothing(cache.timings) ? 𝓂.timings : cache.timings
    
    if isempty(ndc.var_axis)
        var_has_curly = any(x -> contains(string(x), "◖"), T.var)
        if var_has_curly
            var_decomposed = decompose_name.(T.var)
            var_axis = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in var_decomposed]
        else
            var_axis = T.var
        end

        exo_has_curly = any(x -> contains(string(x), "◖"), T.exo)
        if exo_has_curly
            exo_decomposed = decompose_name.(T.exo)
            exo_axis_plain = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in exo_decomposed]
            exo_axis_with_subscript = exo_axis_plain .* "₍ₓ₎"
        else
            exo_axis_plain = T.exo
            exo_axis_with_subscript = map(x -> Symbol(string(x) * "₍ₓ₎"), T.exo)
        end

        cache.name_display_cache = name_display_cache(
            var_axis,
            exo_axis_plain,
            exo_axis_with_subscript,
            var_has_curly,
            exo_has_curly,
        )
    end

    return cache.name_display_cache
end

function ensure_computational_constants_cache!(𝓂)
    cache = 𝓂.caches
    cc = cache.computational_constants
    if isempty(cc.s_in_s⁺)
        # Use timings from cache if available, otherwise from model
        T = isnothing(cache.timings) ? 𝓂.timings : cache.timings
        nᵉ = T.nExo
        nˢ = T.nPast_not_future_and_mixed

        s_in_s⁺ = BitVector(vcat(ones(Bool, nˢ + 1), zeros(Bool, nᵉ)))
        s_in_s = BitVector(vcat(ones(Bool, nˢ), zeros(Bool, nᵉ + 1)))

        kron_s⁺_s⁺ = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_s⁺_s = ℒ.kron(s_in_s⁺, s_in_s)

        e_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ + 1), ones(Bool, nᵉ)))
        v_in_s⁺ = BitVector(vcat(zeros(Bool, nˢ), 1, zeros(Bool, nᵉ)))

        diag_nVars = ℒ.diagm(ones(T.nVars))

        kron_s_s = ℒ.kron(s_in_s⁺, s_in_s⁺)
        kron_e_e = ℒ.kron(e_in_s⁺, e_in_s⁺)
        kron_v_v = ℒ.kron(v_in_s⁺, v_in_s⁺)
        kron_s_e = ℒ.kron(s_in_s⁺, e_in_s⁺)
        kron_e_s = ℒ.kron(e_in_s⁺, s_in_s⁺)

        # Compute sparse index patterns for filter operations
        shockvar_idxs = sparse(ℒ.kron(e_in_s⁺, s_in_s⁺)).nzind
        shock_idxs = sparse(ℒ.kron(e_in_s⁺, zero(e_in_s⁺) .+ 1)).nzind
        shock_idxs2 = sparse(ℒ.kron(zero(e_in_s⁺) .+ 1, e_in_s⁺)).nzind
        shock²_idxs = sparse(ℒ.kron(e_in_s⁺, e_in_s⁺)).nzind
        var_vol²_idxs = sparse(ℒ.kron(s_in_s⁺, s_in_s⁺)).nzind

        cache.computational_constants = computational_constants_cache(
            s_in_s⁺,
            s_in_s,
            kron_s⁺_s⁺,
            kron_s⁺_s,
            nˢ,
            e_in_s⁺,
            v_in_s⁺,
            diag_nVars,
            kron_s_s,
            kron_e_e,
            kron_v_v,
            kron_s_e,
            kron_e_s,
            shockvar_idxs,
            shock_idxs,
            shock_idxs2,
            shock²_idxs,
            var_vol²_idxs,
        )
    end

    return cache.computational_constants
end

function build_first_order_index_cache(T, I_nVars)
    dyn_index = T.nPresent_only + 1:T.nVars

    reverse_dynamic_order = indexin([T.past_not_future_idx; T.future_not_past_and_mixed_idx], T.present_but_not_only_idx)

    comb = union(T.future_not_past_and_mixed_idx, T.past_not_future_idx)
    sort!(comb)

    future_not_past_and_mixed_in_comb = indexin(T.future_not_past_and_mixed_idx, comb)
    past_not_future_and_mixed_in_comb = indexin(T.past_not_future_and_mixed_idx, comb)

    Ir = ℒ.I(length(comb))

    nabla_zero_cols = (T.nFuture_not_past_and_mixed + 1):(T.nFuture_not_past_and_mixed + T.nVars)
    nabla_minus_cols = (T.nFuture_not_past_and_mixed + T.nVars + 1):(T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed)
    nabla_e_start = T.nFuture_not_past_and_mixed + T.nVars + T.nPast_not_future_and_mixed + 1

    expand_future = I_nVars[T.future_not_past_and_mixed_idx,:]
    expand_past = I_nVars[T.past_not_future_and_mixed_idx,:]

    return first_order_index_cache(true,
                                    dyn_index,
                                    reverse_dynamic_order,
                                    comb,
                                    future_not_past_and_mixed_in_comb,
                                    past_not_future_and_mixed_in_comb,
                                    Ir,
                                    nabla_zero_cols,
                                    nabla_minus_cols,
                                    nabla_e_start,
                                    expand_future,
                                    expand_past)
end

function ensure_first_order_index_cache!(𝓂)
    cache = 𝓂.caches
    if !cache.first_order_index_cache.initialized
        cc = ensure_computational_constants_cache!(𝓂)
        # Use timings from cache if available, otherwise from model
        T = isnothing(cache.timings) ? 𝓂.timings : cache.timings
        cache.first_order_index_cache = build_first_order_index_cache(T, cc.diag_nVars)
    end
    return cache.first_order_index_cache
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
    cache = 𝓂.caches.model_structure_cache
    if isempty(cache.SS_and_pars_names)
        SS_and_pars_names = vcat(
            Symbol.(replace.(string.(sort(union(𝓂.var, 𝓂.exo_past, 𝓂.exo_future))),
                    r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")),
            𝓂.calibration_equations_parameters,
        )

        all_variables = sort(union(𝓂.var, 𝓂.aux, 𝓂.exo_present))

        NSSS_labels = [sort(union(𝓂.exo_present, 𝓂.var))..., 𝓂.calibration_equations_parameters...]

        aux_indices = Int.(indexin(𝓂.aux, all_variables))
        processed_all_variables = copy(all_variables)
        processed_all_variables[aux_indices] = map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.aux)

        full_NSSS = copy(processed_all_variables)
        if any(x -> contains(string(x), "◖"), full_NSSS)
            full_NSSS_decomposed = decompose_name.(full_NSSS)
            full_NSSS = [length(a) > 1 ? string(a[1]) * "{" * join(a[2],"}{") * "}" * (a[end] isa Symbol ? string(a[end]) : "") : string(a[1]) for a in full_NSSS_decomposed]
        end
        full_NSSS_display = Vector{Union{Symbol, String}}(full_NSSS)

        steady_state_expand_matrix = create_selector_matrix(processed_all_variables, NSSS_labels)

        vars_in_ss_equations = sort(collect(setdiff(reduce(union, get_symbols.(𝓂.ss_aux_equations)), union(𝓂.parameters_in_equations, 𝓂.➕_vars))))
        extended_SS_and_pars = vcat(map(x -> Symbol(replace(string(x), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.var), 𝓂.calibration_equations_parameters)
        custom_ss_expand_matrix = create_selector_matrix(extended_SS_and_pars, vcat(vars_in_ss_equations, 𝓂.calibration_equations_parameters))

        SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var, 𝓂.exo_past, 𝓂.exo_future)))), 𝓂.calibration_equations_parameters)
        SS_and_pars_names_no_exo = vcat(Symbol.(replace.(string.(sort(setdiff(𝓂.var, 𝓂.exo_past, 𝓂.exo_future))), r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => "")), 𝓂.calibration_equations_parameters)
        SS_and_pars_no_exo_idx = Int.(indexin(unique(SS_and_pars_names_no_exo), SS_and_pars_names_lead_lag))

        vars_non_obc = 𝓂.var[.!contains.(string.(𝓂.var), "ᵒᵇᶜ")]
        vars_idx_excluding_aux_obc = Int.(indexin(setdiff(vars_non_obc, union(𝓂.aux, 𝓂.exo_present)), all_variables))
        vars_idx_excluding_obc = Int.(indexin(vars_non_obc, all_variables))

        𝓂.caches.model_structure_cache = model_structure_cache(
            SS_and_pars_names,
            all_variables,
            NSSS_labels,
            aux_indices,
            processed_all_variables,
            full_NSSS_display,
            steady_state_expand_matrix,
            custom_ss_expand_matrix,
            vars_in_ss_equations,
            SS_and_pars_names_lead_lag,
            SS_and_pars_names_no_exo,
            SS_and_pars_no_exo_idx,
            vars_idx_excluding_aux_obc,
            vars_idx_excluding_obc,
        )
    end

    return 𝓂.caches.model_structure_cache
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
    cache = 𝓂.caches
    mc = cache.moments_cache
    cc = ensure_computational_constants_cache!(𝓂)
    # Use timings from cache if available, otherwise from model
    T = isnothing(cache.timings) ? 𝓂.timings : cache.timings
    
    if isempty(mc.kron_states)
        mc.kron_states = ℒ.kron(cc.s_in_s, cc.s_in_s)
    end
    if isempty(mc.kron_s_e)
        mc.kron_s_e = ℒ.kron(cc.s_in_s, cc.e_in_s⁺)
    end
    if size(mc.I_plus_s_s, 1) == 0
        nˢ = T.nPast_not_future_and_mixed
        mc.I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
    end
    if isempty(mc.e4)
        mc.e4 = compute_e4(T.nExo)
    end
    if isempty(mc.e6)
        mc.e6 = compute_e6(T.nExo)
    end
    if isempty(mc.kron_e_v)
        mc.kron_e_v = ℒ.kron(cc.e_in_s⁺, cc.v_in_s⁺)
    end
    return mc
end

function ensure_moments_substate_cache!(𝓂, nˢ::Int)
    cache = 𝓂.caches
    mc = cache.moments_cache
    if !haskey(mc.substate_cache, nˢ)
        # Use timings from cache if available, otherwise from model
        T = isnothing(cache.timings) ? 𝓂.timings : cache.timings
        nᵉ = T.nExo
        I_plus_s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2) + ℒ.I)
        e_es = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nᵉ * nˢ)), nˢ * nᵉ^2, nˢ * nᵉ^2))
        e_ss = sparse(reshape(ℒ.kron(vec(ℒ.I(nᵉ)), ℒ.I(nˢ^2)), nᵉ * nˢ^2, nᵉ * nˢ^2))
        ss_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ^2)), ℒ.I(nˢ)), nˢ^3, nˢ^3))
        s_s = sparse(reshape(ℒ.kron(vec(ℒ.I(nˢ)), ℒ.I(nˢ)), nˢ^2, nˢ^2))
        mc.substate_cache[nˢ] = moments_substate_cache(I_plus_s_s, e_es, e_ss, ss_s, s_s)
    end
    return mc.substate_cache[nˢ]
end

function ensure_moments_dependency_kron_cache!(𝓂, dependencies::Vector{Symbol}, s_in_s⁺::BitVector)
    cache = 𝓂.caches.moments_cache
    key = Tuple(dependencies)
    if !haskey(cache.dependency_kron_cache, key)
        cc = ensure_computational_constants_cache!(𝓂)
        cache.dependency_kron_cache[key] = moments_dependency_kron_cache(
            ℒ.kron(s_in_s⁺, s_in_s⁺),
            ℒ.kron(s_in_s⁺, cc.e_in_s⁺),
            ℒ.kron(s_in_s⁺, cc.v_in_s⁺),
        )
    end
    return cache.dependency_kron_cache[key]
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
