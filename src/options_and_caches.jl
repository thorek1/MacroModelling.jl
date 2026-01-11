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
    caches( Higher_order_caches(T = T, S = S),
            Higher_order_caches(T = T, S = S),
            name_display_cache([], [], [], false, false),
            model_structure_cache(Symbol[], Symbol[], Symbol[], Int[], Symbol[],
                                Union{Symbol,String}[], spzeros(Float64, 0, 0), spzeros(Float64, 0, 0),
                                Symbol[], Symbol[], Symbol[], Int[], Int[], Int[]),
            computational_constants_cache(BitVector(), BitVector(), BitVector(), BitVector(), 0, 
                                         BitVector(), BitVector(), ℒ.Diagonal(Float64[]),
                                         BitVector(), BitVector(), BitVector(), BitVector(), BitVector()),
            Moments_cache(),
            First_order_index_cache(),
            Float64[])
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
