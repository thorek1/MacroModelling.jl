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
Function to manually define tolerances for the solvers of various problems: non stochastic steady state solver (NSSS), Sylvester equations, Lyapunov equation, and quadratic matrix equation (qme).

# Keyword Arguments
- `NSSS_acceptance_tol` [Default: `1e-12`, Type: `AbstractFloat`]: Acceptance tolerance for non stochastic steady state solver.
- `NSSS_xtol` [Default: `1e-12`, Type: `AbstractFloat`]: Absolute tolerance for solver steps for non stochastic steady state solver.
- `NSSS_ftol` [Default: `1e-14`, Type: `AbstractFloat`]: Absolute tolerance for solver function values for non stochastic steady state solver.
- `NSSS_rel_xtol` [Default: `eps()`, Type: `AbstractFloat`]: Relative tolerance for solver steps for non stochastic steady state solver.

- `qme_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for quadratic matrix equation solver.
- `qme_acceptance_tol` [Default: `1e-8`, Type: `AbstractFloat`]: Acceptance tolerance for quadratic matrix equation solver.

- `sylvester_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for Sylvester equation solver.
- `sylvester_acceptance_tol` [Default: `1e-10`, Type: `AbstractFloat`]: Acceptance tolerance for Sylvester equation solver.

- `lyapunov_tol` [Default: `1e-14`, Type: `AbstractFloat`]: Tolerance for Lyapunov equation solver.
- `lyapunov_acceptance_tol` [Default: `1e-12`, Type: `AbstractFloat`]: Acceptance tolerance for Lyapunov equation solver.

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

const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 10, 
                                :guidefont => 8, 
                                :legendfontsize => 8, 
                                :tickfontsize => 8,
                                :framestyle => :semi)