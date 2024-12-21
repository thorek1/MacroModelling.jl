struct CalculationOptions
    tol::AbstractFloat
    qme_tol::AbstractFloat
    qme_acceptance_tol::AbstractFloat
    sylvester_tol::AbstractFloat
    sylvester_acceptance_tol::AbstractFloat
    verbose::Bool
    quadratic_matrix_equation_algorithm::Symbol
    sylvester_algorithm²::Symbol
    sylvester_algorithm³::Symbol
    lyapunov_algorithm::Symbol
end

function merge_calculation_options(;tol::AbstractFloat=1e-12,
                                    qme_tol::AbstractFloat=1e-14,
                                    qme_acceptance_tol::AbstractFloat=1e-8,
                                    sylvester_tol::AbstractFloat=1e-14,
                                    sylvester_acceptance_tol::AbstractFloat=1e-10,
                                    verbose::Bool=false,
                                    quadratic_matrix_equation_algorithm::Symbol=:schur,
                                    sylvester_algorithm²::Symbol=:doubling,
                                    sylvester_algorithm³::Symbol=:bicgstab,
                                    lyapunov_algorithm::Symbol=:doubling)
                                    
    return CalculationOptions(tol, qme_tol, qme_acceptance_tol, sylvester_tol, sylvester_acceptance_tol, verbose, quadratic_matrix_equation_algorithm, sylvester_algorithm², sylvester_algorithm³, lyapunov_algorithm)
end


const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 10, 
                                :guidefont => 8, 
                                :legendfontsize => 8, 
                                :tickfontsize => 8,
                                :framestyle => :semi)