struct CalculationOptions
    tol::AbstractFloat

    quadratic_matrix_equation_algorithm::Symbol
    qme_tol::AbstractFloat
    qme_acceptance_tol::AbstractFloat

    sylvester_algorithm²::Symbol
    sylvester_algorithm³::Symbol
    sylvester_tol::AbstractFloat
    sylvester_acceptance_tol::AbstractFloat

    lyapunov_algorithm::Symbol
    lyapunov_tol::AbstractFloat
    lyapunov_acceptance_tol::AbstractFloat

    verbose::Bool
end

function merge_calculation_options(;tol::AbstractFloat = 1e-12,

                                    quadratic_matrix_equation_algorithm::Symbol = :schur,
                                    qme_tol::AbstractFloat = 1e-14,
                                    qme_acceptance_tol::AbstractFloat = 1e-8,

                                    sylvester_algorithm²::Symbol = :doubling,
                                    sylvester_algorithm³::Symbol = :bicgstab,
                                    sylvester_tol::AbstractFloat = 1e-14,
                                    sylvester_acceptance_tol::AbstractFloat = 1e-10,

                                    lyapunov_algorithm::Symbol = :doubling,
                                    lyapunov_tol::AbstractFloat = 1e-14,
                                    lyapunov_acceptance_tol::AbstractFloat = 1e-12,

                                    verbose::Bool = false)
                                    
    return CalculationOptions(tol, 

                                quadratic_matrix_equation_algorithm, 
                                qme_tol, 
                                qme_acceptance_tol, 

                                sylvester_algorithm², 
                                sylvester_algorithm³, 
                                sylvester_tol, 
                                sylvester_acceptance_tol, 

                                lyapunov_algorithm, 
                                lyapunov_tol, 
                                lyapunov_acceptance_tol, 

                                verbose)
end


const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 10, 
                                :guidefont => 8, 
                                :legendfontsize => 8, 
                                :tickfontsize => 8,
                                :framestyle => :semi)