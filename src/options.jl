struct CalculationOptions
    tol::AbstractFloat
    verbose::Bool
    quadratic_matrix_equation_algorithm::Symbol
    sylvester_algorithm::Symbol
    lyapunov_algorithm::Symbol
end

function merge_calculation_options(;tol::AbstractFloat=eps(),
                                    verbose::Bool=false,
                                    quadratic_matrix_equation_algorithm::Symbol=:schur,
                                    sylvester_algorithm::Symbol=:doubling,
                                    lyapunov_algorithm::Symbol=:doubling)
                                    
    return CalculationOptions(tol, verbose, quadratic_matrix_equation_algorithm, sylvester_algorithm, lyapunov_algorithm)
end


const default_plot_attributes = Dict(:size=>(700,500),
                                :plot_titlefont => 10, 
                                :titlefont => 10, 
                                :guidefont => 8, 
                                :legendfontsize => 8, 
                                :tickfontsize => 8,
                                :framestyle => :semi)