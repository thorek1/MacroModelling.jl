# commonly used DocStrings
const MODEL® = "`𝓂`: object created by [`@model`](@ref) and [`@parameters`](@ref)."
const PARAMETER_VALUES® = "`parameters` [Type: `Vector`]: Parameter values in alphabetical order (sorted by parameter name)."
const PARAMETERS® = "`parameters` [Default: `nothing`]: If `nothing` is provided, the solution is calculated for the parameters defined previously. Acceptable inputs are a `Vector` of parameter values, a `Vector` or `Tuple` of `Pair`s of the parameter `Symbol` or `String` and value. If the new parameter values differ from the previously defined the solution will be recalculated."
const VARIABLES® = "`variables` [Default: `:all_excluding_obc`]: variables for which to show the results. Inputs can be a variable name passed on as either a `Symbol` or `String` (e.g. `:y` or \"y\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxilliary_and_obc` contains all shocks less those related to auxilliary variables and related to occasionally binding constraints (obc). `:all_excluding_obc` contains all shocks less those related to auxilliary variables. `:all` will contain all variables."
const SHOCKS® = "`shocks` [Default: `:all_excluding_obc`]: shocks for which to calculate the IRFs. Inputs can be a shock name passed on as either a `Symbol` or `String` (e.g. `:y`, or \"y\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:simulate` triggers random draws of all shocks (excluding occasionally binding constraints (obc) related shocks). `:all_excluding_obc` will contain all shocks but not the obc related ones. `:all` will contain also the obc related shocks. A series of shocks can be passed on using either a `Matrix{Float64}`, or a `KeyedArray{Float64}` as input with shocks (`Symbol` or `String`) in rows and periods in columns. The period of the simulation will correspond to the length of the input in the period dimension + the number of periods defined in `periods`. If the series of shocks is input as a `KeyedArray{Float64}` make sure to name the rows with valid shock names of type `Symbol`. Any shocks not part of the model will trigger a warning. `:none` in combination with an `initial_state` can be used for deterministic simulations."
const DERIVATIVES® = "`derivatives` [Default: `true`, Type: `Bool`]: calculate derivatives with respect to the parameters."
const PERIODS® = "`periods` [Default: `40`, Type: `Int`]: number of periods for which to calculate the output. In case a matrix of shocks was provided, periods defines how many periods after the series of shocks the output continues."
const NEGATIVE_SHOCK® = "`negative_shock` [Default: `false`, Type: `Bool`]: calculate IRFs for a negative shock."
const GENERALISED_IRF® = "`generalised_irf` [Default: `false`, Type: `Bool`]: calculate generalised IRFs. Relevant for nonlinear (higher order perturbation) solutions only. Reference steady state for deviations is the stochastic steady state. `initial_state` has no effect on generalised IRFs. Occasionally binding constraint are not respected for generalised IRF."
const ALGORITHM® = "`algorithm` [Default: `:first_order`, Type: `Symbol`]: algorithm to solve for the dynamics of the model. Available algorithms: `:first_order`, `:second_order`, `:pruned_second_order`, `:third_order`, `:pruned_third_order`"
const FILTER® = "`filter` [Default: `:kalman`, Type: `Symbol`]: filter used to compute the variables, and shocks given the data, model, and parameters. The Kalman filter only works for linear problems, whereas the inversion filter (`:inversion`) works for linear and nonlinear models. If a nonlinear solution algorithm is selected, the inversion filter is used."
const LEVELS® = "`levels` [Default: `false`, Type: `Bool`]: return levels or absolute deviations from the relevant steady state corresponding to the solution algorithm (e.g. stochastic steady state for higher order solution algorithms)."
const CONDITIONS® = "`conditions` [Type: `Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}}`]: conditions for which to find the corresponding shocks. The input can have multiple formats, but for all types of entries the first dimension corresponds to variables and the second dimension to the number of periods. The conditions can be specified using a matrix of type `Matrix{Union{Nothing,Float64}}`. In this case the conditions are matrix elements of type `Float64` and all remaining (free) entries are `nothing`. You can also use a `SparseMatrixCSC{Float64}` as input. In this case only non-zero elements are taken as conditions. Note that you cannot condition variables to be zero using a `SparseMatrixCSC{Float64}` as input (use other input formats to do so). Another possibility to input conditions is by using a `KeyedArray`. You can use a `KeyedArray{Union{Nothing,Float64}}` where, similar to `Matrix{Union{Nothing,Float64}}`, all entries of type `Float64` are recognised as conditions and all other entries have to be `nothing`. Furthermore, you can specify in the primary axis a subset of variables (of type `Symbol` or `String`) for which you specify conditions and all other variables are considered free. The same goes for the case when you use `KeyedArray{Float64}}` as input, whereas in this case the conditions for the specified variables bind for all periods specified in the `KeyedArray`, because there are no `nothing` entries permitted with this type."
const SHOCK_CONDITIONS® = "`shocks` [Default: `nothing`, Type: `Union{Matrix{Union{Nothing,Float64}}, SparseMatrixCSC{Float64}, KeyedArray{Union{Nothing,Float64}}, KeyedArray{Float64}, Nothing}`]: known values of shocks. This argument allows the user to include certain shock values. By entering restrictions on the shocks in this way the problem to match the conditions on endogenous variables is restricted to the remaining free shocks in the repective period. The input can have multiple formats, but for all types of entries the first dimension corresponds to shocks and the second dimension to the number of periods.  `shocks` can be specified using a matrix of type `Matrix{Union{Nothing,Float64}}`. In this case the shocks are matrix elements of type `Float64` and all remaining (free) entries are `nothing`. You can also use a `SparseMatrixCSC{Float64}` as input. In this case only non-zero elements are taken as certain shock values. Note that you cannot condition shocks to be zero using a `SparseMatrixCSC{Float64}` as input (use other input formats to do so). Another possibility to input known shocks is by using a `KeyedArray`. You can use a `KeyedArray{Union{Nothing,Float64}}` where, similar to `Matrix{Union{Nothing,Float64}}`, all entries of type `Float64` are recognised as known shocks and all other entries have to be `nothing`. Furthermore, you can specify in the primary axis a subset of shocks (of type `Symbol` or `String`) for which you specify values and all other shocks are considered free. The same goes for the case when you use `KeyedArray{Float64}}` as input, whereas in this case the values for the specified shocks bind for all periods specified in the `KeyedArray`, because there are no `nothing` entries permitted with this type."
const PARAMETER_DERIVATIVES® = "`parameter_derivatives` [Default: :all]: parameters for which to calculate partial derivatives. Inputs can be a parameter name passed on as either a `Symbol` or `String` (e.g. `:alpha`, or \"alpha\"), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. `:all` will include all parameters."
const DATA® = "`data` [Type: `KeyedArray`]: data matrix with variables (`String` or `Symbol`) in rows and time in columns"
const SMOOTH® = "`smooth` [Default: `true`, Type: `Bool`]: whether to return smoothed (`true`) or filtered (`false`) shocks/variables. Only works for the Kalman filter. The inversion filter only returns filtered shocks/variables."
const DATA_IN_LEVELS® = "`data_in_levels` [Default: `true`, Type: `Bool`]: indicator whether the data is provided in levels. If `true` the input to the data argument will have the non stochastic steady state substracted."
const LYAPUNOV® = "`lyapunov_algorithm` [Default: `:doubling`, Type: `Symbol`]: algorithm to solve Lyapunov equation (`A * X * A' + C = X`). Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:gmres`"
const SYLVESTER® = "`sylvester_algorithm` [Default: function of size of problem, with smaller problems: `:doubling`, and larger problems: `:bicgstab`, Type: `Union{Symbol,Vector{Symbol},Tuple{Symbol,Vararg{Symbol}}}`]: algorithm to solve Sylvester equation (`A * X * B + C = X`). Available algorithms: `:doubling`, `:bartels_stewart`, `:bicgstab`, `:dqgmres`, `:gmres`. Input argument can be up to two elements in a `Vector` or `Tuple`. The first (second) element corresponds to the second (third) order perturbation solutions' Sylvester equation. If only one element is provided it corresponds to the second order perturbation solutions' Sylvester equation."
const QME® = "`quadratic_matrix_equation_algorithm` [Default: `:schur`, Type: `Symbol`]: algorithm to solve quadratic matrix equation (`A * X ^ 2 + B * X + C = 0`). Available algorithms: `:schur`, `:doubling`"
const VERBOSE® = "`verbose` [Default: `false`, Type: `Bool`]: print information about results of the different solvers used to solve the model (non stochastic steady state solver, Sylvester equations, Lyapunov equation, and quadratic matrix equation)."
const TOLERANCES® = "`tol` [Default: `Tolerances()`, Type: `Tolerances`]: define various tolerances for the algorithm used to solve the model. See documentation of [`Tolerances`](@ref) for more details: `?Tolerances`"
const PLOT_ATTRIBUTES® = "`plot_attributes` [Default: `Dict()`, Type: `Dict`]: pass on plot attributes for the top-level plot (see https://docs.juliaplots.org/latest/generated/attributes_plot/). E.g. Dict(:plot_titlefontcolor => :red)."
const PLOTS_PER_PAGE® = "`plots_per_page` [Default: `9`, Type: `Int`]: how many plots to show per page"
const SAVE_PLOTS_PATH® = "`save_plots_path` [Default: `pwd()`, Type: `String`]: path where to save plots"
const SAVE_PLOTS_FORMATH® = "`save_plots_format` [Default: `:pdf`, Type: `Symbol`]: output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats."
const SAVE_PLOTS® = "`save_plots` [Default: `false`, Type: `Bool`]: switch to save plots using path and extension from `save_plots_path` and `save_plots_format`. Separate files per shocks and variables depending on number of variables and `plots_per_page`"
const SHOW_PLOTS® = "`show_plots` [Default: `true`, Type: `Bool`]: show plots. Separate plots per shocks and varibles depending on number of variables and `plots_per_page`."
const EXTRA_LEGEND_SPACE® = "`extra_legend_space` [Default: `0.0`, Type: `Float64`]: space between the plots and the legend (useful if the plots overlap the legend)."
const MAX_ELEMENTS_PER_LEGENDS_ROW® = "`max_elements_per_legend_row` [Default: `4`, Type: `Int`]: maximum number of elements per legend row. In other words, number of columns in legend."
const WARMUP_ITERATIONS® = "`warmup_iterations` [Default: `0`, Type: `Int`]: periods added before the first observation for which shocks are computed such that the first observation is matched. A larger value alleviates the problem that the initial value is the relevant steady state."
const SHOCK_SIZE® = "`shock_size` [Default: `1`, Type: `Real`]: affects the size of shocks as long as they are not set to `:none`."
const IGNORE_OBC® = "`ignore_obc` [Default: `false`, Type: `Bool`]: solve the model ignoring the occasionally binding constraints."
const INITIAL_STATE® = "`initial_state` [Default: `[0.0]`, Type: `Union{Vector{Vector{Float64}},Vector{Float64}}`]: The initial state defines the starting point for the model. In the case of pruned solution algorithms the initial state can be given as multiple state vectors (`Vector{Vector{Float64}}`). In this case the initial state must be given in deviations from the non-stochastic steady state. In all other cases the initial state must be given in levels. If a pruned solution algorithm is selected and `initial_state` is a `Vector{Float64}` then it impacts the first order initial state vector only. The state includes all variables as well as exogenous variables in leads or lags if present. `get_irf(𝓂, shocks = :none, variables = :all, periods = 1)` returns a `KeyedArray` with all variables."
const INITIAL_STATE®1 = "`initial_state` [Default: `[0.0]`, Type: `Vector{Float64}`]: The initial state defines the starting point for the model (in levels, not deviations). The state includes all variables as well as exogenous variables in leads or lags if present. `get_irf(𝓂, shocks = :none, variables = :all, periods = 1)` returns a `KeyedArray` with all variables."