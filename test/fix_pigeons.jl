using Revise
using MacroModelling
import Turing
import ADTypes
import Pigeons
import Zygote
import Turing: NUTS, sample, logpdf
import Optim, LineSearches
using Random, CSV, DataFrames, MCMCChains, AxisKeys
import DynamicPPL

include("../models/FS2000.jl")

# load data
dat = CSV.read("test/data/FS2000_data.csv", DataFrame)
data = KeyedArray(Array(dat)',Variable = Symbol.("log_".*names(dat)),Time = 1:size(dat)[1])
data = log.(data)

# declare observables
observables = sort(Symbol.("log_".*names(dat)))

# subset observables in data
data = data(observables,:)

dists = [
    Beta(0.356, 0.02, μσ = true),           # alp
    Beta(0.993, 0.002, μσ = true),          # bet
    Normal(0.0085, 0.003),                  # gam
    Normal(1.0002, 0.007),                  # mst
    Beta(0.129, 0.223, μσ = true),          # rho
    Beta(0.65, 0.05, μσ = true),            # psi
    Beta(0.01, 0.005, μσ = true),           # del
    InverseGamma(0.035449, Inf, μσ = true), # z_e_a
    InverseGamma(0.008862, Inf, μσ = true)  # z_e_m
]

Turing.@model function FS2000_loglikelihood_function(data, m, filter)
    all_params ~ Turing.arraydist(dists)

    if DynamicPPL.leafcontext(__context__) !== DynamicPPL.PriorContext() 
        Turing.@addlogprob! get_loglikelihood(m, data, all_params, filter = filter)
    end
end

FS2000_lp = Pigeons.TuringLogPotential(FS2000_loglikelihood_function(data, FS2000, :inversion))

init_params = FS2000.parameter_values

const FS2000_LP = typeof(FS2000_lp)

function Pigeons.initialization(target::FS2000_LP, rng::AbstractRNG, _::Int64)
    result = DynamicPPL.VarInfo(rng, target.model, DynamicPPL.SampleFromPrior(), DynamicPPL.PriorContext())
    # DynamicPPL.link!!(result, DynamicPPL.SampleFromPrior(), target.model)
    
    result = DynamicPPL.initialize_parameters!!(result, init_params, target.model)

    return result
end

pt = Pigeons.pigeons(target = FS2000_lp, n_rounds = 0, n_chains = 1)

pt = Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 1,
            multithreaded = false)

# FS2000.SS_solve_func
# FS2000.parameter_values
FS2000.NSSS_solver_cache |> length#[end][end]

@profview Pigeons.pigeons(target = FS2000_lp,
            record = [Pigeons.traces; Pigeons.round_trip; Pigeons.record_default()],
            n_chains = 1,
            n_rounds = 1,
            multithreaded = false)


using Revise
using MacroModelling
using BenchmarkTools
import LinearAlgebra as ℒ
import MacroModelling: Tolerances, block_solver, clear_solution_caches!, reverse_diff_friendly_push!, ss_solve_block, solver_parameters, levenberg_marquardt, solve_ss, function_and_jacobian, transform, undo_transform, has_nonfinite, choose_matrix_format,SparseMatrixCSC, minmax!
import DataStructures: CircularBuffer
import TimerOutputs: TimerOutput, @timeit, @timeit
import LinearSolve as 𝒮

include("../models/FS2000.jl")

clear_solution_caches!(FS2000, :first_order)
parms =  [0.33375504395436606, 0.9913930046304643, 0.005818806080402188, 1.0157358846332918, 0.020273285639885882, 0.6822354134806288, 0.0055759285328469965, 0.016157068974959693, 0.0029673209747231033]
SS(FS2000, verbose = true, derivatives = false, parameters = parms)


FS2000.SS_solve_func
FS2000.NSSS_solver_cache
FS2000 = nothing


A = [0.8188700139804173 -4.026667454790379 0.0 0.0 0.0; -0.062225840500729546 1.0 -2.131831062053371e10 -0.6619380213209612 -2.567121434143265e-6; -0.06167490239553266 -2.3546562664483833e-9 -2217.2014286559124 1.3621367407582286e-14 -2.6699232470279754e-13; 8.272624009065536e-6 -4.067935727808727e-5 5.334769875601394e-18 4.705410429871372e-6 -0.00013843348571258455; 0.01183049932547297 -3.103043787902102e10 -1.320525482705414e21 3.4963886803269994e-7 79658.9022139248]
b = [-4.313053119631981, 2.4443267685589394e-10, -0.2483444074196767, 29.42006606575791, 2.540989509981241e10]
[4.026672432776696, 1.8899932801504653, -7.647443340128324e-11, 5.763839102195619, -212521.43125382307]
A \ b
[-4.301368566921438e-65, -4.36732688421233e-65, 4.7633706639682065e-70, -1.6663128422229728e-50, -5.020837394196622e-34]

A = [0.0 -0.010950987654156773 -0.007762755388887709 0.0 0.007762755388887711; 0.0 1.0 -0.3217620148778505 0.0 -0.6649523266972911; 2.6483308628798294 2.6483308628798294 0.0 0.189306815382423 -3.456055461023388; -0.035837070350234784 0.0 -0.04100621191691378 0.12286319760464957 -0.0818569856877358; 5.282429995876679 5.282429995876679 0.0 0.0 0.0]
b =[0.05784782566879296, 0.07018056057844291, 14.989622389082381, -0.4597094250337016, 26.888330776704404]
# sol_cache = FS2000.ss_solve_blocks_in_place[1].ss_problem.chol_buffer
FS2000.ss_solve_blocks_in_place[1].ss_problem.jac_buffer
sol_cache = FS2000.ss_solve_blocks_in_place[1].ss_problem.lu_buffer
sol_cache.A = A
sol_cache.b = b
𝒮.solve!(sol_cache)
sol_cache.u .= 0
#)
A = [  0.0      -0.517004  -0.366485  0.0         0.366485
  0.0       1.0       -0.321762  0.0        -0.664952
  0.7688    0.7688     0.0       0.0         0.0
 -1.69189   0.0       -0.281754  0.844194   -0.56244
 -7.13928  -7.13928    0.0       1.30073   -25.0407]
 b = [  0.39747280113043293
  0.01021401419703094
 -0.42468244463329174
  0.651712167492304
 -4.488677117094992]
A \ b
# parameters_and_solved_vars: 
# [0.33375504395436606, 0.9913930046304643, 0.6822354134806288, 0.0055759285328469965, 1.0157358846332918, -0.0019420558791265646, -0.005818806080402188, -0.0019420558791265646, -0.005818806080402188]

# parameters_and_solved_vars: [0.33375504395436606, 0.9913930046304643, 0.6822354134806288, 0.0055759285328469965, 1.0157358846332918, -0.0019420558791265646, -0.005818806080402188, -0.0019420558791265646, -0.005818806080402188]
# closest_parameters_and_solved_vars: [0.33375504395436606, 0.9913930046304643, 0.6822354134806288, 0.0055759285328469965, 1.0157358846332918, -0.0019420558791265646, -0.005818806080402188, -0.0019420558791265646, -0.005818806080402188]
# guess: [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
            #   sol = solution[1]
            #   P = sol[1]
            #   c = sol[2]
            #   k = sol[3]
            #   l = sol[4]
            #   n = sol[5]

𝓂 = FS2000
initial_parameters  = 𝓂.parameter_values
tol = Tolerances()
verbose = true
cold_start = true
solver_params = 𝓂.solver_parameters

clear_solution_caches!(𝓂, :first_order)
SS(𝓂, verbose = true, derivatives = false)

function solve_ss_(initial_parameters, 𝓂, tol, verbose, cold_start, solver_params; timer::TimerOutput = TimerOutput())
    @timeit timer "Prepare" begin
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3868 =#
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3875 =#
    initial_parameters = if typeof(initial_parameters) == Vector{Float64}
            initial_parameters
        else
            ℱ.value.(initial_parameters)
        end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3877 =#
    initial_parameters_tmp = copy(initial_parameters)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3879 =#
    parameters = copy(initial_parameters)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3880 =#
    params_flt = copy(initial_parameters)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3882 =#
    current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3883 =#
    closest_solution_init = 𝓂.NSSS_solver_cache[end]
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3885 =#
    for pars = 𝓂.NSSS_solver_cache
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3886 =#
        copy!(initial_parameters_tmp, pars[end])
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3888 =#
        ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3890 =#
        latest = sum(abs2, initial_parameters_tmp)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3891 =#
        if latest <= current_best
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3892 =#
            current_best = latest
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3893 =#
            closest_solution_init = pars
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3895 =#
    end
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3900 =#
    range_iters = 0
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3901 =#
    solution_error = 1.0
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3902 =#
    solved_scale = 0
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3904 =#
    scale = 1.0
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3906 =#
    NSSS_solver_cache_scale = CircularBuffer{Vector{Vector{Float64}}}(500)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3907 =#
    push!(NSSS_solver_cache_scale, closest_solution_init)
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3909 =#

    end # timeit_debug
    
    @timeit timer "while loop" begin

    while range_iters <= if cold_start
                    1
                else
                    500
                end && !(solution_error < tol.NSSS_acceptance_tol && solved_scale == 1)

        @timeit timer "prepare" begin

        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3910 =#
        range_iters += 1
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3911 =#
        fail_fast_solvers_only = if range_iters > 1
                true
            else
                false
            end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3913 =#
        if abs(solved_scale - scale) < 0.0001
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3915 =#
            break
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3928 =#
        current_best = sum(abs2, (NSSS_solver_cache_scale[end])[end] - initial_parameters)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3929 =#
        closest_solution = NSSS_solver_cache_scale[end]
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3931 =#
        for pars = NSSS_solver_cache_scale
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3932 =#
            copy!(initial_parameters_tmp, pars[end])
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3934 =#
            ℒ.axpy!(-1, initial_parameters, initial_parameters_tmp)
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3936 =#
            latest = sum(abs2, initial_parameters_tmp)
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3938 =#
            if latest <= current_best
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3939 =#
                current_best = latest
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3940 =#
                closest_solution = pars
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3942 =#
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3946 =#
        if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3947 =#
            parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
        else
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3949 =#
            parameters = copy(initial_parameters)
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3951 =#
        params_flt = parameters
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3955 =#
        alp = parameters[1]
        bet = parameters[2]
        gam = parameters[3]
        mst = parameters[4]
        psi = parameters[6]
        del = parameters[7]
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3956 =#
        mst = min(max(mst, 2.220446049250313e-16), 1.0e12)
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3957 =#
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3958 =#
        NSSS_solver_cache_tmp = []
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3959 =#
        solution_error = 0.0
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3960 =#
        iters = 0
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3961 =#
        ➕₅ = min(max(-1.0e12, -gam), 600.0)
        solution_error += abs(➕₅ - -gam)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
                println("Failed for analytical aux variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            continue
        end
        ➕₄ = min(max(-1.0e12, -alp * gam), 600.0)
        solution_error += abs(➕₄ - -alp * gam)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
                println("Failed for analytical aux variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            continue
        end
        m = mst
        solution_error += abs(min(max(2.220446049250313e-16, m), 1.0e12) - m)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
                println("Failed for bounded variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            continue
        end
        e = 1.0
        ➕₂ = min(max(-1.0e12, -alp * gam), 600.0)
        solution_error += abs(➕₂ - -alp * gam)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
                println("Failed for analytical aux variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            continue
        end
        ➕₃ = min(max(-1.0e12, -gam), 600.0)
        solution_error += abs(➕₃ - -gam)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
                println("Failed for analytical aux variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            continue
        end
        params_and_solved_vars = [alp, bet, psi, del, m, ➕₂, ➕₃, ➕₄, ➕₅]
        lbs = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12]
        ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 600.0, 600.0, 600.0, 600.0]
        inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]

        end # timeit_debug
        @timeit timer "block solve" begin

        solution = block_solver_(params_and_solved_vars, 1, 𝓂.ss_solve_blocks_in_place[1], inits, lbs, ubs, solver_params, fail_fast_solvers_only, cold_start, verbose, timer = timer)

        end # timeit_debug
        @timeit timer "epilogue" begin

        iters += (solution[2])[2]
        solution_error += (solution[2])[1]
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
                println("Failed after solving block with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:2986 =#
            continue
        end
        sol = solution[1]
        P = sol[1]
        c = sol[2]
        k = sol[3]
        l = sol[4]
        n = sol[5]
        NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
                    sol
                else
                    ℱ.value.(sol)
                end]
        NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
                    params_and_solved_vars
                else
                    ℱ.value.(params_and_solved_vars)
                end]
        ➕₁ = min(max(-1.0e12, gam), 600.0)
        solution_error += abs(➕₁ - gam)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
                println("Failed for analytical aux variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3705 =#
            continue
        end
        dA = exp(➕₁)
        gy_obs = dA
        solution_error += abs(min(max(2.220446049250313e-16, gy_obs), 1.0e12) - gy_obs)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
                println("Failed for bounded variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            continue
        end
        gp_obs = m / dA
        solution_error += abs(min(max(2.220446049250313e-16, gp_obs), 1.0e12) - gp_obs)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
                println("Failed for bounded variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3724 =#
            continue
        end
        ➕₆ = min(1.0e12, max(eps(), gp_obs))
        solution_error += +(abs(➕₆ - gp_obs))
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
                println("Failed for analytical variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            continue
        end
        log_gp_obs = log(➕₆)
        d = (l - m) + 1
        W = l / n
        ➕₇ = min(1.0e12, max(eps(), gy_obs))
        solution_error += +(abs(➕₇ - gy_obs))
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
                println("Failed for analytical variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            continue
        end
        log_gy_obs = log(➕₇)
        ➕₈ = min(1.0e12, max(eps(), k))
        ➕₉ = min(1.0e12, max(eps(), n))
        solution_error += abs(➕₈ - k) + abs(➕₉ - n)
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
                println("Failed for analytical variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            continue
        end
        y = ➕₈ ^ alp * ➕₉ ^ (1 - alp) * exp(➕₄)
        ➕₁₀ = min(1.0e12, max(eps(), k / n))
        solution_error += +(abs(➕₁₀ - k / n))
        if solution_error > tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            if verbose
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
                println("Failed for analytical variables with error $(solution_error)")
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            scale = scale * 0.3 + solved_scale * 0.7
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3716 =#
            continue
        end
        R = (P * ➕₁₀ ^ alp * (1 - alp) * exp(➕₄)) / W
        if length(NSSS_solver_cache_tmp) == 0
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3838 =#
            NSSS_solver_cache_tmp = [copy(params_flt)]
        else
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3838 =#
            NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)]
        end
        if current_best > 1.0e-8 && (solution_error < tol.NSSS_acceptance_tol && scale == 1)
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3848 =#
            reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
        end

        end # timeit_debug

        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3963 =#
        if solution_error < tol.NSSS_acceptance_tol
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3965 =#
            solved_scale = scale
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3966 =#
            if scale == 1
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3970 =#
                return ([P, P, R, W, c, c, d, dA, e, gp_obs, gy_obs, k, l, log_gp_obs, log_gy_obs, m, n, y], (solution_error, iters))
            else
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3972 =#
                reverse_diff_friendly_push!(NSSS_solver_cache_scale, NSSS_solver_cache_tmp)
            end
            #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3975 =#
            if scale > 0.95
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3976 =#
                scale = 1
            else
                #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3979 =#
                scale = scale * 0.4 + 0.6
            end
        end
        #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3989 =#
    end

    end # timeit_debug
    
    #= /Users/thorekockerols/GitHub/MacroModelling.jl/src/MacroModelling.jl:3990 =#
    return ([0.0], (1, 0))
end


function block_solver_(parameters_and_solved_vars::Vector{T}, 
                        n_block::Int, 
                        # ss_solve_blocks::Function, 
                        SS_solve_block::ss_solve_block,
                        # SS_optimizer, 
                        # f::OptimizationFunction, 
                        guess_and_pars_solved_vars::Vector{Vector{T}}, 
                        lbs::Vector{T}, 
                        ubs::Vector{T},
                        parameters::Vector{solver_parameters},
                        fail_fast_solvers_only::Bool,
                        cold_start::Bool,
                        verbose::Bool ;
                        tol::Tolerances = Tolerances(),
                        timer::TimerOutput = TimerOutput()
                        # rtol::AbstractFloat = sqrt(eps()),
                        # timeout = 120,
                        # starting_points::Vector{Float64} = [1.205996189998029, 0.7688, 0.897, 1.2],#, 0.9, 0.75, 1.5, -0.5, 2.0, .25]
                        # verbose::Bool = false
                        )::Tuple{Vector{T},Tuple{T, Int}} where T <: AbstractFloat

    @timeit timer "prepare" begin
    # tol = parameters[1].ftol
    # rtol = parameters[1].rel_xtol

    solved_yet = false

    guess = guess_and_pars_solved_vars[1]

    sol_values = guess

    closest_parameters_and_solved_vars = sum(abs, guess_and_pars_solved_vars[2]) == Inf ? parameters_and_solved_vars : guess_and_pars_solved_vars[2]

    # res = ss_solve_blocks(parameters_and_solved_vars, guess)

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, guess, parameters_and_solved_vars) # TODO: make the block a struct
    # TODO: do the function creation with Symbolics as this will solve the compilation bottleneck for large functions

    res = SS_solve_block.ss_problem.func_buffer

    sol_minimum  = ℒ.norm(res)
 
    if !cold_start
        if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol
            # ∇ = 𝒟.jacobian(x->(ss_solve_blocks(parameters_and_solved_vars, x)), backend, guess)

            # ∇̂ = ℒ.lu!(∇, check = false)

            SS_solve_block.ss_problem.jac(SS_solve_block.ss_problem.jac_buffer, guess, parameters_and_solved_vars)

            ∇ = SS_solve_block.ss_problem.jac_buffer

            ∇̂ = ℒ.lu(∇, check = false)
            
            if ℒ.issuccess(∇̂)
                guess_update = ∇̂ \ res

                new_guess = guess - guess_update

                rel_sol_minimum = ℒ.norm(guess_update) / max(ℒ.norm(new_guess), sol_minimum)
            else
                rel_sol_minimum = 1.0
            end
        else
            rel_sol_minimum = 0.0
        end
    else
        rel_sol_minimum = 1.0
    end
    
    if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
        solved_yet = true

        if verbose
            println("Block: $n_block, - Solved using previous solution; residual norm: $sol_minimum")
        end
    end

    total_iters = [0,0]

    SS_optimizer = levenberg_marquardt_

    end # timeit_debug

    if cold_start
        @timeit timer "cold_start" begin
       
        guesses = any(guess .< 1e12) ? [guess, fill(1e12, length(guess))] : [guess] # if guess were provided, loop over them, and then the starting points only

        for g in guesses
            for p in parameters
                for ext in [true, false] # try first the system where values and parameters can vary, next try the system where only values can vary
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol# || rel_sol_minimum > rtol
                        if solved_yet continue end

                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss__(SS_optimizer, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                        # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(SS_optimizer, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, total_iters, n_block, verbose,
                                                            g, 
                                                            p,
                                                            ext,
                                                            false,
                                                            timer = timer)
                                                            
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol
                            solved_yet = true
                        end
                    end
                end
            end
        end

        end # timeit_debug
       
    else !cold_start

        @timeit timer "no cold start" begin
       
        for p in (fail_fast_solvers_only ? [parameters[end]] : unique(parameters)) #[1:3] # take unique because some parameters might appear more than once
            for s in (fail_fast_solvers_only ? [false] : Any[false,p.starting_value, 1.206, 1.5, 0.7688, 2.0, 0.897]) #, .9, .75, 1.5, -.5, 2, .25] # try first the guess and then different starting values
                # for ext in [false, true] # try first the system where only values can vary, next try the system where values and parameters can vary
                for algo in [newton, levenberg_marquardt_]
                    if !isfinite(sol_minimum) || sol_minimum > tol.NSSS_acceptance_tol # || rel_sol_minimum > rtol
                        if solved_yet continue end
                        # println("Block: $n_block pre GN - $ext - $sol_minimum - $rel_sol_minimum")
                        sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss__(algo, SS_solve_block, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, 
                        # sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(algo, ss_solve_blocks, parameters_and_solved_vars, closest_parameters_and_solved_vars, lbs, ubs, tol, 
                                                                            total_iters, 
                                                                            n_block, 
                                                                            false, # verbose
                                                                            guess, 
                                                                            p, 
                                                                            # parameters[1],
                                                                            false, # ext
                                                                            # false)
                                                                            s,
                                                                            timer = timer) 
                        if isfinite(sol_minimum) && sol_minimum < tol.NSSS_acceptance_tol # || rel_sol_minimum > rtol)
                            solved_yet = true

                            if verbose
                                # println("Block: $n_block, - Solved with $algo using previous solution - $(indexin([ext],[false, true])[1])/2 - $ext - $sol_minimum - $rel_sol_minimum - $total_iters")
                                println("Block: $n_block, - Solved with $algo using previous solution - $sol_minimum - $rel_sol_minimum - $total_iters")
                            end
                        end                      
                    end
                end
            end
        end

        end # timeit_debug
    end

    if verbose
        if !solved_yet
            println("Block: $n_block, - Solution not found after $(total_iters[1]) gradient evaluations and $(total_iters[2]) function evaluations; reltol: $rel_sol_minimum - tol: $sol_minimum")
        end
    end

    return sol_values, (sol_minimum, total_iters[1])
end




function solve_ss__(SS_optimizer::Function,
                    # ss_solve_blocks::Function,
                    SS_solve_block::ss_solve_block,
                    parameters_and_solved_vars::Vector{T},
                    closest_parameters_and_solved_vars::Vector{T},
                    lbs::Vector{T},
                    ubs::Vector{T},
                    tol::Tolerances,
                    total_iters::Vector{Int},
                    n_block::Int,
                    verbose::Bool,
                    guess::Vector{T},
                    solver_params::solver_parameters,
                    extended_problem::Bool,
                    separate_starting_value::Union{Bool,T};
                    timer::TimerOutput = TimerOutput())::Tuple{Vector{T}, Vector{Int}, T, T} where T <: AbstractFloat

    @timeit timer "prepare" begin
       
    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    if separate_starting_value isa Float64
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], fill(separate_starting_value, length(guess))))
        sol_values_init[ubs[1:length(guess)] .<= 1] .= .1 # capture cases where part of values is small
    else
        sol_values_init = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], [g < 1e12 ? g : solver_params.starting_value for g in guess]))
    end
    
    end # timeit_debug

    @timeit timer "optimiser" begin
       
    sol_new_tmp, info = SS_optimizer(   extended_problem ? SS_solve_block.extended_ss_problem : SS_solve_block.ss_problem,
    # if extended_problem
    #     function ext_function_to_optimize(guesses)
    #         gss = guesses[1:length(guess)]
    
    #         parameters_and_solved_vars_guess = guesses[length(guess)+1:end]
    
    #         res = ss_solve_blocks(parameters_and_solved_vars, gss)
    
    #         return vcat(res, parameters_and_solved_vars .- parameters_and_solved_vars_guess)
    #     end
    # else
    #     function function_to_optimize(guesses) ss_solve_blocks(parameters_and_solved_vars, guesses) end
    # end

    # sol_new_tmp, info = SS_optimizer(   extended_problem ? ext_function_to_optimize : function_to_optimize,
                                        extended_problem ? vcat(sol_values_init, closest_parameters_and_solved_vars) : sol_values_init,
                                        parameters_and_solved_vars,
                                        extended_problem ? lbs : lbs[1:length(guess)],
                                        extended_problem ? ubs : ubs[1:length(guess)],
                                        solver_params,
                                        tol = tol   ,
                                        timer = timer)

    end # timeit_debug

    @timeit timer "epilogue" begin
       
    sol_new = isnothing(sol_new_tmp) ? sol_new_tmp : sol_new_tmp[1:length(guess)]

    sol_minimum = info[4] # isnan(sum(abs, info[4])) ? Inf : ℒ.norm(info[4])
    
    rel_sol_minimum = info[3]

    sol_values = max.(lbs[1:length(guess)], min.(ubs[1:length(guess)], sol_new))

    total_iters[1] += info[1]
    total_iters[2] += info[2]

    extended_problem_str = extended_problem ? "(extended problem) " : ""

    if separate_starting_value isa Bool
        starting_value_str = ""
    else
        starting_value_str = "and starting point: $separate_starting_value"
    end

    if all(guess .< 1e12) && separate_starting_value isa Bool
        any_guess_str = "previous solution, "
    elseif any(guess .< 1e12) && separate_starting_value isa Bool
        any_guess_str = "provided guess, "
    else
        any_guess_str = ""
    end

    # max_resid = maximum(abs,ss_solve_blocks(parameters_and_solved_vars, sol_values))

    SS_solve_block.ss_problem.func(SS_solve_block.ss_problem.func_buffer, sol_values, parameters_and_solved_vars)
    
    max_resid = maximum(abs, SS_solve_block.ss_problem.func_buffer)

    if sol_minimum < ftol && verbose
        println("Block: $n_block - Solved $(extended_problem_str) using ",string(SS_optimizer),", $(any_guess_str)$(starting_value_str); maximum residual = $max_resid")
    end

    end # timeit_debug

    return sol_values, total_iters, rel_sol_minimum, sol_minimum
end



function levenberg_marquardt_(
    fnj::function_and_jacobian,
    # f::Function, 
    initial_guess::Array{T,1}, 
    parameters_and_solved_vars::Array{T,1},
    lower_bounds::Array{T,1}, 
    upper_bounds::Array{T,1},
    parameters::solver_parameters;
    tol::Tolerances = Tolerances(),
    timer::TimerOutput = TimerOutput()
    )::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    @timeit timer "prepare" begin
       
    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    
    ϕ̄ = parameters.ϕ̄
    ϕ̂ = parameters.ϕ̂
    μ̄¹ = parameters.μ̄¹
    μ̄² = parameters.μ̄²
    p̄¹ = parameters.p̄¹
    p̄² = parameters.p̄²
    ρ = parameters.ρ
    ρ¹ = parameters.ρ¹
    ρ² = parameters.ρ²
    ρ³ = parameters.ρ³
    ν = parameters.ν
    λ¹ = parameters.λ¹
    λ² = parameters.λ²
    λ̂¹ = parameters.λ̂¹
    λ̂² = parameters.λ̂²
    λ̅¹ = parameters.λ̅¹
    λ̅² = parameters.λ̅²
    λ̂̅¹ = parameters.λ̂̅¹
    λ̂̅² = parameters.λ̂̅²
    transformation_level = parameters.transformation_level
    shift = parameters.shift
    backtracking_order = parameters.backtracking_order

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)
    @assert backtracking_order ∈ [2,3] "Backtracking order can only be quadratic (2) or cubic (3)."

    max_linesearch_iterations = 600

    # function f̂(x) 
    #     f(undo_transform(x,transformation_level))  
    # #     # f(undo_transform(x,transformation_level,shift))  
    # end
    u_bounds = copy(upper_bounds)
    l_bounds = copy(lower_bounds)
    current_guess = copy(initial_guess)

    for _ in 1:transformation_level
        u_bounds .= asinh.(u_bounds)
        l_bounds .= asinh.(l_bounds)
        current_guess .= asinh.(current_guess)
    end

    current_guess_untransformed = copy(current_guess)
    # upper_bounds  = transform(upper_bounds,transformation_level)
    # upper_bounds  = transform(upper_bounds,transformation_level,shift)
    # lower_bounds  = transform(lower_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level,shift)

    # current_guess = copy(transform(initial_guess,transformation_level))
    # current_guess_untransformed = copy(transform(initial_guess,transformation_level))
    # current_guess = copy(transform(initial_guess,transformation_level,shift))
    previous_guess = similar(current_guess)
    previous_guess_untransformed = similar(current_guess)
    guess_update = similar(current_guess)
    factor = similar(current_guess)
    factor_tmp = similar(current_guess)
    best_previous_guess = similar(current_guess)
    best_current_guess = similar(current_guess)

    end # timeit_debug

    @timeit timer "prepare jac and linear problem" begin

    sol_cache = fnj.chol_buffer

    # ∇ = Array{T,2}(undef, length(initial_guess), length(initial_guess))
    ∇ = fnj.jac_buffer
    ∇̂ = sol_cache.A
    ∇̄ = similar(fnj.jac_buffer)

    # ∇̂ = choose_matrix_format(∇' * ∇, multithreaded = false)
    
    # ∇̂ = sparse(∇' * ∇)

    # if ∇̂ isa SparseMatrixCSC
    #     prob = 𝒮.LinearProblem(∇̂, guess_update, 𝒮.CHOLMODFactorization())
    #     sol_cache = 𝒮.init(prob, 𝒮.CHOLMODFactorization())
    # else
        # X = ℒ.Symmetric(∇̂, :U)
        # prob = 𝒮.LinearProblem(X, guess_update, 𝒮.CholeskyFactorization)
        # prob = 𝒮.LinearProblem(∇̂, guess_update, 𝒮.CholeskyFactorization())
        # sol_cache = 𝒮.init(prob, 𝒮.CholeskyFactorization())
    # end
    
    # prep = 𝒟.prepare_jacobian(f̂, backend, current_guess)

    largest_step = T(1.0)
    largest_residual = T(1.0)
    largest_relative_step = T(1.0)

    μ¹ = μ̄¹
    μ² = μ̄²

    p¹ = p̄¹
    p² = p̄²

    grad_iter = 0
    func_iter = 0

    end # timeit_debug

    @timeit timer "iterations" begin

    for iter in 1:iterations

        @timeit timer "jacobian" begin

        # make the jacobian and f calls nonallocating
        copy!(current_guess_untransformed, current_guess)
        
        if transformation_level > 0
            factor .= 1
            for _ in 1:transformation_level
                factor .*= cosh.(current_guess_untransformed)
                current_guess_untransformed .= sinh.(current_guess_untransformed)
            end
        end

        fnj.jac(∇, current_guess_untransformed, parameters_and_solved_vars)
        # 𝒟.jacobian!(f̂, ∇, prep, backend, current_guess)

        if transformation_level > 0
            if ∇ isa SparseMatrixCSC
                # ∇̄ = ∇ .* factor'
                copy!(∇̄.nzval, ∇.nzval)
                @inbounds for j in 1:size(∇, 2)
                    col_start = ∇̄.colptr[j]
                    col_end = ∇̄.colptr[j+1] - 1
                    for k in col_start:col_end
                        ∇̄.nzval[k] *= factor[j]
                    end
                end
            else
                # ℒ.mul!(∇̄, ∇, factor')
                @. ∇̄ = ∇ * factor'
                # ∇ .*= factor'
            end
        end

        grad_iter += 1

        previous_guess .= current_guess

        # ∇̂ .= ∇' * ∇
        if ∇̄ isa SparseMatrixCSC && ∇̂ isa SparseMatrixCSC
            ∇̂ = ∇̄' * ∇̄
        else
            ℒ.mul!(∇̂, ∇̄', ∇̄)
        end

        end # timeit_debug

        @timeit timer "func" begin

            @timeit timer "eval" begin

                fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

            end # timeit_debug

            @timeit timer "copy" begin

            copy!(factor, fnj.func_buffer)

            end # timeit_debug

            @timeit timer "μ¹s" begin

            μ¹s = μ¹ * ℒ.dot(factor, factor)^p¹
            # μ¹s = μ¹ * sum(abs2, f̂(current_guess))^p¹
            func_iter += 1

            end # timeit_debug

            @timeit timer "∇̂" begin

            # tmp = 0.0
    # println(∇̂)
            update_∇̂!(∇̂, μ¹s, μ², p²)
            # @inbounds for i in 1:size(∇̂,1)
            #     ∇̂[i,i] += μ¹s
            #     # tmp = ∇̂[i,i]
            #     # tmp ^= p²
            #     # tmp *= μ²
            #     ∇̂[i,i] += μ² * ∇̂[i,i]^p²
            # end
            # ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²

            end # timeit_debug

            @timeit timer "isfinite" begin

            # finn  = !all(isfinite, ∇̂)
            finn  = has_nonfinite(∇̂)
            # finn  = !isfinite(sum(∇̂))

            end # timeit_debug

            if finn
                largest_relative_step = 1.0
                largest_residual = 1.0
                break
            end


            @timeit timer "mul!" begin

            # fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

            ℒ.mul!(guess_update, ∇̄', factor)

            end # timeit_debug

        end # timeit_debug

        @timeit timer "solve" begin

        # X = ℒ.Symmetric(∇̂, :U)
        sol_cache.A = ∇̂
        sol_cache.b = guess_update
        𝒮.solve!(sol_cache)
        copy!(guess_update, sol_cache.u)

        end # timeit_debug

        @timeit timer "wrangle" begin

        @timeit timer "isfinite" begin

        if !isfinite(sum(guess_update))
            largest_relative_step = 1.0
            largest_residual = 1.0
            break
        end

        end # timeit_debug

        @timeit timer "axpy!" begin

        ℒ.axpy!(-1, guess_update, current_guess)
        # current_guess .-= ∇̄ \ ∇' * f̂(current_guess)

        end # timeit_debug

        @timeit timer "minmax!" begin

        minmax!(current_guess, l_bounds, u_bounds)

        end # timeit_debug

        @timeit timer "copy!" begin

        copy!(previous_guess_untransformed, previous_guess)

        end # timeit_debug

        @timeit timer "transform" begin

        for _ in 1:transformation_level
            previous_guess_untransformed .= sinh.(previous_guess_untransformed)
        end
        
        # fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        end # timeit_debug

        @timeit timer "P" begin

        P = ℒ.dot(factor, factor)
        # P = sum(abs2, f̂(previous_guess))
        P̃ = P
        
        end # timeit_debug

        @timeit timer "copy!" begin

        copy!(current_guess_untransformed, current_guess)

        end # timeit_debug

        @timeit timer "transform 2" begin

        for _ in 1:transformation_level
            current_guess_untransformed .= sinh.(current_guess_untransformed)
        end

        end # timeit_debug

        @timeit timer "func eval" begin

        fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)
      
        end # timeit_debug

        @timeit timer "P̋" begin

        P̋ = ℒ.dot(fnj.func_buffer, fnj.func_buffer)
        # P̋ = ℒ.dot(fnj.func_buffer, fnj.func_buffer)
        # P̋ = ℒ.norm(fnj.func_buffer)^2
        # P̋ = sum(abs2, f̂(current_guess))

        end # timeit_debug

        @timeit timer "alloc vars" begin

        func_iter += 3

        α = 1.0
        ᾱ = 1.0

        ν̂ = ν

        end # timeit_debug

        @timeit timer "guess update" begin

        # guess_update .= current_guess - previous_guess

        guess_update .= current_guess
        guess_update .-= previous_guess

        # fnj.func(fnj.func_buffer, previous_guess_untransformed, parameters_and_solved_vars)

        end # timeit_debug

        @timeit timer "g" begin

        # g = update_g(factor, ∇̄, guess_update)
        # ℒ.mul!(factor_tmp, ∇̄', factor)
        g = ℒ.dot(factor, ∇̄, guess_update)
        # g = f̂(previous_guess)' * ∇ * guess_update

        end # timeit_debug

        @timeit timer "U" begin

        U = sum(abs2,guess_update)
        func_iter += 1

        end # timeit_debug

        end # timeit_debug

        @timeit timer "line search" begin

        if P̋ > ρ * P 
            linesearch_iterations = 0

            @timeit timer "while condition" begin

            # cond  = P̋ > (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U
            cond = condition_P̋(P̋, ν̂, ρ¹, α, P̃, ρ², g, ρ³, U)

            end # timeit_debug

            while cond && linesearch_iterations < max_linesearch_iterations
                if backtracking_order == 2
                    # Quadratic backtracking line search

                    @timeit timer "quad" begin

                    # α̂ = -g * α^2 / (2 * (P̋ - P̃ - g * α))
                    α̂ = update_α̂(g, α, P̋, P̃)

                    end # timeit_debug

                elseif backtracking_order == 3

                    @timeit timer "cubic" begin

                    # Cubic backtracking line search
                    a = (ᾱ^2 * (P̋ - P̃ - g * α) - α^2 * (P - P̃ - g * ᾱ)) / (ᾱ^2 * α^2 * (α - ᾱ))
                    b = (α^3 * (P - P̃ - g * ᾱ) - ᾱ^3 * (P̋ - P̃ - g * α)) / (ᾱ^2 * α^2 * (α - ᾱ))

                    if isapprox(a, zero(a), atol=eps())
                        α̂ = g / (2 * b)
                    else
                        # discriminant
                        d = max(b^2 - 3 * a * g, 0)
                        # quadratic equation root
                        α̂ = (sqrt(d) - b) / (3 * a)
                    end

                    ᾱ = α

                    end # timeit_debug

                end

                @timeit timer "minmax" begin

                α̂, α = minmax_α(α̂, ϕ̄, α, ϕ̂)
                # tmp = ϕ̄ * α
                # if α̂ > tmp
                #     α̂ = tmp
                # end

                # tmp2 = ϕ̂ * α
                # if α̂ > tmp2
                #     α = α̂
                # else
                #     α = tmp2
                # end

                # α̂ = min(α̂, ϕ̄ * α)
                # α = max(α̂, ϕ̂ * α)
                
                end # timeit_debug

                @timeit timer "copy/axpy!" begin

                copy!(current_guess, previous_guess)
                ℒ.axpy!(α, guess_update, current_guess)

                end # timeit_debug

                @timeit timer "minmax!" begin

                # current_guess .= previous_guess + α * guess_update
                minmax!(current_guess, l_bounds, u_bounds)
                
                end # timeit_debug

                @timeit timer "P and copy" begin

                P = P̋

                copy!(current_guess_untransformed, current_guess)

                end # timeit_debug

                @timeit timer "transform" begin

                for _ in 1:transformation_level
                    current_guess_untransformed .= sinh.(current_guess_untransformed)
                end

                end # timeit_debug

                @timeit timer "func eval" begin

                fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

                end # timeit_debug

                @timeit timer "update P" begin

                # P̋ = sum(abs2, fnj.func_buffer)
                P̋ = ℒ.dot(fnj.func_buffer, fnj.func_buffer)
                # P̋ = ℒ.norm(fnj.func_buffer)^2
                # P̋ = sum(abs2, f̂(current_guess))

                end # timeit_debug

                @timeit timer "move counters" begin

                func_iter += 1

                ν̂ *= α

                linesearch_iterations += 1

                end # timeit_debug

                @timeit timer "while condition update" begin

                # cond  = P̋ > (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U
                cond = condition_P̋(P̋, ν̂, ρ¹, α, P̃, ρ², g, ρ³, U)

                end # timeit_debug
            end

            @timeit timer "update μ and p" begin

            μ¹ *= λ̅¹
            μ² *= λ̅²

            p¹ *= λ̂̅¹
            p² *= λ̂̅²

            end # timeit_debug
        else

            @timeit timer "update" begin

            μ¹ = min(μ¹ / λ¹, μ̄¹)
            μ² = min(μ² / λ², μ̄²)

            p¹ = min(p¹ / λ̂¹, p̄¹)
            p² = min(p² / λ̂², p̄²)

            end # timeit_debug
        end

        end # timeit_debug

        @timeit timer "undo transform" begin

        for _ in 1:transformation_level
            best_previous_guess .= sinh.(previous_guess)
            best_current_guess .= sinh.(current_guess)
        end

        # best_previous_guess = undo_transform(previous_guess, transformation_level)
        # best_current_guess = undo_transform(current_guess, transformation_level)

        end # timeit_debug

        @timeit timer "norms" begin

        @. factor = best_previous_guess - best_current_guess
        largest_step = ℒ.norm(factor) # maximum(abs, previous_guess - current_guess)
        largest_relative_step = largest_step / max(ℒ.norm(best_previous_guess), ℒ.norm(best_current_guess)) # maximum(abs, (previous_guess - current_guess) ./ previous_guess)
        
        end # timeit_debug

        @timeit timer "copy and transform" begin

        copy!(current_guess_untransformed, current_guess)

        for _ in 1:transformation_level
            current_guess_untransformed .= sinh.(current_guess_untransformed)
        end

        end # timeit_debug

        @timeit timer "eval" begin

        fnj.func(fnj.func_buffer, current_guess_untransformed, parameters_and_solved_vars)

        end # timeit_debug

        @timeit timer "largest res" begin

        largest_residual = ℒ.norm(fnj.func_buffer)    
        # largest_residual = ℒ.norm(f̂(current_guess)) # maximum(abs, f(undo_transform(current_guess,transformation_level)))
        # largest_residual = maximum(abs, f(undo_transform(current_guess,transformation_level,shift)))

        # allow for norm increases (in both measures) as this can lead to the solution
        
        if largest_residual <= ftol || largest_step <= xtol || largest_relative_step <= rel_xtol
            # println("LM Iteration: $iter; xtol ($xtol): $largest_step; ftol ($ftol): $largest_residual; rel_xtol ($rel_xtol): $largest_relative_step")
            break
            # else
            # end
        end

        end # timeit_debug

    end

    end # timeit_debug

    @timeit timer "epilogue" begin
     
    for _ in 1:transformation_level
        best_current_guess .= sinh.(current_guess)
    end

    end # timeit_debug

    return best_current_guess, (grad_iter, func_iter, largest_relative_step, largest_residual)#, f(best_guess))
end



function update_g(factor, ∇̄, guess_update)
    return factor' * ∇̄ * guess_update
end

function update_∇̂!(∇̂::AbstractMatrix{T}, μ¹s::T, μ²::T, p²::T) where T <: Real
    n = size(∇̂, 1)                # hoist size lookup
    @inbounds for i in 1:n
        x = ∇̂[i,i]                # read once
        x += μ¹s
        x += μ² * (x^p²)          # scalar pow, no array allocation
        ∇̂[i,i] = x               # write back
    end
    return nothing
end

function minmax_α(α̂::T, ϕ̄::T, α::T, ϕ̂::T)::Tuple{T,T} where T <: Real
    α̂ = min(α̂, ϕ̄ * α)
    α = max(α̂, ϕ̂ * α)
    return α̂, α
end

function condition_P̋(P̋::T, ν̂::T, ρ¹::T, α::T, P̃::T, ρ²::T, g::T, ρ³::T, U::T)::Bool where T <: Real
    cond  = (1 + ν̂ - ρ¹ * α^2) * P̃ + ρ² * α^2 * g - ρ³ * α^2 * U
    return P̋ > cond
end

function update_α̂(g::T, α::T, P̋::T, P̃::T)::T where T <: Real
    return -g * α^2 / (2 * (P̋ - P̃ - g * α))
end

function has_nonfinite(A::AbstractArray)
    @inbounds for x in A
        if !isfinite(x)
            return true
        end
    end
    return false
end


# has_nonfinite(FS2000.ss_solve_blocks_in_place[1].ss_problem.jac_buffer)
# !all(isfinite,FS2000.ss_solve_blocks_in_place[1].ss_problem.jac_buffer)
# @benchmark has_nonfinite($FS2000.ss_solve_blocks_in_place[1].ss_problem.jac_buffer)
# @benchmark !all(isfinite,$FS2000.ss_solve_blocks_in_place[1].ss_problem.jac_buffer)

clear_solution_caches!(𝓂, :first_order)
solve_ss_(initial_parameters, 𝓂, tol, true, cold_start, solver_params)

@benchmark solve_ss_(initial_parameters, 𝓂, tol, false, cold_start, solver_params) setup = clear_solution_caches!(𝓂, :first_order)

timer = TimerOutput()
clear_solution_caches!(𝓂, :first_order)
solve_ss_(initial_parameters, 𝓂, tol, false, cold_start, solver_params,timer=timer)
timer

a = rand(500)
@benchmark dot(a,a)
@benchmark norm(a)
@benchmark sum(abs2,a)

A = FS2000.ss_solve_blocks_in_place[1].extended_ss_problem.jac_buffer
AA = A*A'
@benchmark lu(AA)
@benchmark qr(AA)
symAA = Symmetric(AA, :U)
@benchmark bunchkaufman(symAA)
@benchmark cholesky(symAA)
@profview for i in 1:10000 factorize(AA) end
@benchmark solve_ss_($initial_parameters, $𝓂, $tol, false, $cold_start, $solver_params) setup = clear_solution_caches!(𝓂, :first_order)


guess_update = rand(size(A,1))
prob = 𝒮.LinearProblem(A, guess_update, 𝒮.LUFactorization())

sol_cache = 𝒮.init(prob)
@benchmark 𝒮.solve!(sol_cache)
@profview for i in 1:10000 𝒮.solve!(sol_cache) end
sol_cache.u

@profview for i in 1:10000 A \ guess_update end
@profview for i in 1:10000 AA \ guess_update end
@profview for i in 1:10000 
    cholAA = cholesky(AA)
    cholAA \ guess_update 
end


@benchmark begin
    cholAA = cholesky(AA)
    cholAA \ guess_update 
end
@benchmark AA \ guess_update



prob = 𝒮.LinearProblem((AA), guess_update)

sol_cache = 𝒮.init(prob)#,𝒮.SimpleLUFactorization())
@benchmark 𝒮.solve!(sol_cache) setup = sol_cache.b = rand(size(A,1))

sol_cache.u
@profview for i in 1:1000000 𝒮.solve!(sol_cache) end


@profview for i in 1:100 
clear_solution_caches!(𝓂, :first_order)
solve_ss_(initial_parameters, 𝓂, tol, false, cold_start, solver_params,timer=timer)
end

timer = TimerOutput()
clear_solution_caches!(𝓂, :first_order)
solve_ss_(initial_parameters, 𝓂, tol, false, cold_start, solver_params,timer=timer)
timer



A = [18.93748305616677 16.78067900566427 0.5127725231944739 -18.277223724411655 53.900985311550265 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 16.78067900566427 18.679805975295558 -0.190165290010227 -15.962730981367663 46.38101934454539 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.5127725231944739 -0.190165290010227 0.14703754108662906 -0.5538046190793915 1.157347541294703 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; -18.277223724411655 -15.962730981367663 -0.5538046190793915 19.739768550388817 -58.214146238182884 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 53.900985311550265 46.38101934454539 1.157347541294703 -58.214146238182884 186.6106637860385 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.1267359999999997 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.986049 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.4225 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0000999999999998 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0004000399999997 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.000009156676 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.00007225 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.000009156676 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.00007225]

A 
μ¹s = 1.2
p² = 1.3
μ² = 1.1

@benchmark update_∇̂!(A, μ¹s, μ², p²)
tmp = 0.0
@benchmark for i in 1:size(A,1)
    # tmp = A[i,i]
    # tmp += μ¹s
    A[i,i] = 1.0
    # A[i,i] += μ² * A[i,i]^p²
end

@benchmark for i in 1:size(A,1)
    A[i,i] += μ¹s
    A[i,i] += μ² * A[i,i]^p²
end

for i in 1:size(A,1)
    A[i^2] += μ¹s
    A[i^2] += μ² * A[i^2]^p²
end

diag(A) .+= μ¹s
# ∇̂ .+= μ¹ * sum(abs2, f̂(current_guess))^p¹ * ℒ.I + μ² * ℒ.Diagonal(∇̂).^p²


initial_guess = [5.282429995876679, 5.282429995876679, 5.282429995876679, 5.282429995876679, 5.282429995876679]
fnj = FS2000.ss_solve_blocks_in_place[1].ss_problem
parameters_and_solved_vars = [0.33375504395436606, 0.9913930046304643, 0.6822354134806288, 0.0055759285328469965, 1.0157358846332918, -0.0019420558791265646, -0.005818806080402188, -0.0019420558791265646, -0.005818806080402188]
lower_bounds = [-1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16]
upper_bounds = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
tol = Tolerances()
# function newton(
#     # f::Function, 
#     fnj::function_and_jacobian, 
#     initial_guess::Array{T,1}, 
#     parameters_and_solved_vars::Array{T,1},
#     lower_bounds::Array{T,1}, 
#     upper_bounds::Array{T,1},
#     parameters::solver_parameters;
#     tol::Tolerances = Tolerances()
    # )::Tuple{Vector{T}, Tuple{Int, Int, T, T}} where {T <: AbstractFloat}
    # issues with optimization: https://www.gurobi.com/documentation/8.1/refman/numerics_gurobi_guidelines.html

    xtol = tol.NSSS_xtol
    ftol = tol.NSSS_ftol
    rel_xtol = tol.NSSS_rel_xtol

    iterations = 250
    transformation_level = 0 # parameters.transformation_level

    @assert size(lower_bounds) == size(upper_bounds) == size(initial_guess)
    @assert all(lower_bounds .< upper_bounds)

    # function f̂(x) 
    #     f(undo_transform(x,transformation_level))  
    # end

    # upper_bounds  = transform(upper_bounds,transformation_level)
    # lower_bounds  = transform(lower_bounds,transformation_level)

    # new_guess = copy(transform(initial_guess,transformation_level))

    new_guess = copy(initial_guess)
    guess_update = copy(initial_guess)

    new_guess = [1.1,1.9,18,.8,.3]

    fnj.func(fnj.func_buffer, new_guess, parameters_and_solved_vars)

    new_residuals = fnj.func_buffer
    # new_residuals = f(new_guess)

    # sol_cache = fnj.lu_buffer

    # ∇ = copy(fnj.jac_buffer)
    # copy!(sol_cache.A, fnj.jac_buffer)

    # if ∇ isa SparseMatrixCSC
    #     prob = 𝒮.LinearProblem(∇, new_guess, 𝒮.UMFPACKFactorization())
    # else
        # prob = 𝒮.LinearProblem(∇, new_guess)#, 𝒮.CholeskyFactorization)
    # end

    # sol_cache = 𝒮.init(prob)

    # ∇ = Array{T,2}(undef, length(new_guess), length(new_guess))

    # prep = 𝒟.prepare_jacobian(f, backend, new_guess)

    # largest_step = zero(T) + 1
    # largest_residual = zero(T) + 1

    rel_xtol_reached = 1.0
    rel_ftol_reached = 1.0
    new_residuals_norm = 1.0
    guess_update_norm = 1.0
    # init_residuals_norm = ℒ.norm(new_residuals)
    iters = [0,0]
    # resnorm = 1.0
    # relresnorm = 1.0
fnj.jac_buffer .= 0 
    new_guess = [1.1,1.9,18,.8,.3]

    # fnj.jac(fnj.jac_buffer, new_guess, parameters_and_solved_vars)

    # fnj.func(fnj.func_buffer, new_guess, parameters_and_solved_vars)

    # u = fnj.jac_buffer \ fnj.func_buffer

    # new_guess -= u

    for iter in 1:iterations
    # while iter < iterations
        fnj.jac(fnj.jac_buffer, new_guess, parameters_and_solved_vars)
        # println("jac_buffer: $(ℒ.norm(fnj.jac_buffer))")
        # if sol_cache.A isa SparseMatrixCSC
        #     copy!(sol_cache.A.nzval, fnj.jac_buffer.nzval)
        # else
        #     # copy!(sol_cache.A, fnj.jac_buffer)
        #     sol_cache.A .= fnj.jac_buffer
        # end
        # 𝒟.jacobian!(f, ∇, prep, backend, new_guess)

        # old_residuals_norm = ℒ.norm(new_residuals)

        # old_residuals = copy(new_residuals)

        fnj.func(fnj.func_buffer, new_guess, parameters_and_solved_vars)

        copy!(new_residuals, fnj.func_buffer)
        # new_residuals = f(new_guess)

        finn = has_nonfinite(new_residuals)

        if finn
            # println("GN not finite after $iter iteration; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
            rel_xtol_reached = 1.0
            rel_ftol_reached = 1.0
            new_residuals_norm = 1.0
            break 
        end
        
        if new_residuals_norm < ftol || rel_xtol_reached < rel_xtol || guess_update_norm < xtol # || rel_ftol_reached < rel_ftol
            new_guess_norm = ℒ.norm(new_guess)

            old_residuals_norm = new_residuals_norm

            new_residuals_norm = ℒ.norm(new_residuals)
        
            # sol_cache.A = ∇
            sol_cache.A .= fnj.jac_buffer
            sol_cache.b = new_residuals
            𝒮.solve!(sol_cache)
            # u = fnj.jac_buffer \ new_residuals
            u = sol_cache.u
            guess_update_norm = ℒ.norm(u)
    
            ℒ.axpy!(-1, u, new_guess)
    
            iters[1] += 1
            iters[2] += 1

            println("GN worked with $(iter+1) iterations - xtol ($xtol): $guess_update_norm; ftol ($ftol): $new_residuals_norm; rel_xtol ($rel_xtol): $rel_xtol_reached")# rel_ftol: $rel_ftol_reached")
            break
        end

        new_guess_norm = ℒ.norm(new_guess)

        old_residuals_norm = new_residuals_norm

        new_residuals_norm = ℒ.norm(new_residuals)
        
        # if iter > 5 && ℒ.norm(rel_xtol_reached) > sqrt(rel_xtol) && new_residuals_norm > old_residuals_norm
        #     # println("GN: $iter, Norm increase")
        #     break
        # end
        # if resnorm < ftol # && iter > 4
        #     println("GN worked with $iter iterations - norm: $resnorm; relative norm: $relresnorm")
        #     return undo_transform(new_guess,transformation_level), (iter, zero(T), zero(T), resnorm) # f(undo_transform(new_guess,transformation_level)))
        # end
        # println(sol_cache.b)
        # sol_cache.A = ∇
        sol_cache.A .= fnj.jac_buffer
        sol_cache.b = new_residuals

        # u = fnj.jac_buffer \ new_residuals
        # println(sol_cache.A)
        # println(sol_cache.b)
        𝒮.solve!(sol_cache)
        u = sol_cache.u
        # println(sol_cache.u)
        copy!(guess_update, u)

        guess_update_norm = ℒ.norm(guess_update)

        ℒ.axpy!(-1, guess_update, new_guess)

        finn = has_nonfinite(new_guess)

        if finn
            println("GN not finite after $iter iteration; - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")  # rel_ftol: $rel_ftol_reached; 
            rel_xtol_reached = 1.0
            rel_ftol_reached = 1.0
            new_residuals_norm = 1.0
            # iters = [iter,iter]
            break 
        end
        
        minmax!(new_guess, lower_bounds, upper_bounds)

        rel_xtol_reached = guess_update_norm / max(new_guess_norm, ℒ.norm(new_guess))
        # rel_ftol_reached = new_residuals_norm / max(eps(),init_residuals_norm)
        
        iters[1] += 1
        iters[2] += 1

        # println("GN: $(iters[1]) iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    end

    # if iters[1] == iterations
    #     println("GN failed to converge - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")#; rel_ftol: $rel_ftol_reached")
    # else
        println("GN converged after $(iters[1]) iterations - rel_xtol: $rel_xtol_reached; ftol: $new_residuals_norm")
    # end

    # best_guess = undo_transform(new_guess,transformation_level)
    
    # return new_guess, (iters[1], iters[2], rel_xtol_reached, new_residuals_norm)
# end
