using SparseArrays
using MacroModelling: timings
using ForwardDiff, FiniteDifferences, Zygote
import Optim, LineSearches


@testset verbose = true "Basic model solution and std" begin
    Random.seed!(3)

    @model RBC_CME begin
        y[0]=A[0]*k[-1]^alpha
        1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
        1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
        R[0] * beta =(Pi[0]/Pibar)^phi_pi
        A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
        z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
        A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    end


    @parameters RBC_CME verbose = true begin
        # alpha | k[ss] / (4 * y[ss]) = cap_share
        # cap_share = 1.66
        alpha = .157

        # beta | R[ss] = R_ss
        # R_ss = 1.0035
        beta = .999

        # delta | c[ss]/y[ss] = 1 - I_K_ratio
        # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
        # I_K_ratio = .15
        delta = .0226

        # Pibar | Pi[ss] = Pi_ss
        # Pi_ss = 1.0025
        Pibar = 1.0008

        phi_pi = 1.5
        rhoz = .9
        std_eps = .0068
        rho_z_delta = .9
        std_z_delta = .005
    end

    show(RBC_CME) # test printout
    solution = get_solution(RBC_CME)

    RBC_CME # test printout
    @test isapprox(solution(:eps_z₍ₓ₎,:y), 0.009677838924324712, rtol = eps(Float32))


    momm = get_moments(RBC_CME)
    @test isapprox(momm[2][1], 0.015600269903198518, rtol = eps(Float32))
end


SS_and_pars, _ = RBC_CME.SS_solve_func(RBC_CME.parameter_values, RBC_CME, true)

get_irf(RBC_CME, algorithm = :third_order)

∇₁ = calculate_jacobian(RBC_CME.parameter_values, SS_and_pars, RBC_CME)
∇₂ = calculate_hessian(RBC_CME.parameter_values,SS_and_pars,RBC_CME)
∇₃ = calculate_third_order_derivatives(RBC_CME.parameter_values,SS_and_pars,RBC_CME)
#SS = get_steady_state(RBC_CME, derivatives = false)


T = timings([:R, :y], [:Pi, :c], [:k, :z_delta], [:A], [:A, :Pi, :c], [:A, :k, :z_delta], [:A, :Pi, :c, :k, :z_delta], [:A], [:k, :z_delta], [:A], [:delta_eps, :eps_z], [:A, :Pi, :R, :c, :k, :y, :z_delta], Symbol[], Symbol[], 2, 1, 3, 3, 5, 7, 2, [3, 6], [1, 2, 4, 5, 7], [1, 2, 4], [2, 3], [1, 5, 7], [1], [1], [5, 7], [5, 6, 1, 7, 3, 2, 4], [3, 4, 5, 1, 2])

first_order_solution, solved = calculate_first_order_solution(∇₁; T = T, explosive = false)# |> Matrix{Float32}

second_order_solution = calculate_second_order_solution(∇₁, 
∇₂, 
first_order_solution; 
T = T)

third_order_solution = calculate_third_order_solution(∇₁, 
∇₂, 
∇₃,
first_order_solution, 
second_order_solution; 
T = T)

@testset verbose = true "SS, derivatives of model at SS and solutions" begin
    NSSS =  [1.0
    1.0024019205374952
    1.003405325870413
    1.2092444352939415
    9.467573947982233
    1.42321160651834
    1.0]
    @test isapprox(SS_and_pars,NSSS,rtol = eps(Float32))

    jacobian2 = sparse(vec([ 2  3  2  3  1  5  7  4  3  4  2  3  5  2  5  1  5  6  7  1  5  6  6  7]),
    vec([ 1  2  3  3  4  4  4  5  6  6  7  7  7  8  8  9  10  10  11  12  12  13  14  15]),
    [ -0.01949762952125457
    0.8249811251976374
    0.6838672481452972
    0.683867248150105
    -1.42321160651834
    1.42321160651834
    1.0
    -1.5000000000190756
    -0.8241561440666456
    0.999
    -0.6838672481452971
    -0.6838672481452971
    -1.0
    0.0017360837926088356
    -1.0
    1.0
    -0.21396717122439846
    1.0
    -0.9
    -0.023601001001000967
    1.001001001001001
    -0.9
    -0.005
    -0.0068],7,15)
    @test isapprox(sparse(∇₁),jacobian2,rtol = eps(Float32))


    hessian2 = sparse(vec([ 2  2  3  3  3  2  3  2  3  3  2  1  5  4  3  3  2  3  2  2  2  5  1  5  5  1  5]),
    vec([ 3  8  17  18  21  31  32  33  33  36  38  57  57  65  77  78  97  97  106  108  113  147  169  169  175  177  177]),
    [  0.016123811656420906
    0.0017360837926088356
    -1.6460086683698225
    -0.6822285892902225
    0.822181329844946
    0.016123811656420906
    -0.6822285892902225
    -1.1310653631067793
    -1.1310653631147312
    0.6815463606961406
    -0.0014356764785829514
    -0.023601001001000967
    0.023601001001000967
    -0.748202876155088
    0.822181329844946
    0.6815463606961406
    1.1310653631067793
    1.1310653631067793
    0.0017360837926088356
    -0.0014356764785829514
    -0.00033795378281254373
    -0.0226
    -0.023601001001000967
    0.023601001001000967
    -0.0226
    0.0021014511165327685
    -0.0021014511165327685],7,225)
    @test isapprox(∇₂,hessian2,rtol = eps(Float32))


    third_order_derivatives2 = sparse(vec([ 2  2  2  2  3  3  3  3  3  3  3  3  2  2  3  3  3  2  3  2  3  3  2  3  3  2  2  2  1  5  4  3  3  3  3  2  3  2  2  2  2  2  2  2  2  1  5  1  5  1  5]),
    vec([ 33  38  108  113  242  243  246  257  258  261  302  303  453  458  467  468  471  481  482  483 483  486  488  527  528  556  558  563  852  852  965  1142  1143  1157  1158  1447  1447  1578  1583  1606  1608  1613  1681  1683  1688  2532  2532  2644  2644  2652  2652]),
    [ -0.026667580492113735
    -0.0014356764785829514
    -0.0014356764785829514
    -0.00033795378281254373
        4.926193679339384
        1.3611877138551502
    -1.6404224952084816
        1.3611877138551502
        1.1283551437214383
    -0.6799132630658674
    -1.6404224952084818
    -0.6799132630658674
    -0.026667580492113735
    -0.0014356764785829514
        1.3611877138551502
        1.1283551437214383
    -0.6799132630658674
    -0.026667580492113735
        1.1283551437214383
        2.806046478514929
        2.8060464785346566
        -1.1272267885697917
        0.00237450169160211
        -0.6799132630658674
        -1.1272267885697917
        -0.0014356764785829514
        0.00237450169160211
        0.0002794751606447495
        0.0021014511165327685
        -0.0021014511165327685
        0.3732050292530844
        -1.6404224952084816
        -0.6799132630658674
        -0.6799132630658674
        -1.1272267885697917
        -2.8060464785149284
        -2.8060464785149284
        -0.0014356764785829514
        -0.00033795378281254373
        -0.0014356764785829514
        0.00237450169160211
        0.0002794751606447495
        -0.00033795378281254373
        0.0002794751606447495
        0.00010148350673731277
        0.0021014511165327685
        -0.0021014511165327685
        0.0021014511165327685
        -0.0021014511165327685
        -0.0004090778090616675
        0.0004090778090616675],7,3375)
    @test isapprox(∇₃,third_order_derivatives2,rtol = eps(Float32))

    first_order_solution2 = [ 0.9         5.41234e-16  -6.41848e-17   0.0           0.0068
    0.0223801  -0.00364902    0.00121336    6.7409e-6     0.000169094
    0.0336038  -0.00547901    0.00182186    1.01215e-5    0.000253895
    0.28748     0.049647     -0.0660339    -0.000366855   0.00217207
    0.99341     0.951354     -0.126537     -0.000702981   0.00750577
    1.28089     0.023601     -9.12755e-17  -0.0           0.00967784
    0.0         0.0           0.9           0.005        -0.0]

    @test isapprox(first_order_solution,first_order_solution2,rtol = 1e-6)



    @test isapprox(second_order_solution[2,1], -0.006642814796744731,rtol = eps(Float32))

    @test isapprox(third_order_solution[2,1],0.003453249193794699, rtol = 1e-5)
end


@testset verbose = true "Higher order IRFs and GIRFs" begin
    first_order_state_update = function(state::Vector{Float64}, shock::Vector{Float64}) first_order_solution * [state[T.past_not_future_and_mixed_idx]; shock] end

    Tz = [first_order_solution[:,1:T.nPast_not_future_and_mixed] zeros(T.nVars) first_order_solution[:,T.nPast_not_future_and_mixed+1:end]]

    second_order_state_update = function(state::Vector{Float64}, shock::Vector{Float64})
        aug_state = [state[T.past_not_future_and_mixed_idx]
                    1
                    shock]
        return Tz * aug_state + second_order_solution * kron(aug_state, aug_state) / 2
    end

    third_order_state_update = function(state::Vector{Float64}, shock::Vector{Float64})
        aug_state = [state[T.past_not_future_and_mixed_idx]
                        1
                        shock]

        return Tz * aug_state + second_order_solution * kron(aug_state, aug_state) / 2 + third_order_solution * kron(kron(aug_state,aug_state),aug_state) / 6
    end


    iirrff = irf(first_order_state_update, zeros(T.nVars), zeros(T.nVars), T)

    @test isapprox(iirrff[4,1,:],[ -0.00036685520477089503
    0.0021720718769730014],rtol = eps(Float32))
    ggiirrff = girf(first_order_state_update, zeros(T.nVars), zeros(T.nVars), T)
    @test isapprox(iirrff[4,1,:],ggiirrff[4,1,:],rtol = eps(Float32))


    SSS_delta = RBC_CME.solution.non_stochastic_steady_state[1:length(RBC_CME.var)] - RBC_CME.solution.perturbation.second_order.stochastic_steady_state

    ggiirrff2 = girf(second_order_state_update, SSS_delta, zeros(T.nVars), T, draws = 1000,warmup_periods = 100)
    @test isapprox(ggiirrff2[4,1,:],[-0.0003668849861768406
    0.0021711333455274096],rtol = 1e-3)

    iirrff2 = irf(second_order_state_update, zeros(T.nVars), zeros(T.nVars), T)
    @test isapprox(iirrff2[4,1,:],[-0.0004547347878067665, 0.0020831426377533636],rtol = 1e-6)


    SSS_delta = RBC_CME.solution.non_stochastic_steady_state[1:length(RBC_CME.var)] - RBC_CME.solution.perturbation.third_order.stochastic_steady_state

    ggiirrff3 = girf(third_order_state_update, SSS_delta, zeros(T.nVars), T,draws = 1000,warmup_periods = 100)
    @test isapprox(ggiirrff3[4,1,:],[ -0.00036686142588429404
    0.002171120660323429],rtol = 1e-3)

    iirrff3 = irf(third_order_state_update, zeros(T.nVars), zeros(T.nVars), T)
    @test isapprox(iirrff3[4,1,:],[-0.00045473149068020854, 0.002083198241302615], rtol = 1e-6)
end


@testset verbose = true "NSSS and std derivatives" begin
    # derivatives of paramteres wrt standard deviations
    stdev_deriv = ForwardDiff.jacobian(x -> get_statistics(RBC_CME, x, parameters = RBC_CME.parameters, standard_deviation = RBC_CME.var)[1], RBC_CME.parameter_values)
    stdev_deriv[9]
    @test isapprox(stdev_deriv[5,6],1.3135107627695757, rtol = 1e-6)

    # derivatives of paramteres wrt non stochastic steady state
    nsss_deriv = ForwardDiff.jacobian(x -> get_statistics(RBC_CME, x, parameters = RBC_CME.parameters, non_stochastic_steady_state = RBC_CME.var)[1], RBC_CME.parameter_values)
    @test isapprox(nsss_deriv[4,1],3.296074644820076, rtol = 1e-6)
end

@testset verbose = true "Method of moments" begin
    # Method of moments: with varying steady states and derivatives of steady state numerical solved_vars
    sol = Optim.optimize(x -> sum(abs2, get_statistics(RBC_CME, x, parameters = [RBC_CME.parameters[1]], standard_deviation = [RBC_CME.var[5]])[1] - [.21]),
    [0], [1], [.16], 
    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

    @test isapprox(get_statistics(RBC_CME, sol.minimizer, parameters = [RBC_CME.parameters[1]], standard_deviation = [RBC_CME.var[5]])[1] ,[.21], atol = 1e-6)

    # multiple parameter inputs and targets
    sol = Optim.optimize(x -> sum(abs2,get_statistics(RBC_CME, x, parameters = RBC_CME.parameters[1:2], standard_deviation = RBC_CME.var[[2,5]])[1] - [.0008,.21]),
    [0,0], [1,1], [.006,.16], 
    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

    @test isapprox(get_statistics(RBC_CME, sol.minimizer, parameters = RBC_CME.parameters[1:2], standard_deviation = RBC_CME.var[[2,5]])[1], [.0008,.21], atol=1e-6)

    # function combining targets for SS and St.Dev.
    function get_variances_optim(x)
        out = get_statistics(RBC_CME, x, parameters = RBC_CME.parameters[1:2], non_stochastic_steady_state = [RBC_CME.var[6]], standard_deviation = [RBC_CME.var[5]])
        sum(abs2,[out[1][1] - 1.45, out[2][1] - .2])
    end
    out = get_variances_optim([.157,.999])

    out = get_statistics(RBC_CME, [.157,.999], parameters = RBC_CME.parameters[1:2], non_stochastic_steady_state = [RBC_CME.var[6]], standard_deviation = [RBC_CME.var[5]])
    sum(abs2,[out[1][1] - 1.4, out[2][1] - .21])

    sol = Optim.optimize(x -> get_variances_optim(x),
    [0,0.95], [1,1], [.16, .999], 
    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

    @test isapprox(get_statistics(RBC_CME, sol.minimizer, parameters = RBC_CME.parameters[1:2], non_stochastic_steady_state = [RBC_CME.var[6]], standard_deviation = [RBC_CME.var[5]]),[[1.45],[.2]],atol = 1e-6)




    # function combining targets for SS, St.Dev., and parameter
    function get_variances_optim2(x)
        out = get_statistics(RBC_CME, x, parameters = RBC_CME.parameters[1:3], non_stochastic_steady_state = [RBC_CME.var[6]], standard_deviation = [RBC_CME.var[5]])
        sum(abs2,[out[1][1] - 1.45, out[2][1] - .2, x[3] - .02])
    end
    out = get_variances_optim2([.157,.999,.022])

    sol = Optim.optimize(x -> get_variances_optim2(x),
    [0,0.95,0], [1,1,1], [.16, .999,.022], 
    Optim.Fminbox(Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3))); autodiff = :forward)

    @test isapprox([get_statistics(RBC_CME, sol.minimizer, parameters = RBC_CME.parameters[1:3], non_stochastic_steady_state = [RBC_CME.var[6]], standard_deviation = [RBC_CME.var[5]])
    sol.minimizer[3]],[[1.45],[.2],.02],atol = 1e-6)
end

RBC_CME = nothing

@testset verbose = true "Kalman filter and gradient" begin
    Random.seed!(3)

    @model RBC_CME begin
        y[0]=A[0]*k[-1]^alpha
        1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
        1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
        R[0] * beta =(Pi[0]/Pibar)^phi_pi
        A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
        z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
        A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
    end


    @parameters RBC_CME verbose = true begin
        # alpha | k[ss] / (4 * y[ss]) = cap_share
        # cap_share = 1.66
        alpha = .157

        # beta | R[ss] = R_ss
        # R_ss = 1.0035
        beta = .999

        # delta | c[ss]/y[ss] = 1 - I_K_ratio
        # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
        # I_K_ratio = .15
        delta = .0226

        # Pibar | Pi[ss] = Pi_ss
        # Pi_ss = 1.0025
        Pibar = 1.0008

        phi_pi = 1.5
        rhoz = .9
        std_eps = .0068
        rho_z_delta = .9
        std_z_delta = .005
    end

    get_solution(RBC_CME)

    get_irf(RBC_CME; parameters = RBC_CME.parameter_values)

    forw_grad = ForwardDiff.gradient(x->get_irf(RBC_CME, x)[4,1,2],Float64.(RBC_CME.parameter_values))

    fin_grad = FiniteDifferences.grad(central_fdm(2,1),x->get_irf(RBC_CME, x)[4,1,2],RBC_CME.parameter_values)[1]

    @test isapprox(forw_grad,fin_grad,rtol = 1e-5)



    data = simulate(RBC_CME)[:,:,1]
    observables = [:c,:k]
    @test isapprox(420.25039827148197,calculate_kalman_filter_loglikelihood(RBC_CME,data(observables),observables),rtol = 1e-5)

    forw_grad = ForwardDiff.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))
    reverse_grad = Zygote.gradient(x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),Float64.(RBC_CME.parameter_values))[1]

    fin_grad = FiniteDifferences.grad(central_fdm(4,1),x->calculate_kalman_filter_loglikelihood(RBC_CME, data(observables), observables; parameters = x),RBC_CME.parameter_values)[1]

    @test isapprox(forw_grad,fin_grad, rtol = 1e-6)
    @test isapprox(forw_grad,reverse_grad, rtol = eps(Float32))

    RBC_CME = nothing
end

# observables = [:c,:k,:Pi]
# calculate_kalman_filter_loglikelihood(RBC_CME,data(observables), observables)

# irfs = get_irf(RBC_CME; parameters = RBC_CME.parameter_values)




@testset verbose = true "SS + SSS derivatives" begin
    # Test diff of SS and SSS
    include("models/RBC_CME_calibration_equations_and_parameter_definitions.jl")

    SSdiff = get_SS(m)

    SSSdiff2 = get_SSS(m)

    SSSdiff3 = get_SSS(m, algorithm = :third_order)

    𝓂 = m

    parameters = copy(m.parameter_values)

    SSfinitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
            x -> collect(get_SS(m; parameters = x, derivatives = false)), 
            parameters)[1]

    @test isapprox(SSfinitediff, SSdiff[:,2:end], rtol = 1e-8)


    SSS2finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
            x -> collect(get_SSS(m; parameters = x, derivatives = false)), 
            parameters)[1]

    @test isapprox(SSS2finitediff, SSSdiff2[:,2:end], rtol = 1e-8)


    SSS3finitediff = FiniteDifferences.jacobian(central_fdm(4,1), 
            x -> collect(get_SSS(m; parameters = x, derivatives = false, algorithm = :third_order)), 
            parameters)[1]

    @test isapprox(SSS3finitediff, SSSdiff3[:,2:end], rtol = 1e-8)
end
m = nothing
𝓂 = nothing
