# MacroModelling.jl

`MacroModelling.jl` - fast prototyping of dynamic stochastic general equilibrium (DSGE) models

`MacroModelling.jl` currently supports dicsrete-time DSGE models and the timing of a variable reflects when the variable is decided (end of period for stock variables).

As of now `MacroModelling.jl` can:
- parse a model written with user friendly syntax (variables are followed by time indices `...[2], [1], [0], [-1], [-2]...`, or `[x]` for shocks)
- (tries to) solve the model only knowing the model equations and parameter values (no steady state file needed)
- calculate first, second, and third order perturbation solutions using (forward) automatic differentiation (AD)
- calculate (generalised) impulse response functions, and simulate the model
- calibrate parameters using (non stochastic) steady state relationships
- match model moments
- estimate the model on data (kalman filter using first order perturbation)
- **differentiate** (forward AD) the model solution (first order perturbation), kalman filter loglikelihood, model moments, steady state, **with respect to the parameters**


`MacroModelling.jl` helps the modeller:
- Syntax makes variable and parameter definitions obsolete
- `MacroModelling.jl` applies symbolic and numerical tools to solve for the steady state (and mostly succeeds without much help)


```@repl
using MacroModelling

@model RBC_estim begin
    # c[0]^(-sigma)=beta/gammax*c(+1)^(-sigma)*(alpha*exp(z(+1))*(k/l(+1))^(alpha-1)+(1-delta))
    c[0]^(-sigma) = beta * c[1]^(-sigma) * (alpha * exp(z[1]) * (k[0] / l[1])^(alpha - 1) + (1 - delta))
    psi * c[0]^sigma * 1 / (1 - l[0]) = w[0]
    k[0] = (1 - delta) * k[-1] + invest[0]
    y[0] = invest[0] + c[0] + gshare * y[ss] * exp(ghat[0])
    y[0] = exp(z[0]) * k[-1]^alpha * l[0]^(1 - alpha)
    w[0] = (1 - alpha) * y[0] / l[0]
    r[0] = 4 * alpha * y[0] / k[-1]
    z[0] = rhoz * z[-1] + std_z * eps_z[x]
    ghat[0] = rhog * ghat[-1] + std_g * eps_g[x]
    c_obs[0] = log(c[0]) - log(c[-1])
    y_obs[0] = log(y[0]) - log(y[-1])
    g_obs[0] = ghat[0] - ghat[-1]
end

@parameters RBC_estim begin
    std_g  = 0.0105
    std_z  = 0.0068
    sigma  = 1
    alpha  = 0.33
    rhoz   = 0.97
    rhog   = 0.989
    gshare = 0.2038
    delta  = 0.024038461538461536
    beta   = 0.9923664122137404

    psi | l[ss] = .33
    w > 0
    l > 0
    y > 0
    k > 0
    c > 0
    invest > 0

    psi > 0
end

get_solution(RBC_estim)
```
