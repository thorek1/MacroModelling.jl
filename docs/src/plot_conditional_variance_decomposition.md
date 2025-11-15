# Conditional Variance Decomposition

The `plot_conditional_variance_decomposition` function visualizes the forecast error variance decomposition (FEVD), showing how much of the variance in forecast errors for each variable can be attributed to different shocks over various forecast horizons.

First, define and load a model:

```julia
@model Smets_Wouters_2007_linear begin
    a[0] = calfa * rkf[0] + (1 - calfa) * wf[0]
    zcapf[0] = rkf[0] * 1 / (czcap / (1 - czcap))
    rkf[0] = wf[0] + labf[0] - kf[0]
    kf[0] = zcapf[0] + kpf[-1]
    invef[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pkf[0] * 1 / (csadjcost * cgamma ^ 2) + invef[-1] + invef[1] * cgamma * cbetabar)
    pkf[0] = b[0] * (1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma)))) - rrf[0] + rkf[1] * (crk / (crk + (1 - ctou))) + pkf[1] * ((1 - ctou) / (crk + (1 - ctou)))
    cf[0] = b[0] + cf[-1] * chabb / cgamma / (1 + chabb / cgamma) + cf[1] * 1 / (1 + chabb / cgamma) + (labf[0] - labf[1]) * ((csigma - 1) * cwhlc / (csigma  *(1 + chabb / cgamma))) - rrf[0] * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))
    yf[0] = g[0] + cf[0] * ccy + invef[0] * ciy + zcapf[0] * crkky
    yf[0] = cfc * (a[0] + calfa * kf[0] + (1 - calfa) * labf[0])
    wf[0] = labf[0] * csigl + cf[0] * 1 / (1 - chabb / cgamma) - cf[-1] * chabb / cgamma / (1 - chabb / cgamma)
    kpf[0] = kpf[-1] * (1 - cikbar) + invef[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar
    mc[0] = calfa * rk[0] + (1 - calfa) * w[0] - a[0]
    zcap[0] = 1 / (czcap / (1 - czcap)) * rk[0]
    rk[0] = w[0] + lab[0] - k[0]
    k[0] = zcap[0] + kp[-1]
    inve[0] = qs[0] + 1 / (1 + cgamma * cbetabar) * (pk[0] * 1 / (csadjcost * cgamma ^ 2) + inve[-1] + inve[1] * cgamma * cbetabar)
    pk[0] = pinf[1] - r[0] + b[0] * 1 / ((1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))) + rk[1] * (crk / (crk + (1 - ctou))) + pk[1] * ((1 - ctou) / (crk + (1 - ctou)))
    c[0] = b[0] + c[-1] * chabb / cgamma / (1 + chabb / cgamma) + c[1] * 1 / (1 + chabb / cgamma) + 
    (lab[0] - lab[1]) * ((csigma - 1) * cwhlc / (csigma * (1 + chabb / cgamma))) - (r[0] - pinf[1]) * (1 - chabb / cgamma) / (csigma * (1 + chabb / cgamma))
    y[0] = g[0] + c[0] * ccy + inve[0] * ciy + zcap[0] * crkky
    y[0] = cfc * (a[0] + calfa * k[0] + (1 - calfa) * lab[0])
    pinf[0] = spinf[0] + 1 / (1 + cindp * cgamma * cbetabar) * (cindp * pinf[-1] + pinf[1] * cgamma * cbetabar + mc[0] * (1 - cprobp) * (1 - cprobp * cgamma * cbetabar) / cprobp / (1 + (cfc - 1) * curvp))
    w[0] = sw[0] + w[-1] * 1 / (1 + cgamma * cbetabar) + w[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + pinf[-1] * cindw / (1 + cgamma * cbetabar) - pinf[0] * (1 + cindw * cgamma * cbetabar) / (1 + cgamma * cbetabar) + pinf[1] * cgamma * cbetabar / (1 + cgamma * cbetabar) + (csigl * lab[0] + c[0] * 1 / (1 - chabb / cgamma) - c[-1] * chabb / cgamma / (1 - chabb / cgamma) - w[0]) * 1 / (1 + (clandaw - 1) * curvw) * (1 - cprobw) * (1 - cprobw * cgamma * cbetabar) / (cprobw * (1 + cgamma * cbetabar))
    r[0] = pinf[0] * crpi * (1 - crr) + (1 - crr) * cry * (y[0] - yf[0]) + crdy * (y[0] - yf[0] - y[-1] + yf[-1]) + crr * r[-1] + ms[0]
    a[0] = crhoa * a[-1] + z_ea * ea[x]
    b[0] = crhob * b[-1] + z_eb * eb[x]
    g[0] = crhog * g[-1] + z_eg * eg[x] + z_ea * ea[x] * cgy
    qs[0] = crhoqs * qs[-1] + z_eqs * eqs[x]
    ms[0] = crhoms * ms[-1] + z_em * em[x]
    spinf[0] = crhopinf * spinf[-1] + epinfma[0] - cmap * epinfma[-1]
    epinfma[0] = z_epinf * epinf[x]
    sw[0] = crhow * sw[-1] + ewma[0] - cmaw * ewma[-1]
    ewma[0] = z_ew * ew[x]
    kp[0] = kp[-1] * (1 - cikbar) + inve[0] * cikbar + qs[0] * csadjcost * cgamma ^ 2 * cikbar
    dy[0] = ctrend + y[0] - y[-1]
    dc[0] = ctrend + c[0] - c[-1]
    dinve[0] = ctrend + inve[0] - inve[-1]
    pinfobs[0] = constepinf + pinf[0]
    robs[0] = r[0] + conster
    dwobs[0] = ctrend + w[0] - w[-1]
    labobs[0] = lab[0] + constelab
end

@parameters Smets_Wouters_2007_linear begin
    ctou = .025
    clandaw = 1.5
    cg = 0.18
    curvp = 10
    curvw = 10
    calfa = .24
    csigma = 1.5
    cfc = 1.5
    cgy = 0.51
    csadjcost = 6.0144
    chabb = 0.6361
    cprobw = 0.8087
    csigl = 1.9423
    cprobp = 0.6
    cindw = 0.3243
    cindp = 0.47
    czcap = 0.2696
    crpi = 1.488
    crr = 0.8762
    cry = 0.0593
    crdy = 0.2347
    crhoa = 0.9977
    crhob = 0.5799
    crhog = 0.9957
    crhoqs = 0.7165
    crhoms = 0
    crhopinf = 0
    crhow = 0
    cmap = 0
    cmaw = 0
    constelab = 0
    constepinf = 0.7
    constebeta = 0.7420
    ctrend = 0.3982
    z_ea	= 0.4618
    z_eb	= 1.8513
    z_eg	= 0.6090
    z_em	= 0.2397
    z_ew	= 0.2089
    z_eqs	= 0.6017
    z_epinf	= 0.1455
    cpie 	= 1 + constepinf / 100         							# gross inflation rate
    cgamma 	= 1 + ctrend / 100          							# gross growth rate
    cbeta 	= 1 / (1 + constebeta / 100)    						# discount factor
    clandap = cfc                									# fixed cost share/gross price markup
    cbetabar= cbeta * cgamma ^ (-csigma)   							# growth-adjusted discount factor in Euler equation
    cr 		= cpie / cbetabar  										# steady state gross real interest rate
    crk 	= 1 / cbetabar - (1 - ctou) 							# steady state rental rate
    cw 		= (calfa ^ calfa * (1 - calfa) ^ (1 - calfa) / (clandap * crk ^ calfa)) ^ (1 / (1 - calfa))	# steady state real wage
    cikbar 	= 1 - (1 - ctou) / cgamma								# (1-k_1) in equation LOM capital, equation (8)
    cik 	= cikbar * cgamma										# i_k: investment-capital ratio
    clk 	= (1 - calfa) / calfa * crk / cw						# labor to capital ratio
    cky 	= cfc * clk ^ (calfa - 1)								# k_y: steady state output ratio
    ciy 	= cik * cky												# investment-output ratio
    ccy 	= 1 - cg - cik * cky									# consumption-output ratio
    crkky 	= crk * cky												# z_y=R_{*}^k*k_y
    cwhlc 	= (1 / clandaw) * (1 - calfa) / calfa * crk * cky / ccy	# W^{h}_{*}*L_{*}/C_{*} used in c_2 in equation (2)
    conster = (cr - 1) * 100										# steady state federal funds rate ($\bar r$)
end
```

Calling the conditional variance decomposition plot function:

```julia
plot_conditional_variance_decomposition(Smets_Wouters_2007_linear)
```

![Smets and Wouters 2007 FEVD](../assets/fevd__Smets_Wouters_2007_linear__2.png)

This creates conditional variance decomposition plots showing the contribution of each shock to the forecast error variance of each variable across different forecast horizons.

The vertical axis shows the share of the shocks variance contribution, and the horizontal axis the period of the variance decomposition. The stacked bars represent each shocks variance contribution at a specific time horizon.

Note that if occasionally binding constraints are present in the model, they are not taken into account here.

The same function can be called using different names. For example: `plot_fevd`, or `plot_forecast_error_variance_decomposition`. Going forward, `plot_fevd` will be used for brevity.

## Periods Argument

The `periods` argument (default: `40`, type: `Int`) specifies the number of forecast horizons to include in the variance decomposition. This determines how far into the future the decomposition extends.

```julia
# Show variance decomposition up to 12 periods ahead
plot_fevd(Smets_Wouters_2007_linear, periods = 12)
```

![Smets and Wouters 2007 FEVD - 12 periods](../assets/short_period__Smets_Wouters_2007_linear__2.png)

## Variables to Plot

The `variables` argument (default: `:all`) specifies for which variables to show results. Variable names can be specified as either a `Symbol` or `String` (e.g. `:y` or `"y"`), or `Tuple`, `Matrix` or `Vector` of `String` or `Symbol`. Any variables not part of the model will trigger a warning. `:all_excluding_auxiliary_and_obc` includes all variables except auxiliary variables and those related to occasionally binding constraints (OBC). `:all_excluding_obc` includes all variables except those related to occasionally binding constraints. `:all` includes all variables.

Specific variables can be selected to plot. The following example selects output (`Y`), consumption (`c`), investment (`inve`), inflation (`pinf`), wages (`w`), and labor (`lab`) using a `Vector` of `Symbol`s:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = [:inve, :c, :y, :pinf, :w, :lab])
```

![Smets and Wouters 2007 FEVD - selected variables](../assets/var_select__Smets_Wouters_2007_linear__1.png)

The plot now displays only one plot with the six selected variables (sorted alphabetically).

The same can be done using a `Tuple`:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = (:inve, :c, :y, :pinf, :w, :lab))
```

a `Matrix`:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = [:inve :c :y :pinf :w :lab])
```

or providing the variable names as `String`s:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = ["inve", "c", "y", "pinf", "w", "lab"])
```

or a single variable as a `Symbol`:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = :inve)
```

or as a `String`:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = "inve")
```

Then there are some predefined options:

`:all_excluding_auxiliary_and_obc` plots all variables except auxiliary variables and those used to enforce occasionally binding constraints (OBC).

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = :all_excluding_auxiliary_and_obc)
```

`:all_excluding_obc` plots all variables except those used to enforce occasionally binding constraints (OBC).

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = :all_excluding_obc)
```

To see auxiliary variables, use a model that defines them. The `FS2000` model can be used:

```julia
@model FS2000 begin
    dA[0] = exp(gam + z_e_a  *  e_a[x])
    log(m[0]) = (1 - rho) * log(mst)  +  rho * log(m[-1]) + z_e_m  *  e_m[x]
    - P[0] / (c[1] * P[1] * m[0]) + bet * P[1] * (alp * exp( - alp * (gam + log(e[1]))) * k[0] ^ (alp - 1) * n[1] ^ (1 - alp) + (1 - del) * exp( - (gam + log(e[1])))) / (c[2] * P[2] * m[1])=0
    W[0] = l[0] / n[0]
    - (psi / (1 - psi)) * (c[0] * P[0] / (1 - n[0])) + l[0] / n[0] = 0
    R[0] = P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ ( - alp) / W[0]
    1 / (c[0] * P[0]) - bet * P[0] * (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) / (m[0] * l[0] * c[1] * P[1]) = 0
    c[0] + k[0] = exp( - alp * (gam + z_e_a  *  e_a[x])) * k[-1] ^ alp * n[0] ^ (1 - alp) + (1 - del) * exp( - (gam + z_e_a  *  e_a[x])) * k[-1]
    P[0] * c[0] = m[0]
    m[0] - 1 + d[0] = l[0]
    e[0] = exp(z_e_a  *  e_a[x])
    y[0] = k[-1] ^ alp * n[0] ^ (1 - alp) * exp( - alp * (gam + z_e_a  *  e_a[x]))
    gy_obs[0] = dA[0] * y[0] / y[-1]
    gp_obs[0] = (P[0] / P[-1]) * m[-1] / dA[0]
    log_gy_obs[0] = log(gy_obs[0])
    log_gp_obs[0] = log(gp_obs[0])
end

@parameters FS2000 begin
    alp     = 0.356
    bet     = 0.993
    gam     = 0.0085
    mst     = 1.0002
    rho     = 0.129
    psi     = 0.65
    del     = 0.01
    z_e_a   = 0.035449
    z_e_m   = 0.008862
end
```

Since both `c` and `P` appear in `t+2`, they generate auxiliary variables in the model. Plotting the policy functions for all variables excluding OBC-related ones means auxiliary variables are shown (same for the default `:all` option since there are no OBCs in this model):

```julia
plot_fevd(FS2000,
    variables = :all_excluding_obc)
```

![FS2000 FEVD - including auxiliary variables](../assets/aux__FS2000__1.png)

Both `c` and `P` appear twice: once as the variable itself and once as an auxiliary variable with the `ᴸ⁽¹⁾` superscript, representing the value of the variable in `t+1` as expected in `t`.

`:all` (default) plots all variables including auxiliary variables. Since OBCs are not considered with FEVD, variables used to enforce occasionally binding constraints (OBC) are not included.

```julia
plot_fevd(FS2000)
```

## Parameter Values

When no parameters are provided, the solution uses the previously defined parameter values. Parameters can be provided as a `Vector` of values, or as a `Vector` or `Tuple` of `Pair`s mapping parameter `Symbol`s or `String`s to values. The solution is recalculated when new parameter values differ from the previous ones.

Start by changing the discount factor `z_eg` to 1:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    parameters = :z_eg => 1)
```

![Smets and Wouters 2007 FEVD - different parameter values](../assets/param_change__Smets_Wouters_2007_linear__2.png)

The shock contributions changed as a result of changing the discount factor.

Multiple parameters can also be changed simultaneously. This example changes `z_eg` to 1.5 and `crpi` to 1.75 using a `Tuple` of `Pair`s and define the variables with `Symbol`s:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    parameters = (:z_eg => 1.5, :crpi => 1.75))
```

![Smets and Wouters 2007 FEVD - multiple parameter changes](../assets/param_change_2__Smets_Wouters_2007_linear__2.png)

A `Vector` of `Pair`s can also be used:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    parameters = [:z_eg => 1.5, :crpi => 1.75])
```

Alternatively, use a `Vector` of parameter values in the order they were defined in the model. To obtain them:

```julia
params = get_parameters(Smets_Wouters_2007_linear, values = true)

param_vals = [p[2] for p in params]

plot_fevd(Smets_Wouters_2007_linear,
    parameters = param_vals)
```

## Plot Attributes

The `plot_attributes` argument (default: `Dict()`, type: `Dict`) accepts a dictionary of attributes passed on to the plotting function. See the Plots.jl documentation for details.

The color palette can be customized using the `plot_attributes` argument. The following example defines a custom color palette (inspired by the European Commission's economic reports) to plot the FEVD using the `Smets_Wouters_2007_linear` model.
First, define the custom color palette using hex color codes:

```julia
ec_color_palette =
[
    "#FFD724",  # "Sunflower Yellow"
    "#353B73",  # "Navy Blue"
    "#2F9AFB",  # "Sky Blue"
    "#B8AAA2",  # "Taupe Grey"
    "#E75118",  # "Vermilion"
    "#6DC7A9",  # "Mint Green"
    "#F09874",  # "Coral"
    "#907800"   # "Olive"
]
```

Next, plot the FEVD using the custom color palette:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:palette => ec_color_palette))
```

![Smets and Wouters 2007 FEVD - custom color palette](../assets/color_palette__Smets_Wouters_2007_linear__2.png)

The colors of the shock contributions now follow the custom color palette.

Other attributes such as the font family can also be modified (see [here](https://github.com/JuliaPlots/Plots.jl/blob/v1.41.1/src/backends/gr.jl#L61) for font options):

```julia
plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:fontfamily => "computer modern"))
```

![Smets and Wouters 2007 FEVD - custom font](../assets/font_family__Smets_Wouters_2007_linear__2.png)

All text in the plot now uses the Computer Modern font. Note that font rendering inherits the constraints of the plotting backend (GR in this case).

Here is another example that customizes the alpha (transparency) of the filled areas in the FEVD plots:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    plot_attributes = Dict(:fillalpha => .5))
```

![Smets and Wouters 2007 FEVD - custom fill alpha](../assets/fill_alpha__Smets_Wouters_2007_linear__2.png)

## Plots Per Page

The `plots_per_page` argument (default: `9`, type: `Int`) controls the number of subplots per page. When the number of variables exceeds this value, multiple pages are created.
The following example selects 6 variables and sets `plots_per_page` to 4, resulting in 2 pages with the first page having 4 subplots and the second page having 2 subplots:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    variables = [:inve, :c, :y, :pinf, :w, :lab],
    plots_per_page = 4)
```

![Smets and Wouters 2007 FEVD - 4 plots per page](../assets/four_per_page__Smets_Wouters_2007_linear__1.png)

The first page displays the first four variables (sorted alphabetically) with two subplots for each shock. The title indicates this is page 1 of 2.

## Display Plots

The `show_plots` argument (default: `true`, type: `Bool`), when `true`, displays the plots; otherwise, they are only returned as an object.

```julia
plot_fevd(Smets_Wouters_2007_linear,
    show_plots = false)
```

## Saving Plots

The `save_plots` argument (default: `false`, type: `Bool`), when `true`, saves the plots to disk; otherwise, they are only displayed and returned as an object.

Related arguments control the saving behavior:

- `save_plots_format` (default: `:pdf`, type: `Symbol`): output format of saved plots. See [input formats compatible with GR](https://docs.juliaplots.org/latest/output/#Supported-output-file-formats) for valid formats.
- `save_plots_path` (default: `"."`, type: `String`): path where plots are saved. If the path does not exist, it will be created automatically.
- `save_plots_name` (default: `"fevd"`, type: `Union{String, Symbol}`): prefix prepended to the filename when saving plots.

Each plot is saved as a separate file with a name indicating the prefix, model name, shocks, and a sequential number for multiple plots (e.g., `fevd__ModelName__shock__1.pdf`).

The following example saves all policy functions for the `Smets_Wouters_2007_linear` model as PNG files in the `../plots` directory with `fevd_plot` as the filename prefix:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    save_plots = true,
    save_plots_format = :png,
    save_plots_path = "./../plots",
    save_plots_name = :fevd_plot)
```

The plots appear in the specified folder with the specified prefix. Each plot is saved in a separate file with a name reflecting the model, the shock, and a sequential index when the number of variables exceeds the plots per page.

## Variable and Shock Renaming

The `rename_dictionary` argument (default: `Dict()`, type: `AbstractDict{<:Union{Symbol, String}, <:Union{Symbol, String}}`) maps variable or shock symbols to custom display names in plots. This is particularly useful when comparing models with different variable naming conventions, allowing them to be displayed with consistent labels.

For example, to rename variables for clearer display:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    rename_dictionary = Dict(:y => "Output", :pinfobs => "Inflation", :robs => "Interest Rate", :inve => "Investment", :c => "Consumption", :w => "Wages", :lab => "Labor"))
```

![Smets and Wouters 2007 FEVD - rename dictionary](../assets/rename_dict_fevd__Smets_Wouters_2007_linear__1.png)

The `rename_dictionary` accepts flexible type combinations for keys and values—both `Symbol` and `String` types work interchangeably:

```julia
# All of these are valid and equivalent:
Dict(:y => "Output")              # Symbol key, String value
Dict("y" => "Output")             # String key, String value
Dict(:y => :Output)               # Symbol key, Symbol value
Dict("y" => :Output)              # String key, Symbol value
```

This flexibility is particularly useful for models like `Backus_Kehoe_Kydland_1992`, which uses `String` representations of variable and shock names (because of `{}`):

```julia
# Define the Backus model (abbreviated for clarity)
@model Backus_Kehoe_Kydland_1992 begin
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1-theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1/nu{co})
        K{co}[0] = (1-delta{co})*K{co}[-1] + S{co}[0]
        X{co}[0] = for lag in (-4+1):0 phi{co} * S{co}[lag] end
        A{co}[0] = (1-eta{co}) * A{co}[-1] + N{co}[0]
        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1-alpha{co})*eta{co} * A{co}[-1]
        U{co}[0] = (C{co}[0]^mu{co}*L{co}[0]^(1-mu{co}))^gamma{co}
        psi{co} * mu{co} / C{co}[0]*U{co}[0] = LGM[0]
        psi{co} * (1-mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1-theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co}*N{co}[0]^(1-theta{co}))^(-nu{co})*Y{co}[0]^(1+nu{co})

        for lag in 0:(4-1)  
            beta{co}^lag * LGM[lag]*phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1-delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1-theta{co})) ^ (-nu{co})* Y{co}[+4]^(1+nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1+sigma{co} * Z{co}[0]^(-nu{co}-1)*Y{co}[+1]^(1+nu{co}))
        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1]))/Y{co}[0]
    end

    (LAMBDA{H}[0]-1) = rho{H}{H}*(LAMBDA{H}[-1]-1) + rho{H}{F}*(LAMBDA{F}[-1]-1) + Z_E{H} * E{H}[x]
    (LAMBDA{F}[0]-1) = rho{F}{F}*(LAMBDA{F}[-1]-1) + rho{F}{H}*(LAMBDA{H}[-1]-1) + Z_E{F} * E{F}[x]

    for co in [H,F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H,F] Y{co}[0] end
end

@parameters Backus_Kehoe_Kydland_1992 begin
    K_ss = 11
    K[ss] = K_ss | beta
    
    mu      =    0.34
    gamma   =    -1.0
    alpha   =    1
    eta     =    0.5
    theta   =    0.36
    nu      =    3
    sigma   =    0.01
    delta   =    0.025
    phi     =    1/4
    psi     =    0.5

    Z_E = 0.00852
    
    rho{H}{H} = 0.906
    rho{F}{F} = rho{H}{H}
    rho{H}{F} = 0.088
    rho{F}{H} = rho{H}{F}
end

# Backus model example showing String to String mapping
plot_fevd(Backus_Kehoe_Kydland_1992,
    rename_dictionary = Dict("K{H}" => "Capital (Home)", 
                             "K{F}" => "Capital (Foreign)",
                             "Y{H}" => "Output (Home)",
                             "Y{F}" => "Output (Foreign)"))
```

![Backus, Kehoe, Kydland 1992 FEVD - rename dictionary](../assets/rename_dict_string__Backus_Kehoe_Kydland_1992__1.png)

Variables or shocks not included in the dictionary retain their default names. The renaming applies to all plot elements including legends, axis labels, and tables.

## Verbose Output

The `verbose` argument (default: `false`, type: `Bool`), when `true`, enables verbose output related to solving the model

```julia
plot_fevd(Smets_Wouters_2007_linear,
    verbose = true)
```

The code outputs information about solving the steady state blocks.
When parameters change, the first-order solution is recomputed; otherwise, it uses the cached solution:

```julia
plot_fevd(Smets_Wouters_2007_linear,
    parameters = :z_eg => 1.05,
    verbose = true)
# Parameter changes: 
#         z_eg    from 1.5        to 1.05
# New parameters changed the steady state.
# Block: 1, - Solved using previous solution; residual norm: 1.776217186138026e-21
# Block: 2, - Solved using previous solution; residual norm: 6.83217016268833e-19
# Quadratic matrix equation solver previous solution has tolerance: 2.25183977733317e-15
```

## Numerical Tolerances

The `tol` argument (default: `Tolerances()`, type: `Tolerances`) defines various tolerances for the algorithm used to solve the model. See the Tolerances documentation for more details: `?Tolerances`.
The tolerances used by the numerical solvers can be adjusted. The Tolerances object allows setting tolerances for the non-stochastic steady state solver (NSSS), Sylvester equations, Lyapunov equation, and quadratic matrix equation (QME). For example, to set tighter tolerances (this example also changes parameters to force recomputation):

```julia
custom_tol = Tolerances(qme_acceptance_tol = 1e-12,
    sylvester_acceptance_tol = 1e-12)

plot_fevd(Smets_Wouters_2007_linear,
    tol = custom_tol,
    parameters = :z_eg => 1.055,
    verbose = true)
# Parameter changes: 
#         z_eg    from 1.05       to 1.055
# New parameters changed the steady state.
# Block: 1, - Solved using previous solution; residual norm: 1.776217186138026e-21
# Block: 2, - Solved using previous solution; residual norm: 6.83217016268833e-19
# Quadratic matrix equation solver: schur - converged: true in 0 iterations to tolerance: 2.25183977733317e-15
```

This is useful when higher precision is needed or when the default tolerances are insufficient for convergence. Use this argument for specific needs or when encountering issues with the default solver.

## Quadratic Matrix Equation Solver

The `quadratic_matrix_equation_algorithm` argument (default: `:schur`, type: `Symbol`) specifies the algorithm to solve quadratic matrix equation (`A * X ^ 2 + B * X + C = 0`). Available algorithms: `:schur`, `:doubling`
The quadratic matrix equation solver is used internally when solving the model to first order. Different algorithms are available. The `:schur` algorithm is generally faster and more reliable, while `:doubling` can be more precise in some cases (this example also changes parameters to force recomputation):

```julia
plot_fevd(Smets_Wouters_2007_linear,
    quadratic_matrix_equation_algorithm = :doubling,
    parameters = :z_eg => 1.0555,
    verbose = true)
# Parameter changes: 
#         z_eg    from 1.055      to 1.0555
# New parameters changed the steady state.
# Block: 1, - Solved using previous solution; residual norm: 1.776217186138026e-21
# Block: 2, - Solved using previous solution; residual norm: 6.83217016268833e-19
# Quadratic matrix equation solver previous solution has tolerance: 2.25183977733317e-15
```

For most use cases, the default `:schur` algorithm is recommended. Use this argument for specific needs or when encountering issues with the default solver.
