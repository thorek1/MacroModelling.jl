using Revise
using MacroModelling
import MacroModelling: get_NSSS_and_parameters
using BenchmarkTools

include("models/RBC_CME_calibration_equations_and_parameter_definitions_and_specfuns.jl")

include("models/RBC_CME_calibration_equations_and_parameter_definitions_lead_lags.jl")

𝓂 = m
𝓂.model_jacobian_SS_and_pars_vars[2]
𝓂.jacobian_SS_and_pars_vars[2]'

SS(m)
# (:Pibar)   1.00083          0.0       0.332779      0.0       0.00111065   0.0         0.0             0.0       0.0    -0.332779     0.0
include("../models/NAWM_EAUS_2008.jl")

𝓂 = NAWM_EAUS_2008




SS_and_pars, (solution_error, iters) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)

∇₁ = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)# |> Matrix
@benchmark calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

∇₁s = calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)
@benchmark calculate_jacobian(𝓂.parameter_values, SS_and_pars, 𝓂)

isapprox(∇₁s,∇₁)

@model Tmp begin
    ## Resource constraint
    ynetm[0] * 1 + c[0] * (-c_yc/(ynetmcons)) + sk[0] * ( -sk_yc/(ynetmcons) ) + sc[0] * (-sc_yc/(ynetmcons) ) + hc[0] * (- actualhc_yc/(ynetmcons) ) + zc[0] * ( -ac_zc/(1-ac_zc)*actualhc_yc/(ynetmcons) ) + ac[0] * (inv(zc_ac-1)*actualhc_yc/(ynetmcons) ) + hk[0] * (- actualhk_yc/(ynetmcons) ) + zk[0] * ( -ak_zk/(1-ak_zk)*actualhk_yc/(ynetmcons) ) + ak[0] * (inv(zk_ak-1)*actualhk_yc/(ynetmcons) ) = 0
    ## Euler equation
    c[0] * (1) + c[1] * (-1) + r[0] * (1) = 0
    ## Aggregate production function FIXME [0], [-1] or SS (muc and nc)
    yc[0] * (1) + chiz[0] * (-1/(1-bb)) + k[-1] * (-al) + l[0] * (  al) + ly[0] * (-1) + nc[0] * (-(muc[0]-1)/(1-bb)) + muc[0] * ((bb+muc[0]*log(nc[0]))/(1-bb)) + u[0] * (-al) + ac[0] * (-bb/(1-bb)*(theta-1)) = 0
    ## demand of capital
    yc[0] * (1) + kl[0] * (-1) + l[0] * ( 1) + ly[0] * (-1) + muc[0] * (-1) + d[0] * (- (1/(1+del/d_pk))) + pk[0] * (- (del/(d_pk+del))) + u[0] * (-edu*(1/(d_pk/del+1))) = 0
    ## capacity choice
    yc[0] * (1) + u[0] * (-(1+edp)) + muc[0] * (-1) + kl[0] * (-1) + l[0] * ( 1) + ly[0] * (-1) + pk[0] * (-1) = 0
    ## labor demand in final output
    yc[0] * (1) + muc[0] * (-1) + ly[0] * (-1) + w[0] * (-1) = 0
    ## production of new investment goods FIXME [0], [-1] or SS (muk and nk)
    yk[0] * (1) + chi[0]  * (-1/(1-bb)) + kl[0] * (- al) + l[0] * (al-1/(1-lc_l)) + u[0] * (- al) + ly[0] * (lc_l/(1-lc_l)) + nk[0] * (-(muk[0]-1)/(1-bb)) + muk[0] * ((bb+muk[0]*log(nk[0]))/(1-bb)) + pk[0] * (-bb/(1-bb)) + ak[0] * (-bb/(1-bb)*(th-1)) = 0
    ## Real wage
    w[0] * (1) + l[0] * (-fi) + c[0] * (-1) + muw[0] * (-1) = 0
    ## Profits embodied
    prk[0] * (1) + yk[0] * (-1) + pk[0] * (-1) + muk[0] * (1) = 0
    ## Profits disembodied
    prc[0] * (1) + yc[0] * (-1) + muc[0] * (1) = 0
    ## Value of an adopted innovation for embodied
    vk[0] * (1) + ak[0] * (-(1-prk_vk)) + prk[0] * (-prk_vk) + vk[1] * (-(1-prk_vk)) + ak[1] * ((1-prk_vk)) + r[0] * ((1-prk_vk)) = 0
    ## Value of an adopted innovation for disembodied
    vc[0] * (1) + ac[0] * (-(1-prc_vc)) + prc[0] * (-prc_vc) + vc[1] * (-(1-prc_vc)) + ac[1] * ((1-prc_vc)) + r[0] * ((1-prc_vc)) = 0
    ## Capital accumulation
    k[0] * ( 1) + u[0] * ( edu*del/(1+gk)) + k[-1] * (-jcof) + yk[0] * (-(1-jcof)) = 0
    ## Law of motion for embodied productivity
    zk[0] * ( 1) + zk[-1] * (-1) + sk[-1] * (-rho*(gzk+ok)/(1+gzk)) + sf[-1] * ( rho*(gzk+ok)/(1+gzk)) + chik[-1] * (-(gzk+ok)/(1+gzk)) = 0
    ## Law of motion for disembodied productivity
    zc[0] * ( 1) + zc[-1] * (-1) + sc[-1] * (-rho*(gzc+oc)/(1+gzc)) + sf[-1] * ( rho*(gzc+oc)/(1+gzc)) = 0
    ## Free entry for embodied
    sk[0] * ( 1-rho) + zk[0] * (-1) + sf[0] * ( rho) + jk[1] * (-1) + zk[1] * (1) + r[0] * (1) = 0
    ## Free entry for disembodied
    sc[0] * (1-rho) + zc[0] * (-1) + sf[0] * (rho) + jc[1] * (-1) + zc[1] * (1) + r[0] * (1) = 0
    ## Bellman for not adopted disemb innovation
    jc[0] * (-1) + hc[0] * (-(hc_jc+phic*elc*lamc/R*rz*(1-zc_ac*vc_jc))) + r[0] * (-(1+hc_jc)) + zc[0] * ( phic*rz*((1-lamc)+lamc*zc_ac*vc_jc)/R) + ac[1] * (-phic*lamc*rz*zc_ac*vc_jc/R) + vc[1] * ( phic*lamc*rz*zc_ac*vc_jc/R) + sf[0] * (-phic*elc*lamc*rz/R*(zc_ac*vc_jc-1)) + zc[1] * (-phic*rz*(1-lamc)/R) + jc[1] * ( phic*rz*(1-lamc)/R) = 0
    ## law of motion for adopted disembodied innvo
    ac[0] * (1) + ac[-1] * (-phic*(1-lamc)/(1+gzc)) + hc[-1] * (-elc*lamc*((phic/(1+gzc))*zc_ac-phic/(1+gzc))) + sf[-1] * (elc*lamc*((phic/(1+gzc))*zc_ac-phic/(1+gzc))) + zc[-1] * (-(1-phic*(1-lamc)/(1+gzc))) = 0
    ## optimal investment in adoption of disemb innov
    zc[0] * (1) + sf[0] * (-(1+ellc)) + r[0] * (-1) + hc[0] * (ellc) + vc[1] * (1/(1-jc_vc*ac_zc)) + ac[1] * (-1/(1-jc_vc*ac_zc)) + jc[1] * (-1/(vc_jc*zc_ac-1)) + zc[1] * (1/(vc_jc*zc_ac-1))
    ## Bellman for not adopted emb innovation
    jk[0] * (-1) + hk[0] * (-(hk_jk+(1-ok)*elk*lamk/R*ra*(1-zk_ak*vk_jk))) + r[0] * (-(1+hk_jk)) + zk[0] * (phik*ra*((1-lamk)+lamk*zk_ak*vk_jk)/R) + ak[1] * (-phik*lamk*ra*zk_ak*vk_jk/R) + vk[1] * (phik*lamk*ra*zk_ak*vk_jk/R) + sf[0] * (- phik*elk*lamk*ra/R*(zk_ak*vk_jk-1)) + zk[1] * (-phik*ra*(1-lamk)/R) + jk[1] * (phik*ra*(1-lamk)/R) = 0
    ## law of motion for adopted embodied innvo
    ak[0] * (1) + ak[-1] * (-phik*(1-lamk)/(1+gzk)) + hk[-1] * (-elk*lamk*((phik/(1+gzk))*zk_ak-phik/(1+gzk))) + sf[-1] * (elk*lamk*((phik/(1+gzk))*zk_ak-phik/(1+gzk))) + zk[-1] * (-(1-phik*(1-lamk)/(1+gzk))) = 0
    ## optimal investment in adoption of emb innov
    zk[0] * (1) + sf[0] * (-(1+ellk)) + r[0] * (-1) + hk[0] * (ellk) + vk[1] * (1/(1-jk_vk*ak_zk)) + ak[1] * (-1/(1-jk_vk*ak_zk)) + jk[1] * (-1/(vk_jk*zk_ak-1)) + zk[1] * (1/(vk_jk*zk_ak-1)) = 0
    ## Arbitrage
    pk[0] * (1) + r[0] * (1) + d[1] * (- (R-1-gpk)/R) + pk[1] * (-(1+gpk)/R) = 0
    ## entry into final goods sector
    muc[0] * (1) + yc[0] * (mucof) + sf[0] * (-mucof) + nc[0] * (-mucof) = 0
    ## m
    muc[0] * (1) + nc[0] * (-etamuc) = 0
    ## entry into capital goods sector
    muk[0] * (1) + yk[0] * (mukcof) + pk[0] * (mukcof) + sf[0] * (-mukcof) + nk[0] * (-mukcof) = 0
    ## mk
    muk[0] * (1) + nk[0] * (-etamuk) = 0
    ## equivalence between klzero and jlag
    kl[0] * (1) + k[-1] * (-1) = 0
    ## Definition of output net of total overhead costs
    ynet[0] * (1) + yc[0] * (-1/(1-oc_yc)) + nc[0] * (occ_yc/(1-oc_yc)) + nk[0] * (ock_yc/(1-oc_yc)) + sf[0] * (oc_yc/(1-oc_yc)) = 0
    ## definition of scaling factor
    sf[0] * (1) + kl[0] * (-1) + ak[0] * (-bb*(1-th)) + ac[0] * (bb*(1-theta)) = 0
    ## definition of ynetm
    ynetm[0] * (1) + ynet[0] * (- 1/(1-mc_yc*inv(ynet_yc)-mk_yc*inv(ynet_yc))) + yc[0] * (mc_yc/ynetmcons) + muc[0] * (-mc_yc/ynetmcons) + pk[0] * (mk_yc/ynetmcons) + yk[0] * (mk_yc/ynetmcons) + muk[0] * (-mk_yc/ynetmcons) = 0
    ## Definition of total value added
    yT[0] * (1) + ynetm[0] * (-ynetmcons/(ynetmcons+pkyk_yc)) + pk[0] * (-pkyk_yc/(ynetmcons+pkyk_yc)) + yk[0] * (-pkyk_yc/(ynetmcons+pkyk_yc)) = 0
    ## labor demand in capital goods production
    yk[0] * (1) + pk[0] * (1) + muk[0] * (-1) + w[0] * (-1) + l[0] * (- 1/(1-lc_l)) + ly[0] * (lc_l/(1-lc_l)) = 0
    # ## embodied productivity shock process
    chi[0] = ρᵡ * chi[-1] + σᵡ* eps_chi[x] = 0
    # ## Labor augmenting technology shock process
    chiz[0] = ρᶻᵪ * chiz[-1] + σᶻᵪ * eps_chi_z[x] = 0
    # ## Wage markup shock process
    muw[0] = muw[-1] * ρᵐʷ + σᵐʷ * eps_muw[x] = 0
    # ## Wage markup shock process
    chik[0] = ρᵏᵪ * chik[-1] + σᵏᵪ * eps_chi_k[x] = 0

end

@parameters Tmp begin
    bet    = 0.95 	     ## discount factor
    del    = 0.08 	     ## depreciation rate
    fi     = 1          ## labor supply curvature
    al     = 1/3        ## k share
    g_y    = 0.2*0.7    ## ss g/y ratio
    th     = 1/0.6      ## elasticity of substitution intermediate good sector
    rho    = 0.9        ## parameter embodied technology
    eta    = 0.0
    theta  = 1/0.6
    # muw[ss]    = 1.2        ## ss wage markup
    muw_ss    = 1.2        ## ss wage markup
    # muc[ss]    = 1.1
    muc_ss = 1.1
    # nc[ss]     = 1
    nc_ss = 1
    # nk[ss]     = 1
    nk_ss     = 1
    dmuc   = -muc_ss
    etamuc = dmuc*nc_ss/muc_ss
    boc    = (muc_ss-1)/muc_ss
    # muk[ss]    = 1.2
    muk_ss    = 1.2
    etamuk = etamuc
    lamk   = 0.1
    lamc   = 0.1
    elk    = 0.9
    elc    = 0.9
    ellk   = elk-1
    ellc   = elc-1
    o  = 0.03
    oz = 0.03
    oc = 0.03
    ok = 0.03
    phic   = 1-oc
    phik   = 1-ok
    bb     = 0.5 ## intermediate share in final output
    ## Nonstochastic steady state
    gpk    = -0.026
    gy     =  0.024
    gk     =  gy - gpk
    gzc    = (gy-al*gk)/bb*(1-bb)/(theta-1)
    gzk    = (gpk-gzc*bb*(theta-1))/(bb*(1-th))
    gtfp   =  gy-al*gk+gzk*(al*bb*(th-1))/(1-al*(1-bb))
    measbls = (0.014-gy+al*gk)/(gzk*(al*bb*(th-1))/(1-al*(1-bb)))
    gv     = gy
    gvz    = gy
    R      = (1+gy)/bet
    d_pk   = R-(1+gpk)                   ## definition of R
    yc_pkkc = muc_ss/(al*(1-bb))*(d_pk+del) ## foc for k
    yk_kk   = muk_ss/(al*(1-bb))*(d_pk+del) ## new capital to capital in capital production sector
    yk_k   = (gk+del)/(1+gk)             ## new capital to capital ratio
    kk_k   = yk_k/yk_kk                  ## share of capital in capital production.
    kc_k   = 1-kk_k
    kk_kc  = kk_k/kc_k
    lk_lc  = kk_kc
    lk_l   = lk_lc/(lk_lc+1)
    lc_l   = 1-lk_l
    pkyk_yc= kk_kc*muk_ss/muc_ss
    mk_yc  = bb*1/th*pkyk_yc/muk_ss
    mc_yc  = bb*1/theta/muc_ss
    pkk_yc = inv(yc_pkkc)/kc_k
    pik_yc = pkk_yc*muc_ss/muk_ss              ## value of total capital stock removing fluctuations in relative price of capital due to markup variations
    prk_yc   = pkyk_yc*(1-1/th)*bb/muk_ss
    prc_yc  = (1-1/theta)*bb/muc_ss
    prk_vk   = 1-(1+gv)*phik/((1+gzk)*R) ## bellman for va
    prc_vc = 1-(1+gvz)*phic/((1+gzc)*R) ## bellman for vz
    yc_vk    = prk_vk*inv(prk_yc)
    yc_vc   = prc_vc*inv(prc_yc)
    zk_ak   = ((gzk+ok)/(lamk*phik)+1)
    zc_ac   = ((gzc+oc)/(lamc*phic)+1)
    ac_zc   = inv(zc_ac)
    ak_zk   = inv(zk_ak)
    ra     = (1+gy)/(1+gzk)
    rz     = (1+gy)/(1+gzc)
    jk_yc    = inv(1-elk*phik*lamk*ra/R-(1-lamk)*phik*ra/R)*(1-elk)*phik*lamk*ra*zk_ak/R*inv(yc_vk) ## zk * jk /yc bellman for not adopted innov
    jc_yc   = inv(1/phic-elc*lamc*rz/R-(1-lamc)*rz/R)*(1-elc)*lamc*rz*zc_ac/R*inv(yc_vc) ## zc*jc/yc bellman for not adopted innov
    hk_yc    = phik*elk*lamk*ra/R*(inv(yc_vk)*zk_ak-jk_yc) ## zk *hk/yc
    hc_yc    = phic*elc*lamc*rz/R*(inv(yc_vc)*zc_ac-jc_yc) ## zc *hc/yc
    sk_yc    = jk_yc*(gzk+o)*(1+gv)*inv((1+gzk)*R) ## from free entry cond't
    sc_yc   = jc_yc*(gzc+oz)*(1+gvz)*inv((1+gzc)*R)
    hc_jc  = hc_yc/jc_yc
    hk_jk  = hk_yc/jk_yc
    vc_jc  = inv(yc_vc)/jc_yc
    vk_jk  = inv(yc_vk)/jk_yc
    jc_vc=inv(vc_jc)
    jk_vk=inv(vk_jk)
    bock   = boc*pkyk_yc*(muk_ss-1)*muc_ss/(muk_ss*(muc_ss-1))
    occ_yc=boc*pik_yc
    ock_yc=bock*pik_yc
    oc_yc=occ_yc+ock_yc
    c_yc = 1-oc_yc-g_y-mc_yc-mk_yc-sk_yc-sc_yc-((phic/(1+gzc))^2-inv(zc_ac))*hc_yc-((phik/(1+gzk))^2-inv(zk_ak))*hk_yc
    pi_yc=(muc_ss-1)/muc_ss-oc_yc
    # u[ss]=.8
    u_ss=.8
    edu=al*(1-bb)*yc_pkkc/(muc_ss*del) ## from foc wrt utilization, edu = elasticity of depreciation with respect to capacity
    edup=0 ## partial of edu wrt u
    edp=1/3##(edu)-1+edup/(edu*u); ## elasticity of del' (i.e. elasticity of delta prima)
    actualhk_yc=hk_yc*(1-ak_zk) ## total expenses in adoption of capital specific innovations
    actualhc_yc=hc_yc*(1-ac_zc) ## total expenses in adoption of consumption specific innovations
    inv_Y=pkyk_yc/(pkyk_yc+1-mc_yc-mk_yc-occ_yc-ock_yc) ## investment output ratio
    Y_yc=pkyk_yc/inv_Y
    ynet_yc=1-oc_yc
    ## Coefficients for the log-linearization
    qcof = (1-del)*(1+gpk)/R
    jcof = (1-del)/(1+gk)
    vcof = (1+gy)/((1+gzk)*R)
    vzcof= (1+gy)/((1+gzc)*R)
    mucof= muc_ss-1
    mukcof=muk_ss-1
    ycof=(ynet_yc-mc_yc-mk_yc+pkyk_yc-actualhk_yc-actualhc_yc-sk_yc-sc_yc)^(-1)
    ynetmcons=1-oc_yc-mc_yc-mk_yc ## fraction of ynetm in y
    # NOTE Shock to Embodied Technology
    ρᵡ = (0.7)^4   # autoregressive component
    σᵡ = 0.01    # standard deviation
    # Disembodied Technology Shock
    ρᶻᵪ = (0.7)^4
    σᶻᵪ = 0.01
    # Wage markup shock
    ρᵐʷ   = 0.60 # (rhow)
    σᵐʷ = 0.01
    # Chik shock
    ρᵏᵪ = 0.8
    σᵏᵪ = 0.01
end

𝓂 = Tmp




import MacroModelling: match_pattern, get_symbols, normcdf, normpdf, norminvcdf, norminv, qnorm, normlogpdf, normpdf, normcdf, pnorm, dnorm, erfc, erfcinv, solve_quadratic_matrix_equation, get_NSSS_and_parameters
import Symbolics



future_varss  = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₁₎$")))
present_varss = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₀₎$")))
past_varss    = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍₋₁₎$")))
shock_varss   = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₓ₎$")))
ss_varss      = collect(reduce(union,match_pattern.(get_symbols.(𝓂.dyn_equations),r"₍ₛₛ₎$")))

sort!(future_varss  ,by = x->replace(string(x),r"₍₁₎$"=>"")) #sort by name without time index because otherwise eps_zᴸ⁽⁻¹⁾₍₋₁₎ comes before eps_z₍₋₁₎
sort!(present_varss ,by = x->replace(string(x),r"₍₀₎$"=>""))
sort!(past_varss    ,by = x->replace(string(x),r"₍₋₁₎$"=>""))
sort!(shock_varss   ,by = x->replace(string(x),r"₍ₓ₎$"=>""))
sort!(ss_varss      ,by = x->replace(string(x),r"₍ₛₛ₎$"=>""))

dyn_future_list = collect(reduce(union, 𝓂.dyn_future_list))
dyn_present_list = collect(reduce(union, 𝓂.dyn_present_list))
dyn_past_list = collect(reduce(union, 𝓂.dyn_past_list))
dyn_exo_list = collect(reduce(union,𝓂.dyn_exo_list))
dyn_ss_list = Symbol.(string.(collect(reduce(union,𝓂.dyn_ss_list))) .* "₍ₛₛ₎")

future = map(x -> Symbol(replace(string(x), r"₍₁₎" => "")),string.(dyn_future_list))
present = map(x -> Symbol(replace(string(x), r"₍₀₎" => "")),string.(dyn_present_list))
past = map(x -> Symbol(replace(string(x), r"₍₋₁₎" => "")),string.(dyn_past_list))
exo = map(x -> Symbol(replace(string(x), r"₍ₓ₎" => "")),string.(dyn_exo_list))
stst = map(x -> Symbol(replace(string(x), r"₍ₛₛ₎" => "")),string.(dyn_ss_list))


vars_raw = vcat(dyn_future_list[indexin(sort(future),future)],
                dyn_present_list[indexin(sort(present),present)],
                dyn_past_list[indexin(sort(past),past)],
                dyn_exo_list[indexin(sort(exo),exo)])

SS_and_pars_names_lead_lag = vcat(Symbol.(string.(sort(union(𝓂.var,𝓂.exo_past,𝓂.exo_future)))), 𝓂.calibration_equations_parameters)

final_indices = vcat(𝓂.parameters, SS_and_pars_names_lead_lag)


# Symbolics.@syms norminvcdf(x) norminv(x) qnorm(x) normlogpdf(x) normpdf(x) normcdf(x) pnorm(x) dnorm(x) erfc(x) erfcinv(x)

# # overwrite SymPyCall names
# input_args = vcat(future_varss,
#                     present_varss,
#                     past_varss,
#                     ss_varss,
#                     𝓂.parameters,
#                     𝓂.calibration_equations_parameters,
#                     shock_varss)

# eval(:(Symbolics.@variables $(input_args...)))

# Symbolics.@variables 𝔛[1:length(input_args)]

# calib_eq_no_vars = reduce(union, get_symbols.(𝓂.calibration_equations_no_var), init = []) |> collect

# eval(:(Symbolics.@variables $((vcat(SS_and_pars_names_lead_lag, calib_eq_no_vars))...)))

# vars = eval(:(Symbolics.@variables $(vars_raw...)))

# eqs = Symbolics.parse_expr_to_symbolic.(𝓂.dyn_equations,(@__MODULE__,))

# input_X = Pair{Symbolics.Num, Symbolics.Num}[]
# input_X_no_time = Pair{Symbolics.Num, Symbolics.Num}[]

# for (v,input) in enumerate(input_args)
#     push!(input_X, eval(input) => eval(𝔛[v]))

#     if input ∈ shock_varss
#         push!(input_X_no_time, eval(𝔛[v]) => 0)
#     else
#         input_no_time = Symbol(replace(string(input), r"₍₁₎$"=>"", r"₍₀₎$"=>"" , r"₍₋₁₎$"=>"", r"₍ₛₛ₎$"=>"", r"ᴸ⁽⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+⁾" => ""))

#         vv = indexin([input_no_time], final_indices)
        
#         if vv[1] isa Int
#             push!(input_X_no_time, eval(𝔛[v]) => eval(𝔛[vv[1]]))
#         end
#     end
# end

# vars_X = map(x -> Symbolics.substitute(x, input_X), vars)

# eqs
# sort_order = calculate_kahn_topological_sort_order(𝓂.calibration_equations_no_var)

# sort_order = sorted_indices

# all_vars = union(future, present, past)
# sort!(all_vars)


pars_and_SS = Expr[]
for (i, p) in enumerate(vcat(𝓂.parameters, 𝓂.calibration_equations_parameters))
    push!(pars_and_SS, :($p = parameters_and_SS[$i]))
end

nn = length(pars_and_SS)

for (i, p) in enumerate(dyn_ss_list[indexin(sort(stst),stst)])
    push!(pars_and_SS, :($p = parameters_and_SS[$(i + nn)]))
end



deriv_vars = Expr[]
# for (k, m) in enumerate(["₍₁₎", "₍₀₎", "₍₋₁₎"])
    for (i, u) in enumerate(vars_raw)
        # push!(deriv_vars, :($(Symbol(string(u) * m)) = variables[$(i + (k-1) * length(vars_raw))]))
        push!(deriv_vars, :($u = variables[$i]))
    end
# end

# for (i, u) in enumerate(dyn_exo_list)
#     push!(deriv_vars, :($u = variables[$(i + 3 * length(vars_raw))]))
# end


eeqqss = Expr[]
for (i, u) in enumerate(𝓂.dyn_equations)
    push!(eeqqss, :(ϵ[$i] = $u))
end



dyn_eqs = :(function model_dynamics!(ϵ, variables, parameters_and_SS)
    @inbounds begin
        $(pars_and_SS...)
        $(𝓂.calibration_equations_no_var...)
        $(deriv_vars...)
        $(eeqqss...)
    end
    return nothing # [$(𝓂.dyn_equations...)]
end)


eval(dyn_eqs)



using DifferentiationInterface
using BenchmarkTools
using Symbolics, ForwardDiff, Zygote, Enzyme, FastDifferentiation, Mooncake, SparseMatrixColorings, SparseConnectivityTracer


stst = get_irf(𝓂, shocks = :none, variables = :all, levels = true, periods = 1) |> collect
stst_and_calib_pars = SS(𝓂, derivatives = false) |> collect

# stst = stst_and_calib_pars[1:end-length(𝓂.calibration_equations_parameters)]

calib_pars = stst_and_calib_pars[end-length(𝓂.calibration_equations_parameters)+1:end]

jac = zeros(length(𝓂.dyn_equations), length(deriv_vars));

ϵ = zeros(length(𝓂.dyn_equations))

SS_and_pars, (iters, tol_reached) = get_NSSS_and_parameters(𝓂, 𝓂.parameter_values)

STST = SS_and_pars[1:end - length(𝓂.calibration_equations)]
calibrated_parameters = SS_and_pars[(end - length(𝓂.calibration_equations)+1):end]

par = vcat(𝓂.parameter_values, calibrated_parameters)

dyn_var_future_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_future_idx
dyn_var_present_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_present_idx
dyn_var_past_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_var_past_idx
dyn_ss_idx = 𝓂.solution.perturbation.auxilliary_indices.dyn_ss_idx

shocks_ss = 𝓂.solution.perturbation.auxilliary_indices.shocks_ss

X = [STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]]; STST[dyn_ss_idx]; par; shocks_ss]

deriv_vars = vcat(STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],shocks_ss)
SS_and_pars = vcat(par, STST[dyn_ss_idx])

# @benchmark model_dynamics!(ϵ, 
                            # vcat(STST[[dyn_var_future_idx; dyn_var_present_idx; dyn_var_past_idx]],shocks_ss), 
                            # vcat(par, STST[dyn_ss_idx]))




backend = AutoFastDifferentiation()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, zero(deriv_vars), Constant(zero(SS_and_pars))); # 3.3s

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prephess = prepare_jacobian(prep.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

@time prep3rd = prepare_jacobian(prephess.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

jac_buffer = similar(sparsity_pattern(prephess), eltype(stst))

backend = AutoSparse(
    # AutoForwardDiff();  # any object from ADTypes
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prephess = prepare_jacobian(prep.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

@time prep3rd = prepare_jacobian(prephess.jac_exe, backend, (deriv_vars), Constant((SS_and_pars))); # 3.3s

jacobian!(jac_deriv!, jac, jac_buffer, prephess, backend, SS_and_pars, C)



prep.jac_exe(deriv_vars, SS_and_pars)

jac_deriv(SS_and_pars, deriv_vars) = 𝓂.jacobian[4].jac_exe(deriv_vars, SS_and_pars)

jac2 = zeros(length(𝓂.dyn_equations) * length(deriv_vars), length(SS_and_pars))

C = Constant(deriv_vars)

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep2 = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.
jac_buffer = similar(sparsity_pattern(prep2), eltype(stst))


backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep2 = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.


prep.jac_exe(deriv_vars, SS_and_pars)

prep.jac_exe!(jac, deriv_vars, SS_and_pars)

jac_deriv(SS_and_pars, derivvars) = prep.jac_exe(derivvars, SS_and_pars)

jac_deriv!(jacc, SS_and_pars, derivvars) = prep.jac_exe!(jacc, derivvars, SS_and_pars)

C = Constant(deriv_vars)

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv!, jac, backend, SS_and_pars, C); # 3.

jac_buffer = similar(sparsity_pattern(prepjac), eltype(deriv_vars))

backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv!, jac, backend, SS_and_pars, C); # 3.

jacobian!(jac_deriv!, jac, jac_buffer, prepjac, backend, SS_and_pars, C)

@benchmark jacobian!(jac_deriv!, jac, jac_buffer, prepjac, backend, SS_and_pars, C)



backend = AutoSparse(
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

prepjac = prepare_jacobian(jac_deriv, backend, SS_and_pars, C); # 3.

jacobian!(jac_deriv, jac_buffer, prepjac, backend, SS_and_pars, C)

@benchmark jacobian!(jac_deriv, jac_buffer, prepjac, backend, SS_and_pars, C)


DifferentiationInterface.jacobian(jac_deriv, prep2, backend, SS_and_pars, Constant(deriv_vars))

# @benchmark model_dynamics(vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), vcat(𝓂.parameter_values, calib_pars, stst))

prepare_jacobian(x->x, AutoForwardDiff(), [0])

backend = AutoSparse(
    AutoForwardDiff();  # any object from ADTypes
    # AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
# DifferentiationInterface.jacobian!(model_dynamics, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))
jac_buffer = similar(sparsity_pattern(prep), eltype(stst))

# prep

backend = AutoSparse(
    # AutoForwardDiff();  # any object from ADTypes
    AutoFastDifferentiation();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoFastDifferentiation()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, zero(deriv_vars), Constant(zero(SS_and_pars))); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars))


𝓂.jacobian[3]
jacobian!(𝓂.jacobian... ,backend, deriv_vars, Constant(SS_and_pars))
𝓂.jacobian[3]
C = Constant(SS_and_pars)
@benchmark jacobian!(𝓂.jacobian... ,backend, deriv_vars, Constant(SS_and_pars))
@benchmark jacobian!(𝓂.jacobian[1], 𝓂.jacobian[2], 𝓂.jacobian[3], 𝓂.jacobian[4], backend, deriv_vars, C)


backend = AutoSparse(
    AutoSymbolics();  # any object from ADTypes
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoForwardDiff()

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac, prep, backend, deriv_vars, Constant(SS_and_pars))


backend = AutoForwardDiff()

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # 1.3s
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))



backend = AutoMooncake(; config=nothing)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 3.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # 1.3s
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))



backend = AutoMooncake(; config=nothing)

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # crashes
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))


backend = AutoZygote()

@time prep = prepare_jacobian(model_dynamics, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))); # 3.3s
@time jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst))) # 1.3s
@benchmark jacobian!(model_dynamics, jac, prep, backend, vcat(stst,stst,stst,zeros(𝓂.timings.nExo)), Constant(vcat(𝓂.parameter_values, calib_pars, stst)))


backend = AutoEnzyme() # (; mode=pushforward, function_annotation=Nothing)

@time prep = prepare_jacobian(model_dynamics!, ϵ, backend, deriv_vars, Constant(SS_and_pars)); # 0.3s
@time jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars)) # forever
@benchmark jacobian!(model_dynamics!, ϵ, jac_buffer, prep, backend, deriv_vars, Constant(SS_and_pars))



# Reorder the calibration equations accordingly so that for each equation,
# all unknowns on its right-hand side have been defined by an earlier equation.
sorted_calibration_equations_no_var = 𝓂.calibration_equations_no_var[sorted_indices]





eqs_sub = Symbolics.Num[]
for subst in eqs
    for _ in calib_eqs
        for calib_eq in calib_eqs
            subst = Symbolics.substitute(subst, calib_eq)
        end
    end
    # subst = Symbolics.fixpoint_sub(subst, calib_eqs)
    subst = Symbolics.substitute(subst, input_X)
    push!(eqs_sub, subst)
end

if max_perturbation_order >= 2 
    nk = length(vars_raw)
    second_order_idxs = [nk * (i-1) + k for i in 1:nk for k in 1:i]
    if max_perturbation_order == 3
        third_order_idxs = [nk^2 * (i-1) + nk * (k-1) + l for i in 1:nk for k in 1:i for l in 1:k]
    end
end

first_order = Symbolics.Num[]
second_order = Symbolics.Num[]
third_order = Symbolics.Num[]
row1 = Int[]
row2 = Int[]
row3 = Int[]
column1 = Int[]
column2 = Int[]
column3 = Int[]

for (c1, var1) in enumerate(vars_X)
    for (r, eq) in enumerate(eqs_sub)
        if Symbol(var1) ∈ Symbol.(Symbolics.get_variables(eq))
            deriv_first = Symbolics.derivative(eq, var1)
        end
    end
end



## SVD tests (doesnt work)


using LinearAlgebra
SSVVDD = jac_buffer[1:230,1:230] |> collect |> svd
cutoff = 1-1e-7
n_cutoff = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,SSVVDD.S[1:i]) / sum(abs2,SSVVDD.S) > cutoff ? break : n_cutoff += 1
end
(sum(abs2,SSVVDD.S) - sum(abs2,SSVVDD.S[1:30])) / sum(abs2,SSVVDD.S)

SSVVDD.S[31:90]
backend = AutoSymbolics()


n_cutoff = 120
# SSVVDD.U[:,1:n_cutoff]
# SSVVDD.V[:,1:n_cutoff]

A = jac_buffer[1:230,1:230] |> collect
B = jac_buffer[1:230,231:460] |> collect
C = jac_buffer[1:230,461:690] |> collect

# Compute the singular value decomposition of A.
U, S, V = svd(B)
# Determine effective rank r: count singular values above tol.
cutoff = 1-1e-7
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end

# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
# S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V

# In our quadratic equation, we assume that the highest‐order term
# (multiplying X^2) is dominated by A.
# Represent X in the reduced space as:
#    X ≈ U_r * X_tilde * V_r'
#
# To derive a reduced quadratic equation, substitute:
#    X_tilde ≈ the unknown (r×r) matrix,
# and (assuming that V_r' * U_r ≈ I) approximate:
#    X^2 ≈ U_r * (X_tilde^2) * V_r'.
#
# Project the matrices:
A_r = U_r' * A * U_r      # (r x r)
B_r = U_r' * B * V_r      # (r x r)
C_r = V_r' * C * V_r      # (r x r)

X̃, iter, reached_tol = solve_quadratic_matrix_equation(A_r, B_r, C_r, Val(:doubling), 𝓂.timings)#, initial_guess = randn(size(A_r)))

reached_tol
X̃

get_solution(𝓂)

# Define the function for the reduced quadratic equation:
# F( X_tilde ) = A_r * X_tilde^2 + B_r * X_tilde + C_r
norm(A_r * (X̃ * X̃) + B_r * X̃ + C_r) / max(norm(X̃), norm(A_r))


n = size(A,1)

U, S, V = svd(A)


r = sum(S .> 1e-12)

r = 10
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V


# S = Diagonal(S_vec)  # S is diagonal with singular values
# Define Q = V' * U (which is orthogonal)
Q_r = V_r' * U_r

# Create an arbitrary matrix X in the full space
# X = randn(n, n)
# Express X in the transformed coordinates: let Y = U' * X * V, so that X = U * Y * V'
Y = U_r' * XX * V_r


U_r' / Y / (V_r) - XX

# Compute the original residual of the quadratic equation:
R_full = A * XX^2 + B * XX + C

U_r' * A * XX^2 * V_r   +    U_r' * B * XX * V_r    +    U_r' * C * V_r

U_r' \ V_r' * XX * U_r * V_r
# In the transformed space, note that:
#   X^2 = U * Y * (V' * U) * Y * V' = U * Y * Q * Y * V'
# Thus, the transformed (projected) equation is:
# R_proj = S * (Q * Y * Q * Y) + (U' * B * U) * Y + (U' * C * V)
R_proj = S_r * Q_r * Y * Q_r * Y + (U_r' * B * U_r) * Y + (U_r' * C * V_r)

norm(R_full)
norm(R_proj)

U, S, V = svd(XX)

# Determine effective rank r: count singular values above tol.
cutoff = 1-1e-12
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end

# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V

U_r * S_r * V_r' - XX
U * Diagonal(S) * V' - XX

norm(A * (U_r * S_r * V_r' * U_r * S_r * V_r') + B * U_r * S_r * V_r' + C)

A * (XX * XX) + B * XX + C


U, S, V = svd(A)

# Determine effective rank r: count singular values above tol.
cutoff = 1-eps()
r = 1
for i in 1:length(SSVVDD.S)
    sum(abs2,S[1:i]) / sum(abs2,S) > cutoff ? break : r += 1
end
r = sum(S .> 1e-12)
# r = sum(S .> 1e-4)
# r = 230
println("Effective rank r = ", r)
U_r = U[:, 1:r]
S_r = Diagonal(S[1:r])
V_r = V[:, 1:r]  # V_r is the first r columns of V
U_r * V_r'
norm(U_r * S_r * V_r' - A)
norm(U_r * S_r * V_r' * (XX * XX) + B * XX + C)
norm(U_r * S_r * V_r' * (XX * XX) + B * XX + C)
norm(A * (XX * XX) + B * XX + C)


X = U_r * X̃ * V_r'
X = U_r * X̃ * V_r'
X - XX




A = SSVVDD.V[:,1:n_cutoff]' * jac_buffer[1:230,1:230] * SSVVDD.U[:,1:n_cutoff]
B = SSVVDD.V[:,1:n_cutoff]' * jac_buffer[1:230,231:460] * SSVVDD.V[:,1:n_cutoff]
C = SSVVDD.U[:,1:n_cutoff]' * jac_buffer[1:230,461:690] * SSVVDD.V[:,1:n_cutoff]


X̃, iter, reached_tol = solve_quadratic_matrix_equation(A,B,C, Val(:doubling), 𝓂.timings)
reached_tol
X

X = SSVVDD.V[:,1:n_cutoff] * X̃ * SSVVDD.U[:,1:n_cutoff]'

XX, iter, reached_tol = solve_quadratic_matrix_equation(jac_buffer[1:230,1:230],jac_buffer[1:230,231:460],jac_buffer[1:230,461:690], Val(:doubling), 𝓂.timings)
reached_tol

norm(X-XX)/max(norm(X),norm(XX)) 
