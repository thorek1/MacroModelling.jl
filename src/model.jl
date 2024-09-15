using MacroModelling

@model Tmp begin
    ## Resource constraint
    ynetm[0] * 1 + c[0] * (-c_yc/(ynetmcons)) + sk[0] * ( -sk_yc/(ynetmcons) ) + sc[0] * (-sc_yc/(ynetmcons) ) + hc[0] * (- actualhc_yc/(ynetmcons) ) + zc[0] * ( -ac_zc/(1-ac_zc)*actualhc_yc/(ynetmcons) ) + ac[0] * (1/(zc_ac-1)*actualhc_yc/(ynetmcons) ) + hk[0] * (- actualhk_yc/(ynetmcons) ) + zk[0] * ( -ak_zk/(1-ak_zk)*actualhk_yc/(ynetmcons) ) + ak[0] * (1/(zk_ak-1)*actualhk_yc/(ynetmcons) ) = 0
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
    ynetm[0] * (1) + ynet[0] * (- 1/(1-mc_yc*1/(ynet_yc)-mk_yc*1/(ynet_yc))) + yc[0] * (mc_yc/ynetmcons) + muc[0] * (-mc_yc/ynetmcons) + pk[0] * (mk_yc/ynetmcons) + yk[0] * (mk_yc/ynetmcons) + muk[0] * (-mk_yc/ynetmcons) = 0
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
    dmuc   = -1.1 # -muc_ss
    etamuc = -1 # dmuc*nc_ss/muc_ss
    boc    = 0.09090909090909098 # (muc_ss-1)/muc_ss
    # muk[ss]    = 1.2
    muk_ss    = 1.2
    etamuk = -1 # etamuc
    lamk   = 0.1
    lamc   = 0.1
    elk    = 0.9
    elc    = 0.9
    ellk   = -0.09999999999999998 # elk-1
    ellc   = -0.09999999999999998 # elc-1
    o  = 0.03
    oz = 0.03
    oc = 0.03
    ok = 0.03
    phic   = 0.97 # 1-oc
    phik   = 0.97 # 1-ok
    bb     = 0.5 ## intermediate share in final output
    ## Nonstochastic steady state
    gpk    = -0.026
    gy     =  0.024
    gk     =  .05 # gy - gpk
    gzc    = .011 # (gy-al*gk)/bb*(1-bb)/(theta-1)
    gzk    = .089 # (gpk-gzc*bb*(theta-1))/(bb*(1-th))
    gtfp   =  .0192 # gy-al*gk+gzk*(al*bb*(th-1))/(1-al*(1-bb))
    measbls = 0.5617977528089887 # (0.014-gy+al*gk)/(gzk*(al*bb*(th-1))/(1-al*(1-bb)))
    gv     = 0.024 # gy
    gvz    = 0.024 # gy
    R      = 1.0778947368421052 # (1+gy)/bet
    d_pk   = 0.10389473684210526 # R-(1+gpk)                   ## definition of R
    yc_pkkc = 1.2137052631578948 # muc_ss/(al*(1-bb))*(d_pk+del) ## foc for k
    yk_kk   = 1.3240421052631581 # muk_ss/(al*(1-bb))*(d_pk+del) ## new capital to capital in capital production sector
    yk_k   = 0.12380952380952381 # (gk+del)/(1+gk)             ## new capital to capital ratio
    kk_k   = 0.09350875120766361 # yk_k/yk_kk                  ## share of capital in capital production.
    kc_k   = 0.9064912487923364 # 1-kk_k
    kk_kc  = 0.1031546099669905 # kk_k/kc_k
    lk_lc  = 0.1031546099669905 # kk_kc
    lk_l   = 0.09350875120766361 # lk_lc/(lk_lc+1)
    lc_l   = 0.9064912487923364 # 1-lk_l
    pkyk_yc= 0.11253230178217144 # kk_kc*muk_ss/muc_ss
    mk_yc  = 0.02813307544554286 # bb*1/th*pkyk_yc/muk_ss
    mc_yc  = 0.2727272727272727 # bb*1/theta/muc_ss
    pkk_yc = 0.9089147451636925 # 1/(yc_pkkc)/kc_k
    pik_yc = 0.833171849733385 # pkk_yc*muc_ss/muk_ss              ## value of total capital stock removing fluctuations in relative price of capital due to markup variations
    prk_yc   = 0.01875538363036191 # pkyk_yc*(1-1/th)*bb/muk_ss
    prc_yc  = 0.18181818181818182 # (1-1/theta)*bb/muc_ss
    prk_vk   = 0.15381083562901743 # 1-(1+gv)*phik/((1+gzk)*R) ## bellman for va
    prc_vc = 0.08852621167161212 # 1-(1+gvz)*phic/((1+gzc)*R) ## bellman for vz
    yc_vk    = 8.200889870363556 # prk_vk*1/(prk_yc)
    yc_vc   = 0.48689416419386666 # prc_vc*1/(prc_yc)
    zk_ak   = 2.2268041237113403 # ((gzk+ok)/(lamk*phik)+1)
    zc_ac   = 1.4226804123711339 # ((gzc+oc)/(lamc*phic)+1)
    ac_zc   = 0.7028985507246378 # 1/(zc_ac)
    ak_zk   = 0.44907407407407407 # 1/(zk_ak)
    ra     = 0.9403122130394859 # (1+gy)/(1+gzk)
    rz     = 1.0128585558852623 # (1+gy)/(1+gzc)
    jk_yc    = 0.014159338410620156 # 1/(1-elk*phik*lamk*ra/R-(1-lamk)*phik*ra/R)*(1-elk)*phik*lamk*ra*zk_ak/R*1/(yc_vk) ## zk * jk /yc bellman for not adopted innov
    jc_yc   = 0.2727626948903887 # 1/(1/phic-elc*lamc*rz/R-(1-lamc)*rz/R)*(1-elc)*lamc*rz*zc_ac/R*1/(yc_vc) ## zc*jc/yc bellman for not adopted innov
    hk_yc    = 0.019600737056023755 # phik*elk*lamk*ra/R*(1/(yc_vk)*zk_ak-jk_yc) ## zk *hk/yc
    hc_yc    = 0.21731983257587298 # phic*elc*lamc*rz/R*(1/(yc_vc)*zc_ac-jc_yc) ## zc *hc/yc
    sk_yc    = 0.0014698927523605224 # jk_yc*(gzk+o)*(1+gv)*1/((1+gzk)*R) ## from free entry cond't
    sc_yc   = 0.01050851331946651 # jc_yc*(gzc+oz)*(1+gvz)*1/((1+gzc)*R)
    hc_jc  = 0.7967359050445085 # hc_yc/jc_yc
    hk_jk  = 1.3842975206611559 # hk_yc/jk_yc
    vc_jc  = 7.529748283752845 # 1/(yc_vc)/jc_yc
    vk_jk  = 8.61184210526315 # 1/(yc_vk)/jk_yc
    jc_vc= 0.13280656435192248 # 1/(vc_jc)
    jk_vk= 0.11611917494270446 # 1/(vk_jk)
    bock   = 0.018755383630361902 # boc*pkyk_yc*(muk_ss-1)*muc_ss/(muk_ss*(muc_ss-1))
    occ_yc= 0.07574289543030778 # boc*pik_yc
    ock_yc= 0.015626457671767874 # bock*pik_yc
    oc_yc= 0.09136935310207565 # occ_yc+ock_yc
    c_yc = 0.40174590233878776 # 1-oc_yc-g_y-mc_yc-mk_yc-sk_yc-sc_yc-((phic/(1+gzc))^2-1/(zc_ac))*hc_yc-((phik/(1+gzk))^2-1/(zk_ak))*hk_yc
    pi_yc= -0.00046026219298467286 # (muc_ss-1)/muc_ss-oc_yc
    # u[ss]=.8
    u_ss=.8
    edu= 2.2986842105263157 # al*(1-bb)*yc_pkkc/(muc_ss*del) ## from foc wrt utilization, edu = elasticity of depreciation with respect to capacity
    edup=0 ## partial of edu wrt u
    edp=1/3##(edu)-1+edup/(edu*u); ## elasticity of del' (i.e. elasticity of delta prima)
    actualhk_yc= 0.010798554211420494 # hk_yc*(1-ak_zk) ## total expenses in adoption of capital specific innovations
    actualhc_yc= 0.06456603721457094 # hc_yc*(1-ac_zc) ## total expenses in adoption of consumption specific innovations
    inv_Y= 0.1562292038136742 # pkyk_yc/(pkyk_yc+1-mc_yc-mk_yc-occ_yc-ock_yc) ## investment output ratio
    Y_yc= 0.7203026005072803 # pkyk_yc/inv_Y
    ynet_yc= 0.9086306468979244 # 1-oc_yc
    ## Coefficients for the log-linearization
    qcof = 0.83132421875 # (1-del)*(1+gpk)/R
    jcof = 0.8761904761904762 # (1-del)/(1+gk)
    vcof =  0.8723599632690543 # (1+gy)/((1+gzk)*R)
    vzcof= 0.9396636993076164 # (1+gy)/((1+gzc)*R)
    mucof= 0.10000000000000009 # muc_ss-1
    mukcof= 0.19999999999999996 # muk_ss-1
    ycof= 1.5798796562140973 # (ynet_yc-mc_yc-mk_yc+pkyk_yc-actualhk_yc-actualhc_yc-sk_yc-sc_yc)^(-1)
    ynetmcons= 0.6077702987251088 # 1-oc_yc-mc_yc-mk_yc ## fraction of ynetm in y
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

SS(Tmp)

model = Tmp
import StatsPlots

plot_irf(
    model,
    # periods = nimpstep,
    # show_plots = false,
    # shocks = :all,
    # save_plots = true,
    # save_plots_path = "./graphs"
)



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
pkk_yc = 1/(yc_pkkc)/kc_k
pik_yc = pkk_yc*muc_ss/muk_ss              ## value of total capital stock removing fluctuations in relative price of capital due to markup variations
prk_yc   = pkyk_yc*(1-1/th)*bb/muk_ss
prc_yc  = (1-1/theta)*bb/muc_ss
prk_vk   = 1-(1+gv)*phik/((1+gzk)*R) ## bellman for va
prc_vc = 1-(1+gvz)*phic/((1+gzc)*R) ## bellman for vz
yc_vk    = prk_vk*1/(prk_yc)
yc_vc   = prc_vc*1/(prc_yc)
zk_ak   = ((gzk+ok)/(lamk*phik)+1)
zc_ac   = ((gzc+oc)/(lamc*phic)+1)
ac_zc   = 1/(zc_ac)
ak_zk   = 1/(zk_ak)
ra     = (1+gy)/(1+gzk)
rz     = (1+gy)/(1+gzc)
jk_yc    = 1/(1-elk*phik*lamk*ra/R-(1-lamk)*phik*ra/R)*(1-elk)*phik*lamk*ra*zk_ak/R*1/(yc_vk) ## zk * jk /yc bellman for not adopted innov
jc_yc   = 1/(1/phic-elc*lamc*rz/R-(1-lamc)*rz/R)*(1-elc)*lamc*rz*zc_ac/R*1/(yc_vc) ## zc*jc/yc bellman for not adopted innov
hk_yc    = phik*elk*lamk*ra/R*(1/(yc_vk)*zk_ak-jk_yc) ## zk *hk/yc
hc_yc    = phic*elc*lamc*rz/R*(1/(yc_vc)*zc_ac-jc_yc) ## zc *hc/yc
sk_yc    = jk_yc*(gzk+o)*(1+gv)*1/((1+gzk)*R) ## from free entry cond't
sc_yc   = jc_yc*(gzc+oz)*(1+gvz)*1/((1+gzc)*R)
hc_jc  = hc_yc/jc_yc
hk_jk  = hk_yc/jk_yc
vc_jc  = 1/(yc_vc)/jc_yc
vk_jk  = 1/(yc_vk)/jk_yc
jc_vc=1/(vc_jc)
jk_vk=1/(vk_jk)
bock   = boc*pkyk_yc*(muk_ss-1)*muc_ss/(muk_ss*(muc_ss-1))
occ_yc=boc*pik_yc
ock_yc=bock*pik_yc
oc_yc=occ_yc+ock_yc
c_yc = 1-oc_yc-g_y-mc_yc-mk_yc-sk_yc-sc_yc-((phic/(1+gzc))^2-1/(zc_ac))*hc_yc-((phik/(1+gzk))^2-1/(zk_ak))*hk_yc
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