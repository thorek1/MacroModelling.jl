using Revise
using MacroModelling

@model NK_multisector begin
    #### Climate variables ####
    for co in [a]
        # P_EM_t{co}[0] = P_EM_ts{co} + shock_P_EM_t{co}[x] # Law of motion for carbon price 

        log(epsi_t{co}[0] / epsi_ts{co}) = rho_eps{co} * log(epsi_t{co}[-1] / epsi_ts{co}) - shock_epsi_t{co}[x] # Law of motion for aggregate TFP
    end

	# EM_t[0] = (1 - rho_EM) * EM_t[-1] + for co in [a] for se in 1:10 ZZ_t{se}{co}[0] end end # Law of motion for emissions concentration

    # for co in [a] 
    #     for se in 1:10   
    #         Pen_t{se}{co}[0] = gama0{se}{co} + EM_t[0] * gama1{se}{co} + gama2{se}{co} * EM_t[0] ^ 2  # Damage function
    #         ZZ_t{se}{co}[0] = (1 + shock_epsi_carb_int_t{co}[x]) * carb_int{se}{co} * Y_t{se}{co}[0] # Emissions
    #     end 
    # end

    # for co in [a] 
    #     ZZ_t{co}[0] = for se in 1:10 ZZ_t{se}{co}[0] end
    # end
    
    #### Domestic aggregates ####
    for co in [a] 
        log(R_t{co}[0] / R_ts{co}) = 0.8 * log(R_t{co}[-1] / R_ts{co}) + 0.3 * log(pi_cpi_t{co}[-1]) # Modified Taylor-rule

        K_t{co}[0] = (1 - delta) * K_t{co}[-1] + I_t{co}[0] # Law of motion for capital

        lambda_t{co}[0] = C_t{co}[0] ^ (-sig) # FOC wrt consumption

        1 = R_t{co}[0] * betta * lambda_t{co}[1] / lambda_t{co}[0] / pi_cpi_t{co}[1] # HH FOC wrt bonds

        1 = betta * lambda_t{co}[1] / lambda_t{co}[0] * (rk_t{co}[1] + (1 - delta) * PI_t{co}[1]) / PI_t{co}[0] # No arbitrage cond. plus FOC wrt investment/capital

        lambda_t{co}[0] * w_t{co}[0] = kappaN{co} * N_t{co}[0] ^ lab{co} # HH FOC wrt labor

        # 1 = (for se in 1:10 Psi_con{se}{co} * P_t{se}{co}[0] ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # Price index derived from opt. problem of cons.-bundler 
        1 = (for se in 1:10 Psi_con{se}{co} * P_t{se}{co}[0] ^ (( - sigc{co}) / (1 - sigc{co})) end) # Price index derived from opt. problem of cons.-bundler 

        # pi_cpi_t{co}[0] = (for se in 1:10 Psi_con{se}{co} * (pi_ppi_t{se}{co}[0] * P_t{se}{co}[-1]) ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # CPI-inflation 
        pi_cpi_t{co}[0] ^ (( - sigc{co} / (1 - sigc{co})))  = (for se in 1:10 Psi_con{se}{co} * (pi_ppi_t{se}{co}[0] * P_t{se}{co}[-1]) ^ (( - sigc{co}) / (1 - sigc{co})) end)# CPI-inflation 

        # PI_t{co}[0] = (for se in 1:10 Psi_inv{se}{co} * P_t{se}{co}[0] ^ (( - sigi{co}) / (1 - sigi{co})) end) ^ (( - (1 - sigi{co})) / sigi{co}) # Price index derived from opt. problem of inv.-bundler 
        PI_t{co}[0] ^ (( - sigi{co}) / (1 - sigi{co})) = (for se in 1:10 Psi_inv{se}{co} * P_t{se}{co}[0] ^ (( - sigi{co}) / (1 - sigi{co})) end) # Price index derived from opt. problem of inv.-bundler 

        # w_t{co}[0] = (for se in 1:10 omega_N{se}{co} * w_t{se}{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) end) ^ (( - (1 - upsi_N{co})) / upsi_N{co}) # Wage index derived from opt. problem of labor-bundler
        w_t{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) = (for se in 1:10 omega_N{se}{co} * w_t{se}{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) end) # Wage index derived from opt. problem of labor-bundler

        # rk_t{co}[0] = (for se in 1:10 omega_K{se}{co} * rk_t{se}{co}[0] ^ (( - upsi_K{co}) / (1 - upsi_K{co})) end) ^ (( - (1 - upsi_K{co})) / upsi_K{co}) # Rental rate index derived from opt. problem of capital-bundler
        rk_t{co}[0] ^ (( - upsi_K{co}) / (1 - upsi_K{co})) = (for se in 1:10 omega_K{se}{co} * rk_t{se}{co}[0] ^ (( - upsi_K{co}) / (1 - upsi_K{co})) end) # Rental rate index derived from opt. problem of capital-bundler

        Y_VA_t{co}[0] = C_t{co}[0] + I_t{co}[0] * PI_t{co}[0]

        Y_t{co}[0] = for se in 1:10 Y_t{se}{co}[0] end

        #### Consumption and investment demand for different goods, relative prices ####
        for se in 1:10
            C_t{se}{co}[0] = C_t{co}[0] * Psi_con{se}{co} * (1 / P_t{se}{co}[0]) ^ (1 / (1 - sigc{co})) # Specific demand for sectoral consumption goods

            I_t{se}{co}[0] = I_t{co}[0] * Psi_inv{se}{co} * (PI_t{co}[0] / P_t{se}{co}[0]) ^ (1 / (1 - sigi{co})) # Specific demand for sectoral investment goods

            N_t{se}{co}[0] = N_t{co}[0] * omega_N{se}{co} * (w_t{co}[0] / w_t{se}{co}[0]) ^ (1 / (1 - upsi_N{co})) # Specific demand for labor types

            K_t{se}{co}[0] = K_t{co}[0] * omega_K{se}{co} * (rk_t{co}[1] / rk_t{se}{co}[1]) ^ (1 / (1 - upsi_K{co})) # Specific demand for capital types   

            Y_t{se}{co}[0] = C_t{se}{co}[0] + I_t{se}{co}[0] + Y_t{se}{co}[0] * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[1] - 1) ^ 2 + for se2 in 1:10 H_t{se2}{se}{co}[0] end

            # Y_t{se}{co}[0] = epsi_t{co}[0] * epsi_t{se}{co}[0] * (1 - Pen_t{se}{co}[0]) * (N_t{se}{co}[0] ^ alphaN{se}{co} * K_t{se}{co}[-1] ^ (1 - alphaN{se}{co})) ^ alphaH{se}{co} * H_t{se}{co}[0] ^ (1 - alphaH{se}{co}) # Production technology of perfectly competitive firm

            Y_t{se}{co}[0] = epsi_t{co}[0] * epsi_t{se}{co}[0] * (N_t{se}{co}[0] ^ alphaN{se}{co} * K_t{se}{co}[-1] ^ (1 - alphaN{se}{co})) ^ alphaH{se}{co} * H_t{se}{co}[0] ^ (1 - alphaH{se}{co}) # Production technology of perfectly competitive firm

            # EM_cost_t{se}{co}[0] = P_EM_t{co}[0] * (1 + shock_epsi_carb_int_t{co}[x]) * carb_int{se}{co} # Emissions costs

            # mc_tild_t{se}{co}[0] = EM_cost_t{se}{co}[0] + mc_t{se}{co}[0] # Real marginal costs relevant for pricing (including emissions taxes)

            mc_tild_t{se}{co}[0] = mc_t{se}{co}[0] # Real marginal costs relevant for pricing (including emissions taxes)

            1 - thetap{se}{co} + mc_tild_t{se}{co}[0] * thetap{se}{co} / P_t{se}{co}[0] + (pi_ppi_t{se}{co}[1] - 1) * betta * lambda_t{co}[1] / lambda_t{co}[0] * kappap{se}{co} * pi_ppi_t{se}{co}[1] ^ 2 / pi_cpi_t{co}[1] * Y_t{se}{co}[1] / Y_t{se}{co}[0] = pi_ppi_t{se}{co}[0] * kappap{se}{co} * (pi_ppi_t{se}{co}[0] - 1) + (1 - thetap{se}{co}) * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[0] - 1) ^ 2 # Firm's pricing decision in case of nominal price rigidities

            w_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * alphaN{se}{co} * alphaH{se}{co} / N_t{se}{co}[0] # FOC firm cost minimization -> labor

            rk_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * (1 - alphaN{se}{co}) * alphaH{se}{co} / K_t{se}{co}[-1] # FOC firm cost minimization -> capital

            PH_t{se}{co}[0] = Y_t{se}{co}[0] * (1 - alphaH{se}{co}) * mc_t{se}{co}[0] / H_t{se}{co}[0] # FOC firm cost minimization -> intermediates

            PH_t{se}{co}[0] ^ (( - sigh{se}{co}) / (1 - sigh{se}{co})) = (for se2 in 1:10 Psi{se}{se2}{co} * P_t{se2}{co}[0] ^ (( - sigh{se}{co}) / (1 - sigh{se}{co})) end) # Price index derived from opt. problem of intermediate input bundler in sector se

            for se2 in 1:10 
                H_t{se}{se2}{co}[0] = H_t{se}{co}[0] * Psi{se}{se2}{co} * (PH_t{se}{co}[0] / P_t{se2}{co}[0]) ^ (1 / (1 - sigh{se}{co})) # Optimal amount of sector se2 output in bundle used in sector se 
            end

            log(epsi_t{se}{co}[0] / epsi_ts{se}{co}) = rho_eps{co} * log(epsi_t{se}{co}[-1] / epsi_ts{se}{co}) - shock_epsi_t{se}{co}[x] # Law of motion for sectoral TFP
        end
    end

    for co in [a]  
        for se in 2:10 
            pi_ppi_t{1}{co}[0] / pi_ppi_t{se}{co}[0] = P_t{1}{co}[0] / P_t{se}{co}[0] * P_t{se}{co}[-1] / P_t{1}{co}[-1] # (Nominal) PPI-inflation 
        end
    end
end

# NK_multisector.parameters_in_equations

@parameters NK_multisector begin
    # EM_t > 170

    # K_t{a} > 5

    betta   = 0.992 # Discount
    
    delta   = 0.025 # Depr. rate
    
    rho_EM  = 0.0021 # Rate of decay wrt atmosph. carbon concentration

    sig   = 2 # Intertemporal elasticity of substitution

    ## Country level parameters
    P_EM_ts{a} = 1e-07 # Initial carbon price level

    R_ts{a} = 1/betta # Nominal interest rate <-> real interest rate 

    epsi_ts{a} = 1 # Aggregate TFP shock
    
    N_t{a}[ss] = 1/3 | kappaN{a} # = 0.1 # endogenous
    # kappaN{a} = 13.01619

    lab{a} = 2 # Inverse Frisch elasticity of labor supply

    rho_eps{a}  = 0.8 # Persistence TFP shock

    sigc{a} = -0.099989 # Determines elasticity of substitution between sector-level consumption goods 

    sigi{a} = -0.3313806 # Determines elasticity of substitution between sector-level investment goods 

    upsi_K{a} = 2 # Determines the elasticity of substitution of capital across sectors

    upsi_N{a} = 2 # Determines the elasticity of substitution of labor across sectors

    ## Sectoral country level parameters
    # Sectoral TFP shock
    epsi_ts = 1

    # Damage function parameter (level)
    gama0 = 0 

    # Damage function parameter (proportional)
    gama1 = 0 

    # Damage function parameter (quadratic)
    gama2 = 0 

    # Substitution elasticities of intermediate goods bundle in sector se
    sigh = -9

    # Sticky price parameter
    kappap = 75

    thetap{1}{a} = 6 # 1.2/(1.2-1)
    thetap{2}{a} = 6 # 1.2/(1.2-1)
    thetap{3}{a} = 6.555555555555557 # 1.18/(1.18-1)
    thetap{4}{a} = 4.225806451612903 # 1.31/(1.31-1)
    thetap{5}{a} = 5.761904761904763 # 1.21/(1.21-1)
    thetap{6}{a} = 5.761904761904763 # 1.21/(1.21-1)
    thetap{7}{a} = 3.173913043478261 # 1.46/(1.46-1)
    thetap{8}{a} = 2.1627906976744184 # 1.86/(1.86-1)
    thetap{9}{a} = 2.5625000000000004 # 1.64/(1.64-1)
    thetap{10}{a} = 2.7241379310344827 # 1.58/(1.58-1)

    # Elasticity of output wrt intermediate-goods (calibrated to WIOD 2005 data)
    alphaH{1}{a} = 0.46782201078250973  
    alphaH{2}{a} = 0.6106439710640003 
    alphaH{3}{a} = 0.3037291982377759 
    alphaH{4}{a} = 0.3749934743220389 
    alphaH{5}{a} = 0.42832086712615436
    alphaH{6}{a} = 0.37000129577676777
    alphaH{7}{a} = 0.5107540047126684 
    alphaH{8}{a} = 0.5128965292523873 
    alphaH{9}{a} = 0.5490957050537137 
    alphaH{10}{a} = 0.5911693755881954

    # Elasticity of output wrt labour (calibrated to WIOD 2005 data)
    alphaN{1}{a} = 0.6721594280816602  
    alphaN{2}{a} = 0.2595318005172585  
    alphaN{3}{a} = 0.6203608901298171  
    alphaN{4}{a} = 0.29666952638298766 
    alphaN{5}{a} = 0.4636650057684192  
    alphaN{6}{a} = 0.7128189705991886  
    alphaN{7}{a} = 0.6522812367201153  
    alphaN{8}{a} = 0.5515522812752323  
    alphaN{9}{a} = 0.6589905203983675  
    alphaN{10}{a} = 0.6794664394984696 


    carb_int{1}{a} = 0.1918
    carb_int{2}{a} = 0.4023
    carb_int{3}{a} = 0.1531
    carb_int{4}{a} = 2.3625
    carb_int{5}{a} = 0.1297
    carb_int{6}{a} = 0.0264
    carb_int{7}{a} = 0.1570
    carb_int{8}{a} = 0.0130
    carb_int{9}{a} = 0.0196
    carb_int{10}{a} = 0.0323


    omega_N{1}{a} = 1.681360e-01
    omega_N{2}{a} = 4.413774e-03
    omega_N{3}{a} = 1.714096e-01
    omega_N{4}{a} = 3.950297e-03
    omega_N{5}{a} = 6.099488e-03
    omega_N{6}{a} = 1.251327e-01
    omega_N{7}{a} = 3.137282e-01
    omega_N{8}{a} = 2.786590e-02
    omega_N{9}{a} = 2.039494e-01
    omega_N{10}{a} = 6.753223e-02
    

    omega_K{1}{a} = 1.219069e-01
    omega_K{2}{a} = 1.188118e-02
    omega_K{3}{a} = 1.879135e-01
    omega_K{4}{a} = 7.801404e-02
    omega_K{5}{a} = 1.113495e-01
    omega_K{6}{a} = 1.357246e-01
    omega_K{7}{a} = 2.127142e-01
    omega_K{8}{a} = 5.862381e-02
    omega_K{9}{a} = 1.329445e-01
    omega_K{10}{a} = 5.703369e-02
    
    # Sectoral shares in the consumption good bundle
    Psi_con{1}{a} = 3.188625e-02  
    Psi_con{2}{a} = 1.584363e-03
    Psi_con{3}{a} = 3.330783e-01
    Psi_con{4}{a} = 3.825251e-02
    Psi_con{5}{a} = 1.900225e-02
    Psi_con{6}{a} = 1.205428e-02
    Psi_con{7}{a} = 4.046963e-01
    Psi_con{8}{a} = 6.753211e-02
    Psi_con{9}{a} = 2.957865e-02
    Psi_con{10}{a} = 6.233497e-02
    
    # Sectoral shares in the investment good bundle
    Psi_inv{1}{a} = 3.849196e-03   
    Psi_inv{2}{a} = 9.947719e-04     
    Psi_inv{3}{a} = 2.620268e-01  
    Psi_inv{4}{a} = 4.328917e-03     
    Psi_inv{5}{a} = 1.012255e-03
    Psi_inv{6}{a} = 5.094560e-01  
    Psi_inv{7}{a} = 8.545419e-02   
    Psi_inv{8}{a} = 7.197562e-02 
    Psi_inv{9}{a} = 5.493709e-02    
    Psi_inv{10}{a} = 5.965217e-03  


    ## Sectoral interaction country level parameters
    # Weighting parameter for intermediate goods -> share of se2 used in H_se
    Psi{1}{1}{a} =    0.265315742774046 
    Psi{2}{1}{a} =    0.00305573269764116
    Psi{3}{1}{a} =    0.0505045140118459
    Psi{4}{1}{a} =    0.00220102788794719
    Psi{5}{1}{a} =    0.00134582434010930
    Psi{6}{1}{a} =    0.00170310880035921
    Psi{7}{1}{a} =    0.0106051738563881
    Psi{8}{1}{a} =    0.000914181717142059
    Psi{9}{1}{a} =    0.00272523783277506
    Psi{10}{1}{a} =   0.00490455678627902
    
    Psi{1}{2}{a} =    0.00160816898439458
    Psi{2}{2}{a} =    0.121292270636578
    Psi{3}{2}{a} =    0.0215420335683630
    Psi{4}{2}{a} =    0.123994431686331
    Psi{5}{2}{a} =    0.00270296898361670
    Psi{6}{2}{a} =    0.00564336039166173
    Psi{7}{2}{a} =    0.00159771339010712
    Psi{8}{2}{a} =    0.000684569457204696
    Psi{9}{2}{a} =    0.00124315031366425
    Psi{10}{2}{a} =   0.00104399754871801
    
    Psi{1}{3}{a} =    0.385536884585970 
    Psi{2}{3}{a} =    0.291473268464839
    Psi{3}{3}{a} =    0.541521084936126
    Psi{4}{3}{a} =    0.153802635821927 
    Psi{5}{3}{a} =    0.246475281659307
    Psi{6}{3}{a} =    0.363083901298632 
    Psi{7}{3}{a} =    0.219250453570809
    Psi{8}{3}{a} =    0.171002305312862
    Psi{9}{3}{a} =    0.121728104776474
    Psi{10}{3}{a} =   0.168364817317450
    
    Psi{1}{4}{a} =   0.0298324268818055 
    Psi{2}{4}{a} =    0.0932192696168212
    Psi{3}{4}{a} =    0.0295715382562291
    Psi{4}{4}{a} =    0.448688592630379
    Psi{5}{4}{a} =    0.0476125879325132
    Psi{6}{4}{a} =    0.00715961032816918
    Psi{7}{4}{a} =    0.0296275019873404
    Psi{8}{4}{a} =    0.0173341837524038
    Psi{9}{4}{a} =    0.0152506071297297
    Psi{10}{4}{a} =   0.0412150762435442
    
    Psi{1}{5}{a} =    0.0123141203627065
    Psi{2}{5}{a} =    0.0129555013336407
    Psi{3}{5}{a} =    0.0144054114824471
    Psi{4}{5}{a} =    0.0111432011491809
    Psi{5}{5}{a} =    0.238959641039029
    Psi{6}{5}{a} =    0.00436523069569921
    Psi{7}{5}{a} =    0.00915830617201352
    Psi{8}{5}{a} =    0.00501001729064976
    Psi{9}{5}{a} =    0.00845738567254073
    Psi{10}{5}{a} =   0.0189537086056638
    
    Psi{1}{6}{a} =    0.0163045847083446
    Psi{2}{6}{a} =    0.0504728517824097
    Psi{3}{6}{a} =    0.0105977676735216
    Psi{4}{6}{a} =    0.0416874501341756
    Psi{5}{6}{a} =    0.0512680110901663
    Psi{6}{6}{a} =    0.357265197894778
    Psi{7}{6}{a} =    0.0338783011070401
    Psi{8}{6}{a} =    0.0227337522344240
    Psi{9}{6}{a} =    0.0253908401644111
    Psi{10}{6}{a} =   0.0379248914579105
   
    Psi{1}{7}{a} =    0.201728152799602
    Psi{2}{7}{a} =    0.233013666844003
    Psi{3}{7}{a} =    0.206327558097361
    Psi{4}{7}{a} =    0.108417315489401
    Psi{5}{7}{a} =    0.150629508594779
    Psi{6}{7}{a} =    0.121987925876926
    Psi{7}{7}{a} =    0.427010856136804
    Psi{8}{7}{a} =    0.147709560097603
    Psi{9}{7}{a} =    0.145529504015973
    Psi{10}{7}{a} =   0.158037990364537
    
    Psi{1}{8}{a} =    0.00829073953775116
    Psi{2}{8}{a} =    0.0286746550659435
    Psi{3}{8}{a} =    0.0197149288703273
    Psi{4}{8}{a} =    0.0255292065965669
    Psi{5}{8}{a} =    0.0425643261942553
    Psi{6}{8}{a} =    0.0178673599991197
    Psi{7}{8}{a} =    0.0629035460903053
    Psi{8}{8}{a} =    0.363931927787456
    Psi{9}{8}{a} =    0.128841278808088
    Psi{10}{8}{a} =   0.107371228252792
    
    Psi{1}{9}{a} =    0.0730526136609058
    Psi{2}{9}{a} =    0.156446254495284
    Psi{3}{9}{a} =    0.100371495745349
    Psi{4}{9}{a} =    0.0781364158514192
    Psi{5}{9}{a} =    0.205701729650236
    Psi{6}{9}{a} =    0.116000738285638
    Psi{7}{9}{a} =    0.193091467402921
    Psi{8}{9}{a} =    0.235370063889126
    Psi{9}{9}{a} =    0.528410674365983
    Psi{10}{9}{a} =   0.234583667509100
    
    Psi{1}{10}{a} =    0.00601656570447395
    Psi{2}{10}{a} =    0.00939652906283916
    Psi{3}{10}{a} =    0.00544366735843085
    Psi{4}{10}{a} =    0.00639972275267240
    Psi{5}{10}{a} =    0.0127401205159877
    Psi{6}{10}{a} =    0.00492356642901725
    Psi{7}{10}{a} =    0.0128766802862714
    Psi{8}{10}{a} =    0.0353094384611289
    Psi{9}{10}{a} =    0.0224232169203602
    Psi{10}{10}{a} =   0.227600065914005
end

SS(NK_multisector)

get_solution(NK_multisector)

import MacroModelling: block_solver, find_SS_solver_parameters!, solve_ss, levenberg_marquardt
find_SS_solver_parameters!(NK_multisector, verbosity = 2, maxtime = 60)
SS_and_pars, (solution_error, iters) = NK_multisector.SS_solve_func(NK_multisector.parameter_values, NK_multisector, true, true, NK_multisector.solver_parameters)

NK_multisector.solution.non_stochastic_steady_state = SS_and_pars
NK_multisector.solution.outdated_NSSS = false

stst = SS(NK_multisector, derivatives= false)
stst("Y_t{a}")

get_solution(NK_multisector)
get_irf(NK_multisector)

# using StatsPlots
# plot_irf(NK_multisector)

# SS_and_pars, (solution_error, iters) = NK_multisector.SS_solve_func(NK_multisector.parameter_values, NK_multisector, true, true, NK_multisector.solver_parameters)

# ∇₁ = calculate_jacobian(NK_multisector.parameter_values, SS_and_pars, NK_multisector)# |> Matrix
       
# S₁, qme_sol, solved = calculate_first_order_solution(∇₁; T = 𝓂.timings)
         

# NK_multisector.SS_solve_func

# NK_multisector.ss_solve_blocks[6]

# import MacroModelling: block_solver, find_SS_solver_parameters!, solve_ss, levenberg_marquardt
# 𝓂 = NK_multisector
# initial_parameters = NK_multisector.parameter_values
# solver_parameters = 𝓂.solver_parameters
# cold_start = true
# verbose = true
 
#  initial_parameters = if typeof(initial_parameters) == Vector{Float64}
#                   initial_parameters
#               else
#                   ℱ.value.(initial_parameters)
#               end
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3179 =#
#           parameters = copy(initial_parameters)
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3180 =#
#           params_flt = copy(initial_parameters)
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3182 =#
#           current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3183 =#
#           closest_solution_init = 𝓂.NSSS_solver_cache[end]
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3185 =#
#           for pars = 𝓂.NSSS_solver_cache
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3186 =#
#               latest = sum(abs2, pars[end] - initial_parameters)
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3187 =#
#               if latest <= current_best
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3188 =#
#                   current_best = latest
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3189 =#
#                   closest_solution_init = pars
#               end
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3191 =#
#           end
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3196 =#
#           range_iters = 0
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3197 =#
#           solution_error = 1.0
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3198 =#
#           solved_scale = 0
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3200 =#
#           scale = 1.0
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3202 =#
#         #   while range_iters < if cold_start
#                     #       1
#                     #   else
#                     #       500
#                     #   end && !(solution_error < 1.0e-12 && solved_scale == 1)
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3203 =#
#               range_iters += 1
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3215 =#
#               current_best = sum(abs2, (𝓂.NSSS_solver_cache[end])[end] - initial_parameters)
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3216 =#
#               closest_solution = 𝓂.NSSS_solver_cache[end]
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3218 =#
#               for pars = 𝓂.NSSS_solver_cache
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3219 =#
#                   latest = sum(abs2, pars[end] - initial_parameters)
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3220 =#
#                   if latest <= current_best
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3221 =#
#                       current_best = latest
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3222 =#
#                       closest_solution = pars
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3224 =#
#               end
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3229 =#
#               if all(isfinite, closest_solution[end]) && initial_parameters != closest_solution_init[end]
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3230 =#
#                   parameters = scale * initial_parameters + (1 - scale) * closest_solution_init[end]
#               else
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3232 =#
#                   parameters = copy(initial_parameters)
#               end
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3234 =#
#               params_flt = parameters
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3238 =#
#               betta = parameters[1]
#               delta = parameters[2]
#               sig = parameters[4]
#               epsi_ts◖a◗ = parameters[6]
#               lab◖a◗ = parameters[7]
#               rho_eps◖a◗ = parameters[8]
#               sigc◖a◗ = parameters[9]
#               sigi◖a◗ = parameters[10]
#               upsi_K◖a◗ = parameters[11]
#               upsi_N◖a◗ = parameters[12]
#               epsi_ts◖1◗◖a◗ = parameters[13]
#               epsi_ts◖2◗◖a◗ = parameters[14]
#               epsi_ts◖3◗◖a◗ = parameters[15]
#               epsi_ts◖a◗ = parameters[16]
#               sigh◖1◗◖a◗ = parameters[20]
#               sigh◖2◗◖a◗ = parameters[21]
#               sigh◖3◗◖a◗ = parameters[22]
#               kappap◖1◗◖a◗ = parameters[23]
#               kappap◖2◗◖a◗ = parameters[24]
#               kappap◖3◗◖a◗ = parameters[25]
#               thetap◖1◗◖a◗ = parameters[26]
#               thetap◖2◗◖a◗ = parameters[27]
#               thetap◖3◗◖a◗ = parameters[28]
#               alphaH◖1◗◖a◗ = parameters[36]
#               alphaH◖2◗◖a◗ = parameters[37]
#               alphaH◖3◗◖a◗ = parameters[38]
#               alphaN◖1◗◖a◗ = parameters[46]
#               alphaN◖2◗◖a◗ = parameters[47]
#               alphaN◖3◗◖a◗ = parameters[48]
#               omega_N◖1◗◖a◗ = parameters[66]
#               omega_N◖2◗◖a◗ = parameters[67]
#               omega_N◖3◗◖a◗ = parameters[68]
#               omega_K◖1◗◖a◗ = parameters[76]
#               omega_K◖2◗◖a◗ = parameters[77]
#               omega_K◖3◗◖a◗ = parameters[78]
#               Psi_con◖1◗◖a◗ = parameters[86]
#               Psi_con◖2◗◖a◗ = parameters[87]
#               Psi_con◖3◗◖a◗ = parameters[88]
#               Psi_inv◖1◗◖a◗ = parameters[96]
#               Psi_inv◖2◗◖a◗ = parameters[97]
#               Psi_inv◖3◗◖a◗ = parameters[98]
#               Psi◖1◗◖1◗◖a◗ = parameters[106]
#               Psi◖2◗◖1◗◖a◗ = parameters[107]
#               Psi◖3◗◖1◗◖a◗ = parameters[108]
#               Psi◖1◗◖2◗◖a◗ = parameters[116]
#               Psi◖2◗◖2◗◖a◗ = parameters[117]
#               Psi◖3◗◖2◗◖a◗ = parameters[118]
#               Psi◖1◗◖3◗◖a◗ = parameters[126]
#               Psi◖2◗◖3◗◖a◗ = parameters[127]
#               Psi◖3◗◖3◗◖a◗ = parameters[128]
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3239 =#
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3240 =#
#               R_ts◖a◗ = 1 / betta
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3241 =#
#               NSSS_solver_cache_tmp = []
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3242 =#
#               solution_error = 0.0
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3243 =#
#               iters = 0
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3244 =#
#               params_and_solved_vars = [rho_eps◖a◗, epsi_ts◖1◗◖a◗]
#               lbs = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[1])], min.(ubs[1:length(closest_solution[1])], closest_solution[1])), closest_solution[2]]
#               solution = block_solver(params_and_solved_vars, 1, 𝓂.ss_solve_blocks[1], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               sol = solution[1]
#               epsi_t◖1◗◖a◗ = sol[1]
#               ➕₁₆ = sol[2]
#               ➕₁₇ = sol[3]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               params_and_solved_vars = [betta, R_ts◖a◗]
#               lbs = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[3])], min.(ubs[1:length(closest_solution[3])], closest_solution[3])), closest_solution[4]]
#               solution = block_solver(params_and_solved_vars, 2, 𝓂.ss_solve_blocks[2], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               sol = solution[1]
#               R_t◖a◗ = sol[1]
#               pi_cpi_t◖a◗ = sol[2]
#               ➕₃ = sol[3]
#               ➕₄ = sol[4]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               N_t◖a◗ = 0.333333333333333
#               params_and_solved_vars = [rho_eps◖a◗, epsi_ts◖3◗◖a◗]
#               lbs = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[5])], min.(ubs[1:length(closest_solution[5])], closest_solution[5])), closest_solution[6]]
#               solution = block_solver(params_and_solved_vars, 3, 𝓂.ss_solve_blocks[3], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               sol = solution[1]
#               epsi_t◖3◗◖a◗ = sol[1]
#               ➕₃₆ = sol[2]
#               ➕₃₇ = sol[3]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               params_and_solved_vars = [epsi_ts◖a◗, rho_eps◖a◗]
#               lbs = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[7])], min.(ubs[1:length(closest_solution[7])], closest_solution[7])), closest_solution[8]]
#               solution = block_solver(params_and_solved_vars, 4, 𝓂.ss_solve_blocks[4], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               sol = solution[1]
#               epsi_t◖a◗ = sol[1]
#               ➕₁ = sol[2]
#               ➕₂ = sol[3]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               params_and_solved_vars = [rho_eps◖a◗, epsi_ts◖2◗◖a◗]
#               lbs = [-1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[9])], min.(ubs[1:length(closest_solution[9])], closest_solution[9])), closest_solution[10]]
#               solution = block_solver(params_and_solved_vars, 5, 𝓂.ss_solve_blocks[5], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               sol = solution[1]
#               epsi_t◖2◗◖a◗ = sol[1]
#               ➕₂₆ = sol[2]
#               ➕₂₇ = sol[3]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               ➕₃₈ = min(1.0e12, max(eps(), pi_cpi_t◖a◗))
#               params_and_solved_vars = [betta, delta, sigc◖a◗, sigi◖a◗, upsi_K◖a◗, upsi_N◖a◗, sigh◖1◗◖a◗, sigh◖2◗◖a◗, sigh◖3◗◖a◗, kappap◖1◗◖a◗, kappap◖2◗◖a◗, kappap◖3◗◖a◗, thetap◖1◗◖a◗, thetap◖2◗◖a◗, thetap◖3◗◖a◗, alphaH◖1◗◖a◗, alphaH◖2◗◖a◗, alphaH◖3◗◖a◗, alphaN◖1◗◖a◗, alphaN◖2◗◖a◗, alphaN◖3◗◖a◗, omega_N◖1◗◖a◗, omega_N◖2◗◖a◗, omega_N◖3◗◖a◗, omega_K◖1◗◖a◗, omega_K◖2◗◖a◗, omega_K◖3◗◖a◗, Psi_con◖1◗◖a◗, Psi_con◖2◗◖a◗, Psi_con◖3◗◖a◗, Psi_inv◖1◗◖a◗, Psi_inv◖2◗◖a◗, Psi_inv◖3◗◖a◗, Psi◖1◗◖1◗◖a◗, Psi◖2◗◖1◗◖a◗, Psi◖3◗◖1◗◖a◗, Psi◖1◗◖2◗◖a◗, Psi◖2◗◖2◗◖a◗, Psi◖3◗◖2◗◖a◗, Psi◖1◗◖3◗◖a◗, Psi◖2◗◖3◗◖a◗, Psi◖3◗◖3◗◖a◗, epsi_t◖1◗◖a◗, epsi_t◖2◗◖a◗, epsi_t◖3◗◖a◗, epsi_t◖a◗, pi_cpi_t◖a◗, ➕₃₈]
#               lbs = [-1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, 2.220446049250313e-16, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, -1.0e12, 2.220446049250313e-16, 2.220446049250313e-16]
#               ubs = [1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12, 1.0e12]
#               inits = [max.(lbs[1:length(closest_solution[11])], min.(ubs[1:length(closest_solution[11])], randn(length(closest_solution[11])))), params_and_solved_vars]

#               𝓂.ss_solve_blocks[6](params_and_solved_vars, max.(lbs[1:length(closest_solution[11])], min.(ubs[1:length(closest_solution[11])], rand(length(closest_solution[11])))))

# i = 0
# while i < 5000
# init = max.(lbs[1:length(closest_solution[11])], min.(ubs[1:length(closest_solution[11])], rand(length(closest_solution[11]))))
#               sol_values, total_iters, rel_sol_minimum, sol_minimum = solve_ss(levenberg_marquardt, 𝓂.ss_solve_blocks[6], params_and_solved_vars, init, lbs, ubs, 1e-14, [0,0], 6, verbose,
#               init, 
#               solver_parameters[1],
#               false,
#               false)
#               if sol_minimum < 1e-7 println("found it"); break end
#               i += 1
# end


#               solution = block_solver(params_and_solved_vars, 6, 𝓂.ss_solve_blocks[6], inits, lbs, ubs, solver_parameters, cold_start, verbose)
#               iters += (solution[2])[2]
#               solution_error += (solution[2])[1]
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                       println("Failed after solving block with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2431 =#
#                   # continue
#               end
#               solution_error += +(abs(➕₃₈ - pi_cpi_t◖a◗))
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2435 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2435 =#
#                       println("Failed for aux variables with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2435 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:2435 =#
#                   # continue
#               end
#               sol = solution[1]
#               C_t◖1◗◖a◗ = sol[1]
#               C_t◖2◗◖a◗ = sol[2]
#               C_t◖3◗◖a◗ = sol[3]
#               C_t◖a◗ = sol[4]
#               H_t◖1◗◖1◗◖a◗ = sol[5]
#               H_t◖1◗◖2◗◖a◗ = sol[6]
#               H_t◖1◗◖3◗◖a◗ = sol[7]
#               H_t◖1◗◖a◗ = sol[8]
#               H_t◖2◗◖1◗◖a◗ = sol[9]
#               H_t◖2◗◖2◗◖a◗ = sol[10]
#               H_t◖2◗◖3◗◖a◗ = sol[11]
#               H_t◖2◗◖a◗ = sol[12]
#               H_t◖3◗◖1◗◖a◗ = sol[13]
#               H_t◖3◗◖2◗◖a◗ = sol[14]
#               H_t◖3◗◖3◗◖a◗ = sol[15]
#               H_t◖3◗◖a◗ = sol[16]
#               I_t◖1◗◖a◗ = sol[17]
#               I_t◖2◗◖a◗ = sol[18]
#               I_t◖3◗◖a◗ = sol[19]
#               I_t◖a◗ = sol[20]
#               K_t◖1◗◖a◗ = sol[21]
#               K_t◖2◗◖a◗ = sol[22]
#               K_t◖3◗◖a◗ = sol[23]
#               K_t◖a◗ = sol[24]
#               N_t◖1◗◖a◗ = sol[25]
#               N_t◖2◗◖a◗ = sol[26]
#               N_t◖3◗◖a◗ = sol[27]
#               PH_t◖1◗◖a◗ = sol[28]
#               PH_t◖2◗◖a◗ = sol[29]
#               PH_t◖3◗◖a◗ = sol[30]
#               PI_t◖a◗ = sol[31]
#               P_t◖1◗◖a◗ = sol[32]
#               P_t◖2◗◖a◗ = sol[33]
#               P_t◖3◗◖a◗ = sol[34]
#               Y_t◖1◗◖a◗ = sol[35]
#               Y_t◖2◗◖a◗ = sol[36]
#               Y_t◖3◗◖a◗ = sol[37]
#               mc_tild_t◖1◗◖a◗ = sol[38]
#               mc_tild_t◖2◗◖a◗ = sol[39]
#               mc_tild_t◖3◗◖a◗ = sol[40]
#               mc_t◖1◗◖a◗ = sol[41]
#               mc_t◖2◗◖a◗ = sol[42]
#               mc_t◖3◗◖a◗ = sol[43]
#               pi_ppi_t◖1◗◖a◗ = sol[44]
#               pi_ppi_t◖2◗◖a◗ = sol[45]
#               pi_ppi_t◖3◗◖a◗ = sol[46]
#               rk_t◖1◗◖a◗ = sol[47]
#               rk_t◖2◗◖a◗ = sol[48]
#               rk_t◖3◗◖a◗ = sol[49]
#               rk_t◖a◗ = sol[50]
#               w_t◖1◗◖a◗ = sol[51]
#               w_t◖2◗◖a◗ = sol[52]
#               w_t◖3◗◖a◗ = sol[53]
#               w_t◖a◗ = sol[54]
#               ➕₁₀ = sol[55]
#               ➕₁₁ = sol[56]
#               ➕₁₂ = sol[57]
#               ➕₁₃ = sol[58]
#               ➕₁₄ = sol[59]
#               ➕₁₅ = sol[60]
#               ➕₁₈ = sol[61]
#               ➕₁₉ = sol[62]
#               ➕₂₀ = sol[63]
#               ➕₂₁ = sol[64]
#               ➕₂₂ = sol[65]
#               ➕₂₃ = sol[66]
#               ➕₂₄ = sol[67]
#               ➕₂₅ = sol[68]
#               ➕₂₈ = sol[69]
#               ➕₂₉ = sol[70]
#               ➕₃₀ = sol[71]
#               ➕₃₁ = sol[72]
#               ➕₃₂ = sol[73]
#               ➕₃₃ = sol[74]
#               ➕₃₄ = sol[75]
#               ➕₃₅ = sol[76]
#               ➕₅ = sol[77]
#               ➕₆ = sol[78]
#               ➕₇ = sol[79]
#               ➕₈ = sol[80]
#               ➕₉ = sol[81]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(sol) == Vector{Float64}
#                           sol
#                       else
#                           ℱ.value.(sol)
#                       end]
#               NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., if typeof(params_and_solved_vars) == Vector{Float64}
#                           params_and_solved_vars
#                       else
#                           ℱ.value.(params_and_solved_vars)
#                       end]
#               Y_t◖a◗ = Y_t◖1◗◖a◗ + Y_t◖2◗◖a◗ + Y_t◖3◗◖a◗
#               ➕₃₉ = min(1.0e12, max(eps(), C_t◖a◗))
#               solution_error += +(abs(➕₃₉ - C_t◖a◗))
#               if solution_error > 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3015 =#
#                   if verbose
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3015 =#
#                       println("Failed for analytical variables with error $(solution_error)")
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3015 =#
#                   scale = (scale + 2solved_scale) / 3
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3015 =#
#                   # continue
#               end
#               lambda_t◖a◗ = ➕₃₉ ^ -sig
#               kappaN◖a◗ = 3.0 ^ lab◖a◗ * lambda_t◖a◗ * w_t◖a◗
#               Y_VA_t◖a◗ = C_t◖a◗ + I_t◖a◗ * PI_t◖a◗
#               if length(NSSS_solver_cache_tmp) == 0
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3139 =#
#                   NSSS_solver_cache_tmp = [copy(params_flt)]
#               else
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3139 =#
#                   NSSS_solver_cache_tmp = [NSSS_solver_cache_tmp..., copy(params_flt)]
#               end
#               if current_best > 1.0e-8 && solution_error < 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3150 =#
#                   reverse_diff_friendly_push!(𝓂.NSSS_solver_cache, NSSS_solver_cache_tmp)
#               end
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3246 =#
#               if solution_error < 1.0e-12
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3248 =#
#                   solved_scale = scale
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3249 =#
#                   if scale == 1
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3253 =#
#                       return ([C_t◖1◗◖a◗, C_t◖2◗◖a◗, C_t◖3◗◖a◗, C_t◖a◗, H_t◖1◗◖1◗◖a◗, H_t◖1◗◖2◗◖a◗, H_t◖1◗◖3◗◖a◗, H_t◖1◗◖a◗, H_t◖2◗◖1◗◖a◗, H_t◖2◗◖2◗◖a◗, H_t◖2◗◖3◗◖a◗, H_t◖2◗◖a◗, H_t◖3◗◖1◗◖a◗, H_t◖3◗◖2◗◖a◗, H_t◖3◗◖3◗◖a◗, H_t◖3◗◖a◗, I_t◖1◗◖a◗, I_t◖2◗◖a◗, I_t◖3◗◖a◗, I_t◖a◗, K_t◖1◗◖a◗, K_t◖2◗◖a◗, K_t◖3◗◖a◗, K_t◖a◗, N_t◖1◗◖a◗, N_t◖2◗◖a◗, N_t◖3◗◖a◗, N_t◖a◗, PH_t◖1◗◖a◗, PH_t◖2◗◖a◗, PH_t◖3◗◖a◗, PI_t◖a◗, P_t◖1◗◖a◗, P_t◖2◗◖a◗, P_t◖3◗◖a◗, R_t◖a◗, Y_VA_t◖a◗, Y_t◖1◗◖a◗, Y_t◖2◗◖a◗, Y_t◖3◗◖a◗, Y_t◖a◗, epsi_t◖1◗◖a◗, epsi_t◖2◗◖a◗, epsi_t◖3◗◖a◗, epsi_t◖a◗, lambda_t◖a◗, mc_tild_t◖1◗◖a◗, mc_tild_t◖2◗◖a◗, mc_tild_t◖3◗◖a◗, mc_t◖1◗◖a◗, mc_t◖2◗◖a◗, mc_t◖3◗◖a◗, pi_cpi_t◖a◗, pi_ppi_t◖1◗◖a◗, pi_ppi_t◖2◗◖a◗, pi_ppi_t◖3◗◖a◗, rk_t◖1◗◖a◗, rk_t◖2◗◖a◗, rk_t◖3◗◖a◗, rk_t◖a◗, w_t◖1◗◖a◗, w_t◖2◗◖a◗, w_t◖3◗◖a◗, w_t◖a◗, kappaN◖a◗], (solution_error, iters))
#                   end
#                   #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3256 =#
#                   if scale > 0.95
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3257 =#
#                       scale = 1
#                   else
#                       #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3259 =#
#                       scale = (scale + 1) / 2
#                   end
#               end
#               #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3269 =#
#           end
#           #= /home/cdsw/Github/MacroModelling.jl/src/MacroModelling.jl:3270 =#
#           return ([0.0], (1, 0))
#       end))