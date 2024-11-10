using Revise
using MacroModelling

@model NK_multisector begin
    #### Climate variables ####
    for co in [a]
        P_EM_t{co}[0] = P_EM_ts{co} + shock_P_EM_t{co}[x] # Law of motion for carbon price 

        log(epsi_t{co}[0] / epsi_ts{co}) = rho_eps{co} * log(epsi_t{co}[-1] / epsi_ts{co}) - shock_epsi_t{co}[x] # Law of motion for aggregate TFP
    end

	EM_t[0] = (1 - rho_EM) * EM_t[-1] + for co in [a] for se in 1:10 ZZ_t{se}{co}[0] end end # Law of motion for emissions concentration

    for co in [a] 
        for se in 1:10   
            Pen_t{se}{co}[0] = gama0{se}{co} + EM_t[0] * gama1{se}{co} + gama2{se}{co} * EM_t[0] ^ 2  # Damage function
            ZZ_t{se}{co}[0] = (1 + shock_epsi_carb_int_t{co}[x]) * carb_int{se}{co} * Y_t{se}{co}[0] # Emissions
        end 
    end

    for co in [a] 
        ZZ_t{co}[0] = for se in 1:10 ZZ_t{se}{co}[0] end
    end
    
    #### Domestic aggregates ####
    for co in [a] 
        log(R_t{co}[0] / R_ts{co}) = 0.8 * log(R_t{co}[-1] / R_ts{co}) + 0.3 * log(pi_cpi_t{co}[-1]) # Modified Taylor-rule

        K_t{co}[0] = (1 - delta) * K_t{co}[-1] + I_t{co}[0] # Law of motion for capital

        lambda_t{co}[0] = C_t{co}[0] ^ (-sig) # FOC wrt consumption

        1 = R_t{co}[0] * betta * lambda_t{co}[1] / lambda_t{co}[0] / pi_cpi_t{co}[1] # HH FOC wrt bonds

        1 = betta * lambda_t{co}[1] / lambda_t{co}[0] * (rk_t{co}[1] + (1 - delta) * PI_t{co}[1]) / PI_t{co}[0] # No arbitrage cond. plus FOC wrt investment/capital

        lambda_t{co}[0] * w_t{co}[0] = kappaN{co} * N_t{co}[0] ^ lab{co} # HH FOC wrt labor

        1 = (for se in 1:10 Psi_con{se}{co} * P_t{se}{co}[0] ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # Price index derived from opt. problem of cons.-bundler 

        pi_cpi_t{co}[0] = (for se in 1:10 Psi_con{se}{co} * (pi_ppi_t{se}{co}[0] * P_t{se}{co}[-1]) ^ (( - sigc{co}) / (1 - sigc{co})) end) ^ (( - (1 - sigc{co})) / sigc{co}) # CPI-inflation 

        PI_t{co}[0] = (for se in 1:10 Psi_inv{se}{co} * P_t{se}{co}[0] ^ (( - sigi{co}) / (1 - sigi{co})) end) ^ (( - (1 - sigi{co})) / sigi{co}) # Price index derived from opt. problem of inv.-bundler 

        w_t{co}[0] = (for se in 1:10 omega_N{se}{co} * w_t{se}{co}[0] ^ (( - upsi_N{co}) / (1 - upsi_N{co})) end) ^ (( - (1 - upsi_N{co})) / upsi_N{co}) # Wage index derived from opt. problem of labor-bundler

        rk_t{co}[0] = (for se in 1:10 omega_K{se}{co} * rk_t{se}{co}[0] ^ (( - upsi_K{co}) / (1 - upsi_K{co})) end) ^ (( - (1 - upsi_K{co})) / upsi_K{co}) # Rental rate index derived from opt. problem of capital-bundler

        Y_VA_t{co}[0] = C_t{co}[0] + I_t{co}[0] * PI_t{co}[0]

        Y_t{co}[0] = for se in 1:10 Y_t{se}{co}[0] end

        #### Consumption and investmend demand for different goods, relative prices ####
        for se in 1:10
            C_t{se}{co}[0] = C_t{co}[0] * Psi_con{se}{co} * (1 / P_t{se}{co}[0]) ^ (1 / (1 - sigc{co})) # Specific demand for sectoral consumption goods

            I_t{se}{co}[0] = I_t{co}[0] * Psi_inv{se}{co} * (PI_t{co}[0] / P_t{se}{co}[0]) ^ (1 / (1 - sigi{co})) # Specific demand for sectoral investment goods

            N_t{se}{co}[0] = N_t{co}[0] * omega_N{se}{co} * (w_t{co}[0] / w_t{se}{co}[0]) ^ (1 / (1 - upsi_N{co})) # Specific demand for labor types

            K_t{se}{co}[0] = K_t{co}[0] * omega_K{se}{co} * (rk_t{co}[1] / rk_t{se}{co}[1]) ^ (1 / (1 - upsi_K{co})) # Specific demand for capital types   

            Y_t{se}{co}[0] = C_t{se}{co}[0] + I_t{se}{co}[0] + Y_t{se}{co}[0] * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[1] - 1) ^ 2 + for se2 in 1:10 H_t{se2}{se}{co}[0] end

            Y_t{se}{co}[0] = epsi_t{co}[0] * epsi_t{se}{co}[0] * (1 - Pen_t{se}{co}[0]) * (N_t{se}{co}[0] ^ alphaN{se}{co} * K_t{se}{co}[-1] ^ (1 - alphaN{se}{co})) ^ alphaH{se}{co} * H_t{se}{co}[0] ^ (1 - alphaH{se}{co}) # Production technology of perfectly competitive firm

            EM_cost_t{se}{co}[0] = P_EM_t{co}[0] * (1 + shock_epsi_carb_int_t{co}[x]) * carb_int{se}{co} # Emissions costs

            mc_tild_t{se}{co}[0] = EM_cost_t{se}{co}[0] + mc_t{se}{co}[0] # Real marginal costs relevant for pricing (including emissions taxes)

            1 - thetap{se}{co} + mc_tild_t{se}{co}[0] * thetap{se}{co} / P_t{se}{co}[0] + (pi_ppi_t{se}{co}[1] - 1) * betta * lambda_t{co}[1] / lambda_t{co}[0] * kappap{se}{co} * pi_ppi_t{se}{co}[1] ^ 2 / pi_cpi_t{co}[1] * Y_t{se}{co}[1] / Y_t{se}{co}[0] = pi_ppi_t{se}{co}[0] * kappap{se}{co} * (pi_ppi_t{se}{co}[0] - 1) + (1 - thetap{se}{co}) * kappap{se}{co} / 2 * (pi_ppi_t{se}{co}[0] - 1) ^ 2 # Firm's pricing decision in case of nominal price rigidities

            w_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * alphaN{se}{co} * alphaH{se}{co} / N_t{se}{co}[0] # FOC firm cost minimization -> labor

            rk_t{se}{co}[0] = Y_t{se}{co}[0] * mc_t{se}{co}[0] * (1 - alphaN{se}{co}) * alphaH{se}{co} / K_t{se}{co}[-1] # FOC firm cost minimization -> capital

            PH_t{se}{co}[0] = Y_t{se}{co}[0] * (1 - alphaH{se}{co}) * mc_t{se}{co}[0] / H_t{se}{co}[0] # FOC firm cost minimization -> intermediates

            PH_t{se}{co}[0] = (for se2 in 1:10 Psi{se}{se2}{co} * P_t{se2}{co}[0] ^ (( - sigh{se}{co}) / (1 - sigh{se}{co})) end) ^ (( - (1 - sigh{se}{co})) / sigh{se}{co}) # Price index derived from opt. problem of intermediate input bundler in sector se

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
    betta   = 0.992 # Discount
    
    delta   = 0.025 # Depr. rate
    
    rho_EM  = 0.0021 # Rate of decay wrt atmosph. carbon concentration

    sig   = 2 # Intertemporal elasticity of substitution

    ## Country level parameters
    P_EM_ts{a} = 1e-07 # Initial carbon price level

    R_ts{a} = 1/betta # Nominal interest rate <-> real interest rate 

    epsi_ts{a} = 1 # Aggregate TFP shock
    
    kappaN{a} = 0.1 # endogenous

    lab{a} = 2 # Inverse Frisch elasticity of labor supply

    rho_eps{a}  = 0.8 # Persistence TFP shock

    sigc{a} = -0.099989000109999 # Determines elasticity of substitution between sector-level consumption goods 

    sigi{a} = -0.331380641725469 # Determines elasticity of substitution between sector-level investment goods 

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


    omega_N{1}{a} = 0.08383020412010535
    omega_N{2}{a} = 0.005004514234409534
    omega_N{3}{a} = 0.2215786546591248
    omega_N{4}{a} = 0.008139785715816689
    omega_N{5}{a} = 0.009060243313662392
    omega_N{6}{a} = 0.10129796616912531
    omega_N{7}{a} = 0.3359103054274722
    omega_N{8}{a} = 0.036419963307039255
    omega_N{9}{a} = 0.14188891160385741
    omega_N{10}{a} = 0.056869451449386994
    

    omega_K{1}{a} = 0.0641270747425581
    omega_K{2}{a} = 0.017840474868937792
    omega_K{3}{a} = 0.2334610401317096
    omega_K{4}{a} = 0.07164561233067956
    omega_K{5}{a} = 0.0535568702148968
    omega_K{6}{a} = 0.08613798878388712
    omega_K{7}{a} = 0.2597785523090172
    omega_K{8}{a} = 0.061272330275292775
    omega_K{9}{a} = 0.10600538188918682
    omega_K{10}{a} = 0.04617467445383404
    
    # Sectoral shares in the consumption good bundle
    Psi_con{1}{a} = 0.0328     
    Psi_con{2}{a} = 0.0021 
    Psi_con{3}{a} = 0.3570 
    Psi_con{4}{a} = 0.0360 
    Psi_con{5}{a} = 0.0177 
    Psi_con{6}{a} = 0.0113 
    Psi_con{7}{a} = 0.3878 
    Psi_con{8}{a} = 0.0659 
    Psi_con{9}{a} = 0.0297 
    Psi_con{10}{a} = 0.0596
    
    # Sectoral shares in the investment good bundle
    Psi_inv{1}{a} = 0.0037     
    Psi_inv{2}{a} = 0.0012     
    Psi_inv{3}{a} = 0.2961     
    Psi_inv{4}{a} = 0.0041     
    Psi_inv{5}{a} = 0.0009     
    Psi_inv{6}{a} = 0.4827     
    Psi_inv{7}{a} = 0.0821     
    Psi_inv{8}{a} = 0.0706     
    Psi_inv{9}{a} = 0.0529     
    Psi_inv{10}{a} = 0.0057    


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
