include("models/SGU_2003_debt_premium.jl")
moments = get_moments(SGU_2003_debt_premium, derivatives = false)

@test isapprox(moments[1][4:8],[0.7442,exp(0.00739062),exp(-1.07949),exp(1.22309),exp(-3.21888)],rtol = 1e-5)

corrr = get_correlation(SGU_2003_debt_premium)

@test isapprox(corrr(:i,:r),0.0114,rtol = 1e-3)
@test isapprox(corrr(:i,:k),0.4645,rtol = 1e-3)




include("models/JQ_2012_RBC.jl")
moments = get_moments(JQ_2012_RBC, derivatives = false)

@test isapprox(moments[1][1:6],[1.01158,3.63583,0.811166,0.114801,0.251989,10.0795],rtol = 1e-5)

var_dec = get_variance_decomposition(JQ_2012_RBC)

@test isapprox(var_dec(:R,:) * 100, [67.43,32.57],rtol = 1e-4)
@test isapprox(var_dec(:k,:) * 100, [17.02,82.98],rtol = 1e-4)
@test isapprox(var_dec(:c,:) * 100, [13.12,86.88],rtol = 1e-4)



include("models/Ireland_2004.jl")
moments = get_moments(Ireland_2004, derivatives = false)

@test isapprox(moments[2],[0.0709,0.0015,0.0077,0.0153,0.0075,0.0163,0.0062],atol = 1e-4)

var_dec = get_variance_decomposition(Ireland_2004)

@test isapprox(var_dec(:π̂,:) * 100, [4.51, 0.91, 87.44, 7.14],rtol = 1e-4)




include("models/GNSS_2010.jl")
moments = get_moments(GNSS_2010, derivatives = false)

@test isapprox(moments[1]([:x,:C,:Y,:D,:BE,:BH,:B],:),exp.([0.182322,0.1305,0.273583,1.04257,0.674026,0.144027,1.13688]),atol = 1e-4)

var_dec = get_var_decomp(GNSS_2010)

@test isapprox(var_dec(:B,:) * 100, [42.97, 19.31, 11.70,  0.36,  4.45,  1.41,  0.70,  0.00,  0.61,  12.54, 2.85, 2.72,  0.38],rtol = 1e-3)





include("models/Ghironi_Melitz_2005.jl")
moments = get_moments(Ghironi_Melitz_2005, derivatives = false)

@test isapprox(moments[1]([:Nd,:Nd̄,:Ne,:Nē,:Nx,:Nx̄],:),[7.50695, 7.50695,0.192486, 0.192486, 1.57988, 1.57988],atol = 1e-4)

var_dec = get_var_decomp(Ghironi_Melitz_2005)

@test isapprox(var_dec(:z̃x,:) * 100, [96.05, 3.95],rtol = 1e-3)
@test isapprox(var_dec(:C,:) * 100, [99.86, 0.14],rtol = 1e-3)
@test isapprox(var_dec(:Nx,:) * 100, [29.27, 70.73],rtol = 1e-3)



include("models/Gali_Monacelli_2005_CITR.jl")
moments = get_moments(Gali_Monacelli_2005_CITR, derivatives = false)

@test isapprox(moments[2]([:a,:r,:s,:pi],:),[2.2942, 0.4943, 2.7590, 0.3295],atol = 1e-4)

var_dec = get_var_decomp(Gali_Monacelli_2005_CITR)

@test isapprox(var_dec(:pih,:) * 100, [62.06, 37.94],rtol = 1e-3)
@test isapprox(var_dec(:x,:) * 100, [56.22, 43.78],rtol = 1e-3)





include("models/Gali_2015_chapter_3_nonlinear.jl")
moments = get_moments(Gali_2015_chapter_3_nonlinear, derivatives = false)

@test isapprox(moments[1]([:C,:N,:M_real,:Y],:),[0.95058, 0.934655, 0.915236, 0.95058],atol = 1e-4)

var_dec = get_var_decomp(Gali_2015_chapter_3_nonlinear)

@test isapprox(var_dec(:C,:) * 100, [27.53, 0.72,  71.76],rtol = 1e-3)
@test isapprox(var_dec(:R,:) * 100, [15.37, 0.23, 84.40],rtol = 1e-3)
@test isapprox(var_dec(:Y,:) * 100, [27.53, 0.72, 71.76],rtol = 1e-3)




include("models/Caldara_et_al_2012.jl")
moments = get_moments(Caldara_et_al_2012, derivatives = false)

@test isapprox(moments[1]([:Rᵏ,:V,:c,:i],:),[0.00908174, 0.687139, 0.724731, 0.18689],atol = 1e-4)

@test isapprox(moments[2]([:Rᵏ,:V,:c,:i],:),[0.0022, 0.0061, 0.0566, 0.0528],atol = 1e-4)




include("models/Ascari_Sbordone_2014.jl")
moments = get_moments(Ascari_Sbordone_2014, derivatives = false)

@test isapprox(moments[1]([:i,:y,:psi,:phi],:),exp.([-4.59512, -1.09861, 1.25138, 1.35674]),atol = 1e-3)

var_dec = get_var_decomp(Ascari_Sbordone_2014)

@test isapprox(var_dec(:w,:) * 100, [2.02, 95.18, 2.80],rtol = 1e-3)
@test isapprox(var_dec(:y,:) * 100, [0.47, 99.41, 0.12],rtol = 1e-3)




include("models/Aguiar_Gopinath_2007.jl")
moments = get_moments(Aguiar_Gopinath_2007, derivatives = false)

@test isapprox(moments[1]([:b,:c,:u,:ul],:),[0.0645177, 0.496156, -1.66644, -1.597],atol = 1e-3)

var_dec = get_var_decomp(Aguiar_Gopinath_2007)

@test isapprox(var_dec(:c,:) * 100, [0.02, 99.98],rtol = 1e-3)
@test isapprox(var_dec(:i_y,:) * 100, [0.31, 99.69],rtol = 1e-3)