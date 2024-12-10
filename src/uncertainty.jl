using Revise
using MacroModelling
using StatsPlots
using Optim, LineSearches
using Optimization, OptimizationNLopt#, OptimizationOptimJL
using Zygote, ForwardDiff, FiniteDifferences
using BenchmarkTools

include("../models/Smets_Wouters_2007 copy.jl")



# estimation results
## 1st order
### 1990-2024
full_sample = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]

# Summary Statistics
#   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec
#       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64

#         z_ea    0.5155    0.0332    0.0009   1393.8347   456.9051    1.0007        0.2397
#         z_eb    0.0766    0.0230    0.0015    236.4050   371.9235    0.9992        0.0407
#         z_eg    0.4293    0.0262    0.0008   1223.1447   645.6005    0.9992        0.2103
#        z_eqs    1.2212    0.0878    0.0028    990.5226   531.7283    0.9994        0.1703
#         z_em    0.7156    0.0641    0.0026    608.0501   717.2317    1.0043        0.1046
#      z_epinf    0.1307    0.0202    0.0010    376.8047   377.1497    1.0000        0.0648
#         z_ew    0.5072    0.0417    0.0014    829.3918   741.3550    1.0008        0.1426
#        crhoa    0.9772    0.0094    0.0004    552.7456   436.9902    1.0003        0.0951
#        crhob    0.9868    0.0071    0.0002    823.4337   530.2314    0.9992        0.1416
#        crhog    0.9823    0.0090    0.0003    825.1687   516.3776    1.0002        0.1419
#       crhoqs    0.0929    0.0507    0.0017    826.8943   633.0003    1.0017        0.1422
#       crhoms    0.4655    0.0644    0.0037    297.6154   525.3794    1.0036        0.0512
#     crhopinf    0.9371    0.0261    0.0012    427.8513   399.5679    1.0043        0.0736
#        crhow    0.4773    0.1519    0.0078    377.2178   340.4372    1.0004        0.0649
#         cmap    0.4466    0.1233    0.0057    471.3527   482.1029    0.9991        0.0811
#         cmaw    0.4303    0.1590    0.0074    441.1709   404.6062    1.0024        0.0759
#    csadjcost    3.6307    0.8240    0.0410    404.6383   446.9467    0.9992        0.0696
#       csigma    0.3763    0.1087    0.0064    300.7318   317.5665    0.9996        0.0517
#        chabb    0.5440    0.0636    0.0029    470.7825   449.1960    0.9998        0.0810
#       cprobw    0.7490    0.0800    0.0044    325.4168   455.5281    1.0057        0.0560
#        csigl    1.3678    0.5804    0.0275    453.4617   622.9209    1.0009        0.0780
#       cprobp    0.8055    0.0321    0.0012    778.0230   654.4965    1.0009        0.1338
#        cindw    0.4055    0.1289    0.0056    508.1408   686.9825    1.0018        0.0874
#        cindp    0.1037    0.0434    0.0017    732.3059   596.5198    1.0003        0.1259
#        czcap    0.7254    0.0973    0.0028   1198.7610   735.7620    1.0001        0.2062
#          cfc    0.9036    0.0261    0.0011    572.6420   432.9642    1.0047        0.0985
#         crpi    2.7581    0.2739    0.0118    521.1147   700.3810    1.0023        0.0896
#          crr    0.6340    0.0467    0.0017    706.3025   671.7194    1.0003        0.1215
#          cry    0.0275    0.0113    0.0005    561.2729   629.0842    1.0083        0.0965
#   constepinf    0.4373    0.0250    0.0010    608.3042   736.5238    0.9998        0.1046
#   constebeta    0.3430    0.1263    0.0050    603.7135   557.3936    0.9991        0.1038
#    constelab   -0.0582    0.3209    0.0113    805.2932   739.1549    0.9991        0.1385
#       ctrend    0.2940    0.0961    0.0033    858.1565   640.6429    1.0073        0.1476
#          cgy    0.2748    0.0231    0.0007   1110.2154   568.6518    1.0031        0.1909
#        calfa    0.3115    0.0441    0.0014    944.3262   564.1218    1.0020        0.1624
#         ctou    0.0310    0.0039    0.0002    616.4507   534.0709    0.9991        0.1060
#      clandaw    4.7229    0.7167    0.0393    333.9502   466.4962    1.0000        0.0574
#           cg    0.1909    0.0091    0.0003    784.2484   632.1253    0.9995        0.1349
#        curvp    3.7626    1.9059    0.0763    663.2870   626.0803    1.0051        0.1141
#        curvw   18.3477    6.5739    0.2143    853.4059   566.2194    1.0016        0.1468



### no pandemic
no_pandemic = [0.6078559133318278, 0.06836618238325545, 0.4203898197505046, 1.1088241818556892, 0.6541387441075293, 0.1267035923202942, 0.4528480216201151, 0.9941945010996004, 0.9900258724661307, 0.944447821651772, 0.09136974979681929, 0.5469941169752605, 0.9839879182859345, 0.8176542834158012, 0.46404242788618344, 0.7277828188461039, 6.207074468776051, 0.5342528174391462, 0.560325003881225, 0.6329231385353169, 0.8484146558715042, 0.7618268755139341, 0.7816314780804516, 0.07816721962903334, 0.817115418052766, 0.9812936465960612, 2.2188852317152006, 0.626915938550924, 0.02363305569575591, 0.4237043241955714, 0.28392007131192487, -0.7476344687461959, 0.30542058428439206, 0.5032209567712396, 0.2993769847124837, 0.034103710249185064, 4.119095036926654, 0.19636391880348672, 8.22514103090019, 14.633481496900645]

# Summary Statistics
#   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec
#       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64

#         z_ea    0.6079    0.0311    0.0009   1086.0071   376.9466    1.0086        0.0888
#         z_eb    0.0684    0.0113    0.0005    549.1434   548.1892    0.9998        0.0449
#         z_eg    0.4204    0.0210    0.0006   1338.7267   664.0818    0.9995        0.1094
#        z_eqs    1.1088    0.0714    0.0022   1097.2191   699.4664    0.9994        0.0897
#         z_em    0.6541    0.0482    0.0017    833.9144   823.0629    1.0025        0.0682
#      z_epinf    0.1267    0.0210    0.0008    664.2627   660.1217    1.0000        0.0543
#         z_ew    0.4528    0.0356    0.0017    536.2179   378.7769    1.0019        0.0438
#        crhoa    0.9942    0.0026    0.0001    594.1165   578.0779    1.0000        0.0486
#        crhob    0.9900    0.0051    0.0001   1124.8781   455.1738    1.0052        0.0920
#        crhog    0.9444    0.0280    0.0011    628.4527   693.5409    0.9997        0.0514
#       crhoqs    0.0914    0.0478    0.0015    935.3038   606.4320    0.9992        0.0765
#       crhoms    0.5470    0.0637    0.0028    512.8851   549.9773    1.0012        0.0419
#     crhopinf    0.9840    0.0075    0.0002   1108.8513   660.1217    0.9999        0.0906
#        crhow    0.8177    0.0902    0.0047    276.3426   190.9008    1.0034        0.0226
#         cmap    0.4640    0.1204    0.0044    765.6769   514.2185    1.0016        0.0626
#         cmaw    0.7278    0.1266    0.0062    352.4298   484.3414    1.0046        0.0288
#    csadjcost    6.2071    1.2190    0.0569    426.4048   492.0854    1.0003        0.0349
#       csigma    0.5343    0.1212    0.0058    431.3279   528.9952    1.0006        0.0353
#        chabb    0.5603    0.0641    0.0029    473.3219   691.3219    1.0008        0.0387
#       cprobw    0.6329    0.0759    0.0030    659.4448   481.9767    1.0036        0.0539
#        csigl    0.8484    0.4390    0.0207    429.7583   578.5761    1.0003        0.0351
#       cprobp    0.7618    0.0405    0.0022    390.8129   442.1859    1.0084        0.0319
#        cindw    0.7816    0.0914    0.0026   1292.3401   679.8451    0.9995        0.1056
#        cindp    0.0782    0.0321    0.0010   1157.4783   567.7376    1.0025        0.0946
#        czcap    0.8171    0.0760    0.0020   1373.5303   513.9606    0.9995        0.1123
#          cfc    0.9813    0.0309    0.0012    650.4144   692.7343    1.0024        0.0532
#         crpi    2.2189    0.2123    0.0084    687.8052   571.7211    1.0015        0.0562
#          crr    0.6269    0.0435    0.0015    890.1587   792.0778    0.9998        0.0728
#          cry    0.0236    0.0102    0.0004    771.5002   698.1655    1.0075        0.0631
#   constepinf    0.4237    0.0232    0.0008    864.2979   759.0312    1.0056        0.0707
#   constebeta    0.2839    0.1132    0.0032   1244.9109   812.5361    0.9999        0.1018
#    constelab   -0.7476    0.3268    0.0127    692.4406   599.3088    1.0010        0.0566
#       ctrend    0.3054    0.0941    0.0026   1327.6548   705.6728    1.0137        0.1085
#          cgy    0.5032    0.0340    0.0014    624.8698   767.7530    0.9992        0.0511
#        calfa    0.2994    0.0353    0.0010   1252.2293   608.4964    0.9995        0.1024
#         ctou    0.0341    0.0042    0.0002    654.7884   773.8952    1.0015        0.0535
#      clandaw    4.1191    0.6505    0.0294    483.0656   687.7024    1.0037        0.0395
#           cg    0.1964    0.0093    0.0003    780.9807   559.3017    0.9991        0.0638
#        curvp    8.2251    4.7403    0.1611    812.2716   716.4445    1.0033        0.0664
#        curvw   14.6335    5.8372    0.1591   1262.0101   691.6939    1.0004        0.1032


## full sample with parameter from no pandemic and reestimation of shock processes
full_sample_shock_pandemic = [0.523787265696482, 0.1350452339819597, 0.7696992475908612, 1.2151550281758268, 0.2702523358550936, 0.11515649363414246, 0.44249640226612147, 0.9694147651241034, 0.899148694268443, 0.9312052754622525, 0.16132684200160735, 0.5629282423072136, 0.9449367727446553, 0.8547459802623111, 0.23215094358997074, 0.7073492266565654,          6.207074468776051, 0.5342528174391462, 0.560325003881225, 0.6329231385353169, 0.8484146558715042, 0.7618268755139341, 0.7816314780804516, 0.07816721962903334, 0.817115418052766, 0.9812936465960612, 2.2188852317152006, 0.626915938550924, 0.02363305569575591, 0.4237043241955714, 0.28392007131192487, -0.7476344687461959, 0.30542058428439206, 0.5032209567712396, 0.2993769847124837, 0.034103710249185064, 4.119095036926654, 0.19636391880348672, 8.22514103090019, 14.633481496900645]

# Summary Statistics
#   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec 
#       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64 

#         z_ea    0.5238    0.0294    0.0009   1155.9498   715.4054    0.9993        0.6653
#         z_eb    0.1350    0.0157    0.0005    853.4967   580.5292    0.9999        0.4912
#         z_eg    0.7697    0.0483    0.0013   1334.0014   814.6575    0.9998        0.7677
#        z_eqs    1.2152    0.0988    0.0028   1229.4627   703.1632    0.9995        0.7076
#         z_em    0.2703    0.0168    0.0004   1673.5475   826.9916    1.0001        0.9632
#      z_epinf    0.1152    0.0129    0.0005    750.7592   686.3622    1.0013        0.4321
#         z_ew    0.4425    0.0310    0.0010   1014.2128   594.0936    0.9995        0.5837
#        crhoa    0.9694    0.0119    0.0003   1002.9256   405.0649    0.9994        0.5772
#        crhob    0.8991    0.0243    0.0008    883.6921   568.5537    0.9992        0.5086
#        crhog    0.9312    0.0218    0.0006   1448.0706   694.5731    1.0028        0.8334
#       crhoqs    0.1613    0.0772    0.0025    881.4438   639.2438    1.0018        0.5073
#       crhoms    0.5629    0.0451    0.0016    836.9367   635.4538    1.0023        0.4817
#     crhopinf    0.9449    0.0165    0.0006    627.6275   671.7117    1.0031        0.3612
#        crhow    0.8547    0.0237    0.0010    542.5226   508.8381    1.0002        0.3122
#         cmap    0.2322    0.0938    0.0035    697.8830   576.4969    1.0097        0.4016
#         cmaw    0.7073    0.0505    0.0021    601.4647   550.5438    0.9995        0.3462



# FULL SAMPLE with reestimation of shock sizes
# Summary Statistics
#   parameters      mean       std      mcse    ess_bulk   ess_tail      rhat   ess_per_sec 
#       Symbol   Float64   Float64   Float64     Float64    Float64   Float64       Float64 

#         z_ea    0.5259    0.0312    0.0010    998.9972   716.1674    1.0007        1.1543
#         z_eb    0.1302    0.0082    0.0002   1119.4634   741.5955    1.0015        1.2935
#         z_eg    0.7616    0.0490    0.0014   1222.5520   785.5574    0.9995        1.4126
#        z_eqs    1.3036    0.0812    0.0026    969.7095   560.9309    1.0006        1.1204
#         z_em    0.2675    0.0163    0.0005   1009.6180   812.2920    0.9993        1.1666
#      z_epinf    0.1521    0.0093    0.0003   1042.4295   675.1217    1.0003        1.2045
#         z_ew    0.4955    0.0289    0.0009    962.8892   606.0280    0.9998        1.1126

full_sample_shock_std_pandemic = [0.5259, 0.1302, 0.7616, 1.3036, 0.2675, 0.1521, 0.4955,       0.9941945010996004, 0.9900258724661307, 0.944447821651772, 0.09136974979681929, 0.5469941169752605, 0.9839879182859345, 0.8176542834158012, 0.46404242788618344, 0.7277828188461039, 6.207074468776051, 0.5342528174391462, 0.560325003881225, 0.6329231385353169, 0.8484146558715042, 0.7618268755139341, 0.7816314780804516, 0.07816721962903334, 0.817115418052766, 0.9812936465960612, 2.2188852317152006, 0.626915938550924, 0.02363305569575591, 0.4237043241955714, 0.28392007131192487, -0.7476344687461959, 0.30542058428439206, 0.5032209567712396, 0.2993769847124837, 0.034103710249185064, 4.119095036926654, 0.19636391880348672, 8.22514103090019, 14.633481496900645]


pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

std_full_sample = get_std(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

std_no_pandemic = get_std(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)

std_full_sample_shock_pandemic = get_std(Smets_Wouters_2007, parameters = pars .=> full_sample_shock_pandemic, derivatives = false)

std_full_sample_shock_std_pandemic = get_std(Smets_Wouters_2007, parameters = pars .=> full_sample_shock_std_pandemic, derivatives = false)

std_full_sample([:a,:b,:gy,:qs,:ms,:spinf,:sw])
# (:a)      0.02426349697813647
# (:b)      0.005998118330574482
# (:gy)     0.0241028522046634
# (:qs)     0.08951131382441423
# (:ms)     0.008085447272957411
# (:spinf)  0.0335251067024813
# (:sw)     8.324446250434056

std_no_pandemic([:a,:b,:gy,:qs,:ms,:spinf,:sw])
# (:a)      0.05649332222233191
# (:b)      0.009154771190994018
# (:gy)     0.015818645868635427
# (:qs)     0.13897662836032265
# (:ms)     0.007814003846949065
# (:spinf)  0.047651064538222695
# (:sw)     2.0020346229399255

std_full_sample_shock_pandemic([:a,:b,:gy,:qs,:ms,:spinf,:sw])
# (:a)     ↓ 0.02134176978740371
# (:b)     ↓ 0.0058214610923949675
# (:gy)    ↑ 0.022320722694152133
# (:qs)    ↑ 0.15367977595326165
# (:ms)    ↓ 0.0032698178820298263
# (:spinf) ↓ 0.03365828622497589
# (:sw)      2.009285473806119

std_full_sample_shock_std_pandemic([:a,:b,:gy,:qs,:ms,:spinf,:sw])
# (:a)     ↓ 0.048876448357432746
# (:b)     ↑ 0.017434807203149848
# (:gy)    ↑ 0.024531928343334983
# (:qs)    ↑ 0.1633892331129693
# (:ms)    ↓ 0.0031954169476854496
# (:spinf) ↑ 0.05720222121202543
# (:sw)    ↑ 2.1905984089710966


# Interpretation:
# estimating the model on the pre-pandemic sample, then fixing the model parameters and reestimating on the full sample only the AR1 shock process related parameters the impact of the pandemic and inflaiton surge period of the variance of the shock processes is higher variance on the investment and government spending shock, unchanged variance on the wage markup shock and all other variances decrease




using StatsPlots
include("../test/download_EA_data.jl")

plot_shock_decomposition(Smets_Wouters_2007, 
                        data[[1,2,3,4,6,7,8],78:end],
                        # parameters = pars .=> full_sample,
                        # parameters = pars .=> no_pandemic,
                        parameters = pars .=> full_sample_shock_pandemic,
                        variables = [:ygap,:y,:pinfobs,:robs],
                        plots_per_page = 4,
                        filter = :kalman,
                        smooth = true)

shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, 
                                    data[[1,2,3,4,6,7,8],78:end],
                                    parameters = pars .=> full_sample,
                                    # parameters = pars .=> no_pandemic,
                                    # parameters = pars .=> full_sample_shock_pandemic,
);#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)


shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, 
                                    data[[1,2,3,4,6,7,8],78:end],
                                    # parameters = pars .=> full_sample,
                                    parameters = pars .=> no_pandemic,
                                    # parameters = pars .=> full_sample_shock_pandemic,
);#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)


shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, 
                                    data[[1,2,3,4,6,7,8],78:end],
                                    # parameters = pars .=> full_sample,
                                    # parameters = pars .=> no_pandemic,
                                    parameters = pars .=> full_sample_shock_pandemic,
);#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)




SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

SS(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)



# US SW07 sample estims
# estimated_par_vals = [0.4818650901000989, 0.24054470291311028, 0.5186956692202958, 0.4662413867655003, 0.23136135922950385, 0.13132950287219664, 0.2506090809487915, 0.9776707755474057, 0.2595790622654468, 0.9727418060187103, 0.687330720531337, 0.1643636762401503, 0.9593771388356938, 0.9717966717403557, 0.8082505346152592, 0.8950643861525535, 5.869499350284732, 1.4625899840952736, 0.724649200081708, 0.7508616008157103, 2.06747381157293, 0.647865359908012, 0.585642549132298, 0.22857733002230182, 0.4476375712834215, 1.6446238878581076, 2.0421854715489007, 0.8196744223749656, 0.10480818163546246, 0.20376610336806866, 0.7312462829038883, 0.14032972276989308, 1.1915345520903131, 0.47172181998770146, 0.5676468533218533, 0.2071701728019517]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# EA long sample
# estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

# SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# EA tighter priors (no crdy)
# estimated_par_vals = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

# SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

# SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)

# find optimal loss coefficients
# Problem definition, find the loss coefficients such that the derivatives of the Taylor rule coefficients wrt the loss are 0

optimal_taylor_coefficients = [Dict(get_parameters(Smets_Wouters_2007, values = true))[i] for i in ["crpi", "cry", "crr"]]

# taylor_coef_stds = [0.2739, 0.0113, 0.0467]
taylor_coef_stds = [0.2123, 0.0102, 0.0435]

function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation = p

    out = get_statistics(Smets_Wouters_2007,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
end

function find_weights(loss_function_weights_regularisation, optimal_taylor_coefficients)
    loss_function_weights = loss_function_weights_regularisation[1:2]
    regularisation = 1 / loss_function_weights_regularisation[3]
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,loss_function_weights), regularisation)), optimal_taylor_coefficients))
end

# find_weights(vcat(loss_function_wts, 1 / regularisation[1]), optimal_taylor_coefficients)

# get_parameters(Smets_Wouters_2007, values = true)
lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())

prob = OptimizationProblem(f, fill(0.5, 3), optimal_taylor_coefficients, ub = ubs, lb = lbs)

# sol = solve(prob, NLopt.LD_TNEWTON(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

find_weights(sol.u, optimal_taylor_coefficients)

ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,sol.u[1:2]), 1 / sol.u[3])), optimal_taylor_coefficients)


derivs = FiniteDifferences.jacobian(FiniteDifferences.central_fdm(3,1), 
                                    x -> begin
                                        prob = OptimizationProblem(f, fill(0.5, 3), x, ub = ubs, lb = lbs)
                                        sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)
                                        return sol.u
                                    end, optimal_taylor_coefficients)

loss_function_weights_lower = copy(sol.u) - derivs[1] * [0.2123, 0.0102, -0.0435] # taylor_coef_stds
loss_function_weights_upper = copy(sol.u) + derivs[1] * [0.2123, 0.0102, -0.0435] # taylor_coef_stds


loss_function_weights = vcat(1, copy(sol.u[1:2]))

lbs = [eps(),eps(),eps()] #,eps()]
ubs = [1e6, 1e6, 1] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
regularisation = 1 / copy(sol.u[3])  #,1e-5]


get_statistics(Smets_Wouters_2007,   
                    optimal_taylor_coefficients,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = :first_order,
                    verbose = true)


calculate_cb_loss(optimal_taylor_coefficients, (loss_function_weights, regularisation), verbose = true)

# SS(Smets_Wouters_2007, parameters = :crdy => 0, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
# sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000)

# sol = solve(prob, Optimization.LBFGS(), maxiters = 10000)

# sol = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

# sol = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 1000)

calculate_cb_loss(sol.u, (loss_function_weights, regularisation))

# abs2.(sol.u)' * regularisation

# sol.objective
# loop across different levels of std

get_parameters(Smets_Wouters_2007, values = true)

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007)#, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0))
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw],[:crr, :crpi, :cry]))'
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Standard_deviation_and_∂standard_deviation∂parameter ∈ 19-element view(::Vector{Symbol},...)
# →   Variables ∈ 4-element view(::Vector{Symbol},...)
# And data, 19×4 adjoint(view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 23, 24, 25, 26, 27, 28, 29, 30, 31, 20, 19, 21])) with eltype Float64:
#                 (:ygap)      (:pinfobs)   (:drobs)      (:robs)
#   (:z_ea)        0.431892     0.0337031    0.0209319     0.0399137
#   (:z_eb)        0.498688     7.68657      0.2897       18.1822
#   (:z_eg)        0.00730732   0.00281355   0.00143873    0.00418418
#   (:z_em)        1.23788      0.168688     0.73829       0.561376
#   (:z_ew)       15.9711       0.535026     0.104924      0.710824
#   (:z_eqs)       0.067932     0.00306293   0.000692842   0.00469514
#   (:z_epinf)    24.2758       0.535294     0.297274      0.504287
#   (:crhoa)       0.0784198   -0.156896    -0.0374588    -0.270447
#   (:crhob)      -0.537715    29.0378       0.0992404    69.119
#   (:crhog)      -0.201827    -0.00292154   0.00139387   -0.00340863
#   (:crhoqs)      0.113207     0.00511452   0.0019416     0.00756271
#   (:crhoms)      2.09109      0.369009    -0.45944      -0.0168982
#   (:crhopinf)   73.8881       0.226225    -0.0511477    -0.0245807
#   (:crhow)      94.8585       2.11075      0.078647      2.75274
#   (:cmap)       -5.89078     -0.0945698   -0.0291701    -0.0982907
#   (:cmaw)      -57.2493      -1.54286     -0.146276     -2.06334
#   (:crr)         2.58469      0.271689    -0.330182     -0.128754
#   (:crpi)        0.607836    -0.624988    -0.0260916    -0.539138
#   (:cry)       -20.0359       3.69087      0.464149      2.69922

# Zygote.gradient(x->calculate_cb_loss(x,regularisation), sol.u)[1]


# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)

# Zygote.gradient(x->calculate_cb_loss(x,regularisation * 1),sol.u)[1]

# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1] * 1.05, derivatives = false)

# using FiniteDifferences

# FiniteDifferences.hessian(x->calculate_cb_loss(x,regularisation * 0),sol.u)[1]


# SS(Smets_Wouters_2007, parameters = stds[1] => std_vals[1], derivatives = false)



# SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

# nms = []

k_range = .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.1 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "crr", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", zlabel = "(1 - crr) * cry", colorbar=false))

    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm)_surface.png")

    ii += 1
end



ii = 1
for (nm,vl) in zip(stds,std_vals)
    plots = []
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]), label = "", ylabel = "Loss weight: Δr", xlabel = "Std($nm)", title = "crr", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,5]), label = "", ylabel = "Loss weight: Δr", xlabel = "Std($nm)", title = "(1 - crr) * crpi", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,6]), label = "", ylabel = "Loss weight: Δr", xlabel = "Std($nm)", title = "(1 - crr) * cry", colorbar=true))

    p = plot(plots...) # , plot_title = string(nm))

    savefig(p,"OSR_$(nm)_contour.png")
    ii += 1
end


k_range = [.5] # .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.1 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Std($nm)", ylabel = "crr", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Std($nm)", ylabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Std($nm)", ylabel = "(1 - crr) * cry", colorbar=false))
    
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm).png")

    ii += 1
end


# ↑ z_ea    ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.6078559133318278
# ↑ z_eb    ⟹   ↑ inflation, ↑ output, ↑ persistence   -    0.06836618238325545
# ↑ z_eg    ⟹   → inflation, → output, → persistence   -    0.4203898197505046
# ↑ z_em    ⟹   ↑ inflation, ↑ output, ↓ persistence   -    0.6541387441075293
# ↑ z_ew    ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.4528480216201151
# ↑ z_eqs   ⟹   → inflation, ↑ output, → persistence   -    1.1088241818556892
# ↑ z_epinf ⟹   ↓ inflation, ↓ output, ↑ persistence   -    0.1267035923202942

# do the same for the shock autocorrelation

rhos = Smets_Wouters_2007.parameters[end-19:end-11]
rho_vals = copy(Smets_Wouters_2007.parameter_values[end-19:end-11])

k_range = .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(rhos), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(rhos,rho_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.0 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "crr", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * cry", colorbar=false))

    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm)_surface.png")

    ii += 1
end


ii = 1
for (nm,vl) in zip(rhos,rho_vals)
    plots = []
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]), label = "", ylabel = "Loss weight: Δr", xlabel = "$nm", title = "crr", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,5]), label = "", ylabel = "Loss weight: Δr", xlabel = "$nm", title = "(1 - crr) * crpi", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,6]), label = "", ylabel = "Loss weight: Δr", xlabel = "$nm", title = "(1 - crr) * cry", colorbar=true))

    p = plot(plots...) # , plot_title = string(nm))

    savefig(p,"OSR_$(nm)_contour.png")
    ii += 1
end


k_range = [.5] # .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(rhos), n_σ_range, 7);


ii = 1
for (nm,vl) in zip(rhos,rho_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(.9 * vl, 1.0 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
            # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective)")
        end

        
        SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    
    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm).png")

    ii += 1
end


# Interpretation:
# Shock persistence:
# ↑ crhob     ⟹ ↑ inflation, ↑ output, ↑ persistence    -   0.9941945010996004
# ↑ crhoa     ⟹ ↑ inflation, ↑ output, ↓ persistence   -   0.9900258724661307
# ↑ crhog     ⟹ → inflation, → output, → persistence    -   0.944447821651772
# ↑ crhoqs    ⟹ → inflation, → output, → persistence    -   0.09136974979681929
# ↑ crhoms    ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.5469941169752605
# ↑ crhopinf  ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.9839879182859345
# ↑ crhow     ⟹ ↓ inflation, ↓ output, ↑ persistence    -   0.8176542834158012
# ↑ cmap      ⟹ ↑ inflation, ↑ output, ↓ persistence    -   0.46404242788618344
# ↑ cmaw      ⟹ ↑ inflation, ↓ output, ↓ persistence    -   0.7277828188461039

# Interpretation:
# starting from the implied optimal weights by the taylor rule coefficients resulting from the estimation to be optimal on the pre-pandemic period and setting the crdy coefficient to 0 we change the shock standard deviations and recalculate the optimal Taylor rule coefficients given the implied optimal loss weights
# (:a)     ↓ stronger reaction to inflation and output
# (:b)     ↓ weaker reaction to inflation and output
# (:gy)    ↑ no impact on optimal reaction function 
# (:qs)    ↑ slightly weaker response on inflaiton, stronger response to output
# (:ms)    ↓ weaker reaction to inflation and output
# (:spinf) ↓ stronger reaction to inflation and output
# (:sw)    →

# doing the same but letting only the standard deviations vary the following changes apply:
# (:a)     ↓ 0.048876448357432746   (0.05649332222233191)   stronger reaction to inflation and output, less persistence
# (:b)     ↑ 0.017434807203149848   (0.009154771190994018)  stronger reaction to inflation and output, more persistence - dominates
# (:gy)    ↑ 0.024531928343334983   (0.015818645868635427)  no impact on optimal reaction function 
# (:qs)    ↑ 0.1633892331129693     (0.13897662836032265)   slightly weaker response on inflaiton, slightly less persistence, stronger response to output
# (:ms)    ↓ 0.0031954169476854496  (0.007814003846949065)  weaker reaction to inflation and output, more persistence - compensates
# (:spinf) ↑ 0.05720222121202543    (0.047651064538222695)  weaker reaction to inflation and output, more persistence
# (:sw)    ↑ 2.1905984089710966     (2.0020346229399255)    weaker reaction to inflation and output, more persistence



# Demand shocks (Y↑ - pi↑ - R↑)
# z_eb	# risk-premium shock
# z_eg	# government shock
# z_eqs	# investment-specific shock


# Monetary policy (Y↓ - pi↓ - R↑)
# z_em	# interest rate shock

# Supply shock (Y↓ - pi↑ - R↑)
# z_ea	# technology shock

## Mark-up/cost-push shocks (Y↓ - pi↑ - R↑)
# z_ew	# wage mark-up shock
# z_epinf	# price mark-up shock

full_sample_shock_std_pandemic[1:7] - no_pandemic[1:7]

(stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds)) * (full_sample_shock_std_pandemic[1:7] - no_pandemic[1:7]))[1:3] #' * derivs[1]'
# these changes imply lower output gap standard deviation and higher std of inflation and changes in rates


SS(Smets_Wouters_2007, parameters = pars .=> no_pandemic, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
# 2.2188852312707352
# 0.0236330702790334
# 0.626915938497439
sol.objective


SS(Smets_Wouters_2007, parameters = pars .=> full_sample_shock_pandemic, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results
sol = solve(prob, NLopt.LN_PRAXIS(), maxiters = 10000) # this seems to achieve best results
# 1.6438787396024561
# 0.03092869784067731
# 0.849531449088666
sol.objective


SS(Smets_Wouters_2007, parameters = pars .=> full_sample_shock_std_pandemic, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results
sol = solve(prob, NLopt.LN_PRAXIS(), maxiters = 10000) # this seems to achieve best results
# 2.628935541874458
# 0.06795643427038466
# 0.8303750184985217
sol.objective


SS(Smets_Wouters_2007, parameters = pars .=> full_sample, derivatives = false)

f = OptimizationFunction(calculate_cb_loss, AutoZygote())
# f = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation), ub = ubs, lb = lbs)

# Import a solver package and solve the optimization problem

# sol = solve(prob, NLopt.LN_PRAXIS(), maxiters = 10000) # this seems to achieve best results

sol = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results
# 0.6344347396867679
# 1.2479460911230826
# 0.5721511659886132
# sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

sol.objective

# Interpretation:
# taken together the changes in the variance of the shock processes imply a weaker response to inflation, a stronger response to output and overall more persistence
# if you vary only the shock standard deviations and not the persistence the effect is flipped and the optimal response is stronger wrt output, inflation and also more presistent

## Analysis of other sources of uncertainty
include("../models/Smets_Wouters_2007_ext.jl")

model = Smets_Wouters_2007_ext


# EA tighter priors (no crdy)
# estimated_par_vals = [0.5155251475194788, 0.07660166839374086, 0.42934249231657745, 1.221167691146145, 0.7156091225215181, 0.13071182824630584, 0.5072333270577154, 0.9771677130980795, 0.986794686927924, 0.9822502018161883, 0.09286109236460689, 0.4654804216926021, 0.9370552043932711, 0.47725222696887853, 0.44661470121418184, 0.4303294544434745, 3.6306838940222996, 0.3762913949270054, 0.5439881753546603, 0.7489991629811795, 1.367786474803364, 0.8055157457796492, 0.40545058009366347, 0.10369929978953055, 0.7253632750136628, 0.9035647768098533, 2.7581458138927886, 0.6340306336303874, 0.0275348491078362, 0.43733563413301674, 0.34302913866206625, -0.05823832790219527, 0.29395331895770577, 0.2747958016561462, 0.3114891537064354, 0.030983938890070825, 4.7228912586862375, 0.1908504262397911, 3.7626464596678604, 18.34766525498524]

# estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa, :ctou, :clandaw, :cg, :curvp, :curvw]

SS(model, parameters = :crdy => 0, derivatives = false)

SS(model, parameters = pars .=> no_pandemic, derivatives = false)

# SS(model, parameters = :crdy => 0, derivatives = false)

# SS(model, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)


optimal_taylor_coefficients = [Dict(get_parameters(model, values = true))[i] for i in ["crpi", "cry", "crr"]]

# taylor_coef_stds = [0.2739, 0.0113, 0.0467]
taylor_coef_stds = [0.2123, 0.0102, 0.0435]

function calculate_cb_loss(taylor_parameter_inputs,p; verbose = false)
    loss_function_weights, regularisation, model = p

    out = get_statistics(model,   
                    taylor_parameter_inputs,
                    parameters = [:crpi, :cry, :crr],#, :crdy],
                    variance = [:pinfobs, :ygap, :drobs],
                    algorithm = :first_order,
                    verbose = verbose)

    return out[:variance]' * loss_function_weights + sum(abs2, taylor_parameter_inputs) * regularisation
end

function find_weights(loss_function_weights_regularisation, p)
    optimal_taylor_coefficients, model = p
    loss_function_weights = loss_function_weights_regularisation[1:2]
    regularisation = 1 / loss_function_weights_regularisation[3]
    sum(abs2, ForwardDiff.gradient(x->calculate_cb_loss(x, (vcat(1,loss_function_weights), regularisation, model)), optimal_taylor_coefficients))
end

lbs = fill(0.0, 3)
ubs = fill(1e6, 3)

f = OptimizationFunction((x,p)-> find_weights(x,p), AutoForwardDiff())

prob = OptimizationProblem(f, fill(0.5, 3), (optimal_taylor_coefficients, model), ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000)




loss_function_weights = vcat(1, copy(sol.u[1:2]))

lbs = [eps(),eps(),eps()] #,eps()]
ubs = [1e6, 1e6, 1] #, 1e6]
# initial_values = [0.8762 ,1.488 ,0.0593] # ,0.2347]
regularisation = 1 / copy(sol.u[3])  #,1e-5]

f = OptimizationFunction(calculate_cb_loss, AutoZygote())

prob = OptimizationProblem(f, optimal_taylor_coefficients, (loss_function_weights, regularisation, model), ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results



# std cprobw: 0.08
# std cprobp: 0.0321

stds = [:σ_pinf, :σ_cprobp, :σ_cprobw]
std_vals = [0.015, 0.0321, 0.08]

stdderivs = get_std(model)#, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0))
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))



k_range = .45:.025:.55 # [.5] # 
n_σ_range = 10
coeff = zeros(length(k_range), length(stds), n_σ_range, 7);

ii = 1
for (nm,vl) in zip(stds,std_vals)
    for (l,k) in enumerate(k_range)
        σ_range = range(0 * vl, .1 * vl, length = n_σ_range)

        combined_loss_function_weights = vcat(1,k * loss_function_weights_upper[1:2] + (1-k) * loss_function_weights_lower[1:2])

        prob = OptimizationProblem(f, optimal_taylor_coefficients, (combined_loss_function_weights, regularisation, model), ub = ubs, lb = lbs)

        for (ll,σ) in enumerate(σ_range)
            SS(model, parameters = nm => σ, derivatives = false)
            
            soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
            
            # soll = solve(prob, NLopt.LN_NELDERMEAD(), maxiters = 10000) # this seems to achieve best results
            
            coeff[l,ii,ll,:] = vcat(combined_loss_function_weights,σ,soll.u)

            println("$nm $σ $(soll.objective) $(soll.u[1:2])")
        end

        
        SS(model, parameters = nm => 0, derivatives = false)
        
        # display(p)
    end
    
    plots = []
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "crr", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * crpi", colorbar=false))
    push!(plots, surface(vec(coeff[:,ii,:,3]), vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "$nm", zlabel = "(1 - crr) * cry", colorbar=false))

    p = plot(plots...) # , plot_title = string(nm))
    savefig(p,"OSR_$(nm)_surface_ext.png")

    # plots = []
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec(coeff[:,ii,:,7]), label = "", xlabel = "$nm", ylabel = "crr", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,5]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * crpi", colorbar=false))
    # push!(plots, plot(vec(coeff[:,ii,:,4]), vec((1 .- coeff[:,ii,:,7]) .* coeff[:,ii,:,6]), label = "", xlabel = "$nm", ylabel = "(1 - crr) * cry", colorbar=false))
    
    # p = plot(plots...) # , plot_title = string(nm))
    # savefig(p,"OSR_$(nm)_ext.png")

    ii += 1
end



ii = 1
for (nm,vl) in zip(stds,std_vals)
    plots = []
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "crr", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,5]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "(1 - crr) * crpi", colorbar=false))
    push!(plots, contour(vec(coeff[1,ii,:,4]), vec(coeff[end:-1:1,ii,1,3]), ((1 .- coeff[end:-1:1,ii,:,7]) .* coeff[end:-1:1,ii,:,6]), label = "", xlabel = "Loss weight: Δr", ylabel = "Std($nm)", title = "(1 - crr) * cry", colorbar=false))

    p = plot(plots...) # , plot_title = string(nm))

    savefig(p,"OSR_$(nm)_contour_ext.png")
    ii += 1
end


# Interpretation:
# increase in uncertainty regarding the calvo probabilities, or a measurement error of inflation in the Taylor rule all trigger a weaker inflation response and a stronger output response



# look at shock decomposition for pandemic period
using StatsPlots
include("../test/download_EA_data.jl")

plot_shock_decomposition(Smets_Wouters_2007, data[[1,2,3,4,6,7,8],78:end], variables = [:ygap,:y,:pinfobs,:robs], plots_per_page = 1, filter = :kalman, smooth = true)

shck_dcmp = get_shock_decomposition(Smets_Wouters_2007, data[[1,2,3,4,6,7,8],78:end], initial_values = state20q1)#, 
# filter = :kalman, smooth = false)


shck_dcmp([:ygap,:pinfobs,:robs,:y],:,118:125)

# pandemic shock interepreted as mix of productivity and cost push shock
# inflation surge driven by cost push and mon. pol. shocks

# estimate up to start of pandemic and compare stds to gauge difference


xs = [string("x", i) for i = 1:10]
ys = [string("y", i) for i = 1:4]
z = float((1:4) * reshape(1:10, 1, :))
heatmap(xs, ys, z, aspect_ratio = 1)


contour(x,y,Z)

ii = 2

heatmap((coeff[1,ii,:,4]), (coeff[end:-1:1,ii,1,3]), (coeff[end:-1:1,ii,:,7]))#, label = "", xlabel = "Loss weight: Δr", ylabel = "Std()", zlabel = "crr", colorbar=false)
heatmap(vec(coeff[:,1,:,4]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "crpi")
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec(coeff[:,1,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "cry")

shck = 7
surface(vec(coeff[:,1,:,1]), vec(coeff[:,1,:,2]), vec((1 .- coeff[:,1,:,3]) .* coeff[:,1,:,4]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * crpi")
surface(vec(coeff[:,shck,:,1]), vec(coeff[:,shck,:,2]), vec((1 .- coeff[:,shck,:,3]) .* coeff[:,shck,:,5]), label = "", xlabel = "r weight", ylabel = "Std", zlabel = "(1 - crr) * cry")


surface(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr")

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [i[2] for i in coeffs], label = "", ylabel = "crpi"))
    push!(plots, plot(σ_range, [i[3] for i in coeffs], label = "", ylabel = "cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_direct_$nm.png")
    # display(p)
end

# Demand shocks (Y↑ - pi↑ - R↑)
# z_eb	# risk-premium shock
# z_eg	# government shock
# z_eqs	# investment-specific shock


# Monetary policy (Y↓ - pi↓ - R↑)
# z_em	# interest rate shock

# Supply shock (Y↓ - pi↑ - R↑)
# z_ea	# technology shock

## Mark-up/cost-push shocks (Y↓ - pi↑ - R↑)
# z_ew	# wage mark-up shock
# z_epinf	# price mark-up shock



# demand shock (Y↑ - pi↑ - R↑): more aggressive on all three measures
# irf: GDP and inflation in same direction so you can neutralise this shocks at the cost of higher rate volatility

# supply shocks (Y↓ - pi↑ - R↑): more aggressive on inflation and GDP and less so on inflation
# trade off betwen GDP and inflation will probably dampen interest rate voltility so you can allow yourself to smooth less

# mark-up shocks (Y↓ - pi↑ - R↑): less aggressive on inflation and GDP but more smoothing
# low effectiveness wrt inflation, high costs, inflation less sticky



# try with EA parameters from estimation
estimated_par_vals = [0.5508386670366793, 0.1121915320498811, 0.4243377356726877, 1.1480212757573225, 0.15646733079230218, 0.296296659613257, 0.5432042443198039, 0.9902290087557833, 0.9259443641489151, 0.9951289612362465, 0.10142231358290743, 0.39362463001158415, 0.1289134188454152, 0.9186217201941123, 0.335751074044953, 0.9679659067034428, 7.200553443953002, 1.6027080351282608, 0.2951432248740656, 0.9228560491337098, 1.4634253784176727, 0.9395327544812212, 0.1686071783737509, 0.6899027652288519, 0.8752458891177585, 1.0875693299513425, 1.0500350793944067, 0.935445005053725, 0.14728806935911198, 0.05076653598648485, 0.6415024921505285, 0.2033331251651342, 1.3564948300498199, 0.37489234540710886, 0.31427612698706603, 0.12891275085926296]

estimated_pars = [:z_ea, :z_eb, :z_eg, :z_eqs, :z_em, :z_epinf, :z_ew, :crhoa, :crhob, :crhog, :crhoqs, :crhoms, :crhopinf, :crhow, :cmap, :cmaw, :csadjcost, :csigma, :chabb, :cprobw, :csigl, :cprobp, :cindw, :cindp, :czcap, :cfc, :crpi, :crr, :cry, :crdy, :constepinf, :constebeta, :constelab, :ctrend, :cgy, :calfa]

SS(Smets_Wouters_2007, parameters = estimated_pars .=> estimated_par_vals, derivatives = false)


prob = OptimizationProblem(f, initial_values, regularisation / 100, ub = ubs, lb = lbs)

sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results

stds = Smets_Wouters_2007.parameters[end-6:end]
std_vals = copy(Smets_Wouters_2007.parameter_values[end-6:end])

stdderivs = get_std(Smets_Wouters_2007, parameters = [:crr, :crpi, :cry, :crdy] .=> vcat(sol.u,0));
stdderivs([:ygap, :pinfobs, :drobs, :robs],vcat(stds,[:crr, :crpi, :cry]))
# 2-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 4-element view(::Vector{Symbol},...)
# →   Standard_deviation_and_∂standard_deviation∂parameter ∈ 11-element view(::Vector{Symbol},...)
# And data, 4×11 view(::Matrix{Float64}, [65, 44, 17, 52], [36, 37, 38, 39, 40, 41, 42, 20, 19, 21, 22]) with eltype Float64:
#               (:z_ea)     (:z_eb)      (:z_eg)      (:z_em)     (:z_ew)    (:z_eqs)      (:z_epinf)   (:crr)    (:crpi)     (:cry)       (:crdy)
#   (:ygap)      0.0173547   0.151379     0.00335788   0.303146    4.07402    0.0062702     1.15872    -60.0597    0.0553865  -0.0208848   -0.116337
#   (:pinfobs)   0.0112278   2.48401e-5   9.97486e-5   0.134175    8.97266    0.000271719   3.75081     90.5474   -0.0832394   0.0312395    0.105122
#   (:drobs)     0.289815    3.7398       0.0452899    0.0150356   0.731132   0.0148536     0.297607     4.20058  -0.0045197   0.00175464   0.121104
#   (:robs)      0.216192    1.82174      0.0424333    0.115266    7.89551    0.0742737     2.57712     80.8386   -0.0743082   0.0273497    0.14874

Smets_Wouters_2007.parameter_values[indexin([:crr, :crpi, :cry, :crdy],Smets_Wouters_2007.parameters)]

for (nm,vl) in zip(stds,std_vals)
    σ_range = range(vl, 1.5 * vl,length = 10)

    coeffs = []
    
    for σ in σ_range
        SS(Smets_Wouters_2007, parameters = nm => σ, derivatives = false)
        # prob = OptimizationProblem(f, sol.u, regularisation * 100, ub = ubs, lb = lbs)
        soll = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000) # this seems to achieve best results
        push!(coeffs,soll.u)
        println("$nm $σ $(soll.objective)")
    end

    SS(Smets_Wouters_2007, parameters = nm => vl, derivatives = false)

    plots = []
    push!(plots, plot(σ_range, [i[1] for i in coeffs], label = "", ylabel = "crr"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[2] for i in coeffs], label = "", ylabel = "(1 - crr) * crpi"))
    push!(plots, plot(σ_range, [(1 - i[1]) * i[3] for i in coeffs], label = "", ylabel = "(1 - crr) * cry"))
    # push!(plots, plot(σ_range, [i[4] for i in coeffs], label = "", ylabel = "crdy"))

    p = plot(plots..., plot_title = string(nm))
    savefig(p,"OSR_EA_$nm.png")
    # display(p)
end





solopt = solve(prob, Optimization.LBFGS(), maxiters = 10000)
soloptjl = solve(prob, Optim.LBFGS(linesearch = LineSearches.BackTracking(order = 3)), maxiters = 10000)

sol_mlsl = solve(prob,  NLopt.G_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)


f_zyg = OptimizationFunction(calculate_cb_loss, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# 32.749
@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

sol_zyg = solve(prob_zyg, NLopt.LN_PRAXIS(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_SBPLX(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LN_NELDERMEAD(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_SLSQP(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_MMA(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_CCSAQ(), maxiters = 10000)
sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)
# sol_zyg = solve(prob_for, NLopt.LD_TNEWTON(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GD_MLSL_LDS(), local_method = NLopt.LD_SLSQP(), maxiters = 10000)
sol_zyg = solve(prob_zyg,  NLopt.GN_MLSL_LDS(), local_method = NLopt.LN_NELDERMEAD(), maxiters = 1000)
sol_zyg.u

f_for = OptimizationFunction(calculate_cb_loss, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)


ForwardDiff.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

Zygote.gradient(x->calculate_cb_loss(x,[]), [0.5753000884102637, 2.220446049250313e-16, 2.1312643700117118, 2.220446049250313e-16])

sol = Optim.optimize(x->calculate_cb_loss(x,[]), 
                    lbs, 
                    ubs, 
                    initial_values, 
                    # LBFGS(),
                    # NelderMead(),
                    # Optim.Fminbox(NelderMead()), 
                    # Optim.Fminbox(LBFGS()), 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                           show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ))#,ad = AutoZgote())

pars = Optim.minimizer(sol)


get_statistics(Smets_Wouters_2007,   
                    sol.u,
                    parameters = [:crr, :crpi, :cry, :crdy],
                    variance = [:ygap, :pinfobs, :drobs],
                    algorithm = :first_order,
                    verbose = true)

## Central bank loss function: Loss = θʸ * var(Ŷ) + θᵖⁱ * var(π̂) + θᴿ * var(ΔR)
loss_function_weights = [1, .3, .4]

lbs = [eps(),eps(),eps()]
ubs = [1e2,1e2,1-eps()]
initial_values = [1.5, 0.125, 0.75]

function calculate_cb_loss(parameter_inputs; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

calculate_cb_loss(initial_values, verbose = true)


@time sol = Optim.optimize(calculate_cb_loss, 
                    lbs, 
                    ubs, 
                    initial_values, 
                    Optim.Fminbox(LBFGS(linesearch = LineSearches.BackTracking(order = 3))), 
                    # LBFGS(linesearch = LineSearches.BackTracking(order = 3)), 
                    Optim.Options(
                    # time_limit = max_time, 
                                                        #    show_trace = true, 
                                    # iterations = 1000,
                    #                                        extended_trace = true, 
                    #                                        show_every = 10000
                    ));

pars = Optim.minimizer(sol)


out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                initial_values,
                parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                variance = [:Ŷ,:π̂,:ΔR],
                algorithm = :first_order,
                verbose = true)
out[:variance]
dd = Dict{Symbol,Array{<:Real}}()
dd[:variance] = out[1]

init_params = copy(Gali_2015_chapter_3_nonlinear.parameter_values)

function calculate_cb_loss_Opt(parameter_inputs,p; verbose = false)
    out = get_statistics(Gali_2015_chapter_3_nonlinear,   
                    parameter_inputs,
                    parameters = [:ϕᵖⁱ, :ϕʸ, :ϕᴿ],
                    variance = [:Ŷ,:π̂,:ΔR],
                    algorithm = :first_order,
                    verbose = verbose)

    cb_loss = out[:variance]' * loss_function_weights
end

f_zyg = OptimizationFunction(calculate_cb_loss_Opt, AutoZygote())
prob_zyg = OptimizationProblem(f_zyg, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_zyg = solve(prob_zyg, NLopt.LD_LBFGS(), maxiters = 10000)

f_for = OptimizationFunction(calculate_cb_loss_Opt, AutoForwardDiff())
prob_for = OptimizationProblem(f_for, initial_values, [], ub = ubs, lb = lbs)

@benchmark sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)
@profview sol_for = solve(prob_for, NLopt.LD_LBFGS(), maxiters = 10000)

# Import a solver package and solve the optimization problem
# sol = solve(prob, NLopt.LN_PRAXIS());
# sol.u
@time sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

@benchmark sol = solve(prob, NLopt.LD_LBFGS(), maxiters = 10000);

using ForwardDiff
ForwardDiff.gradient(calculate_cb_loss,initial_values)
Zygote.gradient(calculate_cb_loss,initial_values)[1]
# SS(Gali_2015_chapter_3_nonlinear, parameters = :std_std_a => .00001)

SS(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std³ = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_third_order)
std² = get_std(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
std¹ = get_std(Gali_2015_chapter_3_nonlinear)

std³([:π,:Y,:R],:)
std²(:π,:)
std¹(:π,:)

plot_solution(Gali_2015_chapter_3_nonlinear, :ν, algorithm = [:pruned_second_order, :pruned_third_order])

mean² = get_mean(Gali_2015_chapter_3_nonlinear, algorithm = :pruned_second_order)
mean¹ = get_mean(Gali_2015_chapter_3_nonlinear)

mean²(:π,:)
mean¹(:π,:)

get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => 1.5, :σ_σ_a => 2.0), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)

get_parameters(Gali_2015_chapter_3_nonlinear, values = true)

n = 5
res = zeros(3,n^2)

SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_π
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :σ_π => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_π", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Pi.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_Y
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_Y => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_Y", 
                    zlabel = "std(Inflation)")

savefig("measurement_uncertainty_Y.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.8,n) # std_std_a
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_a => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_a", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_tfp.png")


SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.4995,1.5005,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_std_z
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕᵖⁱ => i, :σ_σ_z => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:π)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "σ_σ_z", 
                    zlabel = "std(Inflation)")

savefig("stochastic_volatility_z.png")




SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_θ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_θ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_θ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_calvo.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.2,n) # std_ϕᵖⁱ
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ϕᵖⁱ => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (70,25),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ϕᵖⁱ", 
                    zlabel = "std(Inflation)")

savefig("uncertainty_infl_reaction.png")



SS(Gali_2015_chapter_3_nonlinear, parameters = init_params, derivatives = false)
l = 1
for i in range(1.45,1.55,n) # ϕ̄ᵖⁱ
    for k in range(.01,.02,n) # std_ā
        res[1,l] = i
        res[2,l] = k
        res[3,l] = get_std(Gali_2015_chapter_3_nonlinear, 
                            parameters = (:ϕ̄ᵖⁱ => i, :std_ā => k), 
                            algorithm = :pruned_third_order, 
                            derivatives = false)(:pi_ann)
        l += 1
    end
end

StatsPlots.surface(res[1,:],res[2,:],res[3,:],colorbar=false,
                    # camera = (60,65),
                    # camera = (90,25),
                    xlabel = "ϕᵖⁱ", 
                    ylabel = "std_ā", 
                    zlabel = "std(Inflation)")

savefig("tfp_std_dev.png")



