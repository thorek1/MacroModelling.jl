using MacroModelling

include("../models/Smets_Wouters_2007.jl")

irfs = get_irf(Smets_Wouters_2007, shocks = :none, levels = true)

get_parameters(Smets_Wouters_2007, values = true)

starting_vals = get_irf(Smets_Wouters_2007, shocks = :none, levels = true, periods = 1)

irfalt = get_irf(Smets_Wouters_2007, parameters = :ctrend => .4, shocks = :none, levels = true, initial_state = vec(starting_vals))
# 3-dimensional KeyedArray(NamedDimsArray(...)) with keys:
# ↓   Variables ∈ 66-element Vector{Symbol}
# →   Periods ∈ 40-element UnitRange{Int64}
# ◪   Shocks ∈ 1-element Vector{Symbol}
# And data, 66×40×1 Array{Float64, 3}:
# [:, :, 1] ~ (:, :, :none):
#                  (1)            (2)            (3)            (4)            (5)            (6)            …  (35)            (36)            (37)            (38)            (39)            (40)
#   (:Pratio)        1.00002        1.00001        1.00001        1.0            1.0            0.999997           0.999996        0.999996        0.999996        0.999996        0.999997        0.999997
#   (:Sfunc)         5.20535e-18    5.57709e-18    5.93696e-18    6.17147e-18    6.31052e-18    6.37635e-18        3.24174e-18     3.15795e-18     3.0766e-18      2.99759e-18     2.92084e-18     2.84624e-18
#   (:SfuncD)       -0.000127875   -0.000156973   -0.000167241   -0.000164403   -0.000153445   -0.000138124       -5.82638e-6     -5.98484e-6     -6.12002e-6     -6.23154e-6     -6.31971e-6     -6.38538e-6
#   (:SfuncDflex)   -0.000195576   -0.000153698   -0.000125519   -0.000105025   -8.9291e-5     -7.67916e-5        -1.23046e-5     -1.19915e-5     -1.16906e-5     -1.14006e-5     -1.11208e-5     -1.08501e-5
#   (:Sfuncflex)    -1.66356e-17   -1.8244e-17    -1.93326e-17   -2.01298e-17   -2.06828e-17   -2.10323e-17  …    -1.19355e-17    -1.16417e-17    -1.13557e-17    -1.10773e-17    -1.08061e-17    -1.0542e-17
#   (:a)             1.0            1.0            1.0            1.0            1.0            1.0                1.0             1.0             1.0             1.0             1.0             1.0
#   (:afunc)        -2.24353e-5    -2.2646e-5     -2.28258e-5    -2.29284e-5    -2.29355e-5    -2.28455e-5        -1.15466e-5     -1.12571e-5     -1.09758e-5     -1.07023e-5     -1.04363e-5     -1.01776e-5
#   (:afuncD)        0.0384623      0.0384622      0.0384621      0.0384621      0.0384621      0.0384621          0.0384663       0.0384664       0.0384665       0.0384666       0.0384667       0.0384668
#   (:afuncDflex)    0.0384611      0.0384612      0.0384614      0.0384616      0.0384618      0.0384619          0.0384661       0.0384662       0.0384663       0.0384664       0.0384665       0.0384666
#   (:afuncflex)    -2.57357e-5    -2.53111e-5    -2.4842e-5     -2.43525e-5    -2.38544e-5    -2.3354e-5    …    -1.19586e-5     -1.16813e-5     -1.14103e-5     -1.11457e-5     -1.08871e-5     -1.06346e-5
#   (:b)             1.0            1.0            1.0            1.0            1.0            1.0                1.0             1.0             1.0             1.0             1.0             1.0
#    ⋮                                                                           ⋮                           ⋱     ⋮                                                                               ⋮
#   (:wdot)          1.0            1.0            1.0            1.0            1.0            1.0                1.0             1.0             1.0             1.0             1.0             1.0
#   (:wdotl)         1.0            1.0            1.0            1.0            1.0            1.0          …     1.0             1.0             1.0             1.0             1.0             1.0
#   (:wflex)         0.794744       0.794743       0.794742       0.794741       0.794739       0.794738           0.794711        0.79471         0.794709        0.794709        0.794708        0.794708
#   (:wnew)          0.794783       0.794764       0.794749       0.794736       0.794725       0.794717           0.794694        0.794693        0.794693        0.794693        0.794693        0.794693
#   (:xi)            8.03946        8.03968        8.03986        8.03998        8.04008        8.04015            8.0414          8.04144         8.04147         8.0415          8.04153         8.04156
#   (:xiflex)        8.03989        8.03984        8.03984        8.03987        8.03991        8.03996            8.04129         8.04132         8.04136         8.04139         8.04142         8.04145
#   (:y)             1.35931        1.3593         1.35928        1.35927        1.35925        1.35924      …     1.35923         1.35923         1.35923         1.35923         1.35923         1.35923
#   (:yflex)         1.35927        1.35926        1.35926        1.35926        1.35925        1.35925            1.35924         1.35924         1.35924         1.35924         1.35924         1.35924
#   (:ygap)          0.00308965     0.00271599     0.00183785     0.000800933   -0.000206243   -0.00109057        -0.00109644     -0.00104964     -0.00100651     -0.000966695    -0.000929851    -0.000895679
#   (:zcap)          0.999417       0.999411       0.999407       0.999404       0.999404       0.999406           0.9997          0.999707        0.999715        0.999722        0.999729        0.999735
#   (:zcapflex)      0.999331       0.999342       0.999354       0.999367       0.99938        0.999393           0.999689        0.999696        0.999703        0.99971         0.999717        0.999724