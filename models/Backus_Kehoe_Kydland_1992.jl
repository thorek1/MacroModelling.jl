# Backus, D.K., Kehoe, P.J., and Kydland, F.E. (1992). International real business cycles.
# Journal of Political Economy, 100(4), 745–775. https://www.jstor.org/stable/2138686

@model Backus_Kehoe_Kydland_1992 begin
    # To scale to N countries, add/remove symbols in every [H, F] list below.
    # Update the @parameters block rho definitions and steady-state calibration accordingly.
    for co in [H, F]
        Y{co}[0] = ((LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1 - theta{co}))^(-nu{co}) + sigma{co} * Z{co}[-1]^(-nu{co}))^(-1 / nu{co})

        K{co}[0] = (1 - delta{co}) * K{co}[-1] + S{co}[0]

        X{co}[0] = for lag in (-4 + 1):0 phi{co} * S{co}[lag] end

        A{co}[0] = (1 - eta{co}) * A{co}[-1] + N{co}[0]

        L{co}[0] = 1 - alpha{co} * N{co}[0] - (1 - alpha{co}) * eta{co} * A{co}[-1]

        U{co}[0] = (C{co}[0]^mu{co} * L{co}[0]^(1 - mu{co}))^gamma{co}

        psi{co} * mu{co} / C{co}[0] * U{co}[0] = LGM[0]

        psi{co} * (1 - mu{co}) / L{co}[0] * U{co}[0] * (-alpha{co}) = - LGM[0] * (1 - theta{co}) / N{co}[0] * (LAMBDA{co}[0] * K{co}[-4]^theta{co} * N{co}[0]^(1 - theta{co}))^(-nu{co}) * Y{co}[0]^(1 + nu{co})

        for lag in 0:(4 - 1)  
            beta{co}^lag * LGM[lag] * phi{co}
        end +
        for lag in 1:4
            -beta{co}^lag * LGM[lag] * phi{co} * (1 - delta{co})
        end = beta{co}^4 * LGM[+4] * theta{co} / K{co}[0] * (LAMBDA{co}[+4] * K{co}[0]^theta{co} * N{co}[+4]^(1 - theta{co}))^(-nu{co}) * Y{co}[+4]^(1 + nu{co})

        LGM[0] = beta{co} * LGM[+1] * (1 + sigma{co} * Z{co}[0]^(-nu{co} - 1) * Y{co}[+1]^(1 + nu{co}))

        NX{co}[0] = (Y{co}[0] - (C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1])) / Y{co}[0]
    end

    # Shock process: each country's lambda depends on its own lag and spillovers from all others.
    # The inner accumulator sums over co2 != co, handling any number of countries.
    for co in [H, F]
        (LAMBDA{co}[0] - 1) = rho{co}{co} * (LAMBDA{co}[-1] - 1) + for co2 in [H, F] if co2 != co rho{co}{co2} * (LAMBDA{co2}[-1] - 1) end end + Z_E{co} * E{co}[x]
    end

    # World resource constraint: sum of expenditures = sum of outputs
    for co in [H, F] C{co}[0] + X{co}[0] + Z{co}[0] - Z{co}[-1] end = for co in [H, F] Y{co}[0] end
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

    # To scale to N countries, update the country list in the loops below
    for co1 in [H, F]
        for co2 in [H, F]
            if co1 == co2
                rho{co1}{co2} = 0.906
            else
                rho{co1}{co2} = 0.088 / 1 # divide by N - 1 for the model to be stable with more countries 
            end
        end
    end
end