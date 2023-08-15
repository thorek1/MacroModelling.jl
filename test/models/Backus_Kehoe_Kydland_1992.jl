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

    R[0] = LGM[1] / LGM[0]

    R_ann[0] = for operator = :*, lag in -3:0 R[lag] end
end

@parameters Backus_Kehoe_Kydland_1992 verbose = true begin
    K_ss = 11.0148
    # K[ss] = K_ss | beta
    # K[ss] = 10 | beta
    F_H_ratio = 1
    K{F}[ss] / K{H}[ss] = F_H_ratio | beta{F}
    K{H}[ss] = K_ss | beta{H}

    # beta    =    0.99
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