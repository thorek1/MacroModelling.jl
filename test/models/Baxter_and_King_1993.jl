@model Baxter_and_King_1993 begin
    uc[0] = c[0]^-1
    
    ul[0] = θ_l * l[0]^-1
    
    y[0] = A * k[-1]^θ_k * n[0]^θ_n
    
    fk[0] = θ_k * A * k[-1]^(θ_k - 1) * n[0]^θ_n
    
    fn[0] = θ_n * A * k[-1]^θ_k * n[0]^(θ_n - 1)
    
    γ_x * k[0] = (1- δ_k) * k[-1] + iv[0]
    
    l[0] + n[0] = 1
    
    c[0] + iv[0] = (1 - τ) * y[0] + tr[0] + check_walras[0]
    
    c[0] + iv[0] + gb[0] = y[0]
    
    τ * y[0] = gb[0] + tr[0]
    
    uc[0] = λ[0]
    
    ul[0] = λ[0] * (1 - τ) * fn[0]
    
    β * λ[1] * (q[1] + 1 - δ_k) = γ_x * λ[0]
    
    q[0] = (1 - τ) * fk[0]
    
    gb[0] = GB_BAR + e_gb[x]
    
    1 + r[0] = γ_x * λ[0] / (λ[1] * β)

    w[0] = fn[0]
end

@parameters Baxter_and_King_1993 begin
    A = 1.0
    
    γ_x = 1.016
    
    θ_k = 0.42
    
    θ_n = 1 - θ_k
    
    δ_k = 0.1
    
    N = 0.2
    
    L = 1 - N
    
    R = 0.065
    
    β = γ_x / (1 + R)
    
    sG = 0.2
    
    τBAR = 0.2
    
    τ = 0.2
    
    Q = γ_x / β - 1 + δ_k
    
    FK = Q / (1 - τBAR)
    
    K = (FK / ((θ_k * A * N^θ_n)))^(1 / (θ_k - 1))
    
    FN = θ_n * A * K^θ_k * N^(θ_n - 1)
    
    IV = (γ_x - 1 + δ_k) * K
    
    Y = A * N^(1 - θ_k) * K^θ_k
    
    GB_BAR = sG * Y
    
    C = Y - IV - GB_BAR
    
    UC = C^(-1)
    
    UL = UC * (1 - τBAR) * FN
    
    θ_l = UL * L
end
