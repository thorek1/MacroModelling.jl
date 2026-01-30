
# ===========================================================================
# Tests for get_equations and get_calibration_equations filter functionality
# ===========================================================================

# Test with symbol-based model (simple RBC)
@model TestRBC begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters TestRBC begin
    std_z = 0.01
    ρ = 0.2
    δ = 0.02
    α = 0.5
    β = 0.95
end

# Basic filtering tests - symbol-based model
all_eqs = get_equations(TestRBC)
@test length(all_eqs) == 4

# Variable filtering
@test length(get_equations(TestRBC, filter=:k)) == 3  # k appears in 3 equations
@test length(get_equations(TestRBC, filter=:c)) == 2  # c appears in 2 equations
@test length(get_equations(TestRBC, filter=:z)) == 3  # z appears in 3 equations
@test length(get_equations(TestRBC, filter=:q)) == 2  # q appears in 2 equations

# Exact timing filtering
@test length(get_equations(TestRBC, filter="k[0]")) == 2   # k[0] in 2 equations
@test length(get_equations(TestRBC, filter="k[-1]")) == 2  # k[-1] in 2 equations
@test length(get_equations(TestRBC, filter="c[0]")) == 2   # c[0] in 2 equations  
@test length(get_equations(TestRBC, filter="c[1]")) == 1   # c[1] in 1 equation
@test length(get_equations(TestRBC, filter="z[1]")) == 1   # z[1] in 1 equation
@test length(get_equations(TestRBC, filter="z[0]")) == 2   # z[0] in 2 equations
@test length(get_equations(TestRBC, filter="z[-1]")) == 1  # z[-1] in 1 equation

# Parameter filtering
@test length(get_equations(TestRBC, filter=:δ)) == 2     # δ in 2 equations
@test length(get_equations(TestRBC, filter=:α)) == 2     # α in 2 equations
@test length(get_equations(TestRBC, filter=:β)) == 1     # β in 1 equation
@test length(get_equations(TestRBC, filter=:ρ)) == 1     # ρ in 1 equation
@test length(get_equations(TestRBC, filter=:std_z)) == 1 # std_z in 1 equation

# Shock filtering
@test length(get_equations(TestRBC, filter=:eps_z)) == 1      # shock eps_z
@test length(get_equations(TestRBC, filter="eps_z[x]")) == 1  # shock with timing

# Symbol vs String equivalence
@test get_equations(TestRBC, filter=:k) == get_equations(TestRBC, filter="k")
@test get_equations(TestRBC, filter=:δ) == get_equations(TestRBC, filter="δ")

TestRBC = nothing


# Test with string-based model (curly brace syntax like Backus_Kehoe_Kydland_1992)
@model TestBKK begin
    Y{H}[0] = K{H}[-1]^α * exp(z{H}[0])
    K{H}[0] = (1-δ)*K{H}[-1] + I{H}[0]
    C{H}[0] + I{H}[0] = Y{H}[0]
    1/C{H}[0] = β/C{H}[1] * (α*Y{H}[1]/K{H}[0] + 1 - δ)
    z{H}[0] = ρ*z{H}[-1] + eps{H}[x]
end

@parameters TestBKK begin
    α = 0.33
    δ = 0.1
    β = 0.95
    ρ = 0.9
end

# Test curly brace syntax filtering
all_eqs_bkk = get_equations(TestBKK)
@test length(all_eqs_bkk) == 5

# Filter by variable with subscript
@test length(get_equations(TestBKK, filter="K{H}")) == 3   # K{H} appears in 3 equations
@test length(get_equations(TestBKK, filter="Y{H}")) == 3   # Y{H} appears in 3 equations
@test length(get_equations(TestBKK, filter="C{H}")) == 2   # C{H} appears in 2 equations
@test length(get_equations(TestBKK, filter="z{H}")) == 2   # z{H} appears in 2 equations

# Filter by shock with subscript
@test length(get_equations(TestBKK, filter="eps{H}")) == 1

# Test exact timing with curly braces
@test length(get_equations(TestBKK, filter="K{H}[-1]")) == 2  # K{H}[-1] in 2 equations
@test length(get_equations(TestBKK, filter="K{H}[0]")) == 2   # K{H}[0] in 2 equations

TestBKK = nothing


# Test calibration equation filtering using the same TestRBC model
@model TestCalib begin
    1  /  c[0] = (β  /  c[1]) * (α * exp(z[1]) * k[0]^(α - 1) + (1 - δ))
    c[0] + k[0] = (1 - δ) * k[-1] + q[0]
    q[0] = exp(z[0]) * k[-1]^α
    z[0] = ρ * z[-1] + std_z * eps_z[x]
end

@parameters TestCalib begin
    std_z = 0.01
    ρ = 0.2
    capital_to_output = 1.5
    k[ss] / (4 * q[ss]) = capital_to_output | δ
    alpha = .5
    α = alpha
    β = 0.95
end

# Test calibration equations
calib_eqs = get_calibration_equations(TestCalib)
@test length(calib_eqs) == 1  # One calibration equation

# Filter calibration equations by parameter
@test length(get_calibration_equations(TestCalib, filter=:capital_to_output)) == 1

# Filter calibration equations by steady state variable
@test length(get_calibration_equations(TestCalib, filter="k[ss]")) == 1
@test length(get_calibration_equations(TestCalib, filter="q[ss]")) == 1

# Shocks should not appear in calibration equations
@test length(get_calibration_equations(TestCalib, filter=:eps_z)) == 0

TestCalib = nothing


# Test model with multiple lead/lag indices
@model TestMultiLag begin
    Δk[0] = log(k[0]) - log(k[-4])
    c[0] + k[0] = k[-1]^α + (1-δ)*k[-1]
    1/c[0] = β/c[1] * (α*k[0]^(α-1) + 1-δ)
    z[0] = ρ*z[-1] + eps_z[x] + eps_news[x-1]
end

@parameters TestMultiLag begin
    α = 0.33
    δ = 0.025
    β = 0.99
    ρ = 0.9
end

# Test exact timing with multiple lags
@test length(get_equations(TestMultiLag, filter="k[-1]")) == 1  # Only in capital accumulation
@test length(get_equations(TestMultiLag, filter="k[-4]")) == 1  # Only in Δk equation
@test length(get_equations(TestMultiLag, filter="k[0]")) == 3   # In multiple equations

# Test shock with news (past shock timing)
@test length(get_equations(TestMultiLag, filter="eps_news")) == 1
@test length(get_equations(TestMultiLag, filter="eps_news[x-1]")) == 1

TestMultiLag = nothing
