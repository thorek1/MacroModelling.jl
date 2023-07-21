### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 80b070f2-c26e-11ed-0b2e-d3e0ce0ed50f
using Pkg

# ╔═╡ 97e5810f-595f-46e5-80ca-32cc8180edfa
Pkg.activate(".")

# ╔═╡ 89d9f6a7-e219-4083-9584-befed5f31af6
Pkg.add("PlotlyJS")

# ╔═╡ 0fae708e-f33a-4a38-a7a5-664b130037e3
using MacroModelling, PlutoUI

# ╔═╡ 950069d3-33a1-441c-b0de-bb231feeda65
plotlyjs()

# ╔═╡ 05672b1b-f8b8-4da1-8e1d-102ef773ea63
show_pluto(x) = Text(sprint(show, "text/plain",x))

# ╔═╡ 1e6842cf-efbc-4d5d-a1c8-279f696e1af7
@model m begin
    y[0]=A[0]*k[-1]^alpha
    1/c[0]=beta*1/c[1]*(alpha*A[1]*k[0]^(alpha-1)+(1-delta))
    1/c[0]=beta*1/c[1]*(R[0]/Pi[+1])
    R[0] * beta =(Pi[0]/Pibar)^phi_pi
    A[0]*k[-1]^alpha=c[0]+k[0]-(1-delta*z_delta[0])*k[-1]
    z_delta[0] = 1 - rho_z_delta + rho_z_delta * z_delta[-1] + std_z_delta * delta_eps[x]
    A[0] = 1 - rhoz + rhoz * A[-1]  + std_eps * eps_z[x]
end

# ╔═╡ 6f2335ee-4620-4c5b-8850-44677b1e2563
@parameters m begin
    # alpha | k[ss] / (4 * y[ss]) = cap_share
    # cap_share = 1.66
    alpha = .157

    # beta | R[ss] = R_ss
    # R_ss = 1.0035
    beta = .999

    # delta | c[ss]/y[ss] = 1 - I_K_ratio
    # delta | delta * k[ss]/y[ss] = I_K_ratio #check why this doesnt solve for y
    # I_K_ratio = .15
    delta = .0226

    # Pibar | Pi[ss] = Pi_ss
    # Pi_ss = 1.0025
    Pibar = 1.0008

    phi_pi = 1.5
    rhoz = .9
    std_eps = .0068
    rho_z_delta = .9
    std_z_delta = .005
end

# ╔═╡ 101e97c1-76ed-46f0-ac22-8f96dced39f6
@bind Pibar Slider(1:0.001:1.1,default = 1.0008)

# ╔═╡ 3b9b78e6-e116-47a3-986b-6b3d8bd288e9
Pibar

# ╔═╡ d596521d-3141-4af9-a8fe-28c96ff22e2c
get_SS(m, parameters = :Pibar => Pibar) |> show_pluto

# ╔═╡ a2775fd4-59e9-4335-b4e5-188c7a6a04a4
@bind alpha Slider(.1:0.001:.5,default = .157)

# ╔═╡ 958cf8e7-de09-4d43-ba9c-14485b6aef31
alpha

# ╔═╡ 929aeb92-f606-4957-9d4b-7ea4a71b1dbc
get_solution(m,parameters = [:Pibar => Pibar, :alpha => alpha]) |> show_pluto

# ╔═╡ 5d6cc01f-a31e-422a-be69-fe42af212c7d
plot(m,parameters = :alpha => alpha)

# ╔═╡ 63ca45bd-06d1-4eb3-ad26-48ee372ba006
plot_fevd(m,parameters = :alpha => alpha)

# ╔═╡ 8142a6fb-8440-4f8e-b85a-f8bc0c99d2cd
plot_solution(m, :k,parameters = :alpha => alpha)

# ╔═╡ Cell order:
# ╠═80b070f2-c26e-11ed-0b2e-d3e0ce0ed50f
# ╠═97e5810f-595f-46e5-80ca-32cc8180edfa
# ╠═89d9f6a7-e219-4083-9584-befed5f31af6
# ╠═0fae708e-f33a-4a38-a7a5-664b130037e3
# ╠═950069d3-33a1-441c-b0de-bb231feeda65
# ╠═05672b1b-f8b8-4da1-8e1d-102ef773ea63
# ╠═1e6842cf-efbc-4d5d-a1c8-279f696e1af7
# ╠═6f2335ee-4620-4c5b-8850-44677b1e2563
# ╠═101e97c1-76ed-46f0-ac22-8f96dced39f6
# ╠═3b9b78e6-e116-47a3-986b-6b3d8bd288e9
# ╠═d596521d-3141-4af9-a8fe-28c96ff22e2c
# ╠═a2775fd4-59e9-4335-b4e5-188c7a6a04a4
# ╠═958cf8e7-de09-4d43-ba9c-14485b6aef31
# ╠═929aeb92-f606-4957-9d4b-7ea4a71b1dbc
# ╠═5d6cc01f-a31e-422a-be69-fe42af212c7d
# ╠═63ca45bd-06d1-4eb3-ad26-48ee372ba006
# ╠═8142a6fb-8440-4f8e-b85a-f8bc0c99d2cd
