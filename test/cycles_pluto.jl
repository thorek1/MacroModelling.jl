### A Pluto.jl notebook ###
# v0.19.36

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

# ╔═╡ bad5a8f1-fdf0-47f4-98e5-25f8ec03426b
import Pkg; Pkg.activate("D:\\CustomTools\\MacroModelling.jl-cycles")#; Pkg.add("PlutoUI")

# ╔═╡ 1989ec7e-8d47-11ee-13a7-87c968f56766
using MacroModelling

# ╔═╡ d450c895-1433-4fcc-9905-1c18879e49ec
using PlutoUI

# ╔═╡ 10aca7df-6bb3-4dfc-86e0-a431107ea716
using Random

# ╔═╡ c762bf37-305b-463e-b573-6befd69288ee
using LinearAlgebra

# ╔═╡ 8fc6e932-6d22-46c3-8ad6-b26684c38fa1
show_pluto(x) = Text(sprint(show, "text/plain",x));

# ╔═╡ 97ff64bd-961f-43c3-b46d-f3b17e1aabd2
import StatsPlots

# ╔═╡ ea44d30c-2591-43b5-9cbf-ee301bb14020
import LinearAlgebra as ℒ

# ╔═╡ 1c646fb3-8f43-440f-834e-9c4e239bd5ad
md"# An RBC model with habit formation and a nonlinaer autoregressive investment process."

# ╔═╡ 2bdbbaf9-03dd-4492-abf4-ff7aca63e19e
@model RBC_baselin begin
    λ[0] = (c[0] - h * c[-1]) ^ (-σ) - β * h * (c[1] - h * c[0]) ^ (-σ)

    λ[0] = β *  λ[1] * (α * z[1] * (k[0] / l[1]) ^ (α - 1) + 1 - δ)

	ψ / (1 - l[0]) =  λ[0] * w[0]

	k[0] = (1 - δ) * k[-1] + i[0] + G[0]

    G[0] = a3 * (i[-1] - i[ss]) ^ 3 + a2 * (i[-1] - i[ss]) ^ 2 + a1 * (i[-1] - i[ss])

	y[0] = c[0] + i[0] + g[0]

	y[0] = z[0] * k[-1] ^ α * l[0] ^ (1 - α)

	w[0] = y[0] * (1 - α) / l[0]

	r[0] = y[0] * α * 4 / k[-1]

	z[0] = (1 - ρᶻ) + ρᶻ * z[-1] + σᶻ * ϵᶻ[x]

	g[0] = (1 - ρᵍ) * ḡ + ρᵍ * g[-1] + σᵍ * ϵᵍ[x]

end

# ╔═╡ f5983385-9164-4509-bc44-ab3043d379cf
@parameters RBC_baselin begin
    a1 = 0.0005

    a2 = 0.999
	
    a3 = 0

	h = .5
	
	σᶻ = 0.066

	σᵍ = .104

	σ = 1

	α = 1/3

	i_y = 0.25

	k_y = 10.4

	ρᶻ = 0.97

	ρᵍ = 0.989

	g_y = 0.2038

	ḡ | ḡ = g_y * y[ss]

    δ = i_y / k_y

    β = 1 / (α / k_y + (1 - δ))

	ψ | l[ss] = 1/3
end

# ╔═╡ d61dd34b-7765-4cac-b610-4e7e1031a9e8
md"This model has a reduced form nonlinear structure driving I and thereby K."

# ╔═╡ 534226bf-cfef-4fd0-b40a-f9a3db201e64
get_SS(RBC_baselin) |> show_pluto

# ╔═╡ c365b37a-bef3-4a0d-96aa-3af2db6180ca
md"We approximate around a non-zero non stochastic steady state (NSSS)."

# ╔═╡ dc224752-f3e2-489a-992d-101925299105
get_solution(RBC_baselin) |> show_pluto

# ╔═╡ 382b69e7-f843-47ac-8bb2-33912bfaa473
MacroModelling.get_eigenvalues(RBC_baselin) |> show_pluto

# ╔═╡ 38d40765-66d8-44a0-af2e-bffad2733994
RBC_baselin

# ╔═╡ fc8d5002-3d29-45ec-9665-2cd863b4a748


# ╔═╡ 1508faa7-3e2e-4f3a-90e1-6616b1544610
md"## Plotting"

# ╔═╡ ac41c574-c53e-4127-918b-5be4040cc601
md"h = .292 and a1 = 92.893 gives interesting results"

# ╔═╡ d127d880-bf08-408d-b67d-e13ec65d7b66
@bind h Slider(.29:.0005:.31, default = 0.292)

# ╔═╡ 18c1c69a-6b00-41a5-af68-6e7b4853abf3
h

# ╔═╡ 383372a0-dc81-4343-bff8-74fe688e96e1
@bind a1 Slider(0:.001:100, default = 92.893)

# ╔═╡ 2953265e-0e4c-42c9-a933-534361e2bbd5
a1

# ╔═╡ 47901235-dc5d-45a2-908a-b1c00648727c
@bind a2 Slider(6.2:.0001:6.6, default = .00)

# ╔═╡ 76b30796-b561-43a4-99e1-de09a2e33292
a2

# ╔═╡ f13e7256-b624-41ee-bf4d-aa912875c949
@bind a3 Slider(-1:.01:1, default = .00)

# ╔═╡ 9f292898-b6a8-4bc5-827e-6c8b75309c4a
Random.seed!(1)

# ╔═╡ 7e1fff86-654b-4df0-b875-8135bfbca760
MacroModelling.get_eigenvalues(RBC_baselin, parameters = (:a1 => a1, :a2 => a2, :a3 => a3, :h => h)) |> show_pluto

# ╔═╡ ba6174b3-97cb-4a76-a7e6-68c646fdf916
plot_irf(RBC_baselin, parameters = (:a1 => a1, :a2 => a2, :a3 => a3, :h => h,  :σᵍ => .104), algorithm = :second_order, variables = [:i,:c,:k,:y,:l,:w], periods = 120)[1]

# ╔═╡ 938f34f6-fd79-4fd0-9ad1-82619e2b859f
SSS(RBC_baselin, algorithm = :second_order, derivatives = false, parameters = (:a1 => a1, :a2 => a2, :a3 => a3, :h => h,  :σᵍ => .104)) |> show_pluto

# ╔═╡ 85c6b0cb-5eb1-4fcb-9365-32bca4f35fb8
stds = get_std(RBC_baselin, derivatives = false, parameters = (:a1 => 0.0, :a2 => a2, :a3 => a3, :h => h,  :σᵍ => .104));

# ╔═╡ 9c727850-1ff6-435c-8ea4-926206cd7047
stds |> show_pluto

# ╔═╡ f882d5b1-7da0-458b-b079-33fbbcebd057
plot_solution(RBC_baselin, :i, algorithm = :second_order, parameters = (:a1 => 75.0, :a2 => 0.0, :a3 => 0.0, :h => 0.0,  :σᵍ => .104), standard_deviations = 5*stds, initial_state = :nsss)[1]

# ╔═╡ 9ca2fac3-7695-4b82-8f75-b7db5f35d84d
get_std(RBC_baselin, parameters = (:a1 => 40.0, :a2 => a2, :a3 => a3, :h => h,  :σᵍ => .104), algorithm = :pruned_second_order, derivatives = false) |> show_pluto

# ╔═╡ 8beb523a-102b-4474-8044-4e15586c674f
md"## Only second order needed for cycles"

# ╔═╡ 76c07b1a-0994-47fa-89f9-cc060b51841a
# Generate cyclical time series data
begin 
	n_quarters = 1000
	cycle_length = 80
	time = 0:(n_quarters - 1)
	amperes = 10 * sin.(2 * π * time / cycle_length) + 5 * cos.(2 * π * time / cycle_length) + rand(n_quarters)*1 .+ 10
end

# ╔═╡ 0dc35e06-d94f-4f88-9ebe-bbf6e0bf8366
# Prepare the data for the polynomial model
begin
	X = amperes[1:end-1]  # Exclude the last point for X
	y = amperes[2:end]    # Exclude the first point for y
	# X_matrix = hcat(X,X.^2, X.^3, ones(length(X)))
	# X_matrix = hcat(ones(length(X)), X, X.^2)
	X_matrix = hcat( X, X.^2)
end

# ╔═╡ fdb8b36f-579b-4a9a-81aa-31bc366dce02
coefficients = (X_matrix' * X_matrix) \ (X_matrix' * y)

# ╔═╡ c1ec07df-0104-4b0d-816c-03b999cc4609
begin
	# Generate predictions from the model
	fitted_series = X_matrix * coefficients
	
	# Plotting the original and fitted series
	StatsPlots.plot(time[1:end-1], X, label="Original Series", title="Cyclical Time Series and Fitted Model", xlabel="Time (Quarters)", ylabel="Amperes")
	StatsPlots.plot!(time[1:end-1], fitted_series, label="Fitted Series")
end

# ╔═╡ 8c877be1-5861-45ac-9087-e9e8c2f7786d
md"## Dewachter and Wouter model"

# ╔═╡ e63463de-ed39-4b2f-ad9e-2ba217db931d

@model HeKr_2012 begin
	A[0] = 0.05 * cA + 0.95 * A[-1] + cA * σᵉ * esigma[x] # gA[0] + 

	# gA[0] = 0.8 * gA[-1]

	# us[0] = log(es[0])

	usc[0] = es[0] ^ (-cra_usc)

	ERs[0] = cm_ers * cm * theta[0] * (rrs[1] - rf[0]) ^ 2# + riskshock[0]

	es[0] = es[-1] * (1 + cm / cm_es * (R_tilde[0] - 1) - ceta / cm_es - cexit * (es[-1] - es_bar) ^ 2) + es_bar * (cexit * (es[-1] - es_bar) ^ 2 + .0)# + eshock[x]
	# es appears to be the reputation

	0 = ps[0] * ss_bar - h[0] - pbn[0] * b[0]

	Y[0] = A[0] * L[0] ^ calfa * K[0] ^ (1 - calfa) - cfix * Y_bar

	K[0] = (1 - cdelta) * K[-1] + K[-1] * inves[-1]

	inves[0] = cdelta + (Q[0] - 1) / ckappa

	Y[0] = cb[0] + K[0] * inves[0] + K[0] * ckappa * 0.5 * (inves[0] - cdelta) ^ 2

	K[0] * Q[0] = ps[0] * ss_bar

	theta[0] * h[0] = K[0] * Q[0]

	1.0 * (es[0] - es_low) * Qes[0] = K[0] * Q[0]

	theta[0] = 1 / (1 - clambda) + (1 - clambda) ^ 2 * Qes[-1] ^ 3

	ubc[0] = (cb[0] - chab * cb[-1] - clabut / (1 + cinvfe) * L[0] ^ (1 + cinvfe)) ^ (-cgammab)

	ubn[0] = clabut * (cb[0] - chab * cb[-1] - clabut / (1 + cinvfe) * L[0] ^ (1 + cinvfe)) ^ (-cgammab) * L[0] ^ cinvfe

	pbn[0] * b[0] + h[0] + cb[0] + K[0] * (inves[0] - cdelta + ckappa * 0.5 * (inves[0] - cdelta) ^ 2) = wagebill[0] + b[-1] / (1 + pinf[0]) + R_tilde[0] * h[-1]

	wb[0] = h[0] + pbn[0] * b[0]

	ubc[0] * pbf[0] = cbetab * ubc[1]

	pbn[0] * ubc[0] = cbetab * ubc[1] / (1 + pinf[1])

	wagebill[0] = L[0] * wage[0]

	wage[0] * lagr[0] = K[0] ^ (1 - calfa) * A[0] * calfa * L[0] ^ (calfa - 1)

	winf[0] = cbetas * winf[1] + 0.5 * pinf[-1] - pinf[0] * 0.5 * cbetas + 0.02 * (ubn[0] * clandaw - ubc[0] * wage[0]) / (clandaw * ubn_bar)

	mrs[0] = (ubn[0] * clandaw - ubc[0] * wage[0]) / (clandaw * ubn_bar)

	wage[0] = wage[-1] * (1 + winf[0] - pinf[0])

	pinf[0] = pinf[1] * cbetas - 0.1 * (lagr[0] - lagr_bar) / lagr_bar

	rn[0] = rf_bar + pinf[0] * 1.50# + exor[0]

	ss_bar * div[0] = Y[0] - K[0] * cdelta - wagebill[0]

	rf[0] = (1 - pbf[0]) / pbf[0]

	rn[0] = (1 - pbn[0]) / pbn[0]

	rs[0] = (div[1] + ps[1] - ps[0]) / ps[0]

	# sigma[0] = csigma * (1 + esigma[x])

	# riskshock[0] = 0.00 * riskshock[-1]

	# exor[0] = 0.75 * exor[-1]

	# Y_rel[0] = (Y[0] - Y_bar) / Y_bar

	# L_rel[0] = (L[0] - L_bar) / L_bar

	# K_rel[0] = (K[0] - K_bar) / K_bar

	# wage_rel[0] = (wage[0] - wage_bar) / wage_bar

	# inv_rel[0] = K_bar * (K[0] * inves[0] - K_bar * inves_bar) / inves_bar

	# es_rel[0] = (es[0] - es_bar) / es_bar

	# cb_rel[0] = (cb[0] - cb_bar) / cb_bar

	# div_rel[0] = (div[0] - div_bar) / div_bar

	# wb_rel[0] = (wb[0] - wb_bar) / wb_bar

	# ps_rel[0] = (ps[0] - ps_bar) / ps_bar

	# b_rel[0] = (b[0] - b_bar) / b_bar

	# h_rel[0] = (h[0] - h_bar) / h_bar

	# des[0] = (es[0] * gk[-1] - es[-1]) / es[-1]

	# dcb[0] = (cb[0] * gk[-1] - cb[-1]) / cb[-1]

	# dlambdas[0] = log(es[0] ^ (-cgammas)) - log(es[-1] ^ (-cgammas))

	# dlambdab[0] = log(cb[0] ^ (-cgammab)) - log(cb[-1] ^ (-cgammab))

	rrs[0] = (ps[0] + div[0] - ps[-1]) / ps[-1]

	ERs[0] = rs[0] - rf[0]

	# ER_tilde[0] = R_tilde[1]

	# ERt[0] = ER_tilde[0] - rf[0] - 1

	# Evarrs[0] = (rrs[1] - rs[0]) ^ 2

	# Evarrt[0] = (R_tilde[1] - ER_tilde[0]) ^ 2

	# Evardlcs[0] = dlambdas[1] ^ 2

	# EvardlQ[0] = dlQ[1] ^ 2

	# gk[0] = K[0] / K[-1]

	# EQ[0] = Q[1]

	# dly[0] = 100 * (log(Y[0]) - log(Y[-1]))

	# dlc[0] = 100 * (log(cb[0]) - log(cb[-1]))

	# dli[0] = 100 * (log(K[0] * inves[0]) - log(K[-1] * inves[-1]))

	# dlk[0] = 100 * (log(K[0]) - log(K[-1]))

	# dlh[0] = 100 * (log(h[0]) - log(h[-1]))

	# dlQ[0] = 100 * (log(Q[0]) - log(Q[-1]))

end

# ╔═╡ 7cbcbd5c-4203-46c8-be25-14ca8f016eae
@parameters HeKr_2012 begin
    σᵉ = 0.0075# 0.75*0.01*2/(nper^0.5)

	nper = 4

	cbetas = 1/(1+.04/nper)

	cbetab = 1/(1+.04/nper)

	cgammas = 1.0

	cgammab = 1.0

	clambda = 0.5

	cdelta = 0.1/nper

	csigma = 1

	ckappa = 25

	cm = 2.5

	cls = 0.6

	calfa = 0.6

	cfix = 0.20

	cA = (cdelta+1/cbetas-1)/(1-calfa)*(1+cfix)

	chab = 0.300

	cinvfe = 1.0

	K_bar = 1

	L_bar = 1

	Y_bar = cA*L_bar^calfa*K_bar^(1-calfa)/(1+cfix)

	inves_bar = cdelta

	cb_bar = Y_bar-K_bar*inves_bar

	rf_bar = 1/cbetas-1

	rs_bar = 1/cbetas-1

	pbf_bar = 1/(1+rf_bar)

	Q_bar = 1

	cte = 1.

	es_bar = 1.0

	es_low = 0.2

	Qes_bar = Q_bar/(1.0*(es_bar-es_low))

	theta_bar = 1/(1-clambda)+(1-clambda)^2*Qes_bar^3

	h_bar = Q_bar/theta_bar

	us_bar = log(es_bar)

	usc_bar = es_bar^(-1)

	ss_bar = 1

	wagebill_bar = Y_bar-K_bar*inves_bar-rs_bar

	wage_bar = wagebill_bar/L_bar

	lagr_bar = K_bar^(1-calfa)*cA*calfa*1/wage_bar*L_bar^(calfa-1)

	cpsi = wagebill_bar/(K_bar*Q_bar)

	div_bar = (Y_bar-wagebill_bar-K_bar*inves_bar)/ss_bar

	ps_bar = div_bar/rs_bar

	b_bar = (ss_bar*ps_bar-h_bar)/pbf_bar

	clab = 1

	clabut = wage_bar/L_bar

	ubc_bar = (cb_bar-cb_bar*chab-clab*clabut/(1+cinvfe)*L_bar^(1+cinvfe))^(-cgammab)

	ubn_bar = clabut*(cb_bar-cb_bar*chab-clab*clabut/(1+cinvfe)*L_bar^(1+cinvfe))^(-cgammab)*L_bar^cinvfe

	wb_bar = h_bar+pbf_bar*b_bar

	R_tilde_bar = (-b_bar)/(ss_bar*ps_bar-pbf_bar*b_bar)+(div_bar+ss_bar*ps_bar)/(ss_bar*ps_bar-pbf_bar*b_bar)

	ceta = cm*(R_tilde_bar-1)

	crts = 1

	cflex = 1

	shocksize = 0

	shockper = 0

	shockvar = 0

	clandaw = 1

	cexit = 0.10

	cra_usc = 1

	cm_ers = 1.5

	cm_es = 1.5

	ubc > 50
	theta > 2.48
	ubn > 2
	R_tilde > 1
	# ER_tilde > 1
	wage < .06
	# us < 1e-5
	lagr > 1.1
	pbn < 1
	pbf < 1
	Y < .1
	h > .4
	b > .6
	rf < .02
	rs < .02
	div < .02
	rrs < .02
end

# ╔═╡ f103ce35-74be-4c2c-aee7-691926530174
plot_irf(HeKr_2012)[1]

# ╔═╡ b8d0c055-b7ef-4791-a9b6-5af7d07b0e8e
plot_irf(HeKr_2012, algorithm = :pruned_second_order)[1]

# ╔═╡ 1c3e9adb-ece1-4067-9ac2-2389de62d129
plot_irf(HeKr_2012, algorithm = :pruned_third_order, parameters = :σᵉ => .001)[1]

# ╔═╡ 2e50a12d-6aa5-4e91-bc46-06ba71bf82cd
plot_irf(HeKr_2012, algorithm = :third_order, parameters = :σᵉ => .001)[1]

# ╔═╡ 7e3cc930-8015-43e7-ac18-dc8d1d274c4c
HeKr_2012

# ╔═╡ 427b3fd2-a6cb-4a1e-b286-b88596ba2ac5
get_parameters(HeKr_2012, values = true) |> show_pluto

# ╔═╡ 0f15c89a-4505-4683-9413-ab2881c11b2e
@bind chab Slider(.01:.01:1.0, default = 0.3)

# ╔═╡ b4d095bd-61b0-411f-bb08-910c2c5a7804
chab

# ╔═╡ 77637b57-ac4d-4459-bbcb-e070378bc1ae
@bind ckappa Slider(5.0:1.0:10000.0, default = 25.0)

# ╔═╡ d8fc6d1d-7cc5-42b9-b3c7-33d9666cba2d
ckappa

# ╔═╡ 299b0366-a510-438b-bb41-c0ef679f2403
@bind cexit Slider(-1:.00001, default = 0.0)

# ╔═╡ 41db6cb5-5c5a-4556-b2eb-3cb6c3b40013
get_eigenvalues(HeKr_2012, parameters = (:cexit => cexit, :ckappa => ckappa, :chab => chab)) |> show_pluto

# ╔═╡ 13c9dcce-fa5b-4cc4-b684-c0b791a79492
cexit

# ╔═╡ 23da9f62-5c43-4e8f-b05c-da2799e86474
plot_irf(HeKr_2012, 
	algorithm = :second_order, 
	parameters = (:cexit => cexit, :ckappa => ckappa, :chab => chab), 
	periods = 20)[2]

# ╔═╡ 91ba2048-42a9-49f9-8861-a545ff314b91
plot_solution(HeKr_2012, :K, 
				algorithm = :third_order, 
				parameters = (:σᵉ => .00000001))[1]

# ╔═╡ dafecb7b-0c4f-4084-9d24-302b09d98fa6
SSS(HeKr_2012, algorithm = :second_order, parameters = :σᵉ => .000001) |> show_pluto

# ╔═╡ f7b68127-2de2-4baf-9636-106310af6394
SS(HeKr_2012, parameters = :σᵉ => .000001) |> show_pluto

# ╔═╡ Cell order:
# ╠═bad5a8f1-fdf0-47f4-98e5-25f8ec03426b
# ╠═8fc6e932-6d22-46c3-8ad6-b26684c38fa1
# ╠═1989ec7e-8d47-11ee-13a7-87c968f56766
# ╠═97ff64bd-961f-43c3-b46d-f3b17e1aabd2
# ╠═d450c895-1433-4fcc-9905-1c18879e49ec
# ╠═ea44d30c-2591-43b5-9cbf-ee301bb14020
# ╟─1c646fb3-8f43-440f-834e-9c4e239bd5ad
# ╠═2bdbbaf9-03dd-4492-abf4-ff7aca63e19e
# ╠═f5983385-9164-4509-bc44-ab3043d379cf
# ╟─d61dd34b-7765-4cac-b610-4e7e1031a9e8
# ╠═534226bf-cfef-4fd0-b40a-f9a3db201e64
# ╟─c365b37a-bef3-4a0d-96aa-3af2db6180ca
# ╠═dc224752-f3e2-489a-992d-101925299105
# ╠═382b69e7-f843-47ac-8bb2-33912bfaa473
# ╠═38d40765-66d8-44a0-af2e-bffad2733994
# ╠═fc8d5002-3d29-45ec-9665-2cd863b4a748
# ╟─1508faa7-3e2e-4f3a-90e1-6616b1544610
# ╟─ac41c574-c53e-4127-918b-5be4040cc601
# ╠═d127d880-bf08-408d-b67d-e13ec65d7b66
# ╠═18c1c69a-6b00-41a5-af68-6e7b4853abf3
# ╠═383372a0-dc81-4343-bff8-74fe688e96e1
# ╠═2953265e-0e4c-42c9-a933-534361e2bbd5
# ╠═47901235-dc5d-45a2-908a-b1c00648727c
# ╠═76b30796-b561-43a4-99e1-de09a2e33292
# ╠═f13e7256-b624-41ee-bf4d-aa912875c949
# ╠═10aca7df-6bb3-4dfc-86e0-a431107ea716
# ╠═9f292898-b6a8-4bc5-827e-6c8b75309c4a
# ╠═7e1fff86-654b-4df0-b875-8135bfbca760
# ╠═ba6174b3-97cb-4a76-a7e6-68c646fdf916
# ╠═938f34f6-fd79-4fd0-9ad1-82619e2b859f
# ╠═85c6b0cb-5eb1-4fcb-9365-32bca4f35fb8
# ╠═9c727850-1ff6-435c-8ea4-926206cd7047
# ╠═f882d5b1-7da0-458b-b079-33fbbcebd057
# ╠═9ca2fac3-7695-4b82-8f75-b7db5f35d84d
# ╟─8beb523a-102b-4474-8044-4e15586c674f
# ╠═c762bf37-305b-463e-b573-6befd69288ee
# ╠═76c07b1a-0994-47fa-89f9-cc060b51841a
# ╠═0dc35e06-d94f-4f88-9ebe-bbf6e0bf8366
# ╠═fdb8b36f-579b-4a9a-81aa-31bc366dce02
# ╠═c1ec07df-0104-4b0d-816c-03b999cc4609
# ╟─8c877be1-5861-45ac-9087-e9e8c2f7786d
# ╠═e63463de-ed39-4b2f-ad9e-2ba217db931d
# ╠═7cbcbd5c-4203-46c8-be25-14ca8f016eae
# ╠═f103ce35-74be-4c2c-aee7-691926530174
# ╠═b8d0c055-b7ef-4791-a9b6-5af7d07b0e8e
# ╠═1c3e9adb-ece1-4067-9ac2-2389de62d129
# ╠═2e50a12d-6aa5-4e91-bc46-06ba71bf82cd
# ╠═41db6cb5-5c5a-4556-b2eb-3cb6c3b40013
# ╠═7e3cc930-8015-43e7-ac18-dc8d1d274c4c
# ╠═427b3fd2-a6cb-4a1e-b286-b88596ba2ac5
# ╠═0f15c89a-4505-4683-9413-ab2881c11b2e
# ╠═b4d095bd-61b0-411f-bb08-910c2c5a7804
# ╠═77637b57-ac4d-4459-bbcb-e070378bc1ae
# ╠═d8fc6d1d-7cc5-42b9-b3c7-33d9666cba2d
# ╠═299b0366-a510-438b-bb41-c0ef679f2403
# ╠═13c9dcce-fa5b-4cc4-b684-c0b791a79492
# ╠═23da9f62-5c43-4e8f-b05c-da2799e86474
# ╠═91ba2048-42a9-49f9-8861-a545ff314b91
# ╠═dafecb7b-0c4f-4084-9d24-302b09d98fa6
# ╠═f7b68127-2de2-4baf-9636-106310af6394
