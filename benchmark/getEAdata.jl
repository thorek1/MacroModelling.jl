#!/usr/bin/env julia

"""
getEAdata.jl

Julia translation of getEAdata.R pipeline:
- Import AWM database CSV.
- Import Conference Board annual hours worked.
- Chain-link aggregate hours.
- Interpolate and smooth hours via HP filter & optional Kalman smoothing.
- Retrieve time series from Eurostat JSON-stat API.
- Chain-link and interpolate/smooth population.
- Import and process macro variables (GDP, consumption, investment, wages, interest, employment, population).
- Combine old and new series, chain-link old data to new series.
- Export EA_SW_rawdata.csv and EA_SW_data.csv.
- Plot time series facets via Plots.jl.
"""

using DataFrames
using CSV
using Dates
using HTTP
using JSON
using XLSX
using Interpolations: interpolate, extrapolate, Gridded, Linear, Cubic, Line, Flat
using LinearAlgebra
using Plots

# Optional Kalman smoothing via StateSpaceModels.jl
const HAVE_SSM = try
    @eval using StateSpaceModels
    true
catch
    @warn "StateSpaceModels.jl not available; falling back to HP filter for trend"
    false
end

# ---------------------- Utility Functions ----------------------

# Parse quarter strings or ISO dates to Date
function parse_quarter(q)
    s = string(q)
    # pattern YYYY and quarter digit
    m = match(r"^\s*(\d{4})\D*([1-4])\s*$", s)
    if m !== nothing
        yr = parse(Int, m.captures[1])
        qn = parse(Int, m.captures[2])
        return Date(yr, 3*(qn-1) + 1, 1)
    end
    # ISO date YYYY-MM-DD
    m2 = match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m2 !== nothing
        return Date(parse(Int,m2.captures[1]), parse(Int,m2.captures[2]), parse(Int,m2.captures[3]))
    end
    # numeric year.quarter encoding
    try
        yq = parse(Float64, s)
        yr = floor(Int, yq)
        frac = yq - yr
        qn = round(Int, frac*4 + 1e-8) + 1
        return Date(yr, 3*(qn-1) + 1, 1)
    catch
        error("Unable to parse quarter or date: $s")
    end
end

# Hodrick–Prescott filter for trend extraction
function hpfilter_trend(y::AbstractVector{<:Real}; λ::Real=1600)
    n = length(y)
    D = zeros(n-2, n)
    for i in 1:n-2
        D[i,i]   = 1
        D[i,i+1] = -2
        D[i,i+2] = 1
    end
    M = I + λ*(D'*D)
    yv = collect(Float64, y)
    return M 
        yv
end

# Chain-linking to rebase old series and append new basis
function chain(to_rebase::DataFrame, basis::DataFrame, date_chain::Date)
    cref = filter(r->r.period == date_chain, basis)
    @assert nrow(cref)>0 "chain: no basis at $date_chain"
    tr = filter(r->r.period <= date_chain, to_rebase)
    sort!(tr, :period, rev=true)
    out = DataFrame(period=Date[], var=String[], value=Float64[])
    for g in groupby(tr, :var)
        vals = Float64.(g.value)
        rates = [1.0; vals[2:end] ./ vals[1:end-1]]
        crefval = filter(r->r.var==g.var[1], cref).value[1]
        append!(out, DataFrame(period=g.period, var=g.var, value=cumprod(rates)*crefval))
    end
    aft = filter(r->r.period > date_chain, basis)
    append!(out, aft)
    sort!(out, :period)
    return out
end

# Fetch and flatten JSON-stat dataset from Eurostat API
function get_eurostat(dataset::String; filters=Dict{String,Any}(), time_format::String="date")
    base = "https://api.europa.eu/eurostat/api/dissemination/statistics/1.0/data/" * dataset
    qs = isempty(filters) ? "" :
         join(["$(k)=$(isa(v,AbstractVector) ? join(v,"+") : string(v))" for (k,v) in filters], "&")
    url = isempty(qs) ? (base * "?time_format=$time_format") : (base * "?$qs&time_format=$time_format")
    res = HTTP.get(url)
    @assert res.status==200 "Eurostat API error: $(res.status)"
    js = JSON.parse(String(res.body))
    id_order = js["id"]
    dims = js["dimension"]
    mapping = Dict{String,Vector{String}}()
    for d in id_order
        idx = dims[d]["category"]["index"]
        if isa(idx,Dict)
            pos = Dict(string(k)=>Int(v) for (k,v) in idx)
            mapping[d] = sort(collect(keys(pos)), by=x->pos[x])
        else
            mapping[d] = String.(idx)
        end
    end
    combos = collect(Iterators.product((mapping[d] for d in id_order)...))
    df = DataFrame()
    for (i,d) in enumerate(id_order)
        df[!, Symbol(d)] = [c[i] for c in combos]
    end
    vkey = haskey(js,"value") ? "value" : "values"
    raw = js[vkey]
    df.value = [x===nothing ? missing : Float64(x) for x in raw]
    if "time" in id_order
        if time_format=="date"
            df.time = Date.(df.time)
        else
            df.time = parse_quarter.(df.time)
        end
    end
    return df
end

# Linear interpolation for missing values
function fill_linear(periods::Vector{Date}, vals::Vector{Union{Missing,Float64}})
    idx = Float64.(1:length(vals)); mask = .!ismissing.(vals)
    xs = idx[mask]; ys = Float64.(vals[mask])
    it = interpolate((xs,), ys, Gridded(Linear())); eit = extrapolate(it, Flat())
    return [ismissing(vals[i]) ? eit(idx[i]) : Float64(vals[i]) for i in 1:length(vals)]
end

# Cubic spline interpolation for missing values
function fill_spline(periods::Vector{Date}, vals::Vector{Union{Missing,Float64}})
    idx = Float64.(1:length(vals)); mask = .!ismissing.(vals)
    xs = idx[mask]; ys = Float64.(vals[mask])
    it = interpolate((xs,), ys, Gridded(Cubic(Line()))); eit = extrapolate(it, Flat())
    return [ismissing(vals[i]) ? eit(idx[i]) : Float64(vals[i]) for i in 1:length(vals)]
end

# Optional Kalman smoothing via StateSpaceModels.jl
function kalman_smooth(vals::Vector{Float64})
    @assert HAVE_SSM "StateSpaceModels.jl not available"
    m = UnobservedComponents(vals, level=:local_level)
    res = fit(m)
    return Float64.(res.smoothed_state[1,:])
end

# ---------------------- Import Helpers ----------------------

function import_awm(path::AbstractString)
    df = CSV.read(path, DataFrame)
    rename!(df, Dict(:YER=>:gdp,:YED=>:defgdp,:PCR=>:conso,:PCD=>:defconso,
                    :ITR=>:inves,:ITD=>:definves,:WIN=>:wage,:STN=>:shortrate,
                    :LNN=>:employ))
    df.period = parse_quarter.(String.(df.V1))
    return stack(df, Not(:period); variable_name=:var, value_name=:value)
end

function import_confboard_hours(path::AbstractString)
    tbl = XLSX.readtable(path, "Total Hours Worked"; infer_eltypes=true, header=3)
    df = DataFrame(tbl)
    rename!(df, names(df)[1]=>:country)
    EAtot = ["Austria","Belgium","Cyprus","Estonia","Finland","France",
             "Germany","Greece","Ireland","Italy","Latvia","Lithuania","Luxembourg",
             "Malta","Netherlands","Portugal","Slovak Republic","Slovenia","Spain"]
    df = filter(r->r.country in EAtot, df)
    yrs = names(df)[2:end]
    rows = NamedTuple[]
    for r in eachrow(df), y in yrs
        v = r[y]
        if !ismissing(v)
            yr = parse(Int, String(y))
            push!(rows, (country=r.country, period=Date(yr,7,1), value=float(v)))
        end
    end
    df2 = DataFrame(rows)
    return filter(r->r.period>=Date(1970,7,1)&&r.period<=Date(2012,7,1), df2)
end

# ---------------------- Main Pipeline ----------------------

function main()
    base = joinpath(homedir(),"Github","MacroModelling.jl","benchmark")
    awm_file = joinpath(base,"awm19up15.csv")
    ted_file = joinpath(base,"TED---Output-Labor-and-Labor-Productivity-1950-2015.xlsx")

    # -- AWM data
    awm = import_awm(awm_file)

    # -- Conference Board annual hours
    hrs_cb = import_confboard_hours(ted_file)
    EA14 = unique(filter(r->r.period==Date(1970,7,1), hrs_cb).country)
    hrs14 = combine(groupby(filter(r->r.country in EA14, hrs_cb), :period),
                    :value=>sum=>:value); hrs14.var .= "hours"
    hrsTot = combine(groupby(hrs_cb, :period), :value=>sum=>:value); hrsTot.var .= "hours"

    hours_ch = chain(hrs14, hrsTot, Date(1990,7,1))
    quarters = Date(1970,7,1):Month(3):Date(2012,7,1)
    hours = leftjoin(DataFrame(period=collect(quarters)), hours_ch, on=:period)

    # Interpolate & smooth hours
    hrs_approx = deepcopy(hours)
    hrs_approx.value = fill_linear(hrs_approx.period, hrs_approx.value); hrs_approx.var .= "hours_approx"
    hrs_spline = deepcopy(hours)
    hrs_spline.value = fill_spline(hrs_spline.period, hrs_spline.value); hrs_spline.var .= "hours_spline"
    trend_h = HAVE_SSM ? kalman_smooth(Float64.(hours.value)) : hpfilter_trend(hours.value)
    hrs_kalman = DataFrame(period=hours.period, var=fill("hours_kalman", length(hours.period)), value=trend_h)
    hours_filtered = vcat(hrs_approx, hrs_spline, hrs_kalman)

    # Levels & growth rates for hours
    hrs_levels = DataFrame(period=hours_filtered.period,
                           var=hours_filtered.var,
                           ind2=fill("1- Levels", size(hours_filtered,1)),
                           value=hours_filtered.value)
    grp_h = combine(groupby(hours_filtered, :var)) do sdf
        v = log.(sdf.value[2:end] ./ sdf.value[1:end-1])
        DataFrame(period=sdf.period[2:end], var=sdf.var[2:end],
                  ind2=fill("2- Growth rates", length(v)), value=v)
    end
    hours_filtered_levgr = filter(r->r.period>=Date(1971,1,1), vcat(hrs_levels, grp_h))
    hours = DataFrame(period=hrs_kalman.period, var=fill("hours", length(hrs_kalman.period)), value=trend_h)

    # -- Eurostat quarterly hours (original)
    d0 = get_eurostat("namq_10_a10_e", filters=Dict("geo"=>"EA19","freq"=>"Q","unit"=>"THS_HW",
                                                   "nace_r2"=>"TOTAL","s_adj"=>"SCA","na_item"=>"EMP_DC"))
    euro_hours = DataFrame(period=d0.time, value=d0.value); euro_hours.var .= "Quarterly hours (original, Eurostat)"
    ref_h = mean(filter(r->year(r.period)==2000, euro_hours).value)
    euro_hours_ind = deepcopy(euro_hours); euro_hours_ind.value ./= ref_h

    # interpolated hours index
    ref_hi = mean(filter(r->year(r.period)==2000, hours).value)
    hours_ind = DataFrame(period=hours.period,
                          var=fill("Quarterly hours (interpolated)", length(hours.period)),
                          value=hours.value ./ ref_hi)

    # -- Eurostat annual population by country
    p0 = get_eurostat("demo_pjanbroad", filters=Dict("geo"=>["AT","BE","CY","DE_TOT","EE","IE","EL","ES",
                                                       "FX","IT","LT","LV","LU","NL","PT","SK","FI","MT","SI"],
                                                    "freq"=>"A","unit"=>"NR","sex"=>"T","age"=>"Y15-64"))
    pop0 = DataFrame(country=p0.geo, period=p0.time, value=p0.value)
    pop0 = filter(r->r.period>=Date(1970,1,1)&&r.period<=Date(2013,1,1)&&!ismissing(r.value), pop0)
    EA16 = unique(filter(r->r.period==Date(1970,1,1), pop0).country)
    pop16 = combine(groupby(filter(r->r.country in EA16, pop0), :period), :value=>sum=>:value); pop16.var .= "pop"
    popTot = combine(groupby(pop0,:period), :value=>sum=>:value); popTot.var .= "pop"
    pop_ch = chain(pop16, popTot, Date(1982,1,1))
    qr = Date(1970,1,1):Month(3):Date(2013,1,1)
    pop = leftjoin(DataFrame(period=collect(qr)), pop_ch, on=:period)

    # Interpolate & smooth population
    pop_approx = deepcopy(pop); pop_approx.value = fill_linear(pop.period, pop.value); pop_approx.var .= "pop_approx"
    pop_spline = deepcopy(pop); pop_spline.value = fill_spline(pop.period, pop.value); pop_spline.var .= "pop_spline"
    trend_p = HAVE_SSM ? kalman_smooth(Float64.(pop.value)) : hpfilter_trend(pop.value)
    pop_kalman = DataFrame(period=pop.period, var=fill("pop_kalman", length(pop.period)), value=trend_p)
    pop_filtered = vcat(pop_approx, pop_spline, pop_kalman)
    pop_levels = DataFrame(period=pop_filtered.period, var=pop_filtered.var,
                           ind2=fill("1- Levels", size(pop_filtered,1)), value=pop_filtered.value)
    grp_p = combine(groupby(pop_filtered, :var)) do sdf
        v = log.(sdf.value[2:end] ./ sdf.value[1:end-1])
        DataFrame(period=sdf.period[2:end], var=sdf.var[2:end],
                  ind2=fill("2- Growth rates", length(v)), value=v)
    end
    pop_filtered_levgr = filter(r->r.period>=Date(1970,4,1), vcat(pop_levels, grp_p))
    pop = DataFrame(period=pop_kalman.period, var=fill("pop", length(pop_kalman.period)), value=trend_p)

    # -- Combine old data
    old_df = vcat(awm, hours, pop)

    # -- New macro series: GDP/consumption/investment/deflators
    df1 = get_eurostat("namq_10_gdp",
                       filters=Dict("freq"=>"Q","unit"=>["CLV10_MEUR","PD10_EUR"],
                                    "s_adj"=>"SCA","na_item"=>["B1GQ","P31_S14_S15","P51G"],
                                    "geo"=>"EA19"))
    d1 = DataFrame(period=df1.time, unit=df1.unit, item=df1.na_item, value=df1.value)
    d1.var = [it=="B1GQ" && un=="CLV10_MEUR" ? "gdp" :
              it=="B1GQ" ? "defgdp" :
              it=="P31_S14_S15" && un=="CLV10_MEUR" ? "conso" :
              it=="P31_S14_S15" ? "defconso" :
              it=="P51G" && un=="CLV10_MEUR" ? "inves" : "definves"
              for (it,un) in zip(d1.item, d1.unit)]
    select!(d1, [:period, :var, :value])

    # -- Wage
    df2 = get_eurostat("namq_10_a10", filters=Dict("freq"=>"Q","unit"=>"CP_MEUR",
                                                   "s_adj"=>"SCA","nace_r2"=>"TOTAL",
                                                   "na_item"=>"D1","geo"=>"EA19"))
    d2 = DataFrame(period=df2.time, var=fill("wage", length(df2.time)), value=df2.value)

    # -- Hours & employment
    df3 = get_eurostat("namq_10_a10_e", filters=Dict("freq"=>"Q","unit"=>["THS_HW","THS_PER"],
                                                       "s_adj"=>"SCA","nace_r2"=>"TOTAL",
                                                       "na_item"=>"EMP_DC","geo"=>"EA19"))
    d3 = DataFrame(period=df3.time, unit=df3.unit, value=df3.value)
    d3.var = [u=="THS_HW" ? "hours" : "employ" for u in d3.unit]
    select!(d3, [:period, :var, :value])

    # -- Short-term interest rate
    df4 = get_eurostat("irt_st_q", filters=Dict("freq"=>"Q","int_rt"=>"IRT_M3","geo"=>"EA"))
    d4 = DataFrame(period=df4.time, var=fill("shortrate", length(df4.time)), value=df4.value)

    # -- Quarterly pop
    df5 = get_eurostat("lfsq_pganws", filters=Dict("freq"=>"Q","unit"=>"THS_PER",
                                                       "sex"=>"T","citizen"=>"TOTAL",
                                                       "age"=>"Y15-64","wstatus"=>"POP",
                                                       "geo"=>"EA20"))
    d5 = DataFrame(period=df5.time, var=fill("pop", length(df5.time)), value=df5.value)

    recent = vcat(d1,d2,d3,d4,d5)
    md = combine(groupby(recent, :var), :period=>maximum=>:maxdate)
    cutoff = minimum(md.maxdate)
    recent = filter(r->r.period <= cutoff, recent)

    core = ["gdp","conso","inves","defgdp","defconso","definves",
            "shortrate","hours","wage","employ"]
    new_df = filter(r->r.var in core, recent)
    old_core = vcat(filter(r->r.var in core, awm), hours)
    df_chain = chain(old_core, new_df, Date(1999,1,1))

    pop_q = filter(r->r.var=="pop", recent)
    pop_cut = minimum(pop_q.period)
    pop_chain = chain(pop, pop_q, pop_cut)

    final_df = vcat(df_chain, pop_chain)

    raw = unstack(final_df, :period, :var, :value)
    CSV.write("EA_SW_rawdata.csv", raw)

    ed = raw
    ps = [string(year(d),"Q",((month(d)-1)÷3)+1) for d in ed.period]
    EA_SW_data = DataFrame(period=ps,
        gdp_rpc    = 1e6*ed.gdp    ./(ed.pop*1000),
        conso_rpc  = 1e6*ed.conso  ./(ed.pop*1000),
        inves_rpc  = 1e6*ed.inves  ./(ed.pop*1000),
        defgdp     = ed.defgdp,
        wage_rph   = 1e6*ed.wage   ./ed.defgdp ./(ed.hours*1000),
        hours_pc   = 1000*ed.hours ./(ed.pop*1000),
        pinves_defl= ed.definves   ./ed.defgdp,
        pconso_defl= ed.defconso   ./ed.defgdp,
        shortrate  = ed.shortrate ./100,
        employ     = 1000*ed.employ ./(ed.pop*1000))
    CSV.write("EA_SW_data.csv", dropmissing(EA_SW_data))

    println("Done. EA_SW_rawdata.csv and EA_SW_data.csv created.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end