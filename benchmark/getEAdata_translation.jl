#!/usr/bin/env julia

"""
Translation of `getEAdata.R` to Julia. Generates `EA_SW_rawdata.csv`
and `EA_SW_data.csv` using Euro area data.
This script mirrors the steps of the original R code but avoids any
plotting.
"""

using CSV
using DataFrames
using Dates
using HTTP
using JSON
using XLSX
using Interpolations
using LinearAlgebra

""" parse quarter or date string to `Date`. """
function parse_quarter(x)
    s = String(x)
    if occursin(r"^\d{4}Q[1-4]$", s)
        y = parse(Int, s[1:4])
        q = parse(Int, s[end])
        return Date(y, 3(q-1)+1, 1)
    elseif occursin(r"^\d{4}-\d{2}-\d{2}$", s)
        return Date(s)
    else
        # year.quarter numeric
        yq = parse(Float64, s)
        y = floor(Int, yq)
        q = round(Int, (yq - y)*4 + 1)
        return Date(y, 3(q-1)+1, 1)
    end
end

""" Hodrick Prescott trend component. """
function hp_trend(y; λ=1600.0)
    n = length(y)
    D = zeros(n-2, n)
    for i in 1:n-2
        D[i,i] = 1
        D[i,i+1] = -2
        D[i,i+2] = 1
    end
    trend = (I + λ*(D'D)) \ y
    return trend
end

""" Chain link series so that `to_rebase` is rebased to `basis` at
`date_chain`. Both inputs must have columns `:period`, `:var`, `:value`.
"""
function chain(to_rebase::DataFrame, basis::DataFrame, date_chain::Date)
    ref = filter(row -> row.period == date_chain, basis)
    out = DataFrame(period=Date[], var=String[], value=Float64[])
    for g in groupby(filter(row->row.period <= date_chain, to_rebase), :var)
        growth = [1.0; g.value[2:end] ./ g.value[1:end-1]]
        valref = ref[(ref.var .== g.var[1]), :value][1]
        vals = cumprod(growth) .* valref
        append!(out, DataFrame(period=g.period, var=g.var, value=vals))
    end
    append!(out, filter(row->row.period > date_chain, basis))
    sort!(out, :period)
    return out
end

""" Retrieve Eurostat dataset as DataFrame """
function get_eurostat(dataset; filters=Dict(), time_format="date")
    base = "https://api.europa.eu/eurostat/api/dissemination/statistics/1.0/data/" * dataset
    qs = join(["$(k)=$(isa(v,AbstractVector) ? join(v,"+") : v)" for (k,v) in filters], "&")
    url = isempty(qs) ? base * "?time_format=$time_format" : base * "?$qs&time_format=$time_format"
    r = HTTP.get(url)
    r.status == 200 || error("Eurostat request failed: $(r.status)")
    js = JSON.parse(String(r.body))
    ids = js["id"]
    dims = js["dimension"]
    order = [collect(keys(dims[d]["category"]["index"])) for d in ids]
    combos = collect(Iterators.product(order...))
    df = DataFrame()
    for (i,d) in enumerate(ids)
        df[!, Symbol(d)] = [c[i] for c in combos]
    end
    v = js[haskey(js, "value") ? "value" : "values"]
    df.value = [x===nothing ? missing : Float64(x) for x in v]
    if "time" in ids
        df.time = time_format=="date" ? Date.(df.time) : parse_quarter.(df.time)
    end
    return df
end

""" Simple linear interpolation for missing values. """
function fill_linear(ts::Vector{Union{Missing,Float64}})
    idx = collect(eachindex(ts))
    known = .!ismissing.(ts)
    itp = interpolate((Float64.(idx[known]),), Float64.(ts[known]), Gridded(Linear()))
    ext = extrapolate(itp, Flat())
    return [ismissing(t) ? ext(i) : Float64(t) for (i,t) in enumerate(ts)]
end

""" Cubic spline interpolation for missing values. """
function fill_spline(ts::Vector{Union{Missing,Float64}})
    idx = collect(eachindex(ts))
    known = .!ismissing.(ts)
    itp = interpolate((Float64.(idx[known]),), Float64.(ts[known]), Gridded(Cubic(Line())))
    ext = extrapolate(itp, Flat())
    return [ismissing(t) ? ext(i) : Float64(t) for (i,t) in enumerate(ts)]
end

""" Read AWM CSV and reshape to long format. """
function import_awm(path)
    df = CSV.read(path, DataFrame)
    rename!(df, Dict(:YER=>:gdp, :YED=>:defgdp, :PCR=>:conso, :PCD=>:defconso,
                     :ITR=>:inves, :ITD=>:definves, :WIN=>:wage,
                     :STN=>:shortrate, :LNN=>:employ))
    df.period = parse_quarter.(df.V1)
    return stack(df, Not(:period); variable_name=:var, value_name=:value)
end

""" Load Conference Board hours worked. """
function import_hours(path)
    tbl = XLSX.readtable(path, "Total Hours Worked"; header=3)
    df = DataFrame(tbl)
    rename!(df, names(df)[1] => :country)
    EA = ["Austria","Belgium","Cyprus","Estonia","Finland","France",
          "Germany","Greece","Ireland","Italy","Latvia","Lithuania","Luxembourg",
          "Malta","Netherlands","Portugal","Slovak Republic","Slovenia","Spain"]
    df = filter(row -> row.country in EA, df)
    cols = names(df)[2:end]
    rows = NamedTuple[]
    for r in eachrow(df), c in cols
        v = r[c]
        !ismissing(v) || continue
        push!(rows, (country=r.country, period=Date(parse(Int,c),7,1), value=float(v)))
    end
    DataFrame(rows)
end

function main()
    base = joinpath(@__DIR__)
    awm = import_awm(joinpath(base, "awm19up15.csv"))
    hours_raw = import_hours(joinpath(base, "TED---Output-Labor-and-Labor-Productivity-1950-2015.xlsx"))

    EA14 = unique(filter(r->r.period==Date(1970,7,1), hours_raw).country)
    hours14 = combine(groupby(filter(r->r.country in EA14, hours_raw), :period), :value=>sum=>:value); hours14.var .= "hours"
    hoursTot = combine(groupby(hours_raw, :period), :value=>sum=>:value); hoursTot.var .= "hours"
    chained_hours = chain(hours14, hoursTot, Date(1990,7,1))

    qrange = Date(1970,7,1):Month(3):Date(2012,7,1)
    hours = leftjoin(DataFrame(period=collect(qrange)), chained_hours, on=:period)
    approx = fill_linear(hours.value)
    spline = fill_spline(hours.value)
    trend = hp_trend(coalesce.(hours.value, mean(skipmissing(hours.value))))
    hours_kal = DataFrame(period=hours.period, var="hours", value=trend)

    # Population: annual by country
    p0 = get_eurostat("demo_pjanbroad"; filters=Dict(
        "geo"=>["AT","BE","CY","DE_TOT","EE","IE","EL","ES","FX","IT","LT","LV","LU","NL","PT","SK","FI","MT","SI"],
        "freq"=>"A","unit"=>"NR","sex"=>"T","age"=>"Y15-64"))
    pop_df = DataFrame(country=p0.geo, period=p0.time, value=p0.value)
    pop_df = filter(r->r.period>=Date(1970,1,1)&&r.period<=Date(2013,1,1)&&!ismissing(r.value), pop_df)
    EA16 = unique(filter(r->r.period==Date(1970,1,1), pop_df).country)
    pop16 = combine(groupby(filter(r->r.country in EA16, pop_df), :period), :value=>sum=>:value); pop16.var .= "pop"
    popTot = combine(groupby(pop_df, :period), :value=>sum=>:value); popTot.var .= "pop"
    chained_pop = chain(pop16, popTot, Date(1982,1,1))
    prange = Date(1970,1,1):Month(3):Date(2013,1,1)
    pop = leftjoin(DataFrame(period=collect(prange)), chained_pop, on=:period)
    pop_trend = hp_trend(coalesce.(pop.value, mean(skipmissing(pop.value))))
    pop_final = DataFrame(period=pop.period, var="pop", value=pop_trend)

    # Eurostat macro series
    d_gdp = get_eurostat("namq_10_gdp"; filters=Dict(
        "freq"=>"Q","unit"=>["CLV10_MEUR","PD10_EUR"],
        "s_adj"=>"SCA","na_item"=>["B1GQ","P31_S14_S15","P51G"],
        "geo"=>"EA19"))
    df1 = DataFrame(period=d_gdp.time, unit=d_gdp.unit, item=d_gdp.na_item, value=d_gdp.value)
    df1.var = [it=="B1GQ"&&u=="CLV10_MEUR" ? "gdp" : it=="B1GQ" ? "defgdp" : it=="P31_S14_S15"&&u=="CLV10_MEUR" ? "conso" : it=="P31_S14_S15" ? "defconso" : u=="CLV10_MEUR" ? "inves" : "definves" for (it,u) in zip(df1.item, df1.unit)]
    select!(df1, [:period,:var,:value])

    d_wage = get_eurostat("namq_10_a10"; filters=Dict("freq"=>"Q","unit"=>"CP_MEUR","s_adj"=>"SCA","nace_r2"=>"TOTAL","na_item"=>"D1","geo"=>"EA19"))
    df2 = DataFrame(period=d_wage.time, var="wage", value=d_wage.value)

    d_hours_emp = get_eurostat("namq_10_a10_e"; filters=Dict("freq"=>"Q","unit"=>["THS_HW","THS_PER"],"s_adj"=>"SCA","nace_r2"=>"TOTAL","na_item"=>"EMP_DC","geo"=>"EA19"))
    df3 = DataFrame(period=d_hours_emp.time, unit=d_hours_emp.unit, value=d_hours_emp.value)
    df3.var = df3.unit .== "THS_HW" ? "hours" : "employ"
    select!(df3, [:period,:var,:value])

    d_rate = get_eurostat("irt_st_q"; filters=Dict("freq"=>"Q","int_rt"=>"IRT_M3","geo"=>"EA"))
    df4 = DataFrame(period=d_rate.time, var="shortrate", value=d_rate.value)

    d_pop_q = get_eurostat("lfsq_pganws"; filters=Dict("freq"=>"Q","unit"=>"THS_PER","sex"=>"T","citizen"=>"TOTAL","age"=>"Y15-64","wstatus"=>"POP","geo"=>"EA20"))
    df5 = DataFrame(period=d_pop_q.time, var="pop", value=d_pop_q.value)

    recent = vcat(df1,df2,df3,df4,df5)
    cutoff = minimum(combine(groupby(recent,:var), :period=>maximum).period_maximum)
    recent = filter(r->r.period<=cutoff, recent)

    vars = ["gdp","conso","inves","defgdp","defconso","definves","shortrate","hours","wage","employ"]
    new_df = filter(r->r.var in vars, recent)
    old_df = vcat(filter(r->r.var in vars, awm), hours_kal)
    df_chain = chain(old_df, new_df, Date(1999,1,1))

    pop_chain = chain(pop_final, filter(r->r.var=="pop", recent), minimum(filter(r->r.var=="pop", recent).period))
    final_df = vcat(df_chain, pop_chain)

    wide = unstack(final_df, :period, :var, :value)
    CSV.write(joinpath(base, "EA_SW_rawdata.csv"), wide)

    ed = wide
    EA_SW_data = DataFrame(period=[string(year(d),"Q",((month(d)-1)÷3)+1) for d in ed.period],
        gdp_rpc    = 1e6*ed.gdp    ./ (ed.pop*1000),
        conso_rpc  = 1e6*ed.conso  ./ (ed.pop*1000),
        inves_rpc  = 1e6*ed.inves  ./ (ed.pop*1000),
        defgdp     = ed.defgdp,
        wage_rph   = 1e6*ed.wage   ./ ed.defgdp ./ (ed.hours*1000),
        hours_pc   = 1000*ed.hours ./ (ed.pop*1000),
        pinves_defl= ed.definves ./ ed.defgdp,
        pconso_defl= ed.defconso ./ ed.defgdp,
        shortrate  = ed.shortrate ./ 100,
        employ     = 1000*ed.employ ./ (ed.pop*1000))
    CSV.write(joinpath(base, "EA_SW_data.csv"), dropmissing(EA_SW_data))

    println("Finished writing EA_SW_rawdata.csv and EA_SW_data.csv")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
