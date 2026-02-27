using MacroModelling
using Zygote
using FiniteDifferences
using LinearAlgebra
using SparseArrays
using ChainRulesCore

const FDM_ORDER = 4
const FDM_MAX_RANGE = 1e-4

include(joinpath(@__DIR__, "..", "test", "models", "Caldara_et_al_2012_estim.jl"))

const model = Caldara_et_al_2012_estim
const p0 = copy(model.parameter_values)
const opts = MacroModelling.merge_calculation_options(verbose = false)

function build_third_order_inputs(p)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)

    SS_and_pars, (solution_error, _) = MacroModelling.get_NSSS_and_parameters(model, p, opts = opts, estimation = true)
    abs(solution_error) < opts.tol.NSSS_acceptance_tol || error("get_NSSS_and_parameters did not converge")

    ∇₁ = Matrix(MacroModelling.calculate_jacobian(p, SS_and_pars, model.caches, model.functions.jacobian))
    𝐒₁, _, solved1 = MacroModelling.calculate_first_order_solution(
        ∇₁, model.constants, model.workspaces, model.caches;
        initial_guess = model.caches.qme_solution, opts = opts,
    )
    solved1 || error("1st order failed")

    ∇₂ = MacroModelling.calculate_hessian(p, SS_and_pars, model.caches, model.functions.hessian)
    𝐒₂, solved2 = MacroModelling.calculate_second_order_solution(
        ∇₁, ∇₂, 𝐒₁, model.constants, model.workspaces, model.caches;
        initial_guess = model.caches.second_order_solution, opts = opts,
    )
    solved2 || error("2nd order failed")

    𝐒₂ *= model.constants.second_order.𝐔₂
    𝐒₂ = sparse(𝐒₂)

    ∇₃ = MacroModelling.calculate_third_order_derivatives(p, SS_and_pars, model.caches, model.functions.third_order_derivatives)

    return (Matrix(∇₁), sparse(∇₂), sparse(∇₃), Matrix(𝐒₁), 𝐒₂)
end

function third_order_norm(∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂)
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)
    𝐒₃, solved3 = MacroModelling.calculate_third_order_solution(
        ∇₁, ∇₂, ∇₃, 𝐒₁, 𝐒₂,
        model.constants, model.workspaces, model.caches;
        initial_guess = model.caches.third_order_solution, opts = opts,
    )
    solved3 || error("3rd order failed")
    return norm(𝐒₃)
end

# warm up
MacroModelling.solve!(model, algorithm = :third_order, opts = opts)

(∇₁0, ∇₂0, ∇₃0, 𝐒₁0, 𝐒₂0) = build_third_order_inputs(p0)

fdm = FiniteDifferences.central_fdm(FDM_ORDER, 1, max_range = FDM_MAX_RANGE)

# ---- s1 gradient ----
println("\n=== s1 gradient ===")
dims1 = size(𝐒₁0)
x1 = vec(copy(𝐒₁0))

obj_s1 = x -> begin
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)
    𝐒₁x = reshape(x, dims1)
    third_order_norm(∇₁0, ∇₂0, ∇₃0, 𝐒₁x, 𝐒₂0)
end

g_fd_s1 = FiniteDifferences.grad(fdm, obj_s1, x1)[1]
g_zyg_s1 = Zygote.gradient(obj_s1, x1)[1]

Δ_s1 = g_zyg_s1 .- g_fd_s1
println("max abs diff: ", maximum(abs, Δ_s1))
println("rel norm: ", norm(Δ_s1)/max(norm(g_fd_s1), eps()))

# ---- s2 gradient ----
println("\n=== s2 gradient ===")
x2 = copy(𝐒₂0.nzval)

function sparse_with_new_values(template::SparseMatrixCSC{<:Real,Int}, values::AbstractVector{T}) where {T<:Real}
    SparseMatrixCSC(size(template,1), size(template,2), copy(template.colptr), copy(template.rowval), collect(values))
end
ChainRulesCore.rrule(::typeof(sparse_with_new_values), template::SparseMatrixCSC{<:Real,Int}, values::AbstractVector{T}) where {T<:Real} = begin
    y = sparse_with_new_values(template, values)
    pb(ȳ) = begin
        dv = if ȳ isa AbstractMatrix
            out = similar(template.nzval, promote_type(eltype(template.nzval), eltype(ȳ)))
            for col in 1:size(template,2)
                for k in template.colptr[col]:(template.colptr[col+1]-1)
                    out[k] = ȳ[template.rowval[k], col]
                end
            end
            out
        else
            zero(values)
        end
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.ProjectTo(values)(dv)
    end
    y, pb
end

obj_s2 = x -> begin
    MacroModelling.@ignore_derivatives MacroModelling.clear_solution_caches!(model, :third_order)
    𝐒₂x = sparse_with_new_values(𝐒₂0, x)
    third_order_norm(∇₁0, ∇₂0, ∇₃0, 𝐒₁0, 𝐒₂x)
end

g_fd_s2 = FiniteDifferences.grad(fdm, obj_s2, x2)[1]
g_zyg_s2 = Zygote.gradient(obj_s2, x2)[1]

Δ_s2 = g_zyg_s2 .- g_fd_s2
println("max abs diff: ", maximum(abs, Δ_s2))
println("rel norm: ", norm(Δ_s2)/max(norm(g_fd_s2), eps()))

println("DONE_DEBUG")
