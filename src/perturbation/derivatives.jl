@stable default_mode = "disable" begin


function calculate_jacobian(parameters::Vector{M},
                            SS_and_pars::Vector{N},
                            caches_obj::caches,
                            jacobian_funcs::jacobian_functions,
                            workspaces::workspaces;
                            caching::Bool = true)::Matrix{M} where {M,N}
    # Cache hit: return cached jacobian if valid for current parameters
    if caching && M === Float64 && cache_valid_for_parameters(caches_obj.valid_for.jacobian, parameters) && caches_obj.jacobian isa Matrix{M} && !isempty(caches_obj.jacobian)
        return caches_obj.jacobian
    end

    if eltype(caches_obj.jacobian) != M
        if caches_obj.jacobian isa SparseMatrixCSC
            jac_buffer = similar(caches_obj.jacobian,M)
            jac_buffer.nzval .= 0
        else
            jac_buffer = zeros(M, size(caches_obj.jacobian))
        end
    else
        jac_buffer = caches_obj.jacobian
    end
    
    jacobian_funcs.f(jac_buffer, parameters, SS_and_pars)

    if caching && M === Float64
        caches_obj.jacobian = jac_buffer
        caches_obj.valid_for.jacobian = Float64.(parameters)
    end
    
    return jac_buffer
end

function calculate_hessian(parameters::Vector{M}, 
                            SS_and_pars::Vector{N}, 
                            caches_obj::caches,
                            hessian_funcs::hessian_functions,
                            workspaces::workspaces;
                            caching::Bool = true)::SparseMatrixCSC{M, Int} where {M,N}
    # Always make sure the higher-order workspace matches the eltype expected by
    # downstream consumers (e.g. rrules that grab buffers from it). A previous
    # call with a different eltype (e.g. ForwardDiff.Dual) may have replaced the
    # workspace; the cache short-circuit below would otherwise leave it stale.
    S = promote_type(M, N)
    if eltype(workspaces.second_order.Ŝ) != S
        workspaces.second_order = Higher_order_workspace(T = S)
    end

    # Cache hit: return cached hessian if valid for current parameters
    if caching && M === Float64 && cache_valid_for_parameters(caches_obj.valid_for.hessian, parameters) && caches_obj.hessian isa SparseMatrixCSC{M, Int} && !isempty(caches_obj.hessian)
        return caches_obj.hessian
    end

    if eltype(caches_obj.hessian) != M
        if caches_obj.hessian isa SparseMatrixCSC
            hes_buffer = similar(caches_obj.hessian,M)
            hes_buffer.nzval .= 0
        else
            hes_buffer = zeros(M, size(caches_obj.hessian))
        end
    else
        hes_buffer = caches_obj.hessian
    end

    hessian_funcs.f(hes_buffer, parameters, SS_and_pars)

    if caching && M === Float64
        caches_obj.hessian = hes_buffer
        caches_obj.valid_for.hessian = Float64.(parameters)
    end
    
    return hes_buffer
end


function calculate_third_order_derivatives(parameters::Vector{M}, 
                                            SS_and_pars::Vector{N}, 
                                            caches_obj::caches,
                                            third_order_derivatives_funcs::third_order_derivatives_functions,
                                            workspaces::workspaces;
                                            caching::Bool = true)::SparseMatrixCSC{M, Int} where {M,N}
    # Always make sure the third-order workspace matches the eltype expected by
    # downstream consumers (e.g. rrules that grab buffers from it). A previous
    # call with a different eltype (e.g. ForwardDiff.Dual) may have replaced the
    # workspace; the cache short-circuit below would otherwise leave it stale.
    S = promote_type(M, N)
    if eltype(workspaces.third_order.Ŝ) != S
        workspaces.third_order = Higher_order_workspace(T = S)
    end

    # Cache hit: return cached third order derivatives if valid for current parameters
    if caching && M === Float64 && cache_valid_for_parameters(caches_obj.valid_for.third_order_derivatives, parameters) && caches_obj.third_order_derivatives isa SparseMatrixCSC{M, Int} && !isempty(caches_obj.third_order_derivatives)
        return caches_obj.third_order_derivatives
    end

    if eltype(caches_obj.third_order_derivatives) != M
        if caches_obj.third_order_derivatives isa SparseMatrixCSC
            third_buffer = similar(caches_obj.third_order_derivatives,M)
            third_buffer.nzval .= 0
        else
            third_buffer = zeros(M, size(caches_obj.third_order_derivatives))
        end
    else
        third_buffer = caches_obj.third_order_derivatives
    end

    third_order_derivatives_funcs.f(third_buffer, parameters, SS_and_pars)

    if caching && M === Float64
        caches_obj.third_order_derivatives = third_buffer
        caches_obj.valid_for.third_order_derivatives = Float64.(parameters)
    end
    
    return third_buffer
end


end # @stable
