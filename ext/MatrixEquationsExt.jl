module MatrixEquationsExt
# good overview: https://cscproxy.mpi-magdeburg.mpg.de/mpcsc/benner/talks/Benner-Melbourne2019.pdf
import MacroModelling
import MacroModelling:
    sylvester_workspace, lyapunov_workspace, SolverTolerances,
    solve_sylvester_equation, solve_lyapunov_equation,
    ensure_sylvester_krylov_buffers!, ensure_lyapunov_doubling_buffers!,
    _BARTELS_STEWART_AVAILABLE

import MatrixEquations
import LinearAlgebra as ℒ

function __init__()
    _BARTELS_STEWART_AVAILABLE[] = true
end

# ── Sylvester: Bartels-Stewart via MatrixEquations.sylvd ──

function MacroModelling.solve_sylvester_equation(A::DenseMatrix{T},
                                    B::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::DenseMatrix{T},
                                    ::Val{:bartels_stewart},
                                    𝕊ℂ::sylvester_workspace;
                                    initial_guess::AbstractMatrix{<:AbstractFloat} = zeros(0,0),
                                    preconditioner::Symbol = :none,
                                    verbose::Bool = false,
                                    tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat

    if length(initial_guess) == 0
        initial_guess = zero(C)
    end
    
    n = size(A, 1)
    m = size(B, 2)
    ensure_sylvester_krylov_buffers!(𝕊ℂ, n, m)
    
    𝐂¹ = 𝕊ℂ.𝐂
    tmp̄ = 𝕊ℂ.tmp
      
    # 𝐂¹  = A * initial_guess * B + C - initial_guess
    ℒ.mul!(tmp̄, initial_guess, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, initial_guess, 𝐂¹)

    𝐂 = try 
        MatrixEquations.sylvd(-A, B, 𝐂¹)::Matrix{T}
    catch
        return C, 0, 1.0
    end

    𝐂 += initial_guess

    ℒ.mul!(tmp̄, 𝐂, B)
    ℒ.mul!(𝐂¹, A, tmp̄)
    ℒ.axpy!(1, C, 𝐂¹)
    ℒ.axpy!(-1, 𝐂, 𝐂¹)
    
    reached_tol = ℒ.norm(𝐂¹) / max(ℒ.norm(𝐂), ℒ.norm(C))

    return 𝐂, -1, reached_tol
end

# ── Lyapunov: Bartels-Stewart via MatrixEquations.lyapd ──

function MacroModelling.solve_lyapunov_equation(A::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    C::Union{ℒ.Adjoint{T, Matrix{T}}, DenseMatrix{T}},
                                    ::Val{:bartels_stewart},
                                    workspace::lyapunov_workspace;
                                    tol::SolverTolerances = SolverTolerances())::Tuple{Matrix{T}, Int, T} where T <: AbstractFloat

    𝐂 = try 
        MatrixEquations.lyapd(A, C)::Matrix{T}
    catch
        return C, 0, 1.0
    end
    
    ensure_lyapunov_doubling_buffers!(workspace)
    𝐂A_tmp = workspace.𝐂A
    𝐂¹_tmp = workspace.𝐂¹
    ℒ.mul!(𝐂A_tmp, 𝐂, A')
    ℒ.mul!(𝐂¹_tmp, A, 𝐂A_tmp)
    ℒ.axpy!(1, C, 𝐂¹_tmp)
    ℒ.axpy!(-1, 𝐂, 𝐂¹_tmp)
    
    reached_tol = ℒ.norm(𝐂¹_tmp) / ℒ.norm(𝐂)

    return 𝐂, 0, reached_tol
end

end # module
