module TuringExt

# We only import `truncated` now, as we will be defining our own wrappers
# with the same names as the distributions.
import Turing: truncated
# We explicitly refer to the Turing distributions to avoid method overwriting.
import Turing
import DocStringExtensions: SIGNATURES
using DispatchDoctor
import MacroModelling: Normal, Beta, Cauchy, Gamma, InverseGamma, DEFAULT_TURING_USE_MEAN_STD

@stable default_mode = "disable" begin

#==========================================================================================
                                    Beta Distribution
==========================================================================================#

"""
$(SIGNATURES)
Constructs a `Beta` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The first parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The second parameter (β) of the distribution, or the standard deviation when `μσ=true`.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the `α` and `β` parameters.
"""
function Beta(μ::Real, σ::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    if μσ
        # Calculate alpha and beta from mean (μ) and standard deviation (σ)
        ν = μ * (1 - μ) / σ^2 - 1
        α = μ * ν
        β = (1 - μ) * ν
        return Turing.Beta(α, β)
    end
    # By default, treat μ and σ as the distribution parameters α and β
    return Turing.Beta(μ, σ)
end

"""
$(SIGNATURES)
Constructs a truncated `Beta` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The first parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The second parameter (β) of the distribution, or the standard deviation when `μσ=true`.
- `lower_bound` [Type: `Real`]: The truncation lower bound of the distribution.
- `upper_bound` [Type: `Real`]: The truncation upper bound of the distribution.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the `α` and `β` parameters.
"""
function Beta(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    # Create the base distribution, then truncate it
    dist = Beta(μ, σ; μσ=μσ)
    return truncated(dist, lower_bound, upper_bound)
end


#==========================================================================================
                                InverseGamma Distribution
==========================================================================================#

"""
$(SIGNATURES)
Constructs an `InverseGamma` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The shape parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The scale parameter (β) of the distribution, or the standard deviation when `μσ=true`.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the shape `α` and scale `β` parameters.
"""
function InverseGamma(μ::Real, σ::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    if μσ
        # Calculate shape (α) and scale (β) from mean (μ) and standard deviation (σ)
        α = (μ / σ)^2 + 2
        β = μ * ((μ / σ)^2 + 1)
        return Turing.InverseGamma(α, β)
    end
    # By default, treat μ and σ as the distribution parameters α and β
    return Turing.InverseGamma(μ, σ)
end

"""
$(SIGNATURES)
Constructs a truncated `InverseGamma` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The shape parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The scale parameter (β) of the distribution, or the standard deviation when `μσ=true`.
- `lower_bound` [Type: `Real`]: The truncation lower bound of the distribution.
- `upper_bound` [Type: `Real`]: The truncation upper bound of the distribution.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the shape `α` and scale `β` parameters.
"""
function InverseGamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    # Create the base distribution, then truncate it
    dist = InverseGamma(μ, σ; μσ=μσ)
    return truncated(dist, lower_bound, upper_bound)
end


#==========================================================================================
                                    Gamma Distribution
==========================================================================================#

"""
$(SIGNATURES)
Constructs a `Gamma` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The shape parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The rate parameter (θ) of the distribution, or the standard deviation when `μσ=true`.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the shape `α` and scale `θ` parameters.
"""
function Gamma(μ::Real, σ::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    if μσ
        # Calculate shape (α) and scale (θ) from mean (μ) and standard deviation (σ)
        θ = σ^2 / μ
        α = μ / θ
        return Turing.Gamma(α, θ)
    end
    # By default, treat μ and σ as the distribution parameters α and θ
    return Turing.Gamma(μ, σ)
end

"""
$(SIGNATURES)
Constructs a truncated `Gamma` distribution, optionally parameterized by its mean and standard deviation.

# Arguments
- `μ` [Type: `Real`]: The shape parameter (α) of the distribution, or the mean when `μσ=true`.
- `σ` [Type: `Real`]: The rate parameter (θ) of the distribution, or the standard deviation when `μσ=true`.
- `lower_bound` [Type: `Real`]: The truncation lower bound of the distribution.
- `upper_bound` [Type: `Real`]: The truncation upper bound of the distribution.

# Keyword Arguments
- `μσ` [Type: `Bool`, Default: `false`]: If `true`, `μ` and `σ` are interpreted as the mean and standard deviation to calculate the shape `α` and scale `θ` parameters.
"""
function Gamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real; μσ::Bool=DEFAULT_TURING_USE_MEAN_STD)
    # Create the base distribution, then truncate it
    dist = Gamma(μ, σ; μσ=μσ)
    return truncated(dist, lower_bound, upper_bound)
end


#==========================================================================================
                            Simple Truncation Wrappers
==========================================================================================#

"""
$(SIGNATURES)
Convenience wrapper for the truncated `Normal` distribution.

# Arguments
- `μ` [Type: `Real`]: The mean of the distribution.
- `σ` [Type: `Real`]: The standard deviation of the distribution.
- `lower_bound` [Type: `Real`]: The truncation lower bound of the distribution.
- `upper_bound` [Type: `Real`]: The truncation upper bound of the distribution.
"""
function Normal(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Turing.Normal(μ, σ), lower_bound, upper_bound)
end

function Normal(μ::Real, σ::Real)
    Turing.Normal(μ, σ)
end

"""
$(SIGNATURES)
Convenience wrapper for the truncated `Cauchy` distribution.

# Arguments
- `μ` [Type: `Real`]: The location parameter.
- `σ` [Type: `Real`]: The scale parameter.
- `lower_bound` [Type: `Real`]: The truncation lower bound of the distribution.
- `upper_bound` [Type: `Real`]: The truncation upper bound of the distribution.
"""
function Cauchy(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Turing.Cauchy(μ, σ), lower_bound, upper_bound)
end

function Cauchy(μ::Real, σ::Real)
    Turing.Cauchy(μ, σ)
end


end # @stable

end # module
