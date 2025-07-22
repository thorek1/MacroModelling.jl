module TuringExt

import Turing: Beta, InverseGamma, Gamma, Normal, Cauchy, truncated
import DocStringExtensions: SIGNATURES
using DispatchDoctor

@stable default_mode = "disable" begin
"""
$(SIGNATURES)
Convenience wrapper for the truncated Beta distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
- `lower_bound` [Type: `Real`]: truncation lower bound of the distribution
- `upper_bound` [Type: `Real`]: truncation upper bound of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters
"""
function Beta(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real, ::Val{:μσ})
    truncated(Beta(((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2, ((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2  * (1 / μ - 1)), lower_bound, upper_bound)
end


function Beta(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Beta(μ, σ), lower_bound, upper_bound)
end

"""
$(SIGNATURES)
Convenience wrapper for the Beta distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters

"""
function Beta(μ::Real, σ::Real, ::Val{:μσ})
    Beta(((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2, ((1 - μ) / σ ^ 2 - 1 / μ) * μ ^ 2  * (1 / μ - 1))
end


"""
$(SIGNATURES)
Convenience wrapper for the truncated Inverse Gamma distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
- `lower_bound` [Type: `Real`]: truncation lower bound of the distribution
- `upper_bound` [Type: `Real`]: truncation upper bound of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters

"""
function InverseGamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real, ::Val{:μσ})
    truncated(InverseGamma((μ / σ) ^ 2 + 2, μ * ((μ / σ) ^ 2 + 1)), lower_bound, upper_bound)
end

function InverseGamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(InverseGamma(μ, σ), lower_bound, upper_bound)
end


"""
$(SIGNATURES)
Convenience wrapper for the Inverse Gamma distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters

"""
function InverseGamma(μ::Real, σ::Real, ::Val{:μσ})
    InverseGamma((μ / σ) ^ 2 + 2, μ * ((μ / σ) ^ 2 + 1))
end


"""
$(SIGNATURES)
Convenience wrapper for the truncated Inverse Gamma distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
- `lower_bound` [Type: `Real`]: truncation lower bound of the distribution
- `upper_bound` [Type: `Real`]: truncation upper bound of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters

"""
function Gamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real, ::Val{:μσ})
    truncated(Gamma(μ^2/σ^2, σ^2 / μ), lower_bound, upper_bound)
end

function Gamma(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Gamma(μ, σ), lower_bound, upper_bound)
end

"""
$(SIGNATURES)
Convenience wrapper for the Gamma distribution.

If `μσ = true` then `μ` and `σ` are translated to the parameters of the distribution. Otherwise `μ` and `σ` represent the parameters of the distribution.

# Arguments
- `μ` [Type: `Real`]: mean or first parameter of the distribution,
- `σ` [Type: `Real`]: standard deviation or first parameter of the distribution
# Keyword Arguments
- `μσ` [Type: `Bool`]: switch whether μ and σ represent the moments of the distribution or their parameters

"""
function Gamma(μ::Real, σ::Real, ::Val{:μσ})
    Gamma(μ^2/σ^2, σ^2 / μ)
end




"""
$(SIGNATURES)
Convenience wrapper for the truncated Normal distribution.

# Arguments
- `μ` [Type: `Real`]: mean of the distribution 
- `σ` [Type: `Real`]: standard deviation of the distribution
- `lower_bound` [Type: `Real`]: truncation lower bound of the distribution
- `upper_bound` [Type: `Real`]: truncation upper bound of the distribution

"""
function Normal(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Normal(μ, σ), lower_bound, upper_bound)
end

Normal(x,y) = Normal(x,y)


"""
$(SIGNATURES)
Convenience wrapper for the truncated Cauchy distribution.

# Arguments
- `μ` [Type: `Real`]: location,
- `σ` [Type: `Real`]: scale
- `lower_bound` [Type: `Real`]: truncation lower bound of the distribution
- `upper_bound` [Type: `Real`]: truncation upper bound of the distribution

"""
function Cauchy(μ::Real, σ::Real, lower_bound::Real, upper_bound::Real)
    truncated(Cauchy(μ, σ), lower_bound, upper_bound)
end

end # dispatch_doctor

end # module