
using Distributions, OnlineStats, Statistics
StatsPlots.plotlyjs()


sim = 100000
burnin = 1
x = 0.0
x̂ = zeros(sim)
β = [.95,-1.5,-1.95,.75,.1]

s = Series(Moments())

for i in 1:sim
    # x = β[1] * x + β[end] * randn()
    # x = β[1] * x + β[2] * x^2 + β[end] * randn()
    x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[end] * randn()
    # x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[4] * x^4 + β[end] * randn()
    x̂[i] = x 
    fit!(s, x)
    # if i > burnin fit!(s, x) end
end
s

[mean(x̂[burnin:end]), var(x̂[burnin:end]), skewness(x̂[burnin:end]), kurtosis(x̂[burnin:end])]


density(x̂)
plot!(fit(Normal, x̂))
using Optim

function simulate_moments(β)
    sim = 1000000000
    x = 0.0
    x̄ = 0.0
    x̂ = zeros(sim)
    for i in 1:sim
        # x = β[1] * x + β[end] * randn()
        E = randn()
        x = β[1] * x + β[2] * x̄^2 + β[end] * E
        x̄ = β[1] * x̄ + β[end] * E
        # x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[end] * randn()
        # x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[4] * x^4 + β[end] * randn()
        x̂[i] = x
    end
    [mean(x̂), var(x̂), skewness(x̂), kurtosis(x̂)]
end
simulate_moments([.9,.5,.01])

simulate_moments([.95,-.05,-0.195,.075])
sum(abs2,simulate_moments([.95,-.05,-0.195,.075]) - [.01,.1,.1,3])

sol = Optim.optimize(x->sum(abs2,simulate_moments(x) - [.01,.1,.5,0]),fill(eps(Float32),5), iterations = 1000, allow_f_increases = true)


sol.minimizer |> simulate_moments

β = [.5,.01,.1]

s = Series(Moments())

for i in 1:1000000
    x = β[1] * x + β[2] * x^2 + β[3] * randn()
    fit!(s, x)
end
s

x = Normal()


x̂ = β¹ * x + β² * x

mean(x̂)

# first order
a = .9
b = .01
x = 0
x = a * x + b * randn()

b / sqrt(1 - a^2)

# second order
a = .9
b = .5
c = .01

x = 0
x̂ = 0

x̂ = a * x̂ + c * randn()
# mean of x̂ = 0
# std of x̂ = c / sqrt(1 - a^2)

# mean of b * x̂^2 = b * c^2 / (1 - a^2)
# std of b * x̂^2 =  sqrt(2) * b * c^2 / (1 - a^2)
# skew of b * x̂^2 = sqrt(8)
# kurtosis of b * x̂^2 = 12
x = a * x + b * x̂^2 + c * randn()

using Distributions
mean(.9*Chisq(1))
std(.0009*Chisq(1))
skewness(.0009*Chisq(1))
kurtosis(.0009*Chisq(1))


std(.0009*Normal()*Normal())

using SymPy, Statistics
x =("x",0,1)


xx = randn(1000000) * c / sqrt(1 - a^2)
mean(xx)
std(xx)
skewness(xx)
kurtosis(xx)

b*(c / sqrt(1 - a^2))^2 * sqrt(2)
mean(b*xx.^2)
std(b*xx.^2)
skewness(b*xx.^2)
kurtosis(b*xx.^2)