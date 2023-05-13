using Distributions, OnlineStats, Statistics
StatsPlots.plotlyjs()


sim = 1000000
burnin = 10000
x = 0.0
x̂ = zeros(sim)
β = [.95,-1.5,-1.95,.75,.1]

# s = Series(Moments())

for i in 1:sim
    # x = β[1] * x + β[end] * randn()
    # x = β[1] * x + β[2] * x^2 + β[end] * randn()
    # x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[end] * randn()
    x = β[1] * x + β[2] * x^2 + β[3] * x^3 + β[4] * x^4 + β[end] * randn()
    x̂[i] = x
    # if i > burnin fit!(s, x) end
end
# s
[mean(x̂[burnin:end]), var(x̂[burnin:end]), skewness(x̂[burnin:end]), kurtosis(x̂[burnin:end])]


density(x̂)
plot!(fit(Normal, x̂))



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