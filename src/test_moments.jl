
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
    [mean(x̂), std(x̂), skewness(x̂), kurtosis(x̂)]
end
simulate_moments([.5,.5,.01])

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



# covariance of multivariate normal ^ 2
using Distributions
A = randn(2,2)
X = MultivariateNormal([1 .4;.4 3])
X = rand(X,1000000).^2;
mean(X,dims=2)
std(X,dims=2)
var(X,dims=2)
mean([1 0; .5 -1]*X,dims=2)
std([1 0; .5 -1]*X,dims=2)
cov(X')


# mean of X^2 is var(X)
# std  of X^2 is sqrt(2) * var(X)

# X = A X + C ϵ

# mean of X is 0
# var  of X is solve    A * covar * A' + C * C' - covar    for covar

A = randn(2,2)
C = randn(2,2)

using LinearMaps, IterativeSolvers, LinearAlgebra
CC = C * C'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

covar = reshape(gmres(lm, vec(-CC)), size(CC))

diag(covar)


B = [0.5 0.75 ; 0.5 0]

mean_X2 = diag(covar)
mean_BX2 = B*diag(covar)
std_X2 = sqrt(2) * diag(covar)
var_X2 = 2 * diag(covar).^2
var_BX2 = diag(B * diagm(2 * diag(covar).^2) * B')
std_BX2 = sqrt.(diag(B * diagm(2 * diag(covar).^2) * B'))



X = MultivariateNormal(diagm(diag(covar)))
X = rand(X,1000000).^2;
mean(X,dims=2)
mean(B*X,dims=2)
std(X,dims=2)
var(X,dims=2)
std(B*X,dims=2)
var(B*X,dims=2)


# B * X^2 + C * ϵ
X = MultivariateNormal(diagm(diag(covar)))
X = rand(X,1000000).^2;

e = MultivariateNormal(diagm(ones(2)))
e = rand(e,1000000).^2;
x_up = B * X + C * e

var(x_up, dims = 2)


var_BX2 = diag(B * diagm(2 * diag(covar).^2) * B')

var_BX2Ce = var_BX2 + diag(C * C')
var_BX2CeX = var_BX2 + diag(C * C')



for i in 1:1000
var_BX2CeX = diag(A * diagm(var_BX2CeX) * A') + var_BX2Ce
end

diag(A * diagm(var_BX2Ce) * A') + var_BX2Ce
(diag(B * diagm(2 * diag(covar).^2) * B') + diag(C * C'))
diag((I - A) \ diagm(var_BX2Ce)) + 2*var_BX2Ce





# full AR model
sim = 1000000
X = zeros(2)
X̄ = zeros(2)
X̂ = randn(sim,2)

for i in 1:sim
    ϵ = randn(2)
    X̄ = A * X̄ + C * ϵ
    X = A * X + B * X̄.^2 + C * ϵ
    X̂[i,:] = X
end

mean(X̂, dims = 1)
(I - A) \ B * diag(covar) # closed form mean 
std(X̂, dims = 1)
var(X̂, dims = 1)