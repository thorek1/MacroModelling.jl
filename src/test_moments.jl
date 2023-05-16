
using Distributions, OnlineStats, Statistics, StatsPlots
# StatsPlots.plotlyjs()


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

# A = randn(2,2)
A = [.9 .1 ;-.1 .05]
# C = randn(2,2)
C = [.25 -.1 ;.31 .5]

using LinearMaps, IterativeSolvers, LinearAlgebra
CC = C * C'

lm = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' - reshape(x,size(CC)), length(CC))

covar = reshape(gmres(lm, vec(-CC)), size(CC))

diag(covar)





B = [0.5 0.75 ; 0.5 0]




# Σ_X̂ = A * Σ_X̂ * A' + B * E[X̄̂[i-1,:]^4] - (E[X̄̂[i-1,:]^2])^2 + C * C',

EX22 = diag(covar^2)
EX4 = 3 * diag(A * covar * A').^2 + 6 * diag(A * covar * A') .* diag(C * C') + 3 * diag(C * C').^2 # works
# B * EX4 - covar^2

EX = B * diagm(EX4 - EX22)
EX = B * (EX4) - EX22
EX = B * (3 * (A * covar * A').^2 + 6 * (A * covar * A') .* (C * C') + 3 * (C * C').^2) * B'
lmvar = LinearMap{Float64}(x -> A * reshape(x,size(CC)) * A' + EX - reshape(x,size(CC)), length(CC))

covarvar = reshape(gmres(lmvar, vec(-CC)), size(CC))

CC = C * C'
EX = diagm(diag(B * (3 * (A * covar * A').^2 + 6 * (A * covar * A') .* (C * C') + 3 * (C * C').^2 - covar.^2) * B'))
covX = EX + CC

covX = A * covX * A' + EX + CC
for i in 1:1000
    covX = A * covX * A' + EX + CC
end
covX


var(X̂, dims = 1)

eX4 =  B * mean((X̄̂').^4,dims=2)
eX2 = mean((X̄̂'.^2),dims=2).^2

eX = diagm(vec(eX4 - eX2))

cov(X̂, dims = 1)
EX = cov(B * X̄̂[1:end-1,:]'.^2 + C * ϵ̄[2:end,:]', dims = 2)



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
# var_BX2CeX = var_BX2 + diag(C * C')



# for i in 1:1000
# var_BX2CeX = diag(A * diagm(var_BX2CeX) * A') + var_BX2Ce
# end

# diag(A * diagm(var_BX2Ce) * A') + var_BX2Ce
(diag(B * diagm(2 * diag(covar).^2) * B') + diag(C * C')) # this is the one since the two are independent
# diag((I - A) \ diagm(var_BX2Ce)) + 2*var_BX2Ce



# full AR model
sim = 1000000
X̂ = zeros(sim,2)
X̄̂ = zeros(sim,2)
ϵ̄ = randn(sim,2)


X̂[1,:] = C * ϵ̄[1,:]
X̄̂[1,:] = C * ϵ̄[1,:]
for i in 2:sim
    X̂[i,:] = A * X̂[i-1,:] + B * X̄̂[i-1,:].^2 + C * ϵ̄[i,:]
    X̄̂[i,:] = A * X̄̂[i-1,:] + C * ϵ̄[i,:]
end

mean((A * X̄̂').^4,dims=2)


mean(X̂, dims = 1)
(I - A) \ B * diag(covar) # closed form mean 
std(X̂, dims = 1)
var(X̂, dims = 1)

cov(X̄̂.^2,ϵ̄)
cov((B*(X̄̂.^2)')',(C*ϵ̄')')
cov(X̄̂.^2,X̂)

diag(A^1 * C*C' ./ diag(C*C'))


var(B * (X̄̂[1:end-1,:].^2)' + C * ϵ̄[2:end,:]', dims = 2)
(diag(B * diagm(2 * diag(covar).^2) * B') + diag(C * C')) # this is the one since the two are independent





var(A * X̂', dims = 2)
var(B * (X̄̂.^2)', dims = 2)
var(A * X̂'+ B * (X̄̂.^2)', dims = 2)
(diag(B * diagm(2 * diag(covar).^2) * B') + diag(C * C')) # this is the one since the two are independent

#### var(X̂, dims = 1) ####
# Var[A * X̂[i-1,:] + B * X̄̂[i-1,:].^2 + C * ϵ̄[i,:]]
# Var[A * X̂ + B * X̄̂.^2 + C * ϵ̄]
# Var[A * X̂] + Var[B * X̄̂.^2] + Var[C * ϵ̄] + 2 * Covar[A * X̂,B * X̄̂.^2]
var(X̂, dims = 1) # missing error term
var(A * X̂' + B * (X̄̂.^2)', dims = 2)
var(A * X̂', dims = 2) + var(B * (X̄̂.^2)', dims = 2) + 2 * diag(cov(A * X̂',B * (X̄̂.^2)', dims = 2))
cov(X̂',(X̄̂.^2)', dims = 2)
cov(X̂',(X̄̂.^2)', dims = 2)


diag(cov(A * X̂',B * (X̄̂.^2)', dims = 2))
mean((A * X̂') .* (B * (X̄̂.^2)'), dims = 2) - mean((A * X̂'), dims = 2) .* mean((B * (X̄̂.^2)'), dims = 2)

mean((A * X̂' + B * (X̄̂.^2)').^2, dims = 2) - mean(A * X̂' + B * (X̄̂.^2)', dims = 2).^2

mean((A * X̂' + B * (X̄̂.^2)').^2, dims = 2) - (mean(A * X̂', dims = 2) .+ mean(B * (X̄̂.^2)', dims = 2)).^2

mean((A * X̂' + B * (X̄̂.^2)').^2, dims = 2) - (mean(A * X̂', dims = 2).^2 .+ mean(B * (X̄̂.^2)', dims = 2).^2 .+ 2 * mean(A * X̂', dims = 2) .* mean(B * (X̄̂.^2)', dims = 2))


mean((A * X̂' + B * (X̄̂.^2)').^2, dims = 2) 
mean((A * X̂').^2, dims = 2) + mean((B * (X̄̂.^2)').^2, dims = 2) + mean(2 * ((A * X̂') .* (B * (X̄̂.^2)')), dims = 2)



mean(X̂, dims = 1)
mean(X̄̂, dims = 1)
mean(X̂.*X̄̂, dims = 1)
mean(X̂.^2, dims = 1)
mean(X̄̂.^2, dims = 1)

mean(X̂, dims = 1).*mean(X̄̂, dims = 1)

mean(2 * ((A * X̂') .* (B * (X̄̂.^2)')), dims = 2)
mean(2 * (A * (A * X̂[1:end-1,:]' + B * X̄̂[1:end-1,:]'.^2 + C * ϵ̄[2:end,:]') .* (B * (X̄̂[2:end,:].^2)')), dims = 2)

# mean(2 * (A * (C * ϵ̄[2:end,:]') .* (B * (X̄̂[2:end,:].^2)')), dims = 2)
mean(2 * (A * (B * X̄̂[1:end-1,:]'.^2) .* (B * (X̄̂[2:end,:].^2)')), dims = 2) + mean(2 * (A * (A * X̂[1:end-1,:]') .* (B * (X̄̂[2:end,:].^2)')), dims = 2)




mean((A * X̂').^2, dims = 2)
var(A * X̂', dims = 2)
mean((B * (X̄̂.^2)').^2, dims = 2)
var(B * (X̄̂.^2)', dims = 2)

diag(cov(A * X̂',B * (X̄̂.^2)', dims = 2))
2 * mean((A * X̂') .* (B * (X̄̂.^2)'), dims = 2) 

A^2 * var(X̂', dims = 2) + B^2 * var(X̄̂'.^2, dims=2) + 2 * A * B * var(X̂' .* (X̄̂.^2)', dims = 2)


2 * A * B * var(A * X̂' + B * (X̄̂.^2)', dims = 2)

var(A * X̂' + B * (X̄̂.^2)', dims = 2)

var(A * X̂', dims = 2)
diag(A * cov(X̂', dims = 2) * A')

# Var[B * X̄̂.^2] = B * Var[X̄̂.^2] * B' = B * (E[X̄̂.^4] - E[X̄̂.^2]^2) * B'
diag(B * (3 * (A * covar * A').^2 + 6 * (A * covar * A') .* (C * C') + 3 * (C * C').^2 - covar.^2) * B')
var(B * (X̄̂').^2,dims=2)

# E[X̄̂.^4] = Var[X̄̂.^2] + Var[X̄̂]^2 = (A * X̄̂[i-1,:] + C * ϵ̄[i,:])^4
# E[X̄̂.^4] = 3 * (Var[X̄̂[i,:]])^2 + 4 * E[X̄̂[i,:]] * Var[X̄̂[i,:]] + (E[X̄̂[i,:]])^2


# E[X̄̂.^4] = Var[X̄̂.^2] + Var[X̄̂]^2 = E[(A * X̄̂[i-1,:] + C * ϵ̄[i,:])^4]
# (A * X̄̂[i-1,:] + C * ϵ̄[i,:])^4 = E[(A * X̄̂[i-1,:])^4] + E[4*(A * X̄̂[i-1,:])^3*(C * ϵ̄[i,:])] + E[6*(A * X̄̂[i-1,:])^2*(C * ϵ̄[i,:])^2] + E[4*(A * X̄̂[i-1,:])*(C * ϵ̄[i,:])^3] + E[(C * ϵ̄[i,:])^4]


# E[X̄̂.^4]
3 * diag(A * covar * A').^2 + 6 * diag(A * covar * A') .* diag(C * C') + 3 * diag(C * C').^2 # works
mean((X̄̂').^4,dims=2)

# E[(A * X̄̂[i-1,:])^4] + 
diag(A * covar * A').^2 * 3 # works
mean((A * X̄̂').^4,dims=2)

### E[4*(A * X̄̂[i-1,:])^3*(C * ϵ̄[i,:])] + 

# E[6*(A * X̄̂[i-1,:])^2*(C * ϵ̄[i,:])^2] + 
# E[6*(A * X̄̂[i-1,:])^2] * E[(C * ϵ̄[i,:])^2]
6 * diag(A * covar * A') .* diag(C * C') # thats the one
mean( 6*(A * X̄̂[1:end-1,:]').^2, dims = 2)
mean(X̄̂[1:end-1,:]'.^2, dims = 2)
mean((C * ϵ̄[2:end,:]').^2, dims = 2) .* mean( 6*(A * X̄̂[1:end-1,:]').^2, dims = 2)
mean( 6*(A * X̄̂[1:end-1,:]').^2 .*(C * ϵ̄[2:end,:]').^2, dims = 2)
### E[4*(A * X̄̂[i-1,:])*(C * ϵ̄[i,:])^3] + 

# E[(C * ϵ̄[i,:])^4]
diag(C * C').^2 * 3 # works

mean((C * ϵ̄').^4,dims=2)


(I - A) \ ((diag(C * C').^2 * 3) + 6 * diag(A * covar * A') .* diag(C * C'))


(I - A) \ (diag(C * C').^2 * 3 + 6 * diag(A * covar * A') .* diag(C * C'))

3 * var(X̄̂, dims = 1).^2 + 4 * mean(X̄̂, dims = 1) .* var(X̄̂, dims = 1) + mean(X̄̂, dims = 1).^2

# mean(X̄̂, dims = 1).^4 ./ std(X̄̂, dims = 1).^4
mean(X̄̂.^4, dims = 1)
var(X̄̂.^2, dims = 1) + var(X̄̂, dims = 1).^2

mean(X̄̂.^2, dims = 1)
(I - A) \ (C * C')

(I - A) \ ((B * diagm(2 * diag(covar).^2) * B') + (C * C')) 

(I - A) \ ((I - A) \ ((diag(B * diagm(2 * diag(covar).^2) * B')) + diag(C * C')) + diag(C * C'))

# E[X̂ * X̄̂]

(A * X̂[i-1,:] + B * X̄̂[i-1,:].^2 + C * ϵ̄[i,:]) * (A * X̄̂[i-1,:] + C * ϵ̄[i,:])

(A * X̂ + B * X̄̂.^2 + C * ϵ̄) * (A * X̄̂ + C * ϵ̄)

A * X̂ * A * X̄̂   +   B * X̄̂.^2*A * X̄̂   +   C * ϵ̄ * A * X̄̂   +   A * X̂ * C * ϵ̄    +   B * X̄̂.^2 * C * ϵ̄   +   C * ϵ̄ * C * ϵ̄

mean(X̂ .* X̄̂, dims = 1)
mean(X̄̂.^3, dims = 1)
mean(X̂[1:end-1,:] .* ϵ̄[2:end,:], dims = 1)
mean(X̄̂[1:end-1,:] .* ϵ̄[2:end,:], dims = 1)
mean(X̄̂[1:end-1,:].^2 .* ϵ̄[2:end,:], dims = 1)
mean(ϵ̄.^2, dims = 1)


var(X̄̂.^2, dims = 1) 

A * X̂ * A * X̄̂   +   C * C'
A * X̂ * A * X̄̂   +   C * C'




# lets start from scratch. i am looking for    var(X̂, dims = 1)
# with  X̂[i,:] = A * X̂[i-1,:] + B * X̄̂[i-1,:].^2 + C * ϵ̄[i,:]
# and   X̄̂[i,:] = A * X̄̂[i-1,:] + C * ϵ̄[i,:]

var(A * X̂[1:end-1,:]' + B * X̄̂[1:end-1,:]'.^2 + C * ϵ̄[2:end,:]', dims = 2)
var(A * X̂[1:end-1,:]', dims = 2) + var(B * X̄̂[1:end-1,:]'.^2, dims = 2) + var(C * ϵ̄[2:end,:]', dims = 2) + 2 * diag(cov(A * X̂[1:end-1,:]', B * X̄̂[1:end-1,:]'.^2, dims = 2)) + 2 * diag(cov(A * X̂[1:end-1,:]', C * ϵ̄[2:end,:]', dims = 2)) + 2 * diag(cov(B * X̄̂[1:end-1,:]'.^2, C * ϵ̄[2:end,:]', dims = 2))



#this is the significant part:
var(A * X̂[1:end-1,:]', dims = 2)


var(B * X̄̂[1:end-1,:]'.^2, dims = 2)
# diag(B * (3 * (A * covar * A').^2 + 6 * (A * covar * A') .* (C * C') + 3 * (C * C').^2 - covar.^2) * B') # works

var(C * ϵ̄[2:end,:]', dims = 2)
# C * C' # works

2 * diag(cov(A * X̂[1:end-1,:]', B * X̄̂[1:end-1,:]'.^2, dims = 2)) 

cov(A * X̂[1:end-1,:]', B * X̄̂[1:end-1,:]'.^2, dims = 2)

diagm(vec(mean((A * X̂[1:end-1,:]') .* (B * X̄̂[1:end-1,:]'.^2) , dims = 2))) - mean((B * X̄̂[1:end-1,:]'.^2) , dims = 2) * mean((A * X̂[1:end-1,:]') , dims = 2)'

mean((A * X̂[1:end-1,:]') .* (B * X̄̂[1:end-1,:]'.^2) , dims = 2) - diag(mean((B * X̄̂[1:end-1,:]'.^2) , dims = 2) * mean((A * X̂[1:end-1,:]') , dims = 2)')


var(A * X̂[1:end-1,:]', dims = 2) + var(B * X̄̂[1:end-1,:]'.^2, dims = 2) + var(C * ϵ̄[2:end,:]', dims = 2) + 2 * (mean((A * X̂[1:end-1,:]') .* (B * X̄̂[1:end-1,:]'.^2) , dims = 2) - diag((B * mean(X̄̂[1:end-1,:]'.^2 , dims = 2)) * (A * mean(X̂[1:end-1,:]' , dims = 2))'))


mean((A * X̂[1:end-1,:]') , dims = 2)

diag((B * mean(X̄̂[1:end-1,:]'.^2 , dims = 2)) * (A * mean(X̂[1:end-1,:]' , dims = 2))')

diag(B * diag(covar) * (A * mean(X̂[1:end-1,:]' , dims = 2))')