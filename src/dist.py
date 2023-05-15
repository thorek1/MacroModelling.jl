import sympy

from sympy.stats import MultivariateNormal, density, marginal_distribution, E, Normal, std, skewness, kurtosis, variance
from sympy import symbols, MatrixSymbol, sqrt, exp ,pi

a = .9
b = .5
c = .01 
ahat = 1 / sqrt(1 - a**2)
a, ahat, b, c = symbols('a ahat b c')
x = Normal('x',0,c * ahat)

E(x)
std(x)
skewness(x)
kurtosis(x)


E(b*x**2)
0.5*ahat**2*c**2
std(b*x**2)
sqrt(2)*sqrt(ahat**4*b**2*c**4)
skewness(b*x**2)
2*sqrt(2)*ahat**6*b**3*c**6/(ahat**4*b**2*c**4)**(3/2)
kurtosis(b*x**2)
15


e = Normal('e',0,1)

E(b*x**2 + c*e)
0.5*ahat**2*c**2
std(b*x**2 + c*e)
sqrt(2*ahat**4*b**2*c**4 + c**2)
skewness(b*x**2 + c*e)
(8*ahat**6*b**3*c**6)*(2*ahat**4*b**2*c**4 + c**2)**(-3/2)
kurtosis(b*x**2 + c*e)
(2*ahat**4*b**2*c**4 + c**2)**(-2) * (60*ahat**8*b**4*c**8 + 12*ahat**4*b**2*c**6 + 3*c**4)



E(a*x + b*x**2 + c * e)
0.5*ahat**2*c**2
std(a*x + b*x**2 + c * e)
sqrt(a**2*ahat**2*c**2 + 2*ahat**4*b**2*c**4 + c**2)
skewness(a*xhat + b*x**2 + c * e)
kurtosis(a*xhat + b*x**2 + c * e)

xhat = a*xhat + b*x**2 + c * e


x1 = a*x + b*x**2 + c * e
E(a*x1 + b*x**2 + c * e)
ahat**2*b*c**2 + a*ahat**2*b*c**2

ahat**2*b*c**2 / (1-a)
b*c**2 / ((1-a)*(1-a**2))# this one for E




a, ahat, b, c = symbols('a ahat b c')
x = Normal('x',0,c * ahat)
# std(a*(a*x + b*x**2 + c * e))
# std(a*(a*x + b*x**2 + c * e) + b*x**2 + c * e)
std(a**2*x + (1+a)*b*x**2 + (1+a)*c * e)
sqrt(a**4*ahat**2*c**2 + 2*a**2*ahat**4*b**2*c**4 + a**2*c**2 + 4*a*ahat**4*b**2*c**4 + 2*a*c**2 + 2*ahat**4*b**2*c**4 + c**2)

sqrt(a**2*ahat**2*c**2 + 2*ahat**4*b**2*c**4 + c**2) / (1-a)


variance(a**2*x + (1+a)*b*x**2 + (1+a)*c * e)
a**4*ahat**2*c**2 + 2*a**2*ahat**4*b**2*c**4 + a**2*c**2 + 4*a*ahat**4*b**2*c**4 + 2*a*c**2 + 2*ahat**4*b**2*c**4 + c**2



variance(a*x + b*x**2 + c * e)
a**2*ahat**2*c**2 + 2*ahat**4*b**2*c**4 + c**2

(a**2*ahat**2*c**2 + 2*ahat**4*b**2*c**4 + c**2) / (1-a**2)
sqrt((2*ahat**4*b**2*c**4 + c**2)/(1 - a**2)) # this one for std






# Sk_x * (σ_x / σ_z)^3 + Sk_y * (σ_y / σ_z)^3

a = .5
b = .5
c = .01 
ahat = 1 / sqrt(1 - a**2)


X = MultivariateNormal('X', [0, 0], [[1, 0], [0, 1]])
y, z = symbols('y z')
mean(X)(X[0])
density(X)(y, z)
density(X)(1, 2)
marginal_distribution(X, X[1])(y)
marginal_distribution(X, X[0])(y)