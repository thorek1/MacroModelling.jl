from sympy import symbols, Eq, solve, solveset, S

# Defining the symbols (variables and constants)
c0, l0, w0, y0, sigma, psi, alpha, k_neg1, z0, g0, i0 = symbols('c0 l0 w0 y0 sigma psi alpha k_neg1 z0 g0 i0')
aa, bb, cc, dd = symbols('aa, bb, cc, dd')

# Defining the equations
eq1 = Eq(c0**sigma * psi / (1 - l0) - w0, 0)
eq2 = Eq(-k_neg1**alpha * l0**(1 - alpha) * z0 + y0, 0)
eq3 = Eq(w0 - y0 * (1 - alpha) / l0, 0)
eq4 = Eq(-c0 - g0 - i0 + y0, 0)

# Solving the system of equations
solutions = solve((eq1, eq2, eq3, eq4), (c0, l0, w0, y0))
solutions = solve((eq1, eq2, eq3, eq4), (l0, w0, y0))

solutions
solutions

# Substituting
c0 + g0 + i0
# eq1 = Eq(c0**sigma * psi / (1 - l0) - w0, 0)
eq2 = Eq(-k_neg1**alpha * l0**(1 - alpha) * z0 + c0 + g0 + i0, 0)
eq3 = Eq(c0**sigma * psi / (1 - l0) - (c0 + g0 + i0) * (1 - alpha) / l0, 0)
# eq4 = Eq(-c0 - g0 - i0 + y0, 0)

sigmaa = symbols('sigmaa', extended_Real=True)
eq2 = Eq(-k_neg1**alpha * l0**(1 - alpha) * z0 + c0 + g0 + i0, 0)
eq3 = Eq(c0**sigma * psi / (1 - l0) - (c0 + g0 + i0) * (1 - alpha) / l0, 0)
eq3 = Eq(c0**sigmaa * bb - (c0 + dd) * cc, 0)
eq3 = Eq(c0**sigmaa - (c0 + dd), 0)
solutions = solve([eq3], [c0])

solutions = solve((eq2, eq3), (c0, l0))

from sympy import symbols, Eq, solve
x,a,b = symbols('x,a,b')
EQ2 = Eq(x**a - (x + b), 0)
solutions = solve(EQ2, x,)

solve(solutions)


from sympy import symbols, Eq, solve

x,a,b = symbols('x,a,b')

EQ2 = Eq(x**a - x + b, 0)

solutions = solve(EQ2, x)

# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/Users/thorekockerols/.julia/environments/v1.10/.CondaPkg/env/lib/python3.12/site-packages/sympy/solvers/solvers.py", line 1145, in solve
#     solution = _solve(f[0], *symbols, **flags)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/Users/thorekockerols/.julia/environments/v1.10/.CondaPkg/env/lib/python3.12/site-packages/sympy/solvers/solvers.py", line 1693, in _solve
#     raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))
# NotImplementedError: multiple generators [x, x**a]
# No algorithms are implemented to solve equation b - x + x**a

from sympy.solvers.solveset import nonlinsolve
from sympy import var

var('c k mc n r_k eta w alpha', Rational = True, positive=True)

eq1 = -r_k + alpha * mc * k**(-1 + alpha) * n**(1 - alpha)
eq2 = -w + mc * (1 - alpha) * k**alpha * n**(-alpha)
eq3 = c**-1 * w - n**eta

eq_system = [eq1,eq2,eq3]

solve(eq_system , [n,k,w])


