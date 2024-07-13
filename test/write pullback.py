import sympy as sp

# Use actual matrix dimensions assuming dimensions of P, A, B, C are known
n, m = 7, 27  # Example dimensions
A = sp.MatrixSymbol('A', m, m)
B = sp.MatrixSymbol('B', m, m)
P = sp.MatrixSymbol('P', m, m)
C = sp.MatrixSymbol('C', n, m)
u = sp.MatrixSymbol('u', m, 1)
data = sp.MatrixSymbol('data', n, 1)
loglik = sp.Symbol('loglik')


# Intermediate calculations
v = data - C * u

F = C * P * C.T
invF = F.inv()
detF = F.det()

# Outputs
llh1 = sp.log(detF)
llh2 = (v.T * invF * v)
# llh = loglik + sp.log(detF) + (v.T * invF * v)
u_hat = A * (u + P * C.T * invF * v)
P_hat = A * (P - P * C.T * invF * C * P) * A.T + B



# Inputs: u, P , llh, data

# derive pushforward wrt u
d_llh2_du = sp.diff(llh2, u)
d_u_hat_du = sp.diff(u_hat, u)
d_P_hat_du = sp.diff(P_hat, u)
d_v_du = sp.diff(v, u)
d_v_du.shape

forw_du = sp.MatrixSymbol('forw_du', m, m)
back_du = sp.MatrixSymbol('back_du', m, n)

d_v_du.shape
forw_du.shape

(forw_du * d_v_du).shape

eq = back_du - forw_du * d_v_du
sp.solve(eq, 'forw_du')

sp.solve(d_u_hat_du, u)
d_llh2_du + d_u_hat_du
d_llh2_du.shape

d_u_hat_du.shape

# Differentiate llh w.r.t. P, u_hat w.r.t. A, P, u, and P_hat w.r.t. A, P, B
d_llh_ddata1 = sp.diff(v.T * invF * v, data)
d_llh_ddata2 = sp.diff(sp.log(detF), data)

d_llh_dP = sp.diff(llh, P)
d_uhat_dA = sp.diff(u_hat, A)
d_uhat_dP = sp.diff(u_hat, P)
d_uhat_du = sp.diff(u_hat, u)
d_uhat_ddata = sp.diff(u_hat, data)
d_Phat_dA = sp.diff(P_hat, A)
d_Phat_dP = sp.diff(P_hat, P)
d_Phat_dB = sp.diff(P_hat, B)