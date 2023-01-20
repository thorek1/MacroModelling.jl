// cyclic reduction algorithm
var y, c, k, a, h, b;
varexo e, u;

verbatim;
% I want these comments included in
% example1.m 1999q1 1999y
%
var = 1;
end;

parameters beta, rho, alpha, delta, theta, psi, tau;

alpha = 0.36;
rho   = 0.95;
tau   = 0.025;
beta  = 0.99;
delta = 0.025;
psi   = 0;
theta = 2.95;

phi   = 0.1;

model;
c*theta*h^(1+psi)=(1-alpha)*y;
k = beta*(((exp(b)*c)/(exp(b(+1))*c(+1)))
    *(exp(b(+1))*alpha*y(+1)+(1-delta)*k));
y = exp(a)*(k(-1)^alpha)*(h^(1-alpha));
k = exp(b)*(y-c)+(1-delta)*k(-1);
a = rho*a(-1)+tau*b(-1) + e;
b = tau*a(-1)+rho*b(-1) + u;
end;

steady_state_model;
  K_Y = beta*alpha /(1 - beta*(1 - delta));
  H_Y = K_Y^(-alpha/(1 - alpha));
  C_Y = 1 - delta*K_Y;
  y = (theta*C_Y*H_Y^(1 + psi)/(1 - alpha))^(-1/(1 + psi));
  c = C_Y*y;
  k = K_Y*y;
  h = H_Y*y;
  a = 0;
  b = 0;
end;
  
shocks;
var e; stderr 0.009;
var u; stderr 0.009;
//var e, u = phi*0.009*0.009;
var e, u = 0.009*0.009;
end;

check;

stoch_simul(dr=cycle_reduction, order=1);
