var 
R_k R_f SDF_plus__1 V c i k l s y z sigma ;

varexo 
omega epsilon__z ;

parameters 
beta gamma delta zeta eta lambda nu rho sigma_bar psi ;

% Parameter definitions:
	beta	=	0.991;
	zeta	=	0.3;
	delta	=	0.0196;
	lambda	=	0.95;
	psi	=	0.5;
	gamma	=	40.0;
	sigma_bar	=	0.021;
	eta	=	0.1;
	rho	=	0.9;
	nu	=	0.36218431417051217;

model;
	V(0) = ((1 - beta) * (c(0) ^ nu * (1 - l(0)) ^ (1 - nu)) ^ (1 - 1 / psi) + beta * V(1) ^ (1 - 1 / psi)) ^ (1 / (1 - 1 / psi));

	exp(s(0)) = V(1) ^ (1 - gamma);

	1 = (((1 + zeta * exp(z(1)) * k(0) ^ (zeta - 1) * l(1) ^ (1 - zeta)) - delta) * c(0) * beta * (((1 - l(1)) / (1 - l(0))) ^ (1 - nu) * (c(1) / c(0)) ^ nu) ^ (1 - 1 / psi)) / c(1);

	R_k(0) = zeta * exp(z(1)) * k(0) ^ (zeta - 1) * l(1) ^ (1 - zeta) - delta;

	SDF_plus__1(0) = (c(0) * beta * (((1 - l(1)) / (1 - l(0))) ^ (1 - nu) * (c(1) / c(0)) ^ nu) ^ (1 - 1 / psi)) / c(1);

	1 + R_f(0) = 1 / SDF_plus__1(0);

	(((1 - nu) / nu) * c(0)) / (1 - l(0)) = (1 - zeta) * exp(z(0)) * k(-1) ^ zeta * l(0) ^ -zeta;

	c(0) + i(0) = exp(z(0)) * k(-1) ^ zeta * l(0) ^ (1 - zeta);

	k(0) = i(0) + k(-1) * (1 - delta);

	z(0) = lambda * z(-1) + sigma(0) * epsilon__z;

	y(0) = exp(z(0)) * k(-1) ^ zeta * l(0) ^ (1 - zeta);

	log(sigma(0)) = (1 - rho) * log(sigma_bar) + rho * log(sigma(-1)) + eta * omega;

end;

shocks;
var	omega	=	1;
var	epsilon__z	=	1;
end;

initval;
	R_k	=	0.009081735620585375;
	R_f	=	0.009081735620585276;
	SDF_plus__1	=	0.991;
	V	=	0.687138657856569;
	c	=	0.7247305637488348;
	i	=	0.18688997126148366;
	k	=	9.53520261538182;
	l	=	0.3333333333333333;
	s	=	14.633547871166774;
	y	=	0.9116205350103185;
	z	=	0.0;
	sigma	=	0.021;
end;

stoch_simul(order = 1, irf = 40);
