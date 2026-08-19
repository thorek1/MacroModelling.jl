var 
a c deprec_rate n nx pi pih r real_wage rnat s x y ynat ystar ;

varexo 
epsilon__a epsilon__star ;

parameters 
Gamma Theta Psi alpha beta kappa__a rho__y rho__a sigma sigma__a omega phi phi__p__i ;

% Parameter definitions:
	sigma	=	1.0;
	eta	=	1.0;
	gamma	=	1.0;
	phi	=	3.0;
	theta	=	0.75;
	beta	=	0.99;
	alpha	=	0.4;
	phi__p__i	=	1.5;
	rho__a	=	0.9;
	rho__y	=	0.86;
	rho = 1 / beta - 1;
	omega = sigma * gamma + (1 - alpha) * (sigma * eta - 1);
	sigma__a = sigma / ((1 - alpha) + alpha * omega);
	Theta = ((1 - alpha) * (sigma * eta - 1) + sigma * gamma) - 1;
	lambda = ((1 - beta * theta) * (1 - theta)) / theta;
	kappa__a = lambda * (sigma__a + phi);
	Gamma = (1 + phi) / (sigma__a + phi);
	Psi = (-sigma__a * Theta) / (sigma__a + phi);

model;
	x(0) = x(1) - sigma__a ^ -1 * ((r(0) - pih(1)) - rnat(0));

	pih(0) = pih(1) * beta + x(0) * kappa__a;

	rnat(0) = -sigma__a * Gamma * (1 - rho__a) * a(0) + sigma__a * alpha * (Theta + Psi) * (ystar(1) - ystar(0));

	ynat(0) = Gamma * a(0) + ystar(0) * alpha * Psi;

	x(0) = y(0) - ynat(0);

	y(0) = ystar(0) + sigma__a ^ -1 * s(0);

	pi(0) = pih(0) + alpha * (s(0) - s(-1));

	s(0) = (s(-1) + deprec_rate(0)) - pih(0);

	y(0) = a(0) + n(0);

	nx(0) = s(0) * alpha * (omega / sigma - 1);

	y(0) = c(0) + (s(0) * alpha * omega) / sigma;

	real_wage(0) = sigma * c(0) + n(0) * phi;

	a(0) = rho__a * a(-1) + epsilon__a;

	ystar(0) = rho__y * ystar(-1) + epsilon__star;

	r(0) = pi(0) * phi__p__i;

end;

shocks;
var	epsilon__a	=	1;
var	epsilon__star	=	1;
end;

initval;
	a	=	0.0;
	c	=	0.0;
	deprec_rate	=	0.0;
	n	=	0.0;
	nx	=	0.0;
	pi	=	0.0;
	pih	=	0.0;
	r	=	0.0;
	real_wage	=	0.0;
	rnat	=	0.0;
	s	=	0.0;
	x	=	0.0;
	y	=	0.0;
	ynat	=	0.0;
	ystar	=	0.0;
end;

stoch_simul(order = 1, irf = 40);
