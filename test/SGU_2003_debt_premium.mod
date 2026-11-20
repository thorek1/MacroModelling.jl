var 
a c ca_y d h i k r riskpremium tb_y util y lambda ;

varexo 
e ;

parameters 
dbar rbar alpha beta gamma delta rho sigma__tfp psi__2 omega phi ;

% Parameter definitions:
	gamma	=	2.0;
	omega	=	1.455;
	alpha	=	0.32;
	phi	=	0.028;
	rbar	=	0.04;
	delta	=	0.1;
	rho	=	0.42;
	sigma__tfp	=	0.0129;
	psi__2	=	0.000742;
	dbar	=	0.7442;
	beta = 1 / (1 + rbar);

model;
	d(0) = ((1 + r(-1)) * d(-1) - y(0)) + c(0) + i(0) + (phi / 2) * (k(0) - k(-1)) ^ 2;

	y(0) = exp(a(0)) * k(-1) ^ alpha * h(0) ^ (1 - alpha);

	k(0) = i(0) + k(-1) * (1 - delta);

	lambda(0) = beta * (1 + r(0)) * lambda(1);

	(c(0) - h(0) ^ omega / omega) ^ -gamma = lambda(0);

	(c(0) - h(0) ^ omega / omega) ^ -gamma * h(0) ^ (omega - 1) = (y(0) * (1 - alpha) * lambda(0)) / h(0);

	lambda(0) * (1 + phi * (k(0) - k(-1))) = beta * lambda(1) * (((1 + (alpha * y(1)) / k(0)) - delta) + phi * (k(1) - k(0)));

	a(0) = rho * a(-1) + sigma__tfp * e;

	r(0) = rbar + riskpremium(0);

	riskpremium(0) = psi__2 * (exp(d(0) - dbar) - 1);

	tb_y(0) = 1 - ((phi / 2) * (k(0) - k(-1)) ^ 2 + c(0) + i(0)) / y(0);

	ca_y(0) = (1 / y(0)) * (d(-1) - d(0));

	util(0) = ((c(0) - h(0) ^ omega * omega ^ -1) ^ (1 - gamma) - 1) / (1 - gamma);

end;

shocks;
var	e	=	1;
end;

initval;
	a	=	0.0;
	c	=	1.116950781911717;
	ca_y	=	0.0;
	d	=	0.7442000000000468;
	h	=	1.0074179936054592;
	i	=	0.3397685279738404;
	k	=	3.3976852797384067;
	r	=	0.040000000000000036;
	riskpremium	=	3.469446951953614e-17;
	tb_y	=	0.02002573436183175;
	util	=	-1.3683490243936651;
	y	=	1.4864873098855593;
	lambda	=	5.60907710134642;
end;

stoch_simul(order = 1, irf = 40);
