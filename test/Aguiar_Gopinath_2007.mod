var 
b c c_y delta_y g i_y invest k l nx q u uc ul y z ;

varexo 
eps_g eps_z ;

parameters 
alpha b_star beta delta gamma mu_g phi psi r_star rho_g rho_z sigma sigma__g sigma__z ;

% Parameter definitions:
	gamma	=	0.36;
	b_share	=	0.1;
	psi	=	0.001;
	alpha	=	0.68;
	sigma	=	2.0;
	delta	=	0.05;
	phi	=	4.0;
	rho_z	=	0.95;
	rho_g	=	0.01;
	sigma__z	=	0.01;
	sigma__g	=	0.0005;
	beta	=	0.9803921568627451;
	mu_g	=	0.006578315360122507;
	b_star	=	0.06451768392329371;
	r_star	=	0.02916638148458217;

model;
	y(0) = (exp(g(0)) * l(0)) ^ alpha * exp(z(0)) * k(-1) ^ (1 - alpha);

	z(0) = rho_z * z(-1) + sigma__z * eps_z;

	g(0) = (1 - rho_g) * mu_g + rho_g * g(-1) + sigma__g * eps_g;

	u(0) = (c(0) ^ gamma * (1 - l(0)) ^ (1 - gamma)) ^ (1 - sigma) / (1 - sigma);

	uc(0) = ((1 - sigma) * u(0) * gamma) / c(0);

	ul(0) = ((1 - sigma) * u(0) * -((1 - gamma))) / (1 - l(0));

	c(0) + k(0) * exp(g(0)) = (((y(0) + (1 - delta) * k(-1)) - ((k(-1) * phi) / 2) * ((k(0) * exp(g(0))) / k(-1) - exp(mu_g)) ^ 2) - b(-1)) + b(0) * exp(g(0)) * q(0);

	1 / q(0) = 1 + r_star + psi * (exp(b(0) - b_star) - 1);

	exp(g(0)) * uc(0) * (1 + phi * ((k(0) * exp(g(0))) / k(-1) - exp(mu_g))) = beta * exp(g(0) * gamma * (1 - sigma)) * uc(1) * (((1 - delta) + ((1 - alpha) * y(1)) / k(0)) - (phi / 2) * ((k(1) * exp(g(1)) * -(2 * ((k(1) * exp(g(1))) / k(0) - exp(mu_g)))) / k(0) + ((k(1) * exp(g(1))) / k(0) - exp(mu_g)) ^ 2));

	ul(0) + (y(0) * alpha * uc(0)) / l(0) = 0;

	q(0) * exp(g(0)) * uc(0) = beta * exp(g(0) * gamma * (1 - sigma)) * uc(1);

	invest(0) = (((k(-1) * phi) / 2) * ((k(0) * exp(g(0))) / k(-1) - exp(mu_g)) ^ 2 + k(0) * exp(g(0))) - (1 - delta) * k(-1);

	c_y(0) = c(0) / y(0);

	i_y(0) = invest(0) / y(0);

	nx(0) = (b(-1) - b(0) * exp(g(0)) * q(0)) / y(0);

	delta_y(0) = (g(-1) + log(y(0))) - log(y(-1));

end;

shocks;
var	eps_g	=	1;
var	eps_z	=	1;
end;

initval;
	b	=	0.06451768392329393;
	c	=	0.49615604296422694;
	c_y	=	0.7690233325085201;
	delta_y	=	0.006578315360122507;
	g	=	0.006578315360122507;
	i_y	=	0.22878398204327838;
	invest	=	0.14760612640180737;
	k	=	2.6078820919047296;
	l	=	0.3321686927235319;
	nx	=	0.0021926854482003325;
	q	=	0.9716601882753794;
	u	=	-1.6664438374559871;
	uc	=	1.209135291187837;
	ul	=	-1.5969961940258566;
	y	=	0.645176839232937;
	z	=	0.0;
end;

stoch_simul(order = 1, irf = 40);
