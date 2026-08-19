var 
A A_tilde Average_markup MC_real Marginal_markup N Utility i p_star phi pi price_adjustment_gap psi real_interest s v w y zeta ;

varexo 
e_a e_v e_zeta ;

parameters 
Pi_bar Y_bar alpha beta d_n epsilon i_bar phi_par phi_pi phi_y rho_a rho_i rho_v rho_zeta sigma theta var_rho sigma__zeta sigma__v sigma__a ;

% Parameter definitions:
	beta	=	0.99;
	trend_inflation	=	0.0;
	alpha	=	0.0;
	theta	=	0.75;
	epsilon	=	10.0;
	sigma	=	1.0;
	rho_v	=	0.0;
	rho_a	=	0.0;
	rho_zeta	=	0.0;
	phi_par	=	1.0;
	phi_pi	=	2.0;
	phi_y	=	0.125;
	rho_i	=	0.8;
	var_rho	=	0.0;
	sigma__zeta	=	0.01;
	sigma__a	=	0.01;
	sigma__v	=	0.01;
	d_n	=	8.09999999470471;
	Y_bar	=	0.33333331837004737;
	Pi_bar = (1 + trend_inflation / 100) ^ (1 / 4);
	i_bar = Pi_bar / beta - 1;

model;
	1 / y(0) ^ sigma = (beta * (1 + i(0))) / (pi(1) * y(1) ^ sigma);

	w(0) = y(0) ^ sigma * d_n * exp(zeta(0)) * N(0) ^ phi_par;

	p_star(0) = ((1 - theta * pi(-1) ^ ((1 - epsilon) * var_rho) * pi(0) ^ (epsilon - 1)) / (1 - theta)) ^ (1 / (1 - epsilon));

	p_star(0) ^ (1 + (epsilon * alpha) / (1 - alpha)) = ((epsilon / ((epsilon - 1) * (1 - alpha))) * psi(0)) / phi(0);

	psi(0) = w(0) * exp(A(0)) ^ (-1 / (1 - alpha)) * y(0) ^ (1 / (1 - alpha) - sigma) + beta * theta * pi(0) ^ ((epsilon * -var_rho) / (1 - alpha)) * pi(1) ^ (epsilon / (1 - alpha)) * psi(1);

	phi(0) = y(0) ^ (1 - sigma) + beta * theta * pi(0) ^ ((1 - epsilon) * var_rho) * pi(1) ^ (epsilon - 1) * phi(1);

	N(0) = s(0) * (y(0) / exp(A(0))) ^ (1 / (1 - alpha));

	s(0) = (1 - theta) * p_star(0) ^ (-epsilon / (1 - alpha)) + theta * pi(-1) ^ ((var_rho * -epsilon) / (1 - alpha)) * pi(0) ^ (epsilon / (1 - alpha)) * s(-1);

	(1 + i(0)) / (1 + i_bar) = ((1 + i(-1)) / (1 + i_bar)) ^ rho_i * ((pi(0) / Pi_bar) ^ phi_pi * (y(0) / Y_bar) ^ phi_y) ^ (1 - rho_i) * exp(v(0));

	MC_real(0) = ((w(0) * 1) / (1 - alpha)) * exp(A(0)) ^ (1 / (alpha - 1)) * y(0) ^ (alpha / (1 - alpha));

	real_interest(0) = (1 + i(0)) / pi(1);

	Utility(0) = (log(y(0)) - (d_n * exp(zeta(0)) * N(0) ^ (1 + phi_par)) / (1 + phi_par)) + beta * Utility(1);

	v(0) = rho_v * v(-1) + sigma__v * e_v;

	A(0) = rho_a * A(-1) + sigma__a * e_a;

	zeta(0) = rho_zeta * zeta(-1) + sigma__zeta * e_zeta;

	A_tilde(0) = exp(A(0)) / s(0);

	Average_markup(0) = 1 / MC_real(0);

	Marginal_markup(0) = p_star(0) / MC_real(0);

	price_adjustment_gap(0) = 1 / p_star(0);

end;

shocks;
var	e_a	=	1;
var	e_v	=	1;
var	e_zeta	=	1;
end;

initval;
	A	=	0.0;
	A_tilde	=	1.000000000000001;
	Average_markup	=	1.1111111118374883;
	MC_real	=	0.8999999994116344;
	Marginal_markup	=	1.111111093133382;
	N	=	0.3333333333333333;
	Utility	=	-154.86122883739253;
	i	=	0.010101004433099076;
	p_star	=	0.9999999831663043;
	phi	=	3.883494580117997;
	pi	=	0.9999999943887681;
	price_adjustment_gap	=	1.000000016833696;
	psi	=	3.4951450632699883;
	real_interest	=	1.0101010101010102;
	s	=	0.9999999999999989;
	v	=	0.0;
	w	=	0.8999999994116344;
	y	=	0.3333333333333333;
	zeta	=	0.0;
end;

stoch_simul(order = 1, irf = 40);
