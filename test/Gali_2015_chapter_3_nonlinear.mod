var 
A C MC M_real N Pi Pi_star Q R S W_real Y Z i_ann log_N log_W_real log_y nu pi_ann r_real_ann realinterest x_aux_1 x_aux_2 ;

varexo 
eps_a eps_nu eps_z ;

parameters 
std_a std_nu std_z alpha beta eta theta rho__a rho__z rho__nu sigma tau varphi phi__y phi__p__i epsilon ;

% Parameter definitions:
	sigma	=	1.0;
	varphi	=	5.0;
	phi__p__i	=	1.5;
	phi__y	=	0.125;
	theta	=	0.75;
	rho__nu	=	0.5;
	rho__z	=	0.5;
	rho__a	=	0.9;
	beta	=	0.99;
	eta	=	3.77;
	alpha	=	0.25;
	epsilon	=	9.0;
	tau	=	0.0;
	std_a	=	0.01;
	std_z	=	0.05;
	std_nu	=	0.0025;

model;
	W_real(0) = C(0) ^ sigma * N(0) ^ varphi;

	Q(0) = ((beta * (C(1) / C(0)) ^ -sigma * Z(1)) / Z(0)) / Pi(1);

	R(0) = 1 / Q(0);

	Y(0) = A(0) * (N(0) / S(0)) ^ (1 - alpha);

	R(0) = Pi(1) * realinterest(0);

	R(0) = (1 / beta) * Pi(0) ^ phi__p__i * (Y(0) / STEADY_STATE(Y)) ^ phi__y * exp(nu(0));

	C(0) = Y(0);

	log(A(0)) = rho__a * log(A(-1)) + std_a * eps_a;

	log(Z(0)) = rho__z * log(Z(-1)) - std_z * eps_z;

	nu(0) = rho__nu * nu(-1) + std_nu * eps_nu;

	MC(0) = W_real(0) / ((S(0) * Y(0) * (1 - alpha)) / N(0));

	1 = theta * Pi(0) ^ (epsilon - 1) + (1 - theta) * Pi_star(0) ^ (1 - epsilon);

	S(0) = (1 - theta) * Pi_star(0) ^ (-epsilon / (1 - alpha)) + theta * Pi(0) ^ (epsilon / (1 - alpha)) * S(-1);

	Pi_star(0) ^ (1 + (epsilon * alpha) / (1 - alpha)) = (((epsilon * x_aux_1(0)) / x_aux_2(0)) * (1 - tau)) / (epsilon - 1);

	x_aux_1(0) = MC(0) * Y(0) * Z(0) * C(0) ^ -sigma + beta * theta * Pi(1) ^ (epsilon + (alpha * epsilon) / (1 - alpha)) * x_aux_1(1);

	x_aux_2(0) = Y(0) * Z(0) * C(0) ^ -sigma + beta * theta * Pi(1) ^ (epsilon - 1) * x_aux_2(1);

	log_y(0) = log(Y(0));

	log_W_real(0) = log(W_real(0));

	log_N(0) = log(N(0));

	pi_ann(0) = 4 * log(Pi(0));

	i_ann(0) = 4 * log(R(0));

	r_real_ann(0) = 4 * log(realinterest(0));

	M_real(0) = Y(0) / R(0) ^ eta;

end;

shocks;
var	eps_a	=	1;
var	eps_nu	=	1;
var	eps_z	=	1;
end;

initval;
	A	=	1.0;
	C	=	0.9505798249541406;
	MC	=	0.8888888888888884;
	M_real	=	0.915236383286892;
	N	=	0.934655265184067;
	Pi	=	0.9999999999999996;
	Pi_star	=	0.9999999999999987;
	Q	=	0.9900000000000004;
	R	=	1.0101010101010095;
	S	=	1.0;
	W_real	=	0.6780252644037242;
	Y	=	0.9505798249541406;
	Z	=	1.0;
	i_ann	=	0.04020134341400339;
	log_N	=	-0.06757751801802749;
	log_W_real	=	-0.3885707286036581;
	log_y	=	-0.050683138513520666;
	nu	=	0.0;
	pi_ann	=	-1.776356839400251e-15;
	r_real_ann	=	0.04020134341400514;
	realinterest	=	1.01010101010101;
	x_aux_1	=	3.4519956850053397;
	x_aux_2	=	3.8834951456310276;
end;

stoch_simul(order = 1, irf = 40);
