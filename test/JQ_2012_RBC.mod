var 
R b c d i k n r v w y z mu xi ;

varexo 
epsilon__x__i epsilon__z ;

parameters 
A_2_2 A_2__1 A_1__2 A_1_1 zbar alpha beta delta theta kappa xi_bar sigma sigma__x__i sigma__z tau ;

% Parameter definitions:
	BY_ratio	=	3.36;
	nbar	=	0.3;
	zbar	=	1.0;
	beta	=	0.9825;
	sigma	=	1.0;
	theta	=	0.36;
	delta	=	0.025;
	tau	=	0.35;
	kappa	=	0.146;
	A_1_1	=	0.9457;
	A_1__2	=	-0.0091;
	A_2__1	=	0.0321;
	A_2_2	=	0.9703;
	sigma__z	=	0.0045;
	sigma__x__i	=	0.0098;
	xi_bar	=	0.16337753022030044;
	alpha	=	1.8834086344418162;

model;
	w(0) / c(0) ^ sigma = alpha / (1 - n(0));

	c(0) ^ -sigma = ((beta * (R(0) - tau)) / (1 - tau)) * c(1) ^ -sigma;

	((w(0) * n(0) + b(-1)) - b(0) / R(0)) + d(0) = c(0);

	(1 - theta) * z(0) * k(-1) ^ theta * n(0) ^ -theta = w(0) / (1 - mu(0) * (1 + 2 * kappa * (d(0) - STEADY_STATE(d))));

	((beta * (c(0) / c(1)) ^ sigma * (1 + 2 * kappa * (d(0) - STEADY_STATE(d)))) / (1 + 2 * kappa * (d(1) - STEADY_STATE(d)))) * ((1 - delta) + theta * (1 - (1 + 2 * kappa * (d(1) - STEADY_STATE(d))) * mu(1)) * z(1) * k(0) ^ (theta - 1) * n(1) ^ (1 - theta)) + (1 + 2 * kappa * (d(0) - STEADY_STATE(d))) * mu(0) * xi(0) = 1;

	((1 + 2 * kappa * (d(0) - STEADY_STATE(d))) / (1 + 2 * kappa * (d(1) - STEADY_STATE(d)))) * (c(0) / c(1)) ^ sigma * beta * R(0) + ((1 + 2 * kappa * (d(0) - STEADY_STATE(d))) * mu(0) * xi(0) * R(0) * (1 - tau)) / (R(0) - tau) = 1;

	(((b(0) / R(0) + k(-1) * (1 - delta) + z(0) * k(-1) ^ theta * n(0) ^ (1 - theta)) - w(0) * n(0)) - b(-1)) - k(0) = d(0) + kappa * (d(0) - STEADY_STATE(d)) ^ 2;

	xi(0) * (k(0) - (b(0) * (1 - tau)) / (R(0) - tau)) = z(0) * k(-1) ^ theta * n(0) ^ (1 - theta);

	log(z(0) / zbar) = A_1_1 * log(z(-1) / zbar) + A_1__2 * log(xi(-1) / xi_bar) + sigma__z * epsilon__z;

	log(xi(0) / xi_bar) = log(z(-1) / zbar) * A_2__1 + log(xi(-1) / xi_bar) * A_2_2 + sigma__x__i * epsilon__x__i;

	y(0) = z(0) * k(-1) ^ theta * n(0) ^ (1 - theta);

	k(0) = k(-1) * (1 - delta) + i(0);

	v(0) = d(0) + ((c(0) * beta) / c(1)) * v(1);

	1 + r(0) = (R(0) - tau) / (1 - tau);

end;

shocks;
var	epsilon__x__i	=	1;
var	epsilon__z	=	1;
end;

initval;
	R	=	1.0115776081424936;
	b	=	3.6358259916618034;
	c	=	0.8111657946199857;
	d	=	0.11480054308526633;
	i	=	0.2519886806204083;
	k	=	10.07954722481633;
	n	=	0.3;
	r	=	0.017811704834605608;
	v	=	6.5600310334438054;
	w	=	2.182509516501626;
	y	=	1.0631544752403934;
	z	=	1.0;
	mu	=	0.03772089598850488;
	xi	=	0.16337753022030047;
end;

stoch_simul(order = 1, irf = 40);
