var 
c check_walras fk fn gb iv k l n q r tr uc ul w y lambda ;

varexo 
e_gb ;

parameters 
A GB_BAR beta gamma__x delta__k theta__k theta__l theta__n tau ;

% Parameter definitions:
	A	=	1.0;
	gamma__x	=	1.016;
	theta__k	=	0.42;
	delta__k	=	0.1;
	N	=	0.2;
	R	=	0.065;
	sG	=	0.2;
	tau_BAR	=	0.2;
	tau	=	0.2;
	theta__n = 1 - theta__k;
	L = 1 - N;
	beta = gamma__x / (1 + R);
	Q = (gamma__x / beta - 1) + delta__k;
	FK = Q / (1 - tau_BAR);
	K = (FK / (theta__k * A * N ^ theta__n)) ^ (1 / (theta__k - 1));
	FN = theta__n * A * K ^ theta__k * N ^ (theta__n - 1);
	IV = ((gamma__x - 1) + delta__k) * K;
	Y = A * N ^ (1 - theta__k) * K ^ theta__k;
	GB_BAR = sG * Y;
	C = (Y - IV) - GB_BAR;
	UC = C ^ -1;
	UL = UC * (1 - tau_BAR) * FN;
	theta__l = UL * L;

model;
	uc(0) = c(0) ^ -1;

	ul(0) = theta__l * l(0) ^ -1;

	y(0) = A * k(-1) ^ theta__k * n(0) ^ theta__n;

	fk(0) = theta__k * A * k(-1) ^ (theta__k - 1) * n(0) ^ theta__n;

	fn(0) = theta__n * A * k(-1) ^ theta__k * n(0) ^ (theta__n - 1);

	gamma__x * k(0) = (1 - delta__k) * k(-1) + iv(0);

	l(0) + n(0) = 1;

	c(0) + iv(0) = (1 - tau) * y(0) + tr(0) + check_walras(0);

	c(0) + iv(0) + gb(0) = y(0);

	tau * y(0) = gb(0) + tr(0);

	uc(0) = lambda(0);

	ul(0) = lambda(0) * (1 - tau) * fn(0);

	beta * lambda(1) * ((q(1) + 1) - delta__k) = gamma__x * lambda(0);

	q(0) = (1 - tau) * fk(0);

	gb(0) = GB_BAR + e_gb;

	1 + r(0) = (gamma__x * lambda(0)) / (lambda(1) * beta);

	w(0) = fn(0);

end;

shocks;
var	e_gb	=	1;
end;

initval;
	c	=	0.18871000385176429;
	check_walras	=	8.326672684688674e-17;
	fk	=	0.20625000000000004;
	fn	=	0.9706929055197501;
	gb	=	0.0669443383117069;
	iv	=	0.07906734939506319;
	k	=	0.6816150809919241;
	l	=	0.8000000000000002;
	n	=	0.19999999999999993;
	q	=	0.16500000000000004;
	r	=	0.06499999999999995;
	tr	=	-4.163336342344337e-17;
	uc	=	5.299136132631958;
	ul	=	4.1150670794633655;
	w	=	0.9706929055197501;
	y	=	0.3347216915585343;
	lambda	=	5.299136132631958;
end;

stoch_simul(order = 1, irf = 40);
