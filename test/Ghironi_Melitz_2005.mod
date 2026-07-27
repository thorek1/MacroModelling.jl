var 
C Cbar Nd Ndbar Ne Nx Nxbar Nebar Q Qtilde TOL Z Zbar dtilde dtilde_d dtilde_dbar dtilde_x dtilde_xbar dtilde_bar r rbar w wbar zx zxbar ztilde_x ztilde_xbar rho_tilde_d rho_tilde_dbar rho_tilde_x rho_tilde_xbar vtilde vtilde_bar ;

varexo 
epsilon__z epsilon__z_bar ;

parameters 
L Lbar fe fx fxbar febar k zmin zminbar ztilde_d ztilde_dbar beta gamma delta theta rho_Z rho_Zbar sigma__z sigma__z_bar tau ;

% Parameter definitions:
	sigma__z	=	0.01;
	sigma__z_bar	=	0.01;
	beta	=	0.99;
	gamma	=	2.0;
	delta	=	0.025;
	theta	=	3.8;
	k	=	3.4;
	tau	=	1.3;
	zmin	=	1.0;
	zminbar	=	1.0;
	fe	=	1.0;
	febar	=	1.0;
	L	=	1.0;
	Lbar	=	1.0;
	rho_Z	=	0.9;
	rho_Zbar	=	0.9;
	fx_share	=	0.235;
	fx = ((fx_share * (1 - beta * (1 - delta))) / (beta * (1 - delta))) * fe;
	fxbar = ((fx_share * (1 - beta * (1 - delta))) / (beta * (1 - delta))) * febar;
	ztilde_d = (k / (k - (theta - 1))) ^ (1 / (theta - 1)) * zmin;
	ztilde_dbar = (k / (k - (theta - 1))) ^ (1 / (theta - 1)) * zminbar;

model;
	1 = Nd(0) * rho_tilde_d(0) ^ (1 - theta) + Nxbar(0) * rho_tilde_xbar(0) ^ (1 - theta);

	1 = Ndbar(0) * rho_tilde_dbar(0) ^ (1 - theta) + Nx(0) * rho_tilde_x(0) ^ (1 - theta);

	rho_tilde_d(0) = ((theta / (theta - 1)) * w(0)) / (Z(0) * ztilde_d);

	rho_tilde_dbar(0) = ((theta / (theta - 1)) * wbar(0)) / (Zbar(0) * ztilde_dbar);

	rho_tilde_x(0) = (((theta / (theta - 1)) * tau * w(0)) / (Z(0) * ztilde_x(0))) / Q(0);

	rho_tilde_xbar(0) = (((Q(0) * theta) / (theta - 1)) * tau * wbar(0)) / (Zbar(0) * ztilde_xbar(0));

	dtilde(0) = dtilde_d(0) + (Nx(0) / Nd(0)) * dtilde_x(0);

	dtilde_bar(0) = dtilde_dbar(0) + (Nxbar(0) / Ndbar(0)) * dtilde_xbar(0);

	dtilde_d(0) = ((rho_tilde_d(0) ^ (1 - theta) * 1) / theta) * C(0);

	dtilde_dbar(0) = ((rho_tilde_dbar(0) ^ (1 - theta) * 1) / theta) * Cbar(0);

	vtilde(0) = (w(0) * fe) / Z(0);

	vtilde_bar(0) = (wbar(0) * febar) / Zbar(0);

	dtilde_x(0) = (((w(0) * fx) / Z(0)) * (theta - 1)) / (k - (theta - 1));

	dtilde_xbar(0) = (((theta - 1) / (k - (theta - 1))) * wbar(0) * fxbar) / Zbar(0);

	Nx(0) / Nd(0) = (zmin / ztilde_x(0)) ^ k * (k / (k - (theta - 1))) ^ (k / (theta - 1));

	Nxbar(0) / Ndbar(0) = (k / (k - (theta - 1))) ^ (k / (theta - 1)) * (zminbar / ztilde_xbar(0)) ^ k;

	Nd(0) = (1 - delta) * (Nd(-1) + Ne(-1));

	Ndbar(0) = (1 - delta) * (Ndbar(-1) + Nebar(-1));

	C(0) ^ -gamma = beta * (1 + r(0)) * C(1) ^ -gamma;

	Cbar(0) ^ -gamma = beta * (1 + rbar(0)) * Cbar(1) ^ -gamma;

	vtilde(0) = (1 - delta) * beta * (C(1) / C(0)) ^ -gamma * (vtilde(1) + dtilde(1));

	vtilde_bar(0) = (1 - delta) * beta * (Cbar(1) / Cbar(0)) ^ -gamma * (vtilde_bar(1) + dtilde_bar(1));

	C(0) = (w(0) * L + Nd(0) * dtilde(0)) - vtilde(0) * Ne(0);

	Cbar(0) = (wbar(0) * Lbar + Ndbar(0) * dtilde_bar(0)) - vtilde_bar(0) * Nebar(0);

	Q(0) = (Nxbar(0) * rho_tilde_xbar(0) ^ (1 - theta) * C(0)) / (Nx(0) * rho_tilde_x(0) ^ (1 - theta) * Cbar(0));

	Qtilde(0) = (((Ndbar(0) / (Ndbar(0) + Nx(0))) * TOL(0) ^ (1 - theta) + (Nx(0) / (Ndbar(0) + Nx(0))) * ((tau * ztilde_d) / ztilde_x(0)) ^ (1 - theta)) / (Nd(0) / (Nd(0) + Nxbar(0)) + (Nxbar(0) / (Nd(0) + Nxbar(0))) * ((tau * TOL(0) * ztilde_dbar) / ztilde_xbar(0)) ^ (1 - theta))) ^ (1 / (1 - theta));

	Qtilde(0) = Q(0) * ((Nd(0) + Nxbar(0)) / (Ndbar(0) + Nx(0))) ^ (-1 / (theta - 1));

	Z(0) = (1 - rho_Z) * 1.0 + rho_Z * Z(-1) + sigma__z * epsilon__z;

	Zbar(0) = 1.0 * (1 - rho_Zbar) + rho_Zbar * Zbar(-1) + sigma__z_bar * epsilon__z_bar;

	ztilde_x(0) = (theta * fx * (w(0) / Z(0)) ^ theta * (1 + (theta - 1) / (k - (theta - 1))) * Q(0) ^ -theta * tau ^ (theta - 1) * (theta / (theta - 1)) ^ (theta - 1) * Cbar(0) ^ -1) ^ (1 / (theta - 1));

	ztilde_xbar(0) = ((theta / (theta - 1)) ^ (theta - 1) * theta * tau ^ (theta - 1) * (1 + (theta - 1) / (k - (theta - 1))) * fxbar * (wbar(0) / Zbar(0)) ^ theta * Q(0) ^ theta * C(0) ^ -1) ^ (1 / (theta - 1));

	zx(0) = ztilde_x(0) / (k / (k - (theta - 1))) ^ (1 / (theta - 1));

	zxbar(0) = ztilde_xbar(0) / (k / (k - (theta - 1))) ^ (1 / (theta - 1));

end;

shocks;
var	epsilon__z	=	1;
var	epsilon__z_bar	=	1;
end;

initval;
	C	=	3.386882407731392;
	Cbar	=	3.38688240773139;
	Nd	=	7.506952650706411;
	Ndbar	=	7.506952650706394;
	Ne	=	0.19248596540272866;
	Nx	=	1.5798796065733534;
	Nxbar	=	1.5798796065733551;
	Nebar	=	0.1924859654027283;
	Q	=	1.0000000000000002;
	Qtilde	=	0.9999999999999993;
	TOL	=	0.9999999999999992;
	Z	=	1.0;
	Zbar	=	1.0;
	dtilde	=	0.11313270650973085;
	dtilde_d	=	0.0870217286331077;
	dtilde_dbar	=	0.08702172863310784;
	dtilde_x	=	0.12406886813900507;
	dtilde_xbar	=	0.12406886813900495;
	dtilde_bar	=	0.11313270650973105;
	r	=	0.01010101010101011;
	rbar	=	0.01010101010101011;
	w	=	3.142484747007705;
	wbar	=	3.142484747007702;
	zx	=	1.58150472992854;
	zxbar	=	1.5815047299285385;
	ztilde_x	=	2.938435014025518;
	ztilde_xbar	=	2.9384350140255155;
	rho_tilde_d	=	2.2953723636801207;
	rho_tilde_dbar	=	2.295372363680119;
	rho_tilde_x	=	1.8868005996535888;
	rho_tilde_xbar	=	1.8868005996535897;
	vtilde	=	3.142484747007705;
	vtilde_bar	=	3.142484747007702;
end;

stoch_simul(order = 1, irf = 40);
