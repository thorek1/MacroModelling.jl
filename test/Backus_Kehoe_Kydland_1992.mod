var 
A__F__ A__H__ C__F__ C__H__ K__F__ K__H__ LAMBDA__F__ LAMBDA__H__ LGM L__F__ L__H__ NX__F__ NX__H__ N__F__ N__H__ S__F__ S__H__ U__F__ U__H__ X__F__ X__H__ Y__F__ Y__H__ Z__F__ Z__H__ dLGM dLGM_ann ;

varexo 
E__F__ E__H__ ;

parameters 
Z_E__F__ Z_E__H__ alpha__F__ alpha__H__ beta__F__ beta__H__ delta__F__ delta__H__ eta__F__ eta__H__ gamma__F__ gamma__H__ mu__F__ mu__H__ nu__F__ nu__H__ phi__F__ phi__H__ psi__F__ psi__H__ rho__F____F__ rho__F____H__ rho__H____F__ rho__H____H__ sigma__F__ sigma__H__ theta__F__ theta__H__ ;

% Parameter definitions:
	K_ss	=	11.0148;
	F_H_ratio	=	1.0;
	mu__F__	=	0.34;
	mu__H__	=	0.34;
	gamma__F__	=	-1.0;
	gamma__H__	=	-1.0;
	alpha__F__	=	1.0;
	alpha__H__	=	1.0;
	eta__F__	=	0.5;
	eta__H__	=	0.5;
	theta__F__	=	0.36;
	theta__H__	=	0.36;
	nu__F__	=	3.0;
	nu__H__	=	3.0;
	sigma__F__	=	0.01;
	sigma__H__	=	0.01;
	delta__F__	=	0.025;
	delta__H__	=	0.025;
	psi__F__	=	0.5;
	psi__H__	=	0.5;
	Z_E__F__	=	0.00852;
	Z_E__H__	=	0.00852;
	rho__H____H__	=	0.906;
	rho__H____F__	=	0.088;
	phi__F__	=	0.25;
	phi__H__	=	0.25;
	beta__F__	=	0.9899998184488822;
	beta__H__	=	0.989999818448882;
	rho__F____F__ = rho__H____H__;
	rho__F____H__ = rho__H____F__;

model;
	Y__H__(0) = ((LAMBDA__H__(0) * K__H__(-4) ^ theta__H__ * N__H__(0) ^ (1 - theta__H__)) ^ -nu__H__ + sigma__H__ * Z__H__(-1) ^ -nu__H__) ^ (-1 / nu__H__);

	K__H__(0) = (1 - delta__H__) * K__H__(-1) + S__H__(0);

	X__H__(0) = phi__H__ * S__H__(-3) + phi__H__ * S__H__(-2) + phi__H__ * S__H__(-1) + phi__H__ * S__H__(0);

	A__H__(0) = (1 - eta__H__) * A__H__(-1) + N__H__(0);

	L__H__(0) = (1 - alpha__H__ * N__H__(0)) - (1 - alpha__H__) * eta__H__ * A__H__(-1);

	U__H__(0) = (C__H__(0) ^ mu__H__ * L__H__(0) ^ (1 - mu__H__)) ^ gamma__H__;

	((psi__H__ * mu__H__) / C__H__(0)) * U__H__(0) = LGM(0);

	((psi__H__ * (1 - mu__H__)) / L__H__(0)) * U__H__(0) * -alpha__H__ = ((-(LGM(0)) * (1 - theta__H__)) / N__H__(0)) * (LAMBDA__H__(0) * K__H__(-4) ^ theta__H__ * N__H__(0) ^ (1 - theta__H__)) ^ -nu__H__ * Y__H__(0) ^ (1 + nu__H__);

	(beta__H__ ^ 0 * LGM(0) * phi__H__ + beta__H__ ^ 1 * LGM(1) * phi__H__ + beta__H__ ^ 2 * LGM(2) * phi__H__ + beta__H__ ^ 3 * LGM(3) * phi__H__) + (-(beta__H__ ^ 1) * LGM(1) * phi__H__ * (1 - delta__H__) + -(beta__H__ ^ 2) * LGM(2) * phi__H__ * (1 - delta__H__) + -(beta__H__ ^ 3) * LGM(3) * phi__H__ * (1 - delta__H__) + -(beta__H__ ^ 4) * LGM(4) * phi__H__ * (1 - delta__H__)) = ((beta__H__ ^ 4 * LGM(4) * theta__H__) / K__H__(0)) * (LAMBDA__H__(4) * K__H__(0) ^ theta__H__ * N__H__(4) ^ (1 - theta__H__)) ^ -nu__H__ * Y__H__(4) ^ (1 + nu__H__);

	LGM(0) = beta__H__ * LGM(1) * (1 + sigma__H__ * Z__H__(0) ^ (-nu__H__ - 1) * Y__H__(1) ^ (1 + nu__H__));

	NX__H__(0) = (Y__H__(0) - ((C__H__(0) + X__H__(0) + Z__H__(0)) - Z__H__(-1))) / Y__H__(0);

	Y__F__(0) = ((LAMBDA__F__(0) * K__F__(-4) ^ theta__F__ * N__F__(0) ^ (1 - theta__F__)) ^ -nu__F__ + sigma__F__ * Z__F__(-1) ^ -nu__F__) ^ (-1 / nu__F__);

	K__F__(0) = (1 - delta__F__) * K__F__(-1) + S__F__(0);

	X__F__(0) = phi__F__ * S__F__(-3) + phi__F__ * S__F__(-2) + phi__F__ * S__F__(-1) + phi__F__ * S__F__(0);

	A__F__(0) = (1 - eta__F__) * A__F__(-1) + N__F__(0);

	L__F__(0) = (1 - alpha__F__ * N__F__(0)) - (1 - alpha__F__) * eta__F__ * A__F__(-1);

	U__F__(0) = (C__F__(0) ^ mu__F__ * L__F__(0) ^ (1 - mu__F__)) ^ gamma__F__;

	((psi__F__ * mu__F__) / C__F__(0)) * U__F__(0) = LGM(0);

	((psi__F__ * (1 - mu__F__)) / L__F__(0)) * U__F__(0) * -alpha__F__ = ((-(LGM(0)) * (1 - theta__F__)) / N__F__(0)) * (LAMBDA__F__(0) * K__F__(-4) ^ theta__F__ * N__F__(0) ^ (1 - theta__F__)) ^ -nu__F__ * Y__F__(0) ^ (1 + nu__F__);

	(beta__F__ ^ 0 * LGM(0) * phi__F__ + beta__F__ ^ 1 * LGM(1) * phi__F__ + beta__F__ ^ 2 * LGM(2) * phi__F__ + beta__F__ ^ 3 * LGM(3) * phi__F__) + (-(beta__F__ ^ 1) * LGM(1) * phi__F__ * (1 - delta__F__) + -(beta__F__ ^ 2) * LGM(2) * phi__F__ * (1 - delta__F__) + -(beta__F__ ^ 3) * LGM(3) * phi__F__ * (1 - delta__F__) + -(beta__F__ ^ 4) * LGM(4) * phi__F__ * (1 - delta__F__)) = ((beta__F__ ^ 4 * LGM(4) * theta__F__) / K__F__(0)) * (LAMBDA__F__(4) * K__F__(0) ^ theta__F__ * N__F__(4) ^ (1 - theta__F__)) ^ -nu__F__ * Y__F__(4) ^ (1 + nu__F__);

	LGM(0) = beta__F__ * LGM(1) * (1 + sigma__F__ * Z__F__(0) ^ (-nu__F__ - 1) * Y__F__(1) ^ (1 + nu__F__));

	NX__F__(0) = (Y__F__(0) - ((C__F__(0) + X__F__(0) + Z__F__(0)) - Z__F__(-1))) / Y__F__(0);

	LAMBDA__H__(0) - 1 = rho__H____H__ * (LAMBDA__H__(-1) - 1) + rho__H____F__ * (LAMBDA__F__(-1) - 1) + Z_E__H__ * E__H__;

	LAMBDA__F__(0) - 1 = rho__F____F__ * (LAMBDA__F__(-1) - 1) + rho__F____H__ * (LAMBDA__H__(-1) - 1) + Z_E__F__ * E__F__;

	((C__H__(0) + X__H__(0) + Z__H__(0)) - Z__H__(-1)) + ((C__F__(0) + X__F__(0) + Z__F__(0)) - Z__F__(-1)) = Y__H__(0) + Y__F__(0);

	dLGM(0) = LGM(1) / LGM(0);

	dLGM_ann(0) = dLGM(-3) * dLGM(-2) * dLGM(-1) * dLGM(0);

end;

shocks;
var	E__F__	=	1;
var	E__H__	=	1;
end;

initval;
	A__F__	=	0.6064361690984907;
	A__H__	=	0.6064361690984906;
	C__F__	=	0.8260902429883573;
	C__H__	=	0.8260902429883576;
	K__F__	=	11.0148;
	K__H__	=	11.0148;
	LAMBDA__F__	=	0.9999999999999988;
	LAMBDA__H__	=	0.9999999999999989;
	LGM	=	0.2787328444414798;
	L__F__	=	0.6967819154507546;
	L__H__	=	0.6967819154507549;
	NX__F__	=	2.0159112082212333e-16;
	NX__H__	=	-1.0079556041106167e-16;
	N__F__	=	0.30321808454924537;
	N__H__	=	0.30321808454924526;
	S__F__	=	0.27537;
	S__H__	=	0.27537;
	U__F__	=	1.3544616658441064;
	U__H__	=	1.3544616658441064;
	X__F__	=	0.27537;
	X__H__	=	0.27537;
	Y__F__	=	1.1014602429883575;
	Y__H__	=	1.1014602429883575;
	Z__F__	=	1.0986911684853575;
	Z__H__	=	1.0986911684853493;
	dLGM	=	1.0;
	dLGM_ann	=	1.0;
end;

stoch_simul(order = 1, irf = 40);
