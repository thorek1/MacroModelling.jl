using MacroModelling

@model Backus_Kehoe_Kydland_1992 begin
	Y__H__[0] = (sigma__H__ * Z__H__[-1] ^ (-nu__H__) + (N__H__[0] ^ (1 - theta__H__) * LAMBDA__H__[0] * AUX_ENDO_LAG_5_3[-1] ^ theta__H__) ^ (-nu__H__)) ^ (( - 1) / nu__H__)

	K__H__[0] = (1 - delta__H__) * K__H__[-1] + S__H__[0]

	X__H__[0] = S__H__[0] * phi__H__ + phi__H__ * S__H__[-1] + phi__H__ * AUX_ENDO_LAG_16_2[-1] + phi__H__ * AUX_ENDO_LAG_16_1[-1]

	A__H__[0] = N__H__[0] + (1 - eta__H__) * A__H__[-1]

	L__H__[0] = 1 - N__H__[0] * alpha__H__ - A__H__[-1] * eta__H__ * (1 - alpha__H__)

	U__H__[0] = (C__H__[0] ^ mu__H__ * L__H__[0] ^ (1 - mu__H__)) ^ gamma__H__

	U__H__[0] * mu__H__ * psi__H__ / C__H__[0] = LGM[0]

	U__H__[0] * (1 - mu__H__) * psi__H__ / L__H__[0] * ( - alpha__H__) = Y__H__[0] ^ (1 + nu__H__) * (1 - theta__H__) * ( - LGM[0]) / N__H__[0] * (N__H__[0] ^ (1 - theta__H__) * LAMBDA__H__[0] * AUX_ENDO_LAG_5_3[-1] ^ theta__H__) ^ (-nu__H__)

	phi__H__ * LGM[0] + phi__H__ * beta__H__ * LGM[1] + phi__H__ * beta__H__ ^ 2 * AUX_ENDO_LEAD_107[1] + phi__H__ * beta__H__ ^ 3 * AUX_ENDO_LEAD_112[1] + (1 - delta__H__) * phi__H__ * LGM[1] * ( - beta__H__) + (1 - delta__H__) * phi__H__ * ( - (beta__H__ ^ 2)) * AUX_ENDO_LEAD_107[1] + (1 - delta__H__) * phi__H__ * ( - (beta__H__ ^ 3)) * AUX_ENDO_LEAD_112[1] + (1 - delta__H__) * phi__H__ * ( - (beta__H__ ^ 4)) * AUX_ENDO_LEAD_132[1] = AUX_ENDO_LEAD_151[1]

	LGM[0] = beta__H__ * LGM[1] * (1 + sigma__H__ * Z__H__[0] ^ (( - nu__H__) - 1) * Y__H__[1] ^ (1 + nu__H__))

	NX__H__[0] = (Y__H__[0] - (Z__H__[0] + X__H__[0] + C__H__[0] - Z__H__[-1])) / Y__H__[0]

	Y__F__[0] = (sigma__F__ * Z__F__[-1] ^ (-nu__F__) + (N__F__[0] ^ (1 - theta__F__) * LAMBDA__F__[0] * AUX_ENDO_LAG_4_3[-1] ^ theta__F__) ^ (-nu__F__)) ^ (( - 1) / nu__F__)

	K__F__[0] = (1 - delta__F__) * K__F__[-1] + S__F__[0]

	X__F__[0] = S__F__[0] * phi__F__ + phi__F__ * S__F__[-1] + phi__F__ * AUX_ENDO_LAG_15_2[-1] + phi__F__ * AUX_ENDO_LAG_15_1[-1]

	A__F__[0] = N__F__[0] + (1 - eta__F__) * A__F__[-1]

	L__F__[0] = 1 - N__F__[0] * alpha__F__ - A__F__[-1] * eta__F__ * (1 - alpha__F__)

	U__F__[0] = (C__F__[0] ^ mu__F__ * L__F__[0] ^ (1 - mu__F__)) ^ gamma__F__

	U__F__[0] * mu__F__ * psi__F__ / C__F__[0] = LGM[0]

	U__F__[0] * (1 - mu__F__) * psi__F__ / L__F__[0] * ( - alpha__F__) = Y__F__[0] ^ (1 + nu__F__) * ( - LGM[0]) * (1 - theta__F__) / N__F__[0] * (N__F__[0] ^ (1 - theta__F__) * LAMBDA__F__[0] * AUX_ENDO_LAG_4_3[-1] ^ theta__F__) ^ (-nu__F__)

	LGM[0] * phi__F__ + phi__F__ * LGM[1] * beta__F__ + phi__F__ * beta__F__ ^ 2 * AUX_ENDO_LEAD_107[1] + phi__F__ * beta__F__ ^ 3 * AUX_ENDO_LEAD_112[1] + (1 - delta__F__) * phi__F__ * LGM[1] * ( - beta__F__) + (1 - delta__F__) * phi__F__ * ( - (beta__F__ ^ 2)) * AUX_ENDO_LEAD_107[1] + (1 - delta__F__) * phi__F__ * ( - (beta__F__ ^ 3)) * AUX_ENDO_LEAD_112[1] + (1 - delta__F__) * phi__F__ * ( - (beta__F__ ^ 4)) * AUX_ENDO_LEAD_132[1] = AUX_ENDO_LEAD_302[1]

	LGM[0] = LGM[1] * beta__F__ * (1 + sigma__F__ * Z__F__[0] ^ (( - nu__F__) - 1) * Y__F__[1] ^ (1 + nu__F__))

	NX__F__[0] = (Y__F__[0] - (Z__F__[0] + X__F__[0] + C__F__[0] - Z__F__[-1])) / Y__F__[0]

	LAMBDA__H__[0] - 1 = rho__H____H__ * (LAMBDA__H__[-1] - 1) + rho__H____F__ * (LAMBDA__F__[-1] - 1) + Z_E__H__ * E__H__[x]

	LAMBDA__F__[0] - 1 = (LAMBDA__F__[-1] - 1) * rho__F____F__ + (LAMBDA__H__[-1] - 1) * rho__F____H__ + Z_E__F__ * E__F__[x]

	Z__H__[0] + X__H__[0] + C__H__[0] - Z__H__[-1] + Z__F__[0] + X__F__[0] + C__F__[0] - Z__F__[-1] = Y__H__[0] + Y__F__[0]

	dLGM[0] = LGM[1] / LGM[0]

	dLGM_ann[0] = dLGM[0] * dLGM[-1] * AUX_ENDO_LAG_25_1[-1] * AUX_ENDO_LAG_25_2[-1]

	AUX_ENDO_LEAD_107[0] = LGM[1]

	AUX_ENDO_LEAD_112[0] = AUX_ENDO_LEAD_107[1]

	AUX_ENDO_LEAD_132[0] = AUX_ENDO_LEAD_112[1]

	AUX_ENDO_LEAD_416[0] = Y__H__[1] ^ (1 + nu__H__) * theta__H__ * LGM[1] * beta__H__ ^ 4 / AUX_ENDO_LAG_5_2[-1] * (N__H__[1] ^ (1 - theta__H__) * LAMBDA__H__[1] * AUX_ENDO_LAG_5_2[-1] ^ theta__H__) ^ (-nu__H__)

	AUX_ENDO_LEAD_433[0] = AUX_ENDO_LEAD_416[1]

	AUX_ENDO_LEAD_151[0] = AUX_ENDO_LEAD_433[1]

	AUX_ENDO_LEAD_487[0] = Y__F__[1] ^ (1 + nu__F__) * theta__F__ * LGM[1] * beta__F__ ^ 4 / AUX_ENDO_LAG_4_2[-1] * (N__F__[1] ^ (1 - theta__F__) * LAMBDA__F__[1] * AUX_ENDO_LAG_4_2[-1] ^ theta__F__) ^ (-nu__F__)

	AUX_ENDO_LEAD_504[0] = AUX_ENDO_LEAD_487[1]

	AUX_ENDO_LEAD_302[0] = AUX_ENDO_LEAD_504[1]

	AUX_ENDO_LAG_5_1[0] = K__H__[-1]

	AUX_ENDO_LAG_5_2[0] = AUX_ENDO_LAG_5_1[-1]

	AUX_ENDO_LAG_5_3[0] = AUX_ENDO_LAG_5_2[-1]

	AUX_ENDO_LAG_16_1[0] = S__H__[-1]

	AUX_ENDO_LAG_16_2[0] = AUX_ENDO_LAG_16_1[-1]

	AUX_ENDO_LAG_4_1[0] = K__F__[-1]

	AUX_ENDO_LAG_4_2[0] = AUX_ENDO_LAG_4_1[-1]

	AUX_ENDO_LAG_4_3[0] = AUX_ENDO_LAG_4_2[-1]

	AUX_ENDO_LAG_15_1[0] = S__F__[-1]

	AUX_ENDO_LAG_15_2[0] = AUX_ENDO_LAG_15_1[-1]

	AUX_ENDO_LAG_25_1[0] = dLGM[-1]

	AUX_ENDO_LAG_25_2[0] = AUX_ENDO_LAG_25_1[-1]

end


@parameters Backus_Kehoe_Kydland_1992 begin
	K_ss	=	11.0148

	F_H_ratio	=	1.0

	mu__F__ = 0.34

	mu__H__ = 0.34

	gamma__F__ = (-1.0)

	gamma__H__ = (-1.0)

	alpha__F__ = 1.0

	alpha__H__ = 1.0

	eta__F__ = 0.5

	eta__H__ = 0.5

	theta__F__ = 0.36

	theta__H__ = 0.36

	nu__F__ = 3.0

	nu__H__ = 3.0

	sigma__F__ = 0.01

	sigma__H__ = 0.01

	delta__F__ = 0.025

	delta__H__ = 0.025

	psi__F__ = 0.5

	psi__H__ = 0.5

	Z_E__F__ = 0.00852

	Z_E__H__ = 0.00852

	rho__H____H__ = 0.906

	rho__H____F__ = 0.088

	phi__F__ = 0.25

	phi__H__ = 0.25

	beta__F__ = 0.9899998184488822

	beta__H__ = 0.989999818448882

	rho__F____F__ = rho__H____H__

	rho__F____H__ = rho__H____F__

end

