var 
Pratio Sfunc SfuncD SfuncDflex Sfuncflex a afunc afuncD afuncDflex afuncflex b c cflex dc dinve dp dw dwobs dy epinfma ewma gam1 gam2 gam3 gamw1 gamw2 gamw3 gy inve inveflex k kflex kp kpflex lab labflex labobs mc ms pdot pdotl pinf pinfobs pk pkflex qs qsaux r rk rkflex robs rrflex spinf sw w wdot wdotl wflex wnew xi xiflex y yflex ygap zcap zcapflex ;

varexo 
ea eb egy ems epinf eqs ew ;

parameters 
calfa cfc cg cgy chabb cindp cindw clandaw cmap cmaw constebeta constepinf cprobp cprobw crdy crhoa crhob crhog crhoms crhopinf crhoqs crhow crpi crr cry csadjcost csigl csigma ctou ctrend curvp curvw czcap mcflex pinfss ;

% Parameter definitions:
	ctou	=	0.025;
	cg	=	0.18;
	clandaw	=	1.5;
	curvw	=	10.0;
	crhoa	=	0.95827;
	crhob	=	0.22137;
	crhog	=	0.97391;
	crhoqs	=	0.70524;
	crhoms	=	0.11421;
	crhopinf	=	0.83954;
	crhow	=	0.9745;
	cmap	=	0.69414;
	cmaw	=	0.93617;
	csadjcost	=	5.5811;
	csigma	=	1.4103;
	chabb	=	0.68049;
	cprobw	=	0.80501;
	csigl	=	2.2061;
	cindw	=	0.56351;
	cindp	=	0.24165;
	czcap	=	0.49552;
	cfc	=	1.3443;
	crpi	=	1.931;
	crr	=	0.82512;
	cry	=	0.097844;
	crdy	=	0.25114;
	constepinf	=	0.8731;
	constebeta	=	0.12575;
	ctrend	=	0.4419;
	cgy	=	0.53817;
	calfa	=	0.18003;
	curvp	=	64.5595;
	cprobp	=	0.667;
	mcflex	=	0.7438815740534109;
	pinfss	=	1.008731;

model;
	y(0) = c(0) + inve(0) + STEADY_STATE(y) * gy(0) + (afunc(0) * kp(-1)) / (1 + ctrend / 100);

	(y(0) * (pdot(0) + (curvp * (1 - cfc)) / cfc)) / (1 + (curvp * (1 - cfc)) / cfc) = a(0) * k(0) ^ calfa * lab(0) ^ (1 - calfa) - (cfc - 1) * STEADY_STATE(y);

	k(0) = (kp(-1) * zcap(0)) / (1 + ctrend / 100);

	kp(0) = inve(0) * qs(0) * (1 - Sfunc(0)) + (kp(-1) * (1 - ctou)) / (1 + ctrend / 100);

	pdot(0) = (1 - cprobp) * (Pratio(0) / dp(0)) ^ ((-cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1)) + pdot(-1) * cprobp * (((dp(-1) / dp(0)) * pinf(-1) ^ cindp * STEADY_STATE(pinf) ^ (1 - cindp)) / pinf(0)) ^ ((-cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1));

	wdot(0) = (1 - cprobw) * (wnew(0) / dw(0)) ^ ((-clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) + wdot(-1) * cprobw * (((dw(-1) / dw(0)) * pinf(-1) ^ cindw * STEADY_STATE(pinf) ^ (1 - cindw)) / pinf(0)) ^ ((-clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1));

	1 = (1 - cprobp) * (Pratio(0) / dp(0)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1)) + cprobp * (((dp(-1) / dp(0)) * pinf(-1) ^ cindp * STEADY_STATE(pinf) ^ (1 - cindp)) / pinf(0)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1));

	1 = (1 - cprobw) * (wnew(0) / dw(0)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1)) + cprobw * (((dw(-1) / dw(0)) * pinf(-1) ^ cindw * STEADY_STATE(pinf) ^ (1 - cindw)) / pinf(0)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1));

	1 = (dp(0) * (1 + (pdotl(0) * curvp * (1 - cfc)) / cfc)) / (1 + (curvp * (1 - cfc)) / cfc);

	w(0) = (dw(0) * (1 + ((curvw * (1 - clandaw)) / clandaw) * wdotl(0))) / (1 + (curvw * (1 - clandaw)) / clandaw);

	pdotl(0) = ((1 - cprobp) * Pratio(0)) / dp(0) + ((((cprobp * dp(-1)) / dp(0)) * pinf(-1) ^ cindp * STEADY_STATE(pinf) ^ (1 - cindp)) / pinf(0)) * pdotl(-1);

	wdotl(0) = ((1 - cprobw) * wnew(0)) / dw(0) + ((((cprobw * dw(-1)) / dw(0)) * pinf(-1) ^ cindw * STEADY_STATE(pinf) ^ (1 - cindw)) / pinf(0)) * wdotl(-1);

	xi(0) = exp(((csigma - 1) / (1 + csigl)) * ((lab(0) * ((curvw * (1 - clandaw)) / clandaw + wdot(0))) / (1 + (curvw * (1 - clandaw)) / clandaw)) ^ (1 + csigl)) * (c(0) - (c(-1) * chabb) / (1 + ctrend / 100)) ^ -csigma;

	1 = qs(0) * pk(0) * ((1 - Sfunc(0)) - ((1 + ctrend / 100) * inve(0) * SfuncD(0)) / inve(-1)) + ((((SfuncD(1) * xi(1)) / xi(0)) * qsaux(0) * pk(1) * (((1 + ctrend / 100) * inve(1)) / inve(0)) ^ 2 * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	xi(0) = (((xi(1) * b(0) * r(0) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma) / pinf(1);

	rk(0) = afuncD(0);

	pk(0) = (((((rk(1) * zcap(1) - afunc(1)) + (1 - ctou) * pk(1)) * xi(1) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma) / xi(0);

	k(0) = ((lab(0) * w(0) * calfa) / (1 - calfa)) / rk(0);

	mc(0) = (w(0) ^ (1 - calfa) * rk(0) ^ calfa) / (a(0) * calfa ^ calfa * (1 - calfa) ^ (1 - calfa));

	(wnew(0) * gamw1(0) * (1 + curvw * (1 - clandaw))) / (1 + (curvw * (1 - clandaw)) / clandaw) = clandaw * gamw2(0) + ((((gamw3(0) * curvw * (1 - clandaw)) / clandaw) * (clandaw - 1)) / (1 + (curvw * (1 - clandaw)) / clandaw)) * wnew(0) ^ (1 + (clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1));

	gamw1(0) = lab(0) * dw(0) ^ ((clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) + ((((gamw1(1) * ((STEADY_STATE(pinf) ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) ^ (-((1 + curvw * (1 - clandaw))) / (clandaw - 1)) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobw * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	gamw2(0) = (c(0) - (c(-1) * chabb) / (1 + ctrend / 100)) * lab(0) * sw(0) * dw(0) ^ ((clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) * ((lab(0) * ((curvw * (1 - clandaw)) / clandaw + wdot(0))) / (1 + (curvw * (1 - clandaw)) / clandaw)) ^ csigl + ((((gamw2(1) * ((STEADY_STATE(pinf) ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) ^ ((-clandaw * (1 + (curvw * (1 - clandaw)) / clandaw)) / (clandaw - 1)) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobw * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	gamw3(0) = lab(0) + ((((((gamw3(1) * STEADY_STATE(pinf) ^ (1 - cindw) * pinf(0) ^ cindw) / pinf(1)) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobw * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	(Pratio(0) * gam1(0) * (1 + curvp * (1 - cfc))) / (1 + (curvp * (1 - cfc)) / cfc) = cfc * gam2(0) + (((gam3(0) * (cfc - 1) * curvp * (1 - cfc)) / cfc) / (1 + (curvp * (1 - cfc)) / cfc)) * Pratio(0) ^ (1 + (cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1));

	gam1(0) = y(0) * dp(0) ^ ((cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1)) + ((((gam1(1) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobp * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma * ((STEADY_STATE(pinf) ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) ^ (-((1 + curvp * (1 - cfc))) / (cfc - 1));

	gam2(0) = y(0) * mc(0) * spinf(0) * dp(0) ^ ((cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1)) + ((((gam2(1) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobp * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma * ((STEADY_STATE(pinf) ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) ^ ((-cfc * (1 + (curvp * (1 - cfc)) / cfc)) / (cfc - 1));

	gam3(0) = y(0) + ((((((gam3(1) * STEADY_STATE(pinf) ^ (1 - cindp) * pinf(0) ^ cindp) / pinf(1)) * xi(1)) / xi(0)) * (1 + ctrend / 100) * cprobp * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	qsaux(0) = qs(1);

	r(0) = STEADY_STATE(r) ^ (1 - crr) * r(-1) ^ crr * (pinf(0) / pinfss) ^ ((1 - crr) * crpi) * (y(0) / yflex(0)) ^ ((1 - crr) * cry) * ((y(0) / yflex(0)) / (y(-1) / yflex(-1))) ^ crdy * ms(0);

	afunc(0) = ((STEADY_STATE(rk) * 1) / (czcap / (1 - czcap))) * (exp((czcap / (1 - czcap)) * (zcap(0) - 1)) - 1);

	afuncD(0) = STEADY_STATE(rk) * exp((czcap / (1 - czcap)) * (zcap(0) - 1));

	Sfunc(0) = (csadjcost / 2) * (((1 + ctrend / 100) * inve(0)) / inve(-1) - (1 + ctrend / 100)) ^ 2;

	SfuncD(0) = csadjcost * (((1 + ctrend / 100) * inve(0)) / inve(-1) - (1 + ctrend / 100));

	a(0) = (1 - crhoa) + crhoa * a(-1) + ea / 100;

	b(0) = (1 - crhob) + crhob * b(-1) + (eb * -(((1 - chabb / (1 + ctrend / 100)) / (csigma * (1 + chabb / (1 + ctrend / 100)))) ^ -1)) / 100;

	gy(0) - cg = crhog * (gy(-1) - cg) + egy / 100 + (ea * cgy) / 100;

	qs(0) = (1 - crhoqs) + crhoqs * qs(-1) + (eqs * csadjcost * (1 + ctrend / 100) ^ 2 * (1 + (1 / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ (1 - csigma))) / 100;

	ms(0) = (1 - crhoms) + crhoms * ms(-1) + ems / 100;

	spinf(0) = ((1 - crhopinf) + crhopinf * spinf(-1) + epinfma(0)) - cmap * epinfma(-1);

	epinfma(0) = ((epinf * 1) / ((((1 / (1 + ((cindp * (1 + ctrend / 100) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma)) * (1 - cprobp) * (1 - ((cprobp * (1 + ctrend / 100) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma)) / cprobp) / (1 + curvp * (cfc - 1)))) / 100;

	sw(0) = ((1 - crhow) + crhow * sw(-1) + ewma(0)) - cmaw * ewma(-1);

	ewma(0) = ((ew * 1) / (((1 / (1 + curvw * (clandaw - 1))) * (1 - cprobw) * (1 - ((cprobw * (1 + ctrend / 100) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma)) / (cprobw * (1 + (((1 + ctrend / 100) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma)))) / 100;

	yflex(0) = cflex(0) + inveflex(0) + gy(0) * STEADY_STATE(yflex) + (afuncflex(0) * kpflex(-1)) / (1 + ctrend / 100);

	yflex(0) = a(0) * kflex(0) ^ calfa * labflex(0) ^ (1 - calfa) - (cfc - 1) * STEADY_STATE(yflex);

	kflex(0) = (kpflex(-1) * zcapflex(0)) / (1 + ctrend / 100);

	kpflex(0) = inveflex(0) * qs(0) * (1 - Sfuncflex(0)) + (kpflex(-1) * (1 - ctou)) / (1 + ctrend / 100);

	xiflex(0) = exp(((csigma - 1) / (1 + csigl)) * labflex(0) ^ (1 + csigl)) * (cflex(0) - (cflex(-1) * chabb) / (1 + ctrend / 100)) ^ -csigma;

	1 = qs(0) * pkflex(0) * ((1 - Sfuncflex(0)) - ((1 + ctrend / 100) * inveflex(0) * SfuncDflex(0)) / inveflex(-1)) + ((((SfuncDflex(1) * qsaux(0) * xiflex(1)) / xiflex(0)) * pkflex(1) * (((1 + ctrend / 100) * inveflex(1)) / inveflex(0)) ^ 2 * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	xiflex(0) = ((xiflex(1) * b(0) * rrflex(0) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma;

	rkflex(0) = afuncDflex(0);

	pkflex(0) = (((((rkflex(1) * zcapflex(1) - afuncflex(1)) + (1 - ctou) * pkflex(1)) * xiflex(1) * 1) / (1 + constebeta / 100)) * (1 + ctrend / 100) ^ -csigma) / xiflex(0);

	kflex(0) = (((labflex(0) * calfa) / (1 - calfa)) * wflex(0)) / rkflex(0);

	mcflex = (wflex(0) ^ (1 - calfa) * rkflex(0) ^ calfa) / (a(0) * calfa ^ calfa * (1 - calfa) ^ (1 - calfa));

	(wflex(0) * (1 + curvw * (1 - clandaw))) / (1 + (curvw * (1 - clandaw)) / clandaw) = STEADY_STATE(sw) * (labflex(0) ^ csigl * clandaw * (cflex(0) - (cflex(-1) * chabb) / (1 + ctrend / 100)) + (((wflex(0) * curvw * (1 - clandaw)) / clandaw) * (clandaw - 1)) / (1 + (curvw * (1 - clandaw)) / clandaw));

	afuncflex(0) = ((STEADY_STATE(rkflex) * 1) / (czcap / (1 - czcap))) * (exp((czcap / (1 - czcap)) * (zcapflex(0) - 1)) - 1);

	afuncDflex(0) = STEADY_STATE(rkflex) * exp((czcap / (1 - czcap)) * (zcapflex(0) - 1));

	Sfuncflex(0) = (csadjcost / 2) * (((1 + ctrend / 100) * inveflex(0)) / inveflex(-1) - (1 + ctrend / 100)) ^ 2;

	SfuncDflex(0) = csadjcost * (((1 + ctrend / 100) * inveflex(0)) / inveflex(-1) - (1 + ctrend / 100));

	ygap(0) = 100 * log(y(0) / yflex(0));

	dy(0) = ctrend + 100 * (y(0) / y(-1) - 1);

	dc(0) = ctrend + 100 * (c(0) / c(-1) - 1);

	dinve(0) = ctrend + 100 * (inve(0) / inve(-1) - 1);

	pinfobs(0) = 100 * (pinf(0) - STEADY_STATE(pinf)) + constepinf;

	robs(0) = 100 * (r(0) - 1);

	dwobs(0) = ctrend + 100 * (w(0) / w(-1) - 1);

	labobs(0) = 100 * (lab(0) / STEADY_STATE(lab) - 1);

end;

shocks;
var	ea	=	1;
var	eb	=	1;
var	egy	=	1;
var	ems	=	1;
var	epinf	=	1;
var	eqs	=	1;
var	ew	=	1;
end;

initval;
	Pratio	=	1.0;
	Sfunc	=	0.0;
	SfuncD	=	0.0;
	SfuncDflex	=	0.0;
	Sfuncflex	=	0.0;
	a	=	1.0;
	afunc	=	2.6707642871043906e-18;
	afuncD	=	0.03250310455837918;
	afuncDflex	=	0.032503104558379174;
	afuncflex	=	2.9540420520779154e-18;
	b	=	1.0;
	c	=	0.8963673008108926;
	cflex	=	0.8963673008108919;
	dc	=	0.4419;
	dinve	=	0.4419;
	dp	=	1.0;
	dw	=	0.8323624555939105;
	dwobs	=	0.4419;
	dy	=	0.4419;
	epinfma	=	0.0;
	ewma	=	0.0;
	gam1	=	4.071805532645228;
	gam2	=	3.0289411088635187;
	gam3	=	4.071805532645226;
	gamw1	=	24.58768372203227;
	gamw2	=	13.643909866824808;
	gamw3	=	6.806204556019938;
	gy	=	0.18;
	inve	=	0.22229717097859672;
	inveflex	=	0.22229717097859789;
	k	=	7.55624497700795;
	kflex	=	7.556244977007963;
	kp	=	7.589636023561346;
	kpflex	=	7.58963602356136;
	lab	=	1.3439139854552384;
	labflex	=	1.3439139854552407;
	labobs	=	0.0;
	mc	=	0.7438815740534109;
	ms	=	1.0;
	pdot	=	1.0;
	pdotl	=	1.0;
	pinf	=	1.008731;
	pinfobs	=	0.8731;
	pk	=	1.0;
	pkflex	=	1.0;
	qs	=	1.0;
	qsaux	=	1.0;
	r	=	1.0162996141642786;
	rk	=	0.032503104558379174;
	rkflex	=	0.032503104558379174;
	robs	=	1.6299614164278609;
	rrflex	=	1.0075031045583793;
	spinf	=	1.0;
	sw	=	1.0;
	w	=	0.8323624555939108;
	wdot	=	1.0;
	wdotl	=	1.0000000000000002;
	wflex	=	0.8323624555939109;
	wnew	=	0.8323624555939105;
	xi	=	8.007548000200039;
	xiflex	=	8.007548000200062;
	y	=	1.3642249655969378;
	yflex	=	1.3642249655969414;
	ygap	=	-2.553512956637863e-13;
	zcap	=	1.0000000000000002;
	zcapflex	=	1.0000000000000002;
end;

stoch_simul(order = 1, irf = 40);
