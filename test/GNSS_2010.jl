using MacroModelling

@model GNSS_2010 begin
	(1 - a_i) * exp(ee_z[0]) * (c_p[0] - a_i * c_p[-1]) ^ (-1) = lam_p[0]

	j * exp(ee_j[0]) / h_p[0] - lam_p[0] * q_h[0] + beta_p * lam_p[1] * q_h[1] = 0

	lam_p[0] = beta_p * lam_p[1] * (1 + r_d[0]) / pie[1]

	(1 - exp(eps_l[0])) * l_p[0] + exp(eps_l[0]) * l_p[0] ^ (1 + phi) / w_p[0] / lam_p[0] - pie_wp[0] * kappa_w * (pie_wp[0] - pie[-1] ^ ind_w * piss ^ (1 - ind_w)) + lam_p[1] * beta_p * kappa_w / lam_p[0] * (pie_wp[1] - piss ^ (1 - ind_w) * pie[0] ^ ind_w) * pie_wp[1] ^ 2 / pie[1] = 0

	pie_wp[0] = w_p[0] * pie[0] / w_p[-1]

	c_p[0] + q_h[0] * (h_p[0] - h_p[-1]) + d_p[0] = l_p[0] * w_p[0] + (1 + r_d[-1]) * d_p[-1] / pie[0] + J_R[0] / gamma_p

	(1 - a_i) * exp(ee_z[0]) * (c_i[0] - a_i * c_i[-1]) ^ (-1) = lam_i[0]

	j * exp(ee_j[0]) / h_i[0] - q_h[0] * lam_i[0] + q_h[1] * beta_i * lam_i[1] + q_h[1] * pie[1] * s_i[0] * exp(m_i[0]) = 0

	lam_i[0] - beta_i * lam_i[1] * (1 + r_bh[0]) / pie[1] = s_i[0] * (1 + r_bh[0])

	(1 - exp(eps_l[0])) * l_i[0] + exp(eps_l[0]) * l_i[0] ^ (1 + phi) / w_i[0] / lam_i[0] - kappa_w * pie_wi[0] * (pie_wi[0] - pie[-1] ^ ind_w * piss ^ (1 - ind_w)) + lam_i[1] * kappa_w * beta_i / lam_i[0] * (pie_wi[1] - piss ^ (1 - ind_w) * pie[0] ^ ind_w) * pie_wi[1] ^ 2 / pie[1] = 0

	pie_wi[0] = pie[0] * w_i[0] / w_i[-1]

	c_i[0] + q_h[0] * (h_i[0] - h_i[-1]) + (1 + r_bh[-1]) * b_i[-1] / pie[0] = l_i[0] * w_i[0] + b_i[0]

	(1 + r_bh[0]) * b_i[0] = exp(m_i[0]) * q_h[1] * pie[1] * h_i[0]

	K[0] = (1 - deltak) * K[-1] + I[0] * (1 - kappa_i / 2 * (I[0] * exp(ee_qk[0]) / I[-1] - 1) ^ 2)

	1 = q_k[0] * (1 - kappa_i / 2 * (I[0] * exp(ee_qk[0]) / I[-1] - 1) ^ 2 - (I[0] * exp(ee_qk[0]) / I[-1] - 1) * kappa_i * I[0] * exp(ee_qk[0]) / I[-1]) + kappa_i * exp(ee_qk[1]) * beta_e * lam_e[1] / lam_e[0] * q_k[1] * (exp(ee_qk[1]) * I[1] / I[0] - 1) * (I[1] / I[0]) ^ 2

	(1 - a_i) * (c_e[0] - a_i * c_e[-1]) ^ (-1) = lam_e[0]

	q_k[1] * pie[1] * (1 - deltak) * s_e[0] * exp(m_e[0]) + beta_e * lam_e[1] * ((1 - deltak) * q_k[1] + r_k[1] * u[1] - (eksi_1 * (u[1] - 1) + eksi_2 / 2 * (u[1] - 1) ^ 2)) = q_k[0] * lam_e[0]

	w_p[0] = ni * (1 - alpha) * y_e[0] / (l_pd[0] * x[0])

	w_i[0] = (1 - alpha) * y_e[0] * (1 - ni) / (x[0] * l_id[0])

	lam_e[0] - s_e[0] * (1 + r_be[0]) = beta_e * lam_e[1] * (1 + r_be[0]) / pie[1]

	r_k[0] = eksi_1 + eksi_2 * (u[0] - 1)

	c_e[0] + (1 + r_be[-1]) * b_ee[-1] / pie[0] + w_p[0] * l_pd[0] + w_i[0] * l_id[0] + q_k[0] * k_e[0] + (eksi_1 * (u[0] - 1) + eksi_2 / 2 * (u[0] - 1) ^ 2) * k_e[-1] = y_e[0] / x[0] + b_ee[0] + q_k[0] * (1 - deltak) * k_e[-1]

	y_e[0] = exp(A_e[0]) * (u[0] * k_e[-1]) ^ alpha * (l_pd[0] ^ ni * l_id[0] ^ (1 - ni)) ^ (1 - alpha)

	(1 + r_be[0]) * b_ee[0] = exp(m_e[0]) * q_k[1] * pie[1] * (1 - deltak) * k_e[0]

	r_k[0] = exp(A_e[0]) * alpha * (l_pd[0] ^ ni * l_id[0] ^ (1 - ni)) ^ (1 - alpha) * u[0] ^ (alpha - 1) * k_e[-1] ^ (alpha - 1) / x[0]

	R_b[0] = r_ib[0] + ( - kappa_kb) * (K_b[0] / B[0] - vi) * (K_b[0] / B[0]) ^ 2

	pie[0] * K_b[0] = (1 - delta_kb) * K_b[-1] / exp(eps_K_b[0]) + j_B[-1]

	gamma_b * d_b[0] = d_p[0] * gamma_p

	gamma_b * b_h[0] = b_i[0] * gamma_i

	gamma_b * b_e[0] = b_ee[0] * gamma_e

	b_h[0] + b_e[0] = K_b[0] + d_b[0]

	lam_p[1] * beta_p * kappa_d / lam_p[0] * (r_d[1] / r_d[0] - (r_d[0] / r_d[-1]) ^ ind_d) * (r_d[1] / r_d[0]) ^ 2 * d_b[1] / d_b[0] + exp(mk_d[0]) / (exp(mk_d[0]) - 1) - 1 - r_ib[0] * exp(mk_d[0]) / (exp(mk_d[0]) - 1) / r_d[0] - r_d[0] * kappa_d * (r_d[0] / r_d[-1] - (r_d[-1] / AUX_ENDO_LAG_54_1[-1]) ^ ind_d) / r_d[-1] = 0

	1 + beta_p * lam_p[1] / lam_p[0] * kappa_be * (r_be[1] / r_be[0] - (r_be[0] / r_be[-1]) ^ ind_be) * (r_be[1] / r_be[0]) ^ 2 * b_e[1] / b_e[0] - exp(mk_be[0]) / (exp(mk_be[0]) - 1) + R_b[0] * exp(mk_be[0]) / (exp(mk_be[0]) - 1) / r_be[0] - r_be[0] * kappa_be * (r_be[0] / r_be[-1] - (r_be[-1] / AUX_ENDO_LAG_52_1[-1]) ^ ind_be) / r_be[-1] = 0

	1 + beta_p * lam_p[1] / lam_p[0] * kappa_bh * (r_bh[1] / r_bh[0] - (r_bh[0] / r_bh[-1]) ^ ind_bh) * (r_bh[1] / r_bh[0]) ^ 2 * b_h[1] / b_h[0] - exp(mk_bh[0]) / (exp(mk_bh[0]) - 1) + R_b[0] * exp(mk_bh[0]) / (exp(mk_bh[0]) - 1) / r_bh[0] - r_bh[0] * kappa_bh * (r_bh[0] / r_bh[-1] - (r_bh[-1] / AUX_ENDO_LAG_53_1[-1]) ^ ind_bh) / r_bh[-1] = 0

	j_B[0] = r_bh[0] * b_h[0] + r_be[0] * b_e[0] - r_d[0] * d_b[0] - kappa_d * r_d[0] * d_b[0] / 2 * (r_d[0] / r_d[-1] - 1) ^ 2 - kappa_be * r_be[0] * b_e[0] / 2 * (r_be[0] / r_be[-1] - 1) ^ 2 - kappa_bh * r_bh[0] * b_h[0] / 2 * (r_bh[0] / r_bh[-1] - 1) ^ 2 - kappa_kb * K_b[0] / 2 * (K_b[0] / B[0] - vi) ^ 2

	J_R[0] = Y[0] * (1 - 1 / x[0] - kappa_p / 2 * (pie[0] - pie[-1] ^ ind_p * piss ^ (1 - ind_p)) ^ 2)

	1 - exp(eps_y[0]) + exp(eps_y[0]) / x[0] - (pie[0] - pie[-1] ^ ind_p * piss ^ (1 - ind_p)) * pie[0] * kappa_p + kappa_p * lam_p[1] * beta_p * pie[1] / lam_p[0] * (pie[1] - piss ^ (1 - ind_p) * pie[0] ^ ind_p) * Y[1] / Y[0] = 0

	C[0] = c_p[0] * gamma_p + c_i[0] * gamma_i + c_e[0] * gamma_e

	BH[0] = gamma_b * b_h[0]

	BE[0] = gamma_b * b_e[0]

	B[0] = BH[0] + BE[0]

	D[0] = d_p[0] * gamma_p

	Y[0] = y_e[0] * gamma_e

	J_B[0] = gamma_b * j_B[0]

	l_pd[0] * gamma_e = l_p[0] * gamma_p

	l_id[0] * gamma_e = l_i[0] * gamma_i

	h = h_p[0] * gamma_p + h_i[0] * gamma_i

	K[0] = k_e[0] * gamma_e

	Y1[0] = K[0] + C[0] - (1 - deltak) * K[-1]

	PIW[0] = pie[0] * (w_p[0] + w_i[0]) / (w_p[-1] + w_i[-1])

	1 + r_ib[0] = (1 + r_ib_ss) ^ (1 - rho_ib) * (1 + r_ib[-1]) ^ rho_ib * ((pie[0] / piss) ^ phi_pie * (Y1[0] / Y1[-1]) ^ phi_y) ^ (1 - rho_ib) * (1 + sigma__r_ib * e_r_ib[x])

	exp(ee_z[0]) = 1 - rho_ee_z + rho_ee_z * exp(ee_z[-1]) + sigma__z * e_z[x]

	exp(A_e[0]) = 1 - rho_A_e + rho_A_e * exp(A_e[-1]) + sigma__A_e * e_A_e[x]

	exp(ee_j[0]) = 1 - rho_ee_j + rho_ee_j * exp(ee_j[-1]) - sigma__j * e_j[x]

	exp(m_i[0]) = (1 - rho_mi) * m_i_ss + rho_mi * exp(m_i[-1]) + sigma__mi * e_mi[x]

	exp(m_e[0]) = (1 - rho_me) * m_e_ss + rho_me * exp(m_e[-1]) + sigma__me * e_me[x]

	exp(mk_d[0]) = (1 - rho_mk_d) * mk_d_ss + rho_mk_d * exp(mk_d[-1]) + sigma__mk_d * e_mk_d[x]

	exp(mk_be[0]) = (1 - rho_mk_be) * mk_be_ss + rho_mk_be * exp(mk_be[-1]) + sigma__mk_be * e_mk_be[x]

	exp(mk_bh[0]) = (1 - rho_mk_bh) * mk_bh_ss + rho_mk_bh * exp(mk_bh[-1]) + sigma__mk_bh * e_mk_bh[x]

	exp(ee_qk[0]) = 1 - rho_ee_qk + rho_ee_qk * exp(ee_qk[-1]) + sigma__qk * e_qk[x]

	exp(eps_y[0]) = (1 - rho_eps_y) * eps_y_ss + rho_eps_y * exp(eps_y[-1]) + sigma__y * e_y[x]

	exp(eps_l[0]) = (1 - rho_eps_l) * eps_l_ss + rho_eps_l * exp(eps_l[-1]) + sigma__l * e_l[x]

	exp(eps_K_b[0]) = 1 - rho_eps_K_b + rho_eps_K_b * exp(eps_K_b[-1]) + sigma__eps_K_b * e_eps_K_b[x]

	rr_e[0] = lam_e[0] - beta_e * lam_e[1] * (1 + r_be[0]) / pie[1]

	bm[0] = r_bh[-1] * b_h[-1] / (b_h[-1] + b_e[-1]) + r_be[-1] * b_e[-1] / (b_h[-1] + b_e[-1]) - r_d[-1]

	spr_b[0] = r_bh[0] * 0.5 + r_be[0] * 0.5 - r_d[0]

	AUX_ENDO_LAG_54_1[0] = r_d[-1]

	AUX_ENDO_LAG_52_1[0] = r_be[-1]

	AUX_ENDO_LAG_53_1[0] = r_bh[-1]

end


@parameters GNSS_2010 begin
	beta_p = 0.9943

	beta_i = 0.975

	j = 0.2

	phi = 1.0

	m_i_ss = 0.7

	m_e_ss = 0.35

	alpha = 0.25

	eps_d	=	-1.46025

	eps_bh	=	2.932806

	eps_be	=	2.932806

	eps_y_ss = 6.0

	eps_l_ss = 5.0

	gamma_p = 1.0

	gamma_i = 1.0

	ni = 0.8

	gamma_b = 1.0

	gamma_e = 1.0

	deltak = 0.025

	piss = 1.0

	h = 1.0

	vi = 0.09

	ind_d = 0.0

	ind_be = 0.0

	ind_bh = 0.0

	rho_ee_z = 0.3935275324257052

	rho_A_e = 0.9390001567894549

	rho_ee_j = 0.9211790941478728

	rho_me = 0.8938651443507468

	rho_mi = 0.9286486478061776

	rho_mk_d = 0.8380479641501677

	rho_mk_bh = 0.8194621730335763

	rho_mk_be = 0.8342810056222126

	rho_ee_qk = 0.5474914620444137

	rho_eps_y = 0.3047340963457367

	rho_eps_l = 0.639922254764848

	rho_eps_K_b = 0.8129795852441276

	kappa_p = 28.65019653869527

	kappa_w = 99.89828358530188

	kappa_i = 10.182155670839322

	kappa_d = 3.5029734165847466

	kappa_be = 9.36382331915174

	kappa_bh = 10.086654447226444

	kappa_kb = 11.068335540791962

	phi_pie = 1.9816026561910398

	rho_ib = 0.7685551455946952

	phi_y = 0.3459149657035201

	ind_p = 0.16051347848216171

	ind_w = 0.27569624058316433

	a_i = 0.8559521971842566

	sigma__z = 0.0144

	sigma__A_e = 0.0062

	sigma__j = 0.0658

	sigma__me = 0.0034

	sigma__mi = 0.0023

	sigma__mk_d = 0.0488

	sigma__mk_bh = 0.0051

	sigma__mk_be = 0.1454

	sigma__qk = 0.0128

	sigma__r_ib = 0.0018

	sigma__y = 1.0099

	sigma__l = 0.3721

	sigma__eps_K_b = 0.05

	beta_b = beta_p

	beta_e = beta_i

	mk_d_ss = eps_d/(eps_d-1)

	mk_bh_ss = eps_bh/(eps_bh-1)

	mk_be_ss = eps_be/(eps_be-1)

	r_ib_ss = (eps_d-1)*(piss/beta_p-1)/eps_d

	r_be_ss = (eps_be * r_ib_ss) / (eps_be - 1)

	r_bh_ss = (eps_bh * r_ib_ss) / (eps_bh - 1)

	r_k_ss = (-((1 - deltak)) - ((piss * (1 - deltak) * m_e_ss) / beta_e) * (1 / (1 + r_be_ss) - beta_e / piss)) + 1 / beta_e

	eksi_1 = r_k_ss

	eksi_2 = r_k_ss*0.1

	eps_b = (eps_bh + eps_be) / 2

	delta_kb = r_ib_ss/vi*(eps_d-eps_b+eps_d*vi*(eps_b-1))/((eps_d-1)*(eps_b-1))

end

