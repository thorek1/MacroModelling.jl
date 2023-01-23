using MacroModelling

@model GNSS_2010 begin
	interestPol[0] = 400 * r_ib[0]

	interestH[0] = 400 * exp(r_bh[0])

	interestF[0] = 400 * exp(r_be[0])

	inflation[0] = pie[0] * 100

	loansH[0] = 100 * BH[0]

	loansF[0] = 100 * BE[0]

	output[0] = 100 * Y1[0]

	consumption[0] = 100 * C[0]

	investment[0] = 100 * I[0]

	deposits[0] = 100 * D[0]

	interestDep[0] = 400 * exp(r_d[0])

	bankcapital[0] = 100 * K_b[0]

	(1 - a_i) * exp(ee_z[0]) * (exp(c_p[0]) - a_i * exp(c_p[-1])) ^ (-1) = exp(lam_p[0])

	j * exp(ee_j[0]) / exp(h_p[0]) - exp(lam_p[0]) * exp(q_h[0]) + beta_p * exp(lam_p[1]) * exp(q_h[1]) = 0

	exp(lam_p[0]) = beta_p * exp(lam_p[1]) * (1 + exp(r_d[0])) / exp(pie[1])

	(1 - exp(eps_l[0])) * exp(l_p[0]) + exp(eps_l[0]) * exp(l_p[0]) ^ (1 + phi) / exp(w_p[0]) / exp(lam_p[0]) - exp(pie_wp[0]) * kappa_w * (exp(pie_wp[0]) - exp(pie[-1]) ^ ind_w * piss ^ (1 - ind_w)) + kappa_w * beta_p * exp(lam_p[1]) / exp(lam_p[0]) * (exp(pie_wp[1]) - piss ^ (1 - ind_w) * exp(pie[0]) ^ ind_w) * exp(pie_wp[1]) ^ 2 / exp(pie[1]) = 0

	exp(pie_wp[0]) = exp(pie[0]) * exp(w_p[0]) / exp(w_p[-1])

	exp(c_p[0]) + exp(q_h[0]) * (exp(h_p[0]) - exp(h_p[-1])) + exp(d_p[0]) = exp(l_p[0]) * exp(w_p[0]) + (1 + exp(r_d[-1])) * exp(d_p[-1]) / exp(pie[0]) + exp(J_R[0]) / gamma_p

	(1 - a_i) * exp(ee_z[0]) * (exp(c_i[0]) - a_i * exp(c_i[-1])) ^ (-1) = exp(lam_i[0])

	j * exp(ee_j[0]) / exp(h_i[0]) - exp(q_h[0]) * exp(lam_i[0]) + exp(q_h[1]) * beta_i * exp(lam_i[1]) + exp(pie[1]) * exp(q_h[1]) * exp(s_i[0]) * exp(m_i[0]) = 0

	exp(lam_i[0]) - beta_i * exp(lam_i[1]) * (1 + exp(r_bh[0])) / exp(pie[1]) = exp(s_i[0]) * (1 + exp(r_bh[0]))

	(1 - exp(eps_l[0])) * exp(l_i[0]) + exp(eps_l[0]) * exp(l_i[0]) ^ (1 + phi) / exp(w_i[0]) / exp(lam_i[0]) - exp(pie_wi[0]) * kappa_w * (exp(pie_wi[0]) - exp(pie[-1]) ^ ind_w * piss ^ (1 - ind_w)) + kappa_w * beta_i * exp(lam_i[1]) / exp(lam_i[0]) * (exp(pie_wi[1]) - piss ^ (1 - ind_w) * exp(pie[0]) ^ ind_w) * exp(pie_wi[1]) ^ 2 / exp(pie[1]) = 0

	exp(pie_wi[0]) = exp(pie[0]) * exp(w_i[0]) / exp(w_i[-1])

	exp(c_i[0]) + exp(q_h[0]) * (exp(h_i[0]) - exp(h_i[-1])) + (1 + exp(r_bh[-1])) * exp(b_i[-1]) / exp(pie[0]) = exp(l_i[0]) * exp(w_i[0]) + exp(b_i[0])

	(1 + exp(r_bh[0])) * exp(b_i[0]) = exp(pie[1]) * exp(h_i[0]) * exp(q_h[1]) * exp(m_i[0])

	exp(K[0]) = (1 - deltak) * exp(K[-1]) + exp(I[0]) * (1 - kappa_i / 2 * (exp(I[0]) * exp(ee_qk[0]) / exp(I[-1]) - 1) ^ 2)

	1 = exp(q_k[0]) * (1 - kappa_i / 2 * (exp(I[0]) * exp(ee_qk[0]) / exp(I[-1]) - 1) ^ 2 - exp(ee_qk[0]) * exp(I[0]) * kappa_i * (exp(I[0]) * exp(ee_qk[0]) / exp(I[-1]) - 1) / exp(I[-1])) + exp(ee_qk[1]) * kappa_i * beta_e * exp(lam_e[1]) / exp(lam_e[0]) * exp(q_k[1]) * (exp(I[1]) * exp(ee_qk[1]) / exp(I[0]) - 1) * (exp(I[1]) / exp(I[0])) ^ 2

	(1 - a_i) * (exp(c_e[0]) - a_i * exp(c_e[-1])) ^ (-1) = exp(lam_e[0])

	(1 - deltak) * exp(pie[1]) * exp(q_k[1]) * exp(s_e[0]) * exp(m_e[0]) + beta_e * exp(lam_e[1]) * ((1 - deltak) * exp(q_k[1]) + exp(r_k[1]) * exp(u[1]) - (eksi_1 * (exp(u[1]) - 1) + eksi_2 / 2 * (exp(u[1]) - 1) ^ 2)) = exp(q_k[0]) * exp(lam_e[0])

	exp(w_p[0]) = ni * (1 - alpha) * exp(y_e[0]) / (exp(l_pd[0]) * exp(x[0]))

	exp(w_i[0]) = exp(y_e[0]) * (1 - alpha) * (1 - ni) / (exp(x[0]) * exp(l_id[0]))

	exp(lam_e[0]) - exp(s_e[0]) * (1 + exp(r_be[0])) = beta_e * exp(lam_e[1]) * (1 + exp(r_be[0])) / exp(pie[1])

	exp(r_k[0]) = eksi_1 + eksi_2 * (exp(u[0]) - 1)

	exp(c_e[0]) + (1 + exp(r_be[-1])) * exp(b_ee[-1]) / exp(pie[0]) + exp(w_p[0]) * exp(l_pd[0]) + exp(w_i[0]) * exp(l_id[0]) + exp(q_k[0]) * exp(k_e[0]) + (eksi_1 * (exp(u[0]) - 1) + eksi_2 / 2 * (exp(u[0]) - 1) ^ 2) * exp(k_e[-1]) = exp(y_e[0]) / exp(x[0]) + exp(b_ee[0]) + exp(k_e[-1]) * (1 - deltak) * exp(q_k[0])

	exp(y_e[0]) = exp(A_e[0]) * (exp(u[0]) * exp(k_e[-1])) ^ alpha * (exp(l_pd[0]) ^ ni * exp(l_id[0]) ^ (1 - ni)) ^ (1 - alpha)

	(1 + exp(r_be[0])) * exp(b_ee[0]) = (1 - deltak) * exp(k_e[0]) * exp(pie[1]) * exp(q_k[1]) * exp(m_e[0])

	exp(r_k[0]) = (exp(l_pd[0]) ^ ni * exp(l_id[0]) ^ (1 - ni)) ^ (1 - alpha) * alpha * exp(A_e[0]) * exp(u[0]) ^ (alpha - 1) * exp(k_e[-1]) ^ (alpha - 1) / exp(x[0])

	exp(R_b[0]) = r_ib[0] + ( - kappa_kb) * (exp(K_b[0]) / exp(B[0]) - vi) * (exp(K_b[0]) / exp(B[0])) ^ 2

	exp(pie[0]) * exp(K_b[0]) = (1 - delta_kb) * exp(K_b[-1]) / exp(eps_K_b[0]) + exp(j_B[-1])

	gamma_b * exp(d_b[0]) = exp(d_p[0]) * gamma_p

	gamma_b * exp(b_h[0]) = exp(b_i[0]) * gamma_i

	gamma_b * exp(b_e[0]) = exp(b_ee[0]) * gamma_e

	exp(b_h[0]) + exp(b_e[0]) = exp(K_b[0]) + exp(d_b[0])

	kappa_d * beta_p * exp(lam_p[1]) / exp(lam_p[0]) * (exp(r_d[1]) / exp(r_d[0]) - (exp(r_d[0]) / exp(r_d[-1])) ^ ind_d) * (exp(r_d[1]) / exp(r_d[0])) ^ 2 * exp(d_b[1]) / exp(d_b[0]) + exp(mk_d[0]) / (exp(mk_d[0]) - 1) - 1 - r_ib[0] * exp(mk_d[0]) / (exp(mk_d[0]) - 1) / exp(r_d[0]) - exp(r_d[0]) * kappa_d * (exp(r_d[0]) / exp(r_d[-1]) - (exp(r_d[-1]) / exp(r_d[-2])) ^ ind_d) / exp(r_d[-1]) = 0

	beta_p * exp(lam_p[1]) / exp(lam_p[0]) * kappa_be * (exp(r_be[1]) / exp(r_be[0]) - (exp(r_be[0]) / exp(r_be[-1])) ^ ind_be) * (exp(r_be[1]) / exp(r_be[0])) ^ 2 * exp(b_e[1]) / exp(b_e[0]) + 1 - exp(mk_be[0]) / (exp(mk_be[0]) - 1) + exp(R_b[0]) * exp(mk_be[0]) / (exp(mk_be[0]) - 1) / exp(r_be[0]) - exp(r_be[0]) * kappa_be * (exp(r_be[0]) / exp(r_be[-1]) - (exp(r_be[-1]) / exp(r_be[-2])) ^ ind_be) / exp(r_be[-1]) = 0

	beta_p * exp(lam_p[1]) / exp(lam_p[0]) * kappa_bh * (exp(r_bh[1]) / exp(r_bh[0]) - (exp(r_bh[0]) / exp(r_bh[-1])) ^ ind_bh) * (exp(r_bh[1]) / exp(r_bh[0])) ^ 2 * exp(b_h[1]) / exp(b_h[0]) + 1 - exp(mk_bh[0]) / (exp(mk_bh[0]) - 1) + exp(R_b[0]) * exp(mk_bh[0]) / (exp(mk_bh[0]) - 1) / exp(r_bh[0]) - exp(r_bh[0]) * kappa_bh * (exp(r_bh[0]) / exp(r_bh[-1]) - (exp(r_bh[-1]) / exp(r_bh[-2])) ^ ind_bh) / exp(r_bh[-1]) = 0

	exp(j_B[0]) = exp(r_bh[0]) * exp(b_h[0]) + exp(r_be[0]) * exp(b_e[0]) - exp(r_d[0]) * exp(d_b[0]) - exp(d_b[0]) * exp(r_d[0]) * kappa_d / 2 * (exp(r_d[0]) / exp(r_d[-1]) - 1) ^ 2 - exp(b_e[0]) * exp(r_be[0]) * kappa_be / 2 * (exp(r_be[0]) / exp(r_be[-1]) - 1) ^ 2 - exp(b_h[0]) * exp(r_bh[0]) * kappa_bh / 2 * (exp(r_bh[0]) / exp(r_bh[-1]) - 1) ^ 2 - exp(K_b[0]) * kappa_kb / 2 * (exp(K_b[0]) / exp(B[0]) - vi) ^ 2

	exp(J_R[0]) = exp(Y[0]) * (1 - 1 / exp(x[0]) - kappa_p / 2 * (exp(pie[0]) - exp(pie[-1]) ^ ind_p * piss ^ (1 - ind_p)) ^ 2)

	1 - exp(eps_y[0]) + exp(eps_y[0]) / exp(x[0]) - exp(pie[0]) * kappa_p * (exp(pie[0]) - exp(pie[-1]) ^ ind_p * piss ^ (1 - ind_p)) + exp(pie[1]) * beta_p * exp(lam_p[1]) / exp(lam_p[0]) * kappa_p * (exp(pie[1]) - piss ^ (1 - ind_p) * exp(pie[0]) ^ ind_p) * exp(Y[1]) / exp(Y[0]) = 0

	exp(C[0]) = exp(c_p[0]) * gamma_p + exp(c_i[0]) * gamma_i + exp(c_e[0]) * gamma_e

	exp(BH[0]) = gamma_b * exp(b_h[0])

	exp(BE[0]) = gamma_b * exp(b_e[0])

	exp(B[0]) = exp(BH[0]) + exp(BE[0])

	exp(D[0]) = exp(d_p[0]) * gamma_p

	exp(Y[0]) = exp(y_e[0]) * gamma_e

	exp(J_B[0]) = gamma_b * exp(j_B[0])

	exp(l_pd[0]) * gamma_e = exp(l_p[0]) * gamma_p

	exp(l_id[0]) * gamma_e = exp(l_i[0]) * gamma_i

	h = exp(h_p[0]) * gamma_p + exp(h_i[0]) * gamma_i

	exp(K[0]) = exp(k_e[0]) * gamma_e

	exp(Y1[0]) = exp(C[0]) + exp(K[0]) - (1 - deltak) * exp(K[-1])

	exp(PIW[0]) = exp(pie[0]) * (exp(w_p[0]) + exp(w_i[0])) / (exp(w_p[-1]) + exp(w_i[-1]))

	1 + r_ib[0] = (1 + r_ib_ss) ^ (1 - rho_ib) * (1 + r_ib[-1]) ^ rho_ib * ((exp(pie[0]) / piss) ^ phi_pie * (exp(Y1[0]) / exp(Y1[-1])) ^ phi_y) ^ (1 - rho_ib) * (1 + e_r_ib[x])

	exp(ee_z[0]) = 1 - rho_ee_z + rho_ee_z * exp(ee_z[-1]) + e_z[x]

	exp(A_e[0]) = 1 - rho_A_e + rho_A_e * exp(A_e[-1]) + e_A_e[x]

	exp(ee_j[0]) = 1 - rho_ee_j + rho_ee_j * exp(ee_j[-1]) - e_j[x]

	exp(m_i[0]) = (1 - rho_mi) * m_i_ss + rho_mi * exp(m_i[-1]) + e_mi[x]

	exp(m_e[0]) = (1 - rho_me) * m_e_ss + rho_me * exp(m_e[-1]) + e_me[x]

	exp(mk_d[0]) = (1 - rho_mk_d) * mk_d_ss + rho_mk_d * exp(mk_d[-1]) + e_mk_d[x]

	exp(mk_be[0]) = (1 - rho_mk_be) * mk_be_ss + rho_mk_be * exp(mk_be[-1]) + e_mk_be[x]

	exp(mk_bh[0]) = (1 - rho_mk_bh) * mk_bh_ss + rho_mk_bh * exp(mk_bh[-1]) + e_mk_bh[x]

	exp(ee_qk[0]) = 1 - rho_ee_qk + rho_ee_qk * exp(ee_qk[-1]) + e_qk[x]

	exp(eps_y[0]) = (1 - rho_eps_y) * eps_y_ss + rho_eps_y * exp(eps_y[-1]) + e_y[x]

	exp(eps_l[0]) = (1 - rho_eps_l) * eps_l_ss + rho_eps_l * exp(eps_l[-1]) + e_l[x]

	exp(eps_K_b[0]) = 1 - rho_eps_K_b + rho_eps_K_b * exp(eps_K_b[-1]) + e_eps_K_b[x]

	rr_e[0] = exp(lam_e[0]) - beta_e * exp(lam_e[1]) * (1 + exp(r_be[0])) / exp(pie[1])

	aux1[0] = exp(K_b[0]) / exp(B[0])

	exp(bm[0]) = exp(r_bh[-1]) * exp(b_h[-1]) / (exp(b_h[-1]) + exp(b_e[-1])) + exp(r_be[-1]) * exp(b_e[-1]) / (exp(b_h[-1]) + exp(b_e[-1])) - exp(r_d[-1])

	exp(spr_b[0]) = exp(r_bh[0]) * 0.5 + exp(r_be[0]) * 0.5 - exp(r_d[0])
end

@parameters GNSS_2010 begin
	beta_p = 0.9943

	beta_i = 0.975

	beta_b = beta_p

	beta_e = beta_i

	j = 0.2

	phi = 1.0

	m_i_ss = 0.7

	m_e_ss = 0.35

	alpha = 0.250

	eps_d = -1.46025

	eps_bh = 2.932806

	eps_be = 2.932806

	mk_d_ss = eps_d/(eps_d-1)

	mk_bh_ss = eps_bh/(eps_bh-1)

	mk_be_ss = eps_be/(eps_be-1)

	book_ss = 0 

	eps_y_ss = 6

	eps_l_ss = 5

	gamma_p = 1

	gamma_i = 1

	ni = 0.8

	gamma_b = 1

	gamma_e = 1

	deltak = 0.025

	piss = 1

	r_ib_ss = (eps_d-1)*(piss/beta_p-1)/eps_d

	r_be_ss = eps_be*r_ib_ss/(eps_be-1)

	r_bh_ss = eps_bh*r_ib_ss/(eps_bh-1)

	r_k_ss = (-(1-deltak))-piss*(1-deltak)*m_e_ss/beta_e*(1/(1+r_be_ss)-beta_e/piss)+1/beta_e

	h = 1

	eksi_1 = r_k_ss

	eksi_2 = r_k_ss*0.1

	vi = 0.09

	eps_b = (eps_bh + eps_be) / 2

	delta_kb = r_ib_ss/vi*(eps_d-eps_b+eps_d*vi*(eps_b-1))/((eps_d-1)*(eps_b-1))

	ind_d = 0.0

	ind_be = 0.0

	ind_bh = 0.0

	rho_ee_z = 0.385953438168178

	rho_A_e = 0.93816527333294

	rho_ee_j = 0.921872719102206

	rho_me = 0.90129485520182

	rho_mi = 0.922378382753078

	rho_mk_d = 0.892731352899547

	rho_mk_bh = 0.851229673864555

	rho_mk_be = 0.873901213475799

	rho_ee_qk = 0.571692383714171

	rho_eps_y = 0.294182239567384

	rho_eps_l = 0.596186440884132

	rho_eps_K_b = 0.813022758608552

	kappa_p = 33.7705265016395

	kappa_w = 107.352040072465

	kappa_i = 10.0305562248008

	kappa_d = 2.77537377104213

	kappa_be = 7.98005959044637

	kappa_bh = 9.04426718749482

	kappa_kb = 8.91481958034669

	phi_pie = 2.00384780180824

	rho_ib = 0.750481873084311

	phi_y = 0.303247771697294

	ind_p = 0.158112794106546

	ind_w = 0.300197804017489

	a_i = 0.867003766306404

	a_e = 0.0

	a_p = 0.0
end

get_solution(GNSS_2010,verbose = true)