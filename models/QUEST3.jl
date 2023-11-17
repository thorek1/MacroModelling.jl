@model QUEST3 begin
	interest[0] = ((1 + E_INOM[0]) ^ 4 - interestq_exog ^ 4) / interestq_exog ^ 4

	inflation[0] = 0.25 * (inflationq[0] + inflationq[-1] + inflationq[-2] + inflationq[-3])

	inflationq[0] = (1 + 4 * E_PHIC[0] - inflationannual_exog) / inflationannual_exog

	outputgap[0] = E_LYGAP[0]

	E_INOM[0] = ILAGE * E_INOM[-1] + (1 - ILAGE) * (E_EX_R + GP0 + TINFE * (E_PHIC[0] - GP0) + TYE1 * E_LYGAP[-1]) + TYE2 * (E_LYGAP[0] - E_LYGAP[-1]) + E_ZEPS_M[0]

	exp(E_LUCYN[0]) = exp(E_ZEPS_C[0]) * (exp(E_LCNLCSN[0]) * (1 - HABE / (1 + E_GCNLC[0] - GY0))) ^ (-SIGC) * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (1 - SIGC)

	exp(E_LUCLCYN[0]) = (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (1 - SIGC) * exp(E_LCLCSN[0]) ^ (-SIGC)

	E_VL[0] = exp(E_ZEPS_L[0]) * OMEGE * KAPPAE * exp(E_ZEPS_C[0]) * (exp(E_LCNLCSN[0]) * (1 - HABE / (1 + E_GCNLC[0] - GY0))) ^ (1 - SIGC) * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (-SIGC) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ (KAPPAE - 1)

	E_VLLC[0] = exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ (KAPPAE - 1) * OMEGE * KAPPAE * (1 - OMEGE * exp(E_ZEPS_L[0]) * (exp(E_LL[0]) - HABLE * exp(E_LL[-1])) ^ KAPPAE) ^ (-SIGC) * exp(E_LCLCSN[0]) ^ (1 - SIGC)

	1 / BETAE - 1 = E_INOM[0] + E_GUC[1] - E_PHIC[1]

	E_LUCYN[0] - E_LUCYN[-1] = E_GUC[0] + SIGC * (E_GY[0] - GY0 - E_PHIC[0] + E_PHI[0])

	exp(E_LCLCSN[0]) * (1 + TVAT) = (1 - E_TW[0] - SSC) * E_WS[0] + E_WS[0] * E_TRW[0] - E_TAXYN[0]

	E_WS[0] = exp(E_LL[0] - E_LYWR[0])

	exp(E_LCSN[0]) = exp(E_LCLCSN[0]) * SLCFLAG * SLC + exp(E_LCNLCSN[0]) * (1 - SLCFLAG * SLC)

	(1 + TVAT) * ((E_VL[0] * (1 - SLC) + E_VLLC[0] * SLC) / (exp(E_LUCYN[0]) * (1 - SLC) + exp(E_LUCLCYN[0]) * SLC)) ^ (1 - WRLAG) * ((1 - E_TW[0] - SSC) / (1 + TVAT) * (THETAE - 1) / THETAE / exp(E_LYWR[-1]) / (1 + E_GY[0] - GY0)) ^ WRLAG = (1 - E_TW[0] - SSC) * (THETAE - 1) / THETAE / exp(E_LYWR[0]) + GAMIFLAG * GAMWE / THETAE / exp(E_LYWR[0]) * (E_WPHI[0] - GP0 - GY0 - (1 - SFWE) * (E_PHI[-1] - GP0)) - GAMWE * BETAE * GAMIFLAG / THETAE / exp(E_LYWR[0]) * (E_WPHI[1] - GP0 - GY0 - (1 - SFWE) * (E_PHI[0] - GP0))

	(1 + E_ZEPS_W[0]) / exp(E_LYWR[0]) = E_ETA[0] * ALPHAE / exp(E_LL[0]) * (1 + E_LOL[0]) - 1 / exp(E_LYWR[0]) * GAMLE * (E_LL[0] - E_LL[-1]) + GAMLE * 1 / exp(E_LYWR[1]) * (1 + E_GY[1] - GY0) / (1 + E_R[0]) * (E_LL[1] - E_LL[0])

	GAMIE * (exp(E_LIK[0]) - (GY0 + DELTAE + GPOP0 + GPCPI0)) + GAMI2E * (E_GI[0] - GY0 - GPCPI0) - GAMI2E / (1 + E_INOM[0]) * (E_GI[1] - GY0 - GPCPI0) = E_Q[0] - 1

	E_ETA[0] * (1 - TP) * (1 - ALPHAE) * exp(E_LYKPPI[0]) = E_Q[0] - (GPCPI0 + 1 - E_R[0] - DELTAE - RPREMK - E_ZEPS_RPREMK[0] - E_PHIPI[1]) * E_Q[1] + (1 - TP) * (A1E * (E_UCAP[0] - UCAP0) + A2E * (E_UCAP[0] - UCAP0) ^ 2)

	exp(E_LYKPPI[0]) * E_ETA[0] * (1 - ALPHAE) = E_UCAP[0] * (A1E + (E_UCAP[0] - UCAP0) * 2 * A2E)

	E_ETA[0] = 1 - (TAUE + E_ZEPS_ETA[0]) - GAMIFLAG * GAMPE * (BETAE * (SFPE * E_PHI[1] + E_PHI[-1] * (1 - SFPE) - GP0) - (E_PHI[0] - GP0))

	E_MRY[0] = (1 + E_INOM[0]) ^ (-ZETE)

	E_GK[0] - (GY0 + GPCPI0) = exp(E_LIK[0]) - (GY0 + DELTAE + GPOP0 + GPCPI0)

	E_GKG[0] - (GY0 + GPCPI0) = exp(E_LIKG[0]) - (GPCPI0 + GY0 + GPOP0 + DELTAGE)

	E_LISN[0] = GPCPI0 + GY0 + E_LIK[0] - E_LYKPPI[0] - E_GK[0]

	E_GY[0] = (1 - ALPHAE) * (E_GK[0] + E_GUCAP[0]) + ALPHAE * (E_GTFP[0] + E_GL[0] * (1 + LOL)) + E_GKG[0] * (1 - ALPHAGE)

	E_R[0] = E_INOM[0] - E_PHI[1]

	E_LYGAP[0] = (1 - ALPHAE) * (log(E_UCAP[0]) - log(E_UCAP0[0])) + ALPHAE * (E_LL[0] - E_LL0[0])

	E_LL0[0] = RHOL0 * E_LL0[-1] + E_LL[0] * (1 - RHOL0)

	E_UCAP0[0] = RHOUCAP0 * E_UCAP0[-1] + E_UCAP[0] * (1 - RHOUCAP0)

	exp(E_LPCP[0]) = (SE + (1 - SE) * exp(E_LPMP[0]) ^ (1 - SIGIME)) ^ (1 / (1 - SIGIME))

	exp(E_LPMP[0]) = (1 + E_ZEPS_ETAM[0] + GAMIFLAG * GAMPME * (GP0 + BETAE * (SFPME * E_PHIM[1] + (1 - SFPME) * E_PHIM[-1] - GP0) - E_PHIM[0])) * exp(E_LER[0]) ^ ALPHAX

	exp(E_LPXP[0]) = 1 + E_ZEPS_ETAX[0] + GAMIFLAG * GAMPXE * (GP0 + BETAE * (SFPXE * E_PHIX[1] + (1 - SFPXE) * E_PHIX[-1] - GP0) - E_PHIX[0])

	1 = exp(E_LCSN[0]) + exp(E_LISN[0]) + exp(E_LIGSN[0]) + exp(E_LGSN[0]) + E_TBYN[0]

	E_TBYN[0] = exp(E_LEXYN[0]) - exp(E_LIMYN[0]) + E_ZEPS_EX[0]

	exp(E_LIMYN[0]) = (exp(E_LCSN[0]) + exp(E_LISN[0]) + exp(E_LIGSN[0]) + exp(E_LGSN[0])) * (1 - SE) * exp(RHOPCPM * (E_LPCP[-1] - E_LPMP[-1]) + (1 - RHOPCPM) * (E_LPCP[0] - E_LPMP[0])) ^ SIGIME * exp(E_LPMP[0] - E_LPCP[0])

	exp(E_LEXYN[0]) = exp(E_LPXP[0]) * (1 - SE) * exp(RHOPWPX * (E_LER[-1] * SE * ALPHAX - E_LPXP[-1]) + (1 - RHOPWPX) * (E_LER[0] * SE * ALPHAX - E_LPXP[0])) ^ SIGEXE * exp(E_LYWY[0]) ^ ALPHAX

	E_INOM[0] = E_INOMW[0] + E_GE[1] - RPREME * E_BWRY[0] + E_ZEPS_RPREME[0]

	E_BWRY[0] = E_TBYN[0] + (1 + E_INOM[0] - E_PHI[1] - E_GY[0] - GPOP0) * E_BWRY[-1]

	exp(E_LBGYN[0]) = exp(E_LIGSN[0]) + exp(E_LGSN[0]) + (1 + E_R[0] - E_GY[0] - GPOP0) * exp(E_LBGYN[-1]) + E_TRW[0] * exp(E_LL[0] - E_LYWR[0]) - E_WS[0] * (E_TW[0] + SSC) - TP * (1 - E_WS[0]) - TVAT * exp(E_LCSN[0]) - E_TAXYN[0]

	E_GG[0] - GY0 = GSLAG * (E_GG[-1] - GY0) + GFLAG * GVECM * (E_LGSN[-1] - log(GSN)) + (E_LYGAP[0] - E_LYGAP[-1]) * GFLAG * G1E + GFLAG * GEXOFLAG * E_ZEPS_G[0] + (1 - GFLAG) * (E_ZEPS_G[0] - E_ZEPS_G[-1])

	E_GIG[0] - GY0 - GPCPI0 = IGSLAG * (E_GIG[-1] - GY0 - GPCPI0) + IGFLAG * IGVECM * (E_LIGSN[-1] - log(IGSN)) + (E_LYGAP[0] - E_LYGAP[-1]) * IGFLAG * IG1E + IGFLAG * IGEXOFLAG * E_ZEPS_IG[0] + (1 - IGFLAG) * (E_ZEPS_IG[0] - E_ZEPS_IG[-1])

	E_GIG[0] - E_GI[0] = E_LIGSN[0] - E_LISN[0] - E_LIGSN[-1] + E_LISN[-1]

	E_TRW[0] = TRSN + TRFLAG * TR1E * (1 - exp(E_LL[0]) - (1 - L0)) + E_ZEPS_TR[0]

	E_TAXYN[0] - E_TAXYN[-1] = BGADJ1 * (exp(E_LBGYN[-1]) - BGTAR) + BGADJ2 * (exp(E_LBGYN[0]) - exp(E_LBGYN[-1]))

	E_TW[0] = TW0 * (1 + E_LYGAP[0] * TW1 * TWFLAG)

	E_TRYN[0] = E_TRW[0] * exp(E_LL[0] - E_LYWR[0])

	E_TRTAXYN[0] = E_TRW[0] * exp(E_LL[0] - E_LYWR[0]) - E_TAXYN[0]

	E_WSW[0] = (1 - E_TW[0] - SSC) * E_WS[0]

	E_INOMW[0] = (1 - RII) * E_EX_INOMW + RII * E_INOMW[-1] + RIP * (E_PHIW[-1] - GPW0) + RIX * (E_GYW[-1] - GYW0) + E_EPS_INOMW[x]

	E_PHIW[0] - GPW0 = RPI * (E_INOMW[-1] - E_EX_INOMW) + (E_PHIW[-1] - GPW0) * RPP + (E_GYW[-1] - GYW0) * RPX + E_EPS_PW[x]

	E_GYW[0] - GYW0 = (E_INOMW[-1] - E_EX_INOMW) * RXI + (E_PHIW[-1] - GPW0) * RXP + (E_GYW[-1] - GYW0) * RXX + RXY * (E_LYWY[-1] - LYWY0) + E_EPS_YW[x]

	E_LYWY[0] - E_LYWY[-1] = E_GYW[0] - E_GY[0]

	E_GTFP[0] - GTFP0 = E_EPS_Y[x]

	E_LOL[0] - LOL = RHOLOL * (E_LOL[-1] - LOL) + E_EPS_LOL[x]

	E_PHIPI[0] = GPCPI0 + E_ZEPS_PPI[0]

	E_ZEPS_C[0] = RHOCE * E_ZEPS_C[-1] + E_EPS_C[x]

	E_ZEPS_ETA[0] = RHOETA * E_ZEPS_ETA[-1] + E_EPS_ETA[x]

	E_ZEPS_ETAM[0] = RHOETAM * E_ZEPS_ETAM[-1] + E_EPS_ETAM[x]

	E_ZEPS_ETAX[0] = RHOETAX * E_ZEPS_ETAX[-1] + E_EPS_ETAX[x]

	E_ZEPS_EX[0] = RHOEXE * E_ZEPS_EX[-1] + E_EPS_EX[x]

	E_ZEPS_G[0] = E_ZEPS_G[-1] * RHOGE + E_EPS_G[x]

	E_ZEPS_IG[0] = E_ZEPS_IG[-1] * RHOIG + IGEXOFLAG * E_EPS_IG[x]

	E_ZEPS_L[0] = RHOLE * E_ZEPS_L[-1] + E_EPS_L[x]

	E_ZEPS_M[0] = E_EPS_M[x]

	E_ZEPS_PPI[0] = E_EPS_PPI[x] + RHOPPI1 * E_ZEPS_PPI[-1] + RHOPPI2 * E_ZEPS_PPI[-2] + RHOPPI3 * E_ZEPS_PPI[-3] + RHOPPI4 * E_ZEPS_PPI[-4]

	E_ZEPS_RPREME[0] = RHORPE * E_ZEPS_RPREME[-1] + E_EPS_RPREME[x]

	E_ZEPS_RPREMK[0] = RHORPK * E_ZEPS_RPREMK[-1] + E_EPS_RPREMK[x]

	E_ZEPS_W[0] = E_EPS_W[x]

	E_ZEPS_TR[0] = RHOTR * E_ZEPS_TR[-1] + TREXOFLAG * E_EPS_TR[x]

	E_PHIC[0] + E_GC[0] - E_GY[0] - E_PHI[0] = E_LCSN[0] - E_LCSN[-1]

	E_PHIC[0] + E_GCLC[0] - E_GY[0] - E_PHI[0] = E_LCLCSN[0] - E_LCLCSN[-1]

	E_PHIC[0] + E_GCNLC[0] - E_GY[0] - E_PHI[0] = E_LCNLCSN[0] - E_LCNLCSN[-1]

	E_PHIW[0] + E_GE[0] - E_PHI[0] = E_LER[0] - E_LER[-1]

	E_PHIX[0] + E_GEX[0] - E_GY[0] - E_PHI[0] = E_LEXYN[0] - E_LEXYN[-1]

	E_PHIC[0] + E_GG[0] - E_GY[0] - E_PHI[0] = E_LGSN[0] - E_LGSN[-1]

	E_GI[0] - E_GK[-1] = E_LIK[0] - E_LIK[-1]

	E_GIG[0] - E_GKG[-1] = E_LIKG[0] - E_LIKG[-1]

	E_PHIM[0] + E_GIM[0] - E_GY[0] - E_PHI[0] = E_LIMYN[0] - E_LIMYN[-1]

	E_PHIPI[0] + E_GY[0] - E_GK[0] = E_LYKPPI[0] - E_LYKPPI[-1]

	E_GL[0] = E_LL[0] - E_LL[-1]

	E_GTAX[0] - E_GY[0] - E_PHI[0] = log(E_TAXYN[0]/E_TAXYN[-1])

	E_GTFPUCAP[0] = (1 - ALPHAE) * E_GUCAP[0] + ALPHAE * E_GTFP[0]

	E_GTR[0] - E_GL[0] - E_WRPHI[0] = log(E_TRW[0]/E_TRW[-1])

	E_GUCAP[0] = log(E_UCAP[0]/E_UCAP[-1])

	E_GWRY[0] = E_LYWR[-1] - E_LYWR[0]

	E_GY[0] - E_GYPOT[0] = E_LYGAP[0] - E_LYGAP[-1]

	E_BGYN[0] = exp(E_LBGYN[0])

	E_DBGYN[0] = E_BGYN[0] - E_BGYN[-1]

	E_CLCSN[0] = exp(E_LCLCSN[0])

	E_GSN[0] = exp(E_LGSN[0])

	E_LTRYN[0] = log(E_TRYN[0])

	E_PHIC[0] - E_PHI[0] = E_LPCP[0] - E_LPCP[-1]

	E_PHIM[0] - E_PHI[0] = E_LPMP[0] - E_LPMP[-1]

	E_PHIX[0] - E_PHI[0] = E_LPXP[0] - E_LPXP[-1]

	E_PHI[0] + E_GY[0] - E_WPHI[0] = E_LYWR[0] - E_LYWR[-1]

	E_WPHI[0] = E_PHI[0] + E_WRPHI[0]

	E_GYL[0] = E_GY[0] + GPOP0

	E_GCL[0] = GPOP0 + E_GC[0]

	E_GIL[0] = GPOP0 + E_GI[0]

	E_GGL[0] = GPOP0 + E_GG[0]

	E_GEXL[0] = GPOP0 + E_GEX[0] + DGEX

	E_GIML[0] = GPOP0 + E_GIM[0] + DGIM

	E_PHIML[0] = E_PHIM[0] + DGPM

	E_PHIXL[0] = E_PHIX[0] + DGPX

	E_LCY[0] = E_LCSN[0] - E_LPCP[0]

	E_LGY[0] = E_LGSN[0] - E_LPCP[0]

	E_LWS[0] = E_LL[0] - E_LYWR[0]

end


@parameters QUEST3 begin
	A2E = 0.0453

	G1E = (-0.0754)

	GAMIE = 76.0366

	GAMI2E = 1.1216

	GAMLE = 58.2083

	GAMPE = 61.4415

	GAMPME = 1.6782

	GAMPXE = 26.1294

	GAMWE = 1.2919

	GSLAG = (-0.4227)

	GVECM = (-0.1567)

	HABE = 0.5634

	HABLE = 0.8089

	IG1E = 0.1497

	IGSLAG = 0.4475

	IGVECM = (-0.1222)

	ILAGE = 0.9009

	KAPPAE = 1.9224

	RHOCE = 0.9144

	RHOETA = 0.1095

	RHOETAM = 0.9557

	RHOETAX = 0.8109

	RHOGE = 0.2983

	RHOIG = 0.8530

	RHOLE = 0.9750

	RHOL0 = 0.9334

	RHOPCPM = 0.6652

	RHOPWPX = 0.2159

	RHORPE = 0.9842

	RHORPK = 0.9148

	RHOUCAP0 = 0.9517

	RPREME = 0.0200

	RPREMK = 0.0245

	SE = 0.8588

	SFPE = 0.8714

	SFPME = 0.7361

	SFPXE = 0.9180

	SFWE = 0.7736

	SIGC = 4.0962

	SIGEXE = 2.5358

	SIGIME = 1.1724

	SLC = 0.3507

	TINFE = 1.9590

	TR1E = 0.9183

	RHOTR = 0.8636

	TYE1 = 0.4274

	TYE2 = 0.0783

	WRLAG = 0.2653

	ALPHAX = 0.5

	BETAE = 0.996

	ALPHAE = 0.52

	ALPHAGE = 0.9

	BGADJ2 = 0.004

	BGADJ1 = 0.001*BGADJ2

	BGTAR = 2.4

	DELTAE = 0.025

	DELTAGE = 0.0125

	DGIM = 0.00738619107021

	DGEX = 0.00738619107021

	DGPM = (-0.00396650612294)

	DGPX = (-0.00396650612294)

	LOL = 0

	RHOEXE = 0.975

	RHOLOL = 0.99

	SSC = 0.2

	TAUE = 0.1000

	THETAE = 1.6

	TP = 0.2

	TRSN = 0.36

	TVAT = 0.2

	TW0 = 0.2

	TW1 = 0.8

	ZETE = 0.4000

	GFLAG = 1

	IGFLAG = 1

	TRFLAG = 1

	TWFLAG = 1

	GEXOFLAG = 1

	IGEXOFLAG = 1

	TREXOFLAG = 1

	SLCFLAG = 1

	GAMIFLAG = 1

	A1E = 0.0669

	OMEGE = 1.4836

	GSN = 0.203

	IGSN = 0.025

	GPCPI0 = 0

	GP0 = 0.005

	GPW0 = GP0

	GPOP0 = 0.00113377677398

	GY0 = 0.003

	GTFP0 = (ALPHAE+ALPHAGE-1)/ALPHAE*GY0-(2-ALPHAE-ALPHAGE)/ALPHAE*GPCPI0

	GYW0 = GY0

	UCAP0 = 1

	L0 = 0.65

	E_EX_R = 1/BETAE-1

	E_EX_INOMW = GP0+E_EX_R

	LYWY0 = 0

	RXY = (-0.0001)

	RII = 0.887131978334279

	RIP = 0.147455589872832

	RIX = 0.120095599681076

	RPI = 0.112067224979767

	RPP = 0.502758194258928

	RPX = 0.082535400836409

	RXI = 0.073730131176521

	RXP = (-0.302655015002645)

	RXX = 0.495000246553550

	RHOPPI1 = 0.24797097628284

	RHOPPI2 = 0.13739098460472

	RHOPPI3 = 0.10483962746747

	RHOPPI4 = 0.09282876044442

	interestq_exog = 1.00901606

	inflationannual_exog = 1.02

end
