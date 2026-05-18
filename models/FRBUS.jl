# FRB/US LINVER (2024) model
# Source: https://www.federalreserve.gov/econres/us-models-linver.htm
# Settings:
#   expvers      = "mcap"
#   mprule       = "intay"
#   elb_imposed  = "no"

@model FRBUS begin
    delrff[0] = rff[0] - rff[-1]

    dpadj[0] = dpadj[-1] + dpgap[-1]

    dpgap[0] = y_dpgap_1 * pipxnc[0] + y_dpgap_2 * phr_l[0] - pxp_l[0] + y_dpgap_3 * phr_l[-1] + pxp_l[-1] + y_dpgap_4 * pbfir_l[0] + y_dpgap_5 * pbfir_l[-1] + y_dpgap_6 * pegfr_l[0] + y_dpgap_7 * pegfr_l[-1] + y_dpgap_8 * pegsr_l[0] + y_dpgap_9 * pegsr_l[-1] + y_dpgap_10 * pxr_l[0] + y_dpgap_11 * pxr_l[-1]

    ebfi_l[0] = y_ebfi_l_8 * hgpbfir[-1] + y_ebfi_l_6 * xb_l[-1] + y_ebfi_l_5 * zebfi[0] + y_ebfi_l_1 * ebfi_l[-1] + ebfi_l_aerr[x] + y_ebfi_l_2 * qebfi_l[-1] + y_ebfi_l_3 * ebfi_l[-2] + y_ebfi_l_4 * ebfi_l[-3] + y_ebfi_l_7 * xb_l[-2]

    ebfin_l[0] = pxp_l[0] + pbfir_l[0] + ebfi_l[0]

    # ec_l[0] = ec_l[-1] + y_ec_l_1 * eco_l[0] + y_ec_l_2 * eco_l[-1] + y_ec_l_3 * ech_l[0] + y_ec_l_4 * ech_l[-1] + y_ec_l_5 * yhpcd_l[0] + y_ec_l_6 * jkcd_l[0] + y_ec_l_7 * yhpcd_l[-1] + y_ec_l_8 * jkcd_l[-1]

    ecd_l[0] = y_ecd_l_4 * zgapc2[0] + zecd[0] + y_ecd_l_1 * ecd_l[-1] + ecd_l_aerr[x] + y_ecd_l_2 * qecd_l[-1] + y_ecd_l_3 * ecd_l[-2]

    ech_l[0] = y_ech_l_3 * ech_l_aerr[x] + kh_l[-1] + ech_l[-1] * y_ech_l_1 + y_ech_l_2 * kh_l[-2] + y_ech_l_4 * ech_l[-2] + y_ech_l_5 * kh_l[-3]

    ecnia_l[0] = ecnia_l[-1] + eco_l[0] * y_ecnia_l_1 + eco_l[-1] * y_ecnia_l_2 + ecd_l[0] * y_ecnia_l_3 + ecd_l[-1] * y_ecnia_l_4 + ech_l[0] * y_ecnia_l_5 + ech_l[-1] * y_ecnia_l_6

    ecnian_l[0] = ecnia_l[0] + pcnia_l[0]

    eco_l[0] = y_eco_l_8 * yht_l[-1] + y_eco_l_7 * yhl_l[-1] + y_eco_l_6 * yht_l[0] + y_eco_l_5 * yhl_l[0] + y_eco_l_4 * zeco[0] + eco_l[-1] * y_eco_l_1 + eco_l_aerr[x] + y_eco_l_2 * qeco_l[-1] + y_eco_l_3 * eco_l[-2]

    egfe_l[0] = fiscal_egfe * fiscal[0] + y_egfe_l_7 * xgap2[-1] + y_egfe_l_6 * xgap2[0] + y_egfe_l_5 * egfet_l[0] + y_egfe_l_1 * egfe_l[-1] + egfe_l_aerr[x] + y_egfe_l_2 * egfet_l[-1] + y_egfe_l_3 * egfe_l[-2] + y_egfe_l_4 * egfe_l[-3]

    egfen_l[0] = egfe_l[0] + pxp_l[0] + pegfr_l[0]

    egfet_l[0] = egfet_l[-1] * y_egfet_l_1 + pegfr_l[-1] * y_egfet_l_2 + pxp_l[-1] * y_egfet_l_3 + y_egfet_l_4 * xgdptn_l[-1] + y_egfet_l_5 * hggdpt[0] + y_egfet_l_6 * hggdpt[-1] + y_egfet_l_7 * hggdpt[-2] + y_egfet_l_8 * hggdpt[-3]

    egfl_l[0] = fiscal_egfl * fiscal[0] + xgap2[-1] * y_egfl_l_7 + xgap2[0] * y_egfl_l_6 + y_egfl_l_5 * egflt_l[0] + y_egfl_l_1 * egfl_l[-1] + egfl_l_aerr[x] + y_egfl_l_2 * egflt_l[-1] + y_egfl_l_3 * egfl_l[-2] + y_egfl_l_4 * egfl_l[-3]

    egfln_l[0] = egfl_l[0] + pgfl_l[0]

    egflt_l[0] = egflt_l[-1] * y_egflt_l_1 + y_egflt_l_2 * pgfl_l[-1] + xgdptn_l[-1] * y_egflt_l_3 + hggdpt[0] * y_egflt_l_4 + hggdpt[-1] * y_egflt_l_5 + y_egflt_l_6 * hggdpt[-2] + y_egflt_l_7 * hggdpt[-3]

    egse_l[0] = xgap2[-1] * y_egse_l_7 + xgap2[0] * y_egse_l_6 + y_egse_l_5 * egset_l[0] + y_egse_l_1 * egse_l[-1] + egse_l_aerr[x] + y_egse_l_2 * egset_l[-1] + y_egse_l_3 * egse_l[-2] + y_egse_l_4 * egse_l[-3]

    egsen_l[0] = egse_l[0] + pxp_l[0] + pegsr_l[0]

    egset_l[0] = egset_l[-1] * y_egset_l_1 + pegsr_l[-1] * y_egset_l_2 + pxp_l[-1] * y_egset_l_3 + xgdptn_l[-1] * y_egset_l_4 + hggdpt[0] * y_egset_l_5 + hggdpt[-1] * y_egset_l_6 + y_egset_l_7 * hggdpt[-2] + y_egset_l_8 * hggdpt[-3]

    egsl_l[0] = xgap2[-1] * y_egsl_l_7 + xgap2[0] * y_egsl_l_6 + y_egsl_l_5 * egslt_l[0] + y_egsl_l_1 * egsl_l[-1] + egsl_l_aerr[x] + y_egsl_l_2 * egslt_l[-1] + y_egsl_l_3 * egsl_l[-2] + y_egsl_l_4 * egsl_l[-3]

    egsln_l[0] = egsl_l[0] + pgsl_l[0]

    egslt_l[0] = egslt_l[-1] * y_egslt_l_1 + y_egslt_l_2 * pgsl_l[-1] + xgdptn_l[-1] * y_egslt_l_3 + hggdpt[0] * y_egslt_l_4 + hggdpt[-1] * y_egslt_l_5 + y_egslt_l_6 * hggdpt[-2] + y_egslt_l_7 * hggdpt[-3]

    eh_l[0] = y_eh_l_7 * d83[x] + y_eh_l_5 * rme[-1] + zeh[0] + y_eh_l_1 * eh_l[-1] + eh_l_aerr[x] + y_eh_l_2 * qeh_l[-1] + y_eh_l_3 * eh_l[-2] + y_eh_l_4 * eh_l[-3] + y_eh_l_6 * rme[-2]

    ehn_l[0] = eh_l[0] + phr_l[0] + pxp_l[0]

    # em_l[0] = em_l[-1] + y_em_l_1 * emon_l[0] + y_em_l_2 * emn_l[0] + y_em_l_3 * emon_l[-1] + y_em_l_4 * emn_l[-1] + y_em_l_5 * emo_l[0] + y_em_l_6 * emo_l[-1] + y_em_l_7 * empn_l[0] + y_em_l_8 * empn_l[-1] + y_em_l_9 * emp_l[0] + y_em_l_10 * emp_l[-1]

    emn_l[0] = emon_l[0] * y_emn_l_2 + empn_l[0] * y_emn_l_3

    emo_l[0] = y_emo_l_9 * ddockm[x-1] + y_emo_l_8 * ddockm[x] + y_emo_l_7 * xgap2[-2] + xgap2[-1] * y_emo_l_6 + xgap2[0] * y_emo_l_5 + y_emo_l_4 * xgdpn_l[-1] + emo_l[-1] * y_emo_l_1 + emo_l̃[0] + y_emo_l_2 * pmo_l[-1] + y_emo_l_3 * uemot[x-1]

    emo_l̃[0] = (1 - rho_emo_l) * emo_l̄ + rho_emo_l * emo_l̃[-1] + emo_l_aerr[x]

    emon_l[0] = emo_l[0] + pmo_l[0]

    emp_l[0] = xgdp_l[0] + emp_l_aerr[x] + y_emp_l_1 * emptrt[x] + y_emp_l_2 * pmp_l[0] + y_emp_l_3 * pxb_l[0] + y_emp_l_4 * pmp_l[-1] + y_emp_l_5 * pxb_l[-1] + xgap2[-1] * y_emp_l_6

    empn_l[0] = emp_l[0] + pmp_l[0]

    ex_l[0] = y_ex_l_10 * ddockx[x] + y_ex_l_1 * ex_l[-1] + ex_l_aerr[x] + pxr_l[-1] * y_ex_l_2 + pxp_l[-1] * y_ex_l_3 + y_ex_l_4 * fpx_l[-1] + y_ex_l_5 * fgdp_l[-1] + y_ex_l_6 * fpc_l[-1] + y_ex_l_7 * fxgap[0] + y_ex_l_8 * fxgap[-1] + y_ex_l_9 * fxgap[-2]

    exn_l[0] = ex_l[0] + pxp_l[0] + pxr_l[0]

    fcbn_l[0] = exn_l[0] * y_fcbn_l_2 + emn_l[0] * y_fcbn_l_3 + y_fcbn_l_4 * fynicn_l[0] + y_fcbn_l_5 * fyniln_l[0] + y_fcbn_l_6 * ufcbr[x] + pxb_l[0] * y_fcbn_l_7 + y_fcbn_l_8 * xbt_l[0]

    fgdp_l[0] = fgdpt_l[0] + fxgap[0] * y_fgdp_l_2

    fgdpt_l[0] = y_fgdpt_l_1 * fgdpt_l[-1] + y_fgdpt_l_2 * xgdpt_l[-1] + hggdpt[0] * y_fgdpt_l_3 + hggdpt[-1] * y_fgdpt_l_4 + y_fgdpt_l_5 * hggdpt[-2] + y_fgdpt_l_6 * hggdpt[-3]

    fnicn_l[0] = y_fnicn_l_1 * fnicn_l[-1] + y_fnicn_l_2 * xgdptn_l[0] + y_fnicn_l_4 * fpc_l[0] + fpc_l[-1] * y_fnicn_l_5 + y_fnicn_l_6 * fpx_l[0] + fpx_l[-1] * y_fnicn_l_7 + y_fnicn_l_8 * rfnict[x]

    fniln_l[0] = y_fniln_l_1 * fniln_l[-1] + rfnict[x] * y_fniln_l_3 + xgdptn_l[0] * y_fniln_l_4 + fcbn_l[0] * y_fniln_l_5 + y_fniln_l_6 * pgdp_l[0] + y_fniln_l_7 * pgdp_l[-1] + fpx_l[0] * y_fniln_l_8 + fpx_l[-1] * y_fniln_l_9 + y_fniln_l_10 * fnirn_l[0]

    fnirn_l[0] = y_fnirn_l_2 * ufnir[x] + xgdpn_l[0]

    fpc_l[0] = fpc_l[-1] + y_fpc_l_2 * fpic[0]

    fpi10[0] = fxgap[-1] * y_fpi10_6 + y_fpi10_5 * fpitrg[x] + y_fpi10_1 * fpi10[-1] + y_fpi10_2 * fpi10[-2] + y_fpi10_3 * fpi10[-3] + y_fpi10_4 * fpi10[-4]

    fpi10t[0] = y_fpi10t_1 * fpi10t[-1] + fpi10[0] * y_fpi10t_2

    fpic[0] = fpi10[0] * y_fpic_1 + y_fpic_2 * fpic[-1]

    fpx_l[0] = fpc_l[0] + fpxr_l[0] - pcpi_l[0]

    fpxr_l[0] = fpxrr_l[0] + y_fpxr_l_1 * rg10[0] + y_fpxr_l_2 * zpi10f[0] + y_fpxr_l_3 * frl10[0] + fpi10t[0] * y_fpxr_l_4 + fnicn_l[0] * y_fpxr_l_5 + fniln_l[0] * y_fpxr_l_6 + xgdpn_l[0] * y_fpxr_l_7

    fpxrr_l[0] = y_fpxrr_l_4 * fpxrrt[x] + y_fpxrr_l_3 * fpxrr_l[-2] + y_fpxrr_l_1 * fpxrr_l[-1] + fpxrr_l̃[0] + y_fpxrr_l_2 * fpxrrt[x-1]

    fpxrr_l̃[0] =  (1 - rho_fpxrr_l) * fpxrr_l̄ + rho_fpxrr_l * fpxrr_l̃[-1] + fpxrr_l_aerr[x]

    frl10[0] = fxgap[-1] * y_frl10_6 + fxgap[0] * y_frl10_5 + y_frl10_4 * frs10[0] + y_frl10_1 * frl10[-1] + y_frl10_2 * frs10[-1] + y_frl10_3 * frl10[-2]

    frs10[0] = rfrs10[x] + fxgap[0] * y_frs10_8 + fpitrg[x] * y_frs10_7 + y_frs10_1 * dfmprr[x] + y_frs10_2 * frstar[-1] + fpi10[0] * y_frs10_3 + fpi10[-1] * y_frs10_4 + y_frs10_5 * fpi10[-2] + y_frs10_6 * fpi10[-3]

    frstar[0] = frstar[-1] * y_frstar_1 + frs10[0] * y_frstar_2 + fpi10[0] * y_frstar_3 + fpi10[-1] * y_frstar_4 + y_frstar_5 * fpi10[-2] + y_frstar_6 * fpi10[-3]

    ftcin_l[0] = y_ftcin_l_2 * uftcin[x] + ynicpn_l[0]

    fxgap[0] = xgap2[-1] * y_fxgap_13 + frstar[0] * y_fxgap_12 + fpi10[-1] * y_fxgap_4 + frs10[-1] * y_fxgap_3 + fxgap_aerr[x] + fxgap[-1] * y_fxgap_1 + y_fxgap_2 * fxgap[-2] + y_fxgap_5 * fpi10[-2] + y_fxgap_6 * fpi10[-3] + y_fxgap_7 * fpi10[-4] + y_fxgap_8 * frs10[-2] + y_fxgap_9 * fpi10[-5] + y_fxgap_10 * frs10[-3] + y_fxgap_11 * fpi10[-6]

    fynicn_l[0] = fnicn_l[-1] + y_fynicn_l_2 * rfynic[0]

    fyniln_l[0] = fniln_l[-1] + y_fyniln_l_2 * rfynil[0]

    gfdbtnp_l[0] = ugfdbtp_l[0] + y_gfdbtnp_l_2 * gfdbtnp_l[-1] + y_gfdbtnp_l_3 * gfexpn_l[0] + y_gfdbtnp_l_4 * gfrecn_l[0]
    
    gfdbtn_l[0] = gfdbtnp_l[0] + ugfdbt_l[x]

    ugfdbtp_l[0] = (1 - rho_ugfdbtp_l) * ugfdbtp_l̄ + rho_ugfdbtp_l * ugfdbtp_l[-1] + ugfdbtp_lerr[x]

    ugfsrp[0] = y_ugfsrp_1 * ugfsrp[-1]

    uleg_l[0] = uleg_l[-1] + y_uleg_l_1 * leg_l[-1] + y_uleg_l_2 * lep_l[-1] + y_uleg_l_3 * adjlegrt[x]

    gfexpn_l[0] = egfln_l[0] * y_gfexpn_l_2 + egfen_l[0] * y_gfexpn_l_3 + y_gfexpn_l_4 * gtn_l[0] + y_gfexpn_l_5 * gfintn_l[0]

    gfintn_l[0] = y_gfintn_l_2 * rgfint[0] + gfdbtn_l[-1]

    gfrecn_l[0] = y_gfrecn_l_2 * tpn_l[0] + y_gfrecn_l_3 * tcin_l[0] + ugfsrp[0] * y_gfrecn_l_4 + xgdpn_l[0] * y_gfrecn_l_5
    
    gtn_l[0] = pgdp_l[0] + gtr_l[0]

    gtr_l[0] = y_gtr_l_2 * gtrd[0] + y_gtr_l_3 * gtrt[x] + xgdpt_l[0]

    gtrd[0] = .0014 * (fiscalav[0] - y_gtrd_6 * fiscalav[-1]) + y_gtrd_6 * gtrd[-1] + gtrd_aerr[x] + xgap2[0] * y_gtrd_1 + xgap2[-1] * y_gtrd_2 + y_gtrd_3 * xgap2[-2] + y_gtrd_4 * xgap2[-3] + y_gtrd_5 * xgap2[-4] + y_gtrd_7 * xgap2[-5]

    hgemp[0] = y_hgemp_1 * hgemp[-1] + emp_l[0] * y_hgemp_2 + emp_l[-1] * y_hgemp_3

    hggdp[0] = xgdp_l[0] * y_hggdp_1 + y_hggdp_2 * xgdp_l[-1]

    hggdpt[0] = hxbt[0] + huxb[0]

    hgpbfir[0] = hgpbfir[-1] * y_hgpbfir_1 + pbfir_l[0] * y_hgpbfir_2 + pxp_l[0] * y_hgpbfir_3 + pxb_l[0] * y_hgpbfir_4 + pbfir_l[-1] * y_hgpbfir_5 + pxp_l[-1] * y_hgpbfir_6 + pxb_l[-1] * y_hgpbfir_7

    hgpkir[0] = y_hgpkir_1 * hgpkir[-1] + y_hgpkir_2 * pkir[x] + y_hgpkir_3 * pkir[x-1]

    hgynid[0] = ynicpn_l[0] * y_hgynid_1 + tcin_l[0] * y_hgynid_2 + pxb_l[0] * y_hgynid_3 + y_hgynid_4 * ynicpn_l[-1] + y_hgynid_5 * tcin_l[-1] + pxb_l[-1] * y_hgynid_6

    hks[0] = y_hks_1 * kbfi_l[0] + y_hks_2 * kbfi_l[-1] + y_hks_3 * ki_l[0] + y_hks_4 * ki_l[-1] + hksr[x]

    hlept[0] = y_hlept_1 * hqlfpr[0] + y_hlept_2 * n16_l[x] + y_hlept_3 * n16_l[x-1]

    hlprdt[0] = hxbt[0] - hlept[0] - hqlww[0]

    hmfpt[0] = hmfpt_aerr[x] + y_hmfpt_1 * hmfpt[-1]

    hqlfpr[0] = hqlfpr_aerr[x] + y_hqlfpr_1 * hqlfpr[-1]

    hqlww[0] = hqlww_aerr[x] + y_hqlww_1 * hqlww[-1]

    huqpct[0] = y_huqpct_1 * huqpct[-1]

    huxb[0] = y_huxb_1 * dglprd[x] + y_huxb_2 * huxb[-1]

    hxbt[0] = hmfpt[0] + hks[0] * y_hxbt_5 + hlept[0] * y_hxbt_1 + hqlww[0] * y_hxbt_2 + y_hxbt_3 * lqualt_l[x] + y_hxbt_4 * lqualt_l[x-1]

    # hxbtr[0] = 0

    jccan_l[0] = xgdpn_l[0] + y_jccan_l_2 * jccan_l[-1] + xgdpn_l[-1] * y_jccan_l_3 + y_jccan_l_4 * pkbfir[-1] + kbfi_l[-1] * y_jccan_l_5 + y_jccan_l_6 * jrbfi[x] + pxp_l[-1] * y_jccan_l_7

    jkcd_l[0] = y_jkcd_l_2 * jrcd[x] + kcd_l[-1]

    kbfi_l[0] = pbfir_l[0] * y_kbfi_l_2 + y_kbfi_l_3 * pkbfir[0] + ebfi_l[0] * y_kbfi_l_4 + jrbfi[x] * y_kbfi_l_5 + kbfi_l[-1] * y_kbfi_l_6

    kcd_l[0] = ecd_l[0] * y_kcd_l_2 + jrcd[x] * y_kcd_l_3 + kcd_l[-1] * y_kcd_l_4

    kh_l[0] = eh_l[0] * y_kh_l_2 + y_kh_l_3 * jrh[x] + kh_l[-1] * y_kh_l_4

    ki_l[0] = ki_l[-1] * y_ki_l_1 + ki_l_aerr[x] + y_ki_l_2 * qkir_l[0] + y_ki_l_3 * xfs_l[-1] + y_ki_l_4 * ki_l[-2] + y_ki_l_5 * xfs_l[-2] + y_ki_l_6 * xfs_l[-3]

    ks_l[0] = ks_l[-1] + hks[0] * y_ks_l_1

    leg_l[0] = uleg_l[0] + egfl_l[0] * y_leg_l_1 + egsl_l[0] * y_leg_l_2 - lprdt_l[0]

    leh_l[0] = y_leh_l_2 * lep_l[0] + leg_l[0] * y_leh_l_3 + y_leh_l_4 * leo_l[0]

    leo_l[0] = xgap2[-1] * y_leo_l_5 + y_leo_l_4 * qlf_l[-1] + leo_l_aerr[x] + y_leo_l_1 * qleor[x] + qlf_l[0] + y_leo_l_2 * leo_l[-1] + y_leo_l_3 * qleor[x-1]

    lep_l[0] = lhp_l[0] - lww_l[0]

    leppot_l[0] = qlf_l[0] + y_leppot_l_2 * lurnat[0] + qleor[x] * y_leppot_l_3 + adjlegrt[x] * y_leppot_l_4

    lf_l[0] = n16_l[x] + y_lf_l_2 * lfpr[0]

    lfpr[0] = hqlfpr[0] + y_lfpr_1 * lfpr[-1] + lfpr_aerr[x] + y_lfpr_2 * qlfpr[-1] + y_lfpr_3 * lur[-1] + y_lfpr_4 * lurnat[-1]

    lhp_l[0] = y_lhp_l_7 * hlprdt[-1] + y_lhp_l_6 * xbo_l[-1] + y_lhp_l_5 * xbo_l[0] + y_lhp_l_4 * zlhp[0] + y_lhp_l_1 * lhp_l[-1] + lhp_l_aerr[x] + y_lhp_l_2 * qlhp_l[-1] + y_lhp_l_3 * lhp_l[-2] + y_lhp_l_8 * xbo_l[-2] + y_lhp_l_9 * hlprdt[-2]

    lprdt_l[0] = xbt_l[0] - leppot_l[0] - qlww_l[0]

    lur[0] = leh_l[0] * y_lur_1 + lf_l[0] * y_lur_2

    lurnat[0] = lurnat_aerr[x] + lurnat[-1] * y_lurnat_1

    lww_l[0] = y_lww_l_1 * lww_l[-1] + hqlww[0] * y_lww_l_2 + lww_l_aerr[x] + y_lww_l_3 * qlww_l[-1] + lhp_l[0] * y_lww_l_4 + lhp_l[-1] * y_lww_l_5 + hlept[0] * y_lww_l_6

    # mei_l[0] = mei_l[-1]

    # mep_l[0] = mep_l[-1]

    mfpt_l[0] = mfpt_l_aerr[x] + mfpt_l[-1] + hmfpt[0] * y_mfpt_l_1

    pbfir_l[0] = pxp_l[-1] + dpadj[0] + pbfir_l[-1] + pbfir_l_aerr[x] + pipxnc[0] * y_pbfir_l_1 - pxp_l[0]

    pcdr_l[0] = y_pcdr_l_1 * pcdr_l[-1] + y_pcdr_l_2 * pcdr_l[-2]

    pcer_l[0] = pcer_l[-1] + pcer_l_aerr[x] + pmp_l[0] * y_pcer_l_1 + y_pcer_l_2 * pcxfe_l[0] + pmp_l[-1] * y_pcer_l_3 + y_pcer_l_4 * pcxfe_l[-1]

    pcfr_l[0] = y_pcfr_l_6 * pcfrt[x] + y_pcfr_l_5 * pcfr_l[-4] + y_pcfr_l_4 * pcfr_l[-3] + y_pcfr_l_3 * pcfr_l[-2] + y_pcfr_l_1 * pcfr_l[-1] + pcfr_l_aerr[x] + y_pcfr_l_2 * pcfrt[x-1]

    pchr_l[0] = y_pchr_l_1 * pchr_l[-1] + y_pchr_l_2 * pchr_l[-2]

    pcnia_l[0] = pcnia_l[-1] + y_pcnia_l_1 * picnia[0]

    pcor_l[0] = pcor_l[-1] + pcdr_l[0] * y_pcor_l_1 + pcdr_l[-1] * y_pcor_l_2 + pchr_l[0] * y_pcor_l_3 + pchr_l[-1] * y_pcor_l_4

    pcpi_l[0] = pcnia_l[0] + y_pcpi_l_2 * upcpi[x]

    pcpix_l[0] = pcxfe_l[0] + y_pcpix_l_2 * upcpix[x]

    pcxfe_l[0] = pcxfe_l[-1] + y_pcxfe_l_1 * picxfe[0]

    pegfr_l[0] = pxp_l[-1] + dpadj[0] + pegfr_l[-1] + pegfr_l_aerr[x] + pipxnc[0] * y_pegfr_l_1 - pxp_l[0]

    pegsr_l[0] = pxp_l[-1] + dpadj[0] + pegsr_l[-1] + pegsr_l_aerr[x] + pipxnc[0] * y_pegsr_l_1 - pxp_l[0]

    pgdp_l[0] = xgdpn_l[0] - xgdp_l[0]

    pgfl_l[0] = y_pgfl_l_1 * upgfl[x] + pl_l[0] - lprdt_l[0]

    pgsl_l[0] = pl_l[0] + y_pgsl_l_1 * upgsl[x] - lprdt_l[0]

    phouse_l[0] = pcnia_l[-1] * y_phouse_l_4 + pchr_l[-1] * y_phouse_l_3 + y_phouse_l_1 * phouse_l[-1] + phouse_l_aerr[x] + y_phouse_l_2 * phouse_l[-2]

    phr_l[0] = pxp_l[-1] + dpadj[0] + phr_l[-1] + phr_l_aerr[x] + pipxnc[0] * y_phr_l_1 - pxp_l[0]

    pic4[0] = pcnia_l[0] * y_pic4_1 + y_pic4_2 * pcnia_l[-4]

    picnia[0] = picxfe[0] + pcer_l[0] * y_picnia_1 + pcer_l[-1] * y_picnia_2 + pcfr_l[0] * y_picnia_3 + pcfr_l[-1] * y_picnia_4

    picx4[0] = pcxfe_l[0] * y_picx4_1 + y_picx4_2 * pcxfe_l[-4]

    picxfe[0] = picxfe_aerr[x] + y_picxfe_1 * picxfe[-1] + y_picxfe_2 * zpicxfe[0] + y_picxfe_3 * ptr[-1] + y_picxfe_4 * qpcnia_l[-1] + pcnia_l[-1] * y_picxfe_5

    pieci[0] = y_pieci_12 * pl_l[-1] + y_pieci_11 * qpl_l[-1] + lurnat[-1] * y_pieci_10 + lur[-1] * y_pieci_9 + huqpct[-1] * y_pieci_8 + hlprdt[-1] * y_pieci_7 + ptr[-1] * y_pieci_6 + y_pieci_5 * zpieci[0] + pieci_aerr[x] + y_pieci_1 * pieci[-1] + y_pieci_2 * pieci[-2] + y_pieci_3 * pieci[-3] + y_pieci_4 * pieci[-4]

    pigdp[0] = pgdp_l[0] * y_pigdp_1 + pgdp_l[-1] * y_pigdp_2

    pipl[0] = pieci[0]

    pipxnc[0] = y_pipxnc_11 * pxnc_l[-1] + y_pipxnc_10 * qpxnc_l[-1] + y_pipxnc_9 * fpxr_l[-1] + fpxr_l[0] * y_pipxnc_8 + picnia[0] + huqpct[0] * y_pipxnc_1 + y_pipxnc_2 * pipxnc[-1] + y_pipxnc_3 * picnia[-1] + huqpct[-1] * y_pipxnc_4 + y_pipxnc_5 * pipxnc[-2] + y_pipxnc_6 * picnia[-2] + y_pipxnc_7 * huqpct[-2]

    pkbfir[0] = y_pkbfir_1 * upkbfir[x] + pbfir_l[0] * y_pkbfir_2

    pl_l[0] = pl_l[-1] + pipl[0] * y_pl_l_1

    pmo_l[0] = pmo_l[-1] * y_pmo_l_1 + pmo_l̃[0] + y_pmo_l_2 * qpmo_l + fpc_l[-1] * y_pmo_l_3 + fpx_l[-1] * y_pmo_l_4 + pxb_l[-1] * y_pmo_l_5 + fpc_l[0] * y_pmo_l_6 + fpx_l[0] * y_pmo_l_7 + pxb_l[0] * y_pmo_l_8

    pmo_l̃[0] = (1 - rho_pmo_l) * pmo_l̄ + rho_pmo_l * pmo_l̃[-1] + pmo_l_aerr[x]

    pmp_l[0] = y_pmp_l_2 * upmp[x] + poil_l[0]

    poil_l[0] = pxb_l[0] + poilr_l[0]

    poilr_l[0] = y_poilr_l_4 * poilrt[x] + y_poilr_l_3 * poilr_l[-2] + y_poilr_l_1 * poilr_l[-1] + poilr_l_aerr[x] + y_poilr_l_2 * poilrt[x-1]

    ptr[0] = ptr[-1] * y_ptr_1 + picxfe[-1] * y_ptr_2 + y_ptr_3 * pitarg[x-1]

    pxb_l[0] = pgdp_l[0] + y_pxb_l_2 * upxb[x]

    pxnc_l[0] = pxnc_l[-1] + pipxnc[0] * y_pxnc_l_1

    pxp_l[0] = pxp_l[-1] + pcnia_l[0] * y_pxp_l_1 + pcnia_l[-1] * y_pxp_l_2 + pxnc_l[0] * y_pxp_l_3 + pxnc_l[-1] * y_pxp_l_4

    pxr_l[0] = pxp_l[-1] + dpadj[0] + pxr_l[-1] + pxr_l_aerr[x] + pipxnc[0] * y_pxr_l_1 - pxp_l[0]

    qebfi_l[0] = xb_l[0] + y_qebfi_l_2 * vbfi[0] + hxbt[0] * y_qebfi_l_3 + hgpbfir[0] * y_qebfi_l_4 + jrbfi[x] * y_qebfi_l_5

    qec_l[0] = y_qec_l_1 * zyh_l[0] + y_qec_l_2 * zyht_l[0] + y_qec_l_3 * zyhp_l[0] + y_qec_l_4 * wpo_l[0] + y_qec_l_5 * wps_l[0]

    qecd_l[0] = y_qecd_l_13 * rccd[0] + pcdr_l[0] * y_qecd_l_12 + y_qecd_l_11 * hgpcdr[x] + qec_l[0] + jrcd[x] * y_qecd_l_2 + hggdpt[0] * y_qecd_l_3 + hggdpt[-1] * y_qecd_l_4 + y_qecd_l_5 * hggdpt[-2] + y_qecd_l_6 * hggdpt[-3] + y_qecd_l_7 * hggdpt[-4] + y_qecd_l_8 * hggdpt[-5] + y_qecd_l_9 * hggdpt[-6] + y_qecd_l_10 * hggdpt[-7]

    qeco_l[0] = qec_l[0] - pcor_l[0]

    qeh_l[0] = y_qeh_l_19 * rcch[0] + pcnia_l[0] + qec_l[0] + jrh[x] * y_qeh_l_2 + hggdpt[0] * y_qeh_l_3 + hggdpt[-1] * y_qeh_l_4 + y_qeh_l_5 * hggdpt[-2] + y_qeh_l_6 * hggdpt[-3] + y_qeh_l_7 * hggdpt[-4] + y_qeh_l_8 * hggdpt[-5] + y_qeh_l_9 * hggdpt[-6] + y_qeh_l_10 * hggdpt[-7] + y_qeh_l_11 * hggdpt[-8] + y_qeh_l_12 * hggdpt[-9] + y_qeh_l_13 * hggdpt[-10] + y_qeh_l_14 * hggdpt[-11] + y_qeh_l_15 * hggdpt[-12] + y_qeh_l_16 * hggdpt[-13] + y_qeh_l_17 * hggdpt[-14] + y_qeh_l_18 * hggdpt[-15] - phr_l[0] - pxp_l[0]

    qkir_l[0] = dglprd[x] * y_qkir_l_1 + rho_qkir_l * qkir_l[-1]

    qlf_l[0] = n16_l[x] + y_qlf_l_2 * qlfpr[0]

    qlfpr[0] = hqlfpr[0] + qlfpr[-1]

    qlhp_l[0] = xbo_l[0] - lprdt_l[0]

    qlww_l[0] = qlww_l[-1] + hqlww[-1] * y_qlww_l_1

    qpcnia_l[0] = qpxp_l[0] + uqpct_l[0]

    qpl_l[0] = pxb_l[0] + pl_l[0] - qpxb_l[0]

    # qpmo_l[0] = qpmo_l[-1]

    qpxb_l[0] = pl_l[0] + pwstar_l[x] - lprdt_l[0]

    qpxnc_l[0] = pxnc_l[0] + qpxp_l[0] * y_qpxnc_l_1 + pxp_l[0] * y_qpxnc_l_2 + qpcnia_l[0] * y_qpxnc_l_3 + pcnia_l[0] * y_qpxnc_l_4

    qpxp_l[0] = pxp_l[0] + qpxb_l[0] * y_qpxp_l_1 + pxb_l[0] * y_qpxp_l_2

    qynidn_l[0] = y_qynidn_l_1 * d79a[x] + ynicpn_l[0] * y_qynidn_l_2 + tcin_l[0] * y_qynidn_l_3

    rbbb[0] = rg10[0] + rbbbp[0]

    rbbbp[0] = rbbbp_aerr[x] + y_rbbbp_1 * zgap10[0] + y_rbbbp_2 * rbbbp[-1] + y_rbbbp_3 * zgap10[-1]

    rbfi[0] = y_rbfi_1 * trfcim[x] + y_rbfi_2 * rg5[0] + rbbb[0] * y_rbfi_3 + rg10[0] * y_rbfi_4 + y_rbfi_5 * zpib5[0] + y_rbfi_6 * req[0]

    rcar[0] = rcar_aerr[x] + d79a[x] * y_rcar_1 + y_rcar_2 * t47[x] + y_rcar_3 * rcar[-1] + rg5[0] * y_rcar_4 + y_rcar_5 * rg5[-1]

    rccd[0] = rcar[0] + jrcd[x] * y_rccd_1 - zpi5[0]

    rcch[0] = jrh[x] * y_rcch_1 + y_rcch_2 * trfpm[x] + y_rcch_3 * rme[0] + y_rcch_4 * trspp[x] - zpi10[0]

    rcgain[0] = picx4[0] + rcgain_aerr[x] + xgap2[0] * y_rcgain_1 + y_rcgain_2 * rcgain[-1] + y_rcgain_3 * picx4[-1]

    req[0] = rg30[0] - zpic30[0] + reqp[0]

    reqp[0] = reqp_aerr[x] + rbbbp[0] * y_reqp_1 + y_reqp_2 * reqp[-1] + rbbbp[-1] * y_reqp_3

    rfynic[0] = rfynil[0] * y_rfynic_4 + y_rfynic_1 * rfynic[-1] + rfynic_aerr[x] + y_rfynic_2 * rfynil[-1] + y_rfynic_3 * rfynic[-2]

    rfynil[0] = reqp[0] * y_rfynil_8 + y_rfynil_7 * rtb[0] + rg10[0] * y_rfynil_6 + rfynil[-1] * y_rfynil_1 + rfynil_aerr[x] + y_rfynil_2 * rg10[-1] + y_rfynil_3 * rtb[-1] + reqp[-1] * y_rfynil_4 + y_rfynil_5 * rfynil[-2]

    rg10[0] = zrff10[0] + rg10p[0]

    rg10p[0] = rg10p_aerr[x] + zgap10[0] * y_rg10p_1 + y_rg10p_2 * d8095[x] + y_rg10p_3 * rg10p[-1] + zgap10[-1] * y_rg10p_4 + y_rg10p_5 * d8095[x-1]

    rg30[0] = zrff30[0] + rg30p[0]

    rg30p[0] = rg30p_aerr[x] + y_rg30p_1 * zgap30[0] + d8095[x] * y_rg30p_2 + y_rg30p_3 * rg30p[-1] + y_rg30p_4 * zgap30[-1] + y_rg30p_5 * d8095[x-1]

    rg5[0] = zrff5[0] + rg5p[0]

    rg5p[0] = rg5p_aerr[x] + y_rg5p_1 * zgap05[0] + y_rg5p_2 * rg5p[-1] + y_rg5p_3 * zgap05[-1]

    rgfint[0] = gfdbtn_l[-1] * y_rgfint_4 + rgfint_aerr[x] + y_rgfint_1 * rgfint[-1] + y_rgfint_2 * rgw[-1] + y_rgfint_3 * gfdbtn_l[-2]

    rgw[0] = rtb[0] * y_rgw_1 + rg5[0] * y_rgw_2 + rg10[0] * y_rgw_3 + rg30[0] * y_rgw_4

    rme[0] = rme[-1] * y_rme_1 + rme_aerr[x] + rg10[0] * y_rme_2 + rg10[-1] * y_rme_3 + y_rme_4 * d87[x]

    rrff[0] = rff[0] + picxfe[0] * y_rrff_1 + picxfe[-1] * y_rrff_2 + y_rrff_3 * picxfe[-2] + y_rrff_4 * picxfe[-3]

    rrtr[0] = y_rrtr_1 * rrtr[-1] + rrff[0] * y_rrtr_2

    rspnia[0] = y_rspnia_1 * yhsn_l[0] + y_rspnia_2 * ydn_l[0]

    # rstar[0] = rstar[-1]

    rtb[0] = rff[-1] * y_rtb_4 + rff[0] * y_rtb_3 + rtb[-1] * y_rtb_1 + y_rtb_2 * rtb[-2]

    rtbfi_l[0] = pxp_l[0] + rbfi[0] * y_rtbfi_l_2 + jrbfi[x] * y_rtbfi_l_3 + hgpbfir[0] * y_rtbfi_l_4 + y_rtbfi_l_5 * tritc[x] + trfcim[x] * y_rtbfi_l_6 + y_rtbfi_l_7 * tapddp[x] + y_rtbfi_l_8 * tdpv[x] + pkbfir[0] * y_rtbfi_l_9 - pxb_l[0]

    rtinv[0] = pxb_l[0] * y_rtinv_7 + rbfi[0] * y_rtinv_1 + hgpkir[0] * y_rtinv_2 + pxp_l[0] * y_rtinv_3 + pkir[x] * y_rtinv_4 + pxp_l[-1] * y_rtinv_5 + y_rtinv_6 * pkir[x-1]

    rtr[0] = ptr[0] + rrtr[0]

    tcin_l[0] = ynicpn_l[0] + y_tcin_l_2 * trci[0]

    tpn_l[0] = y_tpn_l_2 * trp[0] + y_tpn_l_3 * ypn_l[0] + gtn_l[0] * y_tpn_l_4

    trci[0] = xgap2[-1] * y_trci_4 + trci_aerr[x] + trcit[x] + xgap2[0] * y_trci_1 + y_trci_2 * trci[-1] + y_trci_3 * trcit[x-1]

    # trp[0] = xgap2[0] * y_trp_5 + trp_aerr[x] + trpt[0] + y_trp_1 * trp[-1] + y_trp_2 * trpt[-1] + y_trp_3 * trp[-2] + y_trp_4 * trpt[-2]
    trp[0] = xgap2[0] * y_trp_5 + trp_a[0] + trpt[0] + y_trp_1 * trp[-1] + y_trp_2 * trpt[-1] + y_trp_3 * trp[-2] + y_trp_4 * trpt[-2]

    trp_a[0] =  (1 - rho_trp_a) * trp_ā + rho_trp_a * trp_a[-1] + trp_aerr[x]

    trpt[0] = trpts[0]

    trptd[0] = y_trptd_6 * gfdrt[x-2] + y_trptd_5 * xgdpn_l[-2] + y_trptd_4 * gfdbtn_l[-2] + trpt[-1] + gfdbtnp_l[-1] * y_trptd_1 + xgdpn_l[-1] * y_trptd_2 + y_trptd_3 * gfdrt[x-1]

    trpts[0] = xgap2[-1] * y_trpts_5 + trpt[-1] + y_trpts_1 * gfrecn_l[-1] + y_trpts_2 * gfexpn_l[-1] + xgdpn_l[-1] * y_trpts_3 + y_trpts_4 * gfsrt[-1]
    
    gfsrt[0] = rho_gfsrt * gfsrt[-1] + gfsrt_err[x]

    tryh[0] = tpn_l[0] * y_tryh_1 + y_tryh_2 * yhln_l[0] + y_tryh_3 * yhptn_l[0]

    uqpct_l[0] = huqpct[0] + uqpct_l[-1]

    uxbt_l[0] = uxbt_l[-1] + huxb[0] * y_uxbt_l_1

    uynicpnr[0] = y_uynicpnr_1 * uynicpnr[-1]

    vbfi[0] = y_vbfi_1 * uvbfi[x] + pkbfir[0] * y_vbfi_2 + pbfir_l[0] * y_vbfi_3 + rtbfi_l[0] * y_vbfi_4

    wpo_l[0] = wpon_l[0] - pcnia_l[0]

    wpon_l[0] = y_wpon_l_2 * wpon_l[-1] + rcgain[0] * y_wpon_l_3 + phouse_l[0] * y_wpon_l_4 + phouse_l[-1] * y_wpon_l_5 + ydn_l[0] * y_wpon_l_6 + ecnian_l[0] * y_wpon_l_7 + y_wpon_l_8 * yhibn_l[0] + pcdr_l[0] * y_wpon_l_9 + pcnia_l[0] * y_wpon_l_10 + ecd_l[0] * y_wpon_l_11 + jkcd_l[0] * y_wpon_l_12

    wps_l[0] = wpsn_l[0] - pcnia_l[0]

    wpsn_l[0] = ynicpn_l[0] * y_wpsn_l_1 + tcin_l[0] * y_wpsn_l_2 + req[0] * y_wpsn_l_3 + y_wpsn_l_4 * zdivgr[0]

    xb_l[0] = y_xb_l_2 * xbn_l[0] + pxb_l[0] * y_xb_l_3

    xbn_l[0] = pxb_l[0] * y_xbn_l_2 + xbo_l[0] * y_xbn_l_3 + xgdpn_l[0] * y_xbn_l_4 + y_xbn_l_5 * xgdo_l[0] + pgdp_l[0] * y_xbn_l_6

    xbo_l[0] = xbt_l[0] + xgap2[0] * y_xbo_l_1

    xbt_l[0] = mfpt_l[0] + leppot_l[0] * y_xbt_l_1 + qlww_l[0] * y_xbt_l_2 + lqualt_l[x] * y_xbt_l_3 + ks_l[0] * y_xbt_l_4 + xbtr_l[0]

    xbtr_l[0] = y_xbtr_l_1 * xbtr_l[-1]

    xfs_l[0] = xfs_l[-1] + ecnia_l[0] * y_xfs_l_1 + ecnia_l[-1] * y_xfs_l_2 + eh_l[0] * y_xfs_l_3 + eh_l[-1] * y_xfs_l_4 + ebfi_l[0] * y_xfs_l_5 + ebfi_l[-1] * y_xfs_l_6 + egfe_l[0] * y_xfs_l_7 + egfe_l[-1] * y_xfs_l_8 + egfl_l[0] * y_xfs_l_9 + egfl_l[-1] * y_xfs_l_10 + egse_l[0] * y_xfs_l_11 + egse_l[-1] * y_xfs_l_12 + egsl_l[0] * y_xfs_l_13 + egsl_l[-1] * y_xfs_l_14 + ex_l[0] * y_xfs_l_15 + ex_l[-1] * y_xfs_l_16 + emo_l[0] * y_xfs_l_17 + emo_l[-1] * y_xfs_l_18 + emp_l[0] * y_xfs_l_19 + emp_l[-1] * y_xfs_l_20

    xfsn_l[0] = xgdpn_l[0] * y_xfsn_l_2 + pkir[x] * y_xfsn_l_3 + pxp_l[0] * y_xfsn_l_4 + ki_l[0] * y_xfsn_l_5 + ki_l[-1] * y_xfsn_l_6

    xgap[0] = xbo_l[0] * y_xgap_1 + xbt_l[0] * y_xgap_2

    xgap2[0] = xgdo_l[0] * y_xgap2_1 + xgdpt_l[0] * y_xgap2_2

    xgdi_l[0] = mei_l + xgdo_l[0]

    xgdin_l[0] = pgdp_l[0] + xgdi_l[0]

    xgdo_l[0] = xgdp_l[0] - mep_l

    xgdp_l[0] = xgdp_l[-1] + xfs_l[0] * y_xgdp_l_1 + xfs_l[-1] * y_xgdp_l_2 + ki_l[0] * y_xgdp_l_3 + ki_l[-1] * y_xgdp_l_4 + y_xgdp_l_5 * ki_l[-2]

    xgdpn_l[0] = y_xgdpn_l_2 * xpn_l[0] + egfln_l[0] * y_xgdpn_l_3 + egsln_l[0] * y_xgdpn_l_4 + emn_l[0] * y_xgdpn_l_5 + pkir[x] * y_xgdpn_l_6 + pxp_l[0] * y_xgdpn_l_7 + ki_l[0] * y_xgdpn_l_8 + ki_l[-1] * y_xgdpn_l_9

    xgdpt_l[0] = xbt_l[0] + uxbt_l[0]

    xgdptn_l[0] = pgdp_l[0] + xgdpt_l[0]

    xp_l[0] = xp_l[-1] + ecnia_l[0] * y_xp_l_1 + ecnia_l[-1] * y_xp_l_2 + eh_l[0] * y_xp_l_3 + eh_l[-1] * y_xp_l_4 + ebfi_l[0] * y_xp_l_5 + ebfi_l[-1] * y_xp_l_6 + egfe_l[0] * y_xp_l_7 + egfe_l[-1] * y_xp_l_8 + egse_l[0] * y_xp_l_9 + egse_l[-1] * y_xp_l_10 + ex_l[0] * y_xp_l_11 + ex_l[-1] * y_xp_l_12

    xpn_l[0] = pxp_l[0] + xp_l[0]

    ydn_l[0] = y_ydn_l_2 * uyd[x] + ypn_l[0] * y_ydn_l_3 + tpn_l[0] * y_ydn_l_4

    yh_l[0] = yhl_l[0] * y_yh_l_2 + yht_l[0] * y_yh_l_3 + y_yh_l_4 * yhp_l[0]

    yhgap[0] = y_yhgap_1 * yhshr_l[0] + y_yhgap_2 * zyhst_l[0]

    yhibn_l[0] = xgdpn_l[0] + y_yhibn_l_2 * uyhibn[x]

    yhl_l[0] = yhln_l[0] + tryh[0] * y_yhl_l_2 - pcnia_l[0]

    yhln_l[0] = y_yhln_l_2 * uyhln[x] + yniln_l[0]

    yhp_l[0] = tryh[0] * y_yhp_l_2 + yhptn_l[0] * y_yhp_l_3 + y_yhp_l_4 * yhpntn_l[0] - pcnia_l[0]

    yhpcd_l[0] = kcd_l[-1]

    yhpgap[0] = y_yhpgap_1 * yhpshr_l[0] + y_yhpgap_2 * zyhpst_l[0]

    yhpntn_l[0] = pcnia_l[0] * y_yhpntn_l_2 + pcdr_l[0] * y_yhpntn_l_3 + yhpcd_l[0] * y_yhpntn_l_4 + yhibn_l[0] * y_yhpntn_l_5 + ynicpn_l[0] * y_yhpntn_l_6 + tcin_l[0] * y_yhpntn_l_7 + y_yhpntn_l_8 * ynidn_l[0] + zpi10[0] * y_yhpntn_l_9 + gfdbtn_l[0] * y_yhpntn_l_10

    yhpshr_l[0] = yhp_l[0] - yh_l[0]

    yhptn_l[0] = y_yhptn_l_2 * uyhptn[x] + y_yhptn_l_3 * ynirn_l[0] + gfintn_l[0] * y_yhptn_l_4 + ynidn_l[0] * y_yhptn_l_5 + yhibn_l[0] * y_yhptn_l_6

    yhshr_l[0] = yh_l[0] * y_yhshr_l_2 + xgdp_l[0] * y_yhshr_l_3

    yhsn_l[0] = yhln_l[0] * y_yhsn_l_2 + y_yhsn_l_3 * yhtn_l[0] + yhptn_l[0] * y_yhsn_l_4 + tpn_l[0] * y_yhsn_l_5 + ecnian_l[0] * y_yhsn_l_6 + yhibn_l[0] * y_yhsn_l_7 + y_yhsn_l_8 * uyhsn[x] + xgdptn_l[0] * y_yhsn_l_9

    yht_l[0] = yhtn_l[0] - pcnia_l[0]

    yhtgap[0] = y_yhtgap_1 * yhtshr_l[0] + y_yhtgap_2 * zyhtst_l[0]

    yhtn_l[0] = gtn_l[0] + y_yhtn_l_2 * uyhtn[x]

    yhtshr_l[0] = yht_l[0] - yh_l[0]

    ykbfin_l[0] = pxb_l[0] + rtbfi_l[0] + kbfi_l[0] * y_ykbfin_l_2 + kbfi_l[-1] * y_ykbfin_l_3

    ykin_l[0] = pxb_l[0] + rtinv[0] * y_ykin_l_2 + ki_l[0] * y_ykin_l_3 + ki_l[-1] * y_ykin_l_4

    ynicpn_l[0] = y_ynicpn_l_2 * ynin_l[0] + yniln_l[0] * y_ynicpn_l_3 + ynirn_l[0] * y_ynicpn_l_4 + uynicpnr[0] * y_ynicpn_l_5 + xgdpn_l[0] * y_ynicpn_l_6

    ynidn_l[0] = zynid[0] + y_ynidn_l_8 * pxb_l[-2] + y_ynidn_l_7 * ymsdn[x-2] + y_ynidn_l_6 * ynidn_l[-2] + y_ynidn_l_5 * qynidn_l[-1] + ynidn_l_aerr[x] + pxb_l[-1] * y_ynidn_l_4 + pxb_l[0] + y_ynidn_l_1 * ymsdn[x] + y_ynidn_l_2 * ynidn_l[-1] + y_ynidn_l_3 * ymsdn[x-1]

    yniln_l[0] = y_yniln_l_2 * uyl[x] + pl_l[0] * y_yniln_l_3 + lhp_l[0] * y_yniln_l_4 + pgfl_l[0] * y_yniln_l_5 + egfl_l[0] * y_yniln_l_6 + pgsl_l[0] * y_yniln_l_7 + egsl_l[0] * y_yniln_l_8

    ynin_l[0] = y_ynin_l_2 * uyni[x] + xgdin_l[0] * y_ynin_l_3 + fynicn_l[0] * y_ynin_l_4 + fyniln_l[0] * y_ynin_l_5 + jccan_l[0] * y_ynin_l_6

    ynirn_l[0] = xgdpn_l[0] + y_ynirn_l_1 * ynirn_l_aerr[x] + y_ynirn_l_2 * ynirn_l[-1] + xgdpn_l[-1] * y_ynirn_l_3 + rbbb[0] * y_ynirn_l_4 + y_ynirn_l_5 * rbbb[-1]

    ypn_l[0] = y_ypn_l_2 * uyp[x] + yhln_l[0] * y_ypn_l_3 + yhtn_l[0] * y_ypn_l_4 + yhptn_l[0] * y_ypn_l_5

    zdivgr[0] = y_zdivgr_1 * hgynid[1] + y_zdivgr_2 * zdivgr[1]

    zebfi[0] = hgpbfir[-1] * y_zebfi_21 + y_zebfi_20 * hxbt[-1] + qebfi_l[-1] * y_zebfi_15 + y_zebfi_11 * xgap[-1] + ptr[-1] * y_zebfi_10 + y_zebfi_9 * rtr[-1] + rff[-1] * y_zebfi_5 + picnia[-1] * y_zebfi_1 + y_zebfi_2 * picnia[-2] + y_zebfi_3 * picnia[-3] + y_zebfi_4 * picnia[-4] + y_zebfi_6 * rff[-2] + y_zebfi_7 * rff[-3] + y_zebfi_8 * rff[-4] + y_zebfi_12 * xgap[-2] + y_zebfi_13 * xgap[-3] + y_zebfi_14 * xgap[-4] + y_zebfi_16 * qebfi_l[-2] + y_zebfi_17 * qebfi_l[-3] + y_zebfi_18 * qebfi_l[-4] + y_zebfi_19 * qebfi_l[-5]

    zecd[0] = y_zecd_33 * qecd_l[-5] + y_zecd_32 * qecd_l[-4] + y_zecd_31 * qecd_l[-3] + y_zecd_30 * qecd_l[-2] + qecd_l[-1] * y_zecd_29 + hggdpt[-1] * y_zecd_27 + y_zecd_23 * yhpgap[-1] + y_zecd_19 * yhtgap[-1] + y_zecd_15 * yhgap[-1] + rtr[-1] * y_zecd_14 + ptr[-1] * y_zecd_13 + xgap2[-1] * y_zecd_9 + rff[-1] * y_zecd_5 + picnia[-1] * y_zecd_1 + y_zecd_2 * picnia[-2] + y_zecd_3 * picnia[-3] + y_zecd_4 * picnia[-4] + y_zecd_6 * rff[-2] + y_zecd_7 * rff[-3] + y_zecd_8 * rff[-4] + y_zecd_10 * xgap2[-2] + y_zecd_11 * xgap2[-3] + y_zecd_12 * xgap2[-4] + y_zecd_16 * yhgap[-2] + y_zecd_17 * yhgap[-3] + y_zecd_18 * yhgap[-4] + y_zecd_20 * yhtgap[-2] + y_zecd_21 * yhtgap[-3] + y_zecd_22 * yhtgap[-4] + y_zecd_24 * yhpgap[-2] + y_zecd_25 * yhpgap[-3] + y_zecd_26 * yhpgap[-4] + y_zecd_28 * hgpcdr[x-1]

    zeco[0] = qeco_l[-1] * y_zeco_28 + hggdpt[-1] * y_zeco_27 + yhpgap[-1] * y_zeco_23 + yhtgap[-1] * y_zeco_19 + yhgap[-1] * y_zeco_15 + rtr[-1] * y_zeco_14 + ptr[-1] * y_zeco_13 + xgap2[-1] * y_zeco_9 + rff[-1] * y_zeco_5 + picnia[-1] * y_zeco_1 + y_zeco_2 * picnia[-2] + y_zeco_3 * picnia[-3] + y_zeco_4 * picnia[-4] + y_zeco_6 * rff[-2] + y_zeco_7 * rff[-3] + y_zeco_8 * rff[-4] + y_zeco_10 * xgap2[-2] + y_zeco_11 * xgap2[-3] + y_zeco_12 * xgap2[-4] + y_zeco_16 * yhgap[-2] + y_zeco_17 * yhgap[-3] + y_zeco_18 * yhgap[-4] + y_zeco_20 * yhtgap[-2] + y_zeco_21 * yhtgap[-3] + y_zeco_22 * yhtgap[-4] + y_zeco_24 * yhpgap[-2] + y_zeco_25 * yhpgap[-3] + y_zeco_26 * yhpgap[-4] + y_zeco_29 * qeco_l[-2] + y_zeco_30 * qeco_l[-3] + y_zeco_31 * qeco_l[-4] + y_zeco_32 * qeco_l[-5]

    zeh[0] = qeh_l[-1] * y_zeh_28 + hggdpt[-1] * y_zeh_27 + yhpgap[-1] * y_zeh_23 + yhtgap[-1] * y_zeh_19 + yhgap[-1] * y_zeh_15 + rtr[-1] * y_zeh_14 + ptr[-1] * y_zeh_13 + xgap2[-1] * y_zeh_9 + rff[-1] * y_zeh_5 + picnia[-1] * y_zeh_1 + y_zeh_2 * picnia[-2] + y_zeh_3 * picnia[-3] + y_zeh_4 * picnia[-4] + y_zeh_6 * rff[-2] + y_zeh_7 * rff[-3] + y_zeh_8 * rff[-4] + y_zeh_10 * xgap2[-2] + y_zeh_11 * xgap2[-3] + y_zeh_12 * xgap2[-4] + y_zeh_16 * yhgap[-2] + y_zeh_17 * yhgap[-3] + y_zeh_18 * yhgap[-4] + y_zeh_20 * yhtgap[-2] + y_zeh_21 * yhtgap[-3] + y_zeh_22 * yhtgap[-4] + y_zeh_24 * yhpgap[-2] + y_zeh_25 * yhpgap[-3] + y_zeh_26 * yhpgap[-4] + y_zeh_29 * qeh_l[-2] + y_zeh_30 * qeh_l[-3] + y_zeh_31 * qeh_l[-4] + y_zeh_32 * qeh_l[-5]

    zgap05[0] = xgap[0] * y_zgap05_1 + y_zgap05_2 * zgap05[1]

    zgap10[0] = xgap[0] * y_zgap10_1 + y_zgap10_2 * zgap10[1]

    zgap30[0] = xgap[0] * y_zgap30_1 + y_zgap30_2 * zgap30[1]

    zgapc2[0] = rtr[-1] * y_zgapc2_14 + ptr[-1] * y_zgapc2_13 + xgap2[-1] * y_zgapc2_9 + rff[-1] * y_zgapc2_5 + picnia[-1] * y_zgapc2_1 + y_zgapc2_2 * picnia[-2] + y_zgapc2_3 * picnia[-3] + y_zgapc2_4 * picnia[-4] + y_zgapc2_6 * rff[-2] + y_zgapc2_7 * rff[-3] + y_zgapc2_8 * rff[-4] + y_zgapc2_10 * xgap2[-2] + y_zgapc2_11 * xgap2[-3] + y_zgapc2_12 * xgap2[-4]

    zlhp[0] = hqlww[-1] * y_zlhp_20 + y_zlhp_19 * hlept[-1] + y_zlhp_17 * lprdt_l[-1] + xbo_l[-1] * y_zlhp_15 + xgap[-1] * y_zlhp_11 + ptr[-1] * y_zlhp_10 + rtr[-1] * y_zlhp_9 + rff[-1] * y_zlhp_5 + picnia[-1] * y_zlhp_1 + y_zlhp_2 * picnia[-2] + y_zlhp_3 * picnia[-3] + y_zlhp_4 * picnia[-4] + y_zlhp_6 * rff[-2] + y_zlhp_7 * rff[-3] + y_zlhp_8 * rff[-4] + y_zlhp_12 * xgap[-2] + y_zlhp_13 * xgap[-3] + y_zlhp_14 * xgap[-4] + y_zlhp_16 * xbo_l[-2] + y_zlhp_18 * lprdt_l[-2]

    zpi10[0] = picnia[0] * y_zpi10_1 + y_zpi10_2 * zpi10[1]

    zpi10f[0] = picnia[0] * y_zpi10f_1 + y_zpi10f_2 * zpi10f[1]

    zpi5[0] = xgap[-1] * y_zpi5_11 + ptr[-1] * y_zpi5_10 + rtr[-1] * y_zpi5_9 + rff[-1] * y_zpi5_5 + picnia[-1] * y_zpi5_1 + y_zpi5_2 * picnia[-2] + y_zpi5_3 * picnia[-3] + y_zpi5_4 * picnia[-4] + y_zpi5_6 * rff[-2] + y_zpi5_7 * rff[-3] + y_zpi5_8 * rff[-4] + y_zpi5_12 * xgap[-2] + y_zpi5_13 * xgap[-3] + y_zpi5_14 * xgap[-4]

    zpib5[0] = pxb_l[0] * y_zpib5_1 + pxb_l[-1] * y_zpib5_2 + y_zpib5_3 * zpib5[1]

    zpic30[0] = picnia[0] * y_zpic30_1 + y_zpic30_2 * zpic30[1]

    zpic58[0] = pic4[8]

    zpicxfe[0] = lurnat[-1] * y_zpicxfe_26 + lur[-1] * y_zpicxfe_25 + huqpct[-1] * y_zpicxfe_24 + hlprdt[-1] * y_zpicxfe_23 + pl_l[-1] * y_zpicxfe_22 + qpl_l[-1] * y_zpicxfe_21 + pcnia_l[-1] * y_zpicxfe_20 + qpcnia_l[-1] * y_zpicxfe_19 + ptr[-1] * y_zpicxfe_18 + rtr[-1] * y_zpicxfe_17 + xgap2[-1] * y_zpicxfe_13 + rff[-1] * y_zpicxfe_9 + pieci[-1] * y_zpicxfe_5 + picxfe[-1] * y_zpicxfe_1 + y_zpicxfe_2 * picxfe[-2] + y_zpicxfe_3 * picxfe[-3] + y_zpicxfe_4 * picxfe[-4] + y_zpicxfe_6 * pieci[-2] + y_zpicxfe_7 * pieci[-3] + y_zpicxfe_8 * pieci[-4] + y_zpicxfe_10 * rff[-2] + y_zpicxfe_11 * rff[-3] + y_zpicxfe_12 * rff[-4] + y_zpicxfe_14 * xgap2[-2] + y_zpicxfe_15 * xgap2[-3] + y_zpicxfe_16 * xgap2[-4] + y_zpicxfe_27 * lur[-2] + y_zpicxfe_28 * lurnat[-2]

    zpieci[0] = lurnat[-1] * y_zpieci_26 + lur[-1] * y_zpieci_25 + huqpct[-1] * y_zpieci_24 + hlprdt[-1] * y_zpieci_23 + pl_l[-1] * y_zpieci_22 + qpl_l[-1] * y_zpieci_21 + pcnia_l[-1] * y_zpieci_20 + qpcnia_l[-1] * y_zpieci_19 + ptr[-1] * y_zpieci_18 + rtr[-1] * y_zpieci_17 + xgap2[-1] * y_zpieci_13 + rff[-1] * y_zpieci_9 + pieci[-1] * y_zpieci_5 + picxfe[-1] * y_zpieci_1 + y_zpieci_2 * picxfe[-2] + y_zpieci_3 * picxfe[-3] + y_zpieci_4 * picxfe[-4] + y_zpieci_6 * pieci[-2] + y_zpieci_7 * pieci[-3] + y_zpieci_8 * pieci[-4] + y_zpieci_10 * rff[-2] + y_zpieci_11 * rff[-3] + y_zpieci_12 * rff[-4] + y_zpieci_14 * xgap2[-2] + y_zpieci_15 * xgap2[-3] + y_zpieci_16 * xgap2[-4] + y_zpieci_27 * lur[-2] + y_zpieci_28 * lurnat[-2]

    zrff10[0] = rff[0] * y_zrff10_1 + y_zrff10_2 * zrff10[1]

    zrff30[0] = rff[0] * y_zrff30_1 + y_zrff30_2 * zrff30[1]

    zrff5[0] = rff[0] * y_zrff5_1 + y_zrff5_2 * zrff5[1]

    zyh_l[0] = xgdpt_l[0] + zyhst_l[0] + yhgap[-1] * y_zyh_l_16 + yhgap[0] * y_zyh_l_15 + rtr[0] * y_zyh_l_14 + ptr[0] * y_zyh_l_13 + xgap2[-1] * y_zyh_l_10 + xgap2[0] * y_zyh_l_9 + rff[-1] * y_zyh_l_6 + rff[0] * y_zyh_l_5 + picnia[0] * y_zyh_l_1 + picnia[-1] * y_zyh_l_2 + y_zyh_l_3 * picnia[-2] + y_zyh_l_4 * picnia[-3] + y_zyh_l_7 * rff[-2] + y_zyh_l_8 * rff[-3] + y_zyh_l_11 * xgap2[-2] + y_zyh_l_12 * xgap2[-3] + y_zyh_l_17 * yhgap[-2] + y_zyh_l_18 * yhgap[-3]

    zyhp_l[0] = xgdpt_l[0] + zyhst_l[0] + zyhpst_l[0] + yhpgap[-1] * y_zyhp_l_20 + yhpgap[0] * y_zyhp_l_19 + yhgap[-1] * y_zyhp_l_16 + yhgap[0] * y_zyhp_l_15 + rtr[0] * y_zyhp_l_14 + ptr[0] * y_zyhp_l_13 + xgap2[-1] * y_zyhp_l_10 + xgap2[0] * y_zyhp_l_9 + rff[-1] * y_zyhp_l_6 + rff[0] * y_zyhp_l_5 + picnia[0] * y_zyhp_l_1 + picnia[-1] * y_zyhp_l_2 + y_zyhp_l_3 * picnia[-2] + y_zyhp_l_4 * picnia[-3] + y_zyhp_l_7 * rff[-2] + y_zyhp_l_8 * rff[-3] + y_zyhp_l_11 * xgap2[-2] + y_zyhp_l_12 * xgap2[-3] + y_zyhp_l_17 * yhgap[-2] + y_zyhp_l_18 * yhgap[-3] + y_zyhp_l_21 * yhpgap[-2] + y_zyhp_l_22 * yhpgap[-3]

    zyhpst_l[0] = zyhpst_l[-1] + yhpgap[-1] * y_zyhpst_l_1

    zyhst_l[0] = zyhst_l[-1] + yhgap[-1] * y_zyhst_l_1

    zyht_l[0] = xgdpt_l[0] + zyhst_l[0] + zyhtst_l[0] + yhtgap[-1] * y_zyht_l_20 + yhtgap[0] * y_zyht_l_19 + yhgap[-1] * y_zyht_l_16 + yhgap[0] * y_zyht_l_15 + rtr[0] * y_zyht_l_14 + ptr[0] * y_zyht_l_13 + xgap2[-1] * y_zyht_l_10 + xgap2[0] * y_zyht_l_9 + rff[-1] * y_zyht_l_6 + rff[0] * y_zyht_l_5 + picnia[0] * y_zyht_l_1 + picnia[-1] * y_zyht_l_2 + y_zyht_l_3 * picnia[-2] + y_zyht_l_4 * picnia[-3] + y_zyht_l_7 * rff[-2] + y_zyht_l_8 * rff[-3] + y_zyht_l_11 * xgap2[-2] + y_zyht_l_12 * xgap2[-3] + y_zyht_l_17 * yhgap[-2] + y_zyht_l_18 * yhgap[-3] + y_zyht_l_21 * yhtgap[-2] + y_zyht_l_22 * yhtgap[-3]

    zyhtst_l[0] = zyhtst_l[-1] + yhtgap[-1] * y_zyhtst_l_1

    zynid[0] = hggdpt[-1] * y_zynid_25 + pxb_l[-1] * y_zynid_16 + qynidn_l[-1] * y_zynid_15 + xgap[-1] * y_zynid_11 + ptr[-1] * y_zynid_10 + rtr[-1] * y_zynid_9 + rff[-1] * y_zynid_5 + picnia[-1] * y_zynid_1 + y_zynid_2 * picnia[-2] + y_zynid_3 * picnia[-3] + y_zynid_4 * picnia[-4] + y_zynid_6 * rff[-2] + y_zynid_7 * rff[-3] + y_zynid_8 * rff[-4] + y_zynid_12 * xgap[-2] + y_zynid_13 * xgap[-3] + y_zynid_14 * xgap[-4] + y_zynid_17 * qynidn_l[-2] + y_zynid_18 * pxb_l[-2] + y_zynid_19 * qynidn_l[-3] + y_zynid_20 * pxb_l[-3] + y_zynid_21 * qynidn_l[-4] + y_zynid_22 * pxb_l[-4] + y_zynid_23 * qynidn_l[-5] + y_zynid_24 * pxb_l[-5]

    ugap[0] = lur[0] - lurnat[0]

    rff[0] = rule[0] + eradd[x]

    rule[0] = rff[-1] * .85 + rstar * .15 + picx4[0] * .225 - 0.075 * pitarg[x] + xgap2[0] * .15

    fiscal[0] = (1 - rho_fiscal) * f̄iscal + rho_fiscal * fiscal[-1] + fiscal_aerr[x]

    fiscalav[0] = av * fiscal[0] + fiscalav[-1] * rho_fiscalav


    gov_exp_share[0] = egfe_l[0] * y_xfs_l_7 * 100

    income_tax_share_of_gdp[0] = 100 * (y_yh_l_2 * (-y_yhl_l_2 - 1) + y_yh_l_4 * (-y_yhp_l_2 - 1)) * tryh[0]

    debt_to_gdp[0] = - gfdbtnp_l[0] * y_gfdbtnp_l_4 * y_gfrecn_l_4 * y_gfrecn_l_5 * 100
end


@parameters FRBUS begin
    mep_l = 0
    
    mei_l = 0

    qpmo_l = 0

    rstar = 0

    rho_qkir_l = 0.8

    y_dpgap_1 = 0.0025

    y_dpgap_2 = (-0.103649883938)

    y_dpgap_3 = 0.103649883938

    y_dpgap_4 = (-0.341041547027)

    y_dpgap_5 = 0.341041547027

    y_dpgap_6 = (-0.121366054939)

    y_dpgap_7 = 0.121366054939

    y_dpgap_8 = (-0.104958882473)

    y_dpgap_9 = 0.104958882473

    y_dpgap_10 = (-0.328983631622)

    y_dpgap_11 = 0.328983631622

    y_ebfi_l_1 = 1.27660626172

    y_ebfi_l_2 = 0.0453619253429

    y_ebfi_l_3 = (-0.135655771316)

    y_ebfi_l_4 = (-0.18631241575)

    y_ebfi_l_5 = 0.616485384319

    y_ebfi_l_6 = 0.383514615681

    y_ebfi_l_7 = (-0.383514615681)

    y_ebfi_l_8 = (-0.000958786539202)

    y_ebfin_l_1 = 0.000349694902126

    y_ec_l_1 = 0.7310605131

    y_ec_l_2 = (-0.7310605131)

    y_ec_l_3 = 0.157421136

    y_ec_l_4 = (-0.157421136)

    y_ec_l_5 = 0.0223688796433

    y_ec_l_6 = 0.0891494712567

    y_ec_l_7 = (-0.0223688796433)

    y_ec_l_8 = (-0.0891494712567)

    y_ecd_l_1 = 0.78385727975

    y_ecd_l_2 = 0.156149940356

    y_ecd_l_3 = 0.0599927798938

    y_ecd_l_4 = 0.0296796460069

    y_ech_l_1 = 1.71348425234

    y_ech_l_2 = (-1.71348425234)

    y_ech_l_3 = 9.76051187168

    y_ech_l_4 = (-0.718706571642)

    y_ech_l_5 = 0.718706571642

    y_ecnia_l_1 = 0.735

    y_ecnia_l_2 = (-0.735)

    y_ecnia_l_3 = 0.1055

    y_ecnia_l_4 = (-0.1055)

    y_ecnia_l_5 = 0.1595

    y_ecnia_l_6 = (-0.1595)

    y_ecnian_l_1 = 7.05661360558e-05

    y_eco_l_1 = 1.17546755467

    y_eco_l_2 = 0.109703169694

    y_eco_l_3 = (-0.285170724366)

    y_eco_l_4 = 0.692476259501

    y_eco_l_5 = 0.229572174835

    y_eco_l_6 = 0.0779515656641

    y_eco_l_7 = (-0.229612885136)

    y_eco_l_8 = (-0.0779108553636)

    y_egfe_l_1 = 0.726276173623

    y_egfe_l_2 = (-1.38339974044)

    y_egfe_l_3 = 0.0497143719338

    y_egfe_l_4 = 0.103593759929

    y_egfe_l_5 = 1.50381543495

    y_egfe_l_6 = (-0.000983552448045)

    y_egfe_l_7 = 0.000725681212301

    y_egfen_l_1 = 0.0010878350668

    y_egfet_l_1 = 0.9

    y_egfet_l_2 = (-0.1)

    y_egfet_l_3 = (-0.1)

    y_egfet_l_4 = 0.1

    y_egfet_l_5 = 0.000625

    y_egfet_l_6 = 0.000625

    y_egfet_l_7 = 0.000625

    y_egfet_l_8 = 0.000625

    y_egfl_l_1 = 1.16197632264

    y_egfl_l_2 = (-1.12731388567)

    y_egfl_l_3 = (-0.302868541805)

    y_egfl_l_4 = 0.0613337937414

    y_egfl_l_5 = 1.2068723111

    y_egfl_l_6 = (-0.00250725401078)

    y_egfl_l_7 = 0.00235067489642

    y_egfln_l_1 = 0.00218479904218

    y_egflt_l_1 = 0.9

    y_egflt_l_2 = (-0.1)

    y_egflt_l_3 = 0.1

    y_egflt_l_4 = 0.000625

    y_egflt_l_5 = 0.000625

    y_egflt_l_6 = 0.000625

    y_egflt_l_7 = 0.000625

    y_egse_l_1 = 1.00049378528

    y_egse_l_2 = (-0.797614647892)

    y_egse_l_3 = (-0.128950321813)

    y_egse_l_4 = (-0.00262964990773)

    y_egse_l_5 = 0.928700834331

    y_egse_l_6 = 0.00158066587876

    y_egse_l_7 = (-0.000853766092194)

    y_egsen_l_1 = 0.00117247778411

    y_egset_l_1 = 0.9

    y_egset_l_2 = (-0.1)

    y_egset_l_3 = (-0.1)

    y_egset_l_4 = 0.1

    y_egset_l_5 = 0.000625

    y_egset_l_6 = 0.000625

    y_egset_l_7 = 0.000625

    y_egset_l_8 = 0.000625

    y_egsl_l_1 = 1.04483163655

    y_egsl_l_2 = (-0.633546297018)

    y_egsl_l_3 = (-0.134688612832)

    y_egsl_l_4 = (-0.0215581541096)

    y_egsl_l_5 = 0.744961427412

    y_egsl_l_6 = (-0.00143256549309)

    y_egsl_l_7 = 0.00176517379444

    y_egsln_l_1 = 0.000707659055882

    y_egslt_l_1 = 0.9

    y_egslt_l_2 = (-0.1)

    y_egslt_l_3 = 0.1

    y_egslt_l_4 = 0.000625

    y_egslt_l_5 = 0.000625

    y_egslt_l_6 = 0.000625

    y_egslt_l_7 = 0.000625

    y_eh_l_1 = 1.3576278254

    y_eh_l_2 = 0.0130993143616

    y_eh_l_3 = (-0.164666195693)

    y_eh_l_4 = (-0.206060944067)

    y_eh_l_5 = (-0.0282729007489)

    y_eh_l_6 = 0.0282729007489

    y_eh_l_7 = (-0.000786966438108)

    y_ehn_l_1 = 0.00124036373046

    y_em_l_1 = 0.0012598389126

    y_em_l_2 = (-0.000856800245907)

    y_em_l_3 = 0.00125870045787

    y_em_l_4 = (-0.000781093277566)

    y_em_l_5 = 0.928320853989

    y_em_l_6 = (-0.928320853989)

    y_em_l_7 = (-0.000403038666697)

    y_em_l_8 = (-0.000477607180303)

    y_em_l_9 = 0.0716791460112

    y_em_l_10 = (-0.0716791460112)

    y_emn_l_1 = 0.000320220965275

    y_emn_l_2 = 0.928554219554

    y_emn_l_3 = 0.0714457804463

    y_emo_l_1 = 0.819289500318

    y_emo_l_2 = (-0.180710499682)

    y_emo_l_3 = 1.31018224516

    y_emo_l_4 = 0.180710499682

    y_emo_l_5 = 0.0135818692772

    y_emo_l_6 = 0.00278890259237

    y_emo_l_7 = (-0.0163707718696)

    y_emo_l_8 = 0.723524924437

    y_emo_l_9 = (-0.404694213855)

    y_emon_l_1 = 0.000344859738432

    y_emp_l_1 = 40.1856146542

    y_emp_l_2 = 0.048026

    y_emp_l_3 = (-0.048026)

    y_emp_l_4 = (-0.048026)

    y_emp_l_5 = 0.048026

    y_emp_l_6 = 0.022115

    y_empn_l_1 = 0.00448201367911

    y_ex_l_1 = 0.892272127137

    y_ex_l_2 = (-0.107727872863)

    y_ex_l_3 = (-0.107727872863)

    y_ex_l_4 = (-0.107727872863)

    y_ex_l_5 = 0.107727872863

    y_ex_l_6 = 0.107727872863

    y_ex_l_7 = 0.0148164224533

    y_ex_l_8 = (-0.0045419370785)

    y_ex_l_9 = (-0.0102744853748)

    y_ex_l_10 = 1.01585705046

    y_exn_l_1 = 0.000395785791626

    y_fcbn_l_1 = (-0.00219688240418)

    y_fcbn_l_2 = (-5.55068537239)

    y_fcbn_l_3 = 6.86052021077

    y_fcbn_l_4 = (-2.52909715822)

    y_fcbn_l_5 = 1.9133340876

    y_fcbn_l_6 = (-35.2463013113)

    y_fcbn_l_7 = 0.305928232246

    y_fcbn_l_8 = 0.305928232246

    y_fgdp_l_1 = 0.00843835585766

    y_fgdp_l_2 = 0.01

    y_fgdpt_l_1 = 0.9

    y_fgdpt_l_2 = 0.1

    y_fgdpt_l_3 = 0.000625

    y_fgdpt_l_4 = 0.000625

    y_fgdpt_l_5 = 0.000625

    y_fgdpt_l_6 = 0.000625

    y_fnicn_l_1 = 0.993277528339

    y_fnicn_l_2 = 0.00672247166135

    y_fnicn_l_3 = 0.892965336399

    y_fnicn_l_4 = 0.537028034851

    y_fnicn_l_5 = (-0.537028034851)

    y_fnicn_l_6 = (-0.66631256176)

    y_fnicn_l_7 = 0.66631256176

    y_fnicn_l_8 = 0.892965336399

    y_fniln_l_1 = 0.982046754178

    y_fniln_l_2 = 0.985640979746

    y_fniln_l_3 = 0.692942512139

    y_fniln_l_4 = 0.0100008124223

    y_fniln_l_5 = 0.00373870870246

    y_fniln_l_6 = 0.315405113519

    y_fniln_l_7 = (-0.315405113519)

    y_fniln_l_8 = (-0.0591384587847)

    y_fniln_l_9 = 0.0591384587847

    y_fniln_l_10 = 0.00421372469752

    y_fnirn_l_1 = (-0.00807177556398)

    y_fnirn_l_2 = (-169.102652771)

    y_fpc_l_1 = 0.00879025119382

    y_fpc_l_2 = 0.0025

    y_fpi10_1 = 0.156993726433

    y_fpi10_2 = 0.156993726433

    y_fpi10_3 = 0.156993726433

    y_fpi10_4 = 0.156993726433

    y_fpi10_5 = 0.372025094268

    y_fpi10_6 = 0.32214582784

    y_fpi10t_1 = 0.95

    y_fpi10t_2 = 0.05

    y_fpic_1 = 0.678829880162

    y_fpic_2 = 0.321170119838

    y_fpx_l_1 = 0.00804862709227

    y_fpxr_l_1 = 0.048

    y_fpxr_l_2 = (-0.048)

    y_fpxr_l_3 = (-0.048)

    y_fpxr_l_4 = 0.048

    y_fpxr_l_5 = 0.563832456119

    y_fpxr_l_6 = (-0.726654492224)

    y_fpxr_l_7 = 0.162822036105

    y_fpxrr_l_1 = 1.18364909386

    y_fpxrr_l_2 = (-0.00291888934318)

    y_fpxrr_l_3 = (-0.211089676177)

    y_fpxrr_l_4 = 0.00302407543125

    y_frl10_1 = 0.988458285734

    y_frl10_2 = (-0.29200997295)

    y_frl10_3 = (-0.0655047670227)

    y_frl10_4 = 0.369056454239

    y_frl10_5 = 0.12455118125

    y_frl10_6 = (-0.12455118125)

    y_frs10_1 = 4.78434763861

    y_frs10_2 = 0

    y_frs10_3 = 0.25

    y_frs10_4 = 0.25

    y_frs10_5 = 0.25

    y_frs10_6 = 0.25

    y_frs10_7 = 0

    y_frs10_8 = 0

    y_frstar_1 = 0.95

    y_frstar_2 = 0.05

    y_frstar_3 = (-0.0125)

    y_frstar_4 = (-0.0125)

    y_frstar_5 = (-0.0125)

    y_frstar_6 = (-0.0125)

    y_ftcin_l_1 = 0.0814929508598

    y_ftcin_l_2 = 190.397828213

    y_fxgap_1 = 1.29072367633

    y_fxgap_2 = (-0.468009114875)

    y_fxgap_3 = (-0.0166666666667)

    y_fxgap_4 = 0.00416666666667

    y_fxgap_5 = 0.00833333333333

    y_fxgap_6 = 0.0125

    y_fxgap_7 = 0.0125

    y_fxgap_8 = (-0.0166666666667)

    y_fxgap_9 = 0.00833333333333

    y_fxgap_10 = (-0.0166666666667)

    y_fxgap_11 = 0.00416666666667

    y_fxgap_12 = 0.05

    y_fxgap_13 = 0.0373455901902

    y_fynicn_l_1 = 0.000868642945186

    y_fynicn_l_2 = 0.203972136271

    y_fyniln_l_1 = 0.00114819592586

    y_fyniln_l_2 = 0.344642504397

    y_gfdbtnp_l_1 = 6.19935005084e-05

    y_gfdbtnp_l_2 = 0.984645217482

    y_gfdbtnp_l_3 = 0.0737924242446

    y_gfdbtnp_l_4 = (-0.0584376417269)

    y_gfdbtn_l_1 = 5.5810037311e-05

    y_ugfsrp_1 = 0.947688

    y_uleg_l_1 = (-0.0162972181781)

    y_uleg_l_2 = 0.0162972181781

    y_uleg_l_3 = 0.1

    y_gfexpn_l_1 = 0.000210646994344

    y_gfexpn_l_2 = 0.0964148144871

    y_gfexpn_l_3 = 0.19363872408

    y_gfexpn_l_4 = 0.600944699108

    y_gfexpn_l_5 = 0.109001762325

    y_gfintn_l_1 = 0.00193250998745

    y_gfintn_l_2 = 34.038852147

    y_gfrecn_l_1 = 0.000265992685534

    y_gfrecn_l_2 = 0.5764571204

    y_gfrecn_l_3 = 0.0743675317358

    y_gfrecn_l_4 = 5.57251231588

    y_gfrecn_l_5 = 0.349175347864

    y_gtn_l_1 = 0.000350526420578

    y_gtr_l_1 = 0.000390220355331

    y_gtr_l_2 = 7.39501037898

    y_gtr_l_3 = 7.39501037898

    y_gtrd_1 = (-0.000176387604876)

    y_gtrd_2 = (-0.000206546235356)

    y_gtrd_3 = (-4.93246174231e-05)

    y_gtrd_4 = (-4.93246174231e-05)

    y_gtrd_5 = (-4.93246174231e-05)

    y_gtrd_6 = 0.862481931486

    y_gtrd_7 = 0.000309352740077

    y_hgemp_1 = 0.9

    y_hgemp_2 = 40

    y_hgemp_3 = (-40)

    y_hggdp_1 = 400

    y_hggdp_2 = (-400)

    y_hgpbfir_1 = 0.975

    y_hgpbfir_2 = 10

    y_hgpbfir_3 = 10

    y_hgpbfir_4 = (-10)

    y_hgpbfir_5 = (-10)

    y_hgpbfir_6 = (-10)

    y_hgpbfir_7 = 10

    y_hgpkir_1 = 0.9

    y_hgpkir_2 = 43.1298484247

    y_hgpkir_3 = (-43.0591386594)

    y_hgynid_1 = 454.348916939

    y_hgynid_2 = (-54.3489169394)

    y_hgynid_3 = (-400)

    y_hgynid_4 = (-455.23665293)

    y_hgynid_5 = 55.2366529304

    y_hgynid_6 = 400

    y_hks_1 = 384.31948476

    y_hks_2 = (-384.31948476)

    y_hks_3 = 15.68051524

    y_hks_4 = (-15.68051524)

    y_hlept_1 = 400

    y_hlept_2 = 400

    y_hlept_3 = (-400)

    y_hmfpt_1 = 0.95

    y_hqlfpr_1 = 0.95

    y_hqlww_1 = 0.95

    y_huqpct_1 = 0.95

    y_huxb_1 = 0.324768405324

    y_huxb_2 = 0.95

    y_hxbt_1 = 0.725

    y_hxbt_2 = 0.725

    y_hxbt_3 = 290

    y_hxbt_4 = (-290)

    y_hxbt_5 = 0.275

    y_jccan_l_1 = 6.24582838478

    y_jccan_l_2 = 0.82051735145

    y_jccan_l_3 = (-0.948637916333)

    y_jccan_l_4 = 0.121328058188

    y_jccan_l_5 = 0.128120564883

    y_jccan_l_6 = 1.35223326447

    y_jccan_l_7 = 0.128120564883

    y_jkcd_l_1 = 0.000730359646

    y_jkcd_l_2 = 4.66817353822

    y_kbfi_l_1 = 4.49018358914e-05

    y_kbfi_l_2 = 0.0281084105505

    y_kbfi_l_3 = (-0.0265200751536)

    y_kbfi_l_4 = 0.0281084105505

    y_kbfi_l_5 = (-0.248867790412)

    y_kbfi_l_6 = 0.971891589449

    y_kcd_l_1 = 0.000154373410789

    y_kcd_l_2 = 0.066147038262

    y_kcd_l_3 = (-0.246673633735)

    y_kcd_l_4 = 0.933852961738

    y_kh_l_1 = 5.72922867013e-05

    y_kh_l_2 = 0.00873032740269

    y_kh_l_3 = (-0.249249311699)

    y_kh_l_4 = 0.991269672597

    y_ki_l_1 = 1.44204786648

    y_ki_l_2 = 0.014692062549

    y_ki_l_3 = 0.250723990347

    y_ki_l_4 = (-0.456739929026)

    y_ki_l_5 = 0.0711962153783

    y_ki_l_6 = (-0.307228143176)

    y_ks_l_1 = 0.0025

    y_leg_l_1 = 0.248485878175

    y_leg_l_2 = 0.751514121825

    y_leh_l_1 = 0.00641807663415

    y_leh_l_2 = 0.813979789462

    y_leh_l_3 = 0.132451786431

    y_leh_l_4 = 0.0535684241064

    y_leo_l_1 = 20.7652726744

    y_leo_l_2 = 0.756667597034

    y_leo_l_3 = (-15.6501866511)

    y_leo_l_4 = (-0.756667597034)

    y_leo_l_5 = (-0.0164258334824)

    y_lep_l_1 = 0.00788481079904

    y_leppot_l_1 = 0.0079687353701

    y_leppot_l_2 = (-0.0110028694424)

    y_leppot_l_3 = (-1.10028694424)

    y_leppot_l_4 = (-0.857254870696)

    y_lf_l_1 = 0.00617558165686

    y_lf_l_2 = 1.58659431972

    y_lfpr_1 = 0.432392517171

    y_lfpr_2 = 0.567607482829

    y_lfpr_3 = (-0.000875189202097)

    y_lfpr_4 = 0.000875189202097

    y_lhp_l_1 = 1.00059088506

    y_lhp_l_2 = 0.202289789801

    y_lhp_l_3 = (-0.202880674857)

    y_lhp_l_4 = 0.372064184885

    y_lhp_l_5 = 0.627935815115

    y_lhp_l_6 = (-0.755331857052)

    y_lhp_l_7 = (-0.00156983953779)

    y_lhp_l_8 = 0.127396041937

    y_lhp_l_9 = 0.000318490104843

    y_lur_1 = (-96.2208093896)

    y_lur_2 = 96.2208093896

    y_lurnat_1 = 0.95

    y_lww_l_1 = 0.804289649347

    y_lww_l_2 = 0.00170379588201

    y_lww_l_3 = 0.195710350653

    y_lww_l_4 = 0.318481647196

    y_lww_l_5 = (-0.318481647196)

    y_lww_l_6 = (-0.00079620411799)

    y_mfpt_l_1 = 0.0025

    y_pbfir_l_1 = 0.0025

    y_pcdr_l_1 = 1.50984819434

    y_pcdr_l_2 = (-0.509848194342)

    y_pcer_l_1 = 0.248860953365

    y_pcer_l_2 = (-0.248860953365)

    y_pcer_l_3 = (-0.248860953365)

    y_pcer_l_4 = 0.248860953365

    y_pcfr_l_1 = 1.21019336782

    y_pcfr_l_2 = (-0.14928038046)

    y_pcfr_l_3 = (-0.365198296745)

    y_pcfr_l_4 = 0.318574001625

    y_pcfr_l_5 = (-0.338884189342)

    y_pcfr_l_6 = 0.333798755712

    y_pchr_l_1 = 1.59806398567

    y_pchr_l_2 = (-0.598063985667)

    y_pcnia_l_1 = 0.0025

    y_pcor_l_1 = (-0.1436)

    y_pcor_l_2 = 0.1436

    y_pcor_l_3 = (-0.217)

    y_pcor_l_4 = 0.217

    y_pcpi_l_1 = 0.00394679077503

    y_pcpi_l_2 = 0.43067430272

    y_pcpix_l_1 = 0.00384060295377

    y_pcpix_l_2 = 0.426412064374

    y_pcxfe_l_1 = 0.0025

    y_pegfr_l_1 = 0.0025

    y_pegsr_l_1 = 0.0025

    y_pgdp_l_1 = 0.00898451406694

    y_pgfl_l_1 = 0.525153490957

    y_pgsl_l_1 = 0.514419453205

    y_phouse_l_1 = 1.89031776892

    y_phouse_l_2 = (-0.901886995515)

    y_phouse_l_3 = 0.0115692265899

    y_phouse_l_4 = 0.0115692265899

    y_phr_l_1 = 0.0025

    y_pic4_1 = 100

    y_pic4_2 = (-100)

    y_picnia_1 = 15.96

    y_picnia_2 = (-15.96)

    y_picnia_3 = 29.04

    y_picnia_4 = (-29.04)

    y_picx4_1 = 100

    y_picx4_2 = (-100)

    y_picxfe_1 = 0.404860664116

    y_picxfe_2 = 0.591171818183

    y_picxfe_3 = 0.00396751770099

    y_picxfe_4 = 0.462045372577

    y_picxfe_5 = (-0.462045372577)

    y_pieci_1 = 0.00293156716662

    y_pieci_2 = 0.00293156716662

    y_pieci_3 = 0.00293156716662

    y_pieci_4 = 0.146578358331

    y_pieci_5 = 0.839226144659

    y_pieci_6 = 0.00540079551024

    y_pieci_7 = 0.00540079551024

    y_pieci_8 = (-2.16031820409)

    y_pieci_9 = (-0.0143209721548)

    y_pieci_10 = 0.0143209721548

    y_pieci_11 = 0.327959270689

    y_pieci_12 = (-0.327959270689)

    y_pigdp_1 = 400

    y_pigdp_2 = (-400)

    y_pipxnc_1 = (-796)

    y_pipxnc_2 = 0.462801

    y_pipxnc_3 = (-0.462801)

    y_pipxnc_4 = 368.389596

    y_pipxnc_5 = 0.229745

    y_pipxnc_6 = (-0.229745)

    y_pipxnc_7 = 182.87702

    y_pipxnc_8 = (-14.9334031956)

    y_pipxnc_9 = 14.9334031956

    y_pipxnc_10 = 10

    y_pipxnc_11 = (-10)

    y_pkbfir_1 = 0.960531663984

    y_pkbfir_2 = 1.05983283594

    y_pl_l_1 = 0.0025

    y_pmo_l_1 = 0.622318401629

    y_pmo_l_2 = 0.377681598371

    y_pmo_l_3 = 0.00731956262431

    y_pmo_l_4 = (-0.00731956262431)

    y_pmo_l_5 = (-0.629637964254)

    y_pmo_l_6 = 0.234396660333

    y_pmo_l_7 = (-0.234396660333)

    y_pmo_l_8 = 0.765603339667

    y_pmp_l_1 = 0.0171179383155

    y_pmp_l_2 = 1.05645668526

    y_poil_l_1 = 0.0162031615251

    y_poilr_l_1 = 1.17135063067

    y_poilr_l_2 = (-0.346197996438)

    y_poilr_l_3 = (-0.390345197801)

    y_poilr_l_4 = 0.79951907837

    y_ptr_1 = 0.9

    y_ptr_2 = 0.05

    y_ptr_3 = 0.05

    y_pxb_l_1 = 0.00914375584343

    y_pxb_l_2 = 1.01772402773

    y_pxnc_l_1 = 0.0025

    y_pxp_l_1 = 0.6469

    y_pxp_l_2 = (-0.6469)

    y_pxp_l_3 = 0.3531

    y_pxp_l_4 = (-0.3531)

    y_pxr_l_1 = 0.0025

    y_qebfi_l_1 = 0.000358162570912

    y_qebfi_l_2 = 0.664481948351

    y_qebfi_l_3 = 0.0787039173848

    y_qebfi_l_4 = (-0.0787039173848)

    y_qebfi_l_5 = 7.87039173848

    y_qec_l_1 = 0.935665935123

    y_qec_l_2 = 0.0166517759473

    y_qec_l_3 = (-0.139711201786)

    y_qec_l_4 = 0.135400942735

    y_qec_l_5 = 0.0519925479811

    y_qecd_l_1 = 0.000593792074211

    y_qecd_l_2 = 3.98656310426

    y_qecd_l_3 = 0.00498320388032

    y_qecd_l_4 = 0.00498320388032

    y_qecd_l_5 = 0.00498320388032

    y_qecd_l_6 = 0.00498320388032

    y_qecd_l_7 = 0.00498320388032

    y_qecd_l_8 = 0.00498320388032

    y_qecd_l_9 = 0.00498320388032

    y_qecd_l_10 = 0.00498320388032

    y_qecd_l_11 = (-0.0232956396718)

    y_qecd_l_12 = (-0.584353967629)

    y_qecd_l_13 = (-0.0242284661483)

    y_qeh_l_1 = 0.0015504186377

    y_qeh_l_2 = 24.6010652056

    y_qeh_l_3 = 0.0153756657535

    y_qeh_l_4 = 0.0153756657535

    y_qeh_l_5 = 0.0153756657535

    y_qeh_l_6 = 0.0153756657535

    y_qeh_l_7 = 0.0153756657535

    y_qeh_l_8 = 0.0153756657535

    y_qeh_l_9 = 0.0153756657535

    y_qeh_l_10 = 0.0153756657535

    y_qeh_l_11 = 0.0153756657535

    y_qeh_l_12 = 0.0153756657535

    y_qeh_l_13 = 0.0153756657535

    y_qeh_l_14 = 0.0153756657535

    y_qeh_l_15 = 0.0153756657535

    y_qeh_l_16 = 0.0153756657535

    y_qeh_l_17 = 0.0153756657535

    y_qeh_l_18 = 0.0153756657535

    y_qeh_l_19 = (-0.0270350700995)

    y_qkir_l_1 = 0.00188536673771

    y_qlf_l_1 = 0.00620858571308

    y_qlf_l_2 = 1.58692282562

    y_qlhp_l_1 = 0.00465728156706

    y_qlww_l_1 = 0.0025

    y_qpxnc_l_1 = 2.98507462687

    y_qpxnc_l_2 = (-2.98507462687)

    y_qpxnc_l_3 = (-1.98507462687)

    y_qpxnc_l_4 = 1.98507462687

    y_qpxp_l_1 = 0.7195976338

    y_qpxp_l_2 = (-0.7195976338)

    y_qynidn_l_1 = 0.354822592523

    y_qynidn_l_2 = 1.13587229235

    y_qynidn_l_3 = (-0.135872292349)

    y_rbbbp_1 = (-0.189051)

    y_rbbbp_2 = 0.848879

    y_rbbbp_3 = 0.160481423829

    y_rbfi_1 = (-2.21124682364)

    y_rbfi_2 = 0.395

    y_rbfi_3 = 0.395

    y_rbfi_4 = (-0.395)

    y_rbfi_5 = (-0.5)

    y_rbfi_6 = 0.5

    y_rcar_1 = 1.22665328945

    y_rcar_2 = 0

    y_rcar_3 = 0.696748171914

    y_rcar_4 = 0.101669335039

    y_rcar_5 = 0.201582493047

    y_rccd_1 = 100

    y_rcch_1 = 100

    y_rcch_2 = (-0.0545840410668)

    y_rcch_3 = 0.7953

    y_rcch_4 = 79.53

    y_rcgain_1 = 0.32854362351

    y_rcgain_2 = 0.225785775119

    y_rcgain_3 = (-0.225785775119)

    y_reqp_1 = 0.808086

    y_reqp_2 = 0.795819

    y_reqp_3 = (-0.643090192434)

    y_rfynic_1 = 1.00400815341

    y_rfynic_2 = (-0.49108746803)

    y_rfynic_3 = (-0.144424360986)

    y_rfynic_4 = 0.631503675605

    y_rfynil_1 = 0.884413145643

    y_rfynil_2 = (-0.00726474303036)

    y_rfynil_3 = (-0.171195169347)

    y_rfynil_4 = 0.0265702779079

    y_rfynil_5 = (-0.132818819092)

    y_rfynil_6 = 0.0876033907073

    y_rfynil_7 = 0.261434600384

    y_rfynil_8 = 0.0179349568622

    y_rg10p_1 = (-0.460658806872)

    y_rg10p_2 = 0.228721864424

    y_rg10p_3 = 0.920104088065

    y_rg10p_4 = 0.423854051406

    y_rg10p_5 = (-0.210447922486)

    y_rg30p_1 = (-0.624829467707)

    y_rg30p_2 = 0.134994250522

    y_rg30p_3 = 0.938108605708

    y_rg30p_4 = 0.586157900756

    y_rg30p_5 = (-0.126639268136)

    y_rg5p_1 = (-0.349564481)

    y_rg5p_2 = 0.90221329312

    y_rg5p_3 = 0.315381721561

    y_rgfint_1 = 0.845677566688

    y_rgfint_2 = 0.154322433312

    y_rgfint_3 = 0.00556931000493

    y_rgfint_4 = (-0.00556931000493)

    y_rgw_1 = 0.00495

    y_rgw_2 = 0.00271

    y_rgw_3 = 0.00129

    y_rgw_4 = 0.00105

    y_rme_1 = 0.660306961037

    y_rme_2 = 0.884200704474

    y_rme_3 = (-0.544507665511)

    y_rme_4 = (-0.102549417082)

    y_rrff_1 = (-0.25)

    y_rrff_2 = (-0.25)

    y_rrff_3 = (-0.25)

    y_rrff_4 = (-0.25)

    y_rrtr_1 = 0.97

    y_rrtr_2 = 0.03

    y_rspnia_1 = 7.62633280279

    y_rspnia_2 = (-7.62633280279)

    y_rtb_1 = 0.799718792152

    y_rtb_2 = 0.11137355158

    y_rtb_3 = 0.770122562667

    y_rtb_4 = (-0.681214906399)

    y_rtbfi_l_1 = 5.40790262847

    y_rtbfi_l_2 = 0.0576949599629

    y_rtbfi_l_3 = 5.76949599629

    y_rtbfi_l_4 = (-0.0576949599629)

    y_rtbfi_l_5 = (-0.0123862111793)

    y_rtbfi_l_6 = 0.129531747065

    y_rtbfi_l_7 = 0

    y_rtbfi_l_8 = (-0.260110434765)

    y_rtbfi_l_9 = 0.943576128374

    y_rtinv_1 = 0.00912489842966

    y_rtinv_2 = (-0.00912489842966)

    y_rtinv_3 = 0.0330561789534

    y_rtinv_4 = 0.0356516398072

    y_rtinv_5 = 0.0329826447805

    y_rtinv_6 = 0.0355066663735

    y_rtinv_7 = (-0.066038823734)

    y_tcin_l_1 = 0.00357673139507

    y_tcin_l_2 = 8.35657418879

    y_tpn_l_1 = 0.000461426663182

    y_tpn_l_2 = 7.01937651683

    y_tpn_l_3 = 1.18756132659

    y_tpn_l_4 = (-0.187561326587)

    y_trci_1 = 0.00706626139452

    y_trci_2 = 0.810247648208

    y_trci_3 = (-0.810247648208)

    y_trci_4 = (-0.00572542167653)

    y_trp_1 = 0.603942358608

    y_trp_2 = (-0.603942358608)

    y_trp_3 = 0.236576213581

    y_trp_4 = (-0.236576213581)

    y_trp_5 = 0.000630587773923

    y_trptd_1 = 0.420215062775

    y_trptd_2 = (-0.420215062775)

    y_trptd_3 = (-0.55)

    y_trptd_4 = (-0.422749789232)

    y_trptd_5 = 0.422749789232

    y_trptd_6 = (-0.5)

    y_trpts_1 = (-0.0180202713644)

    y_trpts_2 = 0.0225987818683

    y_trpts_3 = (-0.00457851050393)

    y_trpts_4 = 0.1

    y_trpts_5 = 0.00075

    y_tryh_1 = 0.144437010525

    y_tryh_2 = (-0.0944218552605)

    y_tryh_3 = (-0.0500151552646)

    y_uxbt_l_1 = 0.0025

    y_uynicpnr_1 = 0.779183

    y_vbfi_1 = 5.96826486935

    y_vbfi_2 = 1.41987523928

    y_vbfi_3 = (-1.50480877253)

    y_vbfi_4 = (-1.50480877253)

    y_wpo_l_1 = 1.36629240684e-05

    y_wpon_l_1 = 1.2518399289e-05

    y_wpon_l_2 = 0.99460869287

    y_wpon_l_3 = 0.00146536714744

    y_wpon_l_4 = 0.408461833894

    y_wpon_l_5 = (-0.408461833894)

    y_wpon_l_6 = 0.0498372673814

    y_wpon_l_7 = (-0.0443498822123)

    y_wpon_l_8 = (-0.00103486242602)

    y_wpon_l_9 = 0.000938784388547

    y_wpon_l_10 = 0.000938784388547

    y_wpon_l_11 = 0.00466659001196

    y_wpon_l_12 = (-0.00372780562341)

    y_wps_l_1 = 3.55473075437e-05

    y_wpsn_l_1 = 1.13587229235

    y_wpsn_l_2 = (-0.135872292349)

    y_wpsn_l_3 = (-0.25)

    y_wpsn_l_4 = 0.25

    y_xb_l_1 = 6.84878469814e-05

    y_xb_l_2 = 1.0

    y_xb_l_3 = (-1.0)

    y_xbn_l_1 = 6.26139265285e-05

    y_xbn_l_2 = 1.0198018271

    y_xbn_l_3 = 1.0198018271

    y_xbn_l_4 = 1.31175365227

    y_xbn_l_5 = (-1.33155547937)

    y_xbn_l_6 = (-1.33155547937)

    y_xbo_l_1 = 0.0132470548943

    y_xbt_l_1 = 0.725

    y_xbt_l_2 = 0.725

    y_xbt_l_3 = 0.725

    y_xbt_l_4 = 0.275

    y_xbtr_l_1 = 0.95

    y_xfs_l_1 = 0.6849

    y_xfs_l_2 = (-0.6849)

    y_xfs_l_3 = 0.0386

    y_xfs_l_4 = (-0.0386)

    y_xfs_l_5 = 0.1324

    y_xfs_l_6 = (-0.1324)

    y_xfs_l_7 = 0.0429

    y_xfs_l_8 = (-0.0429)

    y_xfs_l_9 = 0.0223

    y_xfs_l_10 = (-0.0223)

    y_xfs_l_11 = 0.0395

    y_xfs_l_12 = (-0.0395)

    y_xfs_l_13 = 0.0691

    y_xfs_l_14 = (-0.0691)

    y_xfs_l_15 = 0.1203

    y_xfs_l_16 = (-0.1203)

    y_xfs_l_17 = (-0.1399)

    y_xfs_l_18 = 0.1399

    y_xfs_l_19 = (-0.0101)

    y_xfs_l_20 = 0.0101

    y_xfsn_l_1 = 4.78939925834e-05

    y_xfsn_l_2 = 1.00337294235

    y_xfsn_l_3 = (-0.00363305240167)

    y_xfsn_l_4 = (-0.00337294235067)

    y_xfsn_l_5 = (-0.544852165261)

    y_xfsn_l_6 = 0.541479222911

    y_xgap_1 = 100

    y_xgap_2 = (-100)

    y_xgap2_1 = 100

    y_xgap2_2 = (-100)

    y_xgdi_l_1 = 5.28881809334e-05

    y_xgdin_l_1 = 4.75117496854e-05

    y_xgdo_l_1 = 5.23439257193e-05

    y_xgdp_l_1 = 0.9985

    y_xgdp_l_2 = (-0.9985)

    y_xgdp_l_3 = 0.6264

    y_xgdp_l_4 = (-1.2513)

    y_xgdp_l_5 = 0.6249

    y_xgdpn_l_1 = 4.77329919533e-05

    y_xgdpn_l_2 = 1.0564013312

    y_xgdpn_l_3 = 0.021847772281

    y_xgdpn_l_4 = 0.0674519622926

    y_xgdpn_l_5 = (-0.149062669624)

    y_xgdpn_l_6 = 0.00362083951871

    y_xgdpn_l_7 = 0.00336160385466

    y_xgdpn_l_8 = 0.543020588122

    y_xgdpn_l_9 = (-0.539658984268)

    y_xgdptn_l_1 = 4.74032737815e-05

    y_xp_l_1 = 0.6526679404

    y_xp_l_2 = (-0.6526679404)

    y_xp_l_3 = 0.0361108836

    y_xp_l_4 = (-0.0361108836)

    y_xp_l_5 = 0.11825695358

    y_xp_l_6 = (-0.11825695358)

    y_xp_l_7 = 0.04216893278

    y_xp_l_8 = (-0.04216893278)

    y_xp_l_9 = 0.0365822346

    y_xp_l_10 = (-0.0365822346)

    y_xp_l_11 = 0.114213055

    y_xp_l_12 = (-0.114213055)

    y_xpn_l_1 = 4.51845246156e-05

    y_ydn_l_1 = 6.27963768217e-05

    y_ydn_l_2 = 0.998336445483

    y_ydn_l_3 = 1.13631857158

    y_ydn_l_4 = (-0.136318571582)

    y_yh_l_1 = 6.84845844852e-05

    y_yh_l_2 = 0.526533658207

    y_yh_l_3 = 0.178799300517

    y_yh_l_4 = 0.294667041275

    y_yhgap_1 = 100

    y_yhgap_2 = (-100)

    y_yhibn_l_1 = 0.00302416992206

    y_yhibn_l_2 = 63.3559682371

    y_yhl_l_1 = 0.000130066869264

    y_yhl_l_2 = (-1.16884651007)

    y_yhln_l_1 = 0.00010196408965

    y_yhln_l_2 = 1.14236648073

    y_yhp_l_1 = 0.000232413452786

    y_yhp_l_2 = (-1.10635973458)

    y_yhp_l_3 = 0.946560228789

    y_yhp_l_4 = 0.0534397712107

    y_yhpgap_1 = 100

    y_yhpgap_2 = (-100)

    y_yhpntn_l_1 = 0.00398579631637

    y_yhpntn_l_2 = 1.19125575899

    y_yhpntn_l_3 = 1.19125575899

    y_yhpntn_l_4 = 1.19125575899

    y_yhpntn_l_5 = (-1.31798027859)

    y_yhpntn_l_6 = 9.31230191482

    y_yhpntn_l_7 = (-1.11436836489)

    y_yhpntn_l_8 = (-5.43026982685)

    y_yhpntn_l_9 = (-0.71417194978)

    y_yhpntn_l_10 = (-1.64093920349)

    y_yhpshr_l_1 = 3.39337481391

    y_yhptn_l_1 = 0.000192500076699

    y_yhptn_l_2 = 0.975134043217

    y_yhptn_l_3 = 0.563603989214

    y_yhptn_l_4 = 0.102159206272

    y_yhptn_l_5 = 0.268955804126

    y_yhptn_l_6 = 0.0652810003888

    y_yhshr_l_1 = 1.28884347188

    y_yhshr_l_2 = 0.99999334858

    y_yhshr_l_3 = (-0.99999334858)

    y_yhsn_l_1 = 0.000823471829286

    y_yhsn_l_2 = 8.07609651707

    y_yhsn_l_3 = 2.34637489687

    y_yhsn_l_4 = 4.27777403212

    y_yhsn_l_5 = (-1.78462125185)

    y_yhsn_l_6 = (-11.6695043163)

    y_yhsn_l_7 = (-0.272296812186)

    y_yhsn_l_8 = 17.3716235947

    y_yhsn_l_9 = 0.0261769342453

    y_yht_l_1 = 0.000383024901591

    y_yhtgap_1 = 100

    y_yhtgap_2 = (-100)

    y_yhtn_l_1 = 0.00035095492642

    y_yhtn_l_2 = 1.00122246375

    y_yhtshr_l_1 = 5.59327644091

    y_ykbfin_l_1 = 0.000222515379076

    y_ykbfin_l_2 = 0.501135005995

    y_ykbfin_l_3 = 0.498864994005

    y_ykin_l_1 = 0.0048752071053

    y_ykin_l_2 = 15.1326363291

    y_ykin_l_3 = 0.501553881106

    y_ykin_l_4 = 0.498446118894

    y_ynicpn_l_1 = 0.000428014077811

    y_ynicpn_l_2 = 7.69278337234

    y_ynicpn_l_3 = (-4.79530526339)

    y_ynicpn_l_4 = (-1.22202005096)

    y_ynicpn_l_5 = 8.96683950232

    y_ynicpn_l_6 = (-0.675458057991)

    y_ynidn_l_1 = 0.000734588128108

    y_ynidn_l_2 = 0.683167062078

    y_ynidn_l_3 = (-0.000507771585891)

    y_ynidn_l_4 = (-0.790436568589)

    y_ynidn_l_5 = 0.107269506511

    y_ynidn_l_6 = 0.209563431411

    y_ynidn_l_7 = (-0.000157528146717)

    y_ynidn_l_8 = (-0.209563431411)

    y_yniln_l_1 = 8.92568990506e-05

    y_yniln_l_2 = 0.977159070276

    y_yniln_l_3 = 0.829114291162

    y_yniln_l_4 = 0.829114291162

    y_yniln_l_5 = 0.0418084260005

    y_yniln_l_6 = 0.0418084260005

    y_yniln_l_7 = 0.129077282837

    y_yniln_l_8 = 0.129077282837

    y_ynin_l_1 = 5.56383895262e-05

    y_ynin_l_2 = 0.999999935846

    y_ynin_l_3 = 1.17104491968

    y_ynin_l_4 = 0.0640520865577

    y_ynin_l_5 = (-0.0484572291581)

    y_ynin_l_6 = (-0.186639777078)

    y_ynirn_l_1 = 7.33377354801

    y_ynirn_l_2 = 0.951263114856

    y_ynirn_l_3 = (-0.951263114856)

    y_ynirn_l_4 = 0.00548690935542

    y_ynirn_l_5 = (-0.00548690935542)

    y_ypn_l_1 = 5.53550901718e-05

    y_ypn_l_2 = 0.988173952374

    y_ypn_l_3 = 0.549385156514

    y_ypn_l_4 = 0.159614656674

    y_ypn_l_5 = 0.291000186812

    y_zdivgr_1 = 0.00975726425743

    y_zdivgr_2 = 0.990242735743

    y_zebfi_1 = (-0.000431144211955)

    y_zebfi_2 = (-0.00050714173603)

    y_zebfi_3 = (-3.88181916088e-05)

    y_zebfi_4 = 0.00016798757544

    y_zebfi_5 = (-0.000975251482943)

    y_zebfi_6 = 0.000417269685018

    y_zebfi_7 = 9.80402248148e-06

    y_zebfi_8 = 0.00040254489385

    y_zebfi_9 = 0.000145632881593

    y_zebfi_10 = 0.000809116564154

    y_zebfi_11 = 0.000691481740712

    y_zebfi_12 = (-0.00152462990113)

    y_zebfi_13 = 0.000182102122415

    y_zebfi_14 = 0.000170960242897

    y_zebfi_15 = 0.0142945657655

    y_zebfi_16 = (-0.00425222899975)

    y_zebfi_17 = (-0.00503049733108)

    y_zebfi_18 = (-0.00112440248315)

    y_zebfi_19 = (-0.0038874369515)

    y_zebfi_20 = 0.00035570453849

    y_zebfi_21 = (-0.00035570453849)

    y_zecd_1 = (-0.000424433044911)

    y_zecd_2 = (-0.000566112732916)

    y_zecd_3 = (-0.000427835415485)

    y_zecd_4 = 4.27545061866e-06

    y_zecd_5 = (-0.00133363746841)

    y_zecd_6 = 0.00178510275432

    y_zecd_7 = (-0.000271474405975)

    y_zecd_8 = 0.000459611864377

    y_zecd_9 = 0.000428608849069

    y_zecd_10 = (-0.00111248088805)

    y_zecd_11 = 3.61133130939e-05

    y_zecd_12 = 7.97590705793e-05

    y_zecd_13 = 0.00141410574269

    y_zecd_14 = (-0.000639602744318)

    y_zecd_15 = (-0.00010841426451)

    y_zecd_16 = 0.000210363124201

    y_zecd_17 = 0.000178061664134

    y_zecd_18 = 0.000146912749167

    y_zecd_19 = (-0.000139880426754)

    y_zecd_20 = (-3.38007573296e-05)

    y_zecd_21 = 0.000166975793706

    y_zecd_22 = 0.000113506936821

    y_zecd_23 = 0.000124123127885

    y_zecd_24 = (-0.000203591971486)

    y_zecd_25 = 5.7989188193e-05

    y_zecd_26 = 0.000114280775871

    y_zecd_27 = 0.00255088738447

    y_zecd_28 = (-0.001880611807)

    y_zecd_29 = 0.0308598105755

    y_zecd_30 = (-0.00201324622316)

    y_zecd_31 = (-0.0365513581269)

    y_zecd_32 = (-0.00465896135484)

    y_zecd_33 = 0.0123637551294

    y_zeco_1 = (-7.52202049496e-05)

    y_zeco_2 = (-7.94406933181e-05)

    y_zeco_3 = (-2.05931699614e-05)

    y_zeco_4 = 0.000100439779498

    y_zeco_5 = 2.12832185698e-05

    y_zeco_6 = 1.70353153588e-05

    y_zeco_7 = 5.5012376381e-05

    y_zeco_8 = 3.68085672111e-05

    y_zeco_9 = (-0.000630171036922)

    y_zeco_10 = 0.000273875586514

    y_zeco_11 = 0.000133019756131

    y_zeco_12 = (-3.46619140531e-05)

    y_zeco_13 = 7.48142887307e-05

    y_zeco_14 = (-0.000130139477521)

    y_zeco_15 = (-0.000574849476126)

    y_zeco_16 = 0.000315791553755

    y_zeco_17 = 0.000397005436297

    y_zeco_18 = 2.60636593368e-05

    y_zeco_19 = (-6.06591388527e-05)

    y_zeco_20 = (-5.86151697491e-06)

    y_zeco_21 = 4.60869299242e-05

    y_zeco_22 = (-3.67533909379e-05)

    y_zeco_23 = 0.000205501772024

    y_zeco_24 = (-0.000240937714399)

    y_zeco_25 = (-0.000131812287659)

    y_zeco_26 = (-8.99812036284e-05)

    y_zeco_27 = 0.0011735331967

    y_zeco_28 = 0.0732239724725

    y_zeco_29 = (-0.0439002248803)

    y_zeco_30 = (-0.0221752554553)

    y_zeco_31 = 0.00192935493602

    y_zeco_32 = (-0.00907784707292)

    y_zeh_1 = (-7.82590644729e-05)

    y_zeh_2 = (-5.22177650099e-05)

    y_zeh_3 = (-5.11617748264e-05)

    y_zeh_4 = 1.06118801985e-05

    y_zeh_5 = 0.00015247021427

    y_zeh_6 = 0.000115461916411

    y_zeh_7 = 6.94775905299e-05

    y_zeh_8 = 6.915284272e-06

    y_zeh_9 = 0.000501695975955

    y_zeh_10 = (-0.000531489463237)

    y_zeh_11 = (-0.000152716722883)

    y_zeh_12 = 4.42229725272e-05

    y_zeh_13 = 0.000171026724111

    y_zeh_14 = (-0.000344325005483)

    y_zeh_15 = 5.54746256755e-05

    y_zeh_16 = 4.07823169105e-05

    y_zeh_17 = (-3.07861940478e-05)

    y_zeh_18 = (-4.58464253978e-06)

    y_zeh_19 = 7.13641520669e-06

    y_zeh_20 = 1.47073943648e-05

    y_zeh_21 = 3.83084742308e-05

    y_zeh_22 = 3.27196755109e-05

    y_zeh_23 = (-6.99285491718e-05)

    y_zeh_24 = 4.55009661188e-07

    y_zeh_25 = 5.50461376242e-05

    y_zeh_26 = 2.86419262708e-05

    y_zeh_27 = 0.00106943231558

    y_zeh_28 = 0.00426630302385

    y_zeh_29 = (-0.00573847474041)

    y_zeh_30 = (-0.00187609012218)

    y_zeh_31 = (-0.000659760712587)

    y_zeh_32 = 0.00400802255132

    y_zgap05_1 = 0.0547936526434

    y_zgap05_2 = 0.945206347357

    y_zgap10_1 = 0.0300745581094

    y_zgap10_2 = 0.969925441891

    y_zgap30_1 = 0.014106588982

    y_zgap30_2 = 0.985893411018

    y_zgapc2_1 = (-0.0141848331986)

    y_zgapc2_2 = (-0.00438957847118)

    y_zgapc2_3 = (-0.00608986499063)

    y_zgapc2_4 = 0.00127453586676

    y_zgapc2_5 = (-0.0426889990258)

    y_zgapc2_6 = 0.00775994605046

    y_zgapc2_7 = 0.0191285668792

    y_zgapc2_8 = (-0.00220795277592)

    y_zgapc2_9 = 0.194384536968

    y_zgapc2_10 = (-0.0764007234264)

    y_zgapc2_11 = (-0.0113246023485)

    y_zgapc2_12 = (-0.0155518339662)

    y_zgapc2_13 = 0.0233897407937

    y_zgapc2_14 = 0.018008438872

    y_zlhp_1 = (-0.000202321377434)

    y_zlhp_2 = (-6.54709155556e-05)

    y_zlhp_3 = (-0.000172024683014)

    y_zlhp_4 = 3.13937564958e-05

    y_zlhp_5 = (-0.00104747602255)

    y_zlhp_6 = 0.000259883906459

    y_zlhp_7 = 0.00050790958016

    y_zlhp_8 = (-4.19862687488e-05)

    y_zlhp_9 = 0.000321668804678

    y_zlhp_10 = 0.000408423219508

    y_zlhp_11 = (-0.00575496472936)

    y_zlhp_12 = 0.00581682769254

    y_zlhp_13 = (-0.000203485929498)

    y_zlhp_14 = (-0.00027154348171)

    y_zlhp_15 = 0.685794690175

    y_zlhp_16 = (-0.685794690175)

    y_zlhp_17 = (-0.685794690175)

    y_zlhp_18 = 0.685794690175

    y_zlhp_19 = 0.000278310168582

    y_zlhp_20 = 0.000278310168582

    y_zpi10_1 = 0.0300745581094

    y_zpi10_2 = 0.969925441891

    y_zpi10f_1 = 0.0300745581094

    y_zpi10f_2 = 0.969925441891

    y_zpi5_1 = 0.0817876274963

    y_zpi5_2 = 0.0221868418868

    y_zpi5_3 = 0.0250194521826

    y_zpi5_4 = (-9.00706244808e-05)

    y_zpi5_5 = (-0.145676547176)

    y_zpi5_6 = (-0.0311377360679)

    y_zpi5_7 = (-0.0294931929574)

    y_zpi5_8 = (-0.0275798582146)

    y_zpi5_9 = 0.233887334416

    y_zpi5_10 = 0.871096149059

    y_zpi5_11 = 0.174192252057

    y_zpi5_12 = (-0.0718402312689)

    y_zpi5_13 = 0.0406637195158

    y_zpi5_14 = 0.0449446239851

    y_zpib5_1 = 21.9174610574

    y_zpib5_2 = (-21.9174610574)

    y_zpib5_3 = 0.945206347357

    y_zpic30_1 = 0.014106588982

    y_zpic30_2 = 0.985893411018

    y_zpicxfe_1 = 0.380818884672

    y_zpicxfe_2 = 0.00113182715476

    y_zpicxfe_3 = 0.00146351917605

    y_zpicxfe_4 = 0.00225729733693

    y_zpicxfe_5 = 0.0460967342223

    y_zpicxfe_6 = 0.0338772671906

    y_zpicxfe_7 = 0.0228924215171

    y_zpicxfe_8 = 0.0112105032823

    y_zpicxfe_9 = (-0.0140156100481)

    y_zpicxfe_10 = 0.0011222896601

    y_zpicxfe_11 = 0.00760121840982

    y_zpicxfe_12 = (-0.00299260406007)

    y_zpicxfe_13 = 0.0470383710002

    y_zpicxfe_14 = (-0.0278318348119)

    y_zpicxfe_15 = (-0.00506170904133)

    y_zpicxfe_16 = (-0.00225028901719)

    y_zpicxfe_17 = 0.00828470603822

    y_zpicxfe_18 = 0.500251545448

    y_zpicxfe_19 = 11.937795061

    y_zpicxfe_20 = (-11.937795061)

    y_zpicxfe_21 = 6.84395376806e-05

    y_zpicxfe_22 = (-6.84395376806e-05)

    y_zpicxfe_23 = (-0.114076926212)

    y_zpicxfe_24 = 45.6307704848

    y_zpicxfe_25 = (-0.00383816812034)

    y_zpicxfe_26 = 0.00383816812034

    y_zpicxfe_27 = (-0.000695300677346)

    y_zpicxfe_28 = 0.000695300677346

    y_zpieci_1 = (-0.026022539351)

    y_zpieci_2 = 0.00320414216918

    y_zpieci_3 = 0.00402676215955

    y_zpieci_4 = 0.00650489050087

    y_zpieci_5 = 0.202430424141

    y_zpieci_6 = 0.196252633802

    y_zpieci_7 = 0.195837958296

    y_zpieci_8 = 0.0246831983934

    y_zpieci_9 = (-0.0328787076454)

    y_zpieci_10 = 0.00135903909754

    y_zpieci_11 = 0.0229838005541

    y_zpieci_12 = (-0.00862383586105)

    y_zpieci_13 = 0.148708914616

    y_zpieci_14 = (-0.0777266665551)

    y_zpieci_15 = (-0.0137748704693)

    y_zpieci_16 = (-0.00648469451174)

    y_zpieci_17 = 0.0171597038548

    y_zpieci_18 = 0.393082529889

    y_zpieci_19 = (-4.49541220961)

    y_zpieci_20 = 4.49541220961

    y_zpieci_21 = 0.000154587412169

    y_zpieci_22 = (-0.000154587412169)

    y_zpieci_23 = 0.380795785368

    y_zpieci_24 = (-152.318314147)

    y_zpieci_25 = (-0.0172443115476)

    y_zpieci_26 = 0.0172443115476

    y_zpieci_27 = (-0.00416159167724)

    y_zpieci_28 = 0.00416159167724

    y_zrff10_1 = 0.0300745581094

    y_zrff10_2 = 0.969925441891

    y_zrff30_1 = 0.014106588982

    y_zrff30_2 = 0.985893411018

    y_zrff5_1 = 0.0547936526434

    y_zrff5_2 = 0.945206347357

    y_zyh_l_1 = 8.4030342164e-05

    y_zyh_l_2 = 0.000702312990074

    y_zyh_l_3 = 0.00059883124856

    y_zyh_l_4 = 0.000469068735294

    y_zyh_l_5 = (-0.00211240097558)

    y_zyh_l_6 = 0.000165660225273

    y_zyh_l_7 = (-0.000778039304358)

    y_zyh_l_8 = 0.000290681377414

    y_zyh_l_9 = 0.000905762637463

    y_zyh_l_10 = 0.00268742554252

    y_zyh_l_11 = 0.000682359720547

    y_zyh_l_12 = (-0.00125638870298)

    y_zyh_l_13 = (-0.00185424331619)

    y_zyh_l_14 = 0.00243409867739

    y_zyh_l_15 = 0.00416166538503

    y_zyh_l_16 = 0.000750155785081

    y_zyh_l_17 = (-0.000192123546578)

    y_zyh_l_18 = (-0.000360282907619)

    y_zyhp_l_1 = 0.000863735995891

    y_zyhp_l_2 = 0.00104399392499

    y_zyhp_l_3 = 0.000941268897539

    y_zyhp_l_4 = 0.000458916642717

    y_zyhp_l_5 = (-0.00154443220559)

    y_zyhp_l_6 = (-0.000640772573525)

    y_zyhp_l_7 = (-0.000877216982815)

    y_zyhp_l_8 = (-4.0750285327e-05)

    y_zyhp_l_9 = (-0.00203693491032)

    y_zyhp_l_10 = 0.00307902287603

    y_zyhp_l_11 = 0.00153367057932

    y_zyhp_l_12 = (-0.000979443369808)

    y_zyhp_l_13 = (-0.00330791546136)

    y_zyhp_l_14 = 0.00310317204747

    y_zyhp_l_15 = 0.00422069758369

    y_zyhp_l_16 = 0.000130762149267

    y_zyhp_l_17 = 5.34207087507e-05

    y_zyhp_l_18 = (-0.00083494424337)

    y_zyhp_l_19 = 0.00225147864855

    y_zyhp_l_20 = 0.000201405002604

    y_zyhp_l_21 = (-0.000427498545256)

    y_zyhp_l_22 = (-0.000252078634627)

    y_zyhpst_l_1 = 0.0005

    y_zyhst_l_1 = 0.0005

    y_zyht_l_1 = (-0.000334830912493)

    y_zyht_l_2 = 0.000473468268016

    y_zyht_l_3 = 0.000279807258553

    y_zyht_l_4 = 0.000327960879431

    y_zyht_l_5 = (-0.00250027976398)

    y_zyht_l_6 = 0.00088556815263

    y_zyht_l_7 = (-0.00121488126811)

    y_zyht_l_8 = 8.52432970988e-05

    y_zyht_l_9 = 0.00217284254807

    y_zyht_l_10 = 0.00313028920932

    y_zyht_l_11 = 0.00211420687194

    y_zyht_l_12 = 0.000248144569686

    y_zyht_l_13 = (-0.000746405493548)

    y_zyht_l_14 = 0.00274434958239

    y_zyht_l_15 = 0.00270824568765

    y_zyht_l_16 = 0.000679378874986

    y_zyht_l_17 = 0.000308986169352

    y_zyht_l_18 = (-6.93511660157e-05)

    y_zyht_l_19 = 0.00183760811329

    y_zyht_l_20 = 0.000509649917682

    y_zyht_l_21 = 9.47084299939e-05

    y_zyht_l_22 = 0.0002426388839

    y_zyhtst_l_1 = 0.0005

    y_zynid_1 = (-0.000102077846072)

    y_zynid_2 = 0.000348695205252

    y_zynid_3 = 0.000252250306328

    y_zynid_4 = 0.00020597993691

    y_zynid_5 = 0.000352649102887

    y_zynid_6 = (-0.00091171528933)

    y_zynid_7 = (-0.000833252281803)

    y_zynid_8 = 0.000222028205174

    y_zynid_9 = 0.00117029026307

    y_zynid_10 = (-0.000704847602418)

    y_zynid_11 = (-0.00509852446865)

    y_zynid_12 = 0.00166112741624

    y_zynid_13 = 0.000634556266817

    y_zynid_14 = 0.0013968199402

    y_zynid_15 = 0.00251911216766

    y_zynid_16 = (-0.00251911216766)

    y_zynid_17 = 0.0129976207113

    y_zynid_18 = (-0.0129976207113)

    y_zynid_19 = (-0.00311724318294)

    y_zynid_20 = 0.00311724318294

    y_zynid_21 = (-0.0193734997949)

    y_zynid_22 = 0.0193734997949

    y_zynid_23 = 0.00697401009889

    y_zynid_24 = (-0.00697401009889)

    y_zynid_25 = 0.00296525365916

    rho_trp_a = 0.0

    trp_ā = 0.0

    rho_fiscal = 0.97

    rho_fiscalav = 0.9

    fiscal_egfe  = .01

    fiscal_egfl  = .01

    av = 1

    f̄iscal = 0

    fpxrr_l̄ = 0.0

    rho_fpxrr_l = 0.0

    pmo_l̄ = 0.0

    rho_pmo_l = 0.0

    emo_l̄ = 0.0

    rho_emo_l = 0.0

    ugfdbtp_l̄ = 0.0

    rho_ugfdbtp_l = 0.0

    rho_gfsrt = 0.0
end
