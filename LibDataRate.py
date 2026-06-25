# Jan 28, 2026

import numpy as np

# Radio Communications
def Func_DataRateDP2SAT(BeamT_DP2SAT,BeamR_DP2SAT,ChlDP2SAT,P_DP2SAT,BWw,sq_sigma_w):
    num_SAT = np.shape(ChlDP2SAT)[0]
    num_DT = np.shape(ChlDP2SAT)[1]
    DR_up = np.zeros((num_SAT,num_DT))
    for k in range(num_DT):
        for s in range(num_SAT):
            DR_up[s,k] = BWw*np.log2(1 + np.linalg.norm(BeamR_DP2SAT[s,k,:]@ChlDP2SAT[s,k,:,:]@BeamT_DP2SAT[s,k,:])**2*P_DP2SAT[s,k]/sq_sigma_w)
    return DR_up

def Func_DataRateSAT2DT(BeamT_SAT2DT,BeamR_SAT2DT,ChlSAT2DT,P_SAT2DT,BWw,sq_sigma_w):
    num_SAT = np.shape(ChlSAT2DT)[0]
    num_DT = np.shape(ChlSAT2DT)[1]
    DR_down = np.zeros((num_SAT,num_DT))
    for k in range(num_DT):
        for s in range(num_SAT):
            DR_down[s,k] = BWw*np.log2(1 + np.linalg.norm(BeamR_SAT2DT[s,k,:]@ChlSAT2DT[s,k,:,:]@BeamT_SAT2DT[s,k,:])**2*P_SAT2DT[s,k]/sq_sigma_w)
    return DR_down

# Laser Communications
def Func_DataRateSAT2SAT(ChlSAT2SAT,P_SAT2SAT,BWl,sq_sigma_l=1e-12):
    num_SAT = np.shape(ChlSAT2SAT)[0]
    DR_isl = np.zeros((num_SAT,num_SAT))
    for s in range(num_SAT):
        for sp in range(num_SAT):
            DR_isl[s,sp] = BWl*np.log2(1 + ChlSAT2SAT[s,sp]*P_SAT2SAT[s,sp]/sq_sigma_l)
    return DR_isl

def Func_DataRateSAT2SAT(BeamT_SAT2SAT,BeamR_SAT2SAT,ChlSAT2SAT,P_SAT2SAT,BWw,sq_sigma_w):
    num_SAT = np.shape(ChlSAT2SAT)[0]
    DR_isl = np.zeros((num_SAT,num_SAT))
    for s in range(num_SAT):
        for sp in range(num_SAT):
            if s!=sp:
                DR_isl[s,sp] = BWw*np.log2(1 + np.linalg.norm(BeamR_SAT2SAT[s,sp,:]@ChlSAT2SAT[s,sp,:,:]@BeamT_SAT2SAT[s,sp,:])**2*P_SAT2SAT[s,sp]/sq_sigma_w)
    return DR_isl