import numpy as np

# # Channel from one satellite to one satellite
def Chl_1SAT1SAT(N_rS, N_tS, Loc1SAT_0, Loc1SAT_1, fw):
    # N_tS number of transmitted antennas on satellite
    # N_rG number of received antennas on ground station
    pl_exponent = 2 # path-loss exponent
    c = 3e8
    dist = np.linalg.norm(Loc1SAT_0-Loc1SAT_1, ord=2)*1000 # km to m
    PL_mk = 20*np.log10(4*np.pi*fw/c) + 10*pl_exponent*np.log10(dist)
    patlossSha_mac = PL_mk
    # Small-scale fading
    Small_scale = (np.random.randn(N_rS,N_tS)+1j*np.random.randn(N_rS,N_tS))/np.sqrt(2)
    # Combine large-scale fading and small-scale fading
    H = np.sqrt(1/(10**(patlossSha_mac/10)))*Small_scale
    return H

def Func_ChlSAT2SAT(N_tS, N_rS, Loc_SAT, fw):
    num_SAT = np.shape(Loc_SAT)[0]
    ChlSAT2SAT = np.zeros((num_SAT,num_SAT,N_rS,N_tS), dtype=np.complex128)
    DistSAT2SAT = np.zeros((num_SAT,num_SAT))
    G_T_linear = 40 # dBi
    G_R_linear = 30 # dBi
    antenna_gain_amplitude = np.sqrt(10**(G_T_linear/10) * 10**(G_R_linear/10))
    for s in range(num_SAT):
        for sp in range(num_SAT):
            z = np.linalg.norm(Loc_SAT[s,:]-Loc_SAT[sp,:], ord=2) # km to m
            DistSAT2SAT[s,sp] = z
            if z!=0:
                ChlSAT2SAT[s,sp] = Chl_1SAT1SAT(N_rS, N_tS, Loc_SAT[s,:], Loc_SAT[sp,:], fw)*antenna_gain_amplitude
    return ChlSAT2SAT, DistSAT2SAT

# # Channel from one satellite to ground users
def Chl_1SAT1DT(N_tS, N_rG, Loc1SAT, Loc1DT, fw):
    # N_tS number of transmitted antennas on satellite
    # N_rG number of received antennas on ground station
    pl_exponent = 2 # path-loss exponent
    # fw = 20*10**9;
    c = 3e8
    dist = np.linalg.norm(Loc1SAT-Loc1DT, ord=2)*1000 # km to m
    PL_mk = 20*np.log10(4*np.pi*fw/c) + 10*pl_exponent*np.log10(dist)
    patlossSha_mac = PL_mk
    # Small-scale fading
    Small_scale = (np.random.randn(N_rG,N_tS)+1j*np.random.randn(N_rG,N_tS))/np.sqrt(2)
    # Combine large-scale fading and small-scale fading
    H = np.sqrt(1/(10**(patlossSha_mac/10)))*Small_scale
    return H, dist

def Chl_1DP1SAT(N_rS, N_tG, Loc1SAT, Loc1DP, fw):
    # N_rS number of received antennas on satellite
    # N_tG number of transmitted antennas on ground station
    pl_exponent = 2 # path-loss exponent
    # fw = 20*10**9;
    c = 3e8
    dist = np.linalg.norm(Loc1SAT-Loc1DP, ord=2)*1000 # km to m
    PL_mk = 20*np.log10(4*np.pi*fw/c) + 10*pl_exponent*np.log10(dist)
    patlossSha_mac = PL_mk
    # Small-scale fading
    Small_scale = (np.random.randn(N_rS,N_tG)+1j*np.random.randn(N_rS,N_tG))/np.sqrt(2)
    # Combine large-scale fading and small-scale fading
    H = np.sqrt(1/(10**(patlossSha_mac/10)))*Small_scale
    return H, dist

def Func_ChlGS2SAT(N_tS, N_rS, N_tG, N_rG, Loc_SAT, Loc_DP, Loc_DT, fw):
    num_SAT = np.shape(Loc_SAT)[0]
    num_DP = np.shape(Loc_DP)[0]
    num_DT = num_DP
    ChlSAT2DT = np.zeros((num_SAT,num_DT,N_rG,N_tS), dtype=np.complex128)
    DistSAT2DT = np.zeros((num_SAT,num_DT))
    ChlDP2SAT = np.zeros((num_SAT,num_DT,N_rS,N_tG), dtype=np.complex128)
    DistDP2SAT = np.zeros((num_SAT,num_DT))
    antenna_gain_amplitudeDP2SAT = np.sqrt(10**(40/10) * 10**(30/10))
    antenna_gain_amplitudeSAT2DT = np.sqrt(10**(40/10) * 10**(30/10))
    for s in range(num_SAT):
        for m in range(num_DP):
            ChlDP2SAT[s,m], DistDP2SAT[s,m] = Chl_1DP1SAT(N_rS, N_tG, Loc_SAT[s,:], Loc_DP[m,:], fw)
            ChlSAT2DT[s,m], DistSAT2DT[s,m]  = Chl_1SAT1DT(N_tS, N_rG, Loc_SAT[s,:], Loc_DT[m,:], fw)
    return ChlDP2SAT*antenna_gain_amplitudeDP2SAT, ChlSAT2DT*antenna_gain_amplitudeSAT2DT, DistSAT2DT, DistDP2SAT
