import numpy as np

def precoder_w(H: np.ndarray,alpha: float | None = None,eps: float = 1e-12,) -> np.ndarray:
    """
    Single-stream RZF/MMSE precoder *direction* (unit-norm).
    Power is NOT inside w. You will enforce E|s|^2 = pt outside.

    H: (Nr x Nt)
    returns w: (Nt x 1) with ||w||=1
    """
    Nr, Nt = H.shape

    HHh = H @ H.conj().T                          # (Nr x Nr)
    R = HHh + (alpha + eps) * np.eye(Nr, dtype=H.dtype)
    Rinv = np.linalg.solve(R, np.eye(Nr, dtype=H.dtype))
    W0 = H.conj().T @ Rinv                         # (Nt x Nr)

    w = W0[:, 0:1]                                 # pick 1 stream (first column)

    # unit-norm so that E||x||^2 = E|s|^2 = pt
    wnorm = np.linalg.norm(w)
    if wnorm < eps:
        raise ValueError("w is ~zero; check H.")
    w = w / wnorm
    return w


def combiner_g(H: np.ndarray,w: np.ndarray,eps: float = 1e-12,) -> np.ndarray:
    """
    Single-stream combiner direction, explicitly pt-free.
    For single stream with white noise, MMSE direction == MRC direction: g ∝ h = H w.

    H: (Nr x Nt)
    w: (Nt x 1) unit-norm recommended
    returns g: (Nr x 1) (optionally unit-norm)
    """
    Nr, Nt = H.shape
    w = np.asarray(w)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    if w.shape != (Nt, 1):
        raise ValueError(f"w must be shape (Nt,1)=({Nt},1). Got {w.shape}.")

    h = H @ w                                      # (Nr x 1)
    g = h.copy()

    gnorm = np.linalg.norm(g)
    if gnorm < eps:
        raise ValueError("g is ~zero; check H or w.")
    g = g / gnorm

    return g

def Func_BeamDP2SAT(ChlDP2SAT):
    alpha = 1e-2
    num_SAT = np.shape(ChlDP2SAT)[0]
    num_DT = np.shape(ChlDP2SAT)[1]
    N_rS = np.shape(ChlDP2SAT)[2]
    N_tG = np.shape(ChlDP2SAT)[3]
    BeamT_DP2SAT = np.zeros((num_SAT,num_DT,N_tG), dtype=np.complex128)
    BeamR_DP2SAT = np.zeros((num_SAT,num_DT,N_rS), dtype=np.complex128)
    for s in range(num_SAT):
        for m in range(num_DT):
            H = ChlDP2SAT[s,m,:,:]
            w = precoder_w(H, alpha)
            BeamT_DP2SAT[s,m,:] = w.squeeze()
            g = combiner_g(H, w)
            BeamR_DP2SAT[s,m,:] = g.squeeze()
    return BeamT_DP2SAT, BeamR_DP2SAT

def Func_BeamSAT2DT(ChlSAT2DT):
    alpha = 1e-2
    num_SAT = np.shape(ChlSAT2DT)[0]
    num_DT = np.shape(ChlSAT2DT)[1]
    N_rS = np.shape(ChlSAT2DT)[2]
    N_tG = np.shape(ChlSAT2DT)[3]
    BeamT_SAT2DT = np.zeros((num_SAT,num_DT,N_tG), dtype=np.complex128)
    BeamR_SAT2DT = np.zeros((num_SAT,num_DT,N_rS), dtype=np.complex128)
    for s in range(num_SAT):
        for m in range(num_DT):
            H = ChlSAT2DT[s,m,:,:]
            w = precoder_w(H, alpha)
            BeamT_SAT2DT[s,m,:] = w.squeeze()
            g = combiner_g(H, w)
            BeamR_SAT2DT[s,m,:] = g.squeeze()
    return BeamT_SAT2DT, BeamR_SAT2DT

def Func_BeamSAT2SAT(ChlSAT2SAT):
    alpha = 1e-2
    num_SAT = np.shape(ChlSAT2SAT)[0]
    num_SAT = np.shape(ChlSAT2SAT)[1]
    N_rS = np.shape(ChlSAT2SAT)[2]
    N_tS = np.shape(ChlSAT2SAT)[3]
    BeamT_SAT2SAT = np.zeros((num_SAT,num_SAT,N_tS), dtype=np.complex128)
    BeamR_SAT2SAT = np.zeros((num_SAT,num_SAT,N_rS), dtype=np.complex128)
    for s in range(num_SAT):
        for sp in range(num_SAT):
            if s!=sp:
                H = ChlSAT2SAT[s,sp,:,:]
                w = precoder_w(H, alpha)
                BeamT_SAT2SAT[s,sp,:] = w.squeeze()
                g = combiner_g(H, w)
                BeamR_SAT2SAT[s,sp,:] = g.squeeze()
    return BeamT_SAT2SAT, BeamR_SAT2SAT
