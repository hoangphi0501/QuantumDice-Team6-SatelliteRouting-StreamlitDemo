# Feb 1, 2026

import numpy as np

def Func_AllLinks(num_DT, num_SAT):
    Link_DP2SAT = []
    Link_SAT2DT = []
    for m in range(num_DT):
        for s in range(num_SAT):
            Link_DP2SAT.append((m,s))
            Link_SAT2DT.append((s,m))
            
    Link_SAT2SAT = []
    for s in range(num_SAT):
        for sp in range(num_SAT):
            if s != sp:
                Link_SAT2SAT.append((s,sp))
    return Link_DP2SAT,Link_SAT2SAT,Link_SAT2DT

# Two layers
def Func_2LayerLinks(num_DT, Layer1, Layer2):
    Link_DP2SAT = []
    Link_SAT2DT = []
    for m in range(num_DT):
        for s in Layer1:
            Link_DP2SAT.append((m,s))
            
    for m in range(num_DT):
        for s in Layer2:
            Link_SAT2DT.append((s,m))

    Link_SAT2SAT = []
    for s in Layer1:
        for sp in Layer2:
            Link_SAT2SAT.append((s,sp))
    return Link_DP2SAT,Link_SAT2SAT,Link_SAT2DT

def Func_VariableIndex(num_DT,Link_DP2SAT,Link_SAT2SAT,Link_SAT2DT):
    Index_DP2SAT = []
    for m in range(num_DT):
        for link in Link_DP2SAT:
            if link[0]==m:
                Index_DP2SAT.append(("xp", m, link))

    Index_SAT2SAT = []
    for m in range(num_DT):
        for link in Link_SAT2SAT:
            Index_SAT2SAT.append(("xs", m, link))
    return Index_DP2SAT, Index_SAT2SAT

