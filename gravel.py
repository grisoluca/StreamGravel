import numpy as np
from numpy import log,exp
import matplotlib.pyplot as plt
import streamlit as st

def gravel(R,data,x,tolerance,energy_file):
    """
    R --> Response matrix, shape is (n,m)
    N --> data from each detector, shape is (n,)
    x --> initial guess at neutron spectrum, shape is (m,)
    tolerance --> user-defined stopping condition
    """
    x = x.copy()
    n = R.shape[0]
    m = R.shape[1]
    # eliminate any channel with 0 count
    R = np.array([R[i] for i in range(n) if data[i] != 0])
    data = np.array([val for val in data if val > 0])
    # redefine number of rows after the reduction
    n = R.shape[0]
    error = []
    stepcount = 1
    
    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    E_bin_sx = energies[:,0] # bin sx
    E_bin_dx = energies[:,1] # bin dx
    #E_new = energies[:, 2] # bin centrale
    dE = E_bin_dx-E_bin_sx

    rdot = np.array([np.sum(R[i, :] * x * dE) for i in range(n)])
    J0 = np.sum((rdot - data) ** 2) / np.sum(rdot)
    logIter = f"Initial chi-squared J = {J0:.2e}\n"
    
   
    
    logIter = ""
    
    while J0 > tolerance:
        W = np.zeros((n, m))
        rdot = np.array([np.sum(R[i, :] * x * dE) for i in range(n)])

        for j in range(m):
            W[:, j] = data * R[:, j] * x[j] * dE[j] / rdot
            num = np.dot(W[:, j], log(data / rdot))
            num = np.nan_to_num(num)
            den = np.sum(W[:, j])

            if den != 0:
                x[j] *= exp(num / den)

        rdot = np.array([np.sum(R[i, :] * x * dE) for i in range(n)])
        J = np.sum((rdot - data) ** 2) / np.sum(rdot)
        error.append(J)

        logIter += f"Iteration {stepcount}, chi-squared J = {J:.2e}\n"
        stepcount += 1
        J0 = J
        
    with st.expander("📘 Iteration log"):
        st.text_area("Output GRAVEL", logIter, height=300)    
    
    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(data, label="measured")
    axC.plot(rdot, label="evaluated")
    axC.legend()
    #col.pyplot(figC)

    return(x,np.array(error),figC,logIter)
