import numpy as np
from numpy import log,exp
import matplotlib.pyplot as plt
import streamlit as st

def gravel(R,data,x,tolerance,energy_file,max_iter):
    """
    R --> Response matrix, shape is (n,m)
    N --> data from each detector, shape is (n,)
    x --> initial guess at neutron spectrum, shape is (m,)
    tolerance --> user-defined stopping condition
    """
    x = x.copy()
    eps = 1e-300
    n = R.shape[0]
    m = R.shape[1]
    # eliminate any channel with 0 count
    R = np.array([R[i] for i in range(n) if data[i,0] != 0])
    data = np.array(data)
    meas = np.clip(data[data[:, 0] > 0][:, 0], eps, None)
    uncer = np.clip(data[data[:, 0] > 0][:, 1], eps, None) #relative uncertainties rho=sigma/misura
    inv_rho2 = 1 / np.square(uncer)
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

    x = np.clip(x, eps, None)
    rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)
    chi2 = np.sum(np.square(np.log(rdot) - np.log(meas)) * inv_rho2)
    J0 = chi2 / n
    logIter = f"Initial reduced chi-squared J = {J0:.2e}\n"
    while J0 > tolerance and stepcount<=max_iter:
        W = np.zeros((n, m))
        rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)

        for j in range(m):
            W[:, j] = R[:, j] * x[j] * dE[j] / rdot
            weighted_W = W[:, j] * inv_rho2
            num = np.dot(weighted_W, log(meas / rdot))
            num = np.nan_to_num(num)
            den = np.sum(weighted_W)

            if den != 0:
                x[j] *= exp(num / den)

        x = np.clip(x, eps, None)
        rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)
        chi2 = np.sum(np.square(np.log(rdot) - np.log(meas)) * inv_rho2)
        J = chi2 / n
        error.append(J)

        logIter += f"Iteration {stepcount}, reduced chi-squared J = {J:.2e}\n"
        stepcount += 1
        J0 = J
        
    #with st.expander("📘 Iteration log"):
     #   st.text_area("Output GRAVEL", logIter, height=300)    
    
    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(meas, label="measured")
    axC.plot(rdot, label="evaluated")
    axC.set_ylabel("Counts")
    axC.legend()
    #col.pyplot(figC)

    return(x,np.array(error),figC,logIter)
