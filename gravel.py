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
    J0 = 0 ; dJ0 = 1 ; ddJ = 1
    error = []
    stepcount = 1
    
    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    E_bin_sx = energies[:,0] # bin sx
    E_bin_dx = energies[:,1] # bin dx
    #E_new = energies[:, 2] # bin centrale
    dE = E_bin_dx-E_bin_sx
    
    log = ""
    
    while ddJ > tolerance:
        W = np.zeros((n,m))
        rdot = np.zeros((n,))
        for i in range(n):
            rdot[i] = np.sum(R[i,:]*x*dE)#primo folding tra spettro di guess e matrici di risposta

        for j in range(m):

            W[:,j] = data*R[:,j]*x[j]*dE[j] / rdot
            num = np.dot(W[:,j],log(data/rdot))

            num = np.nan_to_num(num)
            den = sum(W[:,j])

            if den == 0:
                x[j] *= 1
            else:
                x[j] *= exp(num/den)

        J = sum((rdot-data)**2) / sum(rdot)
        dJ = J0-J
        ddJ = abs(dJ-dJ0)
        J0 = J
        error.append(ddJ)
        log += f"Iteration {stepcount}, ddJ = {ddJ:.2e}\n"
        stepcount += 1
        dJ0 = dJ
        
    with st.expander("📘 Iteration log"):
        st.text_area("Output GRAVEL", log, height=300)    
    
    figC, axC = plt.subplots()
    axC.plot(data, label="measured")
    axC.plot(rdot, label="evaluated")
    axC.legend()
    st.pyplot(figC)

    return(x,np.array(error))
