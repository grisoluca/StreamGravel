import numpy as np
from numpy import log,exp
import matplotlib.pyplot as plt

def gravel(R,data,x,tolerance,energy_file,max_iter, make_plot=True):
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
    while J0 > tolerance and stepcount <= max_iter:
        rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)
        log_ratio = log(meas / rdot)
        log_correction = np.zeros(m)

        for j in range(m):
            weights = R[:, j] * x[j] * dE[j] / rdot * inv_rho2
            num = np.dot(weights, log_ratio)
            num = np.nan_to_num(num)
            den = np.sum(weights)

            if den != 0:
                log_correction[j] = num / den

        # Reject overly aggressive multiplicative steps instead of letting the
        # spectrum and the refolded counts diverge.
        accepted = False
        damping = 1.0
        for _ in range(20):
            exponent = np.clip(damping * log_correction, -50.0, 50.0)
            candidate = np.clip(x * exp(exponent), eps, None)
            candidate_rdot = np.clip(
                np.array([np.sum(R[i, :] * candidate * dE) for i in range(n)]),
                eps,
                None,
            )
            candidate_chi2 = np.sum(
                np.square(np.log(candidate_rdot) - np.log(meas)) * inv_rho2
            )
            candidate_J = candidate_chi2 / n
            if np.isfinite(candidate_J) and candidate_J <= J0:
                x = candidate
                rdot = candidate_rdot
                J = candidate_J
                accepted = True
                break
            damping *= 0.5

        if not accepted:
            logIter += (
                f"Iteration {stepcount}: stopped because no update reduced "
                f"chi-squared J below {J0:.2e}\n"
            )
            break

        error.append(J)
        logIter += (
            f"Iteration {stepcount}, reduced chi-squared J = {J:.2e}, "
            f"damping = {damping:.3f}\n"
        )
        stepcount += 1
        J0 = J
        
    #with st.expander("📘 Iteration log"):
     #   st.text_area("Output GRAVEL", logIter, height=300)    
    
    figC = None
    if make_plot:
        figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
        axC.plot(meas, label="measured")
        axC.plot(rdot, label="evaluated")
        axC.set_ylabel("Counts")
        axC.legend()
    #col.pyplot(figC)

    return(x,np.array(error),figC,logIter)
