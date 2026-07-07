import numpy as np
import matplotlib.pyplot as plt


def mlem(R, data, x, tolerance, energy_file,max_iter, make_plot=True):
    x = x.copy()
    eps = 1e-300
    n, m = R.shape

    # Elimina canali con 0 conteggi
    R = np.array([R[i] for i in range(n) if data[i,0] != 0])
    data = np.array(data)
    meas = np.clip(data[data[:, 0] > 0][:, 0], eps, None)
    uncer = np.clip(data[data[:, 0] > 0][:, 1], eps, None) #relative uncertainties rho=sigma/misura
    inv_rho2 = 1 / np.square(uncer)
    n = R.shape[0]

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    x = np.clip(x, eps, None)
    rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)
    chi2 = np.sum(np.square(np.log(rdot) - np.log(meas)) * inv_rho2)
    J0 = chi2 / n
    logIter = f"Initial reduced chi-squared J = {J0:.2e}\n"

    error = []
    stepcount = 1

    while J0 > tolerance and stepcount <=max_iter:
        vector = np.zeros(n)

        for i in range(n):
            rdot_i = np.sum(R[i, :] * x * dE)
            if rdot_i > 0:
                vector[i] = meas[i] / rdot_i
            else:
                vector[i] = 0.0

        for j in range(m):
            num = np.sum(R[:, j] * vector)
            den = np.sum(R[:, j])
            if den != 0:
                x[j] *= num / den

        x = np.clip(x, eps, None)
        rdot = np.clip(np.array([np.sum(R[i, :] * x * dE) for i in range(n)]), eps, None)
        chi2 = np.sum(np.square(np.log(rdot) - np.log(meas)) * inv_rho2)
        J = chi2 / n
        error.append(J)

        logIter += f"Iteration {stepcount}, reduced chi-squared J = {J:.2e}\n"
        stepcount += 1
        J0 = J

    figC = None
    if make_plot:
        figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
        axC.plot(meas, label="measured")
        axC.plot(rdot, label="evaluated")
        axC.legend()

    return x, np.array(error), figC, logIter
