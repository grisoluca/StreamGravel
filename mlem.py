import numpy as np
import matplotlib.pyplot as plt


def mlem(R, data, x, tolerance, energy_file,max_iter):
    x = x.copy()
    n, m = R.shape

    # Elimina canali con 0 conteggi
    R = np.array([R[i] for i in range(n) if data[i] != 0])
    data = np.array([val for val in data if val > 0])
    n = R.shape[0]

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    rdot = np.array([np.sum(R[i, :] * x * dE) for i in range(n)])
    J0 = np.sum((rdot - data) ** 2) / np.sum(rdot)
    logIter = f"Initial chi-squared J = {J0:.2e}\n"

    error = []
    stepcount = 1

    while J0 > tolerance or stepcount <=max_iter:
        vector = np.zeros(n)

        for i in range(n):
            rdot_i = np.sum(R[i, :] * x * dE)
            if rdot_i > 0:
                vector[i] = data[i] / rdot_i
            else:
                vector[i] = 0.0

        for j in range(m):
            num = np.sum(R[:, j] * vector)
            den = np.sum(R[:, j])
            if den != 0:
                x[j] *= num / den

        rdot = np.array([np.sum(R[i, :] * x * dE) for i in range(n)])
        J = np.sum((rdot - data) ** 2) / np.sum(rdot)
        error.append(J)

        logIter += f"Iteration {stepcount}, chi-squared J = {J:.2e}\n"
        stepcount += 1
        J0 = J

    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(data, label="measured")
    axC.plot(rdot, label="evaluated")
    axC.legend()

    return x, np.array(error), figC, logIter