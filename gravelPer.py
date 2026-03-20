import numpy as np
import matplotlib.pyplot as plt

def gravel(R, data, x, tolerance, energy_file, max_iter):
    x = x.astype(float).copy()
    n, m = R.shape
    eps = 1e-300

    mask = data[:, 0] > 0
    R = R[mask, :]
    data = np.array(data)[mask]
    meas = data[:, 0]
    uncer = data[:, 1]

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    error = []
    stepcount = 1

    rdot = R @ (x * dE)
    J0 = np.sum((np.log(np.clip(rdot, eps, None)) - np.log(np.clip(meas, eps, None)))**2)
    logIter = f"Initial J = {J0:.2e}\n"

    while J0 > tolerance and stepcount <= max_iter:
        rdot = np.clip(R @ (x * dE), eps, None)

        for j in range(m):
            Wij = R[:, j] * x[j] * dE[j] / rdot
            den = np.sum(Wij)
            if den > 0:
                num = np.sum(Wij * np.log(meas / rdot))
                x[j] *= np.exp(num / den)

        rdot = np.clip(R @ (x * dE), eps, None)
        J = np.sum((np.log(rdot) - np.log(meas))**2)
        error.append(J)
        logIter += f"Iteration {stepcount}, J = {J:.2e}\n"
        J0 = J
        stepcount += 1

    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(meas, label="measured")
    axC.plot(rdot, label="evaluated")
    axC.set_ylabel("Counts")
    axC.legend()

    return x, np.array(error), figC, logIter
