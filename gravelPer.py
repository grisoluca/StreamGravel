import numpy as np
import matplotlib.pyplot as plt

def gravel(R, data, x, tolerance, energy_file, max_iter):
    x = np.array(x, dtype=float).copy()
    data = np.array(data, dtype=float)
    eps = 1e-30

    mask = data[:, 0] > 0
    R = R[mask, :]
    meas = data[mask, 0]
    n, m = R.shape

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    error = []
    stepcount = 1

    rdot = R @ (x * dE)
    rdot = np.clip(rdot, eps, None)
    meas = np.clip(meas, eps, None)

    J0 = np.sum((np.log(rdot) - np.log(meas))**2)
    logIter = f"Initial J = {J0:.2e}\n"

    while J0 > tolerance and stepcount <= max_iter:
        rdot = np.clip(R @ (x * dE), eps, None)
        x_old = x.copy()

        for j in range(m):
            Wij = R[:, j] * x_old[j] * dE[j] / rdot
            den = np.sum(Wij)
            if den > 0:
                num = np.sum(Wij * np.log(meas / rdot))
                x[j] = x_old[j] * np.exp(num / den)

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
