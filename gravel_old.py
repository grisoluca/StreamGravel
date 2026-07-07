import numpy as np
from numpy import exp, log
import matplotlib.pyplot as plt


def gravel_old(R, data, x, tolerance, energy_file, max_iter, make_plot=True):
    """Historical GRAVEL update with measured counts inside the weights."""
    x = x.copy()
    eps = 1e-300
    n, m = R.shape

    valid = data[:, 0] != 0
    R = np.asarray(R)[valid]
    data = np.asarray(data)
    meas = np.clip(data[valid, 0], eps, None)
    rho = np.clip(data[valid, 1], eps, None)
    inv_rho2 = 1 / np.square(rho)
    n = R.shape[0]

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter="\t")
    dE = energies[:, 1] - energies[:, 0]

    x = np.clip(x, eps, None)
    rdot = np.clip(np.sum(R * x[None, :] * dE[None, :], axis=1), eps, None)
    J0 = np.sum(np.square(log(rdot) - log(meas)) * inv_rho2) / n
    error = []
    stepcount = 1
    log_iter = f"Initial reduced chi-squared J = {J0:.2e}\n"

    while J0 > tolerance and stepcount <= max_iter:
        rdot = np.clip(np.sum(R * x[None, :] * dE[None, :], axis=1), eps, None)
        log_ratio = log(meas / rdot)

        for j in range(m):
            weights = meas * R[:, j] * x[j] * dE[j] / rdot
            denominator = np.sum(weights)
            if denominator != 0:
                numerator = np.nan_to_num(np.dot(weights, log_ratio))
                x[j] *= exp(np.clip(numerator / denominator, -50.0, 50.0))

        x = np.clip(x, eps, None)
        rdot = np.clip(np.sum(R * x[None, :] * dE[None, :], axis=1), eps, None)
        J = np.sum(np.square(log(rdot) - log(meas)) * inv_rho2) / n
        error.append(J)
        log_iter += f"Iteration {stepcount}, reduced chi-squared J = {J:.2e}\n"
        stepcount += 1
        J0 = J

    fig_counts = None
    if make_plot:
        fig_counts, ax_counts = plt.subplots(figsize=(6, 4), layout="constrained")
        ax_counts.plot(meas, label="measured")
        ax_counts.plot(rdot, label="evaluated")
        ax_counts.set_ylabel("Counts")
        ax_counts.legend()

    return x, np.asarray(error), fig_counts, log_iter
