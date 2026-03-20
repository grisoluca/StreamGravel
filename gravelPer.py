import numpy as np
import matplotlib.pyplot as plt

def gravel(R, data, x0, tolerance, energy_file, max_iter):
    """
    GRAVEL / modified SAND-II unfolding

    Parameters
    ----------
    R : array, shape (n_det, n_bin)
        Response matrix per unit differential fluence.
    data : array, shape (n_det, 2)
        Column 0: measured counts N_i
        Column 1: relative uncertainty rho_i = sigma_i / N_i
    x0 : array, shape (n_bin,)
        Initial guess for differential spectrum dPhi/dE
    tolerance : float
        Target chi2 per degree of freedom
    energy_file : uploaded txt
        Columns: E_left, E_right, E_center
    max_iter : int
        Maximum number of iterations
    """

    eps = 1e-30

    x = np.asarray(x0, dtype=float).copy()
    data = np.asarray(data, dtype=float)

    mask = data[:, 0] > 0
    R = np.asarray(R, dtype=float)[mask, :]
    data = data[mask]

    meas = np.clip(data[:, 0], eps, None)
    rho = data[:, 1]

    rho = np.where(rho > 0, rho, 1.0)
    sigma = np.clip(rho * meas, eps, None)

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    A = R * dE[np.newaxis, :]

    n, m = A.shape
    error = []
    log_lines = []

    x = np.clip(x, eps, None)

    rdot = np.clip(A @ x, eps, None)
    chi2 = np.sum(((meas - rdot) / sigma) ** 2)
    chi2_ndf = chi2 / n
    log_lines.append(f"Initial chi2/n = {chi2_ndf:.6e}")

    for step in range(1, max_iter + 1):
        rdot = np.clip(A @ x, eps, None)
        x_old = x.copy()

        for j in range(m):
            Wij = (A[:, j] * x_old[j]) / (rdot * sigma**2)
            den = np.sum(Wij)

            if den > 0:
                num = np.sum(Wij * np.log(meas / rdot))
                x[j] = x_old[j] * np.exp(num / den)
            else:
                x[j] = x_old[j]

        x = np.clip(x, eps, None)
        rdot = np.clip(A @ x, eps, None)

        chi2 = np.sum(((meas - rdot) / sigma) ** 2)
        chi2_ndf = chi2 / n
        error.append(chi2_ndf)
        log_lines.append(f"Iteration {step}, chi2/n = {chi2_ndf:.6e}")

        rel_change = np.linalg.norm(x - x_old) / np.linalg.norm(x_old)

        if chi2_ndf <= tolerance:
            log_lines.append(f"Stopped: chi2/n <= tolerance at iteration {step}")
            break

        if rel_change < 1e-6:
            log_lines.append(f"Stopped: relative spectrum change < 1e-6 at iteration {step}")
            break

    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(meas, 'o-', label="measured")
    axC.plot(rdot, 's--', label="refolded")
    axC.set_xlabel("Detector / channel index")
    axC.set_ylabel("Counts")
    axC.grid(True, alpha=0.4)
    axC.legend()

    logIter = "\n".join(log_lines)
    return x, np.array(error), figC, logIter
