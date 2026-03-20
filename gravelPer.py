import numpy as np
import matplotlib.pyplot as plt


def _smooth_log_spectrum(x, strength=0.15):
    if strength <= 0:
        return x.copy()

    y = np.log(np.clip(x, 1e-300, None))
    ys = y.copy()
    ys[1:-1] = (1 - strength) * y[1:-1] + 0.5 * strength * (y[:-2] + y[2:])
    return np.exp(ys)


def gravel(
    R,
    data,
    x0,
    tolerance,
    energy_file,
    max_iter,
    smoothing=0.10,
    use_poisson_sigma=False,
    stop_rise_patience=30,
    rel_change_stop=1e-7
):
    """
    GRAVEL unfolding with:
    - consistent discrete forward operator A = R * dE
    - weighted logarithmic update
    - chi2/n tracking
    - best-iteration storage
    - optional smoothing in log-space

    Parameters
    ----------
    R : ndarray, shape (n_det, n_bin)
    data : ndarray, shape (n_det, 2)
        data[:,0] = measured counts
        data[:,1] = relative uncertainty sigma_i / N_i
    x0 : ndarray, shape (n_bin,)
        initial guess spectrum dPhi/dE
    tolerance : float
        stop when chi2/n <= tolerance
    energy_file : uploaded file-like object
        columns: E_left, E_right, E_center
    max_iter : int
    smoothing : float
        0 = no smoothing, typical 0.05--0.20
    use_poisson_sigma : bool
        if True, use sigma = sqrt(meas) instead of relative uncertainties
    stop_rise_patience : int
        stop if chi2/n keeps worsening after the best point
    rel_change_stop : float
        stop if relative change in spectrum becomes tiny
    """

    eps = 1e-300

    R = np.asarray(R, dtype=float)
    data = np.asarray(data, dtype=float)
    x = np.asarray(x0, dtype=float).copy()

    if R.ndim != 2:
        raise ValueError("R must be 2D")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("data must have at least 2 columns: counts, relative uncertainty")
    if x.ndim != 1:
        raise ValueError("x0 must be 1D")
    if R.shape[1] != x.shape[0]:
        raise ValueError("R columns must match x0 length")

    mask = data[:, 0] > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("All measured counts are zero or negative")

    R = R[mask, :]
    meas = np.clip(data[mask, 0], eps, None)
    rho = data[mask, 1]

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    if energies.ndim != 2 or energies.shape[1] < 2:
        raise ValueError("Energy file must contain at least 2 columns: left and right bin edges")

    dE = energies[:, 1] - energies[:, 0]
    if len(dE) != R.shape[1]:
        raise ValueError("Energy bins length does not match response matrix columns")

    if np.any(dE <= 0):
        raise ValueError("All bin widths dE must be positive")

    A = R * dE[np.newaxis, :]

    if use_poisson_sigma:
        sigma = np.sqrt(meas)
    else:
        rho = np.where(rho > 0, rho, 1.0)
        sigma = rho * meas

    sigma = np.clip(sigma, eps, None)
    x = np.clip(x, eps, None)

    def forward(phi):
        return np.clip(A @ phi, eps, None)

    def chi2_ndf(phi):
        r = forward(phi)
        return np.sum(((meas - r) / sigma) ** 2) / len(meas), r

    chi2_0, rdot = chi2_ndf(x)
    best_x = x.copy()
    best_rdot = rdot.copy()
    best_chi2 = chi2_0
    best_iter = 0

    error = [chi2_0]
    log_lines = [f"Initial chi2/n = {chi2_0:.6e}"]

    worse_count = 0

    for step in range(1, max_iter + 1):
        x_old = x.copy()
        rdot = forward(x_old)

        for j in range(A.shape[1]):
            Wij = (A[:, j] * x_old[j]) / (rdot * sigma**2)
            den = np.sum(Wij)

            if den > 0:
                num = np.sum(Wij * np.log(meas / rdot))
                x[j] = x_old[j] * np.exp(num / den)
            else:
                x[j] = x_old[j]

        x = np.clip(x, eps, None)

        if smoothing > 0:
            x = _smooth_log_spectrum(x, strength=smoothing)

        chi2_now, rdot = chi2_ndf(x)
        error.append(chi2_now)

        rel_change = np.linalg.norm(x - x_old) / max(np.linalg.norm(x_old), eps)
        log_lines.append(
            f"Iter {step:4d} | chi2/n = {chi2_now:.6e} | rel_change = {rel_change:.3e}"
        )

        if chi2_now < best_chi2:
            best_chi2 = chi2_now
            best_x = x.copy()
            best_rdot = rdot.copy()
            best_iter = step
            worse_count = 0
        else:
            worse_count += 1

        if chi2_now <= tolerance:
            log_lines.append(f"Stop: chi2/n <= tolerance at iter {step}")
            break

        if rel_change < rel_change_stop:
            log_lines.append(f"Stop: relative change < {rel_change_stop:.1e} at iter {step}")
            break

        if worse_count >= stop_rise_patience:
            log_lines.append(
                f"Stop: chi2/n worsening for {stop_rise_patience} iterations after best iter {best_iter}"
            )
            break

    log_lines.append(f"Best iteration = {best_iter}")
    log_lines.append(f"Best chi2/n = {best_chi2:.6e}")

    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(meas, 'o-', label="measured")
    axC.plot(best_rdot, 's--', label="best refolded")
    axC.set_xlabel("Detector / channel index")
    axC.set_ylabel("Counts")
    axC.grid(True, alpha=0.4)
    axC.legend()

    return best_x, np.array(error), figC, "\n".join(log_lines)

