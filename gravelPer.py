import numpy as np
import matplotlib.pyplot as plt


def _smooth_positive(x, strength=0.08):
    if strength <= 0:
        return x.copy()
    y = np.log(np.clip(x, 1e-300, None))
    ys = y.copy()
    ys[1:-1] = (1 - strength) * y[1:-1] + 0.5 * strength * (y[:-2] + y[2:])
    return np.exp(ys)


def gravel(R, data, x, tolerance, energy_file, max_iter,
           smoothing=0.08, stop_rise_patience=25):
    """
    Empirical GRAVEL-like unfolding, close to the user's original version
    but with better numerical stability and safer stopping logic.
    """

    eps = 1e-300

    x = np.asarray(x, dtype=float).copy()
    R = np.asarray(R, dtype=float)
    data = np.asarray(data, dtype=float)

    mask = data[:, 0] > 0
    if np.count_nonzero(mask) == 0:
        raise ValueError("No positive measured counts")

    R = R[mask, :]
    meas = np.clip(data[mask, 0], eps, None)
    uncer = np.clip(data[mask, 1], eps, None)
    inv_rho2 = 1 / np.square(uncer)

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    dE = energies[:, 1] - energies[:, 0]

    if len(dE) != R.shape[1]:
        raise ValueError("Energy binning and response matrix do not match")

    x = np.clip(x, eps, None)

    def forward(phi):
        return np.clip(R @ (phi * dE), eps, None)

    def merit(phi):
        r = forward(phi)
        chi2 = np.sum(np.square(np.log(r) - np.log(meas)) * inv_rho2)
        J = chi2 / len(meas)
        return J, r

    error = []
    stepcount = 1
    log_lines = []

    J0, rdot = merit(x)
    best_J = J0
    best_x = x.copy()
    best_rdot = rdot.copy()
    best_iter = 0
    worse_count = 0

    log_lines.append(f"Initial J = {J0:.6e}")

    while stepcount <= max_iter:
        x_old = x.copy()
        rdot = forward(x_old)

        for j in range(R.shape[1]):
            Wij = R[:, j] * x_old[j] * dE[j] / rdot
            weighted_Wij = Wij * inv_rho2
            den = np.sum(weighted_Wij)

            if den > 0:
                num = np.sum(weighted_Wij * np.log(meas / rdot))
                x[j] = x_old[j] * np.exp(num / den)
            else:
                x[j] = x_old[j]

        x = np.clip(x, eps, None)

        if smoothing > 0:
            x = _smooth_positive(x, strength=smoothing)

        J, rdot = merit(x)
        error.append(J)

        rel_change = np.linalg.norm(x - x_old) / max(np.linalg.norm(x_old), eps)
        log_lines.append(f"Iter {stepcount:4d} | J = {J:.6e} | rel_change = {rel_change:.3e}")

        if J < best_J:
            best_J = J
            best_x = x.copy()
            best_rdot = rdot.copy()
            best_iter = stepcount
            worse_count = 0
        else:
            worse_count += 1

        if J <= tolerance:
            log_lines.append(f"Stop: J <= tolerance at iteration {stepcount}")
            break

        #if worse_count >= stop_rise_patience:
        #    log_lines.append(
        #        f"Stop: J worsened for {stop_rise_patience} iterations after best iter {best_iter}"
        #    )
        #    break

        #if rel_change < 1e-8:
        #    log_lines.append(f"Stop: rel_change < 1e-8 at iteration {stepcount}")
        #    break

        stepcount += 1

    log_lines.append(f"Best iteration = {best_iter}")
    log_lines.append(f"Best J = {best_J:.6e}")

    figC, axC = plt.subplots(figsize=(6, 4), layout='constrained')
    axC.plot(meas, 'o-', label="measured")
    axC.plot(best_rdot, 's--', label="evaluated")
    axC.set_xlabel("Detector / channel index")
    axC.set_ylabel("Counts")
    axC.grid(True, alpha=0.4)
    axC.legend()

    return best_x, np.array(error), figC, "\n".join(log_lines)
