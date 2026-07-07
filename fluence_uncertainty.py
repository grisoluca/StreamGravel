import numpy as np

from gravel import gravel
from gravel_old import gravel_old
from mlem import mlem


def _run_unfolding(algorithm, R, data, xguess, tolerance, energy_file, max_iter):
    if algorithm == "Gravel":
        return gravel(R, data, xguess, tolerance, energy_file, max_iter, make_plot=False)[0]
    if algorithm == "GRAVEL_old":
        return gravel_old(R, data, xguess, tolerance, energy_file, max_iter, make_plot=False)[0]
    return mlem(R, data, xguess, tolerance, energy_file, max_iter, make_plot=False)[0]


def estimate_integral_fluence_uncertainty(
    algorithm,
    R,
    data,
    xguess,
    tolerance,
    energy_file,
    max_iter,
    dE,
    n_samples=50,
    seed=12345,
):
    """
    Propagate count uncertainties to the unfolded integral fluence.

    The counts file stores relative standard uncertainties in column 2, so each
    replica perturbs the selected measured counts with an independent Gaussian.
    The returned uncertainty is the sample standard deviation of integral
    fluence values, i.e. a 1-sigma uncertainty conditional on R and the guess.
    """
    if n_samples <= 1:
        return None

    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    counts = data[:, 0]
    rel_unc = np.clip(data[:, 1], 0.0, None)
    sigma_abs = np.abs(counts) * rel_unc
    integrals = []
    eps = 1e-300

    for _ in range(int(n_samples)):
        replica_data = data.copy()
        replica_counts = rng.normal(counts, sigma_abs)
        replica_data[:, 0] = np.where(counts > 0.0, np.clip(replica_counts, eps, None), counts)

        unfolded = _run_unfolding(
            algorithm,
            R,
            replica_data,
            xguess.copy(),
            tolerance,
            energy_file,
            max_iter,
        )
        integral = float(np.sum(unfolded * dE))
        if np.isfinite(integral):
            integrals.append(integral)

    integrals = np.asarray(integrals, dtype=float)
    if integrals.size < 2:
        return None

    std = float(np.std(integrals, ddof=1))
    mean = float(np.mean(integrals))
    p16, p84 = np.percentile(integrals, [16, 84])

    return {
        "mean": mean,
        "std": std,
        "rel_std": std / abs(mean) if mean != 0.0 else np.nan,
        "p16": float(p16),
        "p84": float(p84),
        "n_success": int(integrals.size),
        "n_requested": int(n_samples),
        "seed": int(seed),
    }
