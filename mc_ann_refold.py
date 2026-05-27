from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ann_guess import EPS, _build_absolute_model, _integrate_trapezoid


def _log_interp_positive(x_new, x_old, y_old):
    x_new = np.asarray(x_new, dtype=np.float64)
    x_old = np.asarray(x_old, dtype=np.float64)
    y_old = np.asarray(y_old, dtype=np.float64)
    mask = (x_old > 0.0) & (y_old > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.zeros_like(x_new, dtype=np.float64)

    log_y = np.interp(
        np.log10(np.clip(x_new, EPS, None)),
        np.log10(x_old[mask]),
        np.log10(np.clip(y_old[mask], EPS, None)),
        left=np.log10(EPS),
        right=np.log10(EPS),
    )
    return np.power(10.0, log_y)


def _load_absolute_checkpoint(torch, model_source, device):
    if hasattr(model_source, "seek"):
        model_source.seek(0)
    checkpoint = torch.load(model_source, map_location=device, weights_only=False)
    required = ["model_state_dict", "x_mean", "x_std", "logphi_mean", "logphi_std", "energies"]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(
            "The selected checkpoint is not an absolute ANN model. "
            f"Missing keys: {missing}"
        )
    return checkpoint


def _predict_absolute_batch(counts_batch, model_source):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for ANN + MC shaking. Install torch or choose another mode."
        ) from exc

    counts_batch = np.asarray(counts_batch, dtype=np.float32)
    if counts_batch.ndim != 2 or counts_batch.shape[1] != 11:
        raise ValueError(f"The absolute ANN expects samples with 11 counts, got {counts_batch.shape}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = _load_absolute_checkpoint(torch, model_source, device)
    energies_eV = np.asarray(checkpoint["energies"], dtype=np.float32)

    model = _build_absolute_model(torch, in_dim=11, out_dim=len(energies_eV)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = np.log10(np.clip(counts_batch, EPS, None)).astype(np.float32)
    x = ((x - checkpoint["x_mean"]) / checkpoint["x_std"]).astype(np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_shape, _, pred_logphi_scaled = model(x_tensor)

    pred_shape = pred_shape.cpu().numpy()
    pred_logphi = (
        pred_logphi_scaled.cpu().numpy() * float(checkpoint["logphi_std"])
        + float(checkpoint["logphi_mean"])
    )
    pred_phi_total = 10.0 ** pred_logphi

    phi_shape = pred_shape / (energies_eV[None, :] + EPS)
    area = _integrate_trapezoid(phi_shape, energies_eV, axis=1)[:, None]
    spectra_per_eV = phi_shape / (area + EPS) * pred_phi_total[:, None]
    spectra_per_mev = spectra_per_eV * 1.0e6

    return {
        "energies_mev": energies_eV / 1.0e6,
        "spectra_per_mev": spectra_per_mev,
        "lethargy": spectra_per_mev * (energies_eV[None, :] / 1.0e6),
        "fluence_total": pred_phi_total,
        "device": device,
        "model": getattr(model_source, "name", str(model_source)),
    }


def make_count_replicas(counts, sigma_abs, n_samples, sigma_scale, seed):
    rng = np.random.default_rng(seed)
    replicas = rng.normal(
        loc=counts[None, :],
        scale=(sigma_scale * sigma_abs)[None, :],
        size=(n_samples, len(counts)),
    )
    return np.clip(replicas, EPS, None).astype(np.float32)


def run_mc_ann_refold(
    counts_data,
    model_source,
    response_matrix,
    energy_bins,
    selected_detectors,
    n_samples,
    sigma_scale,
    chi_sigma_scale,
    seed,
    uncertainty_is_relative=True,
):
    counts_data = np.asarray(counts_data, dtype=np.float64)
    response_matrix = np.asarray(response_matrix, dtype=np.float64)
    energy_bins = np.asarray(energy_bins, dtype=np.float64)

    counts = counts_data[:, 0]
    sigma_input = counts_data[:, 1] if counts_data.shape[1] > 1 else np.sqrt(np.clip(counts, 1.0, None))
    sigma_abs = sigma_input * counts if uncertainty_is_relative else sigma_input
    sigma_abs = np.clip(sigma_abs, EPS, None)

    if len(counts) != 11:
        raise ValueError(f"ANN + MC shaking expects 11 detector counts, got {len(counts)}.")

    replicas = make_count_replicas(counts, sigma_abs, n_samples, sigma_scale, seed)
    prediction = _predict_absolute_batch(replicas, model_source)

    xbins = energy_bins[:, 2]
    dE = energy_bins[:, 1] - energy_bins[:, 0]
    spectra_on_grid = np.vstack(
        [
            _log_interp_positive(xbins, prediction["energies_mev"], spectrum)
            for spectrum in prediction["spectra_per_mev"]
        ]
    )

    folded_counts = np.sum(
        spectra_on_grid[:, None, :] * response_matrix[None, :, :] * dE[None, None, :],
        axis=2,
    )

    selected = np.asarray(selected_detectors, dtype=int)
    valid = (counts[selected] > 0.0) & (sigma_abs[selected] > 0.0)
    if not np.any(valid):
        raise ValueError("No valid selected detector is available for MC refold scoring.")
    score_detectors = selected[valid]

    sigma_eff = np.clip(chi_sigma_scale * sigma_abs[score_detectors], EPS, None)
    residuals = (
        folded_counts[:, score_detectors] - counts[score_detectors][None, :]
    ) / sigma_eff[None, :]
    chi2 = np.mean(np.square(residuals), axis=1)
    best_idx = int(np.argmin(chi2))

    fig = plot_mc_refold_summary(
        xbins,
        counts[score_detectors],
        sigma_abs[score_detectors],
        folded_counts[:, score_detectors],
        spectra_on_grid,
        chi2,
        best_idx,
    )

    return {
        "energies_mev": xbins,
        "best_spectrum_per_mev": spectra_on_grid[best_idx],
        "best_index": best_idx,
        "best_chi2": float(chi2[best_idx]),
        "median_chi2": float(np.median(chi2)),
        "p10_chi2": float(np.percentile(chi2, 10)),
        "p90_chi2": float(np.percentile(chi2, 90)),
        "best_fluence_total": float(np.sum(spectra_on_grid[best_idx] * dE)),
        "ann_fluence_total": float(prediction["fluence_total"][best_idx]),
        "folded_best_counts": folded_counts[best_idx],
        "chi2": chi2,
        "residuals": residuals,
        "replicas": replicas,
        "fig_summary": fig,
        "device": prediction["device"],
        "model": Path(prediction["model"]).name,
    }


def plot_mc_refold_summary(xbins, measured_counts, sigma_abs, folded_counts, spectra_on_grid, chi2, best_idx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

    x_det = np.arange(1, len(measured_counts) + 1)
    axes[0].errorbar(x_det, measured_counts, yerr=sigma_abs, fmt="o", color="black", label="Measured")
    axes[0].plot(x_det, folded_counts[best_idx], "s-", color="tab:blue", label=f"Best refold J={chi2[best_idx]:.3g}")
    axes[0].fill_between(
        x_det,
        np.percentile(folded_counts, 16, axis=0),
        np.percentile(folded_counts, 84, axis=0),
        alpha=0.2,
        color="tab:blue",
        label="MC refold 16-84%",
    )
    axes[0].set_xlabel("Selected detector index")
    axes[0].set_ylabel("Counts")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    lethargy = spectra_on_grid * xbins[None, :]
    axes[1].fill_between(
        xbins,
        np.percentile(lethargy, 16, axis=0),
        np.percentile(lethargy, 84, axis=0),
        step="mid",
        alpha=0.25,
        color="tab:orange",
        label="ANN MC 16-84%",
    )
    axes[1].step(xbins, lethargy[best_idx], where="mid", color="black", linewidth=2.0, label="Best MC guess")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Neutron Energy [MeV]")
    axes[1].set_ylabel("Fluence per unit lethargy")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("ANN + MC shaking with response refold selection")
    return fig
