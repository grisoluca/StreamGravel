from pathlib import Path

import numpy as np


DEFAULT_ANN_PROJECT_DIR = Path(
    r"C:\Users\griso\Desktop\UNI\DOC\Simulations\Untitled Folder\DirSpe_spectra\ANN"
)
APP_MODELS_DIR = Path(__file__).resolve().parent / "models"
EPS = 1e-30


def _clean_path(path):
    return Path(str(path).strip().strip('"').strip("'")).expanduser()


def resolve_ann_models_dir(project_dir):
    """Accept either the ANN project folder or its models folder."""
    if not project_dir:
        return None

    base_dir = _clean_path(project_dir)
    candidates = []

    if base_dir.name.lower() in {"model", "models"}:
        candidates.extend([base_dir, base_dir.with_name("models")])

    candidates.extend([base_dir / "models", base_dir])

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def _candidate_model_dirs(project_dir=None):
    candidates = []

    selected_dir = resolve_ann_models_dir(project_dir)
    if selected_dir is not None:
        candidates.append(selected_dir)

    default_dir = resolve_ann_models_dir(DEFAULT_ANN_PROJECT_DIR)
    if default_dir is not None:
        candidates.append(default_dir)

    candidates.extend([APP_MODELS_DIR, Path.cwd() / "models", Path.cwd()])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen or not candidate.exists() or not candidate.is_dir():
            continue
        seen.add(candidate)
        yield candidate


def find_latest_ann_model(project_dir):
    """Return the newest absolute ANN checkpoint in an ANN project folder."""
    checkpoints = []
    for models_dir in _candidate_model_dirs(project_dir):
        checkpoints.extend(models_dir.glob("*.pt"))
    if not checkpoints:
        return None

    absolute_models = [p for p in checkpoints if "absolute" in p.name.lower()]
    candidates = absolute_models or checkpoints
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_ann_model_path(model_path, project_dir=None):
    """Resolve a checkpoint from an absolute path, a filename, or an ANN folder."""
    model = _clean_path(model_path)
    if model.exists():
        return model

    candidates = []
    for models_dir in _candidate_model_dirs(project_dir):
        candidates.append(models_dir / model.name)

    if project_dir:
        candidates.append(_clean_path(project_dir) / model.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    details = f" Searched also: {searched}" if searched else ""
    raise FileNotFoundError(f"ANN checkpoint not found: {model}.{details}")


def _integrate_trapezoid(y, x, axis=-1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def _build_absolute_model(torch, in_dim, out_dim):
    nn = torch.nn

    class AbsoluteUnfoldingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Dropout(0.05),
                nn.Linear(128, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(0.05),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(0.05),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Dropout(0.03),
            )
            self.shape_head = nn.Linear(128, out_dim)
            self.scale_head = nn.Linear(128, 1)

        def forward(self, x):
            features = self.shared(x)
            shape_logits = self.shape_head(features)
            shape_prob = torch.softmax(shape_logits, dim=1)
            logphi_scaled = self.scale_head(features).squeeze(1)
            return shape_prob, shape_logits, logphi_scaled

    return AbsoluteUnfoldingMLP()


def predict_absolute_guess(counts_raw, model_path, project_dir=None):
    """Predict an absolute differential spectrum from raw detector counts.

    The ANN checkpoint returns dPhi/dE in cm^-2 s^-1 eV^-1. StreamGravel uses
    MeV energy files, so the returned spectrum is converted to cm^-2 s^-1 MeV^-1.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for the neural-network initial guess. "
            "Install torch or choose another initial guess mode."
        ) from exc

    if hasattr(model_path, "read"):
        model_source = model_path
        model_label = getattr(model_path, "name", "uploaded checkpoint")
    else:
        model_source = resolve_ann_model_path(model_path, project_dir)
        model_label = str(model_source)

    counts = np.asarray(counts_raw, dtype=np.float32)
    if counts.ndim == 1:
        counts = counts[None, :]
    if counts.shape[1] != 11:
        raise ValueError(
            f"The absolute ANN expects 11 detector counts, got {counts.shape[1]}."
        )
    if np.any(counts < 0.0):
        raise ValueError("ANN input counts must be non-negative.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    energies_eV = np.asarray(checkpoint["energies"], dtype=np.float32)
    model = _build_absolute_model(torch, in_dim=counts.shape[1], out_dim=len(energies_eV)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = np.log10(np.clip(counts, EPS, None)).astype(np.float32)
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
    spectrum_per_eV = phi_shape / (area + EPS) * pred_phi_total[:, None]

    return {
        "energies_mev": energies_eV / 1.0e6,
        "spectrum_per_mev": spectrum_per_eV[0] * 1.0e6,
        "lethargy_shape": pred_shape[0],
        "fluence_total": float(pred_phi_total[0]),
        "model_path": model_label,
        "device": device,
    }
