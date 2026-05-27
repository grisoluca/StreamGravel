import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gravel import gravel
from mlem import mlem
from rebin import rebin
from response_matrix import response_matrix
from ann_guess import predict_absolute_guess
from mc_ann_refold import run_mc_ann_refold
import matplotlib.ticker as ticker
import io

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLE_LINAC_ZIP = EXAMPLES_DIR / "example-LINAC.zip"


def load_guess_spectrum(uploaded_file, is_lethargic, energy_file, target_xbins, target_len):
    uploaded_file.seek(0)
    guess_spect = np.loadtxt(uploaded_file, delimiter='\t')
    xbins_guess = guess_spect[:, 0]
    xguess_raw = guess_spect[:, 1]

    if is_lethargic:
        xguess_raw = xguess_raw / xbins_guess

    needs_rebin = (
        len(xguess_raw) != target_len
        or not np.allclose(xbins_guess, target_xbins, rtol=1e-3, atol=0.0)
    )
    if needs_rebin:
        _, xguess, fig_rebin = rebin(xbins_guess, xguess_raw, energy_file)
    else:
        xguess = xguess_raw
        fig_rebin = None

    return xguess, fig_rebin


# --------------------- CONFIGURAZIONE ---------------------
st.set_page_config(
    page_title="GRAVEL Unfolding",
    page_icon="LogoNML-Black.png",  # può essere un'emoji o un file
    initial_sidebar_state="expanded",
    layout="wide"
)

#st.set_page_config(layout="wide")
st.title("Neutron Spectrum Unfolding")
#st.title("fdfffffffffffffffffff")

# --------------------- SIDEBAR (Controlli) ---------------------
st.sidebar.header("⚙️ Unfolding parameters")
initial_guess_type = st.sidebar.selectbox(
    "Initial Guess Spectrum:",
    ["From file", "Constant", "Neural network", "ANN + MC shaking"],
)
mmin = st.sidebar.number_input(
    "Min energy [MeV] for constant guess", 
    min_value=0.0, 
    max_value=100000.0, 
    value=1e-8, 
    step=1e1, 
    format="%.1e")
mmax = st.sidebar.number_input(
    "Max energy [MeV] for constant guess", 
    min_value=0.0, 
    max_value=100000.0, 
    value=1.0, 
    step=1e1, 
    format="%.1e")
unfolding_type = st.sidebar.selectbox("Unfolding algorithm:", ["Gravel", "MLEM"])
tol = st.sidebar.number_input(
    "🔍 Reduced chi-squared value to stop iterations",
    min_value=1e-12, 
    max_value=10.0, 
    value=1.0,
    step=1e-2, 
    format="%.2e"
)
max_iter = st.sidebar.number_input(
    "🔁 Max number of iterations", 
    min_value=1, 
    max_value=1000000, 
    value=100, 
    step=1
)
log_plot_ymin = st.sidebar.number_input(
    "Log-log plot y min",
    min_value=1e-30,
    max_value=1e30,
    value=1e-4,
    step=1e-4,
    format="%.1e"
)
log_plot_ymax = st.sidebar.number_input(
    "Log-log plot y max",
    min_value=1e-30,
    max_value=1e30,
    value=1e13,
    step=1e12,
    format="%.1e"
)
if log_plot_ymax <= log_plot_ymin:
    st.sidebar.error("Log-log plot y max must be greater than y min.")
    st.stop()

ann_uploaded_model = None
selected_ann_model_path = None
selected_ann_model_key = ""
mc_n_samples = 400
mc_sigma_scale = 1.0
mc_chi_sigma_scale = 1.0
mc_seed = 42
mc_plot_top = 20
mc_uncertainty_is_relative = True
if initial_guess_type in ("Neural network", "ANN + MC shaking"):
    st.sidebar.markdown("#### Neural-network guess")
    ann_uploaded_model = st.sidebar.file_uploader(
        "Upload ANN checkpoint (.pt)",
        type="pt",
        help="Upload the absolute ANN checkpoint to use as the initial GRAVEL guess.",
    )
    selected_ann_model_path = ann_uploaded_model
    selected_ann_model_key = ann_uploaded_model.name if ann_uploaded_model is not None else ""
    if ann_uploaded_model is None:
        st.sidebar.info("Upload a .pt checkpoint to enable the neural-network guess.")

if initial_guess_type == "ANN + MC shaking":
    st.sidebar.markdown("#### MC shaking")
    mc_n_samples = st.sidebar.number_input(
        "MC samples",
        min_value=10,
        max_value=10000,
        value=400,
        step=10,
    )
    mc_sigma_scale = st.sidebar.number_input(
        "Shaking sigma scale",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.2f",
    )
    mc_chi_sigma_scale = st.sidebar.number_input(
        "Refold chi sigma scale",
        min_value=1e-12,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.2f",
    )
    mc_seed = st.sidebar.number_input(
        "MC random seed",
        min_value=0,
        max_value=1000000,
        value=42,
        step=1,
    )
    mc_plot_top = st.sidebar.number_input(
        "MC spectra shown",
        min_value=1,
        max_value=200,
        value=20,
        step=1,
    )
    mc_uncertainty_is_relative = st.sidebar.checkbox(
        "Counts uncertainty column is relative",
        value=True,
    )

# --------------------- ESEMPI SCARICA---------------------
st.markdown("### 📦 Download here an example file")
with st.expander("📦 Expand for example"):
    with EXAMPLE_LINAC_ZIP.open("rb") as f:
        st.download_button(
            label="⬇️ Download Example - LINAC",
            data=f,
            file_name="example_data_unfolding.zip",
            mime="application/zip"
        )

    st.markdown("Example files include:\n"
                "- Response Matrix (TXT, Response Function for each column, tab-separated) \n"
                "- Measurede Counts (TXT, 1st col: counts, 2nd col: relative uncertainty) \n"
                "- Energy bins (TXT, 1st col: left boundary of the bin, 2nd col: right boundary of the bin, 3rd col: central energy) \n"
                "- Guess spectrum (TXT, 1st col: energy in MeV, 2nd col: differential spectrum in energy dPhi/dE) \n"
                "\n➡️ Upload them below to try the unfolding application."
                "\n➡️ ATTENTION: Example Guess Spectrum is per Unit lethargy")


# --------------------- FILE UPLOAD ---------------------
st.markdown("Upload datas: Response Matrix, Measured Counts, Energy bins of the repsonse functions, Initial Guess Spectrum")

# File uploader
with st.container ():
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        response_file = st.file_uploader("📁 Response matrix", type="txt")
        counts_file = st.file_uploader("📈 Measured counts", type="txt")

    with col_u2:
        energy_file = st.file_uploader("⚡ Energy bins (MeV)", type="txt")
        comparison_guess_file = None
        comparison_is_letargic = False
        if initial_guess_type == "From file":
            guess_file = st.file_uploader("🧠 Initial guess spectrum", type="txt")
            is_letargic = st.checkbox("Guess spectrum per unit lethargy (in dΦ/dE*E)", value=False)
        elif initial_guess_type in ("Neural network", "ANN + MC shaking"):
            guess_file = None
            is_letargic = False
            comparison_guess_file = st.file_uploader(
                "Optional guess spectrum to compare",
                type="txt",
                help="This spectrum is plotted only for comparison; GRAVEL still starts from the ANN-based guess.",
            )
            comparison_is_letargic = st.checkbox(
                "Comparison guess per unit lethargy (in dΦ/dE*E)",
                value=False,
            )
        else:
            guess_file = None
            is_letargic = False
            st.caption("No initial-guess file is needed for this mode.")

    

# --------------------- AVVIO ---------------------
#run_button = st.button("Run Unfolding")

# -- Assicura che lo stato sia inizializzato
if 'load_matrices_clicked' not in st.session_state:
    st.session_state.load_matrices_clicked = False

if st.sidebar.button("🚀 Load Data"):
    st.session_state.load_matrices_clicked = True

# Ora, solo se i file sono caricati
guess_input_ready = (
    (initial_guess_type == "From file" and guess_file is not None)
    or (initial_guess_type in ("Neural network", "ANN + MC shaking") and ann_uploaded_model is not None)
    or initial_guess_type == "Constant"
)
if st.session_state.load_matrices_clicked and response_file and energy_file and counts_file and guess_input_ready:
    
    response_tab, fit_tab, spectra_tab, mc_tab = st.tabs(["Response", "Fit", "Spectra", "ANN + MC"])
    
    # Rileva se uno dei file è cambiato
    file_changed = (
        'last_response_file' not in st.session_state or response_file != st.session_state.last_response_file or
        'last_counts_file' not in st.session_state or counts_file != st.session_state.last_counts_file or
        'last_energy_file' not in st.session_state or energy_file != st.session_state.last_energy_file or
        'last_guess_file' not in st.session_state or guess_file != st.session_state.last_guess_file or
        'last_comparison_guess_file' not in st.session_state or comparison_guess_file != st.session_state.last_comparison_guess_file or
        'last_comparison_is_letargic' not in st.session_state or comparison_is_letargic != st.session_state.last_comparison_is_letargic or
        'last_initial_guess_type' not in st.session_state or initial_guess_type != st.session_state.last_initial_guess_type or
        'last_ann_model_path' not in st.session_state or selected_ann_model_key != st.session_state.last_ann_model_path or
        'last_mc_n_samples' not in st.session_state or mc_n_samples != st.session_state.last_mc_n_samples or
        'last_mc_sigma_scale' not in st.session_state or mc_sigma_scale != st.session_state.last_mc_sigma_scale or
        'last_mc_chi_sigma_scale' not in st.session_state or mc_chi_sigma_scale != st.session_state.last_mc_chi_sigma_scale or
        'last_mc_seed' not in st.session_state or mc_seed != st.session_state.last_mc_seed or
        'last_mc_plot_top' not in st.session_state or mc_plot_top != st.session_state.last_mc_plot_top or
        'last_mc_uncertainty_is_relative' not in st.session_state or mc_uncertainty_is_relative != st.session_state.last_mc_uncertainty_is_relative
    )

    if file_changed:
        with response_tab:
            R, data = response_matrix(response_file, counts_file, energy_file, response_tab)
        st.session_state.R = R
        st.session_state.data = data
        st.session_state.last_response_file = response_file
        st.session_state.last_counts_file = counts_file
        st.session_state.last_energy_file = energy_file
        st.session_state.last_guess_file = guess_file
        st.session_state.last_comparison_guess_file = comparison_guess_file
        st.session_state.last_comparison_is_letargic = comparison_is_letargic
        st.session_state.last_initial_guess_type = initial_guess_type
        st.session_state.last_ann_model_path = selected_ann_model_key
        st.session_state.last_mc_n_samples = mc_n_samples
        st.session_state.last_mc_sigma_scale = mc_sigma_scale
        st.session_state.last_mc_chi_sigma_scale = mc_chi_sigma_scale
        st.session_state.last_mc_seed = mc_seed
        st.session_state.last_mc_plot_top = mc_plot_top
        st.session_state.last_mc_uncertainty_is_relative = mc_uncertainty_is_relative
        st.session_state.pop("unfolding_outputs", None)

    R = st.session_state.R
    data = st.session_state.data

    num_detectors = R.shape[0]
    detectors_list = list(range(num_detectors))

    # -- Gestione checkbox: salva stato
    if 'detector_states' not in st.session_state:
        st.session_state.detector_states = {i: True for i in detectors_list}

    response_tab.subheader("Select/Deselect Response Functions")

    detector_cols = response_tab.columns(4)
    for pos, i in enumerate(detectors_list):
        with detector_cols[pos % 4]:
            st.session_state.detector_states[i] = st.checkbox(
                f"Detector #{i}",
                value=st.session_state.detector_states[i],
                key=f"detector_{i}"
            )

    selected_detectors = [i for i in detectors_list if st.session_state.detector_states[i]]

    if not selected_detectors:
        response_tab.warning("You need to select at least one response function.")
        st.stop()

    # --- Anteprima grafica delle funzioni selezionate
    #st.subheader("👀 Preview of the selected response function:")
    response_tab.subheader("Selected response preview")
    fig_preview, ax_preview = plt.subplots(figsize=(7, 4), layout='constrained')

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    E_new = energies[:, 2]
    E_bin_sx = energies[:,0] # bin sx
    E_bin_dx = energies[:,1] # bin dx
    dE = E_bin_dx-E_bin_sx

    for idx in selected_detectors:
        ax_preview.plot(E_new, R[idx, :], marker='o', linestyle='-', label=f"Det {idx}")

    ax_preview.set_xscale("log")
    ax_preview.set_xlabel("Energy [MeV]")
    ax_preview.set_ylabel("Response")
    ax_preview.legend(loc='best',fontsize='small')
    ax_preview.grid(True, which="both", linestyle="--", alpha=0.35)
    response_tab.pyplot(fig_preview, use_container_width=True)

   # Flag di stato per mostrare i grafici solo dopo il click su "Run Unfolding"
    if 'unfolding_done' not in st.session_state:
        st.session_state.unfolding_done = False

    # --- Secondo bottone: Run unfolding
    if st.sidebar.button("Run Unfolding"):
        # Filtro matrice e dati
        data_for_guess = data.copy()
        R_for_mc = R.copy()
        R = R[selected_detectors, :]
        data = data[selected_detectors,:]
        
    
        energy_file.seek(0)
        energies = np.loadtxt(energy_file, delimiter='\t')
        xbins = energies[:, 2]  # bin centrali
        figInt = None
        ann_info = None
        mc_info = None
        comparison_guess = None
        comparison_rebin_fig = None
    
        if initial_guess_type == "From file":
            xguess, figInt = load_guess_spectrum(
                guess_file,
                is_letargic,
                energy_file,
                xbins,
                R.shape[1],
            )
        elif initial_guess_type == "Neural network":
            if not selected_ann_model_path:
                st.error("Select an ANN checkpoint before running the neural-network guess.")
                st.stop()

            try:
                ann_guess = predict_absolute_guess(data_for_guess[:, 0], selected_ann_model_path)
            except Exception as exc:
                st.error(f"Neural-network guess failed: {exc}")
                st.stop()

            xbins_guess = ann_guess["energies_mev"]
            xguess_raw = ann_guess["spectrum_per_mev"]
            if len(xguess_raw) != R.shape[1] or not np.allclose(xbins_guess, xbins, rtol=1e-3, atol=0.0):
                xbins, xguess, figInt = rebin(xbins_guess, xguess_raw, energy_file)
            else:
                xguess = xguess_raw
            ann_info = {
                "model": Path(ann_guess["model_path"]).name,
                "fluence_total": ann_guess["fluence_total"],
                "device": ann_guess["device"],
            }
            if comparison_guess_file is not None:
                comparison_guess, comparison_rebin_fig = load_guess_spectrum(
                    comparison_guess_file,
                    comparison_is_letargic,
                    energy_file,
                    xbins,
                    R.shape[1],
                )
        elif initial_guess_type == "ANN + MC shaking":
            if not selected_ann_model_path:
                st.error("Upload an ANN checkpoint before running ANN + MC shaking.")
                st.stop()

            try:
                mc_result = run_mc_ann_refold(
                    counts_data=data_for_guess,
                    model_source=selected_ann_model_path,
                    response_matrix=R_for_mc,
                    energy_bins=energies,
                    selected_detectors=selected_detectors,
                    n_samples=int(mc_n_samples),
                    sigma_scale=float(mc_sigma_scale),
                    chi_sigma_scale=float(mc_chi_sigma_scale),
                    seed=int(mc_seed),
                    uncertainty_is_relative=mc_uncertainty_is_relative,
                    plot_top=int(mc_plot_top),
                )
            except Exception as exc:
                st.error(f"ANN + MC shaking failed: {exc}")
                st.stop()

            xguess = mc_result["best_spectrum_per_mev"]
            ann_info = {
                "model": mc_result["model"],
                "fluence_total": mc_result["ann_fluence_total"],
                "device": mc_result["device"],
            }
            mc_info = {
                "best_index": mc_result["best_index"],
                "best_chi2": mc_result["best_chi2"],
                "median_chi2": mc_result["median_chi2"],
                "p10_chi2": mc_result["p10_chi2"],
                "p90_chi2": mc_result["p90_chi2"],
                "n_samples": int(mc_n_samples),
                "sigma_scale": float(mc_sigma_scale),
                "chi_sigma_scale": float(mc_chi_sigma_scale),
                "seed": int(mc_seed),
                "plot_top": int(mc_plot_top),
                "uncertainty_is_relative": mc_uncertainty_is_relative,
                "fig_summary": mc_result["fig_summary"],
            }
            if comparison_guess_file is not None:
                comparison_guess, comparison_rebin_fig = load_guess_spectrum(
                    comparison_guess_file,
                    comparison_is_letargic,
                    energy_file,
                    xbins,
                    R.shape[1],
                )
        else:
            xguess = None

        m = R.shape[1]
        x_const = np.zeros((m,))
        mask = (xbins > mmin) & (xbins < mmax)
        x_const[mask] = 1 / xbins[mask]
    
    
        if initial_guess_type == "Constant":
            xguess = x_const
            suffix = "constant"
        elif initial_guess_type == "Neural network":
            suffix = "ann"
        elif initial_guess_type == "ANN + MC shaking":
            suffix = "ann_mc"
        else:
            suffix = "guess"

        if unfolding_type == "Gravel":
            xg, errorg, figC, logIter = gravel(R, data, xguess.copy(), tol, energy_file,max_iter)
        else:
            xg, errorg, figC, logIter = mlem(R, data, xguess.copy(), tol, energy_file,max_iter)
        
        #d_col1.pyplot(figC)
        # Normalizzazione
        #xguess /= np.sum(xguess)
        #xg /= np.sum(xg)
        
        # --- Normalizzazione sugli integrali in energia ∫ x(E) dE = 1
        energy_file.seek(0)
        energies = np.loadtxt(energy_file, delimiter='\t')
        E_bin_sx = energies[:, 0]
        E_bin_dx = energies[:, 1]
        dE = E_bin_dx - E_bin_sx

        # integrali (area fisica)
        integral_fluence_guess = np.sum(xguess * dE)
        integral_fluence_unf = np.sum(xg * dE)
        integral_fluence_comparison = (
            np.sum(comparison_guess * dE) if comparison_guess is not None else None
        )

        # evita divisioni per zero
        xguess_norm = xguess
        xg_norm = xg

        # --- Plot risultati
        fig1, ax1 = plt.subplots(figsize=(6, 4), layout='constrained')
        ax1.step(xbins, xguess_norm * xbins,where='mid',color='blue', label="Guess Spectrum")
        ax1.step(xbins, xg_norm * xbins,where='mid',color='red', label=f'{unfolding_type}')
        ax1.set_xscale("log")
        ax1.set_xlabel("Neutron Energy (MeV)")
        ax1.set_ylabel("Fluence per unit lethargy (dΦ/dE*E) [cm-2 s-1]")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend()
        
        # --- Secondo plot: x log, y log
        fig2, ax2 = plt.subplots(figsize=(6, 4), layout='constrained')
        ax2.step(xbins, xguess_norm * xbins, where='mid', color='blue', label="Guess Spectrum")
        ax2.step(xbins, xg_norm * xbins, where='mid', color='red', label=f'{unfolding_type}')
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_ylim(log_plot_ymin, log_plot_ymax)
        ax2.set_xlabel("Neutron Energy (MeV)")
        ax2.set_ylabel("Fluence per unit lethargy (dΦ/dE*E) [cm-2 s-1]")
        ax2.grid(True, which="both", ls="--", alpha=0.5)
        ax2.legend()
        
        figJ, axJ = plt.subplots(figsize=(6, 4), layout='constrained')
        axJ.plot(range(1, len(errorg) + 1), errorg, marker='o')
        axJ.set_xlabel("Iteration")
        axJ.set_ylabel("Reduced chi-squared J")
        axJ.set_title("Reduced chi-squared convergence")
        axJ.grid(True, which='both', linestyle='--', alpha=0.5)
        axJ.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axJ.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axJ.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
        
        # Segna che abbiamo fatto il run
        st.session_state.unfolding_done = True
        st.session_state.unfolding_outputs = {
            "algorithm": unfolding_type,
            "fig_counts": figC,
            "fig_rebin": figInt if 'figInt' in locals() else None,
            "fig_comparison_rebin": comparison_rebin_fig,
            "fig_chi": figJ,
            "xbins": xbins,
            "xguess": xguess_norm,
            "comparison_guess": comparison_guess,
            "xg": xg,
            "log": logIter,
            "integral_guess": integral_fluence_guess,
            "integral_comparison": integral_fluence_comparison,
            "integral_unfolded": integral_fluence_unf,
            "guess_source": suffix,
            "ann_info": ann_info,
            "mc_info": mc_info,
        }
        
    if 'unfolding_outputs' in st.session_state:
        outputs = st.session_state.unfolding_outputs

        fit_tab.subheader("Fit quality")
        fit_col1, fit_col2 = fit_tab.columns(2)
        fit_col1.pyplot(outputs["fig_counts"], use_container_width=True)
        fit_col2.pyplot(outputs["fig_chi"], use_container_width=True)

        with fit_tab.expander("Iteration log", expanded=True):
            st.text_area(
                f"Output {outputs['algorithm']}",
                value=outputs["log"],
                height=300,
                disabled=True
            )

        spectra_tab.subheader("Spectrum results")
        flu_col1, flu_col2, flu_col3 = spectra_tab.columns(3)
        flu_col1.metric("Integral fluence - Guess [cm-2 s-1]", f"{outputs['integral_guess']:.4e}")
        flu_col2.metric(
            f"Integral fluence - {outputs['algorithm']} [cm-2 s-1]",
            f"{outputs['integral_unfolded']:.4e}"
        )
        if outputs.get("integral_comparison") is not None:
            flu_col3.metric(
                "Integral fluence - Uploaded guess [cm-2 s-1]",
                f"{outputs['integral_comparison']:.4e}",
            )
        if outputs.get("ann_info"):
            ann_info = outputs["ann_info"]
            spectra_tab.caption(
                "ANN initial guess: "
                f"{ann_info['model']} on {ann_info['device']} "
                f"(Phi_tot ANN = {ann_info['fluence_total']:.4e} cm-2 s-1)"
            )
        if outputs.get("mc_info"):
            mc_info = outputs["mc_info"]
            spectra_tab.caption(
                "ANN + MC shaking selected best sample "
                f"#{mc_info['best_index']} with refold J = {mc_info['best_chi2']:.4e}."
            )

        if outputs["fig_rebin"] is not None:
            with spectra_tab.expander("Rebinning preview"):
                st.pyplot(outputs["fig_rebin"], use_container_width=True)
        if outputs.get("fig_comparison_rebin") is not None:
            with spectra_tab.expander("Uploaded guess rebinning preview"):
                st.pyplot(outputs["fig_comparison_rebin"], use_container_width=True)

        fig_linear, ax_linear = plt.subplots(figsize=(7, 4), layout='constrained')
        if outputs.get("mc_info"):
            guess_label = "ANN + MC Best Guess Spectrum"
        elif outputs.get("ann_info"):
            guess_label = "ANN Guess Spectrum"
        else:
            guess_label = "Guess Spectrum"
        ax_linear.step(outputs["xbins"], outputs["xguess"] * outputs["xbins"], where='mid', color='blue', label=guess_label)
        if outputs.get("comparison_guess") is not None:
            ax_linear.step(
                outputs["xbins"],
                outputs["comparison_guess"] * outputs["xbins"],
                where='mid',
                color='green',
                linestyle='--',
                label="Uploaded Guess Spectrum",
            )
        ax_linear.step(outputs["xbins"], outputs["xg"] * outputs["xbins"], where='mid', color='red', label=outputs["algorithm"])
        ax_linear.set_xscale("log")
        ax_linear.set_xlabel("Neutron Energy (MeV)")
        ax_linear.set_ylabel("Fluence per unit lethargy (dÎ¦/dE*E) [cm-2 s-1]")
        ax_linear.grid(True, which="both", ls="--", alpha=0.35)
        ax_linear.legend()

        fig_log, ax_log = plt.subplots(figsize=(7, 4), layout='constrained')
        ax_log.step(outputs["xbins"], outputs["xguess"] * outputs["xbins"], where='mid', color='blue', label=guess_label)
        if outputs.get("comparison_guess") is not None:
            ax_log.step(
                outputs["xbins"],
                outputs["comparison_guess"] * outputs["xbins"],
                where='mid',
                color='green',
                linestyle='--',
                label="Uploaded Guess Spectrum",
            )
        ax_log.step(outputs["xbins"], outputs["xg"] * outputs["xbins"], where='mid', color='red', label=outputs["algorithm"])
        ax_log.set_xscale("log")
        ax_log.set_yscale("log")
        ax_log.set_ylim(log_plot_ymin, log_plot_ymax)
        ax_log.set_xlabel("Neutron Energy (MeV)")
        ax_log.set_ylabel("Fluence per unit lethargy (dÎ¦/dE*E) [cm-2 s-1]")
        ax_log.grid(True, which="both", ls="--", alpha=0.35)
        ax_log.legend()

        spectrum_col1, spectrum_col2 = spectra_tab.columns(2)
        spectrum_col1.pyplot(fig_linear, use_container_width=True)
        spectrum_col2.pyplot(fig_log, use_container_width=True)

        spectra_tab.markdown("### Results Download")
        csv_out = np.column_stack((outputs["xbins"], outputs["xg"]))
        csv_str = io.StringIO()
        np.savetxt(csv_str, csv_out, delimiter='\t', header='Energy (MeV)\tUnfolded spectrum (dPhi/dE)', comments='')
        spectra_tab.download_button("Download unfolded spectrum", csv_str.getvalue(), file_name="unfolded_spectrum.txt")

        mc_tab.subheader("ANN + MC shaking")
        if outputs.get("mc_info"):
            mc_info = outputs["mc_info"]
            metric_cols = mc_tab.columns(4)
            metric_cols[0].metric("Best sample", mc_info["best_index"])
            metric_cols[1].metric("Best refold J", f"{mc_info['best_chi2']:.4e}")
            metric_cols[2].metric("Median refold J", f"{mc_info['median_chi2']:.4e}")
            metric_cols[3].metric("MC samples", mc_info["n_samples"])
            mc_tab.caption(
                f"J p10/p90 = {mc_info['p10_chi2']:.4e} / {mc_info['p90_chi2']:.4e}; "
                f"shake scale = {mc_info['sigma_scale']:.2f}; "
                f"chi sigma scale = {mc_info['chi_sigma_scale']:.2f}; "
                f"shown spectra = {mc_info['plot_top']}; seed = {mc_info['seed']}."
            )
            mc_tab.pyplot(mc_info["fig_summary"], use_container_width=True)
        else:
            mc_tab.info("Select ANN + MC shaking and run unfolding to show MC refold diagnostics.")
    else:
        fit_tab.info("Run unfolding to show fit quality and iteration log.")
        spectra_tab.info("Run unfolding to show spectrum results.")
        mc_tab.info("Run ANN + MC shaking to show Monte Carlo refold diagnostics.")


else:
    st.info("Upload all files and click on 'Load Data' in the sidebar to start.")
