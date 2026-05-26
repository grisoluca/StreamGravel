import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gravel import gravel
from mlem import mlem
from rebin import rebin
from response_matrix import response_matrix
from ann_guess import DEFAULT_ANN_PROJECT_DIR, find_latest_ann_model, predict_absolute_guess
import matplotlib.ticker as ticker
import io

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"
EXAMPLE_LINAC_ZIP = EXAMPLES_DIR / "example-LINAC.zip"

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
    ["From file", "Constant", "Neural network"],
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

ann_model_path = ""
if initial_guess_type == "Neural network":
    st.sidebar.markdown("#### Neural-network guess")
    ann_project_dir = st.sidebar.text_input(
        "ANN project folder",
        value=str(DEFAULT_ANN_PROJECT_DIR),
        help="Folder containing the ANN models/ directory.",
    )
    latest_ann_model = find_latest_ann_model(ann_project_dir)
    default_ann_model = str(latest_ann_model) if latest_ann_model else ""
    ann_model_path = st.sidebar.text_input(
        "ANN checkpoint (.pt)",
        value=default_ann_model,
        help="By default StreamGravel uses the newest absolute ANN checkpoint.",
    )
    if latest_ann_model:
        st.sidebar.caption(f"Latest detected model: {latest_ann_model.name}")
    else:
        st.sidebar.warning("No .pt model found in the selected ANN project folder.")

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
        if initial_guess_type == "From file":
            guess_file = st.file_uploader("🧠 Initial guess spectrum", type="txt")
            is_letargic = st.checkbox("Guess spectrum per unit lethargy (in dΦ/dE*E)", value=False)
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
guess_input_ready = initial_guess_type != "From file" or guess_file is not None
if st.session_state.load_matrices_clicked and response_file and energy_file and counts_file and guess_input_ready:
    
    response_tab, fit_tab, spectra_tab = st.tabs(["Response", "Fit", "Spectra"])
    
    # Rileva se uno dei file è cambiato
    file_changed = (
        'last_response_file' not in st.session_state or response_file != st.session_state.last_response_file or
        'last_counts_file' not in st.session_state or counts_file != st.session_state.last_counts_file or
        'last_energy_file' not in st.session_state or energy_file != st.session_state.last_energy_file or
        'last_guess_file' not in st.session_state or guess_file != st.session_state.last_guess_file or
        'last_initial_guess_type' not in st.session_state or initial_guess_type != st.session_state.last_initial_guess_type or
        'last_ann_model_path' not in st.session_state or ann_model_path != st.session_state.last_ann_model_path
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
        st.session_state.last_initial_guess_type = initial_guess_type
        st.session_state.last_ann_model_path = ann_model_path
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
        R = R[selected_detectors, :]
        data = data[selected_detectors,:]
        
    
        energy_file.seek(0)
        energies = np.loadtxt(energy_file, delimiter='\t')
        xbins = energies[:, 2]  # bin centrali
        figInt = None
        ann_info = None
    
        if initial_guess_type == "From file":
            guess_file.seek(0)
            guess_spect = np.loadtxt(guess_file, delimiter='\t')
        
            xbins_guess = guess_spect[:, 0]
            xguess_raw = guess_spect[:, 1]

            if is_letargic:
                # Conversione da dΦ/dlnE → dΦ/dE
                xguess_raw = xguess_raw / xbins_guess

            if len(xguess_raw) != R.shape[1]:
                xbins, xguess, figInt = rebin(xbins_guess, xguess_raw,energy_file)
                #d_col2.pyplot(figInt)
            else:
                xguess = xguess_raw
        elif initial_guess_type == "Neural network":
            if not ann_model_path:
                st.error("Select an ANN checkpoint before running the neural-network guess.")
                st.stop()

            try:
                ann_guess = predict_absolute_guess(data_for_guess[:, 0], ann_model_path)
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
            "fig_chi": figJ,
            "xbins": xbins,
            "xguess": xguess_norm,
            "xg": xg,
            "log": logIter,
            "integral_guess": integral_fluence_guess,
            "integral_unfolded": integral_fluence_unf,
            "guess_source": suffix,
            "ann_info": ann_info,
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
        flu_col1, flu_col2 = spectra_tab.columns(2)
        flu_col1.metric("Integral fluence - Guess [cm-2 s-1]", f"{outputs['integral_guess']:.4e}")
        flu_col2.metric(
            f"Integral fluence - {outputs['algorithm']} [cm-2 s-1]",
            f"{outputs['integral_unfolded']:.4e}"
        )
        if outputs.get("ann_info"):
            ann_info = outputs["ann_info"]
            spectra_tab.caption(
                "ANN initial guess: "
                f"{ann_info['model']} on {ann_info['device']} "
                f"(Phi_tot ANN = {ann_info['fluence_total']:.4e} cm-2 s-1)"
            )

        if outputs["fig_rebin"] is not None:
            with spectra_tab.expander("Rebinning preview"):
                st.pyplot(outputs["fig_rebin"], use_container_width=True)

        fig_linear, ax_linear = plt.subplots(figsize=(7, 4), layout='constrained')
        ax_linear.step(outputs["xbins"], outputs["xguess"] * outputs["xbins"], where='mid', color='blue', label="Guess Spectrum")
        ax_linear.step(outputs["xbins"], outputs["xg"] * outputs["xbins"], where='mid', color='red', label=outputs["algorithm"])
        ax_linear.set_xscale("log")
        ax_linear.set_xlabel("Neutron Energy (MeV)")
        ax_linear.set_ylabel("Fluence per unit lethargy (dÎ¦/dE*E) [cm-2 s-1]")
        ax_linear.grid(True, which="both", ls="--", alpha=0.35)
        ax_linear.legend()

        fig_log, ax_log = plt.subplots(figsize=(7, 4), layout='constrained')
        ax_log.step(outputs["xbins"], outputs["xguess"] * outputs["xbins"], where='mid', color='blue', label="Guess Spectrum")
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
    else:
        fit_tab.info("Run unfolding to show fit quality and iteration log.")
        spectra_tab.info("Run unfolding to show spectrum results.")


else:
    st.info("Upload all files and click on 'Load Data' in the sidebar to start.")
