import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gravel import gravel
from mlem import mlem
from rebin import rebin
from response_matrix import response_matrix
import matplotlib.ticker as ticker
import io

# --------------------- CONFIGURAZIONE ---------------------
st.set_page_config(layout="wide")
st.title("GRAVELZS Neutron Spectrum UnfoldingZS")
#st.title("figghesti")

# --------------------- SIDEBAR (Controlli) ---------------------
st.sidebar.header("⚙️ Parametri di Unfolding")
initial_guess_type = st.sidebar.selectbox("Initial Guess Spectrum:", ["Constant", "From file"])
unfolding_type = st.sidebar.selectbox("Unfolding algorithm:", ["Gravel", "MLEM"])
tol = st.sidebar.number_input(
    "🔍 Chi-squared value to stop iterations", 
    min_value=1e-12, 
    max_value=1.0, 
    value=1e-1, 
    step=1e-9, 
    format="%.1e"
)

# --------------------- FILE UPLOAD ---------------------
st.markdown("Upload datas: Response Matrix, Measured Counts, Energy bins of the repsonse functions, Initial Guess Spectrum")

# File uploader
with st.container ():
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        response_file = st.file_uploader("📁 Response matrix (TXT, Response Function for each column, tab-separated)", type="txt")
        counts_file = st.file_uploader("📈 Measured counts, in column (TXT)", type="txt")

    with col_u2:
        energy_file = st.file_uploader("⚡ Energy bins (MeV), 1st col: left boundary of the energy bin, 2nd col: right boundary of the energy bin, 3rd col: central energy (TXT)", type="txt")
        guess_file = st.file_uploader("🧠 Initial guess spectrum, 1st col: energy in MeV, 2nd col: differential spectrum in energy dPhi/dE  (TXT)", type="txt")


    

# --------------------- AVVIO ---------------------
#run_button = st.button("Run Unfolding")

# -- Assicura che lo stato sia inizializzato
if 'load_matrices_clicked' not in st.session_state:
    st.session_state.load_matrices_clicked = False

if st.button("🚀 Load Data"):
    st.session_state.load_matrices_clicked = True

# Ora, solo se i file sono caricati
if st.session_state.load_matrices_clicked and response_file and energy_file and counts_file and guess_file:
    
    col1, col2 = st.columns(2)
    d_col1, d_col2 = st.columns(2)
    
    # Solo se non già caricate
    if 'R' not in st.session_state:
        # Carica e salva in session_state
        R, data = response_matrix(response_file, counts_file, energy_file,col1)
        st.session_state.R = R
        st.session_state.data = data

    R = st.session_state.R
    data = st.session_state.data

    num_detectors = R.shape[0]
    detectors_list = list(range(num_detectors))

    # -- Gestione checkbox: salva stato
    if 'detector_states' not in st.session_state:
        st.session_state.detector_states = {i: True for i in detectors_list}

    st.subheader("📦 Select/Deselect Response Functions:")

    for i in detectors_list:
        st.session_state.detector_states[i] = st.checkbox(
            f"Detector #{i}",
            value=st.session_state.detector_states[i],
            key=f"detector_{i}"
        )

    selected_detectors = [i for i in detectors_list if st.session_state.detector_states[i]]

    if not selected_detectors:
        st.warning("⚠️You need to select at least one response function.")
        st.stop()

    # --- Anteprima grafica delle funzioni selezionate
    #st.subheader("👀 Preview of the selected response function:")
    fig_preview, ax_preview = plt.subplots(figsize=(6, 4), layout='constrained')

    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    E_new = energies[:, 2]
    E_bin_sx = energies[:,0] # bin sx
    E_bin_dx = energies[:,1] # bin dx
    dE = E_bin_dx-E_bin_sx

    for idx in selected_detectors:
        ax_preview.plot(E_new, R[idx, :], marker='o', linestyle='-', label=f"Detector {idx}")

    ax_preview.set_xscale("log")
    ax_preview.set_xlabel("Energy [MeV]")
    ax_preview.set_ylabel("Response (a.u.)")
    ax_preview.legend()
    ax_preview.grid(True, which="both", linestyle="--", alpha=0.5)
    d_col1.pyplot(fig_preview)

   # Flag di stato per mostrare i grafici solo dopo il click su "Run Unfolding"
    if 'unfolding_done' not in st.session_state:
        st.session_state.unfolding_done = False

    # --- Secondo bottone: Run unfolding
    if st.sidebar.button("Run Unfolding"):
        # Filtro matrice e dati
        R = R[selected_detectors, :]
        data = data[selected_detectors]
        
    
        energy_file.seek(0)
        energies = np.loadtxt(energy_file, delimiter='\t')
        xbins = energies[:, 2]  # bin centrali
    
        guess_file.seek(0)
        guess_spect = np.loadtxt(guess_file, delimiter='\t')
    
        xbins_guess = guess_spect[:, 0]
        xguess_raw = guess_spect[:, 1]

        if len(xguess_raw) != R.shape[1]:
            xbins, xguess, figInt = rebin(xbins_guess, xguess_raw,energy_file)
            #d_col2.pyplot(figInt)
        else:
            xguess = xguess_raw

        m = R.shape[1]
        x_const = np.zeros((m,))
        mask = (xbins > 1e-8) & (xbins < 1)
        x_const[mask] = 1 / xbins[mask]
    
    
        if initial_guess_type == "Constant":
            xguess = x_const
            suffix = "constant"
        else:
            suffix = "guess"

        if unfolding_type == "Gravel":
            xg, errorg, figC, logIter = gravel(R, data, xguess.copy(), tol, energy_file)
        else:
            xg, errorg, figC, logIter = mlem(R, data, xguess.copy(), tol, energy_file)
        
        #d_col1.pyplot(figC)
        # Normalizzazione
        xguess /= np.sum(xguess)
        xg /= np.sum(xg)

        # --- Plot risultati
        fig1, ax1 = plt.subplots(figsize=(6, 4), layout='constrained')
        ax1.step(xbins, xguess * xbins,where='mid',color='blue', label="Guess Spectrum")
        ax1.step(xbins, xg * xbins,where='mid',color='red', label="GRAVEL")
        ax1.set_xscale("log")
        ax1.set_xlabel("Neutron Energy (MeV)")
        ax1.set_ylabel("Normalized Counts")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend()
        
        figJ, axJ = plt.subplots(figsize=(6, 4), layout='constrained')
        axJ.plot(range(1, len(errorg) + 1), errorg, marker='o')
        axJ.set_xlabel("Iteration")
        axJ.set_ylabel("Chi-squared J")
        axJ.set_title("Chi-squared convergence")
        axJ.grid(True, which='both', linestyle='--', alpha=0.5)
        axJ.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axJ.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axJ.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
        
        # Segna che abbiamo fatto il run
        st.session_state.unfolding_done = True
        
        if st.session_state.unfolding_done:
            
            with st.container():  # 👈 questo fissa la posizione
                d_col1.pyplot(figC)
                d_col2.pyplot(figInt)
                d_col2.pyplot(fig1)
                    
                with st.sidebar.expander("📘 Iteration log"):
                    st.text_area("Output GRAVEL", logIter, height=300,key="log_iter_output")
                    st.pyplot(figJ)

        #d_col2.pyplot(fig1)
        # --- DOWNLOAD
        st.sidebar.markdown("### 📦 Download Risultati")
        csv_out = np.column_stack((xbins, xg))
        csv_str = io.StringIO()
        np.savetxt(csv_str, csv_out, delimiter='\t', header='Energy (MeV)\tUnfolded spectrum (norm)', comments='')
        st.sidebar.download_button("📥 Scarica spettro unfolding", csv_str.getvalue(), file_name="unfolded_spectrum.txt")


else:
    st.info("Upload all files and click on 'Run Unfolding' to start.")