import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gravel import gravel
from rebin import rebin
from response_matrix import response_matrix

st.set_page_config(layout="wide")
st.title("GRAVEL Neutron Spectrum Unfolding")
st.title("🚀 Versione nuova!")

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

initial_guess_type = st.selectbox("Initial Guess Spectrum:", ["Constant", "From file"])
    
tol = st.number_input(
    "🔍 Soglia di arresto (tolleranza su ΔJ)", 
    min_value=1e-12, 
    max_value=1.0, 
    value=1e-7, 
    step=1e-9, 
    format="%.1e"
)

#run_button = st.button("Run Unfolding")

load_matrices_button = st.button("Load Response matrix and go to Detector Selection")
#st.write("Tipo response_file:", type(response_file))
#st.write("Tipo counts_file:", type(counts_file))
#st.write("Tipo energy_file:", type(energy_file))
#st.write("Tipo energy_file:", type(guess_file))

if load_matrices_button and response_file and energy_file and counts_file and guess_file:
    
    col1, col2 = st.columns(2)

    # Caricamento dati
    #st.write("✅ Tutti i file sono stati caricati correttamente...")
    R,data = response_matrix(response_file,counts_file,energy_file,col1)
    
    # --- Multiselect dentro un expander
    with st.expander("📦 Response function selection"):
        num_detectors = R.shape[0]
        detectors_list = list(range(num_detectors))

        selected_detectors = st.multiselect(
            "Select the response functions you want to use:",
            detectors_list,
            default=detectors_list
        )

        # --- Preview delle funzioni selezionate
        if selected_detectors:
            st.subheader("👀 Preview of the selected Response functions:")
            fig_preview, ax_preview = plt.subplots(figsize=(7, 5))

            energy_file.seek(0)
            energies = np.loadtxt(energy_file, delimiter='\t')
            E_new = energies[:, 2]

            for idx in selected_detectors:
                ax_preview.plot(E_new, R[idx, :], label=f"Detector {idx}")

            ax_preview.set_xscale("log")
            ax_preview.set_xlabel("Energy [MeV]")
            ax_preview.set_ylabel("Response")
            ax_preview.legend()
            ax_preview.grid(True, which="both", linestyle="--", alpha=0.5)
            st.pyplot(fig_preview)
        else:
            st.info("Select at least one response function to see the preview.")

    # --- Secondo bottone: Run unfolding
    run_button = st.button("Run Unfolding")

    if run_button and selected_detectors:
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
            xbins, xguess = rebin(xbins_guess, xguess_raw,energy_file,col2)
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

        xg, errorg = gravel(R, data, xguess.copy(), tol, energy_file,col1)

        # Normalizzazione
        xguess /= np.sum(xguess)
        xg /= np.sum(xg)

        # --- Plot risultati
        fig1, ax1 = plt.subplots(figsize=(6, 4), layout='constrained')
        ax1.semilogx(xbins, xguess * xbins, label="Guess Spectrum")
        ax1.semilogx(xbins, xg * xbins, label="GRAVEL")
        ax1.set_xlabel("Neutron Energy (MeV)")
        ax1.set_ylabel("Normalized Counts")
        ax1.grid(True, which="both", ls="--", alpha=0.5)
        ax1.legend()
        col2.pyplot(fig1)

    elif run_button and not selected_detectors:
        st.error("❌ Devi selezionare almeno una funzione di risposta prima di eseguire l'unfolding.")
    # --- Plot errore di convergenza
    #fig2, ax2 = plt.subplots()
    #ax2.plot(np.arange(len(errorg)), errorg, color="k", label="GRAVEL")
    #ax2.axhline(tol, color="r", linestyle="--", linewidth=0.75)
    #ax2.set_ylabel("$\\Delta J$")
    #ax2.set_xlabel("Iteration")
    #ax2.grid(True, which="both", ls=":")
    #ax2.legend()
    #st.pyplot(fig2)

    # --- Calcolo errore relativo
    #nonzero_mask = xguess != 0
    #diff_g = abs(xguess[nonzero_mask] - xg[nonzero_mask]) / xguess[nonzero_mask]
    #rel_norm = np.linalg.norm(diff_g)

    #st.metric(label="Normalized Relative error", value=f"{rel_norm:.4f}")
else:
    st.info("Upload all files and click on 'Run Unfolding' to start.")