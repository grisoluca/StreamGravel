import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gravel import gravel
from rebin import rebin

st.set_page_config(layout="wide")
st.title("GRAVEL Neutron Spectrum Unfolding")

st.markdown("Upload i file necessari per l'unfolding:")

# File uploader
col1, col2 = st.columns(2)

with col1:
    response_file = st.file_uploader("📁 Response matrix (TXT, Response Function for each column, tab-separated)", type="txt")
    counts_file = st.file_uploader("📈 Measured counts, in column (TXT)", type="txt")

with col2:
    energy_file = st.file_uploader("⚡ Energy bins (MeV), 1st col: left boundary of the energy bin, 2nd col: right boundary of the energy bin, 3rd col: central energy (TXT)", type="txt")
    guess_file = st.file_uploader("🧠 Initial guess spectrum, 1st col: energy in MeV, 2nd col: differential spectrum in energy dPhi/dE  (TXT)", type="txt")

initial_guess_type = st.selectbox("Tipo di guess iniziale:", ["Costante", "Da file"])
run_button = st.button("Esegui unfolding")

if run_button and response_file and energy_file and counts_file and guess_file:
    # Caricamento dati
    R_raw = np.loadtxt(response_file, delimiter='\t')
    R = R_raw.T
    funz = [0,2,4,6,8]
    R = R[funz, :]

    energies = np.loadtxt(energy_file)
    xbins = energies[:, 2]  # bin centrali

    data_full = np.loadtxt(counts_file)
    data = data_full[funz]

    guess_spect = np.loadtxt(guess_file)
    xbins_guess = guess_spect[:, 0]
    xguess_raw = guess_spect[:, 1]

    if len(xguess_raw) != R.shape[1]:
        xbins, xguess = rebin(xbins_guess, xguess_raw)
    else:
        xguess = xguess_raw

    m = R.shape[1]
    x_const = np.zeros((m,))
    mask = (xbins > 1e-8) & (xbins < 1)
    x_const[mask] = 1 / xbins[mask]

    if initial_guess_type == "Costante":
        tol = 1e-9
        xguess = x_const
        suffix = "constant"
    else:
        tol = 0.8
        suffix = "guess"

    xg, errorg = gravel(R, data, xguess.copy(), tol)

    # Normalizzazione
    xguess /= np.sum(xguess)
    xg /= np.sum(xg)

    # --- Plot risultati
    fig1, ax1 = plt.subplots()
    ax1.semilogx(xbins, xguess * xbins, label="Guess Spectrum")
    ax1.semilogx(xbins, xg * xbins, label="GRAVEL")
    ax1.set_xlabel("Neutron Energy (MeV)")
    ax1.set_ylabel("Normalized Counts")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()
    st.pyplot(fig1)

    # --- Plot errore di convergenza
    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(len(errorg)), errorg, color="k", label="GRAVEL")
    ax2.axhline(tol, color="r", linestyle="--", linewidth=0.75)
    ax2.set_ylabel("$\\Delta J$")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, which="both", ls=":")
    ax2.legend()
    st.pyplot(fig2)

    # --- Calcolo errore relativo
    nonzero_mask = xguess != 0
    diff_g = abs(xguess[nonzero_mask] - xg[nonzero_mask]) / xguess[nonzero_mask]
    rel_norm = np.linalg.norm(diff_g)

    st.metric(label="Errore relativo normalizzato", value=f"{rel_norm:.4f}")
else:
    st.info("Carica tutti i file e premi 'Esegui unfolding' per iniziare.")

