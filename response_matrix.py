import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def response_matrix(response_file,counts_file,energy_file,col):
    #st.write("✅ Caricamento della matrice di risposta...")
    R = np.loadtxt(response_file, delimiter='\t')

    #st.write("✅ Trasposizione della matrice...")
    R = R.T
    funz = [0,1,2,3,4,5,6,7,8,9]
    stop = len(funz)
    
    #st.write("✅ Selezione dei detector:", funz)
    R = R[funz, :]
    
    #st.write("✅ Caricamento dei conteggi...")
    data_full = np.loadtxt(counts_file, delimiter='\t')
    
    if data_full.ndim == 1:
        uncertainties = np.ones((data_full.shape[0],1))
        data_full = data_full.reshape(-1, 1)  # Converti in colonna
        data_full = np.hstack((data_full, uncertainties))
    
    data = data_full[funz,:]
    
    #st.write("✅ Caricamento delle energie...")
    energy_file.seek(0)
    energies = np.loadtxt(energy_file,delimiter='\t')
    
    
    if energies.shape[1] < 3:
        st.error("❌ Il file delle energie deve avere almeno 3 colonne. Controlla che sia nel formato giusto.")
        return None, None
    
    E_new = energies[:, 2]  # bin centrali
    
    #st.write("✅ Plotting...")
    figRM, axRM = plt.subplots(figsize=(6, 4), layout='constrained')  # puoi cambiare le dimensioni se vuoi

    for i in range(stop):
        axRM.plot(E_new, R[i, :], marker='o', linestyle='-', label=f"Scint {i+1}")

    axRM.set_xscale("log")
    axRM.set_xlabel("Energy [MeV]")
    axRM.set_ylabel("Detector response per unit fluence [optph cm2]")
    axRM.legend()
    axRM.grid(True, which="both", linestyle="--", linewidth=0.5)
    axRM.axvline(x=0.5e-6, color='r', linestyle='--', label="0.5")

    col.pyplot(figRM)  # Mostra la figura nella colonna specificata

    st.success("✅ Response matrix and datas succesfully uploaded.")
    
    return R,data
    
