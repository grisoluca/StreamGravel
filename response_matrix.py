import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def response_matrix(response_file,counts_file,energy_file):
    #st.write("✅ Caricamento della matrice di risposta...")
    R = np.loadtxt(response_file, delimiter='\t')

    #st.write("✅ Trasposizione della matrice...")
    R = R.T
    funz = [0,1,2,3,4,5,6,7,8,9]
    stop = len(funz)
    
    #st.write("✅ Selezione dei detector:", funz)
    R = R[funz, :]
    
    #st.write("✅ Caricamento dei conteggi...")
    data_full = np.loadtxt(counts_file)
    data = data_full[funz]
    
    #st.write("✅ Caricamento delle energie...")
    energies = np.loadtxt(energy_file,delimiter='\t')
    
    if energies.shape[1] < 3:
        st.error("❌ Il file delle energie deve avere almeno 3 colonne. Controlla che sia nel formato giusto.")
        return None, None
    
    E_new = energies[:, 2]  # bin centrali
    
    #st.write("✅ Plotting...")
    for i in range(stop):
        plt.plot(E_new, R[i, :], marker='o', linestyle='-', label=f"Scint {i+1}")
        
        plt.xscale("log")
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Detector response per unit fluence [optph cm2]")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    
    plt.axvline(x=0.5e-6, color='r', linestyle='--', label="0.5")   
    st.pyplot(plt.gcf())  # Mostra il grafico nella webapp

    st.success("✅ Response matrix and datas succesfully uploaded.")
    
    return R,data
    
