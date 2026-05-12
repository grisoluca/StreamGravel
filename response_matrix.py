import numpy as np
import streamlit as st

def response_matrix(response_file,counts_file,energy_file,col):
    #st.write("✅ Caricamento della matrice di risposta...")
    R = np.loadtxt(response_file, delimiter='\t')

    #st.write("✅ Trasposizione della matrice...")
    R = R.T
    #funz = [0,1,2,3,4,5,6,7,8,9]
    #stop = len(funz)
    
    #st.write("✅ Selezione dei detector:", funz)
    #R = R[funz, :]
    
    #st.write("✅ Caricamento dei conteggi...")
    data_full = np.loadtxt(counts_file, delimiter='\t')
    
    if data_full.ndim == 1:
        uncertainties = np.ones((data_full.shape[0],1))
        data_full = data_full.reshape(-1, 1)  # Converti in colonna
        data_full = np.hstack((data_full, uncertainties))
    
    data = data_full#[funz,:]
    
    #st.write("✅ Caricamento delle energie...")
    energy_file.seek(0)
    energies = np.loadtxt(energy_file,delimiter='\t')
    
    
    if energies.shape[1] < 3:
        st.error("❌ Il file delle energie deve avere almeno 3 colonne. Controlla che sia nel formato giusto.")
        return None, None
    
    




    st.success("✅ Response matrix and datas succesfully uploaded.")
    
    return R,data
    
