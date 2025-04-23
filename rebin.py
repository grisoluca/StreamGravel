import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import streamlit as st

def rebin(E_old,S_old,energy_file):
    
    # Interpolazione
    interp_func = interp1d(E_old, S_old, kind='linear', fill_value=0.0, bounds_error=False)
    
    energy_file.seek(0)
    energies = np.loadtxt(energy_file, delimiter='\t')
    #E_bin_sx = energies[:,0] # bin sx
    #E_bin_dx = energies[:,1] # bin dx
    E_new = energies[:,2] # bin centrale
    
    S_new = interp_func(E_new)
    
    E_interp = np.logspace(np.log10(E_new.min()), np.log10(E_new.max()), 1000)
    S_interp = interp_func(E_interp)
    
    # Plot per confronto visivo
    plt.semilogx(E_old, S_old*E_old, 'o', label='Originale (35)')
    plt.semilogx(E_new, S_new*E_new, 'x-', label='Interpolato (60)')
    plt.semilogx(E_interp, S_interp*E_interp, '-', label='funz con')
    plt.xlabel('Energia')
    plt.ylabel('Spettro')
    plt.legend()
    plt.grid(True)
    plt.title('Rebinning dello spettro')
    st.pyplot(plt.gcf())

    return(E_new,S_new)
