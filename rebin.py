import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def rebin(E_old,S_old):
    
    # Interpolazione
    interp_func = interp1d(E_old, S_old, kind='linear', fill_value=0.0, bounds_error=False)
    
    dir = "unfolding_inputs/"
    energies = np.loadtxt(dir+"response_M_energies.txt")
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
    plt.show()

    return(E_new,S_new)
