import numpy as np
import matplotlib.pyplot as plt


def response_matrix(response_file,counts_file,energy_file):
        

    R = np.loadtxt(response_file)
    # R.shape sarà (60, 10)
    R = R.T #ora R.shape è (10,60)
    funz = [0,1,2,3,4,5,6,7,8,9]
    stop = len(funz)
    R = R[funz,:]
    
    data_full = np.loadtxt(counts_file)
    data = data_full[funz]
    
    energies = np.loadtxt(energy_file)
    E_new = energies[:,2] # bin centrale
    
    for i in range(stop):
        plt.plot(E_new, R[i, :], marker='o', linestyle='-', label=f"Scint {i+1}")

    # Settaggi del grafico
    plt.xscale("log")  # Scala logaritmica per l'energia
    plt.xlabel("Energy [MeV]")
    plt.ylabel("Detector response per unit fluence [optph cm2]")
    
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axvline(x=0.5e-6, color='r', linestyle='--', label="0.5")
    # Mostra il grafico
    plt.show()
    
    return R,data
    
