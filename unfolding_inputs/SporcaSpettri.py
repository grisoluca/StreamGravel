import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Carica dati
data = np.loadtxt("R_75/nagoyaspect.txt")
energies = data[:, 0]
spectrum = data[:, 1]

# Modifica: sfocatura + taglio >1MeV
#spectrum_mod = gaussian_filter1d(spectrum, sigma=1.5)
#spectrum_mod[energies > 1e-2] *= 0.6

# Aggiungi rumore random (±5%)
noise_level = 0.25  # puoi aumentare o ridurre l’intensità
spectrum_mod = spectrum * (1 + noise_level * np.random.randn(len(spectrum)))

# Plot
plt.figure(figsize=(10,5))
plt.semilogx(energies, spectrum*energies, label="Original")
plt.semilogx(energies, spectrum_mod*energies, label="Modified", linestyle='--')
plt.xlabel("Energy [MeV]")
plt.ylabel("Fluence")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()

np.savetxt("R_100/nagoya_MOD.txt", 
           np.column_stack((energies, spectrum_mod)), 
           fmt="%.6e", delimiter='\t')