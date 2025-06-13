import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

# Lade die Datei
data = np.load("Phasediagram_mu_without_V_no_mask2.npz")

# special chemical potential
c_spec = 0.017 * (1/2**11)

# Zugriff auf einzelne Variablen:
p_result = data['p']
mu_result = data['mu']
Magnetisation_per_N = np.real(data['Magnetisation_per_N']).reshape(200, 200)
N_par = data['N_par']
print(Magnetisation_per_N)
Magnetisation_per_N.T[np.round(N_par.T[0], 10) < 1] = 0

p_grid = p_result.T
mu_grid = mu_result.T[0]
mu_grid = np.linspace(0.5 * c_spec, 1.5 * c_spec, 200)

# calculate meshgrid
p_mesh, mu_mesh = np.meshgrid(p_grid/c_spec, mu_grid/c_spec)

# Define levels for more color bins (e.g., 20 levels from 0 to 1)
bounds = np.linspace(-0.0001, 1.025, 41)  # 40 color bins

# Colormap 1
h = plt.pcolormesh(mu_mesh, p_mesh, Magnetisation_per_N, cmap="viridis")
#plt.scatter(mu_mesh, p_mesh, label = "grid")
cbar = plt.colorbar(h, ticks=[-1, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 1.])
cbar.set_label("Magnetisation per N")
plt.title("Magnetisation per Particle")
plt.ylabel('p')
plt.xlabel(r'$\mu$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.show()