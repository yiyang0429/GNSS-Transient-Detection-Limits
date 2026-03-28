import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})

N_vals = np.logspace(1, 4.5, 100)
area = 400.0
L_vals = [0.0, 0.1, 0.5, 1.0]
colors = ['gray', 'blue', 'red', 'purple']

fig, ax = plt.subplots(figsize=(10, 6))

for L, color in zip(L_vals, colors):
    if L == 0:
        N_eff = N_vals
        label = 'L = 0 km (Ideal White Noise)'
        ax.plot(N_vals, N_eff, color=color, ls='--', lw=2, label=label)
    else:
        N_eff = 1 / ((1 / N_vals) + (np.pi * L**2 / area))
        label = f'L = {L} km (Urban CME)'
        ax.plot(N_vals, N_eff, color=color, lw=2.5, label=label)

L_target = 0.5
N_opt = 1000
N_eff_opt = 1 / ((1 / N_opt) + (np.pi * L_target**2 / area))
ax.plot(N_opt, N_eff_opt, 'ro', markersize=10, zorder=5)
ax.annotate(f'Economic Sweet Spot\nN=1000, N_eff={N_eff_opt:.0f}', 
            xy=(N_opt, N_eff_opt), xytext=(N_opt*1.2, N_eff_opt-200),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5), fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Physical Node Count ($N$)', fontweight='bold')
ax.set_ylabel('Effective Degrees of Freedom ($N_{eff}$)', fontweight='bold')
ax.set_title('Spatial Aliasing and $N_{eff}$ Asymptotes', fontweight='bold') # Removed "Figure X"
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('Fig1_Neff_Asymptote.pdf', dpi=300)
print("Saved Fig1_Neff_Asymptote.pdf")
