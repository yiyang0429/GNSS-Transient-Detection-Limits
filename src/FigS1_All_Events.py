import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BSSA Publication Style
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 10, 
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9
})

# 12 Historical Events Catalog
catalog = [
    {"event": "Tohoku-Oki", "slip_cm": 0.50}, {"event": "Iquique (Edge)", "slip_cm": 0.74},
    {"event": "Sanriku-Oki", "slip_cm": 0.80}, {"event": "Izmit", "slip_cm": 1.00},
    {"event": "Valparaiso", "slip_cm": 1.20}, {"event": "Nicoya", "slip_cm": 1.50},
    {"event": "Kaikoura", "slip_cm": 1.80}, {"event": "Papanoa", "slip_cm": 2.00},
    {"event": "Gisborne", "slip_cm": 2.20}, {"event": "Iquique (Center)", "slip_cm": 2.50},
    {"event": "Boso SSE", "slip_cm": 3.00}, {"event": "Guerrero SSE", "slip_cm": 4.00}
]

# System Parameters
threshold = 0.87
window_hours = 120
std_cm = 0.088
N_eff = 337
sigma_iot_spatial = 30.0 / np.sqrt(N_eff)
MC_iterations = 1000
days_total = 40
n_samples = days_total * 24
t_hourly = np.linspace(-days_total, 0, n_samples)
idx_start = int((days_total - 15) * 24)
tau = 3.0

# Generate Common Noise Library
np.random.seed(42)
noise_cm_lib = np.zeros((MC_iterations, n_samples))
for i in range(MC_iterations):
    noise_cm_lib[i, 0] = np.random.normal(0, std_cm)
    for j in range(1, n_samples):
        noise_cm_lib[i, j] = 0.95 * noise_cm_lib[i, j-1] + np.random.normal(0, std_cm * np.sqrt(1 - 0.95**2))
iot_noise_lib = np.random.normal(0, sigma_iot_spatial, (MC_iterations, n_samples))
total_noise_lib = noise_cm_lib + iot_noise_lib

# Figure Setup
fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True)
axes = axes.flatten()

# Simulation & Plotting Loop
for idx, item in enumerate(catalog):
    mag = item["slip_cm"]
    event_name = item["event"]
    
    # Generate True Slip
    true_slip = np.zeros(n_samples)
    t_active = t_hourly[idx_start:]
    true_slip[idx_start:] = mag * (np.exp((t_active - t_active[0])/tau) - 1) / (np.exp((0 - t_active[0])/tau) - 1)
    
    # Process MC Iterations
    all_filtered = np.zeros((MC_iterations, n_samples))
    for i in range(MC_iterations):
        obs = true_slip + total_noise_lib[i]
        all_filtered[i, :] = pd.Series(obs).rolling(window=window_hours, min_periods=window_hours).mean().fillna(0).values
    
    mean_filtered = np.mean(all_filtered, axis=0)
    std_filtered = np.std(all_filtered, axis=0)
    ci_upper = mean_filtered + 1.96 * std_filtered
    ci_lower = mean_filtered - 1.96 * std_filtered

    # Plotting
    ax = axes[idx]
    ax.plot(t_hourly, true_slip, 'k--', lw=1.5, label='True Transient Slip')
    ax.plot(t_hourly, mean_filtered, 'b-', lw=1.5, label='Filtered Array Output')
    ax.fill_between(t_hourly, ci_lower, ci_upper, color='gray', alpha=0.3, label='95% MC Interval')
    ax.axhline(threshold, color='red', ls=':', lw=1.5, label='5$\\sigma$ Threshold')
    
    # Trigger Status
    triggered = np.any(mean_filtered > threshold)
    status_text = "TRIGGERED" if triggered else "MISSED"
    status_color = "red" if triggered else "black"
    ax.set_title(f"{event_name} ({mag} cm) - {status_text}", fontsize=11, color=status_color, fontweight='bold')
    
    ax.set_xlim(-25, 5)
    ax.set_ylim(-0.5, max(4.5, mag + 1.0))
    ax.grid(True, ls=':', alpha=0.5)
    
    if idx >= 9:
        ax.set_xlabel('Days to Mainshock')
    if idx % 3 == 0:
        ax.set_ylabel('Displacement (cm)')

# Global Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout()
plt.savefig('FigS1_All_Events.pdf', dpi=300, bbox_inches='tight')
print("Successfully generated FigS1_All_Events.pdf")
