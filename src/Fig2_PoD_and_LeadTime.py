import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11, 'lines.linewidth': 2.0
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
N_array, corr_area, area = 1000, np.pi * (0.5**2), 400.0
N_eff = 1 / ((1 / N_array) + (corr_area / area))
sigma_iot_spatial = 30.0 / np.sqrt(N_eff)
window_hours = 120 
std_cm = 0.088 
std_iot_time = sigma_iot_spatial / np.sqrt(window_hours)
threshold_5sigma = 5 * np.sqrt(std_cm**2 + std_iot_time**2) # ~0.87 cm

MC_iterations = 1000
magnitudes = np.linspace(0.0, 4.5, 50) 
days_total = 40
n_samples = days_total * 24

print(f"Running {MC_iterations} MC iterations for PoD and Lead Time Curves...")
np.random.seed(42)

noise_cm_lib = np.zeros((MC_iterations, n_samples))
for i in range(MC_iterations):
    noise_cm_lib[i, 0] = np.random.normal(0, std_cm)
    for j in range(1, n_samples):
        noise_cm_lib[i, j] = 0.95 * noise_cm_lib[i, j-1] + np.random.normal(0, std_cm * np.sqrt(1 - 0.95**2))
iot_noise_lib = np.random.normal(0, sigma_iot_spatial, (MC_iterations, n_samples))
total_noise_lib = noise_cm_lib + iot_noise_lib

pod_results = []
lead_time_results = []

t_hourly = np.linspace(-days_total, 0, n_samples)
idx_start = int((days_total - 15) * 24)
tau = 3.0
t_active = t_hourly[idx_start:]
shape_curve = np.zeros(n_samples)
shape_curve[idx_start:] = (np.exp((t_active - t_active[0])/tau) - 1) / (np.exp((0 - t_active[0])/tau) - 1)

for mag in magnitudes:
    trigger_count = 0
    lead_times = []
    signal_hourly = shape_curve * mag
    for i in range(MC_iterations):
        obs = signal_hourly + total_noise_lib[i]
        filtered = pd.Series(obs).rolling(window=window_hours, min_periods=window_hours).mean().values
        
        cond = filtered > threshold_5sigma
        for k in range(window_hours, len(cond)-12):
            if np.all(cond[k:k+12]):
                trigger_count += 1
                trigger_day = (k+12) / 24.0
                lead_times.append(days_total - trigger_day)
                break
                
    pod_results.append((trigger_count / MC_iterations) * 100)
    if trigger_count > 0:
        lead_time_results.append(np.mean(lead_times))
    else:
        lead_time_results.append(0.0)

# ----------------- 新增：計算 PoD 的二項式 95% 信心區間 -----------------
pod_array = np.array(pod_results)
# Formula: 1.96 * sqrt(p * (1-p) / N)
ci_95 = 1.96 * np.sqrt((pod_array/100.0) * (1.0 - pod_array/100.0) / MC_iterations) * 100.0

lead_time_smoothed = np.maximum.accumulate(lead_time_results)

pod_interp = np.interp([item["slip_cm"] for item in catalog], magnitudes, pod_results)
lead_time_interp = np.interp([item["slip_cm"] for item in catalog], magnitudes, lead_time_smoothed)

# ----------------- 繪圖 -----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
arrow_style = dict(arrowstyle="-", color="gray", lw=1.2)

# Panel A
ax1.plot(magnitudes, pod_results, 'k-', lw=2.5, label='Theoretical PoD')
# 畫上誤差帶！
ax1.fill_between(magnitudes, pod_array - ci_95, pod_array + ci_95, color='gray', alpha=0.4, label='95% Confidence Interval')

ax1.scatter([item["slip_cm"] for item in catalog], pod_interp, c=pod_interp, cmap='RdYlGn', s=140, edgecolor='black', zorder=5)

labels_ax1 = {
    "Tohoku-Oki": (0.3, 20), "Iquique (Edge)": (0.74, 40), "Sanriku-Oki": (1.1, 15),
    "Izmit": (1.0, 55), "Valparaiso": (1.4, 35), "Nicoya": (1.3, 110),
    "Kaikoura": (1.8, 115), "Papanoa": (2.2, 110), "Gisborne": (2.6, 115),
    "Iquique (Center)": (2.9, 110), "Boso SSE": (3.4, 115), "Guerrero SSE": (4.0, 110)
}
for i, item in enumerate(catalog):
    ax1.annotate(item["event"], xy=(item["slip_cm"], pod_interp[i]), xycoords='data',
                 xytext=labels_ax1[item["event"]], textcoords='data', 
                 arrowprops=arrow_style, ha='center', va='center', fontsize=10, fontweight='bold', bbox=bbox_props)

ax1.axvline(threshold_5sigma, color='red', ls='--', lw=2, label=f'5$\sigma$ Threshold ({threshold_5sigma:.2f} cm)')
ax1.axhline(80, color='gray', ls=':', lw=2)
# ===== BSSA 標籤更新：Panel A X軸 =====
ax1.set_xlabel('Pre-rupture Slip Magnitude (cm)', fontweight='bold')
ax1.set_ylabel('Probability of Detection (%)', fontweight='bold')
ax1.set_title('A. System Detection Capability (PoD)', fontweight='bold')
ax1.grid(True, ls=':', alpha=0.7)
ax1.set_xlim(0, 4.3); ax1.set_ylim(-5, 125)
ax1.legend(loc='lower right')

# Panel B
ax2.plot(magnitudes, lead_time_smoothed, 'b-', lw=2.5, label='Average Lead Time')
ax2.scatter([item["slip_cm"] for item in catalog], lead_time_interp, c=pod_interp, cmap='RdYlGn', s=140, edgecolor='black', zorder=5)

labels_ax2 = {
    "Tohoku-Oki": (0.3, 4), "Iquique (Edge)": (0.6, 6), "Sanriku-Oki": (0.9, 2),
    "Izmit": (0.8, 9), "Valparaiso": (1.0, 11), "Nicoya": (1.2, 13),
    "Kaikoura": (1.5, 15), "Papanoa": (1.7, 17), "Gisborne": (1.9, 19),
    "Iquique (Center)": (2.2, 21), "Boso SSE": (2.7, 23), "Guerrero SSE": (3.5, 23)
}
for i, item in enumerate(catalog):
    ax2.annotate(item["event"], xy=(item["slip_cm"], lead_time_interp[i]), xycoords='data',
                 xytext=labels_ax2[item["event"]], textcoords='data', 
                 arrowprops=arrow_style, ha='center', va='center', fontsize=10, fontweight='bold', bbox=bbox_props)

ax2.axvline(threshold_5sigma, color='red', ls='--', lw=2, label=f'5$\sigma$ Threshold ({threshold_5sigma:.2f} cm)')
# ===== BSSA 標籤更新：Panel B 標籤與標題 =====
ax2.set_xlabel('Pre-rupture Slip Magnitude (cm)', fontweight='bold')
ax2.set_ylabel('Average Lead Time (Days)', fontweight='bold')
ax2.set_title('B. Average Trigger Lead Time', fontweight='bold')
ax2.grid(True, ls=':', alpha=0.7)
ax2.set_xlim(0, 4.3); ax2.set_ylim(-1, 25)
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('Fig2_PoD_and_LeadTime.pdf', dpi=300)
print("Saved Fig2_PoD_and_LeadTime.pdf (BSSA terminology compliant!)")
