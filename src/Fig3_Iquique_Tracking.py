import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})

# 假設 IQQE.tenv3 已存在於同一目錄
cols_ngl = ['site', 'YYMMMDD', 'yyyy_yyyy', 'MJD', 'week', 'd', 'reflon', 'e0', 'east_m', 'n0', 'north_m', 'u0', 'up_m', 'ant', 'sig_e', 'sig_n', 'sig_u', 'corr_en', 'corr_eu', 'corr_nu', 'lat', 'lon', 'height']
try:
    iqqe_data = pd.read_csv('IQQE.tenv3', sep=r'\s+', header=None, names=cols_ngl, skiprows=1)
except FileNotFoundError:
    print("Error: IQQE.tenv3 not found. Please ensure it is in the working directory.")
    exit()

mainshock_year = 2014.249 
mask_iqqe = (iqqe_data['yyyy_yyyy'] >= mainshock_year - (40/365.25)) & (iqqe_data['yyyy_yyyy'] <= mainshock_year)
iqqe_window = iqqe_data[mask_iqqe].copy()

t_daily = (iqqe_window['yyyy_yyyy'].values - mainshock_year) * 365.25
slip_daily_edge = np.abs(iqqe_window['east_m'].values * 100.0 - (iqqe_window['east_m'].values[0] * 100.0))
slip_daily_center = slip_daily_edge * (2.5 / np.max(slip_daily_edge))

t_hourly = np.arange(t_daily[0], t_daily[-1], 1/24.0)
slip_edge = interp1d(t_daily, slip_daily_edge, fill_value='extrapolate')(t_hourly)
slip_center = interp1d(t_daily, slip_daily_center, fill_value='extrapolate')(t_hourly)

# 1000 MC Iterations for Confidence Intervals
MC_iterations = 1000
window_hours = 120
threshold = 0.87
std_cm = 0.088
sigma_iot_spatial = 30.0 / np.sqrt(337)

print(f"Running {MC_iterations} MC iterations for Iquique Tracking...")
np.random.seed(42)

res_center = np.zeros((MC_iterations, len(t_hourly)))
res_edge = np.zeros((MC_iterations, len(t_hourly)))

for i in range(MC_iterations):
    noise_cm = np.zeros(len(t_hourly))
    noise_cm[0] = np.random.normal(0, std_cm)
    for j in range(1, len(t_hourly)): noise_cm[j] = 0.95 * noise_cm[j-1] + np.random.normal(0, std_cm * np.sqrt(1 - 0.95**2))
    iot_noise = np.random.normal(0, sigma_iot_spatial, len(t_hourly))
    
    total_noise = noise_cm + iot_noise
    res_center[i, :] = pd.Series(slip_center + total_noise).rolling(window=window_hours, min_periods=window_hours).mean().values
    res_edge[i, :] = pd.Series(slip_edge + total_noise).rolling(window=window_hours, min_periods=window_hours).mean().values

# Calculate mean and 95% CI
mean_center = np.nanmean(res_center, axis=0)
ci_center = 1.96 * np.nanstd(res_center, axis=0)
mean_edge = np.nanmean(res_edge, axis=0)
ci_edge = 1.96 * np.nanstd(res_edge, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Panel A: PSAD (Center)
ax1.plot(t_hourly, slip_center, 'k--', alpha=0.7, label='True Slip (PSAD ~2.5 cm)')
ax1.plot(t_hourly, mean_center, 'g-', lw=2, label='MC Mean Output')
ax1.fill_between(t_hourly, mean_center - ci_center, mean_center + ci_center, color='green', alpha=0.2, label='95% Confidence Interval')
ax1.axhline(threshold, color='r', ls=':', lw=2, label='5$\sigma$ Threshold (0.87 cm)')
ax1.set_title('A. Rupture Center (PSAD Station)', fontweight='bold') # Removed "Figure X"
ax1.set_xlabel('Days to Mainshock', fontweight='bold')
ax1.set_ylabel('Displacement (cm)', fontweight='bold')
ax1.grid(ls=':'); ax1.legend(loc='upper left')

# Panel B: IQQE (Edge)
ax2.plot(t_hourly, slip_edge, 'k--', alpha=0.7, label='True Slip (IQQE ~0.74 cm)')
ax2.plot(t_hourly, mean_edge, 'C0-', lw=2, label='MC Mean Output')
ax2.fill_between(t_hourly, mean_edge - ci_edge, mean_edge + ci_edge, color='C0', alpha=0.2, label='95% Confidence Interval')
ax2.axhline(threshold, color='r', ls=':', lw=2, label='5$\sigma$ Threshold (0.87 cm)')
ax2.set_title('B. Rupture Edge (IQQE Station)', fontweight='bold') # Removed "Figure X"
ax2.set_xlabel('Days to Mainshock', fontweight='bold')
ax2.grid(ls=':'); ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('Fig3_Iquique_Tracking.pdf', dpi=300)
print("Saved Fig3_Iquique_Tracking.pdf")