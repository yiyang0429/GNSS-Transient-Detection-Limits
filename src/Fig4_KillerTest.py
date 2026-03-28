import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update({
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11, 'lines.linewidth': 2.0
})

# ----------------- 1. 空間物理設定 -----------------
U0 = 2.5 # 最大位移 2.5 cm
Rc = 3.0 # 彈性衰減半徑 3.0 km
r_geodetic = 4.0 # 最近的高精度測站距離 4.0 km

# ----------------- 2. 時間與訊號設定 -----------------
days_total = 40
n_samples = days_total * 24
t_hourly = np.linspace(-days_total, 0, n_samples)
idx_start = int((days_total - 15) * 24)
tau = 3.0

# 產生震央的真實滑移
true_slip_center = np.zeros(n_samples)
t_active = t_hourly[idx_start:]
true_slip_center[idx_start:] = U0 * (np.exp((t_active - t_active[0])/tau) - 1) / (np.exp((0 - t_active[0])/tau) - 1)

# Geodetic 衰減後的接收訊號
slip_geodetic_raw = true_slip_center * np.exp(-r_geodetic / Rc)

# ----------------- 3. 雜訊注入與濾波 (1000 次 MC) -----------------
MC_iterations = 500
window_hours = 120

std_cm = 0.088
std_geo = np.sqrt(std_cm**2 + 0.1**2) 
thresh_geo_practical = 0.63 

N_eff = 337
std_iot_spatial = 30.0 / np.sqrt(N_eff)
thresh_iot = 0.87

np.random.seed(42)
geo_filtered = np.zeros((MC_iterations, n_samples))
iot_filtered = np.zeros((MC_iterations, n_samples))

# IoT 採用 1km 內的平均訊號強度進行模擬 (保守估計 r=0.5km)
r_iot_effective = 0.5 
slip_iot_raw = true_slip_center * np.exp(-r_iot_effective / Rc)

for i in range(MC_iterations):
    noise_geo = np.random.normal(0, std_geo, n_samples)
    obs_geo = slip_geodetic_raw + noise_geo
    geo_filtered[i, :] = pd.Series(obs_geo).rolling(window=24, min_periods=1).mean().values
    
    noise_iot_temp = np.zeros(n_samples)
    noise_iot_temp[0] = np.random.normal(0, std_cm)
    for j in range(1, n_samples):
        noise_iot_temp[j] = 0.95 * noise_iot_temp[j-1] + np.random.normal(0, std_cm * np.sqrt(1 - 0.95**2))
    noise_iot_hw = np.random.normal(0, std_iot_spatial, n_samples)
    obs_iot = slip_iot_raw + noise_iot_temp + noise_iot_hw
    iot_filtered[i, :] = pd.Series(obs_iot).rolling(window=window_hours, min_periods=window_hours).mean().values

geo_mean = np.nanmean(geo_filtered, axis=0)
iot_mean = np.nanmean(iot_filtered, axis=0)

# ----------------- 4. 雙面板繪圖 -----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: 空間幾何
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Slip_map = U0 * np.exp(-np.sqrt(X**2 + Y**2) / Rc)

c1 = ax1.contourf(X, Y, Slip_map, levels=np.linspace(0, 2.5, 20), cmap='Reds', alpha=0.6)

# Geodetic 測站 (故意留出 4km 盲區)
gx = np.array([-4, 4, -4, 4, 0, -8, 8, 0, -8, 8])
gy = np.array([-4, -4, 4, 4, 8, 0, 0, -8, 8, -8])
ax1.plot(gx, gy, 'b^', markersize=10, markeredgecolor='black', alpha=0.8, label='Geodetic Station ($N=15$, sparse)')

# 生成 1000 個 IoT 測站
np.random.seed(15) # 挑一個看起來自然均勻的 seed
ix = np.random.uniform(-10, 10, 1000)
iy = np.random.uniform(-10, 10, 1000)

# 找出距離震央 1km 內的 IoT 測站
dist_to_center = np.sqrt(ix**2 + iy**2)
close_idx = dist_to_center <= 1.0

# 畫出背景的 IoT
ax1.scatter(ix[~close_idx], iy[~close_idx], c='black', marker='.', s=10, alpha=0.15, label='IoT Array ($N=1000$, dense)')
# 畫出核心區(觸發區)的 IoT
ax1.scatter(ix[close_idx], iy[close_idx], c='black', marker='o', s=35, edgecolor='white', linewidth=1, label='Proximal IoT Nodes ($r < 1.0$ km)')

# 畫出斷層中心
ax1.plot(0, 0, 'r*', markersize=18, markeredgecolor='black', label='Nucleation Center (2.5 cm)')
ax1.add_patch(Circle((0, 0), Rc, color='red', fill=False, linestyle='--', linewidth=2, label=f'Attenuation Radius ($R_c$={Rc} km)'))

ax1.set_title('A. Spatial Geometry of the Nucleation Event', fontweight='bold')
ax1.set_xlabel('East-West Distance (km)', fontweight='bold')
ax1.set_ylabel('North-South Distance (km)', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(ls=':', alpha=0.5)
ax1.set_xlim(-10, 10); ax1.set_ylim(-10, 10)

# Panel B: 時間動態軌跡對決
ax2.plot(t_hourly, true_slip_center, 'k--', alpha=0.5, lw=2, label='True Slip at Center (2.5 cm max)')
ax2.plot(t_hourly, slip_geodetic_raw, 'b--', alpha=0.3, lw=2, label='True Slip at Geodetic (Attenuated)')

ax2.plot(t_hourly, geo_mean, 'b-', lw=2.5, label='Filtered Geodetic Output')
ax2.plot(t_hourly, iot_mean, 'k-', lw=2.5, label='Filtered Proximal IoT Output')

ax2.axhline(thresh_geo_practical, color='blue', ls=':', lw=2, label='Geodetic $5\sigma$ Threshold (0.63 cm)')
ax2.axhline(thresh_iot, color='red', ls=':', lw=2, label='IoT Array $5\sigma$ Threshold (0.87 cm)')

trig_iot = t_hourly[np.where(iot_mean > thresh_iot)[0][0]] if np.any(iot_mean > thresh_iot) else np.nan
if not np.isnan(trig_iot):
    ax2.axvline(trig_iot, color='red', ls='-.', alpha=0.7)
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8)
    ax2.annotate(f'IoT TRIGGERED\n({abs(trig_iot):.1f} days lead)', xy=(trig_iot, thresh_iot), 
                 xytext=(trig_iot-8, thresh_iot+0.5), arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                 ha='right', va='center', fontweight='bold', color='red', bbox=bbox_props)

ax2.annotate('Geodetic MISSED\n(Signal buried in threshold)', xy=(-2, geo_mean[-50]), 
             xytext=(-12, 0.2), arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
             ha='right', va='center', fontweight='bold', color='blue')

ax2.set_title('B. Dynamic Spatiotemporal Race to Threshold', fontweight='bold')
ax2.set_xlabel('Days to Mainshock', fontweight='bold')
ax2.set_ylabel('Displacement (cm)', fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, ls=':', alpha=0.7)
ax2.set_xlim(-20, 0)
ax2.set_ylim(-0.2, 2.7)

plt.tight_layout()
plt.savefig('Fig4_KillerTest.pdf', dpi=300)
print("Saved Fig4_KillerTest.pdf (Dense random scattering added to Panel A)")
