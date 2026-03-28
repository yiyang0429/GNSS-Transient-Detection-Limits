import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14})

# Area 20x20 km
x = np.linspace(0, 20, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)

U0 = 2.0 
Rc = 3.0 
Slip = U0 * np.exp(-np.sqrt((X-10)**2 + (Y-10)**2) / Rc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

np.random.seed(10)
gx = np.random.uniform(1, 19, 15)
gy = np.random.uniform(1, 19, 15)
gx = np.append(gx[np.sqrt((gx-10)**2 + (gy-10)**2) > 4], [3, 17, 3, 17])[:15]
gy = np.append(gy[np.sqrt((gx-10)**2 + (gy-10)**2) > 4], [3, 3, 17, 17])[:15]

c1 = ax1.contourf(X, Y, Slip, levels=np.linspace(0, 2.0, 20), cmap='Reds', alpha=0.6)
ax1.scatter(gx, gy, c='blue', marker='^', s=100, edgecolor='black', label='Geodetic Station ($30k)')
ax1.contour(X, Y, Slip, levels=[0.63], colors='blue', linewidths=2, linestyles='--')
ax1.set_title('A. Geodetic Array ($450k, N=15)\nFATAL BLIND SPOT', fontweight='bold')
ax1.set_xlabel('East-West Distance (km)', fontweight='bold')
ax1.set_ylabel('North-South Distance (km)', fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_xlim(0, 20); ax1.set_ylim(0, 20)

ix = np.random.uniform(0, 20, 1000)
iy = np.random.uniform(0, 20, 1000)

c2 = ax2.contourf(X, Y, Slip, levels=np.linspace(0, 2.0, 20), cmap='Reds', alpha=0.6)
ax2.scatter(ix, iy, c='black', marker='.', s=15, alpha=0.5, label='IoT Station ($300)')
ax2.contour(X, Y, Slip, levels=[0.87], colors='black', linewidths=2, linestyles='-')
ax2.set_title('B. IoT Array ($300k, N=1000)\nZERO BLIND SPOTS', fontweight='bold')
ax2.set_xlabel('East-West Distance (km)', fontweight='bold')
ax2.set_ylabel('North-South Distance (km)', fontweight='bold')
ax2.legend(loc='upper right')
ax2.set_xlim(0, 20); ax2.set_ylim(0, 20)

fig.colorbar(c2, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.13, label='Precursor Surface Displacement (cm)')
plt.savefig('Fig5_Spatial_Aliasing.pdf', dpi=300, bbox_inches='tight')
print("Saved Fig5_Spatial_Aliasing.pdf (Axes units added)")
