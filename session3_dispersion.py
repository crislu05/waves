import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# -----------------------
# Load experimental data
# -----------------------
df = pd.read_csv('data/session3/data.csv', skipinitialspace=True)

# Extract frequency range and phase number
f_min = df['fmin (kHz)'].values
f_max = df['fmax (kHz)'].values
f_mean = df['fmean (kHz)'].values
omega = df['omega'].values  # Angular frequency in rad/s
k = df['k'].values

# Remove any NaN values
valid_mask = ~(pd.isna(f_min) | pd.isna(f_max) | pd.isna(f_mean) | pd.isna(omega) | pd.isna(k))
f_min = f_min[valid_mask]
f_max = f_max[valid_mask]
f_mean = f_mean[valid_mask]
omega = omega[valid_mask]
k = k[valid_mask]

# Sort by k to ensure correct order
sort_idx = np.argsort(k)
k = k[sort_idx]
f_min = f_min[sort_idx]
f_max = f_max[sort_idx]
f_mean = f_mean[sort_idx]
omega = omega[sort_idx]

# Convert frequency ranges to angular frequency (rad/s)
omega_min = 2 * np.pi * f_min * 1000  # Convert kHz to Hz, then to rad/s
omega_max = 2 * np.pi * f_max * 1000
omega_mean = 2 * np.pi * f_mean * 1000  # Convert kHz to Hz, then to rad/s (for consistency)

# Calculate wavenumber k_n = nπ/L where n is the phase number (k)
# L is the total number of sections (40 sections: L1-L40)
L_total = 40  # Total number of sections in the LC ladder
k_n = np.pi * k / L_total  # Wavenumber k_n = nπ/L

# Get L and C values from session3.py
L = 330e-6  # H (inductance per section)
C = 15e-9   # F (capacitance per section)

# Theoretical dispersion relation: ω = (2/√(LC)) * sin(k/2)
# where k is the wavenumber k_n
omega_cutoff = 2 / np.sqrt(L * C)  # Cutoff angular frequency
# Create fine grid of k_n for theoretical curve
k_n_theory = np.linspace(0, k_n.max(), 1000)
omega_theory = omega_cutoff * np.sin(k_n_theory / 2)  # ω = (2/√(LC)) * sin(k_n/2)

# Perform linear fit: omega_mean = a * k_n + b
slope_rad_per_s, intercept, r_value, p_value, std_err_rad_per_s = stats.linregress(k_n, omega_mean)
omega_fit = slope_rad_per_s * k_n + intercept

# Convert gradient to sections/s
# k_n = nπ/L where n is phase number and L = 40 sections
# dω/dk_n = slope_rad_per_s (in rad/s per unit k_n)
# To get dω/dn (change in angular frequency per section number):
#   dω/dn = dω/dk_n * dk_n/dn = slope_rad_per_s * (π/L)
# Velocity in sections/s = (dω/dn) / π = slope_rad_per_s * (π/L) / π = slope_rad_per_s / L
slope_sections_per_s = slope_rad_per_s   # Convert to sections/s
std_err_sections_per_s = std_err_rad_per_s 


# -----------------------
# Plot (Scientific style)
# -----------------------
fig, ax = plt.subplots(figsize=(10, 7))
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 100

# Plot vertical bars from omega_min to omega_max for each k_n (experimental data range)
for i in range(len(k_n)):
    ax.plot([k_n[i], k_n[i]], [omega_min[i], omega_max[i]], 
            color='#2E86AB', linewidth=4, alpha=0.7, zorder=2)

# Create a dummy line for legend
ax.plot([], [], color='#2E86AB', linewidth=4, label='Experimental data range')

# Plot linear fit
ax.plot(k_n, omega_fit, '--', color='#C9184A', linewidth=2.5, 
        label=f'Linear fit: gradient = ${slope_sections_per_s:.0f}$ sections/s, $R^2 = {r_value**2:.4f}$', 
        zorder=4)

# Plot theoretical dispersion relation
ax.plot(k_n_theory, omega_theory, '-', color='#06A77D', linewidth=2.5,
        label=f'Theoretical: $\\omega = \\frac{{2}}{{\\sqrt{{LC}}}} \\sin(k_n/2)$', 
        zorder=3, alpha=0.8)

# Formatting
ax.set_xlabel('Wavenumber $k_n = n\\pi/L$', fontweight='bold')
ax.set_ylabel('Angular frequency $\\omega$ (rad/s)', fontweight='bold')
ax.set_title('Frequency dispersion in LC ladder transmission line', 
             fontweight='bold', pad=15)

# Grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)


# Improve axis appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Set tight layout
plt.tight_layout()
plt.show()

print(f"Loaded {len(k)} data points")
print(f"Phase number range: k = {k.min():.0f} to {k.max():.0f}")
print(f"Angular frequency range: {omega_min.min():.1f} to {omega_max.max():.1f} rad/s")
print(f"\nLinear fit results:")
print(f"  Slope: {slope_sections_per_s:.0f} sections/s")
print(f"  Intercept: {intercept:.1f} rad/s")
print(f"  R² = {r_value**2:.4f}")
print(f"  Standard error: {std_err_sections_per_s:.0f} sections/s")

