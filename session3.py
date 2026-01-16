import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------
# Parameters from experiment
# -----------------------
Za = 90      # Ohms (coaxial cable)
L = 330e-6     # H (fixed - physical inductor value)
C = 15e-9      # F (fixed - physical capacitor value)


# Cutoff frequency
omega_c = 2 / np.sqrt(L*C)
f_c = omega_c / (2*np.pi)

# Frequency range (below cutoff)
f = np.linspace(1e3, 0.999*f_c, 2000)
omega = 2*np.pi*f

# -----------------------
# Ladder characteristic impedance
# -----------------------
Zb = np.sqrt(L/C) / np.sqrt(1 - omega**2 * L * C / 4)

# -----------------------
# Voltage transmission ratio (LAB FORMULA)
# -----------------------
Vratio = 2*Za / (Zb + Za)

# -----------------------
# Load experimental data
# -----------------------
df = pd.read_csv('data/session3/data.csv', skipinitialspace=True)

f_exp = df['fmean (kHz)'].values
V_in_min = df['V_in_min (V)'].values
V_in_max = df['V_in_max (V)'].values
V_out_min = df['V_out_min (V)'].values
V_out_max = df['V_out_max (V)'].values

valid_mask = ~(pd.isna(f_exp) | pd.isna(V_in_min) | pd.isna(V_out_min))
f_exp = f_exp[valid_mask]
V_in_min = V_in_min[valid_mask]
V_in_max = V_in_max[valid_mask]
V_out_min = V_out_min[valid_mask]
V_out_max = V_out_max[valid_mask]

ratio_min = V_out_min / V_in_max
ratio_max = V_out_max / V_in_min

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10, 6))

# Check if impedances are matched
Zb_dc = np.sqrt(L/C)
match_status = "matched" if abs(Za - Zb_dc) < 1 else "unmatched"
mismatch_percent = abs(Za - Zb_dc) / Za * 100

plt.plot(f/1e3, Vratio, linewidth=2,
         label='Theoretical', color='blue')

plt.axvline(f_c/1e3, linestyle='--', color='gray', label="Cutoff frequency")

for i in range(len(f_exp)):
    plt.plot([f_exp[i], f_exp[i]], [ratio_min[i], ratio_max[i]],
             'r-', linewidth=4, alpha=0.8)

plt.plot([], [], 'r-', linewidth=4, label='Experimental (V_out/V_in range)')

plt.xlabel("Frequency (kHz)")
plt.ylabel(r"$V_{\mathrm{transmitted}} / V_{\mathrm{in}}$")
plt.title("Voltage transmission ratio")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Cutoff frequency = {:.1f} kHz".format(f_c/1e3))
print("Source impedance Za = {:.1f} Ohms".format(Za))
print("LC ladder impedance Zb(DC) = sqrt(L/C) = {:.1f} Ohms".format(np.sqrt(L/C)))
print("Impedance match status: {}".format(match_status))
if not match_status == "matched":
    print("  Mismatch: {:.1f}%".format(mismatch_percent))
