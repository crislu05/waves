import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def load_dataset(path):
	"""Load thermal dataset from CSV file."""
	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	output_voltage = data.iloc[:, 1].to_numpy()
	output_current = data.iloc[:, 2].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()
	
	return timestamp, output_voltage, output_current, thermistor_temperatures


def calculate_qc(alpha, I, Tc, R, K, Th):
	"""
	Calculate heat flow rate at cold side.
	Heat added to cold plate: - Peltier removal + 1/2 Joule heating + Conductive leak from hot side
	
	Parameters:
	- alpha: Seebeck coefficient (V/K)
	- I: Current through Peltier device (A)
	- Tc: Cold side temperature (K)
	- R: Electrical resistance (Ohms)
	- K: Thermal conductance (W/K)
	- Th: Hot side temperature (K)
	"""
	# Heat added to cold plate: 
	# - Peltier removal + 1/2 Joule heating + Conductive leak from hot side
	qc = - (alpha * I * Tc) + (0.5 * I**2 * R) + K * (Th - Tc)
	return qc


def calculate_qh(alpha, I, Th, R, K, Tc):
	"""
	Calculate heat flow rate at hot side.
	Heat added to hot plate: + Peltier dumping + 1/2 Joule heating - Conductive leak to cold side
	
	Parameters:
	- alpha: Seebeck coefficient (V/K)
	- I: Current through Peltier device (A)
	- Th: Hot side temperature (K)
	- R: Electrical resistance (Ohms)
	- K: Thermal conductance (W/K)
	- Tc: Cold side temperature (K)
	"""
	# Heat added to hot plate: 
	# + Peltier dumping + 1/2 Joule heating - Conductive leak to cold side
	qh = (alpha * I * Th) + (0.5 * I**2 * R) - K * (Th - Tc)
	return qh


def solve_1d_heat_equation(t_span, x_grid, T_initial, voltage_func,
                           L, radius, alpha, K, T_cold, R, D_brass, k_brass, 
                           h, rho, c, T_infinity, C_ceramic_hot, A_ceramic_hot, h_ceramic_hot, rtol, atol):
	"""
	Solve 1D heat equation with convective heat loss: ∂T/∂t = D * ∂²T/∂x² - (hP/(ρcA))[T - T∞]
	Also solves coupled differential equation for Th (hot side ceramic plate temperature).
	
	Boundary conditions:
	- At x=0: -k_rod * (∂T/∂x)|_(x=0) = Q_c(t)
	  where Q_c(t) = -αIT_c + (1/2)I²R + K(T_h - T_c)  (heat added to cold plate)
	  and T_c(t) = T_rod(0, t)  (rod temperature at interface equals cold plate temperature)
	- At x=L: (∂T/∂x)|_(x=L) = 0 (insulated boundary - avoids double-counting convection)
	
	Coupled differential equation:
	- C_h * Ṫ_h = Q_h + h_h * A_h * (T_∞ - T_h)
	
	Note: No separate ceramic ODE for cold side - T[0] is the cold plate temperature.
	
	Parameters:
	- t_span: Time span (t0, tf) in seconds
	- x_grid: Spatial grid points (array)
	- T_initial: Initial temperature distribution (array, same length as x_grid)
	- voltage_func: Function voltage(t) returning voltage at time t
	- L: Length of the cylinder (m)
	- radius: Radius of the cylinder (m)
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance of Peltier (W/K)
	- T_cold: Cold side temperature in Kelvin (initial/reference)
	- R: Electrical resistance (Ohms)
	- D_brass: Thermal diffusivity of brass (m²/s)
	- k_brass: Thermal conductivity of brass (W/(m·K))
	- h: Convective heat transfer coefficient for brass rod (W/(m²·K))
	- rho: Density of brass (kg/m³)
	- c: Specific heat capacity of brass (J/(kg·K))
	- T_infinity: Ambient/environment temperature (K)
	- C_ceramic_hot: Heat capacity of hot ceramic plate (J/K)
	- A_ceramic_hot: Surface area of hot ceramic plate (m²)
	- h_ceramic_hot: Convective heat transfer coefficient for hot ceramic plate (W/(m²·K))
	- rtol: Relative tolerance for ODE solver
	- atol: Absolute tolerance for ODE solver
	"""
	# Cross-sectional area and perimeter
	A = np.pi * radius**2
	P = 2 * np.pi * radius  # Perimeter of cylinder
	
	# Convective heat loss coefficient: hP / (ρcA)
	convective_coeff = (h * P) / (rho * c * A)
	
	# Spatial grid
	dx = x_grid[1] - x_grid[0]
	N = len(x_grid)
	
	# State vector: T[0] to T[N-1] are rod temperatures (T[0] = Tc), T[N] is Th
	def heat_equation_rhs(t, T_full):
		"""Right-hand side of the coupled heat equation system."""
		# Split state vector: rod temperatures and hot side ceramic plate temperature
		T_rod = T_full[:N]  # Rod temperatures T[0] to T[N-1], where T[0] = Tc
		Tc = T_rod[0]       # Cold side temperature (rod temperature at x=0)
		Th = T_full[N]      # Hot side ceramic plate temperature
		
		dTdt = np.zeros_like(T_full)
		
		# Calculate current from voltage using: I(t) = (V(t) - α(T_h - T_c)) / R
		V = voltage_func(t)
		I = (V - alpha * (Th - Tc)) / R
		
		# Calculate Qc and Qh using coupled equations
		Qc = calculate_qc(alpha, I, Tc, R, K, Th)
		Qh = calculate_qh(alpha, I, Th, R, K, Tc)
		
		# Boundary condition at x=0: -k_rod * (∂T/∂x)|_(x=0) = Q_c(t)
		# Rearranging: (∂T/∂x)|_(x=0) = -Q_c / k_rod
		# Using ghost point: (T[1] - T[-1]) / (2*dx) = -Q_c / k_rod
		# So: T[-1] = T[1] - (2*dx / k) * Q_c
		# T[0] = Tc (rod temperature at interface equals cold plate temperature)
		# No separate ceramic ODE for Tc - it evolves only through the rod equation
		T_ghost_0 = T_rod[1] - 2 * dx * Qc / k_brass
		diffusion_term_0 = D_brass * (T_rod[1] - 2*Tc + T_ghost_0) / dx**2
		convective_term_0 = -convective_coeff * (Tc - T_infinity)
		dTdt[0] = diffusion_term_0 + convective_term_0
		
		# Interior points: ∂T/∂t = D * ∂²T/∂x² - (hP/(ρcA))[T - T∞]
		# Using central difference for spatial derivative
		for i in range(1, N-1):
			diffusion_term = D_brass * (T_rod[i+1] - 2*T_rod[i] + T_rod[i-1]) / dx**2
			convective_term = -convective_coeff * (T_rod[i] - T_infinity)
			dTdt[i] = diffusion_term + convective_term
		
		# Solve for Th: C_h * Ṫ_h = Q_h + h_h * A_h * (T_∞ - T_h)
		dTdt[N] = (Qh + h_ceramic_hot * A_ceramic_hot * (T_infinity - Th)) / C_ceramic_hot
		
		# Boundary condition at x=L: Insulated boundary (∂T/∂x)|_(x=L) = 0
		# This avoids double-counting convection, since distributed convection -hP/(ρcA)(T - T∞)
		# is already applied everywhere including at x=L
		# Using ghost point method: (∂T/∂x)|_(x=L) ≈ (T[N] - T[N-2]) / (2*dx) = 0
		# So: T[N] = T[N-2]
		# For the heat equation at x=L: dT[N-1]/dt = D * (T[N] - 2*T[N-1] + T[N-2]) / dx² - (hP/(ρcA))[T[N-1] - T∞]
		T_ghost_L = T_rod[N-2]  # Insulated: no heat flux across boundary
		diffusion_term_L = D_brass * (T_ghost_L - 2*T_rod[N-1] + T_rod[N-2]) / dx**2
		convective_term_L = -convective_coeff * (T_rod[N-1] - T_infinity)
		dTdt[N-1] = diffusion_term_L + convective_term_L
		
		return dTdt
	
	# Initial conditions: rod temperatures + Th
	T_hot_initial = T_cold  # Initial hot side temperature (same as cold side initially)
	T_initial_full = np.concatenate([T_initial, [T_hot_initial]])  # Add Th to state vector
	
	# Solve the ODE system
	sol = solve_ivp(heat_equation_rhs, t_span, T_initial_full, 
	                method='RK45', dense_output=True, rtol=rtol, atol=atol)
	
	return sol


def plot_temperature_vs_time(sol, timestamp, thermistor_0_data, voltage_func,
                            radius, alpha, K, R):
	"""
	Plot temperature as a function of time at x=0, thermistor 0 data, and heat flux Q at x=0.
	
	Parameters:
	- sol: Solution object from solve_ivp
	- timestamp: Original time data for reference
	- thermistor_0_data: Thermistor 0 temperature data (array)
	- voltage_func: Function voltage(t) returning voltage at time t
	- radius: Radius of cylinder (m)
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance of Peltier (W/K)
	- R: Electrical resistance (Ohms)
	"""
	# Evaluate solution at evenly spaced time points
	t_eval = np.linspace(timestamp[0], timestamp[-1], min(500, len(timestamp)))
	T_solution = sol.sol(t_eval)  # Shape: (N_x+1, len(t_eval)) - last row is Th
	
	# Extract rod temperatures and Th
	N_x = T_solution.shape[0] - 1  # Number of rod grid points
	T_rod_solution = T_solution[:N_x, :]  # Rod temperatures
	Th_solution = T_solution[N_x, :]  # Hot side temperature
	
	# Convert to Celsius and get temperature at x=0 (Tc)
	T_at_x0_C = T_rod_solution[0, :] - 273.15
	T_at_x0_K = T_rod_solution[0, :]  # Keep in Kelvin for heat flux calculation
	Th_C = Th_solution - 273.15  # Hot side temperature in Celsius
	
	# Calculate Qc (heat flow rate) at x=0 for each time point using form from heatpump.py
	A = np.pi * radius**2
	Qc_at_x0 = np.zeros_like(t_eval)
	
	for i, t in enumerate(t_eval):
		# Calculate I from V using: I(t) = (V(t) - α(T_h - T_c)) / R
		V = voltage_func(t)
		Tc = T_at_x0_K[i]
		Th = Th_solution[i]
		I = (V - alpha * (Th - Tc)) / R
		# Qc = -αITc + (1/2)I²R + K(Th - Tc)  (heat added to cold plate)
		Qc_at_x0[i] = calculate_qc(alpha, I, Tc, R, K, Th)
	
	# Convert to heat flux for plotting (W/m²)
	Q_at_x0 = Qc_at_x0 / A
	
	# Create figure with three subplots
	fig, axes = plt.subplots(3, 1, figsize=(10, 12))
	
	# Plot 1: Temperature at x=0 (Tc) and Th from heat equation solution
	axes[0].plot(t_eval, T_at_x0_C, 'b-', linewidth=2, label='Tc (cold side, x=0)')
	axes[0].plot(t_eval, Th_C, 'r-', linewidth=2, label='Th (hot side)')
	axes[0].set_xlabel('Time (s)', fontsize=12)
	axes[0].set_ylabel('Temperature (°C)', fontsize=12)
	axes[0].set_title('Temperatures vs Time (Heat Equation Solution)', fontsize=14, fontweight='bold')
	axes[0].grid(True, alpha=0.3)
	axes[0].legend(loc='best', fontsize=11)
	
	# Plot 2: Thermistor 0 data
	axes[1].plot(timestamp, thermistor_0_data, 'r-', linewidth=2, label='Thermistor 0')
	axes[1].set_xlabel('Time (s)', fontsize=12)
	axes[1].set_ylabel('Temperature (°C)', fontsize=12)
	axes[1].set_title('Thermistor 0 Temperature vs Time (Experimental Data)', fontsize=14, fontweight='bold')
	axes[1].grid(True, alpha=0.3)
	axes[1].legend(loc='best', fontsize=11)
	
	# Plot 3: Heat flux Q at x=0
	axes[2].plot(t_eval, Q_at_x0, 'g-', linewidth=2, label='Heat flux Q at x=0')
	axes[2].set_xlabel('Time (s)', fontsize=12)
	axes[2].set_ylabel('Heat Flux Q (W/m²)', fontsize=12)
	axes[2].set_title('Heat Flux Q at x=0 Boundary vs Time', fontsize=14, fontweight='bold')
	axes[2].grid(True, alpha=0.3)
	axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
	axes[2].legend(loc='best', fontsize=11)
	
	plt.tight_layout()
	plt.show()


def main():
	"""Main function to run the thermal analysis."""
	# Load data
	filepath = 'data/session6/brass_7V_10s.csv'
	timestamp, voltage, current, thermistor_temperatures = load_dataset(filepath)
	thermistor_0 = thermistor_temperatures[:, 0]  # Extract thermistor 0 data
	
	# Thermoelectric device parameters
	alpha = 0.05  # V/K (Seebeck coefficient)
	K = 0.5  # W/K (Thermal conductance of Peltier)
	R = 2.5  # Ohm (Electrical resistance, can be estimated from V/I)
	T_cold = 298.15  # K (25°C, cold side temperature, initial/reference)
	# Note: delta_T is calculated dynamically from V = IR + αΔT
	
	# Brass cylinder parameters
	radius = 0.015  # m (1.5 cm)
	L = 0.041  # m (assumed length, adjust as needed)
	
	# Brass material properties
	D_brass = 3.4e-5  # m²/s (Thermal diffusivity of brass)
	k_brass = 109.0  # W/(m·K) (Thermal conductivity of brass)
	rho_brass = 8520.0  # kg/m³ (Density of brass)
	c_brass = 380.0  # J/(kg·K) (Specific heat capacity of brass)
	
	# Convective heat transfer parameters
	h = 10.0  # W/(m²·K) (Convective heat transfer coefficient, typical for natural convection)
	T_infinity = 298.15  # K (Ambient/environment temperature, 25°C)
	
	# Ceramic plate parameters (only for hot side - solving Th via ODE)
	# Ceramic properties: Aluminum Oxide (Al2O3) - typical for Peltier devices
	rho_ceramic = 3970.0  # kg/m³ (Al2O3 density)
	c_ceramic = 775.0  # J/(kg·K) (Al2O3 specific heat capacity)
	thickness_ceramic = 0.002  # m (typical ceramic plate thickness ~2mm)
	volume_ceramic_hot = np.pi * radius**2 * thickness_ceramic  # m³
	mass_ceramic_hot = rho_ceramic * volume_ceramic_hot  # kg
	C_ceramic_hot = mass_ceramic_hot * c_ceramic  # J/K (heat capacity)
	A_ceramic_hot = 2 * np.pi * radius**2 + 2 * np.pi * radius * thickness_ceramic  # m²
	h_ceramic_hot = 10.0  # W/(m²·K) (Convective heat transfer coefficient)
	# Note: No ceramic parameters for cold side - T[0] is the cold plate temperature, no separate ODE
	
	# Numerical parameters
	N_x = 100  # Number of spatial points
	rtol = 1e-6  # Relative tolerance for ODE solver
	atol = 1e-8  # Absolute tolerance for ODE solver
	
	# Calculate Qc statistics (using average values for initial estimate)
	# delta_T is calculated from V = IR + αΔT, so ΔT = (V - IR) / α
	# For statistics, estimate Th_avg = T_cold + delta_T_avg
	delta_T_avg = np.mean((voltage - current * R) / alpha)
	Th_avg = T_cold + delta_T_avg
	V_avg = np.mean(voltage)
	I_avg = (V_avg - alpha * delta_T_avg) / R
	qc_dot = calculate_qc(alpha, I_avg, T_cold, R, K, Th_avg)
	
	# Set up spatial grid for 1D heat equation
	x_grid = np.linspace(0, L, N_x)
	
	# Initial temperature distribution (uniform at ambient)
	T_initial = np.ones(N_x) * T_cold
	
	# Create interpolation function for voltage
	from scipy.interpolate import interp1d
	voltage_interp = interp1d(timestamp, voltage, kind='linear', 
	                          fill_value=(voltage[0], voltage[-1]), bounds_error=False)
	
	# Time span for solving
	t_span = (timestamp[0], timestamp[-1])
	
	# Solve 1D heat equation with coupled Th (Tc = T[0], no separate ODE)
	print("Solving 1D heat equation with coupled Th...")
	print("  T[0] is the cold plate temperature (no separate ceramic ODE)")
	print("  Th is solved via: C_h * Ṫ_h = Q_h + h_h * A_h * (T_∞ - T_h)")
	sol = solve_1d_heat_equation(t_span, x_grid, T_initial, voltage_interp,
	                             L, radius, alpha, K, T_cold, R, D_brass, k_brass,
	                             h, rho_brass, c_brass, T_infinity, 
	                             C_ceramic_hot, A_ceramic_hot, h_ceramic_hot, rtol, atol)
	
	# Print summary statistics
	print(f"\nData Summary for {filepath}:")
	print(f"  Time range: {timestamp[0]:.2f} s to {timestamp[-1]:.2f} s")
	print(f"  Duration: {timestamp[-1] - timestamp[0]:.2f} s")
	print(f"  Voltage range: {voltage.min():.3f} V to {voltage.max():.3f} V")
	print(f"  Current range: {current.min():.3f} A to {current.max():.3f} A")
	print(f"\nPhysical Parameters:")
	print(f"  Cylinder radius: {radius*100:.1f} cm")
	print(f"  Cylinder length: {L*100:.1f} cm")
	print(f"  Cross-sectional area: {np.pi * radius**2 * 1e4:.2f} cm²")
	print(f"  Brass thermal diffusivity (D): {D_brass:.2e} m²/s")
	print(f"  Brass thermal conductivity (k): {k_brass:.1f} W/(m·K)")
	print(f"  Brass density (ρ): {rho_brass:.0f} kg/m³")
	print(f"  Brass specific heat (c): {c_brass:.0f} J/(kg·K)")
	print(f"\nConvective Heat Transfer Parameters:")
	print(f"  Heat transfer coefficient (h): {h:.1f} W/(m²·K)")
	print(f"  Ambient temperature (T∞): {T_infinity-273.15:.1f} °C ({T_infinity:.2f} K)")
	print(f"\nThermoelectric Parameters:")
	print(f"  Seebeck coefficient (α): {alpha*1e6:.0f} μV/K")
	print(f"  Thermal conductance (K): {K:.2f} W/K")
	print(f"  Electrical resistance (R): {R:.2f} Ω")
	print(f"  Cold side temp (Tc, initial): {T_cold-273.15:.1f} °C ({T_cold:.2f} K)")
	print(f"  Temperature difference (ΔT, average): {delta_T_avg:.1f} K")
	print(f"    (ΔT calculated dynamically from V = IR + αΔT)")
	print(f"\nQ̇c Statistics:")
	print(f"  Q̇c range: {qc_dot.min():.4f} W to {qc_dot.max():.4f} W")
	print(f"  Average Q̇c: {np.mean(qc_dot):.4f} W")
	print(f"\nHeat Equation Solution:")
	print(f"  Solution computed successfully")
	print(f"  Number of time steps: {len(sol.t)}")
	N_x = len(x_grid)
	print(f"  Final temperature at x=0 (Tc): {sol.y[0, -1]-273.15:.2f} °C")
	print(f"  Final temperature at x=L: {sol.y[N_x-1, -1]-273.15:.2f} °C")
	print(f"  Final hot side temperature (Th): {sol.y[N_x, -1]-273.15:.2f} °C")
	print(f"  Final ΔT (Th - Tc): {sol.y[N_x, -1] - sol.y[0, -1]:.2f} K")
	
	# Plot temperature as a function of time at x=0 and thermistor 0 data
	plot_temperature_vs_time(sol, timestamp, thermistor_0, voltage_interp,
	                         radius, alpha, K, R)


if __name__ == '__main__':
	main()

