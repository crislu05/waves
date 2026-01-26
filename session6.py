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


def calculate_qc_dot(voltage, current, alpha, K, T_cold, delta_T):
	"""
	Calculate net heat flow rate at cold side: Q̇c = αITc - (1/2)IV + (1/2)αIΔT - KΔT
	
	Parameters:
	- voltage: Voltage across Peltier device (V)
	- current: Current through Peltier device (A)
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance (W/K)
	- T_cold: Cold side temperature in Kelvin
	- delta_T: Temperature difference across Peltier (Thot - Tcold) in Kelvin
	"""
	# Calculate Q̇c components
	# Q̇c = αITc - (1/2)IV + (1/2)αIΔT - KΔT
	peltier_pumping = alpha * current * T_cold
	electrical_power_cold = -0.5 * current * voltage
	seebeck_correction = 0.5 * alpha * current * delta_T
	conduction_leak = -K * delta_T
	
	qc_dot = peltier_pumping + electrical_power_cold + seebeck_correction + conduction_leak
	
	return qc_dot


def calculate_heat_flux_at_boundary(voltage, current, T_cold, alpha, K, delta_T, R, A):
	"""
	Calculate heat flux at x=0 boundary: -k * (∂T/∂x)|_(x=0) = (1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
	
	Parameters:
	- voltage: Voltage across Peltier device (V)
	- current: Current through Peltier device (A)
	- T_cold: Cold side temperature in Kelvin
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance (W/K)
	- delta_T: Temperature difference across Peltier (Th - Tc) in Kelvin
	- R: Electrical resistance (Ohms)
	- A: Cross-sectional area (m²)
	"""
	# Calculate heat flux: (1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
	# Note: Th = Tc + delta_T
	heat_flux = (1.0 / A) * (alpha * current * T_cold - 0.5 * current**2 * R - K * delta_T)
	
	return heat_flux


def solve_1d_heat_equation(t_span, x_grid, T_initial, voltage_func, current_func, 
                           L, radius, alpha, K, T_cold, R, D_brass, k_brass, 
                           h, rho, c, T_infinity, rtol, atol):
	"""
	Solve 1D heat equation with convective heat loss: ∂T/∂t = D * ∂²T/∂x² - (hP/(ρcA))[T - T∞]
	
	Boundary conditions:
	- At x=0: k * (∂T/∂x)|_(x=0) = -(1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
	- At x=L: -k * (∂T/∂x)|_(x=L) = h * (T(L) - T∞) (convective heat exchange)
	
	ΔT is calculated dynamically from: V = IR + αΔT, so ΔT = (V - IR) / α
	
	Parameters:
	- t_span: Time span (t0, tf) in seconds
	- x_grid: Spatial grid points (array)
	- T_initial: Initial temperature distribution (array, same length as x_grid)
	- voltage_func: Function voltage(t) returning voltage at time t
	- current_func: Function current(t) returning current at time t
	- L: Length of the cylinder (m)
	- radius: Radius of the cylinder (m)
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance of Peltier (W/K)
	- T_cold: Cold side temperature in Kelvin (initial/reference)
	- R: Electrical resistance (Ohms)
	- D_brass: Thermal diffusivity of brass (m²/s)
	- k_brass: Thermal conductivity of brass (W/(m·K))
	- h: Convective heat transfer coefficient (W/(m²·K))
	- rho: Density of brass (kg/m³)
	- c: Specific heat capacity of brass (J/(kg·K))
	- T_infinity: Ambient/environment temperature (K)
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
	
	def heat_equation_rhs(t, T):
		"""Right-hand side of the heat equation."""
		dTdt = np.zeros_like(T)
		
		# Get current voltage and current at time t
		V = voltage_func(t)
		I = current_func(t)
		
		# Calculate delta_T dynamically from: V = IR + αΔT
		# So: ΔT = (V - IR) / α
		delta_T = (V - I * R) / alpha
		
		# Calculate heat flux at x=0 boundary
		# Boundary condition: k * (∂T/∂x)|_(x=0) = -(1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
		# Convention: Negative flux = heat INPUT (flowing INTO the system)
		#             Positive flux = heat OUTPUT (flowing OUT of the system)
		# Using Tc = T[0] (temperature at x=0)
		Tc = T[0]
		# Calculate heat flux: negative means heat input, positive means heat output
		# For Peltier cooling, we want to remove heat (positive flux = heat out)
		# The equation gives: (1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
		# We need to negate this to match our convention where negative = heat in
		heat_flux = -(1.0 / A) * (alpha * I * Tc - 0.5 * I**2 * R - K * delta_T)
		
		# Interior points: ∂T/∂t = D * ∂²T/∂x² - (hP/(ρcA))[T - T∞]
		# Using central difference for spatial derivative
		for i in range(1, N-1):
			diffusion_term = D_brass * (T[i+1] - 2*T[i] + T[i-1]) / dx**2
			convective_term = -convective_coeff * (T[i] - T_infinity)
			dTdt[i] = diffusion_term + convective_term
		
		# Boundary condition at x=0: k * (∂T/∂x)|_(x=0) = heat_flux
		# Rearranging: (∂T/∂x)|_(x=0) = heat_flux / k
		# Using ghost point method: introduce T[-1] such that
		# (∂T/∂x)|_(x=0) ≈ (T[1] - T[-1]) / (2*dx) = heat_flux / k
		# So: T[1] - T[-1] = 2*dx*heat_flux / k
		# Therefore: T[-1] = T[1] - 2*dx*heat_flux / k
		# For the heat equation at x=0: dT[0]/dt = D * (T[1] - 2*T[0] + T[-1]) / dx² - (hP/(ρcA))[T[0] - T∞]
		T_ghost_0 = T[1] + 2 * dx * heat_flux / k_brass
		diffusion_term_0 = D_brass * (T[1] - 2*T[0] + T_ghost_0) / dx**2
		convective_term_0 = -convective_coeff * (T[0] - T_infinity)
		dTdt[0] = diffusion_term_0 + convective_term_0
		
		# Boundary condition at x=L: -k * (∂T/∂x)|_(x=L) = h * (T(L) - T∞)
		# Rearranging: (∂T/∂x)|_(x=L) = -h/k * (T(L) - T∞)
		# Using ghost point method: introduce T[N] such that
		# (∂T/∂x)|_(x=L) ≈ (T[N] - T[N-2]) / (2*dx) = -h/k * (T[N-1] - T∞)
		# So: T[N] - T[N-2] = -2*dx * h/k * (T[N-1] - T∞)
		# Therefore: T[N] = T[N-2] - 2*dx * h/k_brass * (T[N-1] - T_infinity)
		# For the heat equation at x=L: dT[N-1]/dt = D * (T[N] - 2*T[N-1] + T[N-2]) / dx² - (hP/(ρcA))[T[N-1] - T∞]
		T_ghost_L = T[N-2] - 2 * dx * h / k_brass * (T[N-1] - T_infinity)
		diffusion_term_L = D_brass * (T_ghost_L - 2*T[N-1] + T[N-2]) / dx**2
		convective_term_L = -convective_coeff * (T[N-1] - T_infinity)
		dTdt[N-1] = diffusion_term_L + convective_term_L
		
		return dTdt
	
	# Solve the ODE system
	sol = solve_ivp(heat_equation_rhs, t_span, T_initial, 
	                method='RK45', dense_output=True, rtol=rtol, atol=atol)
	
	return sol


def plot_temperature_vs_time(sol, timestamp, thermistor_0_data, voltage_func, current_func,
                            radius, alpha, K, R):
	"""
	Plot temperature as a function of time at x=0, thermistor 0 data, and heat flux Q at x=0.
	
	Parameters:
	- sol: Solution object from solve_ivp
	- timestamp: Original time data for reference
	- thermistor_0_data: Thermistor 0 temperature data (array)
	- voltage_func: Function voltage(t) returning voltage at time t
	- current_func: Function current(t) returning current at time t
	- radius: Radius of cylinder (m)
	- alpha: Seebeck coefficient (V/K)
	- K: Thermal conductance of Peltier (W/K)
	- R: Electrical resistance (Ohms)
	"""
	# Evaluate solution at evenly spaced time points
	t_eval = np.linspace(timestamp[0], timestamp[-1], min(500, len(timestamp)))
	T_solution = sol.sol(t_eval)  # Shape: (N_x, len(t_eval))
	
	# Convert to Celsius and get temperature at x=0
	T_at_x0_C = T_solution[0, :] - 273.15
	T_at_x0_K = T_solution[0, :]  # Keep in Kelvin for heat flux calculation
	
	# Calculate heat flux Q at x=0 for each time point
	A = np.pi * radius**2
	Q_at_x0 = np.zeros_like(t_eval)
	
	for i, t in enumerate(t_eval):
		V = voltage_func(t)
		I = current_func(t)
		Tc = T_at_x0_K[i]
		delta_T = (V - I * R) / alpha
		# Heat flux: Q = -(1/A) * [αITc - (1/2)I²R - K(Th - Tc)]
		Q_at_x0[i] = -(1.0 / A) * (alpha * I * Tc - 0.5 * I**2 * R - K * delta_T)
	
	# Create figure with three subplots
	fig, axes = plt.subplots(3, 1, figsize=(10, 12))
	
	# Plot 1: Temperature at x=0 from heat equation solution
	axes[0].plot(t_eval, T_at_x0_C, 'b-', linewidth=2, label='x=0 (cold end, Peltier side)')
	axes[0].set_xlabel('Time (s)', fontsize=12)
	axes[0].set_ylabel('Temperature (°C)', fontsize=12)
	axes[0].set_title('Temperature vs Time at x=0 (Heat Equation Solution)', fontsize=14, fontweight='bold')
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
	R = 1.0  # Ohm (Electrical resistance, can be estimated from V/I)
	T_cold = 298.15  # K (25°C, cold side temperature, initial/reference)
	# Note: delta_T is calculated dynamically from V = IR + αΔT
	
	# Brass cylinder parameters
	radius = 0.015  # m (1.5 cm)
	L = 0.1  # m (assumed length, adjust as needed)
	
	# Brass material properties
	D_brass = 3.4e-5  # m²/s (Thermal diffusivity of brass)
	k_brass = 109.0  # W/(m·K) (Thermal conductivity of brass)
	rho_brass = 8520.0  # kg/m³ (Density of brass)
	c_brass = 380.0  # J/(kg·K) (Specific heat capacity of brass)
	
	# Convective heat transfer parameters
	h = 10.0  # W/(m²·K) (Convective heat transfer coefficient, typical for natural convection)
	T_infinity = 298.15  # K (Ambient/environment temperature, 25°C)
	
	# Numerical parameters
	N_x = 100  # Number of spatial points
	rtol = 1e-6  # Relative tolerance for ODE solver
	atol = 1e-8  # Absolute tolerance for ODE solver
	
	# Calculate Q̇c (using average delta_T for statistics)
	# delta_T is calculated from V = IR + αΔT, so ΔT = (V - IR) / α
	delta_T_avg = np.mean((voltage - current * R) / alpha)
	qc_dot = calculate_qc_dot(voltage, current, alpha, K, T_cold, delta_T_avg)
	
	# Set up spatial grid for 1D heat equation
	x_grid = np.linspace(0, L, N_x)
	
	# Initial temperature distribution (uniform at ambient)
	T_initial = np.ones(N_x) * T_cold
	
	# Create interpolation functions for voltage and current
	from scipy.interpolate import interp1d
	voltage_interp = interp1d(timestamp, voltage, kind='linear', 
	                          fill_value=(voltage[0], voltage[-1]), bounds_error=False)
	current_interp = interp1d(timestamp, current, kind='linear',
	                          fill_value=(current[0], current[-1]), bounds_error=False)
	
	# Time span for solving
	t_span = (timestamp[0], timestamp[-1])
	
	# Solve 1D heat equation
	print("Solving 1D heat equation...")
	sol = solve_1d_heat_equation(t_span, x_grid, T_initial, voltage_interp, current_interp,
	                             L, radius, alpha, K, T_cold, R, D_brass, k_brass,
	                             h, rho_brass, c_brass, T_infinity, rtol, atol)
	
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
	print(f"  Final temperature at x=0: {sol.y[0, -1]-273.15:.2f} °C")
	print(f"  Final temperature at x=L: {sol.y[-1, -1]-273.15:.2f} °C")
	
	# Plot temperature as a function of time at x=0 and thermistor 0 data
	plot_temperature_vs_time(sol, timestamp, thermistor_0)


if __name__ == '__main__':
	main()

