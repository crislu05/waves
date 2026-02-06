import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path


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
    Heat added to cold plate: + Peltier pumping + 1/2 Joule heating + Conductive leak from hot side
    
    Note: Positive voltage (and thus positive current) should result in positive flux (input energy) at cold plate.
    
    Parameters:
    - alpha: Seebeck coefficient (V/K)
    - I: Current through Peltier device (A)
    - Tc: Cold side temperature (K)
    - R: Electrical resistance (Ohms)
    - K: Thermal conductance (W/K)
    - Th: Hot side temperature (K)
    """
    # Heat added to cold plate: 
    # + Peltier pumping (positive I pumps heat TO cold plate) + 1/2 Joule heating + Conductive leak from hot side
    qc = (alpha * I * Tc) + (0.5 * I**2 * R) + K * (Th - Tc)
    return qc


def calculate_qh(alpha, I, Th, R, K, Tc):
    """
    Calculate heat flow rate at hot side.
    Heat added to hot plate: - Peltier pumping + 1/2 Joule heating - Conductive leak to cold side
    
    Note: If positive current pumps heat TO cold plate, it removes heat FROM hot plate.
    
    Parameters:
    - alpha: Seebeck coefficient (V/K)
    - I: Current through Peltier device (A)
    - Th: Hot side temperature (K)
    - R: Electrical resistance (Ohms)
    - K: Thermal conductance (W/K)
    - Tc: Cold side temperature (K)
    """
    # Heat added to hot plate: 
    # - Peltier pumping (positive I removes heat FROM hot plate) + 1/2 Joule heating - Conductive leak to cold side
    qh = - (alpha * I * Th) + (0.5 * I**2 * R) - K * (Th - Tc)
    return qh


def solve_coupled_heat_pump(t_span, T_initial, voltage_func, params, rtol, atol, t_eval=None, method=None):
    """
    Solve coupled heat pump equations with 1D brass rod PDE.
    
    The system includes:
    - Tc, Th: Peltier cold and hot plate temperatures (ODEs)
    - T_brass: Array of temperatures along the brass rod (1D PDE)
    
    The differential equations are:
    - C_cold_plate * Ṫc = Qc_peltier + q_interface
    - C_hot_plate * Ṫh = Qh_peltier + h_hot * A_hot * (T_inf - Th)
    - ∂T_brass/∂t = α * ∂²T_brass/∂x²  (1D heat equation)
    
    Boundary conditions:
    - At x=0: -k_brass * (∂T/∂x)|_0 = q''
      where q'' = (T_brass[0] - Tc) / R''_grease is heat flux density (W/m²)
      and R''_grease = thickness_grease / k_grease (specific thermal resistance in m²·K/W)
    - At x=L: (∂T/∂x)|_L = 0 (insulated)
    
    where:
    - Qc_peltier = -αITc + (1/2)I²R + K(Th - Tc)
    - Qh_peltier = αITh + (1/2)I²R - K(Th - Tc)
    - q_interface = q'' * A_contact is total heat flux (W) for Peltier ODE
    - I(t) = (V(t) - α(T_h - T_c)) / R_elec
    - α = k_brass / (ρ_brass * c_brass) is thermal diffusivity
    
    Parameters:
    - t_span: Time span (t0, tf) in seconds
    - T_initial: Initial temperatures [Tc, Th, T_brass[0], ..., T_brass[N]] in Kelvin
    - voltage_func: Function V(t) returning voltage at time t
    - params: Dictionary containing all physical parameters
    - rtol: Relative tolerance for ODE solver
    - atol: Absolute tolerance for ODE solver
    - t_eval: Optional array of time points at which to evaluate solution.
              If provided, dense_output=False for efficiency.
    - method: Optional solver method ('Radau', 'BDF', 'RK45'). 
              Default is 'Radau' for stiff thermal problems.
    """
    def heat_pump_rhs(t, State, params):
        """Right-hand side of the coupled heat pump equations with 1D brass rod PDE."""
        # Unpack state: Peltier plates and the array of brass temperatures
        Tc = State[0]
        Th = State[1]
        T_brass = State[2:]  # Array of brass temperatures along the rod
        
        # 1. Peltier Current and Heat Pump Logic
        V = voltage_func(t)
        I = (V - params['alpha'] * (Th - Tc)) / params['R_elec']
        Qc_peltier = calculate_qc(params['alpha'], I, Tc, params['R_elec'], params['K_therm'], Th)
        Qh_peltier = calculate_qh(params['alpha'], I, Th, params['R_elec'], params['K_therm'], Tc)
        
        # 2. Coupling Flux through Grease at x=0
        # Boundary condition: -k_brass * (∂T/∂x)|_(x=0) = q''
        # where q'' = (T_brass[0] - Tc) / R''_grease is heat flux density (W/m²)
        # and R''_grease = thickness_grease / k_grease (specific thermal resistance in m²·K/W)
        R_grease_specific = params['thickness_grease'] / params['k_grease']  # m²·K/W
        q_double_prime = (T_brass[0] - Tc) / R_grease_specific  # W/m² (heat flux density)
        
        # Total heat flux for Peltier ODE: Q = q'' * A
        q_interface = q_double_prime * params['A_contact']  # W (total heat flux)
        
        # 3. Peltier Plate ODEs
        dTc_dt = (Qc_peltier + q_interface) / params['C_cold_plate']
        dTh_dt = (Qh_peltier + params['h_hot'] * params['A_hot'] * (params['T_inf'] - Th)) / params['C_hot_plate']
        
        # 4. Brass Rod PDE (Finite Difference)
        # 1D Heat Equation with Convection: ∂T/∂t = α * ∂²T/∂x² - (hP / (ρ * c_p * A)) * (T - T_∞)
        # where α = k_brass / (ρ_brass * c_brass) is thermal diffusivity
        # and h is convective heat transfer coefficient (different from h_hot)
        dT_brass_dt = np.zeros_like(T_brass)
        dx = params['L_brass'] / (len(T_brass) - 1)
        diff_coeff = params['k_brass'] / (params['rho_brass'] * params['c_brass'])  # Thermal diffusivity
        
        # Convection term coefficient: (hP / (ρ * c_p * A))
        # For a cylindrical rod: P = 2πr (perimeter), A = πr² (cross-sectional area)
        # So: hP / (ρ * c_p * A) = h * 2πr / (ρ * c_p * πr²) = 2h / (ρ * c_p * r)
        h = params.get('h', 0.0)  # Convective heat transfer coefficient for brass rod (to be determined)
        rho_cp = params['rho_brass'] * params['c_brass']  # ρ * c_p
        radius_brass = params.get('radius_brass', 0.015)  # m
        conv_coeff = (2 * h) / (rho_cp * radius_brass)  # Convection coefficient
        
        # Node 0 (Interface with grease at x=0)
        # Boundary condition: -k_brass * (∂T/∂x)|_0 = q''
        # Using ghost point: (T[1] - T_ghost) / (2*dx) = -q'' / k_brass
        # T_ghost = T[1] + 2*dx*q''/k_brass
        # Include convection term: -conv_coeff * (T[0] - T_∞)
        dT_brass_dt[0] = (2 * diff_coeff / dx**2) * (T_brass[1] - T_brass[0] - (q_double_prime * dx / params['k_brass'])) \
                        - conv_coeff * (T_brass[0] - params['T_inf'])
        
        # Interior Nodes (1 to N-2) - Vectorized for speed
        # Laplacian: ∂²T/∂x² ≈ (T[i+1] - 2*T[i] + T[i-1]) / dx²
        # Include convection term: -conv_coeff * (T[i] - T_∞)
        dT_brass_dt[1:-1] = diff_coeff * (T_brass[2:] - 2*T_brass[1:-1] + T_brass[:-2]) / dx**2 \
                           - conv_coeff * (T_brass[1:-1] - params['T_inf'])
        
        # Last Node (Convective Boundary at x=L)
        # Boundary condition: -k_brass * (∂T/∂x)|_(x=L) = h * (T[L] - T_∞)
        # Using ghost point method: (T[N+1] - T[N-1]) / (2*dx) = -h*(T[N] - T_∞) / k_brass
        # T[N+1] = T[N-1] - 2*dx*h*(T[N] - T_∞) / k_brass
        # Then: dT[N]/dt = diff_coeff * (T[N+1] - 2*T[N] + T[N-1]) / dx² - conv_coeff * (T[N] - T_∞)
        # Substituting T[N+1]: dT[N]/dt = diff_coeff * (2*T[N-1] - 2*T[N] - 2*dx*h*(T[N] - T_∞)/k_brass) / dx² - conv_coeff * (T[N] - T_∞)
        # Simplifying: dT[N]/dt = (2*diff_coeff/dx²) * (T[N-1] - T[N]) - (2*diff_coeff*h/(k_brass*dx)) * (T[N] - T_∞) - conv_coeff * (T[N] - T_∞)
        h_end = h  # Same convective coefficient for the end
        conv_boundary_coeff = (2 * diff_coeff * h_end) / (params['k_brass'] * dx)  # Convection at boundary
        dT_brass_dt[-1] = (2 * diff_coeff / dx**2) * (T_brass[-2] - T_brass[-1]) \
                         - conv_boundary_coeff * (T_brass[-1] - params['T_inf']) \
                         - conv_coeff * (T_brass[-1] - params['T_inf'])
        
        return np.concatenate([[dTc_dt, dTh_dt], dT_brass_dt])
    
    # Create wrapper function for solve_ivp (it expects signature (t, T))
    def rhs_wrapper(t, T):
        return heat_pump_rhs(t, T, params)
    
    # Solve the ODE system
    # Use 'Radau' or 'BDF' for stiff systems (thermal problems are typically stiff)
    # 'Radau' is generally faster for moderately stiff problems
    if method is None:
        solver_method = 'Radau'  # Default: 'Radau' for stiff thermal problems
    else:
        solver_method = method
    
    # If t_eval is provided, use it directly and disable dense_output for efficiency
    # This allows the integrator to optimize step size for the specific time points
    if t_eval is not None: 
        sol = solve_ivp(rhs_wrapper, t_span, T_initial,
                        method=solver_method, t_eval=t_eval, dense_output=False,
                        rtol=rtol, atol=atol)
    else:
        # Default behavior: use dense_output for interpolation later
        sol = solve_ivp(rhs_wrapper, t_span, T_initial,
                        method=solver_method, dense_output=True, rtol=rtol, atol=atol)
    
    return sol


def get_thermistor_positions():
    """
    Get thermistor positions dictionary.
    
    Returns:
    - Dictionary mapping thermistor_id to position in meters from x=0
    """
    return {
        0: 0.003,   # 3mm
        1: 0.008,   # 8mm
        2: 0.013,   # 13mm
        3: 0.018,   # 18mm
        4: 0.023,   # 23mm
        5: 0.028,   # 28mm
        6: 0.033,   # 33mm
        7: 0.038    # 38mm
    }


def load_h_values_from_cooling():
	"""
	Load h values from h_vs_voltage.csv.
	
	Returns:
	- Dictionary mapping fan voltage (int) to h value (float)
	"""
	h_vs_voltage_file = Path('data/cooling/h_vs_voltage.csv')
	h_dict = {}
	
	if h_vs_voltage_file.exists():
		try:
			h_data = pd.read_csv(h_vs_voltage_file)
			for _, row in h_data.iterrows():
				voltage = int(row['voltage'])
				# Use h_after_correction column (corrected h values)
				h = float(row['h_after_correction'])
				
				# Skip fan off (voltage = 0), only include fan voltages
				if voltage > 0:
					h_dict[voltage] = h
					print(f"  Loaded h for {voltage}V fan: {h:.2f} W/(m²·K)")
		except Exception as e:
			print(f"Warning: Could not load h values from h_vs_voltage.csv: {e}")
			import traceback
			traceback.print_exc()
	else:
		print(f"Warning: {h_vs_voltage_file} not found.")
		print(f"  Run cooling.py first to generate h values.")
	
	return h_dict


def plot_all_thermistors(sol, timestamp, thermistor_temperatures, x_grid, thermistor_positions, 
                         fan_voltage, h_value, save_path=None):
    """
    Plot all 7 thermistors (0-6) experimental vs model in one image.
    
    Parameters:
    - sol: Solution object from solve_ivp
    - timestamp: Time data from experimental measurements
    - thermistor_temperatures: Array of shape (n_time, n_thermistors) with experimental temperatures
    - x_grid: Spatial grid positions along the brass rod
    - thermistor_positions: Dictionary mapping thermistor_id to position in meters
    - fan_voltage: Fan voltage (for title)
    - h_value: Convective heat transfer coefficient used (for title)
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Use actual solution points
    t_eval = sol.t
    T_solution = sol.y
    T_brass_all = T_solution[2:, :]  # Shape: (N_nodes, len(t_eval))
    
    # Interpolate model solution to experimental time points
    T_brass_interp_time = np.zeros((len(thermistor_positions), len(timestamp)))
    for therm_id, x_pos in thermistor_positions.items():
        if therm_id >= 7:  # Only plot thermistors 0-6
            continue
        # Interpolate spatially at each model time point
        T_at_pos_model_times = np.zeros(len(t_eval))
        for i in range(len(t_eval)):
            T_brass_interp = interp1d(x_grid, T_brass_all[:, i], kind='linear',
                                     fill_value='extrapolate', bounds_error=False)
            T_at_pos_model_times[i] = T_brass_interp(x_pos)
        
        # Interpolate to experimental time points
        T_interp_to_exp = interp1d(t_eval, T_at_pos_model_times, kind='linear',
                                   fill_value='extrapolate', bounds_error=False)
        T_brass_interp_time[therm_id, :] = T_interp_to_exp(timestamp)
    
    # Convert to Celsius
    T_brass_interp_time_C = T_brass_interp_time - 273.15
    
    # Create figure with subplots for each thermistor (0-6, 7 thermistors)
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for therm_id in range(7):  # Thermistors 0-6
        ax = axes[therm_id]
        x_pos = thermistor_positions[therm_id]
        x_pos_mm = x_pos * 1000
        
        # Experimental data
        exp_data = thermistor_temperatures[:, therm_id]
        ax.plot(timestamp, exp_data, 'b-', linewidth=1.5, label='Experimental', alpha=0.7)
        
        # Model data
        model_data = T_brass_interp_time_C[therm_id, :]
        ax.plot(timestamp, model_data, 'r-', linewidth=1.5, label='Model', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Thermistor {therm_id} (x={x_pos_mm:.0f}mm)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Remove the 8th subplot (index 7)
    fig.delaxes(axes[7])
    
    # Add overall title
    fig.suptitle(f'Experimental vs Model: All Thermistors (Fan Voltage: {fan_voltage}V, h={h_value:.2f} W/(m²·K))',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_temperatures_vs_time(sol, timestamp, thermistor_0, x_grid, thermistor_x_pos, voltage_func, params, save_path=None):
    """
    Plot T_brass at thermistor position, thermistor_0 temperature, and Qc as a function of time.
    
    Parameters:
    - sol: Solution object from solve_ivp
    - timestamp: Original time data for reference
    - thermistor_0: Array of thermistor 0 temperatures from experimental data
    - x_grid: Spatial grid positions along the brass rod
    - thermistor_x_pos: x position of thermistor 0 in meters (e.g., 0.003 for 3mm)
    - voltage_func: Function V(t) returning voltage at time t
    - params: Dictionary containing physical parameters
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Use actual solution points to avoid interpolation artifacts
    # sol.t contains the actual time points where the solver evaluated the solution
    # sol.y contains the solution at those points: shape (2 + N_nodes, len(sol.t))
    t_eval = sol.t  # Use actual solver time points
    T_solution = sol.y  # Use actual solution values (no interpolation)
    
    # Extract temperatures
    Tc = T_solution[0, :]  # Cold plate temperature in Kelvin
    Th = T_solution[1, :]  # Hot plate temperature in Kelvin
    T_brass_all = T_solution[2:, :]  # Shape: (N_nodes, len(t_eval))
    
    # Interpolate to get exact temperature at thermistor position (3mm)
    T_brass_thermistor = np.zeros(len(t_eval))
    for i in range(len(t_eval)):
        # Interpolate spatially at each time point
        T_brass_interp = interp1d(x_grid, T_brass_all[:, i], kind='linear', 
                                  fill_value='extrapolate', bounds_error=False)
        T_brass_thermistor[i] = T_brass_interp(thermistor_x_pos)
    
    # Calculate Qc for each time point (vectorized for efficiency and numerical stability)
    V_eval = np.array([voltage_func(t) for t in t_eval])
    I_eval = (V_eval - params['alpha'] * (Th - Tc)) / params['R_elec']
    # Vectorized Qc calculation
    Qc = (params['alpha'] * I_eval * Tc) + (0.5 * I_eval**2 * params['R_elec']) + params['K_therm'] * (Th - Tc)
    
    # Apply light smoothing to reduce numerical oscillations (especially in negative flux region)
    # Use Savitzky-Golay filter to preserve signal shape while reducing high-frequency noise
    if len(Qc) > 11:  # Need enough points for smoothing
        window_length = min(11, (len(Qc)//10)*2+1)  # Odd number, at least 11 points
        if window_length >= 5:  # Minimum window size for polyorder=3
            Qc = savgol_filter(Qc, window_length=window_length, polyorder=3)
    
    # Convert to Celsius for plotting
    T_brass_thermistor_C = T_brass_thermistor - 273.15
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Temperature comparison
    axes[0].plot(t_eval, T_brass_thermistor_C, 'g-', linewidth=2, 
                 label=f'T_brass (Model, x={thermistor_x_pos*1000:.1f}mm)')
    axes[0].plot(timestamp, thermistor_0, 'b-', linewidth=2, label='Thermistor 0 (Experimental)', alpha=0.7)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0].set_title(f'Convective Model: Brass Temperature at x={thermistor_x_pos*1000:.1f}mm vs Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=11)
    
    # Plot 2: Heat flux at cold plate (Qc)
    axes[1].plot(t_eval, Qc, 'r-', linewidth=2, label='Qc (Cold plate heat flux)')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Heat Flux Qc (W)', fontsize=12)
    axes[1].set_title('Convective Model: Cold Plate Heat Flux vs Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def solve_and_plot_for_fan_voltage(filepath, fan_voltage, h_value, thermistor_positions):
    """
    Solve the model and generate plot for a specific fan voltage.
    
    Parameters:
    - filepath: Path to the CSV file for this fan voltage
    - fan_voltage: Fan voltage (int, e.g., 4, 6, 8, 10, 12)
    - h_value: Convective heat transfer coefficient for this fan voltage
    - thermistor_positions: Dictionary mapping thermistor_id to position in meters
    """
    # Load data
    timestamp, voltage, _, thermistor_temperatures = load_dataset(filepath)
    
    # Trim data to between 200 and 800 seconds
    mask = (timestamp >= 200) & (timestamp <= 800)
    timestamp = timestamp[mask]
    voltage = voltage[mask]
    thermistor_temperatures = thermistor_temperatures[mask, :]
    
    if len(timestamp) == 0:
        print(f"No data found between 200s and 800s for {filepath}. Skipping.")
        return
    
    # Extract initial temperatures from first data point of each thermistor
    # thermistor_temperatures shape: (n_time, n_thermistors)
    T_initial_thermistors = thermistor_temperatures[0, :] + 273.15  # Convert from Celsius to Kelvin
    # T_initial_thermistors[0] is thermistor 0, T_initial_thermistors[1] is thermistor 1, etc.
    
    # Get thermistor positions for interpolation
    thermistor_positions_dict = get_thermistor_positions()
    
    # Extract positions and temperatures for interpolation
    thermistor_x_positions = []
    thermistor_T_values = []
    for therm_id in range(thermistor_temperatures.shape[1]):  # For each thermistor column
        if therm_id in thermistor_positions_dict:
            thermistor_x_positions.append(thermistor_positions_dict[therm_id])
            thermistor_T_values.append(T_initial_thermistors[therm_id])
    
    # Convert to numpy arrays for interpolation
    thermistor_x_positions = np.array(thermistor_x_positions)
    thermistor_T_values = np.array(thermistor_T_values)
    
    # Thermoelectric device parameters (load from optimization results)
    import json
    from pathlib import Path
    params_file = Path('data/netflux/optimized_parameters.json')
    
    if params_file.exists():
        with open(params_file, 'r') as f:
            optimized_params = json.load(f)
        alpha = optimized_params.get('alpha', 0.05)  # V/K (Seebeck coefficient)
        K_therm = optimized_params.get('K_therm', 0.5)  # W/K (Thermal conductance of Peltier)
        R_elec = optimized_params.get('R_elec', 2.5)  # Ohm (Electrical resistance)
        print(f"Loaded optimized parameters from {params_file}:")
        print(f"  alpha: {alpha:.6f} V/K")
        print(f"  K_therm: {K_therm:.6f} W/K")
        print(f"  R_elec: {R_elec:.6f} Ω")
    else:
        # Default values if optimization file doesn't exist
        alpha = 0.05  # V/K (Seebeck coefficient)
        K_therm = 0.5  # W/K (Thermal conductance of Peltier)
        R_elec = 2.5  # Ohm (Electrical resistance)
        print(f"Warning: {params_file} not found. Using default parameters.")
        print(f"  Run optimse.py first to generate optimized parameters.")
    
    # Ambient temperature
    T_inf = 298.15  # K (25°C, ambient temperature)
    
    # Ceramic plate properties (Al2O3 aluminum oxide)
    rho_ceramic = 3970.0  # kg/m³ (Al2O3 density)
    c_ceramic = 775.0  # J/(kg·K) (Al2O3 specific heat capacity)
    
    # Ceramic plate geometry
    radius_plate = 0.015  # m (1.5 cm)
    thickness_plate = 0.002  # m (2mm typical)
    volume_plate = np.pi * radius_plate**2 * thickness_plate  # m³
    mass_plate = rho_ceramic * volume_plate  # kg
    C_cold_plate = mass_plate * c_ceramic  # J/K (heat capacity of cold ceramic plate, ~1 J/K)
    
    # Hot side thermal mass: Large finned heat sink with fan
    # The hot side is attached to a large finned heat sink, so C_hot is much larger than the ceramic plate alone
    C_hot_plate = 300.0  # J/K (heat capacity of hot side including heat sink, matches optimse.py)
    
    # Brass cylinder properties (from session6.py)
    rho_brass = 8520.0  # kg/m³ (Density of brass)
    c_brass = 380.0  # J/(kg·K) (Specific heat capacity of brass)
    k_brass = 109.0  # W/(m·K) (Thermal conductivity of brass)
    radius_brass = 0.015  # m (1.5 cm, same as ceramic plate)
    L_brass = 0.041  # m (length of brass cylinder)
    
    # Use the h value passed as parameter (from cooling.py results)
    h = h_value  # W/(m²·K) (from cooling fit results for this fan voltage)
    
    # Grease layer properties
    thickness_grease = 0.0001  # m (0.1 mm)
    k_grease = 1.0  # W/(m·K) (Thermal conductivity of grease)
    A_contact = np.pi * radius_plate**2  # m² (contact area between brass and cold plate)
    
    # Spatial discretization for brass rod
    N_nodes = 50  # Number of nodes along the brass rod
    x_grid = np.linspace(0, L_brass, N_nodes)  # Spatial grid
    dx = L_brass / (N_nodes - 1)  # Spatial step size
    
    # Initial temperatures: [Tc, Th, T_brass[0], ..., T_brass[N-1]]
    # Use first thermistor 0 measurement for Peltier plates
    T_cold_initial = T_initial_thermistors[0]  # K (from thermistor 0 first data point)
    T_hot_initial = T_initial_thermistors[0]  # K (from thermistor 0 first data point)
    
    # Interpolate thermistor temperatures to spatial grid for brass rod initial condition
    # Use linear interpolation to map thermistor temperatures to x_grid positions
    if len(thermistor_x_positions) > 1:
        # Interpolate using scipy's interp1d
        T_brass_interp_func = interp1d(thermistor_x_positions, thermistor_T_values, 
                                       kind='linear', fill_value='extrapolate', bounds_error=False)
        T_brass_array = T_brass_interp_func(x_grid)  # Interpolate to all grid points
    else:
        # Fallback: if only one thermistor, use uniform temperature
        T_brass_array = np.full(N_nodes, T_initial_thermistors[0])
    
    T_initial = np.concatenate([[T_cold_initial, T_hot_initial], T_brass_array])
    
    # Convective heat transfer parameters
    h_hot = 200  # W/(m²·K) (Convective heat transfer coefficient for hot plate with fan)
    
    # Hot side surface area: Finned heat sink dimensions 10 × 14 × 1 cm with 18 fins (2.5 × 14 cm each)
    heat_sink_length = 0.10  # m (10 cm)
    heat_sink_width = 0.14  # m (14 cm)
    heat_sink_height = 0.01  # m (1 cm)
    n_fins = 18  # Number of fins
    fin_length = 0.025  # m (2.5 cm)
    fin_width = 0.14  # m (14 cm, same as heat sink width)
    
    # Calculate total surface area including fins
    base_area = heat_sink_length * heat_sink_width  # m² (top/bottom base)
    # Each fin has 2 sides: 2 × (fin_length × fin_width)
    fin_area_per_fin = 2 * fin_length * fin_width  # m²
    total_fin_area = n_fins * fin_area_per_fin  # m²
    # Top surface (base with fins): base_area + total_fin_area
    # Bottom surface: base_area
    # Sides: 2 × (length × height) + 2 × (width × height)
    side_area = 2 * (heat_sink_length * heat_sink_height) + 2 * (heat_sink_width * heat_sink_height)  # m²
    A_hot = base_area + total_fin_area + base_area + side_area  # m² (total surface area)
    
    # Numerical parameters (match optimse.py for consistency)
    rtol = 1e-6  # Relative tolerance for ODE solver (higher precision)
    atol = 1e-8  # Absolute tolerance for ODE solver (higher precision)
    
    # Create interpolation function for voltage
    voltage_interp = interp1d(timestamp, voltage, kind='linear',
                              fill_value=(voltage[0], voltage[-1]), bounds_error=False)
    
    # Time span for solving
    t_span = (timestamp[0], timestamp[-1])
    
    # Create parameters dictionary
    params = {
        'alpha': alpha,
        'R_elec': R_elec,
        'K_therm': K_therm,
        'C_cold_plate': C_cold_plate,
        'C_hot_plate': C_hot_plate,
        'T_inf': T_inf,
        'h_hot': h_hot,
        'A_hot': A_hot,
        'thickness_grease': thickness_grease,
        'k_grease': k_grease,
        'A_contact': A_contact,
        'k_brass': k_brass,
        'rho_brass': rho_brass,
        'c_brass': c_brass,
        'L_brass': L_brass,
        'radius_brass': radius_brass,
        'h': h,  # Convective heat transfer coefficient for brass rod
        'N_nodes': N_nodes
    }
    
    # Solve coupled heat pump equations
    print("Solving coupled heat pump equations with 1D brass rod PDE...")
    print(f"\nDifferential Equations:")
    print(f"  C_cold_plate * Ṫc = Qc_peltier + q_interface")
    print(f"  C_hot_plate * Ṫh = Qh_peltier + h_hot * A_hot * (T_inf - Th)")
    print(f"  ∂T_brass/∂t = α * ∂²T_brass/∂x² - (hP / (ρ * c_p * A)) * (T - T_∞)  (1D heat equation with convection)")
    print(f"\nBoundary Conditions:")
    print(f"  At x=0: -k_brass * (∂T/∂x)|_0 = q''")
    print(f"    where q'' = (T_brass[0] - Tc) / R''_grease (heat flux density in W/m²)")
    print(f"    and R''_grease = thickness_grease / k_grease = {thickness_grease/k_grease:.6e} m²·K/W")
    print(f"  At x=L: -k_brass * (∂T/∂x)|_L = h * (T[L] - T_∞) (convective heat transfer)")
    print(f"\n  where q_interface = q'' * A_contact (total heat flux in W)")
    print(f"  and I(t) = (V(t) - α(T_h - T_c)) / R_elec")
    print(f"  and α = k_brass / (ρ_brass * c_brass) = thermal diffusivity")
    print(f"\nParameters:")
    print(f"  α (Seebeck coefficient): {alpha:.3f} V/K")
    print(f"  K_therm (Thermal conductance): {K_therm:.2f} W/K")
    print(f"  R_elec (Electrical resistance): {R_elec:.2f} Ω")
    print(f"\nThermal Masses:")
    print(f"  C_cold_plate: {C_cold_plate:.2f} J/K (ceramic plate)")
    print(f"  C_hot_plate: {C_hot_plate:.2f} J/K (heat sink with fan, 200-400 J/K range)")
    print(f"\nHot Side Heat Sink:")
    print(f"  Dimensions: {heat_sink_length*100:.0f} × {heat_sink_width*100:.0f} × {heat_sink_height*100:.0f} cm")
    print(f"  Number of fins: {n_fins}")
    print(f"  Fin dimensions: {fin_length*100:.1f} × {fin_width*100:.0f} cm each")
    print(f"  Surface area (A_hot): {A_hot:.4f} m² (including fins)")
    print(f"  Convective coefficient (h_hot): {h_hot:.0f} W/(m²·K) (with fan)")
    print(f"\nBrass Rod Properties:")
    print(f"  Length: {L_brass*100:.1f} cm")
    print(f"  Radius: {radius_brass*100:.1f} cm")
    print(f"  Thermal conductivity (k_brass): {k_brass:.1f} W/(m·K)")
    print(f"  Density (ρ_brass): {rho_brass:.0f} kg/m³")
    print(f"  Specific heat (c_brass): {c_brass:.0f} J/(kg·K)")
    print(f"  Thermal diffusivity (α): {k_brass/(rho_brass*c_brass):.2e} m²/s")
    print(f"  Convective coefficient (h): {h:.2f} W/(m²·K) (from cooling fit for {fan_voltage}V fan)")
    print(f"  Convection term coefficient: {2*h/(rho_brass*c_brass*radius_brass):.4f} 1/s")
    print(f"  Number of nodes: {N_nodes}")
    print(f"  Spatial step (dx): {dx*1000:.3f} mm")
    print(f"\nGrease Layer:")
    print(f"  Thickness: {thickness_grease*1000:.2f} mm")
    print(f"  Thermal conductivity: {k_grease:.1f} W/(m·K)")
    print(f"  R''_grease (specific): {thickness_grease/k_grease:.6e} m²·K/W")
    print(f"  Contact area (A_contact): {A_contact*1e6:.2f} mm²")
    print(f"\nInitial Conditions (from first data point of each thermistor):")
    print(f"  Initial Tc: {T_cold_initial-273.15:.2f} °C ({T_cold_initial:.2f} K)")
    print(f"  Initial Th: {T_hot_initial-273.15:.2f} °C ({T_hot_initial:.2f} K)")
    print(f"  Initial T_brass profile (interpolated from thermistors):")
    print(f"    T_brass[0] (x=0): {T_brass_array[0]-273.15:.2f} °C ({T_brass_array[0]:.2f} K)")
    x_mid = x_grid[N_nodes//2]
    print(f"    T_brass[L/2] (x={x_mid*1000:.1f}mm): {T_brass_array[N_nodes//2]-273.15:.2f} °C ({T_brass_array[N_nodes//2]:.2f} K)")
    print(f"    T_brass[L] (x={L_brass*1000:.0f}mm): {T_brass_array[-1]-273.15:.2f} °C ({T_brass_array[-1]:.2f} K)")
    print(f"  Thermistor initial temperatures:")
    for therm_id, x_pos in thermistor_positions_dict.items():
        if therm_id < len(T_initial_thermistors):
            print(f"    Thermistor {therm_id} (x={x_pos*1000:.0f}mm): {T_initial_thermistors[therm_id]-273.15:.2f} °C ({T_initial_thermistors[therm_id]:.2f} K)")
    
    sol = solve_coupled_heat_pump(t_span, T_initial, voltage_interp, params, rtol, atol, t_eval=None, method='Radau')
    
    # Print summary statistics
    print(f"\nSolution Summary:")
    print(f"  Time range: {timestamp[0]:.2f} s to {timestamp[-1]:.2f} s")
    print(f"  Duration: {timestamp[-1] - timestamp[0]:.2f} s")
    print(f"  Number of time steps: {len(sol.t)}")
    
    # Evaluate final temperatures
    t_final = timestamp[-1]
    T_final = sol.sol(t_final)
    Tc_final = T_final[0]
    Th_final = T_final[1]
    T_brass_final = T_final[2:]  # Array of brass temperatures
    T_brass_0_final = T_brass_final[0]  # Temperature at x=0 (interface)
    T_brass_L_final = T_brass_final[-1]  # Temperature at x=L (far end)
    
    print(f"  Final Tc: {Tc_final-273.15:.2f} °C ({Tc_final:.2f} K)")
    print(f"  Final Th: {Th_final-273.15:.2f} °C ({Th_final:.2f} K)")
    print(f"  Final T_brass[0] (x=0): {T_brass_0_final-273.15:.2f} °C ({T_brass_0_final:.2f} K)")
    print(f"  Final T_brass[L] (x=L): {T_brass_L_final-273.15:.2f} °C ({T_brass_L_final:.2f} K)")
    print(f"  Final ΔT (Th - Tc): {Th_final - Tc_final:.2f} K")
    print(f"  Final ΔT (T_brass[0] - Tc): {T_brass_0_final - Tc_final:.2f} K")
    
    # Calculate final heat flows
    V_final = voltage_interp(t_final)
    I_final = (V_final - params['alpha'] * (Th_final - Tc_final)) / params['R_elec']
    Qc_final = calculate_qc(params['alpha'], I_final, Tc_final, params['R_elec'], params['K_therm'], Th_final)
    Qh_final = calculate_qh(params['alpha'], I_final, Th_final, params['R_elec'], params['K_therm'], Tc_final)
    print(f"  Final Qc: {Qc_final:.4f} W")
    print(f"  Final Qh: {Qh_final:.4f} W")
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for convective plots
    convective_plots_dir = plots_dir / 'convective'
    convective_plots_dir.mkdir(exist_ok=True)
    
    # Generate plot filename
    plot_filename = f'all_thermistors_{fan_voltage}Vfan.png'
    plot_path = convective_plots_dir / plot_filename
    
    # Plot all thermistors (0-6) and save
    print(f"\nGenerating plot for {fan_voltage}V fan (h={h:.2f} W/(m²·K))...")
    plot_all_thermistors(sol, timestamp, thermistor_temperatures, x_grid, thermistor_positions,
                         fan_voltage, h_value, save_path=str(plot_path))
    print(f"✓ Completed {fan_voltage}V fan plot\n")


def main():
    """Main function to solve coupled heat pump equations for all fan voltages."""
    import json
    import re
    
    # Load h values from cooling results
    print("Loading h values from cooling fit results...")
    h_dict = load_h_values_from_cooling()
    
    if not h_dict:
        print("Error: No h values loaded. Please run cooling.py first.")
        return
    
    # Get thermistor positions
    thermistor_positions = get_thermistor_positions()
    
    # Find all fan voltage CSV files in data/fan folder
    fan_dir = Path('data/fan')
    if not fan_dir.exists():
        print(f"Error: {fan_dir} does not exist.")
        return
    
    # Find all CSV files matching the pattern: 7V_10s_*Vfan.csv
    csv_files = sorted(fan_dir.glob('7V_10s_*Vfan.csv'))
    
    if not csv_files:
        print(f"No fan voltage CSV files found in {fan_dir}")
        return
    
    print(f"\nFound {len(csv_files)} fan voltage files:")
    for f in csv_files:
        print(f"  {f.name}")
    
    # Process each file
    for filepath in csv_files:
        # Extract fan voltage from filename (e.g., "7V_10s_12Vfan.csv" -> 12)
        match = re.search(r'(\d+)Vfan\.csv', filepath.name)
        if match:
            fan_voltage = int(match.group(1))
            
            # Get corresponding h value
            if fan_voltage in h_dict:
                h_value = h_dict[fan_voltage]
                print(f"\n{'='*60}")
                print(f"Processing {fan_voltage}V fan (h={h_value:.2f} W/(m²·K))")
                print(f"{'='*60}")
                
                try:
                    solve_and_plot_for_fan_voltage(filepath, fan_voltage, h_value, thermistor_positions)
                except Exception as e:
                    print(f"Error processing {fan_voltage}V fan: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: No h value found for {fan_voltage}V fan. Skipping {filepath.name}")
        else:
            print(f"Warning: Could not extract fan voltage from {filepath.name}. Skipping.")
    
    print(f"\n{'='*60}")
    print("All fan voltage plots generated successfully!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
