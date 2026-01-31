import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
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
        # 1D Heat Equation: ∂T/∂t = α * ∂²T/∂x²
        # where α = k_brass / (ρ_brass * c_brass) is thermal diffusivity
        dT_brass_dt = np.zeros_like(T_brass)
        dx = params['L_brass'] / (len(T_brass) - 1)
        diff_coeff = params['k_brass'] / (params['rho_brass'] * params['c_brass'])  # Thermal diffusivity
        
        # Node 0 (Interface with grease at x=0)
        # Boundary condition: -k_brass * (∂T/∂x)|_0 = q''
        # Using ghost point: (T[1] - T_ghost) / (2*dx) = -q'' / k_brass
        # T_ghost = T[1] + 2*dx*q''/k_brass
        dT_brass_dt[0] = (2 * diff_coeff / dx**2) * (T_brass[1] - T_brass[0] - (q_double_prime * dx / params['k_brass']))
        
        # Interior Nodes (1 to N-2) - Vectorized for speed
        # Laplacian: ∂²T/∂x² ≈ (T[i+1] - 2*T[i] + T[i-1]) / dx²
        dT_brass_dt[1:-1] = diff_coeff * (T_brass[2:] - 2*T_brass[1:-1] + T_brass[:-2]) / dx**2
        
        # Last Node (Insulated Boundary at x=L)
        # Boundary condition: (∂T/∂x)|_(x=L) = 0 (adiabatic)
        # No heat flux out, so T[N+1] is effectively T[N-1]
        dT_brass_dt[-1] = (2 * diff_coeff / dx**2) * (T_brass[-2] - T_brass[-1])
        
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
    # Evaluate solution at evenly spaced time points
    t_eval = np.linspace(timestamp[0], timestamp[-1], min(500, len(timestamp)))
    T_solution = sol.sol(t_eval)  # Shape: (2 + N_nodes, len(t_eval)) - [Tc, Th, T_brass[0], ..., T_brass[N-1]]
    
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
    
    # Calculate Qc for each time point
    Qc = np.zeros_like(t_eval)
    for i, t in enumerate(t_eval):
        V = voltage_func(t)
        I = (V - params['alpha'] * (Th[i] - Tc[i])) / params['R_elec']
        Qc[i] = calculate_qc(params['alpha'], I, Tc[i], params['R_elec'], params['K_therm'], Th[i])
    
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
    axes[0].set_title(f'Brass Temperature at x={thermistor_x_pos*1000:.1f}mm vs Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=11)
    
    # Plot 2: Heat flux at cold plate (Qc)
    axes[1].plot(t_eval, Qc, 'r-', linewidth=2, label='Qc (Cold plate heat flux)')
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Heat Flux Qc (W)', fontsize=12)
    axes[1].set_title('Cold Plate Heat Flux vs Time', fontsize=14, fontweight='bold')
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


def main():
    """Main function to solve coupled heat pump equations."""
    # Load data (same as session6.py)
    filepath = 'data/session6/brass_7V_10s.csv'
    timestamp, voltage, _, thermistor_temperatures = load_dataset(filepath)
    
    # Extract thermistor_0 data and get initial temperature
    thermistor_0 = thermistor_temperatures[:, 0]  # First column contains thermistor 0 data
    T_initial_from_data = thermistor_0[0] + 273.15  # Convert from Celsius to Kelvin
    
    # Thermoelectric device parameters
    alpha = 0.05  # V/K (Seebeck coefficient)
    K_therm = 0.5  # W/K (Thermal conductance of Peltier)
    R_elec = 2.5 # Ohm (Electrical resistance)
    
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
    C_hot_plate = 300.0  # J/K (heat capacity of hot side including heat sink, 200-400 J/K range)
    
    # Brass cylinder properties (from session6.py)
    rho_brass = 8520.0  # kg/m³ (Density of brass)
    c_brass = 380.0  # J/(kg·K) (Specific heat capacity of brass) d
    k_brass = 109.0  # W/(m·K) (Thermal conductivity of brass)
    radius_brass = 0.015  # m (1.5 cm, same as ceramic plate)
    L_brass = 0.041  # m (length of brass cylinder)
    
    # Grease layer properties
    thickness_grease = 0.0001  # m (0.1 mm)
    k_grease = 1.0  # W/(m·K) (Thermal conductivity of grease)
    A_contact = np.pi * radius_plate**2  # m² (contact area between brass and cold plate)
    
    # Spatial discretization for brass rod
    N_nodes = 50  # Number of nodes along the brass rod
    x_grid = np.linspace(0, L_brass, N_nodes)  # Spatial grid
    dx = L_brass / (N_nodes - 1)  # Spatial step size
    
    # Initial temperatures: [Tc, Th, T_brass[0], ..., T_brass[N-1]]
    # Use first thermistor 0 measurement as initial temperature
    T_cold_initial = T_initial_from_data  # K (from thermistor 0 first data point)
    T_hot_initial = T_initial_from_data  # K (from thermistor 0 first data point)
    T_brass_initial = T_initial_from_data  # K (from thermistor 0 first data point)
    T_brass_array = np.full(N_nodes, T_brass_initial)  # Initial temperature profile (uniform)
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
    
    # Numerical parameters
    rtol = 1e-6  # Relative tolerance for ODE solver
    atol = 1e-8  # Absolute tolerance for ODE solver
    
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
        'N_nodes': N_nodes
    }
    
    # Solve coupled heat pump equations
    print("Solving coupled heat pump equations with 1D brass rod PDE...")
    print(f"\nDifferential Equations:")
    print(f"  C_cold_plate * Ṫc = Qc_peltier + q_interface")
    print(f"  C_hot_plate * Ṫh = Qh_peltier + h_hot * A_hot * (T_inf - Th)")
    print(f"  ∂T_brass/∂t = α * ∂²T_brass/∂x²  (1D heat equation)")
    print(f"\nBoundary Conditions:")
    print(f"  At x=0: -k_brass * (∂T/∂x)|_0 = q''")
    print(f"    where q'' = (T_brass[0] - Tc) / R''_grease (heat flux density in W/m²)")
    print(f"    and R''_grease = thickness_grease / k_grease = {thickness_grease/k_grease:.6e} m²·K/W")
    print(f"  At x=L: (∂T/∂x)|_L = 0 (insulated)")
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
    print(f"  Number of nodes: {N_nodes}")
    print(f"  Spatial step (dx): {dx*1000:.3f} mm")
    print(f"\nGrease Layer:")
    print(f"  Thickness: {thickness_grease*1000:.2f} mm")
    print(f"  Thermal conductivity: {k_grease:.1f} W/(m·K)")
    print(f"  R''_grease (specific): {thickness_grease/k_grease:.6e} m²·K/W")
    print(f"  Contact area (A_contact): {A_contact*1e6:.2f} mm²")
    print(f"\nInitial Conditions (from thermistor 0 first data point):")
    print(f"  Initial Tc: {T_cold_initial-273.15:.2f} °C ({T_cold_initial:.2f} K)")
    print(f"  Initial Th: {T_hot_initial-273.15:.2f} °C ({T_hot_initial:.2f} K)")
    print(f"  Initial T_brass (uniform): {T_brass_initial-273.15:.2f} °C ({T_brass_initial:.2f} K)")
    
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
    
    # Thermistor 0 position: 3mm from x=0
    thermistor_x_pos = 0.003  # m (3mm)
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for heatpump plots
    heatpump_plots_dir = plots_dir / 'heatpump'
    heatpump_plots_dir.mkdir(exist_ok=True)
    
    # Generate plot filename
    plot_filename = 'heatpump_analysis.png'
    plot_path = heatpump_plots_dir / plot_filename
    
    # Plot results and save
    plot_temperatures_vs_time(sol, timestamp, thermistor_0, x_grid, thermistor_x_pos, 
                              voltage_interp, params, save_path=str(plot_path))
    
    # Create data directory for CSV
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create subdirectory for netflux data
    netflux_dir = data_dir / 'netflux'
    netflux_dir.mkdir(exist_ok=True)
    
    # Save data to CSV
    # Evaluate solution at evenly spaced time points for CSV export
    t_eval_csv = np.linspace(timestamp[0], timestamp[-1], min(1000, len(timestamp)))
    T_solution_csv = sol.sol(t_eval_csv)
    
    # Extract temperatures
    Tc_csv = T_solution_csv[0, :]  # Cold plate temperature in Kelvin
    Th_csv = T_solution_csv[1, :]  # Hot plate temperature in Kelvin
    
    # Calculate Qc, Qh, current, and power for each time point
    Qc_csv = np.zeros_like(t_eval_csv)
    Qh_csv = np.zeros_like(t_eval_csv)
    V_csv = np.zeros_like(t_eval_csv)
    I_csv = np.zeros_like(t_eval_csv)
    P_csv = np.zeros_like(t_eval_csv)
    
    for i, t in enumerate(t_eval_csv):
        V = voltage_interp(t)
        V_csv[i] = V
        I = (V - params['alpha'] * (Th_csv[i] - Tc_csv[i])) / params['R_elec']
        I_csv[i] = I
        P = V * I  # Power = Voltage × Current
        P_csv[i] = P
        Qc_csv[i] = calculate_qc(params['alpha'], I, Tc_csv[i], params['R_elec'], params['K_therm'], Th_csv[i])
        Qh_csv[i] = calculate_qh(params['alpha'], I, Th_csv[i], params['R_elec'], params['K_therm'], Tc_csv[i])
    
    # Create DataFrame with only required columns
    data_dict = {
        'Time (s)': t_eval_csv,
        'Tc (K)': Tc_csv,
        'Th (K)': Th_csv,
        'Qc (W)': Qc_csv,
        'Qh (W)': Qh_csv,
        'Voltage (V)': V_csv,
        'Current (A)': I_csv,
        'Power (W)': P_csv
    }
    
    df = pd.DataFrame(data_dict)
    
    # Save to CSV in data/netflux directory
    csv_filename = 'heatpump_data.csv'
    csv_path = netflux_dir / csv_filename
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")


if __name__ == '__main__':
    main()
