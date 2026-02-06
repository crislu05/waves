import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json

# Import functions from convection.py
from convection import (
    load_dataset,
    calculate_qc,
    calculate_qh,
    solve_coupled_heat_pump,
    get_thermistor_positions
)

# Import functions from transient.py for exponential fitting
from transient import exponential_model, fit_transient_single_series

def setup_parameters():
    """Set up physical parameters for the model."""
    # Thermoelectric device parameters (load from optimization results)
    params_file = Path('data/netflux/optimized_parameters.json')
    
    if params_file.exists():
        with open(params_file, 'r') as f:
            optimized_params = json.load(f)
        alpha = optimized_params.get('alpha', 0.05)
        K_therm = optimized_params.get('K_therm', 0.5)
        R_elec = optimized_params.get('R_elec', 2.5)
    else:
        alpha = 0.05
        K_therm = 0.5
        R_elec = 2.5
    
    # Ambient temperature
    T_inf = 298.15  # K
    
    # Ceramic plate properties
    rho_ceramic = 3970.0  # kg/m³
    c_ceramic = 775.0  # J/(kg·K)
    radius_plate = 0.015  # m
    thickness_plate = 0.002  # m
    volume_plate = np.pi * radius_plate**2 * thickness_plate
    mass_plate = rho_ceramic * volume_plate
    C_cold_plate = mass_plate * c_ceramic
    C_hot_plate = 300.0  # J/K
    
    # Brass cylinder properties
    rho_brass = 8520.0  # kg/m³
    c_brass = 380.0  # J/(kg·K)
    k_brass = 109.0  # W/(m·K)
    radius_brass = 0.015  # m
    L_brass = 0.041  # m
    
    # Grease layer properties
    thickness_grease = 0.0001  # m
    k_grease = 1.0  # W/(m·K)
    A_contact = np.pi * radius_plate**2
    
    # Hot side parameters
    h_hot = 200  # W/(m²·K)
    heat_sink_length = 0.10  # m
    heat_sink_width = 0.14  # m
    heat_sink_height = 0.01  # m
    n_fins = 18
    fin_length = 0.025  # m
    fin_width = 0.14  # m
    base_area = heat_sink_length * heat_sink_width
    fin_area_per_fin = 2 * fin_length * fin_width
    total_fin_area = n_fins * fin_area_per_fin
    side_area = 2 * (heat_sink_length * heat_sink_height) + 2 * (heat_sink_width * heat_sink_height)
    A_hot = base_area + total_fin_area + base_area + side_area
    
    # Spatial discretization
    N_nodes = 50
    x_grid = np.linspace(0, L_brass, N_nodes)
    
    return {
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
        'N_nodes': N_nodes,
        'x_grid': x_grid
    }


def solve_with_fixed_h(filepath, h_value):
    """
    Solve the PDE system with fixed h value and return temperature data.
    
    Parameters:
    - filepath: Path to CSV file
    - h_value: Fixed h value in W/(m²·K)
    
    Returns:
    - T_brass_3d_C: Temperature array in Celsius, shape (n_positions, n_times)
    - timestamp: Time array
    - x_grid: Position array
    """
    # Load data
    timestamp, voltage, _, thermistor_temperatures = load_dataset(filepath)
    
    # Trim data to between 200 and 800 seconds
    mask = (timestamp >= 200) & (timestamp <= 800)
    timestamp = timestamp[mask]
    voltage = voltage[mask]
    thermistor_temperatures = thermistor_temperatures[mask, :]
    
    if len(timestamp) == 0:
        print(f"No data found between 200s and 800s for {filepath}.")
        return None, None, None
    
    # Set up parameters
    params = setup_parameters()
    params['h'] = h_value  # Set fixed h value
    x_grid = params['x_grid']
    N_nodes = params['N_nodes']
    
    # Extract initial temperatures from first data point of each thermistor
    T_initial_thermistors = thermistor_temperatures[0, :] + 273.15  # Convert to Kelvin
    
    # Get thermistor positions for interpolation
    thermistor_positions_dict = get_thermistor_positions()
    thermistor_x_positions = []
    thermistor_T_values = []
    for therm_id in range(thermistor_temperatures.shape[1]):
        if therm_id in thermistor_positions_dict:
            thermistor_x_positions.append(thermistor_positions_dict[therm_id])
            thermistor_T_values.append(T_initial_thermistors[therm_id])
    
    thermistor_x_positions = np.array(thermistor_x_positions)
    thermistor_T_values = np.array(thermistor_T_values)
    
    # Set up initial conditions
    T_cold_initial = T_initial_thermistors[0]
    T_hot_initial = T_initial_thermistors[0]
    
    # Interpolate thermistor temperatures to spatial grid
    if len(thermistor_x_positions) > 1:
        T_brass_interp_func = interp1d(thermistor_x_positions, thermistor_T_values,
                                      kind='linear', fill_value='extrapolate', bounds_error=False)
        T_brass_array = T_brass_interp_func(x_grid)
    else:
        T_brass_array = np.full(N_nodes, T_initial_thermistors[0])
    
    T_initial = np.concatenate([[T_cold_initial, T_hot_initial], T_brass_array])
    
    # Create interpolation function for voltage
    voltage_interp = interp1d(timestamp, voltage, kind='linear',
                              fill_value=(voltage[0], voltage[-1]), bounds_error=False)
    
    # Time span
    t_span = (timestamp[0], timestamp[-1])
    rtol = 1e-6
    atol = 1e-8
    
    # Solve the PDE system with fixed h
    try:
        sol = solve_coupled_heat_pump(t_span, T_initial, voltage_interp, params,
                                     rtol, atol, t_eval=timestamp, method='Radau')
    except Exception as e:
        print(f"Error solving PDE: {e}")
        return None, None, None
    
    # Extract temperature data for all time points
    # When t_eval is provided, sol.y contains the solution at those time points
    # sol.y shape: (n_states, n_times) where n_states = 2 (Tc, Th) + N_nodes (T_brass)
    T_brass_3d = sol.y[2:, :]  # Extract brass temperatures (skip Tc and Th), shape: (N_nodes, n_times)
    
    # Convert to Celsius
    T_brass_3d_C = T_brass_3d - 273.15
    
    return T_brass_3d_C, timestamp, x_grid


def main():
    """Main function to generate two 2D heatmap plots in the same image with shared temperature scale."""
    # Use a representative file (e.g., 12V fan)
    fan_dir = Path('data/fan')
    filepath = fan_dir / '7V_10s_12Vfan.csv'
    
    if not filepath.exists():
        print(f"Error: {filepath} not found.")
        return
    
    # h values: min = still air (11.2), max = max fan (44.3 from 10V fan)
    h_min = 11.2  # Still air
    h_max = 44.3  # Maximum fan (from h_vs_voltage.csv)
    
    # Solve for both h values
    print("Solving PDE system with h = 11.2 W/(m²·K) (Still Air)...")
    T_brass_min, timestamp, x_grid = solve_with_fixed_h(filepath, h_min)
    
    print("Solving PDE system with h = 44.3 W/(m²·K) (Max Fan)...")
    T_brass_max, _, _ = solve_with_fixed_h(filepath, h_max)
    
    if T_brass_min is None or T_brass_max is None:
        print("Error: Failed to solve PDE systems.")
        return
    
    # Find global min and max temperature for consistent color scale
    T_min_global = min(T_brass_min.min(), T_brass_max.min())
    T_max_global = max(T_brass_min.max(), T_brass_max.max())
    
    # Convert x_grid to mm for plotting
    x_grid_mm = x_grid * 1000
    
    # Create figure with two subplots side by side, leaving space for colorbar on the right
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Minimum h (Still Air) - 2D heatmap
    im1 = ax1.pcolormesh(timestamp, x_grid_mm, T_brass_min,
                        cmap='coolwarm', shading='auto',
                        vmin=T_min_global, vmax=T_max_global)
    ax1.set_xlabel('Time (s)', fontsize=20)
    ax1.set_ylabel('Position along Brass Rod (mm)', fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_title(f'Still Air (h = {h_min:.1f} W/(m²·K))', fontsize=18, fontweight='bold')
    ax1.set_aspect('auto')
    
    # Plot 2: Maximum h (Max Fan) - 2D heatmap
    im2 = ax2.pcolormesh(timestamp, x_grid_mm, T_brass_max,
                        cmap='coolwarm', shading='auto',
                        vmin=T_min_global, vmax=T_max_global)
    ax2.set_xlabel('', fontsize=20)  # Remove x-axis label for right graph
    ax2.set_ylabel('', fontsize=20)  # Remove y-axis label for right graph
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_title(f'7V fan (h = {h_max:.1f} W/(m²·K))', fontsize=18, fontweight='bold')
    ax2.set_aspect('auto')
    
    # Adjust subplot layout to make room for colorbar on the right
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Leave 8% space on the right for colorbar
    
    # Add shared colorbar to the right of the plots
    cbar = fig.colorbar(im1, ax=[ax1, ax2], label='Temperature (°C)', 
                       shrink=0.8, aspect=20, pad=0.02)
    cbar.ax.set_ylabel('Temperature (°C)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Save plot
    save_path = Path('plots/convective/temperature_heatmap_comparison.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 2D heatmap comparison plot saved to: {save_path}")
    plt.close()


def solve_steady_state_with_h(filepath, h_value):
    """
    Solve the PDE system with fixed h value and determine steady state by exponential fitting.
    
    Uses experimental voltage profile and fits exponential decay to determine T_inf (steady state).
    
    Parameters:
    - filepath: Path to CSV file
    - h_value: Fixed h value in W/(m²·K)
    
    Returns:
    - T_steady_profile: Steady state temperature profile along rod in Celsius, shape (N_nodes,)
    """
    # Solve the PDE system with experimental voltage (not constant)
    T_brass_3d_C, timestamp, x_grid = solve_with_fixed_h(filepath, h_value)
    
    if T_brass_3d_C is None:
        return None
    
    # T_brass_3d_C shape: (N_nodes, n_times)
    # For each position along the rod, fit exponential to get T_inf (steady state)
    N_nodes = T_brass_3d_C.shape[0]
    T_steady_profile = np.zeros(N_nodes)
    
    for i in range(N_nodes):
        T_time_series = T_brass_3d_C[i, :]  # Temperature at position i over time
        # Fit exponential: T = T_inf + A * exp(-t / tau)
        T_inf, A, tau, tau_unc = fit_transient_single_series(timestamp, T_time_series)
        if not np.isnan(T_inf):
            T_steady_profile[i] = T_inf
        else:
            # Fallback: use final temperature if fit fails
            T_steady_profile[i] = T_time_series[-1]
    
    return T_steady_profile


def plot_steady_state_3d(filepath):
    """
    Generate a 3D plot of steady state temperature vs position and convective coefficient h.
    
    Parameters:
    - filepath: Path to CSV file to use for initial conditions and voltage
    """
    # Range of h values (from still air to extended range)
    h_min = 11.2  # Still air
    h_max = 100.0  # Extended range up to 100 W/(m²·K)
    n_h_points = 120  # Number of h values to solve for (maintains resolution ~0.74 W/(m²·K) step)
    h_values = np.linspace(h_min, h_max, n_h_points)
    
    print(f"Solving steady state for {n_h_points} h values from {h_min:.1f} to {h_max:.1f} W/(m²·K)...")
    
    # Get spatial grid from first solve (all solves use same grid)
    params = setup_parameters()
    x_grid = params['x_grid']
    x_grid_mm = x_grid * 1000  # Convert to mm for plotting
    
    T_steady_profiles = []
    successful_h = []
    
    for i, h_val in enumerate(h_values):
        print(f"  Progress: {i+1}/{n_h_points} (h = {h_val:.2f} W/(m²·K))")
        T_steady_profile = solve_steady_state_with_h(filepath, h_val)
        if T_steady_profile is not None:
            T_steady_profiles.append(T_steady_profile)
            successful_h.append(h_val)
    
    if len(T_steady_profiles) == 0:
        print("Error: No successful solutions.")
        return
    
    # Convert to numpy arrays
    h_array = np.array(successful_h)
    T_profiles_array = np.array(T_steady_profiles)  # Shape: (n_h, n_positions)
    
    # Save data to CSV
    save_steady_state_data_to_csv(h_array, x_grid_mm, T_profiles_array)
    
    # Use shared plotting functions from convective_coeff_plot
    from convective_coeff_plot import plot_steady_state_3d_from_data, plot_steady_state_2d_middle
    
    # Generate plots using shared functions
    plot_steady_state_3d_from_data(h_array, x_grid_mm, T_profiles_array)
    plot_steady_state_2d_middle(h_array, T_profiles_array, x_grid)


def save_convective_comparison_data_to_csv(filepath, h_value, save_path=None):
    """
    Save temperature comparison data (model vs experimental) for a given h value to CSV.
    
    Parameters:
    - filepath: Path to CSV file for this fan voltage
    - h_value: Convective heat transfer coefficient (W/(m²·K))
    - save_path: Path to save the CSV file (if None, uses default path based on h value)
    
    Returns:
    - save_path: Path where CSV was saved
    """
    # Solve the PDE system
    T_brass_3d_C, timestamp, x_grid = solve_with_fixed_h(filepath, h_value)
    
    if T_brass_3d_C is None:
        print(f"  Warning: Failed to solve PDE for h={h_value:.2f}, skipping CSV save")
        return None
    
    # Load experimental data
    timestamp_exp, _, _, thermistor_temperatures_exp = load_dataset(filepath)
    
    # Trim experimental data to same range as model (200-800 seconds)
    mask = (timestamp_exp >= 200) & (timestamp_exp <= 800)
    timestamp_trimmed = timestamp_exp[mask]
    thermistor_temperatures_trimmed = thermistor_temperatures_exp[mask, :]
    
    # Get thermistor positions
    thermistor_positions_dict = get_thermistor_positions()
    
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp_trimmed)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for saving
    timestamp_plot = timestamp_trimmed[start_idx:end_idx]
    
    # Process each thermistor
    thermistor_results = {}
    for therm_id in sorted(thermistor_positions_dict.keys()):
        x_pos = thermistor_positions_dict[therm_id]
        
        # Interpolate model temperature at this thermistor's position
        T_model_at_pos = []
        for t_idx in range(T_brass_3d_C.shape[1]):
            T_profile = T_brass_3d_C[:, t_idx]
            interp_func = interp1d(x_grid, T_profile, kind='linear',
                                 fill_value='extrapolate', bounds_error=False)
            T_model_at_pos.append(interp_func(x_pos))
        T_model_at_pos = np.array(T_model_at_pos)
        
        # Filter to middle 90%
        T_model_middle90 = T_model_at_pos[start_idx:end_idx]
        
        # Get experimental data for this thermistor (middle 90%)
        T_exp_middle90 = thermistor_temperatures_trimmed[start_idx:end_idx, therm_id]
        
        thermistor_results[therm_id] = {
            'T_model_C': T_model_middle90,
            'T_exp_C': T_exp_middle90,
            'x_pos_mm': x_pos * 1000
        }
    
    # Create DataFrame with time and all thermistor data
    data_dict = {'Time (s)': timestamp_plot}
    
    # Add columns for each thermistor (sorted by ID)
    for therm_id in sorted(thermistor_results.keys()):
        result = thermistor_results[therm_id]
        data_dict[f'T_model_{therm_id}_x{result["x_pos_mm"]:.1f}mm (°C)'] = result['T_model_C']
        data_dict[f'T_exp_{therm_id}_x{result["x_pos_mm"]:.1f}mm (°C)'] = result['T_exp_C']
    
    df = pd.DataFrame(data_dict)
    
    # Save to CSV
    if save_path is None:
        # Create filename based on h value
        h_str = f"{h_value:.1f}".replace('.', 'p')  # Replace . with p for filename
        save_path = Path(f'data/convective/temperature_comparison_h{h_str}.csv')
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Temperature comparison data saved to: {save_path}")
    print(f"    Saved {len(df)} time points with data for {len(thermistor_results)} thermistors")
    
    return save_path


def save_steady_state_data_to_csv(h_array, x_grid_mm, T_profiles_array):
    """
    Save steady state temperature data to CSV file.
    
    Parameters:
    - h_array: Array of h values (W/(m²·K))
    - x_grid_mm: Position array in mm
    - T_profiles_array: Temperature profiles array, shape (n_h, n_positions) in °C
    """
    # Create long format data: each row is (h, x, temperature)
    data_rows = []
    for i, h_val in enumerate(h_array):
        for j, x_pos in enumerate(x_grid_mm):
            temp = T_profiles_array[i, j]
            data_rows.append({
                'h_W_per_m2K': h_val,
                'x_mm': x_pos,
                'temperature_C': temp
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    save_path = Path('data/convective/steady_state_3d_data.csv')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ Steady state data saved to: {save_path}")
    print(f"  Shape: {len(data_rows)} rows (h: {len(h_array)} values, x: {len(x_grid_mm)} positions)")




def plot_thermistor_temperatures_vs_time(filepath, h_value):
    """
    Generate plots of model temperature vs time at each thermistor position.
    
    Parameters:
    - filepath: Path to CSV file
    - h_value: Fixed h value in W/(m²·K)
    """
    # Solve the PDE system
    print(f"Solving PDE system with h = {h_value:.2f} W/(m²·K) for thermistor plots...")
    T_brass_3d_C, timestamp, x_grid = solve_with_fixed_h(filepath, h_value)
    
    if T_brass_3d_C is None:
        print("Error: Failed to solve PDE system.")
        return
    
    # Get thermistor positions
    thermistor_positions_dict = get_thermistor_positions()
    
    # Select 5 thermistors (0, 1, 2, 3, 4)
    thermistor_ids = [0, 1, 2, 3, 4]
    
    # Interpolate temperatures at thermistor positions
    # T_brass_3d_C shape: (N_nodes, n_times)
    # x_grid: positions of nodes
    # We need to interpolate to get temperatures at thermistor positions
    
    T_thermistors = {}
    fit_results = {}
    for therm_id in thermistor_ids:
        if therm_id in thermistor_positions_dict:
            x_therm = thermistor_positions_dict[therm_id]
            # Interpolate temperature at this position for all time points
            T_at_therm = []
            for t_idx in range(T_brass_3d_C.shape[1]):
                T_profile = T_brass_3d_C[:, t_idx]  # Temperature profile at time t
                interp_func = interp1d(x_grid, T_profile, kind='linear', 
                                     fill_value='extrapolate', bounds_error=False)
                T_at_therm.append(interp_func(x_therm))
            T_at_therm = np.array(T_at_therm)
            T_thermistors[therm_id] = T_at_therm
            
            # Fit exponential: T = T_inf + A * exp(-t / tau)
            T_inf, A, tau, tau_unc = fit_transient_single_series(timestamp, T_at_therm)
            fit_results[therm_id] = {
                'T_inf': T_inf,
                'A': A,
                'tau': tau,
                'tau_unc': tau_unc
            }
    
    # Create subplots: 5 plots in a column
    fig, axes = plt.subplots(5, 1, figsize=(10, 14))
    
    for idx, therm_id in enumerate(thermistor_ids):
        ax = axes[idx]
        if therm_id in T_thermistors:
            x_therm = thermistor_positions_dict[therm_id]
            T_data = T_thermistors[therm_id]
            
            # Plot data
            ax.plot(timestamp, T_data, linewidth=2, color='#1976D2', label='Model')
            
            # Plot exponential fit if available
            if therm_id in fit_results:
                fit = fit_results[therm_id]
                if not (np.isnan(fit['T_inf']) or np.isnan(fit['A']) or np.isnan(fit['tau'])):
                    # Generate smooth fit curve
                    t_fit = np.linspace(timestamp[0], timestamp[-1], 200)
                    T_fit = exponential_model(t_fit, fit['T_inf'], fit['A'], fit['tau'])
                    ax.plot(t_fit, T_fit, '--', linewidth=2, color='#E74C3C', 
                           label=f'Fit: T_inf = {fit["T_inf"]:.2f}°C, τ = {fit["tau"]:.1f}s')
            
            ax.set_xlabel('Time (s)', fontsize=11)
            ax.set_ylabel('Temperature (°C)', fontsize=11)
            ax.set_title(f'Thermistor {therm_id} at x = {x_therm*1000:.1f} mm', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(timestamp[0], timestamp[-1])
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(f'plots/convective/thermistor_temperatures_vs_time_h{h_value:.1f}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Thermistor temperature plots saved to: {save_path}")
    plt.close()


def plot_example_thermistor_with_fit(filepath, h_value, thermistor_id=0):
    """
    Generate an example plot showing model temperature vs time with exponential fit for a single thermistor.
    
    Parameters:
    - filepath: Path to CSV file
    - h_value: Fixed h value in W/(m²·K)
    - thermistor_id: Thermistor ID to plot (default: 0)
    """
    # Solve the PDE system
    print(f"Solving PDE system with h = {h_value:.2f} W/(m²·K) for example plot...")
    T_brass_3d_C, timestamp, x_grid = solve_with_fixed_h(filepath, h_value)
    
    if T_brass_3d_C is None:
        print("Error: Failed to solve PDE system.")
        return
    
    # Get thermistor positions
    thermistor_positions_dict = get_thermistor_positions()
    
    if thermistor_id not in thermistor_positions_dict:
        print(f"Error: Thermistor {thermistor_id} not found.")
        return
    
    x_therm = thermistor_positions_dict[thermistor_id]
    
    # Interpolate temperature at this position for all time points
    T_at_therm = []
    for t_idx in range(T_brass_3d_C.shape[1]):
        T_profile = T_brass_3d_C[:, t_idx]  # Temperature profile at time t
        interp_func = interp1d(x_grid, T_profile, kind='linear', 
                             fill_value='extrapolate', bounds_error=False)
        T_at_therm.append(interp_func(x_therm))
    T_at_therm = np.array(T_at_therm)
    
    # Fit exponential: T = T_inf + A * exp(-t / tau)
    T_inf, A, tau, tau_unc = fit_transient_single_series(timestamp, T_at_therm)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data
    ax.plot(timestamp, T_at_therm, linewidth=2, color='#1976D2', label='Model Temperature', alpha=0.8)
    
    # Plot exponential fit if available
    if not (np.isnan(T_inf) or np.isnan(A) or np.isnan(tau)):
        # Generate smooth fit curve
        t_fit = np.linspace(timestamp[0], timestamp[-1], 200)
        T_fit = exponential_model(t_fit, T_inf, A, tau)
        ax.plot(t_fit, T_fit, '--', linewidth=2.5, color='#E74C3C', 
               label=f'Exponential Fit: $T_\\infty$ = {T_inf:.2f}°C, $\\tau$ = {tau:.1f} ± {tau_unc:.1f}s')
        print(f"\nFit results for Thermistor {thermistor_id}:")
        print(f"  T_inf (steady state) = {T_inf:.2f} °C")
        print(f"  A (amplitude) = {A:.2f} °C")
        print(f"  τ (time constant) = {tau:.1f} ± {tau_unc:.1f} s")
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Example: Thermistor {thermistor_id} at x = {x_therm*1000:.1f} mm\n'
                f'Exponential Fit: $T(t) = T_\\infty + A \\exp(-t/\\tau)$', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(timestamp[0], timestamp[-1])
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    save_path = Path(f'plots/convective/example_thermistor_{thermistor_id}_fit_h{h_value:.1f}.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Example plot saved to: {save_path}")
    plt.close()


def save_all_convective_comparison_data():
    """
    Save temperature comparison data for all fan voltages to CSV files.
    """
    # Load h values from cooling data
    h_vs_voltage_file = Path('data/cooling/h_vs_voltage.csv')
    h_dict = {}
    
    if h_vs_voltage_file.exists():
        h_data = pd.read_csv(h_vs_voltage_file)
        for _, row in h_data.iterrows():
            voltage = int(row['voltage'])
            h = float(row['h_after_correction'])
            h_dict[voltage] = h
    
    # Find all fan voltage CSV files
    fan_dir = Path('data/fan')
    csv_files = list(fan_dir.glob('*Vfan.csv'))
    
    print("Saving temperature comparison data for all fan voltages...")
    for csv_file in sorted(csv_files):
        # Extract voltage from filename
        import re
        match = re.search(r'(\d+)Vfan', csv_file.name)
        if match:
            voltage = int(match.group(1))
            if voltage in h_dict:
                h_value = h_dict[voltage]
                print(f"\nProcessing {csv_file.name} (voltage={voltage}V, h={h_value:.2f} W/(m²·K))...")
                save_convective_comparison_data_to_csv(csv_file, h_value)
            else:
                print(f"  Warning: No h value found for {voltage}V fan, skipping")
        else:
            print(f"  Warning: Could not extract voltage from {csv_file.name}, skipping")


if __name__ == '__main__':
    main()
    # Also generate 3D steady state plot
    fan_dir = Path('data/fan')
    filepath = fan_dir / '7V_10s_12Vfan.csv'
    if filepath.exists():
        plot_steady_state_3d(filepath)
        # Generate example plot for thermistor 0
        h_representative = 11.2  # Still air
        print("\n" + "="*60)
        print("Generating example plot with exponential fit...")
        print("="*60)
        plot_example_thermistor_with_fit(filepath, h_representative, thermistor_id=0)
        # Generate all thermistor temperature plots for a representative h value
        plot_thermistor_temperatures_vs_time(filepath, h_representative)
        
        # Save comparison data for all fan voltages
        print("\n" + "="*60)
        print("Saving temperature comparison data for all fan voltages...")
        print("="*60)
        save_all_convective_comparison_data()
    else:
        print(f"Warning: {filepath} not found. Skipping steady state plot.")


