"""
Parameter Optimization for Coupled Heat Pump ODE System
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

from heatpump import load_dataset, solve_coupled_heat_pump


def objective_function(params_vector, fixed_params, experimental_data):
    """Objective function: MSE between model and experimental data."""
    alpha, K_therm, R_elec, h_hot = params_vector
    
    # Fixed parameters
    rho_ceramic = fixed_params['rho_ceramic']
    c_ceramic = fixed_params['c_ceramic']
    radius_plate = fixed_params['radius_plate']
    thickness_plate = fixed_params['thickness_plate']
    C_hot_plate = fixed_params['C_hot_plate']
    thickness_grease = fixed_params['thickness_grease']
    k_grease = fixed_params['k_grease']
    
    # Calculate derived parameters
    volume_plate = np.pi * radius_plate**2 * thickness_plate
    mass_plate = rho_ceramic * volume_plate
    C_cold_plate = mass_plate * c_ceramic
    A_contact = np.pi * radius_plate**2
    
    params = {
        'alpha': alpha, 'R_elec': R_elec, 'K_therm': K_therm,
        'C_cold_plate': C_cold_plate, 'C_hot_plate': C_hot_plate,
        'T_inf': fixed_params['T_inf'], 'h_hot': h_hot,
        'A_hot': fixed_params['A_hot'], 'thickness_grease': thickness_grease,
        'k_grease': k_grease, 'A_contact': A_contact,
        'k_brass': fixed_params['k_brass'], 'rho_brass': fixed_params['rho_brass'],
        'c_brass': fixed_params['c_brass'], 'L_brass': fixed_params['L_brass'],
        'N_nodes': fixed_params['N_nodes']
    }
    
    try:
        # Subsample for efficiency
        max_points = 200 if len(experimental_data['timestamp']) > 1000 else len(experimental_data['timestamp'])
        subsample_factor = max(1, len(experimental_data['timestamp']) // max_points)
        timestamp_eval = experimental_data['timestamp'][::subsample_factor]
        thermistor_0_eval = experimental_data['thermistor_0'][::subsample_factor]
        
        sol = solve_coupled_heat_pump(
            experimental_data['t_span'], experimental_data['T_initial'],
            experimental_data['voltage_interp'], params,
            fixed_params['rtol'], fixed_params['atol'],
            t_eval=timestamp_eval, method='Radau'
        )
        
        if not sol.success or sol.y is None or sol.y.shape[0] < 3:
            return 1e10
        
        T_brass_all = sol.y[2:, :]
        thermistor_node_idx = fixed_params['thermistor_node_idx']
        T_brass_thermistor = T_brass_all[thermistor_node_idx, :]
        T_brass_thermistor_C = T_brass_thermistor - 273.15
        
        return np.mean((T_brass_thermistor_C - thermistor_0_eval)**2)
    except Exception:
        return 1e10


def setup_optimization():
    """Set up optimization problem."""
    # Load data
    timestamp, voltage, _, thermistor_temperatures = load_dataset('data/session6/brass_7V_10s.csv')
    thermistor_0 = thermistor_temperatures[:, 0]
    T_initial_from_data = thermistor_0[0] + 273.15
    
    # Fixed physical parameters
    T_inf = 298.15
    rho_brass, c_brass, k_brass = 8520.0, 380.0, 109.0
    L_brass = 0.041
    
    # Heat sink geometry
    heat_sink_length, heat_sink_width, heat_sink_height = 0.10, 0.14, 0.01
    n_fins, fin_length, fin_width = 18, 0.025, 0.14
    base_area = heat_sink_length * heat_sink_width
    A_hot = base_area + n_fins * 2 * fin_length * fin_width + base_area + \
            2 * (heat_sink_length * heat_sink_height) + 2 * (heat_sink_width * heat_sink_height)
    
    # Spatial grid with node at thermistor position (3mm)
    thermistor_x_pos = 0.003
    N_nodes = 20
    n_before = max(1, int((thermistor_x_pos / L_brass) * (N_nodes - 1)))
    n_after = N_nodes - n_before - 1
    
    x_grid_before = np.linspace(0, thermistor_x_pos, n_before + 1)
    x_grid_after = np.linspace(thermistor_x_pos, L_brass, n_after + 1)[1:] if n_after > 0 else np.array([])
    x_grid = np.concatenate([x_grid_before, x_grid_after])
    thermistor_node_idx = np.argmin(np.abs(x_grid - thermistor_x_pos))
    
    # Initial conditions
    T_initial = np.concatenate([
        [T_initial_from_data, T_initial_from_data],
        np.full(N_nodes, T_initial_from_data)
    ])
    
    voltage_interp = interp1d(timestamp, voltage, kind='linear',
                              fill_value=(voltage[0], voltage[-1]), bounds_error=False)
    
    fixed_params = {
        'T_inf': T_inf, 'k_brass': k_brass, 'rho_brass': rho_brass,
        'c_brass': c_brass, 'L_brass': L_brass, 'N_nodes': N_nodes,
        'x_grid': x_grid, 'thermistor_node_idx': thermistor_node_idx,
        'A_hot': A_hot, 'rtol': 1e-3, 'atol': 1e-5,
        'rho_ceramic': 3970.0, 'c_ceramic': 775.0,
        'radius_plate': 0.015, 'thickness_plate': 0.002,
        'C_hot_plate': 300.0, 'thickness_grease': 0.0001, 'k_grease': 1.0
    }
    
    experimental_data = {
        'timestamp': timestamp, 'thermistor_0': thermistor_0,
        'voltage_interp': voltage_interp, 't_span': (timestamp[0], timestamp[-1]),
        'T_initial': T_initial
    }
    
    initial_params = np.array([0.05, 0.5, 2.5, 200.0])  # alpha, K_therm, R_elec, h_hot
    bounds = [(0.01, 0.10), (0.1, 2.0), (1.0, 5.0), (20.0, 300.0)]
    
    return initial_params, bounds, fixed_params, experimental_data


def optimize_parameters(method='L-BFGS-B', quick_test=False):
    """Optimize parameters."""
    initial_params, bounds, fixed_params, experimental_data = setup_optimization()
    
    param_names = ['alpha', 'K_therm', 'R_elec', 'h_hot']
    print(f"\nInitial Parameters: {dict(zip(param_names, initial_params))}")
    print(f"Bounds: {dict(zip(param_names, bounds))}\n")
    
    def objective_wrapper(params):
        return objective_function(params, fixed_params, experimental_data)
    
    maxiter = 3 if quick_test else 50
    result = minimize(objective_wrapper, initial_params, method=method,
                     bounds=bounds, options={'maxiter': maxiter, 'disp': True})
    
    print(f"\nOptimization completed!")
    print(f"  Success: {result.success}")
    print(f"  Final MSE: {result.fun:.6f} °C², RMSE: {np.sqrt(result.fun):.4f} °C")
    print(f"  Iterations: {result.nit}, Function evaluations: {result.nfev}")
    print(f"  Optimized Parameters: {dict(zip(param_names, result.x))}")
    
    return result, dict(zip(param_names, result.x))


def plot_optimization_results(result, fixed_params, experimental_data):
    """Plot comparison between optimized model and experimental data."""
    alpha, K_therm, R_elec, h_hot = result.x
    
    # Calculate derived parameters
    radius_plate = fixed_params['radius_plate']
    thickness_plate = fixed_params['thickness_plate']
    rho_ceramic = fixed_params['rho_ceramic']
    c_ceramic = fixed_params['c_ceramic']
    volume_plate = np.pi * radius_plate**2 * thickness_plate
    C_cold_plate = rho_ceramic * volume_plate * c_ceramic
    
    params = {
        'alpha': alpha, 'R_elec': R_elec, 'K_therm': K_therm,
        'C_cold_plate': C_cold_plate, 'C_hot_plate': fixed_params['C_hot_plate'],
        'T_inf': fixed_params['T_inf'], 'h_hot': h_hot,
        'A_hot': fixed_params['A_hot'], 'thickness_grease': fixed_params['thickness_grease'],
        'k_grease': fixed_params['k_grease'], 'A_contact': np.pi * radius_plate**2,
        'k_brass': fixed_params['k_brass'], 'rho_brass': fixed_params['rho_brass'],
        'c_brass': fixed_params['c_brass'], 'L_brass': fixed_params['L_brass'],
        'N_nodes': fixed_params['N_nodes']
    }
    
    sol = solve_coupled_heat_pump(
        experimental_data['t_span'], experimental_data['T_initial'],
        experimental_data['voltage_interp'], params,
        fixed_params['rtol'], fixed_params['atol'],
        t_eval=None, method='Radau'
    )
    
    if hasattr(sol, 'sol') and sol.sol is not None:
        T_brass_all = sol.sol(experimental_data['timestamp'])[2:, :]
    else:
        sol_full = solve_coupled_heat_pump(
            experimental_data['t_span'], experimental_data['T_initial'],
            experimental_data['voltage_interp'], params,
            fixed_params['rtol'], fixed_params['atol'],
            t_eval=experimental_data['timestamp'], method='Radau'
        )
        T_brass_all = sol_full.y[2:, :]
    
    T_brass_thermistor_C = T_brass_all[fixed_params['thermistor_node_idx'], :] - 273.15
    residuals = T_brass_thermistor_C - experimental_data['thermistor_0']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(experimental_data['timestamp'], experimental_data['thermistor_0'],
                 'b-', linewidth=2, label='Experimental', alpha=0.7)
    axes[0].plot(experimental_data['timestamp'], T_brass_thermistor_C,
                 'r--', linewidth=2, label='Optimized Model')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title(f'Optimized Model vs Experimental (RMSE: {np.sqrt(result.fun):.4f} °C)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(experimental_data['timestamp'], residuals, 'g-', linewidth=1.5)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Residual (°C)')
    axes[1].set_title('Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = Path('plots/optimization/optimization_results.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.close()


def main():
    """Main function."""
    print("=" * 70)
    print("Parameter Optimization for Coupled Heat Pump ODE System")
    print("=" * 70)
    
    result, optimized_params = optimize_parameters(method='L-BFGS-B', quick_test=False)
    
    _, _, fixed_params, experimental_data = setup_optimization()
    plot_optimization_results(result, fixed_params, experimental_data)
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
