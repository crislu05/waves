"""
Parameter Optimization for Coupled Heat Pump ODE System
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

from heatpump import load_dataset, solve_coupled_heat_pump


def objective_function(params_vector, fixed_params, experimental_data):
    """Objective function: MSE between model and experimental data."""
    alpha, K_therm, R_elec = params_vector
    
    # Fixed parameters
    rho_ceramic = fixed_params['rho_ceramic']
    c_ceramic = fixed_params['c_ceramic']
    radius_plate = fixed_params['radius_plate']
    thickness_plate = fixed_params['thickness_plate']
    C_hot_plate = fixed_params['C_hot_plate']
    thickness_grease = fixed_params['thickness_grease']
    k_grease = fixed_params['k_grease']
    h_hot = fixed_params['h_hot']  # Fixed at 200.0
    
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
        # Extract middle 90% of dataset (remove first 5% and last 5%)
        # Use stored indices if available, otherwise calculate
        if 'middle90_start_idx' in experimental_data and 'middle90_end_idx' in experimental_data:
            start_idx = experimental_data['middle90_start_idx']
            end_idx = experimental_data['middle90_end_idx']
        else:
            # Fallback: calculate if not stored
            n_total = len(experimental_data['timestamp'])
            start_idx = int(0.05 * n_total)
            end_idx = int(0.95 * n_total)
        
        # Get middle 90% data for optimization
        timestamp_middle90 = experimental_data['timestamp'][start_idx:end_idx]
        thermistor_0_middle90 = experimental_data['thermistor_0'][start_idx:end_idx]
        
        # Subsample for efficiency
        max_points = 200 if len(timestamp_middle90) > 1000 else len(timestamp_middle90)
        subsample_factor = max(1, len(timestamp_middle90) // max_points)
        timestamp_eval = timestamp_middle90[::subsample_factor]
        thermistor_0_eval = thermistor_0_middle90[::subsample_factor]
        
        # Solve over full time span for proper physics, but evaluate at middle 90% points
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
        
        # For sinusoidal/oscillatory data with transients, use a combination metric:
        # 1. MSE (captures amplitude and overall fit) - only on middle 90%
        mse = np.mean((T_brass_thermistor_C - thermistor_0_eval)**2)
        
        # 2. Correlation coefficient (captures phase alignment and shape similarity)
        # Remove mean to focus on oscillatory behavior
        model_centered = T_brass_thermistor_C - np.mean(T_brass_thermistor_C)
        exp_centered = thermistor_0_eval - np.mean(thermistor_0_eval)
        if np.std(model_centered) > 1e-6 and np.std(exp_centered) > 1e-6:
            correlation = np.corrcoef(model_centered, exp_centered)[0, 1]
            # Convert correlation (0-1) to error metric (0 = perfect, 1 = worst)
            correlation_error = 1.0 - correlation
        else:
            correlation_error = 1.0
        
        # 3. Weighted combination: 70% MSE, 30% correlation error
        # This balances amplitude accuracy with phase/shape matching
        combined_error = 0.7 * mse + 0.3 * correlation_error * np.var(thermistor_0_eval)
        
        return combined_error
    except Exception:
        return 1e10


def setup_optimization():
    """Set up optimization problem."""
    # Load data
    timestamp, voltage, _, thermistor_temperatures = load_dataset('data/session6/brass_7V_10s.csv')
    thermistor_0 = thermistor_temperatures[:, 0]
    T_initial_from_data = thermistor_0[0] + 273.15
    
    # Extract middle 90% of dataset for optimization (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    print(f"Dataset size: {n_total} points")
    print(f"Using middle 90% for optimization: indices {start_idx} to {end_idx} ({end_idx - start_idx} points)")
    print(f"  Excluded: first {start_idx} points (5%) and last {n_total - end_idx} points (5%)")
    
    # Store full dataset for solving (needed for proper physics)
    # But optimization will only compare against middle 90%
    timestamp_full = timestamp
    thermistor_0_full = thermistor_0
    voltage_full = voltage
    
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
    # Use higher resolution for better accuracy (matches heatpump.py)
    thermistor_x_pos = 0.003
    N_nodes = 50  # Higher resolution (matches heatpump.py)
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
    
    # Create interpolation using full dataset (needed for solving over full time span)
    voltage_interp = interp1d(timestamp_full, voltage_full, kind='linear',
                              fill_value=(voltage_full[0], voltage_full[-1]), bounds_error=False)
    
    fixed_params = {
        'T_inf': T_inf, 'k_brass': k_brass, 'rho_brass': rho_brass,
        'c_brass': c_brass, 'L_brass': L_brass, 'N_nodes': N_nodes,
        'x_grid': x_grid, 'thermistor_node_idx': thermistor_node_idx,
        'A_hot': A_hot, 'rtol': 1e-4, 'atol': 1e-6,  # More relaxed tolerance for optimization speed
        'rho_ceramic': 3970.0, 'c_ceramic': 775.0,
        'radius_plate': 0.015, 'thickness_plate': 0.002,
        'C_hot_plate': 300.0, 'thickness_grease': 0.0001, 'k_grease': 1.0,
        'h_hot': 200.0  # Fixed convective coefficient
    }
    
    # Store full dataset for solving, but optimization will use middle 90% for comparison
    experimental_data = {
        'timestamp': timestamp_full,  # Full timestamp for solving
        'thermistor_0': thermistor_0_full,  # Full thermistor data for solving
        'voltage_interp': voltage_interp, 
        't_span': (timestamp_full[0], timestamp_full[-1]),  # Full time span
        'T_initial': T_initial,
        'middle90_start_idx': start_idx,  # Store indices for middle 90%
        'middle90_end_idx': end_idx
    }
    
    initial_params = np.array([0.05, 0.5, 2.5])  # alpha, K_therm, R_elec
    bounds = [(0.01, 0.06), (0.1, 2.0), (1.0, 5.0)]
    
    return initial_params, bounds, fixed_params, experimental_data


def optimize_parameters(quick_test=False):
    """Optimize parameters."""
    import json
    
    initial_params, bounds, fixed_params, experimental_data = setup_optimization()
    
    param_names = ['alpha', 'K_therm', 'R_elec']
    print(f"\nInitial Parameters: {dict(zip(param_names, initial_params))}")
    print(f"Bounds: {dict(zip(param_names, bounds))}\n")
    
    # Initialize iteration history and exploration tracking
    iteration_history = []
    history_file = Path('data/netflux/optimization_history.json')
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Track visited regions for heat map (for parameter pair: alpha vs K_therm)
    visited_regions = {}  # (alpha_bin, K_therm_bin) -> visit_count
    trajectory = []  # List of (alpha, K_therm, fx) tuples
    
    def objective_wrapper(params):
        return objective_function(params, fixed_params, experimental_data)
    
    def callback(xk, convergence):
        """Callback to save iteration parameters and track exploration."""
        fx = objective_function(xk, fixed_params, experimental_data)
        
        iteration_data = {
            'iteration': len(iteration_history),
            'fx': float(fx),
            'alpha': float(xk[0]),
            'K_therm': float(xk[1]),
            'R_elec': float(xk[2]),
            'convergence': float(convergence)
        }
        
        iteration_history.append(iteration_data)
        
        # Track trajectory for alpha vs K_therm (fixing R_elec at current value)
        trajectory.append((float(xk[0]), float(xk[1]), float(fx)))
        
        # Track visited regions (bin the parameter space)
        alpha_bin = int((xk[0] - bounds[0][0]) / (bounds[0][1] - bounds[0][0]) * 50)
        K_therm_bin = int((xk[1] - bounds[1][0]) / (bounds[1][1] - bounds[1][0]) * 50)
        key = (alpha_bin, K_therm_bin)
        visited_regions[key] = visited_regions.get(key, 0) + 1
        
        # Save to JSON file after each iteration
        with open(history_file, 'w') as f:
            json.dump(iteration_history, f, indent=2)
    
    # Use global optimizer to systematically explore all values within bounds
    print("Using differential_evolution (global optimizer)")
    print("This will systematically search all values within the bounds...\n")
    maxiter = 5 if quick_test else 20  # Reduced to ~20 steps for faster optimization
    popsize = 5 if quick_test else 15  # Reduced population size proportionally
    result = differential_evolution(
        objective_wrapper, bounds,
        seed=42, maxiter=maxiter, popsize=popsize,
        tol=1e-4, atol=1e-5, polish=True, disp=True,
        callback=callback
    )
    
    print(f"\nOptimization completed!")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Final Objective: {result.fun:.6f} (combined MSE + correlation error)")
    print(f"  Iterations: {result.nit}/{maxiter}, Function evaluations: {result.nfev}")
    print(f"  Optimized Parameters: {dict(zip(param_names, result.x))}")
    
    # Save optimized parameters to file for use in heatpump.py
    optimized_params = dict(zip(param_names, result.x))
    params_file = Path('data/netflux/optimized_parameters.json')
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(optimized_params, f, indent=4)
    print(f"\nOptimized parameters saved to: {params_file}")
    print(f"Iteration history saved to: {history_file}")
    print(f"  Total iterations: {len(iteration_history)}")
    
    # Save trajectory and visited regions data for GIF generation
    if len(trajectory) > 0:
        gif_data = {
            'trajectory': trajectory,
            'visited_regions': {f"{k[0]}_{k[1]}": v for k, v in visited_regions.items()},
            'bounds': bounds,
            'param_names': param_names,
            'optimized_R_elec': float(result.x[2]),  # Save optimized R_elec value
            'fixed_params': {k: (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) 
                           for k, v in fixed_params.items() if k not in ['x_grid', 'thermistor_node_idx']}
        }
        gif_data_file = Path('data/netflux/gif_data.json')
        gif_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(gif_data_file, 'w') as f:
            json.dump(gif_data, f, indent=2)
        print(f"GIF data saved to: {gif_data_file}")
        print(f"  Run visualise.py to generate the evolution GIF")
    
    return result, optimized_params


def create_heatmap_evolution_gif(trajectory, visited_regions, bounds, fixed_params, 
                                 experimental_data, param_names, n_frames=50):
    """
    Create an animated GIF showing heat map evolution of optimization process.
    
    Parameters:
    - trajectory: List of (alpha, K_therm, fx) tuples
    - visited_regions: Dict mapping (alpha_bin, K_therm_bin) -> visit_count
    - bounds: Parameter bounds
    - fixed_params: Fixed parameters
    - experimental_data: Experimental data
    - param_names: Parameter names
    - n_frames: Number of frames in animation
    """
    print("\nCreating heat map evolution GIF...")
    
    # Get bounds for alpha and K_therm
    alpha_bounds = bounds[0]
    K_therm_bounds = bounds[1]
    
    # Create parameter grids for heat map
    n_grid = 40
    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], n_grid)
    K_therm_grid = np.linspace(K_therm_bounds[0], K_therm_bounds[1], n_grid)
    alpha_mesh, K_therm_mesh = np.meshgrid(alpha_grid, K_therm_grid)
    
    # Calculate objective function over grid (fix R_elec at optimized value)
    # Use the last R_elec value from trajectory (or could use optimized value)
    R_elec_fixed = bounds[2][0] + (bounds[2][1] - bounds[2][0]) * 0.5  # Midpoint as default
    
    print("  Computing objective function landscape...")
    fx_mesh = np.zeros_like(alpha_mesh)
    for i in range(n_grid):
        for j in range(n_grid):
            params_vector = np.array([alpha_mesh[i, j], K_therm_mesh[i, j], R_elec_fixed])
            fx_mesh[i, j] = objective_function(params_vector, fixed_params, experimental_data)
    
    # Prepare trajectory data
    trajectory_array = np.array(trajectory)
    n_iterations = len(trajectory)
    frame_indices = np.linspace(0, n_iterations - 1, n_frames, dtype=int)
    
    # Create visited regions heat map
    visited_map = np.zeros((50, 50))
    for (alpha_bin, K_therm_bin), count in visited_regions.items():
        if 0 <= alpha_bin < 50 and 0 <= K_therm_bin < 50:
            visited_map[K_therm_bin, alpha_bin] = count
    
    # Normalize visited map
    if visited_map.max() > 0:
        visited_map = visited_map / visited_map.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initial plot
    im = ax.contourf(alpha_mesh, K_therm_mesh, fx_mesh, levels=20, cmap='viridis', alpha=0.7)
    ax.set_xlabel('$\\alpha$', fontsize=20)
    ax.set_ylabel('$k$', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # Remove title
    ax.set_title('', fontsize=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Objective Function (fx, log scale)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Initialize trajectory line and best point
    traj_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.6, label='Trajectory')
    best_point, = ax.plot([], [], 'r*', markersize=15, label='Current Best')
    visited_overlay = ax.imshow(visited_map, extent=[alpha_bounds[0], alpha_bounds[1], 
                                                     K_therm_bounds[0], K_therm_bounds[1]],
                               cmap='hot', alpha=0.3, origin='lower', aspect='auto')
    
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Text for iteration info
    iter_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        """Update plot for each frame."""
        idx = frame_indices[frame]
        
        # Get trajectory up to current iteration
        traj_up_to_now = trajectory_array[:idx+1]
        
        if len(traj_up_to_now) > 0:
            # Update trajectory line
            traj_line.set_data(traj_up_to_now[:, 0], traj_up_to_now[:, 1])
            
            # Update best point (current position)
            best_point.set_data([traj_up_to_now[-1, 0]], [traj_up_to_now[-1, 1]])
            
            # Update visited regions (accumulate up to current iteration)
            current_visited = {}
            for i in range(idx + 1):
                alpha, K_therm, _ = trajectory[i]
                alpha_bin = int((alpha - alpha_bounds[0]) / (alpha_bounds[1] - alpha_bounds[0]) * 50)
                K_therm_bin = int((K_therm - K_therm_bounds[0]) / (K_therm_bounds[1] - K_therm_bounds[0]) * 50)
                key = (alpha_bin, K_therm_bin)
                current_visited[key] = current_visited.get(key, 0) + 1
            
            visited_map_current = np.zeros((50, 50))
            for (alpha_bin, K_therm_bin), count in current_visited.items():
                if 0 <= alpha_bin < 50 and 0 <= K_therm_bin < 50:
                    visited_map_current[K_therm_bin, alpha_bin] = count
            
            if visited_map_current.max() > 0:
                visited_map_current = visited_map_current / visited_map_current.max()
            
            visited_overlay.set_array(visited_map_current)
            
            # Update text
            iter_text.set_text(f'Iteration: {idx}\n'
                             f'fx: {traj_up_to_now[-1, 2]:.4f}\n'
                             f'α: {traj_up_to_now[-1, 0]:.4f}\n'
                             f'K: {traj_up_to_now[-1, 1]:.4f}')
        
        return traj_line, best_point, visited_overlay, iter_text
    
    # Create animation
    print(f"  Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False, repeat=True)
    
    # Save as GIF
    gif_path = Path('plots/optimization/heatmap_evolution.gif')
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving GIF to {gif_path}...")
    anim.save(gif_path, writer='pillow', fps=10)
    print(f"  ✓ GIF saved successfully!")
    
    plt.close()


def plot_optimization_results(result, fixed_params, experimental_data):
    """Plot comparison between optimized model and experimental data."""
    alpha, K_therm, R_elec = result.x
    h_hot = fixed_params['h_hot']  # Fixed at 200.0
    
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
    
    # Use global optimizer to systematically search all values within bounds
    result, optimized_params = optimize_parameters(quick_test=False)
    
    _, _, fixed_params, experimental_data = setup_optimization()
    plot_optimization_results(result, fixed_params, experimental_data)
    
    print("\n" + "=" * 70)
    print("Optimization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
