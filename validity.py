import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, gaussian_kde, ttest_1samp, chi2, normaltest, shapiro, t
import matplotlib.pyplot as plt
from pathlib import Path


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


def interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval):
    """
    Interpolate brass temperature at a specific position for all time points.
    
    Parameters:
    - T_brass_all: Array of brass temperatures at grid points, shape (N_nodes, len(t_eval))
    - x_grid: Spatial grid positions along the brass rod
    - x_pos: Position along rod to interpolate to (in meters)
    - t_eval: Time points where solution is evaluated
    
    Returns:
    - T_at_pos: Temperature at x_pos for all time points (in Kelvin)
    """
    T_at_pos = np.zeros(len(t_eval))
    for i in range(len(t_eval)):
        T_brass_interp = interp1d(x_grid, T_brass_all[:, i], kind='linear', 
                                  fill_value='extrapolate', bounds_error=False)
        T_at_pos[i] = T_brass_interp(x_pos)
    return T_at_pos


def calculate_residual(T_model_C, t_model, thermistor_data, t_exp):
    """
    Calculate residual between model and experimental data.
    
    Parameters:
    - T_model_C: Model temperature in Celsius at model time points
    - t_model: Model time points
    - thermistor_data: Experimental thermistor data
    - t_exp: Experimental time points
    
    Returns:
    - residual: Model - Experimental (in Celsius)
    """
    T_model_interp_func = interp1d(t_model, T_model_C, kind='linear',
                                   fill_value='extrapolate', bounds_error=False)
    T_model_at_exp_times = T_model_interp_func(t_exp)
    return T_model_at_exp_times - thermistor_data


def plot_thermistor_comparison(ax, t_eval, T_model_C, timestamp, thermistor_data, 
                               x_pos_mm, thermistor_id, model_color, exp_color):
    """
    Plot temperature comparison for a single thermistor.
    
    Parameters:
    - ax: Matplotlib axis to plot on
    - t_eval: Model time points
    - T_model_C: Model temperature in Celsius
    - timestamp: Experimental time points
    - thermistor_data: Experimental thermistor data
    - x_pos_mm: Position in mm
    - thermistor_id: Thermistor ID number
    - model_color: Color for model line
    - exp_color: Color for experimental line
    """
    ax.plot(t_eval, T_model_C, model_color, linewidth=2, 
            label=f'T_brass (Model, x={x_pos_mm:.1f}mm)')
    ax.plot(timestamp, thermistor_data, exp_color, linewidth=2, 
            label=f'Thermistor {thermistor_id} (Experimental)', alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Brass Temperature at x={x_pos_mm:.1f}mm (Thermistor {thermistor_id}) vs Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)


def plot_residual(ax, timestamp, residual, thermistor_id, residual_color):
    """
    Plot residual for a single thermistor.
    
    Parameters:
    - ax: Matplotlib axis to plot on
    - timestamp: Time points
    - residual: Residual values (Model - Experimental)
    - thermistor_id: Thermistor ID number
    - residual_color: Color for residual line
    """
    ax.plot(timestamp, residual, residual_color, linewidth=2, 
            label=f'Residual (Model - Thermistor {thermistor_id})')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Residual (°C)', fontsize=12)
    ax.set_title(f'Residual: Numerical Model - Thermistor {thermistor_id} Data', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', fontsize=11)


def calculate_rmse_and_correlation(T_model, T_exp):
    """
    Calculate RMSE and Pearson correlation coefficient between model and experimental data.
    
    Parameters:
    - T_model: Model temperature array
    - T_exp: Experimental temperature array
    
    Returns:
    - rmse: Root Mean Square Error in °C
    - correlation: Pearson correlation coefficient
    """
    # Remove any NaN or inf values
    valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
    if np.sum(valid_mask) == 0:
        return np.nan, np.nan
    
    T_model_valid = T_model[valid_mask]
    T_exp_valid = T_exp[valid_mask]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((T_model_valid - T_exp_valid)**2))
    
    # Calculate Pearson correlation
    if len(T_model_valid) > 1 and np.std(T_model_valid) > 0 and np.std(T_exp_valid) > 0:
        correlation, _ = pearsonr(T_model_valid, T_exp_valid)
    else:
        correlation = np.nan
    
    return rmse, correlation


def save_temperature_data_to_csv(sol_dict, timestamp, thermistor_data_dict, x_grid, save_path=None):
    """
    Save temperature comparison data (model vs experimental) for all thermistors to CSV.
    Only saves the middle 90% of the dataset (excludes first 5% and last 5%).
    
    Parameters:
    - sol_dict: Dictionary mapping thermistor_id to solution object from solve_ivp
    - timestamp: Original time data
    - thermistor_data_dict: Dictionary mapping thermistor_id to {'data': array, 'x_pos': float}
                           x_pos is in meters (e.g., 0.003 for 3mm)
    - x_grid: Spatial grid positions along the brass rod
    - save_path: Path to save the CSV file (if None, uses default path)
    """
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for saving
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Process each thermistor's solution
    thermistor_results = {}
    for therm_id, therm_info in thermistor_data_dict.items():
        sol = sol_dict[therm_id]
        x_pos = therm_info['x_pos']
        
        # Use actual solution points to avoid interpolation artifacts
        t_eval = sol.t
        T_solution = sol.y
        T_brass_all = T_solution[2:, :]
        
        # Interpolate model temperature at this thermistor's position
        T_brass_K = interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval)
        T_brass_C = T_brass_K - 273.15
        
        # Filter experimental data to middle 90% for comparison
        thermistor_data_middle90 = therm_info['data'][start_idx:end_idx]
        
        # Interpolate model to experimental time points in middle 90%
        T_model_interp_func = interp1d(t_eval, T_brass_C, kind='linear',
                                       fill_value='extrapolate', bounds_error=False)
        T_model_middle90 = T_model_interp_func(timestamp_plot)
        
        thermistor_results[therm_id] = {
            'T_model_C': T_model_middle90,  # Filtered to middle 90%
            'T_exp_C': thermistor_data_middle90,  # Experimental data (middle 90%)
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
        save_path = Path('data/comparison/temperature_comparison_data.csv')
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Temperature comparison data saved to: {save_path}")
    print(f"  Saved {len(df)} time points with data for {len(thermistor_results)} thermistors")
    
    return save_path


def plot_temperatures_vs_time(sol_dict, timestamp, thermistor_data_dict, x_grid, voltage_func, params, 
                              t_span, rtol, atol, N_nodes, save_path=None):
    """
    Plot T_brass at thermistor positions and thermistor temperatures as a function of time.
    Each thermistor uses its own T0 for numerical integration.
    Only plots the middle 90% of the dataset (excludes first 5% and last 5%).
    
    Parameters:
    - sol_dict: Dictionary mapping thermistor_id to solution object from solve_ivp
    - timestamp: Original time data for reference
    - thermistor_data_dict: Dictionary mapping thermistor_id to {'data': array, 'x_pos': float}
                           x_pos is in meters (e.g., 0.003 for 3mm)
    - x_grid: Spatial grid positions along the brass rod
    - voltage_func: Function V(t) returning voltage at time t
    - params: Dictionary containing physical parameters
    - t_span: Time span tuple (t0, tf)
    - rtol: Relative tolerance for ODE solver
    - atol: Absolute tolerance for ODE solver
    - N_nodes: Number of nodes in spatial discretization
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for plotting
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Process each thermistor's solution
    thermistor_results = {}
    for therm_id, therm_info in thermistor_data_dict.items():
        sol = sol_dict[therm_id]
        x_pos = therm_info['x_pos']
        
        # Use actual solution points to avoid interpolation artifacts
        t_eval = sol.t
        T_solution = sol.y
        
        # Extract temperatures
        Tc = T_solution[0, :]
        Th = T_solution[1, :]
        T_brass_all = T_solution[2:, :]
        
        # Interpolate model temperature at this thermistor's position
        T_brass_K = interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval)
        T_brass_C = T_brass_K - 273.15
        
        # Filter experimental data to middle 90% for comparison
        thermistor_data_middle90 = therm_info['data'][start_idx:end_idx]
        
        # Filter model data to middle 90% for plotting
        # Interpolate model to experimental time points in middle 90%
        T_model_interp_func = interp1d(t_eval, T_brass_C, kind='linear',
                                       fill_value='extrapolate', bounds_error=False)
        T_model_middle90 = T_model_interp_func(timestamp_plot)
        
        thermistor_results[therm_id] = {
            'T_model_C': T_model_middle90,  # Filtered to middle 90%
            'T_exp_C': thermistor_data_middle90,  # Experimental data (middle 90%)
            'x_pos_mm': x_pos * 1000,
            't_eval': timestamp_plot,  # Use filtered timestamp for plotting
            'Tc': Tc,
            'Th': Th
        }
    
    # Define colors for each thermistor
    model_colors = ['g', 'orange', 'brown', 'red', 'purple', 'pink', 'olive', 'cyan']
    exp_colors = ['b', 'purple', 'pink', 'coral', 'indigo', 'magenta', 'darkgreen', 'teal']
    
    # Create subplots: 1 plot per thermistor (temperature comparison only)
    n_thermistors = len(thermistor_data_dict)
    fig, axes = plt.subplots(n_thermistors, 1, figsize=(10, 3*n_thermistors))
    
    # Handle case where there's only one thermistor (axes becomes 1D instead of array)
    if n_thermistors == 1:
        axes = [axes]
    
    # Plot each thermistor (using middle 90% data)
    for i, (therm_id, therm_info) in enumerate(sorted(thermistor_data_dict.items())):
        result = thermistor_results[therm_id]
        
        # Plot temperature comparison (middle 90% only)
        plot_thermistor_comparison(
            axes[i], result['t_eval'], result['T_model_C'], timestamp_plot, 
            result['T_exp_C'], result['x_pos_mm'], therm_id,
            model_colors[i % len(model_colors)], exp_colors[i % len(exp_colors)]
        )
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_residual_distributions(sol_dict, timestamp, thermistor_data_dict, x_grid, save_path=None):
    """
    Plot residual distributions (histograms with kernel density estimation) for all thermistors.
    Uses KDE instead of assuming a Gaussian distribution.
    
    Parameters:
    - sol_dict: Dictionary mapping thermistor_id to solution object from solve_ivp
    - timestamp: Original time data for reference
    - thermistor_data_dict: Dictionary mapping thermistor_id to {'data': array, 'x_pos': float}
                           x_pos is in meters (e.g., 0.003 for 3mm)
    - x_grid: Spatial grid positions along the brass rod
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for analysis
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Calculate residuals for each thermistor
    n_thermistors = len(thermistor_data_dict)
    
    # Create subplots: arrange in a grid (2 columns, or as needed)
    n_cols = 2
    n_rows = (n_thermistors + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    # Handle different subplot arrangements
    if n_thermistors == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process each thermistor
    for idx, (therm_id, therm_info) in enumerate(sorted(thermistor_data_dict.items())):
        sol = sol_dict[therm_id]
        x_pos = therm_info['x_pos']
        
        # Use actual solution points to avoid interpolation artifacts
        t_eval = sol.t
        T_solution = sol.y
        T_brass_all = T_solution[2:, :]
        
        # Interpolate model temperature at this thermistor's position
        T_brass_K = interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval)
        T_brass_C = T_brass_K - 273.15
        
        # Filter experimental data to middle 90% for comparison
        thermistor_data_middle90 = therm_info['data'][start_idx:end_idx]
        
        # Interpolate model to experimental time points in middle 90%
        T_model_interp_func = interp1d(t_eval, T_brass_C, kind='linear',
                                       fill_value='extrapolate', bounds_error=False)
        T_model_middle90 = T_model_interp_func(timestamp_plot)
        
        # Calculate residuals (Model - Experimental)
        residuals = T_model_middle90 - thermistor_data_middle90
        
        # Remove NaN and inf values
        valid_mask = np.isfinite(residuals)
        residuals_clean = residuals[valid_mask]
        
        if len(residuals_clean) == 0:
            print(f"  Warning: No valid residuals for thermistor {therm_id}")
            continue
        
        # Calculate statistics for display
        mu = np.mean(residuals_clean)
        std = np.std(residuals_clean)
        median = np.median(residuals_clean)
        
        # Create histogram
        ax = axes[idx]
        n_bins = min(50, int(np.sqrt(len(residuals_clean))))  # Adaptive number of bins
        counts, bins, patches = ax.hist(residuals_clean, bins=n_bins, density=True, 
                                       color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Fit kernel density estimation (non-parametric, doesn't assume Gaussian)
        try:
            kde = gaussian_kde(residuals_clean)
            x_fit = np.linspace(residuals_clean.min(), residuals_clean.max(), 200)
            y_fit = kde(x_fit)
            ax.plot(x_fit, y_fit, 'k-', linewidth=2, 
                   label=f'KDE (μ={mu:.2f}, σ={std:.2f}, med={median:.2f})')
        except Exception as e:
            # Fallback if KDE fails (e.g., too few points or constant values)
            print(f"  Warning: KDE fitting failed for thermistor {therm_id}: {e}")
            # Plot simple statistics line
            ax.axvline(mu, color='k', linestyle='--', linewidth=1.5, 
                      label=f'Mean={mu:.2f}, σ={std:.2f}')
        
        # Formatting
        ax.set_xlabel('Error (°C)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Residual Distribution (Thermistor {therm_id} at {x_pos*1000:.0f}mm)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_thermistors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual distributions plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_mean_residual_distribution(sol_dict, timestamp, thermistor_data_dict, x_grid, save_path=None):
    """
    Plot mean residual distribution combining all thermistors into one plot.
    
    Parameters:
    - sol_dict: Dictionary mapping thermistor_id to solution object from solve_ivp
    - timestamp: Original time data for reference
    - thermistor_data_dict: Dictionary mapping thermistor_id to {'data': array, 'x_pos': float}
                           x_pos is in meters (e.g., 0.003 for 3mm)
    - x_grid: Spatial grid positions along the brass rod
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for analysis
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Collect all residuals from all thermistors
    all_residuals = []
    
    for therm_id, therm_info in sorted(thermistor_data_dict.items()):
        sol = sol_dict[therm_id]
        x_pos = therm_info['x_pos']
        
        # Use actual solution points to avoid interpolation artifacts
        t_eval = sol.t
        T_solution = sol.y
        T_brass_all = T_solution[2:, :]
        
        # Interpolate model temperature at this thermistor's position
        T_brass_K = interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval)
        T_brass_C = T_brass_K - 273.15
        
        # Filter experimental data to middle 90% for comparison
        thermistor_data_middle90 = therm_info['data'][start_idx:end_idx]
        
        # Interpolate model to experimental time points in middle 90%
        T_model_interp_func = interp1d(t_eval, T_brass_C, kind='linear',
                                       fill_value='extrapolate', bounds_error=False)
        T_model_middle90 = T_model_interp_func(timestamp_plot)
        
        # Calculate residuals (Model - Experimental)
        residuals = T_model_middle90 - thermistor_data_middle90
        
        # Remove NaN and inf values
        valid_mask = np.isfinite(residuals)
        residuals_clean = residuals[valid_mask]
        
        # Add to combined list
        all_residuals.extend(residuals_clean.tolist())
    
    all_residuals = np.array(all_residuals)
    
    if len(all_residuals) == 0:
        print("  Warning: No valid residuals found!")
        return
    
    # Calculate statistics for display
    mu = np.mean(all_residuals)
    std = np.std(all_residuals)
    median = np.median(all_residuals)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n_bins = min(50, int(np.sqrt(len(all_residuals))))  # Adaptive number of bins
    counts, bins, patches = ax.hist(all_residuals, bins=n_bins, density=True, 
                                   color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Fit kernel density estimation (non-parametric, doesn't assume Gaussian)
    try:
        kde = gaussian_kde(all_residuals)
        x_fit = np.linspace(all_residuals.min(), all_residuals.max(), 200)
        y_fit = kde(x_fit)
        ax.plot(x_fit, y_fit, 'k-', linewidth=2, 
               label=f'KDE (μ={mu:.2f}, σ={std:.2f}, med={median:.2f})')
    except Exception as e:
        # Fallback if KDE fails (e.g., too few points or constant values)
        print(f"  Warning: KDE fitting failed: {e}")
        # Plot simple statistics line
        ax.axvline(mu, color='k', linestyle='--', linewidth=1.5, 
                  label=f'Mean={mu:.2f}, σ={std:.2f}')
    
    # Formatting
    ax.set_xlabel('Error (°C)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Mean Residual Distribution (All Thermistors Combined)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mean residual distribution plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_rmse_and_correlation_vs_position(sol_dict, timestamp, thermistor_data_dict, x_grid, save_path=None):
    """
    Plot RMSE and Pearson Correlation vs position for all thermistors in one image.
    
    Parameters:
    - sol_dict: Dictionary mapping thermistor_id to solution object from solve_ivp
    - timestamp: Original time data for reference
    - thermistor_data_dict: Dictionary mapping thermistor_id to {'data': array, 'x_pos': float}
                           x_pos is in meters (e.g., 0.003 for 3mm)
    - x_grid: Spatial grid positions along the brass rod
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    # Extract middle 90% of dataset (remove first 5% and last 5%)
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)  # Start at 5%
    end_idx = int(0.95 * n_total)    # End at 95%
    
    # Filter data to middle 90% for analysis
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Calculate RMSE and correlation for each thermistor
    positions = []
    rmse_values = []
    correlation_values = []
    
    for therm_id, therm_info in sorted(thermistor_data_dict.items()):
        sol = sol_dict[therm_id]
        x_pos = therm_info['x_pos']
        
        # Use actual solution points to avoid interpolation artifacts
        t_eval = sol.t
        T_solution = sol.y
        T_brass_all = T_solution[2:, :]
        
        # Interpolate model temperature at this thermistor's position
        T_brass_K = interpolate_temperature_at_position(T_brass_all, x_grid, x_pos, t_eval)
        T_brass_C = T_brass_K - 273.15
        
        # Filter experimental data to middle 90% for comparison
        thermistor_data_middle90 = therm_info['data'][start_idx:end_idx]
        
        # Interpolate model to experimental time points in middle 90%
        T_model_interp_func = interp1d(t_eval, T_brass_C, kind='linear',
                                       fill_value='extrapolate', bounds_error=False)
        T_model_middle90 = T_model_interp_func(timestamp_plot)
        
        # Calculate RMSE and correlation
        rmse, correlation = calculate_rmse_and_correlation(T_model_middle90, thermistor_data_middle90)
        
        positions.append(x_pos * 1000)  # Convert to mm
        rmse_values.append(rmse)
        correlation_values.append(correlation)
        
        print(f"  Thermistor {therm_id} (x={x_pos*1000:.1f}mm): RMSE={rmse:.4f} °C, r={correlation:.6f}")
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot RMSE on left y-axis (red dots only, no line)
    color_rmse = 'red'
    ax1.set_xlabel('Position along rod (mm)', fontsize=12)
    ax1.set_ylabel('RMSE (°C)', fontsize=12, color=color_rmse)
    line1 = ax1.plot(positions, rmse_values, 'o', color=color_rmse, 
                     markersize=8, label='RMSE', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    ax1.grid(True, alpha=0.3)
    
    # Plot Pearson Correlation on right y-axis (blue line with squares)
    ax2 = ax1.twinx()
    color_corr = 'blue'
    ax2.set_ylabel('Pearson Correlation (r)', fontsize=12, color=color_corr)
    line2 = ax2.plot(positions, correlation_values, 's-', color=color_corr, linewidth=2,
                     markersize=8, label='Pearson Correlation', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_corr)
    
    # Set title
    ax1.set_title('Model Performance vs. Sensor Position', fontsize=14, fontweight='bold')
    
    # Add legend - position it in upper left to avoid overlapping with curves
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RMSE and Correlation plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_temperatures_from_csv(csv_path, save_path=None):
    """
    Plot temperature comparisons from CSV data.
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    df = pd.read_csv(csv_path)
    time = df['Time (s)'].values
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    n_thermistors = len(model_cols)
    
    # Define colors for each thermistor
    model_colors = ['g', 'orange', 'brown', 'red', 'purple', 'pink', 'olive', 'cyan']
    exp_colors = ['b', 'purple', 'pink', 'coral', 'indigo', 'magenta', 'darkgreen', 'teal']
    
    # Create subplots: 1 plot per thermistor
    fig, axes = plt.subplots(n_thermistors, 1, figsize=(10, 3*n_thermistors))
    
    # Handle case where there's only one thermistor
    if n_thermistors == 1:
        axes = [axes]
    
    # Plot each thermistor
    for i in range(n_thermistors):
        model_col = f'T_model_{i}_x'
        exp_col = f'T_exp_{i}_x'
        
        # Find the actual column names (they have position info)
        model_col_name = [col for col in model_cols if col.startswith(model_col)][0]
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract position from column name (e.g., "T_model_0_x3.0mm (°C)" -> 3.0)
        x_pos_mm = float(model_col_name.split('_x')[1].split('mm')[0])
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Plot
        axes[i].plot(time, T_model, model_colors[i % len(model_colors)], linewidth=2, 
                    label=f'T_brass (Model, x={x_pos_mm:.1f}mm)')
        axes[i].plot(time, T_exp, exp_colors[i % len(exp_colors)], linewidth=2, 
                    label=f'Thermistor {i} (Experimental)', alpha=0.7)
        axes[i].set_xlabel('Time (s)', fontsize=12)
        axes[i].set_ylabel('Temperature (°C)', fontsize=12)
        axes[i].set_title(f'Brass Temperature at x={x_pos_mm:.1f}mm (Thermistor {i}) vs Time', 
                         fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_residuals_from_csv(csv_path, save_path=None):
    """
    Plot residual distributions from CSV data.
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    df = pd.read_csv(csv_path)
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    n_thermistors = len(model_cols)
    
    # Create subplots: arrange in a grid (2 columns)
    n_cols = 2
    n_rows = (n_thermistors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    
    # Handle different subplot arrangements
    if n_thermistors == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Process each thermistor
    for i in range(n_thermistors):
        model_col_name = model_cols[i]
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract position from column name
        x_pos_mm = float(model_col_name.split('_x')[1].split('mm')[0])
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate residuals
        residuals = T_model - T_exp
        
        # Remove NaN and inf values
        valid_mask = np.isfinite(residuals)
        residuals_clean = residuals[valid_mask]
        
        if len(residuals_clean) == 0:
            print(f"  Warning: No valid residuals for thermistor {i}")
            continue
        
        # Calculate statistics
        mu = np.mean(residuals_clean)
        std = np.std(residuals_clean)
        median = np.median(residuals_clean)
        
        # Create histogram
        ax = axes[i]
        n_bins = min(50, int(np.sqrt(len(residuals_clean))))
        counts, bins, patches = ax.hist(residuals_clean, bins=n_bins, density=True, 
                                       color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Fit kernel density estimation
        try:
            kde = gaussian_kde(residuals_clean)
            x_fit = np.linspace(residuals_clean.min(), residuals_clean.max(), 200)
            y_fit = kde(x_fit)
            ax.plot(x_fit, y_fit, 'k-', linewidth=2, 
                   label=f'KDE (μ={mu:.2f}, σ={std:.2f}, med={median:.2f})')
        except Exception as e:
            print(f"  Warning: KDE fitting failed for thermistor {i}: {e}")
            ax.axvline(mu, color='k', linestyle='--', linewidth=1.5, 
                      label=f'Mean={mu:.2f}, σ={std:.2f}')
        
        # Formatting
        ax.set_xlabel('Error (°C)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Residual Distribution (Thermistor {i} at {x_pos_mm:.0f}mm)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_thermistors, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual distributions plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_rmse_correlation_from_csv(csv_path, save_path=None):
    """
    Plot RMSE and correlation vs position from CSV data.
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    df = pd.read_csv(csv_path)
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    
    positions = []
    rmse_values = []
    correlation_values = []
    mse_values = []
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract thermistor ID and position from column name
        # e.g., "T_model_0_x3.0mm (°C)" -> therm_id=0, x_pos=3.0
        parts = model_col_name.split('_')
        therm_id = int(parts[2])
        # Extract position: "x3.0mm (°C)" -> "3.0"
        # parts[3] is "x3.0mm" but if there's a space, it might be "x3.0mm (°C)"
        pos_str = parts[3].split()[0]  # Take first part before any space
        x_pos_mm = float(pos_str.replace('x', '').replace('mm', ''))
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate RMSE and correlation
        rmse, correlation = calculate_rmse_and_correlation(T_model, T_exp)
        
        # Calculate MSE
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
        if np.sum(valid_mask) > 0:
            T_model_valid = T_model[valid_mask]
            T_exp_valid = T_exp[valid_mask]
            mse = np.mean((T_model_valid - T_exp_valid)**2)
        else:
            mse = np.nan
        
        positions.append(x_pos_mm)
        rmse_values.append(rmse)
        correlation_values.append(correlation)
        mse_values.append(mse)
        
        print(f"  Thermistor {therm_id} (x={x_pos_mm:.1f}mm): RMSE={rmse:.4f} °C, r={correlation:.6f}, MSE={mse:.6f} °C²")
    
    # Calculate mean residuals and standard deviations for the 4th plot
    mean_residuals = []
    std_residuals = []
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate residuals
        residuals = T_model - T_exp
        
        # Remove NaN and inf values
        valid_mask = np.isfinite(residuals)
        residuals_clean = residuals[valid_mask]
        
        if len(residuals_clean) > 0:
            mean_residual = np.mean(residuals_clean)
            std_residual = np.std(residuals_clean)
        else:
            mean_residual = np.nan
            std_residual = np.nan
        
        mean_residuals.append(mean_residual)
        std_residuals.append(std_residual)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    
    # Plot 1: RMSE
    ax1 = axes[0]
    ax1.plot(positions, rmse_values, 'o-', color='red', linewidth=2,
             markersize=8, label='RMSE', alpha=0.8)
    ax1.set_xlabel('Position along rod (mm)', fontsize=12)
    ax1.set_ylabel('RMSE (°C)', fontsize=12, color='red')
    ax1.set_title('Root Mean Square Error vs. Sensor Position', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11)
    
    # Plot 2: Pearson Correlation
    ax2 = axes[1]
    ax2.plot(positions, correlation_values, 's-', color='blue', linewidth=2,
             markersize=8, label='Pearson Correlation', alpha=0.8)
    ax2.set_xlabel('Position along rod (mm)', fontsize=12)
    ax2.set_ylabel('Pearson Correlation (r)', fontsize=12, color='blue')
    ax2.set_title('Pearson Correlation vs. Sensor Position', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=11)
    
    # Plot 3: MSE
    ax3 = axes[2]
    ax3.plot(positions, mse_values, '^-', color='green', linewidth=2,
             markersize=8, label='MSE', alpha=0.8)
    ax3.set_xlabel('Position along rod (mm)', fontsize=12)
    ax3.set_ylabel('MSE (°C²)', fontsize=12, color='green')
    ax3.set_title('Mean Squared Error vs. Sensor Position', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=11)
    
    # Plot 4: Mean Residual
    ax4 = axes[3]
    ax4.errorbar(positions, mean_residuals, yerr=std_residuals, 
                fmt='o-', color='purple', linewidth=2, markersize=8, 
                capsize=5, capthick=2, alpha=0.8, label='Mean Residual ± 1σ')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero Error')
    ax4.set_xlabel('Position along rod (mm)', fontsize=12)
    ax4.set_ylabel('Mean Residual (°C)', fontsize=12)
    ax4.set_title('Mean Residual (Center Position) vs. Distance Along Rod', 
                 fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=11)
    
    # Overall title
    fig.suptitle('Model Performance Metrics vs. Sensor Position', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"RMSE and Correlation plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_metrics_bar_chart_normalized(csv_path, save_path=None):
    """
    Plot RMSE, MSE, and Mean Residual as unnormalized bar charts (in degrees Celsius).
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    df = pd.read_csv(csv_path)
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    
    positions = []
    rmse_values = []
    mse_values = []
    mean_residuals = []
    std_residuals = []
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract thermistor ID and position from column name
        parts = model_col_name.split('_')
        therm_id = int(parts[2])
        pos_str = parts[3].split()[0]
        x_pos_mm = float(pos_str.replace('x', '').replace('mm', ''))
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate RMSE
        rmse, _ = calculate_rmse_and_correlation(T_model, T_exp)
        
        # Calculate MSE
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
        if np.sum(valid_mask) > 0:
            T_model_valid = T_model[valid_mask]
            T_exp_valid = T_exp[valid_mask]
            mse = np.mean((T_model_valid - T_exp_valid)**2)
        else:
            mse = np.nan
        
        # Calculate residuals
        residuals = T_model - T_exp
        valid_residuals = residuals[np.isfinite(residuals)]
        if len(valid_residuals) > 0:
            mean_residual = np.mean(valid_residuals)
            std_residual = np.std(valid_residuals)
        else:
            mean_residual = np.nan
            std_residual = np.nan
        
        positions.append(x_pos_mm)
        rmse_values.append(rmse)
        mse_values.append(mse)
        mean_residuals.append(mean_residual)
        std_residuals.append(std_residual)
    
    # Create single figure with grouped bars (unnormalized)
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = np.arange(len(positions))
    width = 0.25  # Width of each bar group (3 bars now)
    
    # Use colorblind-friendly scientific color palette
    # Colors: Blue (#1f77b4), Orange (#ff7f0e), Purple (#9467bd) from matplotlib's default color cycle
    colors = ['#1f77b4', '#ff7f0e', '#9467bd']  # Blue, Orange, Purple
    
    # Create grouped bars with unnormalized values
    bars1 = ax.bar(x_pos - width, rmse_values, width, color=colors[0], alpha=0.8, 
                   label='RMSE (°C)')
    bars2 = ax.bar(x_pos, mse_values, width, color=colors[1], alpha=0.8, 
                   label='MSE (°C²)')
    bars3 = ax.bar(x_pos + width, mean_residuals, width, color=colors[2], alpha=0.8, 
                   label='Residual (°C)')
    
    # Add reference line for zero error (mean residual) at y=0 (no label)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Position along rod (mm)', fontsize=20)
    ax.set_ylabel('Error (°C or °C²)', fontsize=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{p:.1f}' for p in positions], rotation=45, ha='right', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=14)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Normalized bar chart plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def plot_mean_residual_vs_position(csv_path, save_path=None):
    """
    Plot mean residual (center position) vs distance along the rod.
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    df = pd.read_csv(csv_path)
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    
    positions = []
    mean_residuals = []
    std_residuals = []
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract thermistor ID and position from column name
        parts = model_col_name.split('_')
        therm_id = int(parts[2])
        # Extract position: "x3.0mm (°C)" -> "3.0"
        pos_str = parts[3].split()[0]  # Take first part before any space
        x_pos_mm = float(pos_str.replace('x', '').replace('mm', ''))
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate residuals
        residuals = T_model - T_exp
        
        # Remove NaN and inf values
        valid_mask = np.isfinite(residuals)
        residuals_clean = residuals[valid_mask]
        
        if len(residuals_clean) == 0:
            print(f"  Warning: No valid residuals for thermistor {therm_id}")
            continue
        
        # Calculate mean and standard deviation
        mean_residual = np.mean(residuals_clean)
        std_residual = np.std(residuals_clean)
        
        positions.append(x_pos_mm)
        mean_residuals.append(mean_residual)
        std_residuals.append(std_residual)
        
        print(f"  Thermistor {therm_id} (x={x_pos_mm:.1f}mm): Mean={mean_residual:.4f} °C, Std={std_residual:.4f} °C")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot mean residual with error bars (standard deviation)
    ax.errorbar(positions, mean_residuals, yerr=std_residuals, 
                fmt='o-', color='purple', linewidth=2, markersize=8, 
                capsize=5, capthick=2, alpha=0.8, label='Mean Residual ± 1σ')
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero Error')
    
    ax.set_xlabel('Position along rod (mm)', fontsize=12)
    ax.set_ylabel('Mean Residual (°C)', fontsize=12)
    ax.set_title('Mean Residual (Center Position) vs. Distance Along Rod', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save plot if save_path is provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mean residual vs position plot saved to: {save_path}")
        plt.close()
    else:
        plt.show(block=True)


def perform_threshold_testing(csv_path, save_path=None, u_measurement=0.1, alpha=0.05):
    """
    Perform rigorous threshold testing for model validation metrics.
    
    Parameters:
    - csv_path: Path to temperature_comparison_data.csv
    - save_path: Path to save the CSV results (if None, prints to console)
    - u_measurement: Measurement uncertainty in °C (default: 0.1°C for thermistors)
    - alpha: Significance level for statistical tests (default: 0.05)
    
    Returns:
    - DataFrame with validation results
    """
    df = pd.read_csv(csv_path)
    
    # Get all thermistor IDs from column names
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    
    results = []
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract thermistor ID and position from column name
        parts = model_col_name.split('_')
        therm_id = int(parts[2])
        pos_str = parts[3].split()[0]
        x_pos_mm = float(pos_str.replace('x', '').replace('mm', ''))
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate metrics
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
        if np.sum(valid_mask) < 3:
            # Not enough data points
            results.append({
                'thermistor_id': therm_id,
                'position_mm': x_pos_mm,
                'n_data_points': np.sum(valid_mask),
                'rmse': np.nan,
                'mse': np.nan,
                'mean_residual': np.nan,
                'std_residual': np.nan,
                'rmse_valid': False,
                'mse_valid': False,
                'mean_residual_valid': False,
                'residuals_normal': np.nan,
                'chi_squared': np.nan,
                'chi_squared_pvalue': np.nan,
                't_statistic': np.nan,
                't_pvalue': np.nan,
                'pearson_r': np.nan,
                'overall_valid': False
            })
            continue
        
        T_model_valid = T_model[valid_mask]
        T_exp_valid = T_exp[valid_mask]
        residuals = T_model_valid - T_exp_valid
        n_points = len(residuals)
        
        # Calculate RMSE and Pearson correlation
        rmse, pearson_r = calculate_rmse_and_correlation(T_model_valid, T_exp_valid)
        
        # Calculate MSE
        mse = np.mean(residuals**2)
        
        # Calculate mean residual and standard error
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals, ddof=1)  # Sample standard deviation
        se_mean_residual = std_residual / np.sqrt(n_points)  # Standard error of mean
        
        # ===== THRESHOLD TESTING =====
        
        # 1. RMSE threshold: RMSE < 2 * u_measurement (95% coverage)
        rmse_threshold = 2.0 * u_measurement
        # Round to 3 decimal places before comparison for consistency with display
        rmse_rounded = np.round(rmse, 3)
        rmse_threshold_rounded = np.round(rmse_threshold, 3)
        rmse_valid = rmse_rounded <= rmse_threshold_rounded
        
        # 2. MSE threshold: MSE < (2 * u_measurement)^2
        mse_threshold = (2.0 * u_measurement)**2
        # Round to 3 decimal places before comparison for consistency with display
        mse_rounded = np.round(mse, 3)
        mse_threshold_rounded = np.round(mse_threshold, 3)
        mse_valid = mse_rounded <= mse_threshold_rounded
        
        # 3. Mean Residual constraint test: Independent threshold check
        # Mean residual threshold based on measurement uncertainty
        # |mean_residual| <= 1.5 * u_measurement
        mean_residual_threshold = 1.5 * u_measurement
        # Round to 3 decimal places before comparison for consistency with display
        mean_residual_rounded = np.round(np.abs(mean_residual), 3)
        mean_residual_threshold_rounded = np.round(mean_residual_threshold, 3)
        mean_residual_valid = mean_residual_rounded <= mean_residual_threshold_rounded
        
        # T-test: Test if mean residual is significantly different from zero (for information only)
        if n_points >= 3 and std_residual > 0:
            t_stat, t_pvalue = ttest_1samp(residuals, 0.0)
        else:
            t_stat = np.nan
            t_pvalue = np.nan
        
        # 4. Chi-squared goodness-of-fit test
        # χ² = Σ[(T_model - T_exp)² / σ²_measurement]
        chi_squared = np.sum((residuals**2) / (u_measurement**2))
        df_chi2 = n_points - 1  # Degrees of freedom (approximate, assuming 1 parameter)
        chi_squared_pvalue = 1.0 - chi2.cdf(chi_squared, df_chi2)
        chi_squared_valid = chi_squared_pvalue >= alpha
        
        # 5. Normality test for residuals
        # Use Shapiro-Wilk for small samples (< 5000), otherwise use D'Agostino-Pearson
        if n_points >= 3 and n_points < 5000:
            try:
                _, normality_pvalue = shapiro(residuals)
                residuals_normal = normality_pvalue >= alpha
            except:
                # Fallback to D'Agostino-Pearson test
                _, normality_pvalue = normaltest(residuals)
                residuals_normal = normality_pvalue >= alpha
        elif n_points >= 5000:
            _, normality_pvalue = normaltest(residuals)
            residuals_normal = normality_pvalue >= alpha
        else:
            normality_pvalue = np.nan
            residuals_normal = np.nan
        
        # Overall validity: All criteria must pass
        overall_valid = (rmse_valid and mse_valid and mean_residual_valid and 
                        chi_squared_valid and (residuals_normal if not np.isnan(residuals_normal) else True))
        
        results.append({
            'thermistor_id': therm_id,
            'position_mm': x_pos_mm,
            'n_data_points': n_points,
            'rmse': rmse,
            'rmse_threshold': rmse_threshold,
            'rmse_valid': rmse_valid,
            'mse': mse,
            'mse_threshold': mse_threshold,
            'mse_valid': mse_valid,
            'mean_residual': mean_residual,
            'mean_residual_threshold': mean_residual_threshold,
            'mean_residual_valid': mean_residual_valid,
            'std_residual': std_residual,
            't_statistic': t_stat if not np.isnan(t_stat) else np.nan,
            't_pvalue': t_pvalue if not np.isnan(t_pvalue) else np.nan,
            'chi_squared': chi_squared,
            'chi_squared_pvalue': chi_squared_pvalue,
            'chi_squared_valid': chi_squared_valid,
            'residuals_normal': residuals_normal if not np.isnan(residuals_normal) else np.nan,
            'normality_pvalue': normality_pvalue if not np.isnan(normality_pvalue) else np.nan,
            'pearson_r': pearson_r if not np.isnan(pearson_r) else np.nan,
            'overall_valid': overall_valid
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Calculate averaged row across all thermistors
    # Need to be careful about how different properties propagate
    avg_row = {}
    
    # Simple identifiers
    avg_row['thermistor_id'] = 'average'  # or 'all'
    avg_row['position_mm'] = df_results['position_mm'].mean()
    
    # Sum total data points
    avg_row['n_data_points'] = df_results['n_data_points'].sum()
    
    # Thresholds (same for all, take first)
    avg_row['rmse_threshold'] = df_results['rmse_threshold'].iloc[0]
    avg_row['mse_threshold'] = df_results['mse_threshold'].iloc[0]
    avg_row['mean_residual_threshold'] = df_results['mean_residual_threshold'].iloc[0]
    
    # Metrics: Use RMS for RMSE (root mean square of RMSEs), average for others
    # RMSE: RMS of individual RMSEs (conservative approach)
    valid_rmse = df_results['rmse'].dropna()
    if len(valid_rmse) > 0:
        avg_row['rmse'] = np.sqrt(np.mean(valid_rmse**2))
        # Uncertainty: Standard error of RMSE values
        avg_row['rmse_uncertainty'] = np.std(valid_rmse, ddof=1) / np.sqrt(len(valid_rmse)) if len(valid_rmse) > 1 else 0.0
    else:
        avg_row['rmse'] = np.nan
        avg_row['rmse_uncertainty'] = np.nan
    
    # MSE: Average MSE
    valid_mse = df_results['mse'].dropna()
    if len(valid_mse) > 0:
        avg_row['mse'] = valid_mse.mean()
        # Uncertainty: Standard error of MSE values
        avg_row['mse_uncertainty'] = np.std(valid_mse, ddof=1) / np.sqrt(len(valid_mse)) if len(valid_mse) > 1 else 0.0
    else:
        avg_row['mse'] = np.nan
        avg_row['mse_uncertainty'] = np.nan
    
    # Standard Residual: Pooled standard deviation (calculate first, needed for mean_residual_uncertainty)
    # Pooled std = sqrt(sum((n_i - 1) * std_i^2) / sum(n_i - 1))
    valid_std = df_results['std_residual'].dropna()
    valid_n = df_results.loc[valid_std.index, 'n_data_points']
    if len(valid_std) > 0:
        # Calculate pooled variance
        degrees_of_freedom = (valid_n - 1).sum()
        if degrees_of_freedom > 0:
            pooled_variance = ((valid_n - 1) * valid_std**2).sum() / degrees_of_freedom
            avg_row['std_residual'] = np.sqrt(pooled_variance)
        else:
            avg_row['std_residual'] = valid_std.mean()
    else:
        avg_row['std_residual'] = np.nan
    
    # Mean Residual: Weighted average by n_data_points
    valid_mean_res = df_results['mean_residual'].dropna()
    valid_n_points = df_results.loc[valid_mean_res.index, 'n_data_points']
    if len(valid_mean_res) > 0 and valid_n_points.sum() > 0:
        avg_row['mean_residual'] = np.average(valid_mean_res, weights=valid_n_points)
        # Uncertainty: Standard error from pooled std_residual (now available)
        if not np.isnan(avg_row.get('std_residual', np.nan)) and avg_row['n_data_points'] > 0:
            avg_row['mean_residual_uncertainty'] = avg_row['std_residual'] / np.sqrt(avg_row['n_data_points'])
        else:
            avg_row['mean_residual_uncertainty'] = np.std(valid_mean_res, ddof=1) / np.sqrt(len(valid_mean_res)) if len(valid_mean_res) > 1 else 0.0
    else:
        avg_row['mean_residual'] = np.nan
        avg_row['mean_residual_uncertainty'] = np.nan
    
    # Validity flags: Recalculate based on averaged values (not logical AND)
    # Round to 3 decimal places before comparison for consistency with display
    rmse_rounded = np.round(avg_row['rmse'], 3)
    rmse_threshold_rounded = np.round(avg_row['rmse_threshold'], 3)
    avg_row['rmse_valid'] = rmse_rounded <= rmse_threshold_rounded
    
    mse_rounded = np.round(avg_row['mse'], 3)
    mse_threshold_rounded = np.round(avg_row['mse_threshold'], 3)
    avg_row['mse_valid'] = mse_rounded <= mse_threshold_rounded
    
    mean_residual_rounded = np.round(np.abs(avg_row['mean_residual']), 3)
    mean_residual_threshold_rounded = np.round(avg_row['mean_residual_threshold'], 3)
    avg_row['mean_residual_valid'] = mean_residual_rounded <= mean_residual_threshold_rounded
    
    # For other flags, use logical AND
    avg_row['chi_squared_valid'] = df_results['chi_squared_valid'].all()
    avg_row['residuals_normal'] = df_results['residuals_normal'].dropna().all() if df_results['residuals_normal'].notna().any() else np.nan
    avg_row['overall_valid'] = df_results['overall_valid'].all()
    
    # Statistical tests: Need to recalculate from combined data
    # For t-test: Use weighted mean residual and pooled std
    if not np.isnan(avg_row['mean_residual']) and not np.isnan(avg_row['std_residual']) and avg_row['n_data_points'] > 0:
        se_mean = avg_row['std_residual'] / np.sqrt(avg_row['n_data_points'])
        if se_mean > 0:
            avg_row['t_statistic'] = avg_row['mean_residual'] / se_mean
            # Calculate p-value from t-distribution
            df_t = avg_row['n_data_points'] - 1
            avg_row['t_pvalue'] = 2 * (1 - t.cdf(abs(avg_row['t_statistic']), df_t))
        else:
            avg_row['t_statistic'] = np.nan
            avg_row['t_pvalue'] = np.nan
    else:
        avg_row['t_statistic'] = np.nan
        avg_row['t_pvalue'] = np.nan
    
    # Chi-squared: Sum of individual chi-squared values (if independent)
    valid_chi2 = df_results['chi_squared'].dropna()
    if len(valid_chi2) > 0:
        avg_row['chi_squared'] = valid_chi2.sum()
        # Recalculate p-value for summed chi-squared
        df_chi2_total = (df_results.loc[valid_chi2.index, 'n_data_points'] - 1).sum()
        avg_row['chi_squared_pvalue'] = 1.0 - chi2.cdf(avg_row['chi_squared'], df_chi2_total)
    else:
        avg_row['chi_squared'] = np.nan
        avg_row['chi_squared_pvalue'] = np.nan
    
    # Normality p-value: Can't meaningfully average, use minimum (most conservative)
    valid_norm_p = df_results['normality_pvalue'].dropna()
    avg_row['normality_pvalue'] = valid_norm_p.min() if len(valid_norm_p) > 0 else np.nan
    
    # Pearson correlation: Average (weighted by n_data_points)
    valid_pearson = df_results['pearson_r'].dropna()
    if len(valid_pearson) > 0:
        # Use weighted average by n_data_points
        valid_n_for_pearson = df_results.loc[valid_pearson.index, 'n_data_points']
        if valid_n_for_pearson.sum() > 0:
            avg_row['pearson_r'] = np.average(valid_pearson, weights=valid_n_for_pearson)
        else:
            avg_row['pearson_r'] = valid_pearson.mean()
        # Uncertainty: Standard error of Pearson r values
        avg_row['pearson_r_uncertainty'] = np.std(valid_pearson, ddof=1) / np.sqrt(len(valid_pearson)) if len(valid_pearson) > 1 else 0.0
    else:
        avg_row['pearson_r'] = np.nan
        avg_row['pearson_r_uncertainty'] = np.nan
    
    # T-test p-value uncertainty: Not typically reported, but we can use 0 or show confidence interval
    # For simplicity, we'll set it to 0 or calculate from t-statistic if needed
    avg_row['t_pvalue_uncertainty'] = 0.0  # p-values don't typically have uncertainties in this context
    
    # Append averaged row to DataFrame
    df_results = pd.concat([df_results, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save to CSV
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"\nThreshold testing results saved to: {save_path}")
        print(f"\nSummary:")
        print(f"  Total thermistors analyzed: {len(df_results)}")
        print(f"  Valid thermistors: {df_results['overall_valid'].sum()}")
        print(f"  Invalid thermistors: {(~df_results['overall_valid']).sum()}")
        print(f"\n  RMSE valid: {df_results['rmse_valid'].sum()}/{len(df_results)}")
        print(f"  MSE valid: {df_results['mse_valid'].sum()}/{len(df_results)}")
        print(f"  Mean Residual valid: {df_results['mean_residual_valid'].sum()}/{len(df_results)}")
        print(f"  Chi-squared valid: {df_results['chi_squared_valid'].sum()}/{len(df_results)}")
    else:
        print("\nThreshold Testing Results:")
        print(df_results.to_string(index=False))
    
    return df_results


def main():
    """Main function to generate plots from saved CSV data."""
    # Path to CSV file
    csv_path = Path('data/comparison/temperature_comparison_data.csv')
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("  Run heatpump.py first to generate the comparison data.")
        return
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} time points")
    
    # Create plots directory
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    validity_plots_dir = plots_dir / 'validity'
    validity_plots_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\nGenerating temperature comparison plots...")
    plot_temperatures_from_csv(csv_path, save_path=str(validity_plots_dir / 'temperature_comparison.png'))
    
    print("\nGenerating residual distribution plots...")
    plot_residuals_from_csv(csv_path, save_path=str(validity_plots_dir / 'residual_distributions.png'))
    
    print("\nGenerating RMSE and correlation plot...")
    plot_rmse_correlation_from_csv(csv_path, save_path=str(validity_plots_dir / 'rmse_correlation_vs_position.png'))
    
    print("\nGenerating mean residual vs position plot...")
    plot_mean_residual_vs_position(csv_path, save_path=str(validity_plots_dir / 'mean_residual_vs_position.png'))
    
    print("\nGenerating normalized bar chart plot...")
    plot_metrics_bar_chart_normalized(csv_path, save_path=str(validity_plots_dir / 'metrics_bar_chart_normalized.png'))
    
    print("\nPerforming threshold testing...")
    perform_threshold_testing(
        csv_path, 
        save_path=str(validity_plots_dir.parent / 'data' / 'comparison' / 'threshold_testing_results.csv'),
        u_measurement=0.1,  # Thermistor uncertainty: ±0.1°C
        alpha=0.05  # 95% confidence level
    )
    
    print(f"\nAll plots saved to: {validity_plots_dir}")


if __name__ == '__main__':
    main()

