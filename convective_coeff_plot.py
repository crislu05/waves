import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, griddata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import re

# Import functions from convection.py (only for loading data and thermistor positions)
from convection import load_dataset, get_thermistor_positions

# Import functions from transient.py for exponential fitting
from transient import exponential_model, fit_transient_single_series


def fit_steady_state_temperature(timestamp, temperature_data, data_type='model'):
    """
    Global function to extract steady-state temperature from time series data.
    Handles both experimental and model data with proper validation and bounds.
    
    Parameters:
    - timestamp: Time array
    - temperature_data: Temperature array
    - data_type: 'model' or 'experimental' (for different validation ranges)
    
    Returns:
    - T_inf: Steady-state temperature (°C), or None if fitting fails
    """
    # Remove NaN values
    valid_mask = np.isfinite(temperature_data) & np.isfinite(timestamp)
    if np.sum(valid_mask) < 10:
        return None
    
    timestamp_valid = timestamp[valid_mask]
    temp_valid = temperature_data[valid_mask]
    
    # Check if data is already at steady-state (small variation)
    temp_std = np.std(temp_valid)
    temp_mean = np.mean(temp_valid)
    temp_min = np.min(temp_valid)
    temp_max = np.max(temp_valid)
    
    # If variation is very small (< 0.1°C), treat as steady-state
    if temp_std < 0.1:
        return temp_mean
    
    # Try exponential fit with bounds to ensure reasonable values
    from scipy.optimize import curve_fit
    
    # Determine if data is increasing or decreasing
    temp_change = temp_valid[-1] - temp_valid[0]
    
    # Set reasonable bounds based on data type
    if data_type == 'model':
        # Model data should be in reasonable temperature range
        T_inf_min = max(0, temp_min - 10)  # Don't allow negative
        T_inf_max = min(50, temp_max + 10)  # Cap at reasonable max
    else:  # experimental
        # Experimental data can have wider range
        T_inf_min = max(-10, temp_min - 20)
        T_inf_max = min(100, temp_max + 20)
    
    # A can be positive or negative (for growth or decay)
    data_range = temp_max - temp_min
    A_min = -2 * data_range if temp_change > 0 else -data_range
    A_max = 2 * data_range if temp_change < 0 else data_range
    
    # tau should be positive and reasonable
    tau_min = 1.0
    tau_max = 10000.0
    
    # Initial guess
    T_inf0 = temp_mean
    A0 = temp_valid[0] - T_inf0
    if abs(A0) < 0.01:
        A0 = 0.1 if temp_change > 0 else -0.1
    tau0 = (timestamp_valid[-1] - timestamp_valid[0]) / 3.0
    
    try:
        popt, _ = curve_fit(
            exponential_model,
            timestamp_valid,
            temp_valid,
            p0=[T_inf0, A0, tau0],
            bounds=([T_inf_min, A_min, tau_min], [T_inf_max, A_max, tau_max]),
            maxfev=20000  # Increase max iterations
        )
        T_inf_fit, A_fit, tau_fit = popt
        
        # Validate fit result
        if (T_inf_min <= T_inf_fit <= T_inf_max and 
            tau_fit > 0 and np.isfinite(T_inf_fit) and 
            np.isfinite(A_fit) and np.isfinite(tau_fit)):
            
            # Additional reasonable range check
            if data_type == 'model':
                if 0 <= T_inf_fit <= 50:
                    return T_inf_fit
            else:  # experimental
                if -10 <= T_inf_fit <= 100:
                    return T_inf_fit
        
        # Fit produced unreasonable values, use mean as fallback
        return temp_mean
    except Exception:
        # curve_fit failed, try the original function (suppress warnings and print statements)
        import warnings
        import sys
        from io import StringIO
        
        # Suppress print statements from fit_transient_single_series
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = fit_transient_single_series(timestamp_valid, temp_valid)
        finally:
            sys.stdout = old_stdout
        
        if fit_result is not None and not np.isnan(fit_result[0]):
            T_inf_fit, _, tau_fit, _ = fit_result
            
            # Validate fit result
            if tau_fit > 0 and np.isfinite(T_inf_fit):
                if data_type == 'model':
                    if 0 <= T_inf_fit <= 50:
                        return T_inf_fit
                else:  # experimental
                    if -10 <= T_inf_fit <= 100:
                        return T_inf_fit
        
        # Fit failed or produced unreasonable values, use mean as fallback
        return temp_mean


def load_steady_state_data_from_csv():
    """
    Load steady state temperature data from CSV file.
    
    Returns:
    - h_array: Array of h values (W/(m²·K))
    - x_grid_mm: Position array in mm
    - T_profiles_array: Temperature profiles array, shape (n_h, n_positions) in °C
    """
    csv_path = Path('data/convective/steady_state_3d_data.csv')
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Run convective_plot.py first to generate the data.")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get unique h and x values, ensuring ascending order (0 to 40 for x)
    h_array = np.sort(df['h_W_per_m2K'].unique())
    x_grid_mm = np.sort(df['x_mm'].unique())  # Should be 0 to 40 in ascending order
    
    # Reshape temperature data into 2D array: (n_h, n_positions)
    n_h = len(h_array)
    n_positions = len(x_grid_mm)
    T_profiles_array = np.zeros((n_h, n_positions))
    
    for i, h_val in enumerate(h_array):
        for j, x_pos in enumerate(x_grid_mm):
            # Find matching row
            mask = (df['h_W_per_m2K'] == h_val) & (df['x_mm'] == x_pos)
            if mask.sum() > 0:
                T_profiles_array[i, j] = df.loc[mask, 'temperature_C'].iloc[0]
            else:
                print(f"Warning: No data found for h={h_val:.2f}, x={x_pos:.2f}")
    
    return h_array, x_grid_mm, T_profiles_array


def plot_steady_state_3d_from_data(h_array, x_grid_mm, T_profiles_array):
    """
    Fixed 3D plot function:
    1. Ensures X-axis is 0 to 40 (ascending).
    2. Uses specific viewing angles and padding to keep Z-label visible.
    """
    # 1. FIX X-AXIS ORIENTATION: Force ascending order (0 to 40)
    # This ensures x_grid_mm starts at 0, and T_profiles are reordered accordingly
    sort_idx = np.argsort(x_grid_mm)
    x_grid_mm_sorted = x_grid_mm[sort_idx]
    T_profiles_array_sorted = T_profiles_array[:, sort_idx]

    print(f"Plotting Position range: {x_grid_mm_sorted[0]} to {x_grid_mm_sorted[-1]} mm")
    
    # Create meshgrid
    X_mesh, H_mesh = np.meshgrid(x_grid_mm_sorted, h_array)
    T_mesh = T_profiles_array_sorted 
    
    # Larger figure size to help with label clearance
    fig = plt.figure(figsize=(14, 10)) 
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X_mesh, H_mesh, T_mesh, cmap='coolwarm', 
                          alpha=0.9, linewidth=0, antialiased=True)
    
    # 2. IMPROVE VISUALIZATION PARAMETERS
    # Set labels with padding to avoid overlapping with tick marks
    ax.set_xlabel('Position along rod (mm)', fontsize=20, labelpad=15)
    ax.set_ylabel('Convective Coefficient $h$ (W/m²K)', fontsize=20, labelpad=15)
    
    # Set Z-axis label with significant padding to ensure it's not cut off
    ax.set_zlabel('Steady State Temp (°C)', fontsize=20, labelpad=25)
    
    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    
    # 3. SET VIEWING ANGLE
    # azim=135 views from the left-hand side (LHS)
    # elev=30 gives enough 'height' to see the temperature drop along x
    ax.view_init(elev=30, azim=135)
    
    # Force X-axis range
    ax.set_xlim(0, 40)
    
    # Formatting
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('Temperature (°C)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    # Use tight_layout to help with spacing
    plt.tight_layout()
    
    save_path = Path('plots/convective/steady_state_3d.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Corrected 3D plot saved to: {save_path}")
    plt.close()


def plot_steady_state_3d():
    """
    Generate a 3D plot of steady state temperature vs position and convective coefficient h.
    Loads data from CSV instead of solving PDE.
    """
    print("Loading steady state data from CSV...")
    h_array, x_grid_mm, T_profiles_array = load_steady_state_data_from_csv()
    
    print(f"Loaded data: {len(h_array)} h values, {len(x_grid_mm)} positions")
    
    # Use shared plotting function
    plot_steady_state_3d_from_data(h_array, x_grid_mm, T_profiles_array)
    
    # Generate 2D plot at middle of rod (x = L/2)
    x_grid = x_grid_mm / 1000.0  # Convert to meters for compatibility
    plot_steady_state_2d_middle(h_array, T_profiles_array, x_grid)


def exponential_decay_model(h, T_inf, A, h_tau):
    """
    Exponential decay model for temperature vs convective coefficient.
    T(h) = T_inf + A * exp(-h / h_tau)
    
    Parameters:
    - h: Convective heat transfer coefficient
    - T_inf: Asymptotic temperature at high h
    - A: Amplitude (temperature difference from T_inf at h=0)
    - h_tau: Decay constant (h value at which temperature drops by 1/e)
    """
    return T_inf + A * np.exp(-h / h_tau)


def plot_steady_state_2d_middle(h_array, T_profiles_array, x_grid):
    """
    Generate a 2D plot of steady state temperature vs h at the middle of the rod.
    Fits an exponential decay curve and extracts the decay constant.
    
    Parameters:
    - h_array: Array of h values
    - T_profiles_array: Temperature profiles array, shape (n_h, n_positions)
    - x_grid: Position array in meters
    """
    # Find middle position (x = L/2)
    L = x_grid[-1]  # Total length in meters
    x_middle = L / 2.0
    
    # Find index closest to middle
    middle_idx = np.argmin(np.abs(x_grid - x_middle))
    x_middle_actual = x_grid[middle_idx] * 1000  # Convert to mm for display
    
    # Extract temperature at middle position for all h values
    T_middle = T_profiles_array[:, middle_idx]  # Shape: (n_h,)
    
    # Fit exponential decay: T(h) = T_inf + A * exp(-h / h_tau)
    # Initial guess: T_inf = min(T), A = max(T) - min(T), h_tau = mean(h)
    T_min = np.min(T_middle)
    T_max = np.max(T_middle)
    h_mean = np.mean(h_array)
    
    initial_guess = [T_min, T_max - T_min, h_mean]
    
    try:
        popt, pcov = curve_fit(exponential_decay_model, h_array, T_middle, 
                              p0=initial_guess, maxfev=10000)
        T_inf_fit, A_fit, h_tau_fit = popt
        
        # Calculate uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        T_inf_unc, A_unc, h_tau_unc = perr
        
        # Generate smooth fit curve
        h_fit = np.linspace(h_array[0], h_array[-1], 200)
        T_fit = exponential_decay_model(h_fit, T_inf_fit, A_fit, h_tau_fit)
        
        fit_success = True
        print(f"\nExponential decay fit results:")
        print(f"  T_inf (asymptotic temperature) = {T_inf_fit:.3f} ± {T_inf_unc:.3f} °C")
        print(f"  A (amplitude) = {A_fit:.3f} ± {A_unc:.3f} °C")
        print(f"  h_tau (decay constant) = {h_tau_fit:.2f} ± {h_tau_unc:.2f} W/(m²·K)")
    except Exception as e:
        print(f"Warning: Exponential decay fit failed: {e}")
        fit_success = False
        h_fit = None
        T_fit = None
        h_tau_fit = None
        h_tau_unc = None
    
    # Create 2D plot (similar style to transient.py)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot fit curve only (no data points) - solid line with label (no stats)
    if fit_success:
        ax.plot(h_fit, T_fit, '-', linewidth=2, color='#C0392B', alpha=0.8,
               label='Exponential Fit')
    
    ax.set_xlabel('Convective Heat Transfer Coefficient, $h$ (W/(m²·K))', fontsize=20)
    ax.set_ylabel('Steady State Temperature (°C)', fontsize=20)
    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # Title removed
    ax.grid(True, alpha=0.3)
    if fit_success:
        ax.legend(loc='best', fontsize=10)
    
    # Save plot
    save_path = Path('plots/convective/steady_state_vs_h_middle.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 2D plot at middle of rod saved to: {save_path}")
    plt.close()
    
    # Return decay constant if fit was successful
    if fit_success:
        return h_tau_fit, h_tau_unc
    else:
        return None, None


def load_h_values_from_cooling():
    """
    Load h values from h_vs_voltage.csv.
    
    Returns:
    - Dictionary mapping fan voltage (int) to h value (float)
    """
    h_vs_voltage_file = Path('data/cooling/h_vs_voltage.csv')
    h_dict = {}
    
    if h_vs_voltage_file.exists():
        h_data = pd.read_csv(h_vs_voltage_file)
        for _, row in h_data.iterrows():
            voltage = int(row['voltage'])
            h = float(row['h_after_correction'])
            h_dict[voltage] = h
    else:
        print(f"Warning: {h_vs_voltage_file} not found.")
        print(f"  Run cooling.py first to generate h values.")
    
    return h_dict


def plot_experimental_steady_state_3d():
    """
    Generate a 3D plot of experimental steady-state temperature vs position and convective coefficient h.
    Uses discrete thermistor positions and discrete fan voltages (h values).
    """
    print("Loading experimental steady-state data from temperature comparison CSV files...")
    
    # Get thermistor positions
    thermistor_positions = get_thermistor_positions()
    thermistor_ids = sorted(thermistor_positions.keys())
    positions_mm = np.array([thermistor_positions[tid] * 1000 for tid in thermistor_ids])  # Convert to mm
    
    # Find all temperature comparison CSV files
    comparison_dir = Path('data/convective')
    if not comparison_dir.exists():
        print(f"Error: {comparison_dir} not found.")
        return
    
    csv_files = list(comparison_dir.glob('temperature_comparison_h*.csv'))
    if not csv_files:
        print(f"Error: No temperature comparison CSV files found in {comparison_dir}")
        print("  Please run convective_plot.py first to generate comparison data.")
        return
    
    # Extract steady-state temperatures for each h value/thermistor combination
    h_values = []
    T_steady_array = []  # Will be list of lists: [h_value, [T_therm0, T_therm1, ...]]
    
    for csv_file in sorted(csv_files):
        # Extract h value from filename (e.g., "temperature_comparison_h23p7.csv" -> h=23.7)
        match = re.search(r'h(\d+p?\d*)\.csv', csv_file.name)
        if not match:
            print(f"  Warning: Could not extract h value from {csv_file.name}, skipping")
            continue
        
        h_str = match.group(1).replace('p', '.')
        try:
            h_value = float(h_str)
        except ValueError:
            print(f"  Warning: Could not parse h value from {csv_file.name}, skipping")
            continue
        
        print(f"  Processing {csv_file.name} (h={h_value:.2f} W/(m²·K))...")
        
        # Load CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"    Error loading {csv_file.name}: {e}")
            continue
        
        # Extract time and experimental temperature columns
        time_col = 'Time (s)'
        if time_col not in df.columns:
            print(f"    Error: Time column not found in {csv_file.name}")
            continue
        
        timestamp = df[time_col].values
        
        # Extract steady-state temperature for each thermistor using exponential fitting
        T_steady_thermistors = []
        for therm_id in thermistor_ids:
            # Find experimental column for this thermistor
            exp_col = f'T_exp_{therm_id}_x'
            exp_col_name = [col for col in df.columns if col.startswith(exp_col)]
            
            if not exp_col_name:
                print(f"    Warning: No experimental data found for thermistor {therm_id}")
                T_steady_thermistors.append(np.nan)
                continue
            
            exp_col_name = exp_col_name[0]
            therm_data = df[exp_col_name].values
            
            # Remove NaN values for fitting
            valid_mask = np.isfinite(therm_data) & np.isfinite(timestamp)
            if np.sum(valid_mask) < 10:
                print(f"    Warning: Not enough valid data for thermistor {therm_id}")
                T_steady_thermistors.append(np.nan)
                continue
            
            timestamp_valid = timestamp[valid_mask]
            therm_data_valid = therm_data[valid_mask]
            
            # Fit exponential decay to extract steady-state temperature
            T_inf = fit_steady_state_temperature(timestamp_valid, therm_data_valid, data_type='experimental')
            if T_inf is not None:
                T_steady_thermistors.append(T_inf)
            else:
                # Fallback: use mean of last 10% of data if fitting fails
                n_last = max(1, int(len(therm_data_valid) * 0.1))
                T_steady_thermistors.append(np.mean(therm_data_valid[-n_last:]))
        
        # Only add if we have valid data for at least some thermistors
        if not all(np.isnan(T_steady_thermistors)):
            h_values.append(h_value)
            T_steady_array.append(T_steady_thermistors)
    
    if not h_values:
        print("Error: No valid data found!")
        return
    
    # Convert to numpy arrays
    h_array = np.array(h_values)
    T_profiles_array = np.array(T_steady_array)  # Shape: (n_h, n_thermistors)
    
    print(f"\nLoaded experimental data: {len(h_array)} h values, {len(positions_mm)} thermistor positions")
    
    # Create 3D plot similar to model version
    plot_experimental_steady_state_3d_from_data(h_array, positions_mm, T_profiles_array)


def plot_experimental_steady_state_3d_from_data(h_array, positions_mm, T_profiles_array):
    """
    Plot experimental 3D steady-state temperature vs position and h.
    
    Parameters:
    - h_array: Array of h values (W/(m²·K))
    - positions_mm: Array of thermistor positions in mm
    - T_profiles_array: Temperature array, shape (n_h, n_thermistors) in °C
    """
    # Ensure positions are sorted
    sort_idx = np.argsort(positions_mm)
    positions_mm_sorted = positions_mm[sort_idx]
    T_profiles_array_sorted = T_profiles_array[:, sort_idx]
    
    # Create meshgrid from actual discrete data points (no interpolation)
    # X-axis: discrete thermistor positions
    # Y-axis: discrete h values
    H_mesh, X_mesh = np.meshgrid(h_array, positions_mm_sorted)
    T_mesh = T_profiles_array_sorted.T  # Transpose to match meshgrid shape: (n_positions, n_h)
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface using only discrete data points (no interpolation)
    # This creates a surface that connects only the actual measured points
    surf = ax.plot_surface(X_mesh, H_mesh, T_mesh, cmap='coolwarm', 
                          alpha=0.9, linewidth=0.5, antialiased=True, edgecolor='black')
    
    # Also plot discrete points as scatter for emphasis
    for i, h_val in enumerate(h_array):
        ax.scatter(positions_mm_sorted, [h_val] * len(positions_mm_sorted), 
                  T_profiles_array_sorted[i, :], 
                  color='black', s=30, alpha=0.8, zorder=5)
    
    # Set labels with padding
    ax.set_xlabel('Position along rod (mm)', fontsize=20, labelpad=15)
    ax.set_ylabel('Convective Coefficient $h$ (W/m²K)', fontsize=20, labelpad=15)
    ax.set_zlabel('Steady State Temp (°C)', fontsize=20, labelpad=25)
    
    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    
    # Set viewing angle (same as model version)
    ax.view_init(elev=30, azim=135)
    
    # Force X-axis range
    ax.set_xlim(positions_mm_sorted.min(), positions_mm_sorted.max())
    
    # Formatting
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
    cbar.set_label('Temperature (°C)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    save_path = Path('plots/convective/steady_state_3d_experimental.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Experimental 3D plot saved to: {save_path}")
    plt.close()


def plot_residual_heatmap():
    """
    Generate a 2D heatmap of residual (experimental - model) vs position and h.
    Uses discrete thermistor positions and discrete h values.
    """
    print("Loading data for residual heatmap...")
    
    # Get thermistor positions
    thermistor_positions = get_thermistor_positions()
    thermistor_ids = sorted(thermistor_positions.keys())
    positions_mm = np.array([thermistor_positions[tid] * 1000 for tid in thermistor_ids])  # Convert to mm
    
    # Find all temperature comparison CSV files
    comparison_dir = Path('data/convective')
    if not comparison_dir.exists():
        print(f"Error: {comparison_dir} not found.")
        return
    
    csv_files = list(comparison_dir.glob('temperature_comparison_h*.csv'))
    if not csv_files:
        print(f"Error: No temperature comparison CSV files found in {comparison_dir}")
        return
    
    # Extract steady-state temperatures for each h value/thermistor combination
    h_values = []
    T_exp_steady_array = []  # Experimental steady-state temperatures
    T_model_steady_array = []  # Model steady-state temperatures
    
    for csv_file in sorted(csv_files):
        # Extract h value from filename
        match = re.search(r'h(\d+p?\d*)\.csv', csv_file.name)
        if not match:
            continue
        
        h_str = match.group(1).replace('p', '.')
        try:
            h_value = float(h_str)
        except ValueError:
            continue
        
        print(f"  Processing {csv_file.name} (h={h_value:.2f} W/(m²·K))...")
        
        # Load CSV file
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"    Error loading {csv_file.name}: {e}")
            continue
        
        # Extract time column
        time_col = 'Time (s)'
        if time_col not in df.columns:
            continue
        
        timestamp = df[time_col].values
        
        # Extract steady-state temperatures for experimental and model
        T_exp_steady_thermistors = []
        T_model_steady_thermistors = []
        
        for therm_id in thermistor_ids:
            # Find experimental and model columns
            exp_col = f'T_exp_{therm_id}_x'
            model_col = f'T_model_{therm_id}_x'
            
            exp_col_name = [col for col in df.columns if col.startswith(exp_col)]
            model_col_name = [col for col in df.columns if col.startswith(model_col)]
            
            if not exp_col_name or not model_col_name:
                T_exp_steady_thermistors.append(np.nan)
                T_model_steady_thermistors.append(np.nan)
                continue
            
            exp_col_name = exp_col_name[0]
            model_col_name = model_col_name[0]
            
            exp_data = df[exp_col_name].values
            model_data = df[model_col_name].values
            
            # Remove NaN values for fitting
            valid_mask_exp = np.isfinite(exp_data) & np.isfinite(timestamp)
            valid_mask_model = np.isfinite(model_data) & np.isfinite(timestamp)
            
            # Extract steady-state for experimental
            if np.sum(valid_mask_exp) >= 10:
                timestamp_exp = timestamp[valid_mask_exp]
                exp_data_valid = exp_data[valid_mask_exp]
                T_inf_exp = fit_steady_state_temperature(timestamp_exp, exp_data_valid, data_type='experimental')
                if T_inf_exp is not None:
                    T_exp_steady_thermistors.append(T_inf_exp)
                else:
                    n_last = max(1, int(len(exp_data_valid) * 0.1))
                    T_exp_steady_thermistors.append(np.mean(exp_data_valid[-n_last:]))
            else:
                T_exp_steady_thermistors.append(np.nan)
            
            # Extract steady-state for model
            if np.sum(valid_mask_model) >= 10:
                timestamp_model = timestamp[valid_mask_model]
                model_data_valid = model_data[valid_mask_model]
                T_inf_model = fit_steady_state_temperature(timestamp_model, model_data_valid, data_type='model')
                if T_inf_model is not None:
                    T_model_steady_thermistors.append(T_inf_model)
                else:
                    n_last = max(1, int(len(model_data_valid) * 0.1))
                    T_model_steady_thermistors.append(np.mean(model_data_valid[-n_last:]))
            else:
                T_model_steady_thermistors.append(np.nan)
        
        # Only add if we have valid data
        if not all(np.isnan(T_exp_steady_thermistors)) and not all(np.isnan(T_model_steady_thermistors)):
            h_values.append(h_value)
            T_exp_steady_array.append(T_exp_steady_thermistors)
            T_model_steady_array.append(T_model_steady_thermistors)
    
    if not h_values:
        print("Error: No valid data found!")
        return
    
    # Convert to numpy arrays
    h_array = np.array(h_values)
    T_exp_array = np.array(T_exp_steady_array)  # Shape: (n_h, n_thermistors)
    T_model_array = np.array(T_model_steady_array)  # Shape: (n_h, n_thermistors)
    
    # Calculate residuals: experimental - model
    residuals = T_exp_array - T_model_array  # Shape: (n_h, n_thermistors)
    
    # Ensure positions are sorted
    sort_idx = np.argsort(positions_mm)
    positions_mm_sorted = positions_mm[sort_idx]
    residuals_sorted = residuals[:, sort_idx]  # Shape: (n_h, n_thermistors)
    
    print(f"\nCalculated residuals for {len(h_array)} h values and {len(positions_mm_sorted)} positions")
    
    # Create 2D heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create meshgrid for heatmap
    H_mesh, X_mesh = np.meshgrid(h_array, positions_mm_sorted)
    
    # Use absolute values for viridis colormap (similar to metrics heatmaps)
    abs_residuals = np.abs(residuals_sorted)
    
    # Find global min and max for log scale
    valid_residuals = abs_residuals[np.isfinite(abs_residuals)]
    if len(valid_residuals) > 0:
        vmin = np.nanmin(valid_residuals)
        vmax = np.nanmax(valid_residuals)
        # Ensure vmin is positive for log scale
        if vmin <= 0:
            vmin = np.percentile(valid_residuals[valid_residuals > 0], 1) if np.any(valid_residuals > 0) else 0.001
    else:
        vmin = 0.001
        vmax = 1.0
    
    # Plot heatmap with viridis colormap and log scale
    im = ax.pcolormesh(H_mesh, X_mesh, abs_residuals.T, 
                      cmap='viridis', shading='auto',
                      norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    
    # Add colorbar with simplified label (same style as metrics heatmaps)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label('°C', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Set labels
    ax.set_xlabel('Convective Coefficient $h$ (W/(m²·K))', fontsize=20)
    ax.set_ylabel('Position along rod (mm)', fontsize=20)
    
    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Set ticks to show discrete values
    ax.set_xticks(h_array)
    ax.set_yticks(positions_mm_sorted)
    
    plt.tight_layout()
    
    save_path = Path('plots/convective/residual_heatmap.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Residual heatmap saved to: {save_path}")
    plt.close()
    
    # Also generate 3D bar chart version
    plot_residual_3d_barchart(h_array, positions_mm_sorted, abs_residuals, vmin, vmax)


def plot_residual_3d_barchart(h_array, positions_mm, abs_residuals, vmin, vmax):
    """
    Plot residual data as a 3D bar chart.
    
    Parameters:
    - h_array: Array of h values (W/(m²·K))
    - positions_mm: Array of thermistor positions in mm
    - abs_residuals: Absolute residual array, shape (n_h, n_positions)
    - vmin, vmax: Min and max values for color scale
    """
    # Create figure with 3D axes
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for bar positions
    H_mesh, X_mesh = np.meshgrid(h_array, positions_mm)
    
    # Flatten arrays for bar plotting
    h_flat = H_mesh.flatten()
    x_flat = X_mesh.flatten()
    residual_flat = abs_residuals.T.flatten()  # Transpose to match meshgrid
    
    # Set bar dimensions
    dx = np.min(np.diff(np.unique(h_array))) * 0.6 if len(np.unique(h_array)) > 1 else 1.0
    dy = np.min(np.diff(np.unique(positions_mm))) * 0.6 if len(np.unique(positions_mm)) > 1 else 1.0
    
    # Normalize residuals for color mapping (log scale)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    colors = plt.cm.viridis(norm(residual_flat))
    
    # Plot 3D bars
    ax.bar3d(h_flat - dx/2, x_flat - dy/2, np.zeros_like(residual_flat),
             dx, dy, residual_flat,
             color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set labels
    ax.set_xlabel('Convective Coefficient $h$ (W/(m²·K))', fontsize=20, labelpad=10)
    ax.set_ylabel('Position along rod (mm)', fontsize=20, labelpad=10)
    ax.set_zlabel('|Residual| (°C)', fontsize=20, labelpad=10)
    
    # Increase tick label font size
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    
    # Set viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('|Residual| (°C)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    save_path = Path('plots/convective/residual_3d_barchart.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Residual 3D bar chart saved to: {save_path}")
    plt.close()


def plot_all_fits_4v_fan():
    """
    Plot all thermistor exponential fits (experimental and model) for 4V fan.
    Shows time series, exponential fits, and extracted T_inf values for all thermistors.
    """
    print("Generating all fits plot for 4V fan...")
    
    # Load h values to find h for 4V fan
    h_dict = load_h_values_from_cooling()
    if 4 not in h_dict:
        print("Error: No h value found for 4V fan. Please check h_vs_voltage.csv")
        return
    
    h_value = h_dict[4]
    print(f"  Using h={h_value:.2f} W/(m²·K) for 4V fan")
    
    # Find temperature comparison CSV file for this h value
    comparison_dir = Path('data/convective')
    h_str = f"{h_value:.1f}".replace('.', 'p')
    csv_file = comparison_dir / f'temperature_comparison_h{h_str}.csv'
    
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    print(f"  Loading {csv_file.name}")
    
    # Get thermistor positions
    thermistor_positions = get_thermistor_positions()
    thermistor_ids = sorted(thermistor_positions.keys())
    
    # Load CSV file
    df = pd.read_csv(csv_file)
    time_col = 'Time (s)'
    timestamp = df[time_col].values
    
    # Create subplots: 2 columns, 4 rows (for 8 thermistors)
    n_cols = 2
    n_rows = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    axes = axes.flatten()
    
    # Process each thermistor
    for idx, therm_id in enumerate(thermistor_ids):
        ax = axes[idx]
        
        # Find experimental and model columns
        exp_col = f'T_exp_{therm_id}_x'
        model_col = f'T_model_{therm_id}_x'
        
        exp_col_name = [col for col in df.columns if col.startswith(exp_col)]
        model_col_name = [col for col in df.columns if col.startswith(model_col)]
        
        if not exp_col_name or not model_col_name:
            ax.text(0.5, 0.5, f'No data for Thermistor {therm_id}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Thermistor {therm_id}', fontsize=11, fontweight='bold')
            continue
        
        exp_col_name = exp_col_name[0]
        model_col_name = model_col_name[0]
        
        exp_data = df[exp_col_name].values
        model_data = df[model_col_name].values
        
        # Extract position from column name
        x_pos_mm = float(exp_col_name.split('_x')[1].split('mm')[0])
        
        # Remove NaN values for fitting
        valid_mask_exp = np.isfinite(exp_data) & np.isfinite(timestamp)
        valid_mask_model = np.isfinite(model_data) & np.isfinite(timestamp)
        
        timestamp_exp = timestamp[valid_mask_exp]
        exp_data_valid = exp_data[valid_mask_exp]
        timestamp_model = timestamp[valid_mask_model]
        model_data_valid = model_data[valid_mask_model]
        
        # Fit exponential decay to extract steady-state
        T_inf_exp = None
        T_inf_model = None
        fit_exp = None
        fit_model = None
        t_fit_exp = None
        t_fit_model = None
        
        # Experimental fit
        T_inf_exp = None
        fit_exp = None
        t_fit_exp = None
        if len(timestamp_exp) >= 10:
            T_inf_exp = fit_steady_state_temperature(timestamp_exp, exp_data_valid, data_type='experimental')
            if T_inf_exp is not None:
                # Get full fit parameters for plotting
                try:
                    fit_result = fit_transient_single_series(timestamp_exp, exp_data_valid)
                    if fit_result is not None and not np.isnan(fit_result[0]):
                        _, A_exp, tau_exp, _ = fit_result
                        # Generate fit curve
                        t_fit_exp = np.linspace(timestamp_exp.min(), timestamp_exp.max(), 200)
                        fit_exp = exponential_model(t_fit_exp, T_inf_exp, A_exp, tau_exp)
                except:
                    pass
        
        # Model fit
        T_inf_model = None
        fit_model = None
        t_fit_model = None
        if len(timestamp_model) >= 10:
            T_inf_model = fit_steady_state_temperature(timestamp_model, model_data_valid, data_type='model')
            if T_inf_model is not None:
                # Get full fit parameters for plotting
                try:
                    fit_result = fit_transient_single_series(timestamp_model, model_data_valid)
                    if fit_result is not None and not np.isnan(fit_result[0]):
                        _, A_model, tau_model, _ = fit_result
                        # Generate fit curve
                        t_fit_model = np.linspace(timestamp_model.min(), timestamp_model.max(), 200)
                        fit_model = exponential_model(t_fit_model, T_inf_model, A_model, tau_model)
                except:
                    # If fit fails, just use flat line at T_inf
                    t_fit_model = np.linspace(timestamp_model.min(), timestamp_model.max(), 200)
                    fit_model = np.full_like(t_fit_model, T_inf_model)
        
        # Plot experimental data
        ax.plot(timestamp_exp, exp_data_valid, 'o', color='blue', markersize=2, 
               alpha=0.5, label='Experimental')
        
        # Plot model data
        ax.plot(timestamp_model, model_data_valid, 's', color='red', markersize=2, 
               alpha=0.5, label='Model')
        
        # Plot exponential fits
        if fit_exp is not None:
            ax.plot(t_fit_exp, fit_exp, '-', color='blue', linewidth=2, 
                   label=f'Exp: $T_\\infty$={T_inf_exp:.2f}°C', alpha=0.8)
            ax.axhline(y=T_inf_exp, color='blue', linestyle='--', linewidth=1, alpha=0.4)
        
        if fit_model is not None:
            ax.plot(t_fit_model, fit_model, '-', color='red', linewidth=2, 
                   label=f'Model: $T_\\infty$={T_inf_model:.2f}°C', alpha=0.8)
            ax.axhline(y=T_inf_model, color='red', linestyle='--', linewidth=1, alpha=0.4)
        
        # Calculate residual if both are available
        if T_inf_exp is not None and T_inf_model is not None:
            residual = T_inf_exp - T_inf_model
            ax.text(0.02, 0.98, f'Residual: {residual:.3f}°C', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Thermistor {therm_id} (x={x_pos_mm:.1f}mm)', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    # Overall title
    fig.suptitle(f'Exponential Fits for All Thermistors (4V Fan, h={h_value:.2f} W/(m²·K))', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    save_path = Path('plots/convective/all_fits_4v_fan.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ All fits plot for 4V fan saved to: {save_path}")
    plt.close()


def main():
    """Main function to generate plots."""
    # Generate model steady-state 3D plot
    try:
        plot_steady_state_3d()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Please run convective_plot.py first to generate the steady state data CSV.")
    
    # Generate experimental steady-state 3D plot
    print("\n" + "="*60)
    print("Generating experimental steady-state 3D plot...")
    print("="*60)
    plot_experimental_steady_state_3d()
    
    # Generate residual heatmap
    print("\n" + "="*60)
    print("Generating residual heatmap...")
    print("="*60)
    plot_residual_heatmap()
    
    # Generate all fits for 4V fan
    print("\n" + "="*60)
    print("Generating all fits for 4V fan...")
    print("="*60)
    plot_all_fits_4v_fan()


if __name__ == '__main__':
    main()

