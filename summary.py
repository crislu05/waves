"""
Summary table combining validation metrics and phase/amplitude coefficients.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_1samp

# Import functions from other modules
from convective_validity import (
    load_h_values_from_cooling,
    find_csv_for_h_value,
    calculate_metrics_from_csv
)
from phase_and_amplitude import (
    load_comparison_data,
    analyse_temperature_data,
    estimate_D_from_gamma,
    estimate_D_from_phase
)
# Import improved fitting function from convective_coeff_plot.py
from convective_coeff_plot import fit_steady_state_temperature
from transient import exponential_model, fit_transient_single_series
# Import numerical integration functions from convection.py
from convection import (
    load_dataset,
    get_thermistor_positions,
    solve_coupled_heat_pump
)
from scipy.interpolate import interp1d


def plot_curve_fits(timestamp, temps_exp, temps_model, positions, 
                    analysis_exp, analysis_model,
                    alpha_exp, alpha_exp_unc, alpha_model, alpha_model_unc,
                    beta_exp, beta_exp_unc, beta_model, beta_model_unc,
                    tau_exp_dict, tau_model_dict, thermistor_positions_dict,
                    intercept_exp_phi=None, intercept_model_phi=None,
                    save_dir=None):
    """
    Plot curve fitting results for α, β, and τ.
    
    Parameters:
    - timestamp: Time array
    - temps_exp: Experimental temperatures (n_time, n_thermistors)
    - temps_model: Model temperatures (n_time, n_thermistors)
    - positions: Array of thermistor positions in meters
    - analysis_exp, analysis_model: Phase/amplitude analysis results
    - alpha_exp, alpha_exp_unc, etc.: Fitted coefficients
    - tau_exp_list, tau_model_list: Lists of tau values for each thermistor
    - thermistor_positions_dict: Dictionary mapping thermistor_id to position
    - save_dir: Directory to save plots
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Attenuation (γ) vs position with exponential fit (for α)
    ax1 = plt.subplot(2, 3, 1)
    x_exp = positions
    mask_exp = (analysis_exp["gamma"] > 0.05) & np.isfinite(analysis_exp["gamma"])
    mask_exp[0] = False  # Skip thermistor 0
    mask_model = (analysis_model["gamma"] > 0.05) & np.isfinite(analysis_model["gamma"])
    mask_model[0] = False
    
    if np.any(mask_exp):
        x_fit_exp = x_exp[mask_exp]
        gamma_fit_exp = analysis_exp["gamma"][mask_exp]
        ax1.semilogy(x_fit_exp * 1000, gamma_fit_exp, 'o', color='#2980B9', 
                    markersize=8, label='Experimental', alpha=0.7)
        # Plot fit: γ = A * exp(-α*x)
        if not np.isnan(alpha_exp):
            x_fit_plot = np.linspace(x_fit_exp[0], x_fit_exp[-1], 100)
            A_exp = np.exp(np.polyfit(x_fit_exp, np.log(gamma_fit_exp), 1)[1])
            gamma_fit_plot = A_exp * np.exp(-alpha_exp * x_fit_plot)
            ax1.semilogy(x_fit_plot * 1000, gamma_fit_plot, '--', color='#2980B9', 
                        linewidth=2, label=f'Exp fit: α={alpha_exp:.2f}±{alpha_exp_unc:.2f} m⁻¹')
    
    if np.any(mask_model):
        x_fit_model = x_exp[mask_model]
        gamma_fit_model = analysis_model["gamma"][mask_model]
        ax1.semilogy(x_fit_model * 1000, gamma_fit_model, 's', color='#C0392B', 
                    markersize=8, label='Model', alpha=0.7)
        if not np.isnan(alpha_model):
            x_fit_plot = np.linspace(x_fit_model[0], x_fit_model[-1], 100)
            A_model = np.exp(np.polyfit(x_fit_model, np.log(gamma_fit_model), 1)[1])
            gamma_fit_plot = A_model * np.exp(-alpha_model * x_fit_plot)
            ax1.semilogy(x_fit_plot * 1000, gamma_fit_plot, '--', color='#C0392B', 
                        linewidth=2, label=f'Model fit: α={alpha_model:.2f}±{alpha_model_unc:.2f} m⁻¹')
    
    ax1.set_xlabel('Position $x$ (mm)', fontsize=12)
    ax1.set_ylabel('Amplitude $\\gamma$ (°C)', fontsize=12)
    ax1.set_title('Attenuation Decay (α fit)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phase lag (Δφ) vs position with linear fit (for β)
    ax2 = plt.subplot(2, 3, 2)
    mask_exp_phi = np.isfinite(analysis_exp["dphi"])
    mask_model_phi = np.isfinite(analysis_model["dphi"])
    
    if np.any(mask_exp_phi):
        x_fit_exp_phi = x_exp[mask_exp_phi]
        dphi_fit_exp = analysis_exp["dphi"][mask_exp_phi]
        ax2.plot(x_fit_exp_phi * 1000, dphi_fit_exp, 'o', color='#2980B9', 
                markersize=8, label='Experimental', alpha=0.7)
        if not np.isnan(beta_exp):
            x_fit_plot = np.linspace(x_fit_exp_phi[0], x_fit_exp_phi[-1], 100)
            # Linear fit: Δφ = β * x + intercept
            # Recalculate intercept from the fit if not provided
            if intercept_exp_phi is None:
                p_fit = np.polyfit(x_fit_exp_phi, dphi_fit_exp, 1)
                if len(p_fit) >= 2:
                    intercept_exp_phi = p_fit[1]
                else:
                    intercept_exp_phi = 0.0
            dphi_fit_plot = beta_exp * x_fit_plot + intercept_exp_phi
            ax2.plot(x_fit_plot * 1000, dphi_fit_plot, '--', color='#2980B9', 
                    linewidth=2, label=f'Exp fit: β={beta_exp:.2f}±{beta_exp_unc:.2f} rad/m')
    
    if np.any(mask_model_phi):
        x_fit_model_phi = x_exp[mask_model_phi]
        dphi_fit_model = analysis_model["dphi"][mask_model_phi]
        ax2.plot(x_fit_model_phi * 1000, dphi_fit_model, 's', color='#C0392B', 
                markersize=8, label='Model', alpha=0.7)
        if not np.isnan(beta_model):
            x_fit_plot = np.linspace(x_fit_model_phi[0], x_fit_model_phi[-1], 100)
            # Linear fit: Δφ = β * x + intercept
            # Recalculate intercept from the fit if not provided
            if intercept_model_phi is None:
                p_fit = np.polyfit(x_fit_model_phi, dphi_fit_model, 1)
                if len(p_fit) >= 2:
                    intercept_model_phi = p_fit[1]
                else:
                    intercept_model_phi = 0.0
            dphi_fit_plot = beta_model * x_fit_plot + intercept_model_phi
            ax2.plot(x_fit_plot * 1000, dphi_fit_plot, '--', color='#C0392B', 
                    linewidth=2, label=f'Model fit: β={beta_model:.2f}±{beta_model_unc:.2f} rad/m')
    
    ax2.set_xlabel('Position $x$ (mm)', fontsize=12)
    ax2.set_ylabel('Phase Lag $\\Delta\\phi$ (rad)', fontsize=12)
    ax2.set_title('Phase Lag (β fit)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3-6: Exponential fits for τ at selected thermistor positions
    # Show fits for thermistors 0, 2, 4, 6
    selected_thermistors = [0, 2, 4, 6]
    for idx, plot_idx in enumerate([3, 4, 5, 6]):
        ax = plt.subplot(2, 3, plot_idx)
        if idx < len(selected_thermistors):
            therm_id = selected_thermistors[idx]
            if therm_id < temps_exp.shape[1]:
                T_exp = temps_exp[:, therm_id]
                T_model = temps_model[:, therm_id]
                
                # Plot data (subsample for clarity)
                step = max(1, len(timestamp) // 1000)
                ax.plot(timestamp[::step], T_exp[::step], 'o', color='#2980B9', markersize=2, 
                       alpha=0.4, label='Experimental')
                ax.plot(timestamp[::step], T_model[::step], 's', color='#C0392B', markersize=2, 
                       alpha=0.4, label='Model')
                
                # Plot exponential fits if available
                valid_mask_exp = np.isfinite(T_exp) & np.isfinite(timestamp)
                if np.sum(valid_mask_exp) >= 10:
                    t_exp_valid = timestamp[valid_mask_exp]
                    T_exp_valid = T_exp[valid_mask_exp]
                    tau_exp_val = tau_exp_dict.get(therm_id, None)
                    
                    if tau_exp_val is not None and tau_exp_val > 0:
                        # Fit exponential: T = T_inf + A * exp(-t/tau)
                        try:
                            from scipy.optimize import curve_fit
                            def exp_model(t, T_inf, A, tau):
                                return T_inf + A * np.exp(-t / tau)
                            
                            # Estimate initial guess
                            T_inf0 = np.mean(T_exp_valid[-100:]) if len(T_exp_valid) > 100 else np.mean(T_exp_valid)
                            A0 = T_exp_valid[0] - T_inf0
                            tau0 = tau_exp_val
                            
                            popt_exp, _ = curve_fit(exp_model, t_exp_valid, T_exp_valid, 
                                                   p0=[T_inf0, A0, tau0], maxfev=10000)
                            T_inf_exp, A_exp, tau_exp_fit = popt_exp
                            
                            t_fit = np.linspace(t_exp_valid[0], t_exp_valid[-1], 200)
                            T_fit_exp = exp_model(t_fit, T_inf_exp, A_exp, tau_exp_fit)
                            ax.plot(t_fit, T_fit_exp, '--', color='#2980B9', 
                                   linewidth=2, label=f'Exp: τ={tau_exp_fit:.1f} s')
                        except:
                            pass
                
                valid_mask_model = np.isfinite(T_model) & np.isfinite(timestamp)
                if np.sum(valid_mask_model) >= 10:
                    t_model_valid = timestamp[valid_mask_model]
                    T_model_valid = T_model[valid_mask_model]
                    tau_model_val = tau_model_dict.get(therm_id, None)
                    
                    if tau_model_val is not None and tau_model_val > 0:
                        try:
                            from scipy.optimize import curve_fit
                            def exp_model(t, T_inf, A, tau):
                                return T_inf + A * np.exp(-t / tau)
                            
                            T_inf0 = np.mean(T_model_valid[-100:]) if len(T_model_valid) > 100 else np.mean(T_model_valid)
                            A0 = T_model_valid[0] - T_inf0
                            tau0 = tau_model_val
                            
                            popt_model, _ = curve_fit(exp_model, t_model_valid, T_model_valid, 
                                                     p0=[T_inf0, A0, tau0], maxfev=10000)
                            T_inf_model, A_model, tau_model_fit = popt_model
                            
                            t_fit = np.linspace(t_model_valid[0], t_model_valid[-1], 200)
                            T_fit_model = exp_model(t_fit, T_inf_model, A_model, tau_model_fit)
                            ax.plot(t_fit, T_fit_model, '--', color='#C0392B', 
                                   linewidth=2, label=f'Model: τ={tau_model_fit:.1f} s')
                        except:
                            pass
                
                x_pos_mm = thermistor_positions_dict.get(therm_id, 0) * 1000
                ax.set_xlabel('Time (s)', fontsize=11)
                ax.set_ylabel('Temperature (°C)', fontsize=11)
                ax.set_title(f'Thermistor {therm_id} (x={x_pos_mm:.0f} mm)', fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = save_dir / "curve_fitting_plots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved curve fitting plots to: {save_path}")
    
    plt.close()


def plot_sine_fits(timestamp, temps_exp, temps_model, positions,
                   analysis_exp, analysis_model, period_s,
                   thermistor_positions_dict, save_dir=None):
    """
    Plot sine wave fits for attenuation and phase analysis.
    
    Parameters:
    - timestamp: Time array
    - temps_exp: Experimental temperatures (n_time, n_thermistors)
    - temps_model: Model temperatures (n_time, n_thermistors)
    - positions: Array of thermistor positions in meters
    - analysis_exp, analysis_model: Phase/amplitude analysis results
    - period_s: Period of thermal wave in seconds
    - thermistor_positions_dict: Dictionary mapping thermistor_id to position
    - save_dir: Directory to save plots
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    omega = 2 * np.pi / period_s
    
    # Create figure with subplots: 2 rows x 4 columns
    # Row 1: Experimental sine fits for selected thermistors
    # Row 2: Model sine fits for selected thermistors
    fig = plt.figure(figsize=(18, 10))
    
    # Select thermistors to plot (0, 1, 3, 5)
    selected_thermistors = [0, 1, 3, 5]
    
    # Map thermistor IDs to position indices
    therm_id_to_idx = {}
    for idx, pos in enumerate(positions):
        # Find closest thermistor ID for this position
        closest_therm_id = None
        min_dist = float('inf')
        for therm_id, therm_pos in thermistor_positions_dict.items():
            dist = abs(pos - therm_pos)
            if dist < min_dist:
                min_dist = dist
                closest_therm_id = therm_id
        if closest_therm_id is not None:
            therm_id_to_idx[closest_therm_id] = idx
    
    # Row 1: Experimental sine fits
    for idx, therm_id in enumerate(selected_thermistors):
        ax = plt.subplot(2, 4, idx + 1)
        therm_idx = therm_id_to_idx.get(therm_id, None)
        if therm_idx is not None and therm_idx < temps_exp.shape[1]:
            T_exp = temps_exp[:, therm_idx]
            x_pos_mm = positions[therm_idx] * 1000
            
            # Get sine fit parameters from analysis
            if 'amps' in analysis_exp and 'phases' in analysis_exp and 'offsets' in analysis_exp:
                if therm_idx < len(analysis_exp['amps']):
                    amp = analysis_exp['amps'][therm_idx]
                    phase = analysis_exp['phases'][therm_idx]
                    offset = analysis_exp['offsets'][therm_idx]
                    
                    # Plot data (subsample for clarity)
                    step = max(1, len(timestamp) // 2000)
                    ax.plot(timestamp[::step], T_exp[::step], 'o', color='#2980B9', 
                           markersize=1.5, alpha=0.3, label='Experimental data')
                    
                    # Plot sine fit: T(t) = offset + A·sin(ωt + φ)
                    # This matches the form used in phase_and_amplitude.py
                    t_fit = np.linspace(timestamp[0], timestamp[-1], 1000)
                    sine_fit = offset + amp * np.sin(omega * t_fit + phase)
                    ax.plot(t_fit, sine_fit, '-', color='#2980B9', 
                           linewidth=2, label=f'Sine fit: A={amp:.3f}°C, φ={phase:.3f} rad')
                    
                    ax.set_xlabel('Time (s)', fontsize=11)
                    ax.set_ylabel('Temperature (°C)', fontsize=11)
                    ax.set_title(f'Exp: Thermistor {therm_id} (x={x_pos_mm:.0f} mm)', 
                               fontsize=12, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
    
    # Row 2: Model sine fits
    for idx, therm_id in enumerate(selected_thermistors):
        ax = plt.subplot(2, 4, idx + 5)
        therm_idx = therm_id_to_idx.get(therm_id, None)
        if therm_idx is not None and therm_idx < temps_model.shape[1]:
            T_model = temps_model[:, therm_idx]
            x_pos_mm = positions[therm_idx] * 1000
            
            # Get sine fit parameters from analysis
            if 'amps' in analysis_model and 'phases' in analysis_model and 'offsets' in analysis_model:
                if therm_idx < len(analysis_model['amps']):
                    amp = analysis_model['amps'][therm_idx]
                    phase = analysis_model['phases'][therm_idx]
                    offset = analysis_model['offsets'][therm_idx]
                    
                    # Plot data (subsample for clarity)
                    step = max(1, len(timestamp) // 2000)
                    ax.plot(timestamp[::step], T_model[::step], 's', color='#C0392B', 
                           markersize=1.5, alpha=0.3, label='Model data')
                    
                    # Plot sine fit: T(t) = offset + A·sin(ωt + φ)
                    # This matches the form used in phase_and_amplitude.py
                    t_fit = np.linspace(timestamp[0], timestamp[-1], 1000)
                    sine_fit = offset + amp * np.sin(omega * t_fit + phase)
                    ax.plot(t_fit, sine_fit, '-', color='#C0392B', 
                           linewidth=2, label=f'Sine fit: A={amp:.3f}°C, φ={phase:.3f} rad')
                    
                    ax.set_xlabel('Time (s)', fontsize=11)
                    ax.set_ylabel('Temperature (°C)', fontsize=11)
                    ax.set_title(f'Model: Thermistor {therm_id} (x={x_pos_mm:.0f} mm)', 
                               fontsize=12, fontweight='bold')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir is not None:
        save_path = save_dir / "sine_fitting_plots.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved sine fitting plots to: {save_path}")
    
    plt.close()


def extract_tau_from_exponential_fit(timestamp, temperature_data, data_type='model'):
    """
    Extract tau from exponential fit using the same approach as convective_coeff_plot.py.
    Based on fit_steady_state_temperature but returns tau instead of T_inf.
    
    Parameters:
    - timestamp: Time array
    - temperature_data: Temperature array
    - data_type: 'model' or 'experimental'
    
    Returns:
    - tau: Time constant (s), or None if fitting fails
    - tau_uncertainty: Uncertainty in tau, or None if fitting fails
    """
    from scipy.optimize import curve_fit
    import warnings
    import sys
    from io import StringIO
    
    # Remove NaN values
    valid_mask = np.isfinite(temperature_data) & np.isfinite(timestamp)
    if np.sum(valid_mask) < 10:
        return None, None
    
    timestamp_valid = timestamp[valid_mask]
    temp_valid = temperature_data[valid_mask]
    
    # Check if data is already at steady-state (small variation)
    temp_std = np.std(temp_valid)
    temp_mean = np.mean(temp_valid)
    temp_min = np.min(temp_valid)
    temp_max = np.max(temp_valid)
    
    # If variation is very small (< 0.1°C), cannot extract tau
    if temp_std < 0.1:
        return None, None
    
    # Determine if data is increasing or decreasing
    temp_change = temp_valid[-1] - temp_valid[0]
    
    # Set reasonable bounds based on data type (same as convective_coeff_plot.py)
    if data_type == 'model':
        T_inf_min = max(0, temp_min - 10)
        T_inf_max = min(50, temp_max + 10)
    else:  # experimental
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
        popt, pcov = curve_fit(
            exponential_model,
            timestamp_valid,
            temp_valid,
            p0=[T_inf0, A0, tau0],
            bounds=([T_inf_min, A_min, tau_min], [T_inf_max, A_max, tau_max]),
            maxfev=20000  # Same as convective_coeff_plot.py
        )
        T_inf_fit, A_fit, tau_fit = popt
        
        # Extract uncertainty for tau (third parameter, index 2)
        tau_uncertainty = np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else None
        
        # Validate fit result
        if (T_inf_min <= T_inf_fit <= T_inf_max and 
            tau_fit > 0 and np.isfinite(T_inf_fit) and 
            np.isfinite(A_fit) and np.isfinite(tau_fit)):
            
            # Additional reasonable range check
            if data_type == 'model':
                if 0 <= T_inf_fit <= 50:
                    return tau_fit, tau_uncertainty
            else:  # experimental
                if -10 <= T_inf_fit <= 100:
                    return tau_fit, tau_uncertainty
        
        return None, None
    except Exception:
        # curve_fit failed, try the fallback function (suppress warnings and print statements)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = fit_transient_single_series(timestamp_valid, temp_valid)
        finally:
            sys.stdout = old_stdout
        
        if fit_result is not None and not np.isnan(fit_result[0]):
            T_inf_fit, _, tau_fit, tau_unc = fit_result
            
            # Validate fit result
            if tau_fit > 0 and np.isfinite(T_inf_fit):
                if data_type == 'model':
                    if 0 <= T_inf_fit <= 50:
                        return tau_fit, tau_unc
                else:  # experimental
                    if -10 <= T_inf_fit <= 100:
                        return tau_fit, tau_unc
        
        return None, None


def check_within_uncertainty(model_value, model_unc, exp_value, exp_unc):
    """
    Check if model and experimental uncertainty ranges overlap.
    
    Parameters:
    - model_value: Model coefficient value
    - model_unc: Model uncertainty
    - exp_value: Experimental coefficient value
    - exp_unc: Experimental uncertainty
    
    Returns:
    - True if the uncertainty ranges overlap, False otherwise
    """
    if np.isnan(model_value) or np.isnan(exp_value) or np.isnan(exp_unc) or np.isnan(model_unc):
        return False
    
    # Experimental range: [exp_value - exp_unc, exp_value + exp_unc]
    exp_lower = exp_value - exp_unc
    exp_upper = exp_value + exp_unc
    
    # Model range: [model_value - model_unc, model_value + model_unc]
    model_lower = model_value - model_unc
    model_upper = model_value + model_unc
    
    # Check if ranges overlap
    # Ranges overlap if: max(lower bounds) <= min(upper bounds)
    return max(exp_lower, model_lower) <= min(exp_upper, model_upper)


def generate_combined_summary_table(save_path=None):
    """
    Generate a combined summary table with:
    1. Validation metrics (RMSE, MSE, Mean Residual, T-test, Pearson r)
    2. Attenuation coefficient (α) comparison
    3. Phase shift coefficient (β) comparison
    
    For α and β, adds a "Valid" column that checks if model is within experimental uncertainty.
    """
    # ===== PART 1: Validation Metrics =====
    print("Calculating validation metrics from convective model (still air only)...")
    
    # Load h values from cooling data
    h_dict = load_h_values_from_cooling()
    if not h_dict:
        print("Error: Could not load h values. Please run cooling.py first.")
        return
    
    # Use brass_7V_10s.csv for experimental data (still air)
    script_dir = Path(__file__).resolve().parent
    still_air_csv = script_dir / "data" / "session6" / "brass_7V_10s.csv"
    
    if not still_air_csv.exists():
        print(f"Error: Still air CSV file not found at: {still_air_csv}")
        return
    
    # Only use still air data (voltage=0)
    if 0 not in h_dict:
        print("Error: No h value found for still air (voltage=0). Please check h_vs_voltage.csv")
        return
    
    h_value = h_dict[0]  # Still air (fan off)
    print(f"  Loading experimental data from brass_7V_10s.csv...")
    print(f"  Calculating validation metrics for still air (h={h_value:.2f} W/(m²·K))...")
    
    # Load experimental data using load_dataset (same as convection.py)
    timestamp, voltage, current, thermistor_temperatures = load_dataset(still_air_csv)
    
    # Trim to middle 90% of dataset (drop first 5% and last 5%)
    n_total = len(timestamp)
    n_drop_start = int(n_total * 0.05)
    n_drop_end = int(n_total * 0.05)
    start_idx = n_drop_start
    end_idx = n_total - n_drop_end
    
    timestamp = timestamp[start_idx:end_idx]
    voltage = voltage[start_idx:end_idx]
    current = current[start_idx:end_idx]
    thermistor_temperatures = thermistor_temperatures[start_idx:end_idx, :]
    
    print(f"  Trimmed dataset: using middle 90% ({len(timestamp)}/{n_total} points)")
    
    # Extract initial temperatures from first data point of trimmed dataset
    T_initial_thermistors = thermistor_temperatures[0, :] + 273.15  # Convert from Celsius to Kelvin
    
    # Get thermistor positions
    thermistor_positions_dict = get_thermistor_positions()
    
    # Extract positions and temperatures for interpolation
    thermistor_x_positions = []
    thermistor_T_values = []
    for therm_id in range(thermistor_temperatures.shape[1]):
        if therm_id in thermistor_positions_dict:
            thermistor_x_positions.append(thermistor_positions_dict[therm_id])
            thermistor_T_values.append(T_initial_thermistors[therm_id])
    
    thermistor_x_positions = np.array(thermistor_x_positions)
    thermistor_T_values = np.array(thermistor_T_values)
    
    # Load optimized parameters
    import json
    params_file = script_dir / 'data' / 'netflux' / 'optimized_parameters.json'
    
    if not params_file.exists():
        print(f"Error: Optimized parameters file not found at: {params_file}")
        return
    
    with open(params_file, 'r') as f:
        opt_params = json.load(f)
    
    alpha = opt_params['alpha']
    K_therm = opt_params['K_therm']
    R_elec = opt_params['R_elec']
    
    # Set up parameters for numerical integration (same as convection.py)
    T_inf = 25.0 + 273.15  # K (ambient temperature)
    C_cold_plate = 50.0  # J/K
    C_hot_plate = 300.0  # J/K
    rho_brass = 8520.0  # kg/m³
    c_brass = 380.0  # J/(kg·K)
    k_brass = 109.0  # W/(m·K)
    radius_brass = 0.015  # m
    L_brass = 0.041  # m
    thickness_grease = 0.0001  # m
    k_grease = 1.0  # W/(m·K)
    radius_plate = 0.015  # m
    A_contact = np.pi * radius_plate**2
    N_nodes = 50
    x_grid = np.linspace(0, L_brass, N_nodes)
    dx = L_brass / (N_nodes - 1)
    
    # Interpolate thermistor temperatures to spatial grid for initial condition
    from scipy.interpolate import interp1d
    if len(thermistor_x_positions) > 1:
        T_brass_interp_func = interp1d(thermistor_x_positions, thermistor_T_values, 
                                       kind='linear', fill_value='extrapolate', bounds_error=False)
        T_brass_array = T_brass_interp_func(x_grid)
    else:
        T_brass_array = np.full(N_nodes, T_initial_thermistors[0])
    
    T_cold_initial = T_initial_thermistors[0]
    T_hot_initial = T_initial_thermistors[0]
    T_initial = np.concatenate([[T_cold_initial, T_hot_initial], T_brass_array])
    
    h_hot = 200  # W/(m²·K)
    heat_sink_length = 0.10
    heat_sink_width = 0.14
    heat_sink_height = 0.01
    n_fins = 18
    fin_length = 0.025
    fin_width = 0.14
    fin_thickness = 0.001
    A_hot = heat_sink_length * heat_sink_width + n_fins * 2 * fin_length * fin_width
    
    params = {
        'alpha': alpha,
        'K_therm': K_therm,
        'R_elec': R_elec,
        'T_inf': T_inf,
        'C_cold_plate': C_cold_plate,
        'C_hot_plate': C_hot_plate,
        'rho_brass': rho_brass,
        'c_brass': c_brass,
        'k_brass': k_brass,
        'radius_brass': radius_brass,
        'L_brass': L_brass,
        'thickness_grease': thickness_grease,
        'k_grease': k_grease,
        'A_contact': A_contact,
        'N_nodes': N_nodes,
        'x_grid': x_grid,
        'dx': dx,
        'h': h_value,  # Use still air h value
        'h_hot': h_hot,
        'A_hot': A_hot,
    }
    
    # Create interpolation function for voltage
    voltage_interp = interp1d(timestamp, voltage, kind='linear',
                                fill_value=(voltage[0], voltage[-1]), bounds_error=False)
    
    t_span = (timestamp[0], timestamp[-1])
    rtol = 1e-6
    atol = 1e-8
    
    # Perform numerical integration to generate model temperatures
    print("  Performing numerical integration to generate model temperatures...")
    sol = solve_coupled_heat_pump(t_span, T_initial, voltage_interp, params, rtol, atol, 
                                    t_eval=timestamp, method='Radau')
    
    # Extract model temperatures at thermistor positions
    print("  Extracting model temperatures at thermistor positions...")
    temps_model = np.zeros((len(timestamp), len(thermistor_positions_dict)))
    for therm_id, x_pos in thermistor_positions_dict.items():
        T_brass_at_t = sol.y[2:, :]
        idx_closest = np.argmin(np.abs(x_grid - x_pos))
        temps_model[:, therm_id] = T_brass_at_t[idx_closest, :] - 273.15  # Convert to Celsius
    
    # Experimental temperatures (already in Celsius)
    temps_exp = thermistor_temperatures
    
    # Collect all residuals and model/experimental data
    all_residuals = []
    all_T_model = []
    all_T_exp = []
    
    for therm_id in range(thermistor_temperatures.shape[1]):
        T_model = temps_model[:, therm_id]
        T_exp = temps_exp[:, therm_id]
        
        # Calculate residuals
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp) & np.isfinite(timestamp)
        if np.sum(valid_mask) > 0:
            T_model_valid = T_model[valid_mask]
            T_exp_valid = T_exp[valid_mask]
            residuals = T_model_valid - T_exp_valid
            # Take absolute value of all residuals (same as convective_validity.py)
            residuals = np.abs(residuals)
            
            all_residuals.extend(residuals.tolist())
            all_T_model.extend(T_model_valid.tolist())
            all_T_exp.extend(T_exp_valid.tolist())
    
    if len(all_residuals) == 0:
        print("Error: No valid data found for validation metrics!")
        return
    
    all_residuals = np.array(all_residuals)
    all_T_model = np.array(all_T_model)
    all_T_exp = np.array(all_T_exp)
    n_points = len(all_residuals)
    
    # Calculate validation metrics (same as convective_validity.py)
    u_measurement = 0.1  # Measurement uncertainty in °C
    alpha = 0.05  # Significance level
    
    # RMSE
    rmse = np.sqrt(np.mean(all_residuals**2))
    rmse_uncertainty = np.std(all_residuals**2) / (2 * rmse * np.sqrt(n_points)) if rmse > 0 else 0.0
    
    # MSE
    mse = np.mean(all_residuals**2)
    mse_uncertainty = np.std(all_residuals**2) / np.sqrt(n_points)
    
    # Mean Residual (already absolute values)
    mean_residual = np.mean(all_residuals)
    std_residual = np.std(all_residuals, ddof=1)
    mean_residual_uncertainty = std_residual / np.sqrt(n_points)
    
    # Pearson correlation
    from scipy.stats import pearsonr
    if len(all_T_model) > 1 and np.std(all_T_model) > 0 and np.std(all_T_exp) > 0:
        pearson_r, _ = pearsonr(all_T_model, all_T_exp)
        pearson_r_uncertainty = (1 - pearson_r**2) / np.sqrt(n_points - 2) if n_points > 2 else 0.0
    else:
        pearson_r = np.nan
        pearson_r_uncertainty = np.nan
    
    # T-test: Test if mean residual is significantly different from zero
    if n_points >= 3 and std_residual > 0:
        _, t_pvalue = ttest_1samp(all_residuals, 0.0)
    else:
        t_pvalue = np.nan
    
    # Thresholds
    rmse_threshold = 2.0 * u_measurement
    mse_threshold = (2.0 * u_measurement)**2
    mean_residual_threshold = 1.5 * u_measurement
    pearson_r_threshold = 0.900
    
    # Validity checks
    rmse_valid = np.round(rmse, 3) <= np.round(rmse_threshold, 3)
    mse_valid = np.round(mse, 3) <= np.round(mse_threshold, 3)
    mean_residual_valid = np.round(mean_residual, 3) <= np.round(mean_residual_threshold, 3)
    t_test_valid = t_pvalue >= 0.05 if not np.isnan(t_pvalue) else False
    pearson_r_valid = pearson_r >= pearson_r_threshold if not np.isnan(pearson_r) else False
    
    # ===== PART 2: Phase/Amplitude Coefficients and Tau from Convective Model =====
    print("Calculating α, β, τ from convective model data...")
    
    script_dir = Path(__file__).resolve().parent
    
    # Collect tau values from all positions and h values
    tau_exp_list = []
    tau_exp_unc_list = []
    tau_model_list = []
    tau_model_unc_list = []
    
    # Collect alpha and beta values from phase/amplitude analysis for each h value
    alpha_exp_list = []
    alpha_exp_unc_list = []
    alpha_model_list = []
    alpha_model_unc_list = []
    beta_exp_list = []
    beta_exp_unc_list = []
    beta_model_list = []
    beta_model_unc_list = []
    
    # Known period for thermal wave analysis (same as phase_and_amplitude.py)
    PERIOD_S = 10.0
    MIN_GAMMA_FOR_FIT = 0.05
    
    # Use brass_7V_10s.csv for still air data (voltage=0, fan off)
    print("  Loading still air data from brass_7V_10s.csv...")
    
    script_dir = Path(__file__).resolve().parent
    still_air_csv = script_dir / "data" / "session6" / "brass_7V_10s.csv"
    
    if not still_air_csv.exists():
        print(f"Error: Still air CSV file not found at: {still_air_csv}")
        alpha_exp = np.nan
        alpha_exp_unc = np.nan
        alpha_model = np.nan
        alpha_model_unc = np.nan
        beta_exp = np.nan
        beta_exp_unc = np.nan
        beta_model = np.nan
        beta_model_unc = np.nan
        tau_exp = np.nan
        tau_exp_unc = np.nan
        tau_model = np.nan
        tau_model_unc = np.nan
    else:
        # Load data using load_dataset (same as convection.py)
        timestamp, voltage, current, thermistor_temperatures = load_dataset(still_air_csv)
        
        # Trim to middle 90% of dataset (drop first 5% and last 5%)
        n_total = len(timestamp)
        n_drop_start = int(n_total * 0.05)
        n_drop_end = int(n_total * 0.05)
        start_idx = n_drop_start
        end_idx = n_total - n_drop_end
        
        timestamp = timestamp[start_idx:end_idx]
        voltage = voltage[start_idx:end_idx]
        current = current[start_idx:end_idx]
        thermistor_temperatures = thermistor_temperatures[start_idx:end_idx, :]
        
        print(f"  Trimmed dataset: using middle 90% ({len(timestamp)}/{n_total} points)")
        
        # Get h value for still air (voltage=0)
        if 0 not in h_dict:
            print("Error: No h value found for still air (voltage=0). Please check h_vs_voltage.csv")
            alpha_exp = np.nan
            alpha_exp_unc = np.nan
            alpha_model = np.nan
            alpha_model_unc = np.nan
            beta_exp = np.nan
            beta_exp_unc = np.nan
            beta_model = np.nan
            beta_model_unc = np.nan
            tau_exp = np.nan
            tau_exp_unc = np.nan
            tau_model = np.nan
            tau_model_unc = np.nan
        else:
            h_value = h_dict[0]  # Still air (fan off)
            print(f"  Using h={h_value:.2f} W/(m²·K) for still air")
            
            # Extract initial temperatures from first data point of trimmed dataset
            T_initial_thermistors = thermistor_temperatures[0, :] + 273.15  # Convert from Celsius to Kelvin
            
            # Get thermistor positions
            thermistor_positions_dict = get_thermistor_positions()
            
            # Extract positions and temperatures for interpolation
            thermistor_x_positions = []
            thermistor_T_values = []
            for therm_id in range(thermistor_temperatures.shape[1]):
                if therm_id in thermistor_positions_dict:
                    thermistor_x_positions.append(thermistor_positions_dict[therm_id])
                    thermistor_T_values.append(T_initial_thermistors[therm_id])
            
            thermistor_x_positions = np.array(thermistor_x_positions)
            thermistor_T_values = np.array(thermistor_T_values)
            positions = thermistor_x_positions.copy()  # For later use
            
            # Load optimized parameters
            import json
            params_file = script_dir / 'data' / 'netflux' / 'optimized_parameters.json'
            
            if not params_file.exists():
                print(f"Error: Optimized parameters file not found at: {params_file}")
                alpha_exp = np.nan
                alpha_exp_unc = np.nan
                alpha_model = np.nan
                alpha_model_unc = np.nan
                beta_exp = np.nan
                beta_exp_unc = np.nan
                beta_model = np.nan
                beta_model_unc = np.nan
                tau_exp = np.nan
                tau_exp_unc = np.nan
                tau_model = np.nan
                tau_model_unc = np.nan
            else:
                with open(params_file, 'r') as f:
                    opt_params = json.load(f)
                
                alpha = opt_params['alpha']
                K_therm = opt_params['K_therm']
                R_elec = opt_params['R_elec']
                
                # Set up parameters for numerical integration (same as convection.py)
                T_inf = 25.0 + 273.15  # K (ambient temperature)
                C_cold_plate = 50.0  # J/K
                C_hot_plate = 300.0  # J/K
                rho_brass = 8520.0  # kg/m³
                c_brass = 380.0  # J/(kg·K)
                k_brass = 109.0  # W/(m·K)
                radius_brass = 0.015  # m
                L_brass = 0.041  # m
                thickness_grease = 0.0001  # m
                k_grease = 1.0  # W/(m·K)
                radius_plate = 0.015  # m
                A_contact = np.pi * radius_plate**2
                N_nodes = 50
                x_grid = np.linspace(0, L_brass, N_nodes)
                dx = L_brass / (N_nodes - 1)
                
                # Interpolate thermistor temperatures to spatial grid for initial condition
                from scipy.interpolate import interp1d
                if len(thermistor_x_positions) > 1:
                    T_brass_interp_func = interp1d(thermistor_x_positions, thermistor_T_values, 
                                                   kind='linear', fill_value='extrapolate', bounds_error=False)
                    T_brass_array = T_brass_interp_func(x_grid)
                else:
                    T_brass_array = np.full(N_nodes, T_initial_thermistors[0])
                
                T_cold_initial = T_initial_thermistors[0]
                T_hot_initial = T_initial_thermistors[0]
                T_initial = np.concatenate([[T_cold_initial, T_hot_initial], T_brass_array])
                
                h_hot = 200  # W/(m²·K)
                heat_sink_length = 0.10
                heat_sink_width = 0.14
                heat_sink_height = 0.01
                n_fins = 18
                fin_length = 0.025
                fin_width = 0.14
                base_area = heat_sink_length * heat_sink_width
                fin_area_per_fin = 2 * fin_length * fin_width
                total_fin_area = n_fins * fin_area_per_fin
                side_area = 2 * (heat_sink_length * heat_sink_height) + 2 * (heat_sink_width * heat_sink_height)
                A_hot = base_area + total_fin_area + base_area + side_area
                
                rtol = 1e-6
                atol = 1e-8
                
                # Create interpolation function for voltage
                voltage_interp = interp1d(timestamp, voltage, kind='linear',
                                          fill_value=(voltage[0], voltage[-1]), bounds_error=False)
                
                t_span = (timestamp[0], timestamp[-1])
                
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
                    'h': h_value,
                    'N_nodes': N_nodes
                }
                
                # Perform numerical integration
                print("  Performing numerical integration...")
                sol = solve_coupled_heat_pump(t_span, T_initial, voltage_interp, params, rtol, atol, 
                                               t_eval=timestamp, method='Radau')
                
                # Extract model temperatures at thermistor positions
                print("  Extracting model temperatures at thermistor positions...")
                temps_model = np.zeros((len(timestamp), len(thermistor_positions_dict)))
                for therm_id, x_pos in thermistor_positions_dict.items():
                    # Interpolate T_brass from solution to thermistor position
                    T_brass_at_t = sol.y[2:, :]  # Shape: (N_nodes, n_time)
                    # Find closest grid point to thermistor position
                    idx_closest = np.argmin(np.abs(x_grid - x_pos))
                    temps_model[:, therm_id] = T_brass_at_t[idx_closest, :] - 273.15  # Convert to Celsius
                
                # Experimental temperatures (already in Celsius)
                temps_exp = thermistor_temperatures
                
                # Trim to last 20% of the trimmed dataset for sine fitting
                n_trimmed = len(timestamp)
                n_last_20 = int(n_trimmed * 0.2)
                start_idx_fit = n_trimmed - n_last_20
                
                timestamp_fit = timestamp[start_idx_fit:]
                temps_exp_fit = temps_exp[start_idx_fit:, :]
                temps_model_fit = temps_model[start_idx_fit:, :]
                
                print(f"  Using last 20% of trimmed data for sine fitting ({len(timestamp_fit)}/{n_trimmed} points)")
                
                # Perform phase/amplitude analysis for still air (same as first table)
                # Use only the last 20% for sine fitting
                analysis_exp = analyse_temperature_data(timestamp_fit, temps_exp_fit, PERIOD_S, label="Experimental")
                analysis_model = analyse_temperature_data(timestamp_fit, temps_model_fit, PERIOD_S, label="Model")
                
                if analysis_exp is None or analysis_model is None:
                    print(f"  Warning: Could not complete phase/amplitude analysis for still air")
                    alpha_exp = np.nan
                    alpha_exp_unc = np.nan
                    alpha_model = np.nan
                    alpha_model_unc = np.nan
                    beta_exp = np.nan
                    beta_exp_unc = np.nan
                    beta_model = np.nan
                    beta_model_unc = np.nan
                    tau_exp = np.nan
                    tau_exp_unc = np.nan
                    tau_model = np.nan
                    tau_model_unc = np.nan
                else:
                    # Update positions in analysis results
                    analysis_exp["x"] = positions
                    analysis_model["x"] = positions
                    
                    # Fit alpha (attenuation decay factor) for experimental
                    x_exp = positions
                    mask_exp = (analysis_exp["gamma"] > MIN_GAMMA_FOR_FIT) & np.isfinite(analysis_exp["gamma"])
                    mask_exp[0] = False  # Skip thermistor 0
                    if np.any(mask_exp) and len(x_exp[mask_exp]) >= 2:
                        x_fit_exp = x_exp[mask_exp]
                        gamma_fit_exp = analysis_exp["gamma"][mask_exp]
                        ln_gamma_exp = np.log(gamma_fit_exp)
                        p_exp, cov_exp = np.polyfit(x_fit_exp, ln_gamma_exp, 1, cov=True)
                        slope_exp, _ = p_exp
                        alpha_exp = -slope_exp
                        alpha_exp_unc = np.sqrt(cov_exp[0, 0])
                    else:
                        alpha_exp = np.nan
                        alpha_exp_unc = np.nan
                    
                    # Fit alpha for model
                    x_model = positions
                    mask_model = (analysis_model["gamma"] > MIN_GAMMA_FOR_FIT) & np.isfinite(analysis_model["gamma"])
                    mask_model[0] = False  # Skip thermistor 0
                    if np.any(mask_model) and len(x_model[mask_model]) >= 2:
                        x_fit_model = x_model[mask_model]
                        gamma_fit_model = analysis_model["gamma"][mask_model]
                        ln_gamma_model = np.log(gamma_fit_model)
                        p_model, cov_model = np.polyfit(x_fit_model, ln_gamma_model, 1, cov=True)
                        slope_model, _ = p_model
                        alpha_model = -slope_model
                        alpha_model_unc = np.sqrt(cov_model[0, 0])
                    else:
                        alpha_model = np.nan
                        alpha_model_unc = np.nan
                    
                    # Fit beta (phase lag coefficient) for experimental: Δφ(x) = β * x (linear fit)
                    mask_exp_phi = np.isfinite(analysis_exp["dphi"])
                    if np.any(mask_exp_phi) and len(x_exp[mask_exp_phi]) >= 2:
                        x_fit_exp_phi = x_exp[mask_exp_phi]
                        dphi_fit_exp = analysis_exp["dphi"][mask_exp_phi]
                        # Linear fit: Δφ = β * x + intercept
                        p_exp_phi, cov_exp_phi = np.polyfit(x_fit_exp_phi, dphi_fit_exp, 1, cov=True)
                        beta_exp, intercept_exp_phi = p_exp_phi  # slope is beta, intercept should be ~0
                        beta_exp_unc = np.sqrt(cov_exp_phi[0, 0])  # uncertainty in slope
                    else:
                        beta_exp = np.nan
                        beta_exp_unc = np.nan
                    
                    # Fit beta for model: Δφ(x) = β * x (linear fit)
                    mask_model_phi = np.isfinite(analysis_model["dphi"])
                    if np.any(mask_model_phi) and len(x_model[mask_model_phi]) >= 2:
                        x_fit_model_phi = x_model[mask_model_phi]
                        dphi_fit_model = analysis_model["dphi"][mask_model_phi]
                        # Linear fit: Δφ = β * x + intercept
                        p_model_phi, cov_model_phi = np.polyfit(x_fit_model_phi, dphi_fit_model, 1, cov=True)
                        beta_model, intercept_model_phi = p_model_phi  # slope is beta, intercept should be ~0
                        beta_model_unc = np.sqrt(cov_model_phi[0, 0])  # uncertainty in slope
                    else:
                        beta_model = np.nan
                        beta_model_unc = np.nan
                    
                    # Extract tau from exponential fitting for each thermistor position
                    # Store tau values indexed by thermistor ID for plotting
                    tau_exp_dict = {}  # Dictionary mapping thermistor_id to tau_exp
                    tau_exp_unc_dict = {}
                    tau_model_dict = {}  # Dictionary mapping thermistor_id to tau_model
                    tau_model_unc_dict = {}
                    tau_exp_list = []  # List for averaging
                    tau_exp_unc_list = []
                    tau_model_list = []  # List for averaging
                    tau_model_unc_list = []
                    
                    # Map positions to thermistor IDs
                    position_to_therm_id = {}
                    for therm_id, x_pos in thermistor_positions_dict.items():
                        # Find closest position
                        closest_idx = np.argmin(np.abs(positions - x_pos))
                        position_to_therm_id[closest_idx] = therm_id
                    
                    for i in range(len(positions)):
                        T_exp = temps_exp[:, i]
                        T_model = temps_model[:, i]
                        therm_id = position_to_therm_id.get(i, i)  # Default to i if not found
                        
                        # Extract tau from exponential fitting for experimental data
                        valid_mask_exp = np.isfinite(T_exp) & np.isfinite(timestamp)
                        if np.sum(valid_mask_exp) >= 10:
                            t_exp_valid = timestamp[valid_mask_exp]
                            T_exp_valid = T_exp[valid_mask_exp]
                            
                            # Use improved exponential fit from convective_coeff_plot.py
                            tau_exp_val, tau_exp_unc_val = extract_tau_from_exponential_fit(
                                t_exp_valid, T_exp_valid, data_type='experimental'
                            )
                            if tau_exp_val is not None and tau_exp_val > 0 and np.isfinite(tau_exp_val):
                                tau_exp_list.append(tau_exp_val)
                                tau_exp_dict[therm_id] = tau_exp_val
                                if tau_exp_unc_val is not None and not np.isnan(tau_exp_unc_val):
                                    tau_exp_unc_list.append(tau_exp_unc_val**2)  # Store squared for later averaging
                                    tau_exp_unc_dict[therm_id] = tau_exp_unc_val
                        
                        # Extract tau from exponential fitting for model data
                        valid_mask_model = np.isfinite(T_model) & np.isfinite(timestamp)
                        if np.sum(valid_mask_model) >= 10:
                            t_model_valid = timestamp[valid_mask_model]
                            T_model_valid = T_model[valid_mask_model]
                            
                            # Use improved exponential fit from convective_coeff_plot.py
                            tau_model_val, tau_model_unc_val = extract_tau_from_exponential_fit(
                                t_model_valid, T_model_valid, data_type='model'
                            )
                            if tau_model_val is not None and tau_model_val > 0 and np.isfinite(tau_model_val):
                                tau_model_list.append(tau_model_val)
                                tau_model_dict[therm_id] = tau_model_val
                                if tau_model_unc_val is not None and not np.isnan(tau_model_unc_val):
                                    tau_model_unc_list.append(tau_model_unc_val**2)  # Store squared for later averaging
                                    tau_model_unc_dict[therm_id] = tau_model_unc_val
                    
                    # Calculate tau averages across positions for still air
                    if len(tau_exp_list) > 0:
                        tau_exp = np.nanmean(tau_exp_list)
                        tau_exp_unc = np.sqrt(np.nanmean(tau_exp_unc_list)) if len(tau_exp_unc_list) > 0 else np.nan
                    else:
                        tau_exp = np.nan
                        tau_exp_unc = np.nan
                    
                    if len(tau_model_list) > 0:
                        tau_model = np.nanmean(tau_model_list)
                        tau_model_unc = np.sqrt(np.nanmean(tau_model_unc_list)) if len(tau_model_unc_list) > 0 else np.nan
                    else:
                        tau_model = np.nan
                        tau_model_unc = np.nan
                    
                    print(f"  Processed still air data (h={h_value:.2f} W/(m²·K))")
                    print(f"  Alpha: exp={alpha_exp:.3f} ± {alpha_exp_unc:.3f}, model={alpha_model:.3f} ± {alpha_model_unc:.3f}")
                    print(f"  Beta: exp={beta_exp:.3f} ± {beta_exp_unc:.3f}, model={beta_model:.3f} ± {beta_model_unc:.3f}")
                    print(f"  Tau: exp={len(tau_exp_list)} fits (avg={tau_exp:.3f} ± {tau_exp_unc:.3f} s), model={len(tau_model_list)} fits (avg={tau_model:.3f} ± {tau_model_unc:.3f} s)")
                    
                    # Generate plots showing the curve fitting
                    print("  Generating curve fitting plots...")
                    # Get intercepts from phase lag fits for plotting
                    intercept_exp_phi_val = None
                    intercept_model_phi_val = None
                    mask_exp_phi = np.isfinite(analysis_exp["dphi"])
                    mask_model_phi = np.isfinite(analysis_model["dphi"])
                    if np.any(mask_exp_phi) and len(positions[mask_exp_phi]) >= 2:
                        x_fit_exp_phi = positions[mask_exp_phi]
                        dphi_fit_exp = analysis_exp["dphi"][mask_exp_phi]
                        p_exp_phi = np.polyfit(x_fit_exp_phi, dphi_fit_exp, 1)
                        if len(p_exp_phi) >= 2:
                            intercept_exp_phi_val = p_exp_phi[1]
                    if np.any(mask_model_phi) and len(positions[mask_model_phi]) >= 2:
                        x_fit_model_phi = positions[mask_model_phi]
                        dphi_fit_model = analysis_model["dphi"][mask_model_phi]
                        p_model_phi = np.polyfit(x_fit_model_phi, dphi_fit_model, 1)
                        if len(p_model_phi) >= 2:
                            intercept_model_phi_val = p_model_phi[1]
                    
                    plot_curve_fits(timestamp, temps_exp, temps_model, positions, 
                                   analysis_exp, analysis_model,
                                   alpha_exp, alpha_exp_unc, alpha_model, alpha_model_unc,
                                   beta_exp, beta_exp_unc, beta_model, beta_model_unc,
                                   tau_exp_dict, tau_model_dict, thermistor_positions_dict,
                                   intercept_exp_phi=intercept_exp_phi_val,
                                   intercept_model_phi=intercept_model_phi_val,
                                   save_dir=script_dir / "plots" / "summary")
                    
                    # Generate sine fitting plots (use last 20% data that was used for fitting)
                    print("  Generating sine fitting plots...")
                    plot_sine_fits(timestamp_fit, temps_exp_fit, temps_model_fit, positions,
                                  analysis_exp, analysis_model, PERIOD_S,
                                  thermistor_positions_dict,
                                  save_dir=script_dir / "plots" / "summary")
    
    # Check validity for α and β
    alpha_valid = check_within_uncertainty(alpha_model, alpha_model_unc, alpha_exp, alpha_exp_unc)
    beta_valid = check_within_uncertainty(beta_model, beta_model_unc, beta_exp, beta_exp_unc)
    tau_valid = check_within_uncertainty(tau_model, tau_model_unc, tau_exp, tau_exp_unc)
    
    # ===== PART 3: Create Combined Table =====
    print("Generating combined summary table...")
    
    def format_with_uncertainty(value, uncertainty, decimals=3):
        """Format value ± uncertainty."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.{decimals}f}'
        return f'{value:.{decimals}f} ± {uncertainty:.{decimals}f}'
    
    def format_with_uncertainty_1dec(value, uncertainty, decimals=1):
        """Format value ± uncertainty with 1 decimal place."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.{decimals}f}'
        return f'{value:.{decimals}f} ± {uncertainty:.{decimals}f}'
    
    # Prepare table data
    data = [
        ['Parameter', 'Value', 'Constraint', 'Valid'],
        # Validation metrics
        ['RMSE (°C)', format_with_uncertainty(rmse, rmse_uncertainty), f'≤ {rmse_threshold:.3f}', '✓' if rmse_valid else '✗'],
        ['MSE (°C²)', format_with_uncertainty(mse, mse_uncertainty), f'≤ {mse_threshold:.3f}', '✓' if mse_valid else '✗'],
        ['Mean Residual (°C)', format_with_uncertainty(mean_residual, mean_residual_uncertainty), f'≤ {mean_residual_threshold:.3f}', '✓' if mean_residual_valid else '✗'],
        ['T-test of Mean Residual (p-value)', f'{t_pvalue:.3f}', f'≥ 0.050', '✓' if t_test_valid else '✗'],
        ['Pearson r', format_with_uncertainty(pearson_r, pearson_r_uncertainty), f'≥ {pearson_r_threshold:.3f}', '✓' if pearson_r_valid else '✗'],
        # Attenuation coefficient - Modeling value in Value column, Experimental in Constraint column
        ['α (attenuation decay coefficient)', format_with_uncertainty(alpha_model, alpha_model_unc), format_with_uncertainty(alpha_exp, alpha_exp_unc), '✓' if alpha_valid else '✗'],
        # Phase shift coefficient - Modeling value in Value column, Experimental in Constraint column
        ['β (phaseshift scaling constant)', format_with_uncertainty(beta_model, beta_model_unc), format_with_uncertainty(beta_exp, beta_exp_unc), '✓' if beta_valid else '✗'],
        # Time constant - Modeling value in Value column, Experimental in Constraint column
        ['τ (transient time constant)', format_with_uncertainty(tau_model, tau_model_unc), format_with_uncertainty(tau_exp, tau_exp_unc), '✓' if tau_valid else '✗'],
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.35, 0.25, 0.25, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # Color code validity cells
    for i in range(1, len(data)):
        # Skip empty separator rows
        if data[i][0] == '':
            continue
        
        # Validity column (last column)
        if data[i][3] == '✓':
            table[(i, 3)].set_facecolor('#90EE90')  # Light green
            table[(i, 3)].set_text_props(weight='bold')
        elif data[i][3] == '✗':
            table[(i, 3)].set_facecolor('#FFB6C1')  # Light pink
            table[(i, 3)].set_text_props(weight='bold')
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white', size=12)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCombined summary table saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_threshold_testing_table(save_path=None):
    """
    Generate threshold testing table from convective model CSV files.
    Same logic as convective_validity.py plot_threshold_testing_table function.
    """
    print("Generating threshold testing table from convective model...")
    
    # Load h values from cooling data
    h_dict = load_h_values_from_cooling()
    if not h_dict:
        print("Error: Could not load h values. Please run cooling.py first.")
        return
    
    # Collect residuals and model/experimental data ONLY from still air (voltage=0, h=11.20)
    all_residuals = []
    all_T_model = []
    all_T_exp = []
    
    # Only use still air data (voltage=0)
    if 0 not in h_dict:
        print("Error: No h value found for still air (voltage=0). Please check h_vs_voltage.csv")
        return
    
    h_value = h_dict[0]  # Still air (fan off)
    print(f"  Calculating validation metrics for still air (h={h_value:.2f} W/(m²·K))...")
    
    csv_path = find_csv_for_h_value(h_value)
    if csv_path is None or not csv_path.exists():
        print(f"  Warning: CSV not found for h={h_value:.2f} (voltage=0V)")
        print("  Please run convective_plot.py first to generate comparison CSV files.")
        return
    
    df = pd.read_csv(csv_path)
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    
    for model_col_name in sorted(model_cols):
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate residuals
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
        if np.sum(valid_mask) > 0:
            T_model_valid = T_model[valid_mask]
            T_exp_valid = T_exp[valid_mask]
            residuals = T_model_valid - T_exp_valid
            # Take absolute value of all residuals (same as convective_validity.py)
            residuals = np.abs(residuals)
            
            all_residuals.extend(residuals.tolist())
            all_T_model.extend(T_model_valid.tolist())
            all_T_exp.extend(T_exp_valid.tolist())
    
    if len(all_residuals) == 0:
        print("Error: No valid data found for threshold testing!")
        print("  Please run convective_plot.py first to generate comparison CSV files.")
        return
    
    all_residuals = np.array(all_residuals)
    all_T_model = np.array(all_T_model)
    all_T_exp = np.array(all_T_exp)
    n_points = len(all_residuals)
    
    # Calculate validation metrics (same as convective_validity.py)
    u_measurement = 0.1  # Measurement uncertainty in °C
    alpha = 0.05  # Significance level
    
    # RMSE
    rmse = np.sqrt(np.mean(all_residuals**2))
    rmse_uncertainty = np.std(all_residuals**2) / (2 * rmse * np.sqrt(n_points)) if rmse > 0 else 0.0
    
    # MSE
    mse = np.mean(all_residuals**2)
    mse_uncertainty = np.std(all_residuals**2) / np.sqrt(n_points)
    
    # Mean Residual (already absolute values)
    mean_residual = np.mean(all_residuals)
    std_residual = np.std(all_residuals, ddof=1)
    mean_residual_uncertainty = std_residual / np.sqrt(n_points)
    
    # Pearson correlation
    from scipy.stats import pearsonr
    if len(all_T_model) > 1 and np.std(all_T_model) > 0 and np.std(all_T_exp) > 0:
        pearson_r, _ = pearsonr(all_T_model, all_T_exp)
        pearson_r_uncertainty = (1 - pearson_r**2) / np.sqrt(n_points - 2) if n_points > 2 else 0.0
    else:
        pearson_r = np.nan
        pearson_r_uncertainty = np.nan
    
    # T-test: Test if mean residual is significantly different from zero
    if n_points >= 3 and std_residual > 0:
        _, t_pvalue = ttest_1samp(all_residuals, 0.0)
    else:
        t_pvalue = np.nan
    
    # Thresholds
    rmse_threshold = 2.0 * u_measurement
    mse_threshold = (2.0 * u_measurement)**2
    mean_residual_threshold = 1.5 * u_measurement
    pearson_r_threshold = 0.900
    
    # Validity checks
    rmse_valid = np.round(rmse, 3) <= np.round(rmse_threshold, 3)
    mse_valid = np.round(mse, 3) <= np.round(mse_threshold, 3)
    mean_residual_valid = np.round(mean_residual, 3) <= np.round(mean_residual_threshold, 3)
    t_test_valid = t_pvalue >= 0.05 if not np.isnan(t_pvalue) else False
    pearson_r_valid = pearson_r >= pearson_r_threshold if not np.isnan(pearson_r) else False
    
    # Format values with uncertainties (3 decimal places)
    def format_with_uncertainty(value, uncertainty, decimals=3):
        """Format value ± uncertainty."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.{decimals}f}'
        return f'{value:.{decimals}f} ± {uncertainty:.{decimals}f}'
    
    # Prepare table data
    data = [
        ['Parameter', 'Value', 'Constraint', 'Valid'],
        ['RMSE (°C)', format_with_uncertainty(rmse, rmse_uncertainty), f'≤ {rmse_threshold:.3f}', '✓' if rmse_valid else '✗'],
        ['MSE (°C²)', format_with_uncertainty(mse, mse_uncertainty), f'≤ {mse_threshold:.3f}', '✓' if mse_valid else '✗'],
        ['Mean Residual (°C)', format_with_uncertainty(mean_residual, mean_residual_uncertainty), f'≤ {mean_residual_threshold:.3f}', '✓' if mean_residual_valid else '✗'],
        ['T-test of Mean Residual (p-value)', f'{t_pvalue:.3f}', f'≥ 0.050', '✓' if t_test_valid else '✗'],
        ['Pearson r', format_with_uncertainty(pearson_r, pearson_r_uncertainty), f'≥ {pearson_r_threshold:.3f}', '✓' if pearson_r_valid else '✗'],
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.35, 0.25, 0.25, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Color code validity cells
    for i in range(1, len(data)):
        # Validity column (last column)
        if data[i][3] == '✓':
            table[(i, 3)].set_facecolor('#90EE90')  # Light green
            table[(i, 3)].set_text_props(weight='bold')
        elif data[i][3] == '✗':
            table[(i, 3)].set_facecolor('#FFB6C1')  # Light pink
            table[(i, 3)].set_text_props(weight='bold')
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white', size=13)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold testing table saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to generate both tables."""
    # Generate threshold testing table
    threshold_path = Path('plots/summary/threshold_testing_table.png')
    generate_threshold_testing_table(save_path=threshold_path)
    
    # Generate combined summary table
    combined_path = Path('plots/summary/combined_summary_table.png')
    generate_combined_summary_table(save_path=combined_path)
    
    print("\nAll tables generation complete!")


if __name__ == '__main__':
    main()

