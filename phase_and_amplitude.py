#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase and Amplitude Analysis: Thermal waves analysis from comparison data

This script analyzes experimental and model temperature data from 
temperature_comparison_data.csv to:
- fit sine waves at the driving frequency
- calculate attenuation γ_i and phase lag Δφ_i vs distance
- extract estimates of thermal diffusivity D from:
    * attenuation vs distance (ln γ vs x)
    * phase lag vs distance (Δφ vs x)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------
# Configuration
# ---------------------------------
DELTA_D_M = 5e-3         # thermistor spacing (5 mm)

REQUESTED_CYCLES_TO_DROP = 10
MIN_CYCLES_TO_KEEP = 2

# When fitting D:
MIN_GAMMA_FOR_FIT = 0.05   # ignore thermistors where amplitude is too tiny

# Known period (from previous analysis)
PERIOD_S = 10.0


# ---------------------------------
# Helper functions
# ---------------------------------
def load_comparison_data(csv_path):
    """
    Load model and experimental temperature data from comparison CSV.
    
    Returns:
    - t: time array
    - temps_model: model temperatures (n_time, n_thermistors)
    - temps_exp: experimental temperatures (n_time, n_thermistors)
    - thermistor_ids: list of thermistor IDs
    - positions: list of positions in meters
    """
    df = pd.read_csv(csv_path)
    
    # Get time
    t = df['Time (s)'].values
    
    # Get all model columns
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    model_cols = sorted(model_cols)  # Sort to ensure consistent ordering
    
    # Extract thermistor IDs and positions
    thermistor_ids = []
    positions = []
    temps_model_list = []
    temps_exp_list = []
    
    for model_col in model_cols:
        # Extract thermistor ID and position: "T_model_0_x3.0mm (°C)" -> 0, 3.0e-3
        parts = model_col.split('_')
        therm_id = int(parts[2])
        thermistor_ids.append(therm_id)
        
        # Extract position from column name: "x3.0mm (°C)" -> 3.0e-3
        # parts[3] is "x3.0mm (°C)", so split by space to get just "x3.0mm"
        pos_part = parts[3].split()[0]  # Gets "x3.0mm" from "x3.0mm (°C)"
        # Remove 'x' and 'mm' to get just the number
        pos_str = pos_part.replace('x', '').replace('mm', '')
        pos_m = float(pos_str) * 1e-3  # Convert mm to m
        positions.append(pos_m)
        
        # Get corresponding experimental column
        exp_col = model_col.replace('T_model_', 'T_exp_')
        
        temps_model_list.append(df[model_col].values)
        temps_exp_list.append(df[exp_col].values)
    
    # Convert to arrays (n_time, n_thermistors)
    temps_model = np.column_stack(temps_model_list)
    temps_exp = np.column_stack(temps_exp_list)
    
    return t, temps_model, temps_exp, thermistor_ids, np.array(positions)


def compute_cycles_available(t, period_s):
    duration = t[-1] - t[0]
    if duration <= 0:
        return 0
    return int(duration // period_s)


def safe_drop_initial_cycles(t, y2d, period_s,
                             requested_drop,
                             min_cycles_to_keep=2):
    """
    Drop the first N cycles, but never so many that fewer than
    min_cycles_to_keep remain. Returns (t2, y2d2, actual_dropped).
    """
    n_cycles = compute_cycles_available(t, period_s)
    if n_cycles < min_cycles_to_keep:
        return None, None, 0

    max_drop_allowed = max(0, n_cycles - min_cycles_to_keep)
    drop_cycles = min(requested_drop, max_drop_allowed)

    t_start = t[0] + drop_cycles * period_s
    mask = t >= t_start
    if not np.any(mask):
        return None, None, drop_cycles

    t2 = t[mask]
    y2 = y2d[mask, :]
    return t2, y2, drop_cycles


def keep_integer_cycles(t, y2d, period_s):
    """Trim the end so we keep an integer number of cycles."""
    n_cycles = compute_cycles_available(t, period_s)
    if n_cycles <= 0:
        return t, y2d
    t_end = t[0] + n_cycles * period_s
    mask = t <= t_end
    return t[mask], y2d[mask, :]


def fit_sine_fixedfreq(t, y, omega):
    """
    Fit y(t) ≈ B sin(ωt) + C cos(ωt) + offset using least squares.
    Return (amplitude, phase) where model = A sin(ωt + φ).
    """
    s = np.sin(omega * t)
    c = np.cos(omega * t)
    M = np.column_stack([s, c, np.ones_like(t)])
    B, C, offset = np.linalg.lstsq(M, y, rcond=None)[0]

    A = np.sqrt(B**2 + C**2)
    phi = np.arctan2(C, B)   # y = B sin + C cos
    return A, phi


# ---------------------------------
# Diffusivity extraction from γ(x) and Δφ(x)
# ---------------------------------
def estimate_D_from_gamma(x, gamma, omega):
    """
    Use the model γ(x) = exp(-sqrt(ω/(2D)) * x).
    Fit ln γ vs x with a straight line (excluding very small γ).
    slope = -sqrt(ω/(2D))  ->  D = ω / (2 * slope^2)
    """
    mask = (gamma > MIN_GAMMA_FOR_FIT) & np.isfinite(gamma)
    mask[0] = False  # skip thermistor 0 (ln(1) = 0 adds little info)
    x_fit = x[mask]
    g_fit = gamma[mask]

    if len(x_fit) < 2:
        return np.nan

    y = np.log(g_fit)
    slope, intercept = np.polyfit(x_fit, y, 1)
    if slope == 0:
        return np.nan

    D = omega / (2.0 * slope**2)
    return D


def estimate_D_from_phase(x, dphi, omega):
    """
    Use the model Δφ(x) = sqrt(ω/(2D)) * x.
    Fit Δφ vs x with a straight line: slope = sqrt(ω/(2D)).
    Then D = ω / (2 * slope^2).
    """
    mask = np.isfinite(dphi)
    x_fit = x[mask]
    p_fit = dphi[mask]

    if len(x_fit) < 2:
        return np.nan

    slope, intercept = np.polyfit(x_fit, p_fit, 1)
    if slope == 0:
        return np.nan

    D = omega / (2.0 * slope**2)
    return D


# ---------------------------------
# Analyse temperature data
# ---------------------------------
def analyse_temperature_data(t, temps, period_s, label="Data"):
    """
    Analyze temperature data to extract attenuation and phase lag.
    No filtering is applied since the CSV already contains filtered data (middle 90%).
    
    Returns:
    - x: positions (m)
    - gamma: attenuation relative to thermistor 0
    - dphi: phase lag relative to thermistor 0 (rad)
    - dropped: number of cycles dropped (always 0 since no filtering)
    - D_att: diffusivity estimate from attenuation
    - D_phase: diffusivity estimate from phase
    - t_fit: time array used for fitting
    - temps_fit: temperature array used for fitting
    - amps: amplitudes for each thermistor
    - phases: phases for each thermistor
    - offsets: DC offsets for each thermistor
    """
    omega = 2 * np.pi / period_s

    # Use full dataset directly (no filtering since CSV is already filtered)
    t3 = t
    temps3 = temps
    dropped = 0

    n_therm = temps3.shape[1]
    amps = np.zeros(n_therm)
    phases = np.zeros(n_therm)
    offsets = np.zeros(n_therm)

    for i in range(n_therm):
        A, phi = fit_sine_fixedfreq(t3, temps3[:, i], omega)
        amps[i] = A
        phases[i] = phi
        # Calculate offset (DC component)
        s = np.sin(omega * t3)
        c = np.cos(omega * t3)
        M = np.column_stack([s, c, np.ones_like(t3)])
        B, C, offset = np.linalg.lstsq(M, temps3[:, i], rcond=None)[0]
        offsets[i] = offset

    # attenuation and phase lag relative to thermistor 0
    gamma = amps / amps[0]

    # raw phase difference, then unwrap & flip sign so positive = lagging
    dphi_raw = phases - phases[0]
    dphi_unwrapped = np.unwrap(dphi_raw)
    dphi_lag = -dphi_unwrapped

    # Use actual positions from data
    # We'll get positions from the data loading function
    # For now, use index-based positions (will be updated with actual positions)
    idx = np.arange(n_therm)
    x = idx * DELTA_D_M

    # ---- Diffusivity estimates from this dataset ----
    D_att = estimate_D_from_gamma(x, gamma, omega)
    D_phase = estimate_D_from_phase(x, dphi_lag, omega)

    return {
        "label": label,
        "period_s": period_s,
        "x": x,
        "gamma": gamma,
        "dphi": dphi_lag,
        "dropped_cycles": dropped,
        "D_att": D_att,
        "D_phase": D_phase,
        "t_fit": t3,
        "temps_fit": temps3,
        "amps": amps,
        "phases": phases,
        "offsets": offsets,
        "omega": omega,
    }


# ---------------------------------
# Plot: comparison of experimental and model
# ---------------------------------
def exponential_decay_model(x, A, alpha):
    """Exponential decay model: γ(x) = A * exp(-α * x)"""
    return A * np.exp(-alpha * x)


def linear_model(x, beta):
    """Linear model: Δφ(x) = β * x"""
    return beta * x


def plot_comparison(analysis_exp, analysis_model, positions, save_path=None):
    """
    Plot comparison of experimental and model attenuation and phase lag.
    Fits exponential decay to attenuation and linear fit to phase lag.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('')  # Ensure no title

    # Use actual positions (convert to mm for display)
    x_exp = positions if len(positions) == len(analysis_exp["gamma"]) else analysis_exp["x"]
    x_model = positions if len(positions) == len(analysis_model["gamma"]) else analysis_model["x"]
    
    # Convert to mm for plotting (assuming positions are in meters)
    x_exp_mm = x_exp * 1000
    x_model_mm = x_model * 1000

    # left panel: attenuation with exponential decay fit
    ax = axes[0]
    
    # Plot data points (using mm for x-axis)
    ax.plot(x_exp_mm, analysis_exp["gamma"], "o", label="Experimental", 
            color='#E74C3C', markersize=6, alpha=0.7)
    ax.plot(x_model_mm, analysis_model["gamma"], "s", label="Model", 
            color='#3498DB', markersize=6, alpha=0.7)
    
    # Fit exponential decay: γ(x) = A * exp(-α * x)
    # Use log-linear fit: ln(γ) = ln(A) - α*x
    # Note: fitting uses original x values in meters
    mask_exp = (analysis_exp["gamma"] > MIN_GAMMA_FOR_FIT) & np.isfinite(analysis_exp["gamma"])
    mask_model = (analysis_model["gamma"] > MIN_GAMMA_FOR_FIT) & np.isfinite(analysis_model["gamma"])
    
    if np.any(mask_exp):
        x_fit_exp = x_exp[mask_exp]  # Use meters for fitting
        gamma_fit_exp = analysis_exp["gamma"][mask_exp]
        ln_gamma_exp = np.log(gamma_fit_exp)
        # Use polyfit with full output to get covariance
        p_exp, cov_exp = np.polyfit(x_fit_exp, ln_gamma_exp, 1, cov=True)
        slope_exp, intercept_exp = p_exp
        alpha_exp = -slope_exp  # attenuation decay factor
        alpha_exp_unc = np.sqrt(cov_exp[0, 0])  # uncertainty in slope = uncertainty in alpha
        A_exp = np.exp(intercept_exp)
        
        # Plot fitted curve (convert x to mm for display)
        x_smooth = np.linspace(x_exp[0], x_exp[-1], 200)
        x_smooth_mm = x_smooth * 1000
        gamma_fit_exp_smooth = exponential_decay_model(x_smooth, A_exp, alpha_exp)
        ax.plot(x_smooth_mm, gamma_fit_exp_smooth, "-", color='#C0392B', linewidth=2,
                alpha=0.8)
    
    if np.any(mask_model):
        x_fit_model = x_model[mask_model]  # Use meters for fitting
        gamma_fit_model = analysis_model["gamma"][mask_model]
        ln_gamma_model = np.log(gamma_fit_model)
        # Use polyfit with full output to get covariance
        p_model, cov_model = np.polyfit(x_fit_model, ln_gamma_model, 1, cov=True)
        slope_model, intercept_model = p_model
        alpha_model = -slope_model  # attenuation decay factor
        alpha_model_unc = np.sqrt(cov_model[0, 0])  # uncertainty in slope = uncertainty in alpha
        A_model = np.exp(intercept_model)
        
        # Plot fitted curve (convert x to mm for display)
        x_smooth = np.linspace(x_model[0], x_model[-1], 200)
        x_smooth_mm = x_smooth * 1000
        gamma_fit_model_smooth = exponential_decay_model(x_smooth, A_model, alpha_model)
        ax.plot(x_smooth_mm, gamma_fit_model_smooth, "-", color='#2980B9', linewidth=2,
                alpha=0.8)
    
    ax.set_xlabel("Distance along the rod (mm)", fontsize=20)
    ax.set_ylabel("Attenuation γᵢ", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # right panel: phase lag with linear fit
    ax = axes[1]
    
    # Plot data points (using mm for x-axis)
    ax.plot(x_exp_mm, analysis_exp["dphi"], "o", label="Experimental", 
            color='#E74C3C', markersize=6, alpha=0.7)
    ax.plot(x_model_mm, analysis_model["dphi"], "s", label="Model", 
            color='#3498DB', markersize=6, alpha=0.7)
    
    # Fit linear: Δφ(x) = β * x
    # Note: fitting uses original x values in meters
    mask_exp_phi = np.isfinite(analysis_exp["dphi"])
    mask_model_phi = np.isfinite(analysis_model["dphi"])
    
    if np.any(mask_exp_phi) and len(x_exp[mask_exp_phi]) >= 2:
        x_fit_exp_phi = x_exp[mask_exp_phi]  # Use meters for fitting
        dphi_fit_exp = analysis_exp["dphi"][mask_exp_phi]
        # Use polyfit with full output to get covariance
        p_exp_phi, cov_exp_phi = np.polyfit(x_fit_exp_phi, dphi_fit_exp, 1, cov=True)
        beta_exp, intercept_exp_phi = p_exp_phi
        beta_exp_unc = np.sqrt(cov_exp_phi[0, 0])  # uncertainty in slope = uncertainty in beta
        
        # Plot fitted line (convert x to mm for display)
        x_smooth = np.linspace(x_exp[0], x_exp[-1], 200)
        x_smooth_mm = x_smooth * 1000
        dphi_fit_exp_smooth = linear_model(x_smooth, beta_exp) + intercept_exp_phi
        ax.plot(x_smooth_mm, dphi_fit_exp_smooth, "-", color='#C0392B', linewidth=2,
                alpha=0.8)
    
    if np.any(mask_model_phi) and len(x_model[mask_model_phi]) >= 2:
        x_fit_model_phi = x_model[mask_model_phi]  # Use meters for fitting
        dphi_fit_model = analysis_model["dphi"][mask_model_phi]
        # Use polyfit with full output to get covariance
        p_model_phi, cov_model_phi = np.polyfit(x_fit_model_phi, dphi_fit_model, 1, cov=True)
        beta_model, intercept_model_phi = p_model_phi
        beta_model_unc = np.sqrt(cov_model_phi[0, 0])  # uncertainty in slope = uncertainty in beta
        
        # Plot fitted line (convert x to mm for display)
        x_smooth = np.linspace(x_model[0], x_model[-1], 200)
        x_smooth_mm = x_smooth * 1000
        dphi_fit_model_smooth = linear_model(x_smooth, beta_model) + intercept_model_phi
        ax.plot(x_smooth_mm, dphi_fit_model_smooth, "-", color='#2980B9', linewidth=2,
                alpha=0.8)
    
    ax.set_xlabel("", fontsize=20)  # Remove x-axis label for right graph
    ax.set_ylabel("Phase lag Δϕᵢ (rad)", fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    # Update positions in analysis results for diffusivity calculation
    analysis_exp["x"] = x_exp
    analysis_model["x"] = x_model
    
    # Store fitted parameters with uncertainties
    if np.any(mask_exp):
        analysis_exp["attenuation_decay_factor"] = alpha_exp
        analysis_exp["attenuation_decay_factor_uncertainty"] = alpha_exp_unc
    if np.any(mask_model):
        analysis_model["attenuation_decay_factor"] = alpha_model
        analysis_model["attenuation_decay_factor_uncertainty"] = alpha_model_unc
    if np.any(mask_exp_phi) and len(x_exp[mask_exp_phi]) >= 2:
        analysis_exp["phase_lag_coeff"] = beta_exp
        analysis_exp["phase_lag_coeff_uncertainty"] = beta_exp_unc
    if np.any(mask_model_phi) and len(x_model[mask_model_phi]) >= 2:
        analysis_model["phase_lag_coeff"] = beta_model
        analysis_model["phase_lag_coeff_uncertainty"] = beta_model_unc

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_fitted_curves(analysis_exp, analysis_model, save_path=None, n_thermistors_to_plot=8):
    """
    Plot fitted sine curves overlaid on experimental and model temperature data.
    Shows a grid of subplots, one for each thermistor.
    """
    # Determine number of thermistors to plot
    n_therm = min(n_thermistors_to_plot, len(analysis_exp["amps"]))
    
    # Create subplot grid (2 columns, multiple rows)
    n_cols = 2
    n_rows = (n_therm + 1) // 2  # Round up
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    omega_exp = analysis_exp["omega"]
    omega_model = analysis_model["omega"]
    t_exp = analysis_exp["t_fit"]
    t_model = analysis_model["t_fit"]
    
    for i in range(n_therm):
        ax = axes[i]
        
        # Experimental data and fit
        temps_exp = analysis_exp["temps_fit"][:, i]
        A_exp = analysis_exp["amps"][i]
        phi_exp = analysis_exp["phases"][i]
        offset_exp = analysis_exp["offsets"][i]
        
        # Model data and fit
        temps_model = analysis_model["temps_fit"][:, i]
        A_model = analysis_model["amps"][i]
        phi_model = analysis_model["phases"][i]
        offset_model = analysis_model["offsets"][i]
        
        # Plot raw data (subsampled for clarity)
        step = max(1, len(t_exp) // 500)  # Show max 500 points
        ax.plot(t_exp[::step], temps_exp[::step], 'o', color='#E74C3C', 
                markersize=2, alpha=0.5, label='Experimental' if i == 0 else '')
        ax.plot(t_model[::step], temps_model[::step], 's', color='#3498DB', 
                markersize=2, alpha=0.5, label='Model' if i == 0 else '')
        
        # Plot fitted curves
        t_fit_smooth = np.linspace(t_exp[0], t_exp[-1], 1000)
        fit_exp = offset_exp + A_exp * np.sin(omega_exp * t_fit_smooth + phi_exp)
        fit_model = offset_model + A_model * np.sin(omega_model * t_fit_smooth + phi_model)
        
        ax.plot(t_fit_smooth, fit_exp, '--', color='#C0392B', linewidth=1.5, 
                label='Exp fit' if i == 0 else '', alpha=0.8)
        ax.plot(t_fit_smooth, fit_model, '--', color='#2980B9', linewidth=1.5, 
                label='Model fit' if i == 0 else '', alpha=0.8)
        
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Temperature (°C)', fontsize=10)
        ax.set_title(f'Thermistor {i}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_therm, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fitted curves plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_coefficient_table(analysis_exp, analysis_model, coefficient_name, coefficient_label, save_path=None):
    """
    Generate a table image with averaged experimental and modeling coefficient values.
    Similar style to the tau comparison table.
    
    Parameters:
    - analysis_exp: experimental analysis results dictionary
    - analysis_model: model analysis results dictionary
    - coefficient_name: key name in analysis dict (e.g., 'attenuation_decay_factor' or 'phase_lag_coeff')
    - coefficient_label: label for the table (e.g., 'α (m⁻¹)' or 'β (rad/m)')
    - save_path: path to save the table image
    """
    # Format values with uncertainties (1 decimal place to match tau table style)
    def format_with_uncertainty(value, uncertainty, decimals=1):
        """Format value ± uncertainty."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.{decimals}f}'
        return f'{value:.{decimals}f} ± {uncertainty:.{decimals}f}'
    
    # Get coefficient values and uncertainties
    coeff_exp = analysis_exp.get(coefficient_name, np.nan)
    coeff_exp_unc = analysis_exp.get(f'{coefficient_name}_uncertainty', np.nan)
    coeff_model = analysis_model.get(coefficient_name, np.nan)
    coeff_model_unc = analysis_model.get(f'{coefficient_name}_uncertainty', np.nan)
    
    # Format values with uncertainties
    exp_str = format_with_uncertainty(coeff_exp, coeff_exp_unc)
    model_str = format_with_uncertainty(coeff_model, coeff_model_unc)
    
    # Prepare table data (only header and two rows)
    data = [['Parameter', 'Value']]
    data.append([f'Experimental {coefficient_label}', exp_str])
    data.append([f'Modeling {coefficient_label}', model_str])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center',
                    colWidths=[0.5, 0.5])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.0)
    
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white', size=10)
    
    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'  Coefficient table saved to: {save_path}')
    else:
        plt.show()
    
    plt.close()


# ---------------------------------
# Main
# ---------------------------------
def main():
    # Load comparison data
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "data" / "comparison" / "temperature_comparison_data.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Comparison data not found at: {csv_path}")
    
    print(f"Loading comparison data from: {csv_path}")
    t, temps_model, temps_exp, thermistor_ids, positions = load_comparison_data(csv_path)
    
    print(f"Loaded data: {len(t)} time points, {len(thermistor_ids)} thermistors")
    print(f"Thermistor IDs: {thermistor_ids}")
    print(f"Positions: {positions * 1e3} mm")
    print(f"Using period: {PERIOD_S:.2f} s")
    print()

    # Analyze experimental data
    print("Analyzing experimental data...")
    analysis_exp = analyse_temperature_data(t, temps_exp, PERIOD_S, label="Experimental")
    
    # Analyze model data
    print("Analyzing model data...")
    analysis_model = analyse_temperature_data(t, temps_model, PERIOD_S, label="Model")

    if analysis_exp is None or analysis_model is None:
        print("ERROR: Could not complete analysis")
        return

    # Update positions in analysis results
    analysis_exp["x"] = positions
    analysis_model["x"] = positions

    # Recalculate diffusivity with correct positions
    omega = 2 * np.pi / PERIOD_S
    analysis_exp["D_att"] = estimate_D_from_gamma(positions, analysis_exp["gamma"], omega)
    analysis_exp["D_phase"] = estimate_D_from_phase(positions, analysis_exp["dphi"], omega)
    analysis_model["D_att"] = estimate_D_from_gamma(positions, analysis_model["gamma"], omega)
    analysis_model["D_phase"] = estimate_D_from_phase(positions, analysis_model["dphi"], omega)

    # Plot comparison
    save_dir = script_dir / "plots" / "phase_amplitude"
    save_path = save_dir / "phase_amplitude_comparison.png"
    plot_comparison(analysis_exp, analysis_model, positions, save_path=save_path)
    
    # Plot fitted curves
    fitted_curves_path = save_dir / "fitted_curves_comparison.png"
    plot_fitted_curves(analysis_exp, analysis_model, save_path=fitted_curves_path)
    
    # Generate coefficient tables
    if 'attenuation_decay_factor' in analysis_exp and 'attenuation_decay_factor' in analysis_model:
        attenuation_table_path = save_dir / "attenuation_coefficient_table.png"
        generate_coefficient_table(
            analysis_exp, analysis_model,
            'attenuation_decay_factor',
            'α',
            save_path=attenuation_table_path
        )
    
    if 'phase_lag_coeff' in analysis_exp and 'phase_lag_coeff' in analysis_model:
        phase_table_path = save_dir / "phase_shift_coefficient_table.png"
        generate_coefficient_table(
            analysis_exp, analysis_model,
            'phase_lag_coeff',
            'β',
            save_path=phase_table_path
        )

    # Print results
    print("\n=== Results ===")
    print(f"Period: {PERIOD_S:.1f} s")
    print(f"Dropped cycles - Experimental: {analysis_exp['dropped_cycles']}, Model: {analysis_model['dropped_cycles']}")
    print()
    print("Experimental:")
    print(f"  D from attenuation: {analysis_exp['D_att']:.3e} m²/s")
    print(f"  D from phase:       {analysis_exp['D_phase']:.3e} m²/s")
    if 'attenuation_decay_factor' in analysis_exp:
        print(f"  Attenuation decay factor (α): {analysis_exp['attenuation_decay_factor']:.3f} m⁻¹")
    if 'phase_lag_coeff' in analysis_exp:
        print(f"  Phase lag coefficient (β):   {analysis_exp['phase_lag_coeff']:.3f} rad/m")
    print()
    print("Model:")
    print(f"  D from attenuation: {analysis_model['D_att']:.3e} m²/s")
    print(f"  D from phase:       {analysis_model['D_phase']:.3e} m²/s")
    if 'attenuation_decay_factor' in analysis_model:
        print(f"  Attenuation decay factor (α): {analysis_model['attenuation_decay_factor']:.3f} m⁻¹")
    if 'phase_lag_coeff' in analysis_model:
        print(f"  Phase lag coefficient (β):   {analysis_model['phase_lag_coeff']:.3f} rad/m")
        print()


if __name__ == "__main__":
    main()
