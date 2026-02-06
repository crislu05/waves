#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 18:47:27 2026

@author: gracemurray1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermal waves: steady-state analysis + transient analysis, with cooling excluded,
and extraction of transient time constant tau and diffusivity D.

For each file like '*_7V_*s*.csv' this script:

1) Trims the dataset to only the time when the heater is ON:
   - uses the output-voltage column (|V| > threshold) to detect heater-on region
   - discards initial "pre-heating" and final "cooling after heater off"

2) Steady-state analysis (same idea as before):
   - extracts the driving period from the filename
   - drops early cycles (transient) for the sine fit
   - fits a sine at the known frequency to each thermistor
   - plots attenuation γ_i and phase lag Δφ_i vs distance

3) Transient analysis (while heater is ON):
   - divides the trimmed dataset into full periods
   - computes cycle-averaged temperature at each thermistor
   - for selected thermistors (default: 0, 3, 7), fits:
         T_mean(t) = T_inf + A * exp(-t / tau)
   - prints the fitted parameters and plots data + fit
   - converts tau to a transient diffusivity estimate via
         D_trans = L_eff^2 / (pi^2 * tau)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Try to use SciPy for a clean non-linear fit if available
try:
    from scipy.optimize import curve_fit
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("SciPy not found; using simple log-linear fit for transient.")

# ---------------------------------
# Configuration
# ---------------------------------
DATA_DIR = None               # None -> current directory / script directory
FILE_GLOB = "*_7V_*s*.csv"
DELTA_D_M = 5e-3              # thermistor spacing (5 mm)

REQUESTED_CYCLES_TO_DROP = 10
MIN_CYCLES_TO_KEEP = 2        # for steady-state fitting

THERMISTORS_FOR_TRANSIENT = [0, 3, 7]   # which thermistors to fit transient on
HEATER_V_THRESHOLD = 0.1      # volts; heater considered ON when |V| > threshold

L_EFF = 0.035                 # effective length of rod for transient (m)
PI2 = np.pi**2

# ---------------------------------
# Basic helpers
# ---------------------------------
def load_dataset(path):
    data = pd.read_csv(path, header=3)
    t = data.iloc[:, 0].to_numpy()
    out_v = data.iloc[:, 1].to_numpy()
    out_i = data.iloc[:, 2].to_numpy()
    temps = data.iloc[:, 3:].to_numpy()
    return t, out_v, out_i, temps


def period_from_filename(fname: str) -> float:
    m = re.search(r"_([0-9]+)s", fname)
    if not m:
        raise ValueError(f"Could not parse period from filename: {fname}")
    return float(m.group(1))


def metal_from_filename(fname: str) -> str:
    m = re.match(r"([A-Za-z]+)_7V_", fname)
    if not m:
        return "Unknown"
    raw = m.group(1)
    if raw.lower() == "al":
        return "Aluminium"
    else:
        return raw.capitalize()


def compute_cycles_available(t, period_s):
    duration = t[-1] - t[0]
    if duration <= 0:
        return 0
    return int(duration // period_s)


def trim_to_heater_on(t, out_v, temps, v_threshold=HEATER_V_THRESHOLD):
    """
    Trim the data so we only keep the interval where the heater is ON:
    defined by |out_v| > v_threshold.
    """
    mask_on = np.abs(out_v) > v_threshold
    if not np.any(mask_on):
        print("  WARNING: no heater-on region detected (voltage never exceeds threshold).")
        return t, out_v, temps

    first_on = np.argmax(mask_on)
    last_on = len(mask_on) - 1 - np.argmax(mask_on[::-1])

    t_trim = t[first_on:last_on + 1]
    out_v_trim = out_v[first_on:last_on + 1]
    temps_trim = temps[first_on:last_on + 1, :]

    return t_trim, out_v_trim, temps_trim


def safe_drop_initial_cycles(t, y2d, period_s,
                             requested_drop,
                             min_cycles_to_keep=2):
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
    n_cycles = compute_cycles_available(t, period_s)
    if n_cycles <= 0:
        return t, y2d
    t_end = t[0] + n_cycles * period_s
    mask = t <= t_end
    return t[mask], y2d[mask, :]


def fit_sine_fixedfreq(t, y, omega):
    s = np.sin(omega * t)
    c = np.cos(omega * t)
    M = np.column_stack([s, c, np.ones_like(t)])
    B, C, offset = np.linalg.lstsq(M, y, rcond=None)[0]
    A = np.sqrt(B**2 + C**2)
    phi = np.arctan2(C, B)
    return A, phi

# ---------------------------------
# Steady-state analysis (as before)
# ---------------------------------
def steady_state_analysis(t, temps, period_s, fname, show_plots=True):
    omega = 2 * np.pi / period_s

    t2, temps2, dropped = safe_drop_initial_cycles(
        t, temps, period_s,
        requested_drop=REQUESTED_CYCLES_TO_DROP,
        min_cycles_to_keep=MIN_CYCLES_TO_KEEP,
    )
    if t2 is None:
        print(f"  WARNING: {fname}: not enough data after dropping cycles; skipped steady-state.")
        return

    t3, temps3 = keep_integer_cycles(t2, temps2, period_s)

    n_therm = temps3.shape[1]
    amps = np.zeros(n_therm)
    phases = np.zeros(n_therm)

    for i in range(n_therm):
        A, phi = fit_sine_fixedfreq(t3, temps3[:, i], omega)
        amps[i] = A
        phases[i] = phi

    gamma = amps / amps[0]

    dphi_raw = phases - phases[0]
    dphi_unwrapped = np.unwrap(dphi_raw)
    dphi_lag = -dphi_unwrapped

    idx = np.arange(n_therm)
    x = idx * DELTA_D_M

    if show_plots:
        plt.figure(figsize=(9, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, gamma, "o-")
        plt.xlabel("Distance x (m)")
        plt.ylabel("Attenuation γᵢ")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x, dphi_lag, "o-")
        plt.xlabel("Distance x (m)")
        plt.ylabel("Phase lag Δϕᵢ (rad)")
        plt.grid(True)

        plt.suptitle(f"{fname} (dropped {dropped} cycles for steady-state)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])

    return x, gamma, dphi_lag, dropped

# ---------------------------------
# Transient analysis: cycle-mean and exponential fit
# ---------------------------------
def compute_cycle_means(t, temps, period_s):
    n_cycles = compute_cycles_available(t, period_s)
    if n_cycles <= 0:
        return None, None

    n_therm = temps.shape[1]
    mean_temp = np.zeros((n_cycles, n_therm))
    t_cycle_centres = np.zeros(n_cycles)

    t0 = t[0]
    for j in range(n_cycles):
        t_start = t0 + j * period_s
        t_end = t0 + (j + 1) * period_s
        mask = (t >= t_start) & (t < t_end)
        if not np.any(mask):
            continue
        t_seg = t[mask]
        temps_seg = temps[mask, :]

        t_cycle_centres[j] = 0.5 * (t_start + t_end)
        mean_temp[j, :] = temps_seg.mean(axis=0)

    return t_cycle_centres, mean_temp


def exponential_model(t, T_inf, A, tau):
    return T_inf + A * np.exp(-t / tau)


def fit_transient_single_series(t_c, T_c):
    """
    Fit exponential decay: T = T_inf + A * exp(-t / tau)
    Returns: (T_inf, A, tau, tau_uncertainty)
    """
    # Remove NaNs
    mask = np.isfinite(t_c) & np.isfinite(T_c)
    t_c = t_c[mask]
    T_c = T_c[mask]
    if len(t_c) < 3:
        return np.nan, np.nan, np.nan, np.nan

    n = len(t_c)
    last_slice = max(1, n // 5)
    T_inf0 = np.mean(T_c[-last_slice:])
    A0 = T_c[0] - T_inf0
    if A0 == 0:
        A0 = 1e-3
    tau0 = (t_c[-1] - t_c[0]) / 3.0 if t_c[-1] > t_c[0] else 10.0

    if HAVE_SCIPY:
        try:
            popt, pcov = curve_fit(
                exponential_model,
                t_c,
                T_c,
                p0=[T_inf0, A0, tau0],
                maxfev=10000,
            )
            # Extract uncertainty for tau (third parameter, index 2)
            tau_uncertainty = np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else np.nan
            return popt[0], popt[1], popt[2], tau_uncertainty  # T_inf, A, tau, tau_uncertainty
        except Exception as e:
            print(f"  SciPy fit failed, falling back to log-linear: {e}")

    # log-linear fallback (no uncertainty estimate)
    T_inf = T_inf0
    y = T_c - T_inf
    mask2 = (np.abs(y) > 1e-3)
    t_fit = t_c[mask2]
    y_fit = y[mask2]
    if len(t_fit) < 2:
        return T_inf, A0, tau0, np.nan
    ln_y = np.log(np.abs(y_fit))
    slope, intercept = np.polyfit(t_fit, ln_y, 1)
    tau = -1.0 / slope if slope != 0 else tau0
    A = np.sign(y_fit[0]) * np.exp(intercept)
    return T_inf, A, tau, np.nan  # No uncertainty estimate for fallback method


def plot_thermistor_comparison(t_cycles, mean_temp_exp, mean_temp_model, 
                                T_inf_exp, A_exp, tau_exp, tau_unc_exp,
                                T_inf_model, A_model, tau_model, tau_unc_model,
                                thermistor_id, save_path=None):
    """
    Plot comparison of experimental vs model cycle-averaged temperatures for a single thermistor.
    
    Parameters:
    - t_cycles: time array for cycle centers
    - mean_temp_exp: experimental cycle-averaged temperatures
    - mean_temp_model: model cycle-averaged temperatures
    - T_inf_exp, A_exp, tau_exp, tau_unc_exp: experimental fit parameters
    - T_inf_model, A_model, tau_model, tau_unc_model: model fit parameters
    - thermistor_id: thermistor ID for labeling
    - save_path: path to save the plot (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot cycle-averaged data points
    ax.plot(t_cycles, mean_temp_exp, 'o', color='#E74C3C', markersize=6, 
            label='Experimental', alpha=0.7)
    ax.plot(t_cycles, mean_temp_model, 's', color='#3498DB', markersize=6, 
            label='Model', alpha=0.7)
    
    # Generate smooth fit curves
    t_fit = np.linspace(t_cycles[0], t_cycles[-1], 200)
    
    # Experimental fit curve (no label)
    if not (np.isnan(tau_exp) or np.isnan(A_exp) or np.isnan(T_inf_exp)):
        T_fit_exp = exponential_model(t_fit, T_inf_exp, A_exp, tau_exp)
        ax.plot(t_fit, T_fit_exp, '--', color='#C0392B', linewidth=2, alpha=0.8)
    
    # Model fit curve (no label)
    if not (np.isnan(tau_model) or np.isnan(A_model) or np.isnan(T_inf_model)):
        T_fit_model = exponential_model(t_fit, T_inf_model, A_model, tau_model)
        ax.plot(t_fit, T_fit_model, '--', color='#2980B9', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel('Cycle-Averaged Temperature (°C)', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def transient_analysis(t, temps_model, temps_exp, period_s, save_dir=None, thermistor_ids=None):
    """
    Computes cycle means, fits exponentials for selected thermistors,
    and saves tau values with uncertainties to CSV.
    
    Parameters:
    - t: time array
    - temps_model: model temperatures array (n_time, n_thermistors)
    - temps_exp: experimental temperatures array (n_time, n_thermistors)
    - period_s: period in seconds
    - save_dir: directory to save CSV (if None, prints to console)
    - thermistor_ids: list of thermistor IDs to analyze (if None, analyzes all available thermistors)
    """
    if thermistor_ids is None:
        # Analyze all available thermistors
        n_thermistors = temps_model.shape[1]
        thermistor_ids = list(range(n_thermistors))
    
    t_cycles_model, mean_temp_model = compute_cycle_means(t, temps_model, period_s)
    t_cycles_exp, mean_temp_exp = compute_cycle_means(t, temps_exp, period_s)
    
    if t_cycles_model is None or t_cycles_exp is None:
        print("  WARNING: could not compute cycle means.")
        return

    # For T = 60 s, drop the first cycle (we started mid-transient)
    if abs(period_s - 60.0) < 1e-6 and len(t_cycles_model) > 1:
        t_cycles_model = t_cycles_model[1:]
        mean_temp_model = mean_temp_model[1:, :]
        t_cycles_exp = t_cycles_exp[1:]
        mean_temp_exp = mean_temp_exp[1:, :]

    print(f"\n  Transient analysis")
    
    # Store results
    results = []
    
    # Store fit parameters for thermistor 0 plotting
    thermistor_0_data = None

    for idx in thermistor_ids:
        if idx >= mean_temp_model.shape[1]:
            print(f"  WARNING: Thermistor {idx} not available (only {mean_temp_model.shape[1]} thermistors)")
            continue
        
        # Model fit
        T_c_model = mean_temp_model[:, idx]
        T_inf_model, A_model, tau_model, tau_unc_model = fit_transient_single_series(t_cycles_model, T_c_model)
        
        # Experimental fit
        T_c_exp = mean_temp_exp[:, idx]
        T_inf_exp, A_exp, tau_exp, tau_unc_exp = fit_transient_single_series(t_cycles_exp, T_c_exp)
        
        results.append({
            'thermistor_id': idx,
            'experimental_tau': tau_exp,
            'experimental_tau_uncertainty': tau_unc_exp,
            'modeling_tau': tau_model,
            'modeling_tau_uncertainty': tau_unc_model
        })
        
        # Store data for thermistor 0
        if idx == 0:
            # Use model time array (should be same as exp, but ensure consistency)
            t_cycles_plot = t_cycles_model if len(t_cycles_model) == len(t_cycles_exp) else t_cycles_exp
            thermistor_0_data = {
                't_cycles': t_cycles_plot,
                'mean_temp_exp': T_c_exp,
                'mean_temp_model': T_c_model,
                'T_inf_exp': T_inf_exp,
                'A_exp': A_exp,
                'tau_exp': tau_exp,
                'tau_unc_exp': tau_unc_exp,
                'T_inf_model': T_inf_model,
                'A_model': A_model,
                'tau_model': tau_model,
                'tau_unc_model': tau_unc_model
            }

        print(
            f"    Thermistor {idx}: "
            f"Model: tau = {tau_model: .1f} ± {tau_unc_model: .1f} s | "
            f"Exp: tau = {tau_exp: .1f} ± {tau_unc_exp: .1f} s"
        )

    # Plot thermistor 0 comparison
    if thermistor_0_data is not None and save_dir:
        plot_path = Path(save_dir) / "thermistor_0_comparison.png"
        plot_thermistor_comparison(
            thermistor_0_data['t_cycles'],
            thermistor_0_data['mean_temp_exp'],
            thermistor_0_data['mean_temp_model'],
            thermistor_0_data['T_inf_exp'],
            thermistor_0_data['A_exp'],
            thermistor_0_data['tau_exp'],
            thermistor_0_data['tau_unc_exp'],
            thermistor_0_data['T_inf_model'],
            thermistor_0_data['A_model'],
            thermistor_0_data['tau_model'],
            thermistor_0_data['tau_unc_model'],
            thermistor_id=0,
            save_path=plot_path
        )

    # Save to CSV
    if results:
        df_results = pd.DataFrame(results)
        
        if save_dir:
            csv_path = Path(save_dir) / "transient_tau_results.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(csv_path, index=False)
            print(f"\n  Saved tau results to: {csv_path}")
        else:
            print("\n  Tau results:")
            print(df_results.to_string(index=False))


def generate_tau_comparison_table(csv_path, save_path=None, k_factor=2.0):
    """
    Generate a table image with averaged experimental and modeling tau values.
    
    Parameters:
    - csv_path: Path to CSV file with tau results (from transient_analysis)
    - save_path: Path to save the table image (if None, saves to same directory as CSV)
    - k_factor: Not used, kept for compatibility
    """
    df = pd.read_csv(csv_path)
    
    # Format values with uncertainties
    def format_with_uncertainty(value, uncertainty, decimals=1):
        """Format value ± uncertainty."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.1f}'
        return f'{value:.1f} ± {uncertainty:.1f}'
    
    # Calculate averaged values
    # For tau: use mean
    # For uncertainty: use RMS (root mean square) of uncertainties
    tau_exp_avg = np.nanmean(df['experimental_tau'].values)
    tau_exp_unc_avg = np.sqrt(np.nanmean(df['experimental_tau_uncertainty'].values**2))
    
    tau_model_avg = np.nanmean(df['modeling_tau'].values)
    tau_model_unc_avg = np.sqrt(np.nanmean(df['modeling_tau_uncertainty'].values**2))
    
    # Format averaged values
    exp_str = format_with_uncertainty(tau_exp_avg, tau_exp_unc_avg)
    model_str = format_with_uncertainty(tau_model_avg, tau_model_unc_avg)
    
    # Prepare table data (only header and one averaged row)
    data = [['Parameter', 'Value']]
    data.append(['Experimental τ (s)', exp_str])
    data.append(['Modeling τ (s)', model_str])
    
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
    if save_path is None:
        save_path = Path(csv_path).parent / "tau_comparison_table.png"
    else:
        save_path = Path(save_path)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\n  Tau comparison table saved to: {save_path}')
    plt.close()


# ---------------------------------
# Load comparison data
# ---------------------------------
def load_comparison_data(csv_path):
    """
    Load model and experimental temperature data from comparison CSV.
    
    Returns:
    - t: time array
    - temps_model: model temperatures (n_time, n_thermistors)
    - temps_exp: experimental temperatures (n_time, n_thermistors)
    - thermistor_ids: list of thermistor IDs
    """
    df = pd.read_csv(csv_path)
    
    # Get time
    t = df['Time (s)'].values
    
    # Get all model columns
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    model_cols = sorted(model_cols)  # Sort to ensure consistent ordering
    
    # Extract thermistor IDs and positions
    thermistor_ids = []
    temps_model_list = []
    temps_exp_list = []
    
    for model_col in model_cols:
        # Extract thermistor ID: "T_model_0_x3.0mm (°C)" -> 0
        parts = model_col.split('_')
        therm_id = int(parts[2])
        thermistor_ids.append(therm_id)
        
        # Get corresponding experimental column
        exp_col = model_col.replace('T_model_', 'T_exp_')
        
        temps_model_list.append(df[model_col].values)
        temps_exp_list.append(df[exp_col].values)
    
    # Convert to arrays (n_time, n_thermistors)
    temps_model = np.column_stack(temps_model_list)
    temps_exp = np.column_stack(temps_exp_list)
    
    return t, temps_model, temps_exp, thermistor_ids


def estimate_period_from_data(t, temps, min_period=10.0, max_period=120.0):
    """
    Estimate the period from the temperature data using FFT.
    For better period detection, we focus on frequencies corresponding to periods
    between min_period and max_period.
    """
    # Use first thermistor for period estimation
    T = temps[:, 0]
    
    # Remove mean and detrend (remove linear trend to focus on oscillations)
    T_detrend = T - np.mean(T)
    # Remove linear trend
    t_norm = t - t[0]
    if len(t_norm) > 1:
        coeffs = np.polyfit(t_norm, T_detrend, 1)
        T_detrend = T_detrend - (coeffs[0] * t_norm + coeffs[1])
    
    # FFT
    dt = np.mean(np.diff(t))
    fft_vals = np.fft.rfft(T_detrend)
    fft_freq = np.fft.rfftfreq(len(T_detrend), dt)
    
    # Focus on frequencies corresponding to periods between min_period and max_period
    valid_mask = (fft_freq > 1.0/max_period) & (fft_freq < 1.0/min_period)
    if np.any(valid_mask):
        power = np.abs(fft_vals[valid_mask])**2
        valid_freqs = fft_freq[valid_mask]
        peak_idx = np.argmax(power)
        peak_freq = valid_freqs[peak_idx]
        
        if peak_freq > 0:
            estimated_period = 1.0 / peak_freq
            estimated_period = np.clip(estimated_period, min_period, max_period)
            return estimated_period
    
    # Fallback: use known period
    return 10.0


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
    t, temps_model, temps_exp, thermistor_ids = load_comparison_data(csv_path)
    
    print(f"Loaded data: {len(t)} time points, {len(thermistor_ids)} thermistors")
    print(f"Thermistor IDs: {thermistor_ids}")
    
    # Use known period (10 seconds)
    period_s = 10.0
    print(f"Using period: {period_s:.2f} s")
    
    # Optional: also estimate from data for verification
    estimated_period = estimate_period_from_data(t, temps_exp)
    print(f"Estimated period from FFT: {estimated_period:.2f} s (for reference)")
    
    # Set up save directory
    save_dir = script_dir / "plots" / "transient"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {save_dir}")
    
    # Run transient analysis for all thermistors
    transient_analysis(t, temps_model, temps_exp, period_s, save_dir=save_dir, 
                      thermistor_ids=None)  # None means analyze all
    
    # Generate comparison table
    csv_path = save_dir / "transient_tau_results.csv"
    if csv_path.exists():
        generate_tau_comparison_table(csv_path, save_path=save_dir / "tau_comparison_table.png")
    
    print("\nTransient analysis complete!")


if __name__ == "__main__":
    main()
