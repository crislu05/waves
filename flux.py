"""
Plot Qc and Qh heat fluxes vs time using the model from heatpump.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from heatpump import (
    load_dataset, solve_coupled_heat_pump,
    calculate_qc, calculate_qh,
    setup_parameters, create_voltage_interpolation,
    setup_initial_conditions
)


def calculate_fluxes_over_time(sol, timestamp, voltage_func, params):
    """
    Calculate Qc and Qh heat fluxes at each time point.
    
    Parameters:
    - sol: Solution object from solve_ivp
    - timestamp: Time points to evaluate at (experimental timestamps, for reference)
    - voltage_func: Function V(t) returning voltage
    - params: Parameters dictionary
    
    Returns:
    - t_eval: Time points (s) - using solution's time points for accuracy
    - Qc: Cold plate heat flux array (W)
    - Qh: Hot plate heat flux array (W)
    """
    # Use solution's own time points directly - this is the most accurate
    # Avoids interpolation artifacts from irregular experimental timestamps
    t_sol = sol.t  # Solution's internal time points
    
    # Evaluate solution at its own time points (most accurate)
    T_solution = sol.y  # Direct access to solution at t_sol points
    
    # Extract temperatures
    Tc_sol = T_solution[0, :]  # Cold plate temperature (K) at solution time points
    Th_sol = T_solution[1, :]  # Hot plate temperature (K) at solution time points
    
    # Calculate voltage and current at solution time points
    V_sol = np.array([voltage_func(t) for t in t_sol])
    I_sol = (V_sol - params['alpha'] * (Th_sol - Tc_sol)) / params['R_elec']
    
    # Calculate heat fluxes at solution time points
    Qc = np.zeros_like(t_sol)
    Qh = np.zeros_like(t_sol)
    
    for i in range(len(t_sol)):
        Qc[i] = calculate_qc(params['alpha'], I_sol[i], Tc_sol[i], params['R_elec'], params['K_therm'], Th_sol[i])
        Qh[i] = calculate_qh(params['alpha'], I_sol[i], Th_sol[i], params['R_elec'], params['K_therm'], Tc_sol[i])
    
    # Return solution's time points and fluxes directly (no interpolation)
    return t_sol, Qc, Qh


def plot_fluxes_vs_time(timestamp, Qc, Qh, save_path=None):
    """
    Plot Qc and Qh heat fluxes vs time.
    
    Parameters:
    - timestamp: Time array (s)
    - Qc: Cold plate heat flux array (W)
    - Qh: Hot plate heat flux array (W)
    - save_path: Path to save the plot (if None, plot is displayed)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Qc and Qh
    ax.plot(timestamp, Qc, 'b-', linewidth=2, label='Qc (Cold Plate Heat Flux)', alpha=0.8)
    ax.plot(timestamp, Qh, 'r-', linewidth=2, label='Qh (Hot Plate Heat Flux)', alpha=0.8)
    
    # Add zero reference line
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Zero Flux Reference')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heat Flux (W)', fontsize=12)
    ax.set_title('Qc and Qh Heat Fluxes vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
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


def main():
    """Main function to plot Qc and Qh vs time."""
    print("=" * 70)
    print("Heat Flux Plotter (Qc and Qh vs Time)")
    print("=" * 70)
    
    # Load data
    filepath = 'data/session6/brass_7V_10s.csv'
    timestamp, voltage, _, thermistor_temperatures = load_dataset(filepath)
    
    # Setup parameters
    params, L_brass, N_nodes = setup_parameters(verbose=False)
    
    # Create voltage interpolation
    voltage_interp = create_voltage_interpolation(timestamp, voltage)
    
    # Set initial conditions from thermistor 0
    thermistor_0 = thermistor_temperatures[:, 0]
    thermistor_data_dict = {0: {'data': thermistor_0, 'x_pos': 0.003}}
    T_initial, T0 = setup_initial_conditions(thermistor_data_dict, N_nodes, thermistor_id=0)
    
    # Solve system
    t_span = (timestamp[0], timestamp[-1])
    rtol = 1e-6
    atol = 1e-8
    
    print("\nSolving coupled heat pump equations...")
    sol = solve_coupled_heat_pump(t_span, T_initial, voltage_interp, params, 
                                  rtol, atol, t_eval=None, method='Radau')
    
    if not sol.success:
        print("Warning: Solver did not converge successfully!")
    
    print(f"Solution completed: {len(sol.t)} time steps")
    
    # Calculate fluxes over time
    print("\nCalculating Qc and Qh heat fluxes...")
    t_flux, Qc, Qh = calculate_fluxes_over_time(sol, timestamp, voltage_interp, params)
    
    # Subsample if too many points (for smoother plotting and to avoid artifacts)
    # Keep at most 5000 points, or use every nth point
    if len(t_flux) > 5000:
        subsample_factor = len(t_flux) // 5000
        t_flux = t_flux[::subsample_factor]
        Qc = Qc[::subsample_factor]
        Qh = Qh[::subsample_factor]
        print(f"  Subsampled to {len(t_flux)} points for plotting")
    
    # Print summary statistics
    print(f"\nFlux Summary:")
    print(f"  Qc range: {np.min(Qc):.4f} to {np.max(Qc):.4f} W")
    print(f"  Qc mean: {np.mean(Qc):.4f} W")
    print(f"  Qh range: {np.min(Qh):.4f} to {np.max(Qh):.4f} W")
    print(f"  Qh mean: {np.mean(Qh):.4f} W")
    
    # Create output directory
    output_dir = Path('plots/flux')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot and save
    plot_path = output_dir / 'Qc_Qh_vs_time.png'
    plot_fluxes_vs_time(t_flux, Qc, Qh, save_path=str(plot_path))
    
    print("\n" + "=" * 70)
    print("Heat flux plot generated successfully!")
    print("=" * 70)
    print(f"  Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()

