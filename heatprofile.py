"""
Generate static heatmaps showing temperature profile along brass rod and plate temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

from heatpump import (
    load_dataset, solve_coupled_heat_pump,
    setup_parameters, create_voltage_interpolation,
    setup_initial_conditions
)


def create_temperature_heatmap(sol, x_grid, timestamp, save_path):
    """
    Create static PNG heatmap showing temperature profile along brass rod vs time.
    
    Parameters:
    - sol: Solution object from solve_ivp
    - x_grid: Spatial grid positions along brass rod (in meters)
    - timestamp: Time points
    - save_path: Path to save PNG
    """
    print(f"Creating temperature profile heatmap...")
    
    # Extract middle 90% of dataset
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)
    end_idx = int(0.95 * n_total)
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Evaluate solution at experimental time points
    T_profiles = []  # List of temperature profiles (in Celsius)
    for t in timestamp_plot:
        T_solution = sol.sol(t)
        T_brass = T_solution[2:]  # Extract brass temperatures
        T_brass_C = T_brass - 273.15  # Convert to Celsius
        T_profiles.append(T_brass_C)
    
    T_profiles = np.array(T_profiles)  # Shape: (n_time_points, N_nodes)
    
    # Convert position to mm
    x_mm = x_grid * 1000  # Convert to mm
    
    # Transpose so that rows = positions, columns = time
    # This way x-axis = time, y-axis = position
    T_profiles_transposed = T_profiles.T  # Shape: (N_nodes, n_time_points)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap: x-axis = time, y-axis = position along rod
    im = ax.imshow(T_profiles_transposed, aspect='auto', origin='lower',
                   extent=[timestamp_plot[0], timestamp_plot[-1], x_mm[0], x_mm[-1]],
                   cmap='hot', interpolation='bilinear')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position along Brass Rod (mm)', fontsize=12)
    ax.set_title('Temperature Profile Along Brass Rod vs Time', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Temperature (°C)')
    
    plt.tight_layout()
    
    # Save as PNG
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Temperature heatmap saved to: {save_path}")
    plt.close()


def create_all_heatmaps_combined(sol, x_grid, timestamp, save_path):
    """
    Create static PNG heatmap showing all three temperature profiles in one image with shared color scale:
    1. Temperature profile along brass rod
    2. Tc (Cold Plate) temperature
    3. Th (Hot Plate) temperature
    
    Parameters:
    - sol: Solution object from solve_ivp
    - x_grid: Spatial grid positions along brass rod (in meters)
    - timestamp: Time points
    - save_path: Path to save PNG
    """
    print(f"Creating combined heatmap with all three temperature profiles...")
    
    # Extract middle 90% of dataset
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)
    end_idx = int(0.95 * n_total)
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Prepare data for temperature profile along brass rod
    T_profiles = []  # List of temperature profiles (in Celsius)
    for t in timestamp_plot:
        T_solution = sol.sol(t)
        T_brass = T_solution[2:]  # Extract brass temperatures
        T_brass_C = T_brass - 273.15  # Convert to Celsius
        T_profiles.append(T_brass_C)
    
    T_profiles = np.array(T_profiles)  # Shape: (n_time_points, N_nodes)
    
    # Convert position to mm
    x_mm = x_grid * 1000  # Convert to mm
    
    # Transpose so that rows = positions, columns = time
    T_profiles_transposed = T_profiles.T  # Shape: (N_nodes, n_time_points)
    
    # Calculate Tc and Th temperatures at each time point
    Tc_values = []
    Th_values = []
    for t in timestamp_plot:
        T_solution = sol.sol(t)
        Tc = T_solution[0] - 273.15  # Cold plate temperature in Celsius
        Th = T_solution[1] - 273.15  # Hot plate temperature in Celsius
        Tc_values.append(Tc)
        Th_values.append(Th)
    
    Tc_values = np.array(Tc_values)
    Th_values = np.array(Th_values)
    
    # Find common temperature range for consistent color scale across all three plots
    all_temps = np.concatenate([T_profiles.flatten(), Tc_values, Th_values])
    vmin = np.min(all_temps)
    vmax = np.max(all_temps)
    
    # Calculate rod length in mm for determining relative heights
    rod_length_mm = x_mm[-1] - x_mm[0]
    # Plate heatmaps should be 5mm wide
    plate_width_mm = 5.0
    
    # Create figure with three subplots stacked vertically
    # Use gridspec to control relative heights: rod gets most space, plates get 5mm each
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[rod_length_mm, plate_width_mm, plate_width_mm], hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # 1. Temperature profile along brass rod (top subplot)
    im1 = ax1.imshow(T_profiles_transposed, aspect='auto', origin='lower',
                     extent=[timestamp_plot[0], timestamp_plot[-1], x_mm[0], x_mm[-1]],
                     cmap='coolwarm', interpolation='bilinear',
                     vmin=vmin, vmax=vmax)
    ax1.set_ylabel('Position along Brass Rod (mm)', fontsize=20)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.set_title('T(brass rod) vs Time', fontsize=18, fontweight='bold')
    
    # 2. Tc temperature (middle subplot) - 5mm wide
    # Use a single row to represent 5mm
    n_rows_tc = 1
    Tc_2d = np.tile(Tc_values, (n_rows_tc, 1))
    im2 = ax2.imshow(Tc_2d, aspect='auto', origin='lower',
                     extent=[timestamp_plot[0], timestamp_plot[-1], 0, plate_width_mm],
                     cmap='coolwarm', interpolation='bilinear',
                     vmin=vmin, vmax=vmax)
    ax2.set_ylabel('', fontsize=20)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', labelsize=16)
    ax2.set_title('Tc (cold plate) vs time', fontsize=18, fontweight='bold')
    
    # 3. Th temperature (bottom subplot) - 5mm wide
    # Use a single row to represent 5mm
    n_rows_th = 1
    Th_2d = np.tile(Th_values, (n_rows_th, 1))
    im3 = ax3.imshow(Th_2d, aspect='auto', origin='lower',
                     extent=[timestamp_plot[0], timestamp_plot[-1], 0, plate_width_mm],
                     cmap='coolwarm', interpolation='bilinear',
                     vmin=vmin, vmax=vmax)
    ax3.set_xlabel('Time (s)', fontsize=20)
    ax3.set_ylabel('', fontsize=20)
    ax3.set_yticks([])
    ax3.tick_params(axis='x', labelsize=16)
    ax3.set_title('Th (hot plate) vs time', fontsize=18, fontweight='bold')
    
    # Add shared colorbar for all three plots
    # Position it on the right side of the figure
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, label='Temperature (°C)')
    cbar.set_label('Temperature (°C)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for colorbar on the right
    
    # Save as PNG
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Combined heatmap saved to: {save_path}")
    plt.close()


def create_temperature_plate_heatmap(sol, timestamp, save_path, temp_type='Tc'):
    """
    Create static PNG heatmap showing plate temperature (Tc or Th) vs time.
    
    Parameters:
    - sol: Solution object from solve_ivp
    - timestamp: Time points
    - save_path: Path to save PNG
    - temp_type: 'Tc' for cold plate or 'Th' for hot plate
    """
    print(f"Creating {temp_type} temperature heatmap...")
    
    # Extract middle 90% of dataset
    n_total = len(timestamp)
    start_idx = int(0.05 * n_total)
    end_idx = int(0.95 * n_total)
    timestamp_plot = timestamp[start_idx:end_idx]
    
    # Calculate temperature at each time point
    temp_values = []
    for t in timestamp_plot:
        T_solution = sol.sol(t)
        if temp_type == 'Tc':
            temp = T_solution[0] - 273.15  # Cold plate temperature in Celsius
        else:  # Th
            temp = T_solution[1] - 273.15  # Hot plate temperature in Celsius
        
        temp_values.append(temp)
    
    temp_values = np.array(temp_values)
    
    # Create figure with narrow height to emphasize 1D nature
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Create heatmap - essentially a 1D line that varies with time
    # Use a thin horizontal strip to visualize as heatmap (small width = essentially 1D)
    # Repeat temperature values vertically (small number of rows) to create thin rectangle
    n_rows = 5  # Small number of rows to make it essentially 1D
    temp_2d = np.tile(temp_values, (n_rows, 1))  # Shape: (n_rows, n_time_points)
    
    # Use a temperature colormap (e.g., 'hot' or 'coolwarm')
    im = ax.imshow(temp_2d, aspect='auto', origin='lower',
                   extent=[timestamp_plot[0], timestamp_plot[-1], 0, n_rows],
                   cmap='coolwarm', interpolation='bilinear')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('', fontsize=12)
    ax.set_yticks([])  # Hide y-axis since it's just for visualization
    ax.set_title(f'{temp_type} Temperature vs Time', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label=f'{temp_type} Temperature (°C)', orientation='horizontal', pad=0.2)
    
    plt.tight_layout()
    
    # Save as PNG
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ {temp_type} temperature heatmap saved to: {save_path}")
    plt.close()


def main():
    """Main function to generate heatmap GIFs."""
    print("=" * 70)
    print("Heat Profile Heatmap GIF Generator")
    print("=" * 70)
    
    # Load data
    filepath = 'data/session6/brass_7V_10s.csv'
    timestamp, voltage, _, thermistor_temperatures = load_dataset(filepath)
    
    # Setup parameters
    params, L_brass, N_nodes = setup_parameters(verbose=False)
    
    # Create spatial grid
    x_grid = np.linspace(0, L_brass, N_nodes)
    
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
    
    # Create output directory
    output_dir = Path('plots/heatprofile')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate heatmaps
    print("\n" + "=" * 70)
    print("Generating Heatmap Plots")
    print("=" * 70)
    
    # Combined heatmap with all three temperature profiles (shared color scale)
    combined_heatmap_path = output_dir / 'all_temperature_heatmaps_combined.png'
    create_all_heatmaps_combined(sol, x_grid, timestamp, combined_heatmap_path)
    
    print("\n" + "=" * 70)
    print("All heatmap plots generated successfully!")
    print("=" * 70)
    print(f"  Combined heatmaps: {combined_heatmap_path}")


if __name__ == '__main__':
    main()

