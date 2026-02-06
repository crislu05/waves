"""
Generate optimization evolution GIF from saved data.
This can be run independently after optimization completes.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from pathlib import Path
from scipy.interpolate import interp1d

from optimse import objective_function, setup_optimization


def load_gif_data():
    """Load saved optimization data for GIF generation."""
    gif_data_file = Path('data/netflux/gif_data.json')
    if not gif_data_file.exists():
        raise FileNotFoundError(f"GIF data file not found: {gif_data_file}. Run optimse.py first.")
    
    with open(gif_data_file, 'r') as f:
        data = json.load(f)
    
    # Reconstruct visited_regions dict
    visited_regions = {}
    for key_str, count in data['visited_regions'].items():
        alpha_bin, K_therm_bin = map(int, key_str.split('_'))
        visited_regions[(alpha_bin, K_therm_bin)] = count
    
    data['visited_regions'] = visited_regions
    return data


def load_optimized_R_elec():
    """Load optimized R_elec value from optimized_parameters.json."""
    params_file = Path('data/netflux/optimized_parameters.json')
    if not params_file.exists():
        raise FileNotFoundError(f"Optimized parameters file not found: {params_file}")
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    return params['R_elec']


def create_heatmap_evolution_gif(n_frames=50, n_grid=40):
    """
    Create an animated GIF showing heat map evolution of optimization process.
    
    Parameters:
    - n_frames: Number of frames in animation
    - n_grid: Grid resolution for objective function landscape
    """
    print("Loading optimization data...")
    gif_data = load_gif_data()
    
    # Get optimized R_elec value
    try:
        R_elec_fixed = load_optimized_R_elec()
        print(f"Using optimized R_elec = {R_elec_fixed:.4f}")
    except FileNotFoundError:
        R_elec_fixed = gif_data.get('optimized_R_elec', 3.0)
        print(f"Using R_elec from GIF data = {R_elec_fixed:.4f}")
    
    trajectory = gif_data['trajectory']
    visited_regions = gif_data['visited_regions']
    bounds = gif_data['bounds']
    param_names = gif_data['param_names']
    
    # Reconstruct fixed_params and experimental_data
    _, _, fixed_params, experimental_data = setup_optimization()
    
    # Get bounds for alpha and K_therm
    alpha_bounds = bounds[0]
    K_therm_bounds = bounds[1]
    
    # Create parameter grids for heat map
    print(f"Computing objective function landscape (grid: {n_grid}x{n_grid})...")
    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], n_grid)
    K_therm_grid = np.linspace(K_therm_bounds[0], K_therm_bounds[1], n_grid)
    alpha_mesh, K_therm_mesh = np.meshgrid(alpha_grid, K_therm_grid)
    
    # Calculate objective function over grid (fix R_elec at optimized value)
    fx_mesh = np.zeros_like(alpha_mesh)
    for i in range(n_grid):
        for j in range(n_grid):
            params_vector = np.array([alpha_mesh[i, j], K_therm_mesh[i, j], R_elec_fixed])
            fx_mesh[i, j] = objective_function(params_vector, fixed_params, experimental_data)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {(i+1)/n_grid*100:.1f}%")
    
    # Prepare trajectory data
    trajectory_array = np.array(trajectory)
    n_iterations = len(trajectory)
    frame_indices = np.linspace(0, n_iterations - 1, n_frames, dtype=int)
    
    # Handle solver failures and prepare for visualization
    # Many parameter combinations return 1e10 (solver failures), which makes the surface appear flat
    valid_mask = fx_mesh < 1e9
    n_failures = np.sum(~valid_mask)
    
    print("Creating animation...")
    if len(fx_mesh[valid_mask]) > 0:
        valid_fx = fx_mesh[valid_mask]
        print(f"  Valid objective function range: {np.min(valid_fx):.6f} to {np.max(valid_fx):.6f}")
        print(f"  Solver failures (fx >= 1e9): {n_failures} / {fx_mesh.size} ({100*n_failures/fx_mesh.size:.1f}%)")
        print(f"  Using logarithmic scale for visualization")
        
        # Only replace actual solver failures with NaN (keep all valid values)
        fx_mesh_plot = fx_mesh.copy()
        fx_mesh_plot[~valid_mask] = np.nan
    else:
        print("  Warning: No valid objective function values found!")
        fx_mesh_plot = fx_mesh
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Create 2D heatmap of objective function landscape
    # Use pcolormesh for better control over the grid
    # Note: pcolormesh expects the mesh to be one cell larger, so we need to handle boundaries
    # For imshow, we need to specify extent
    extent = [alpha_bounds[0], alpha_bounds[1], K_therm_bounds[0], K_therm_bounds[1]]
    
    # Create heatmap using imshow with log scale
    # Get valid values for log scale limits
    plot_valid = fx_mesh_plot[~np.isnan(fx_mesh_plot)]
    if len(plot_valid) > 0:
        vmin = np.min(plot_valid)
        vmax = np.max(plot_valid)
        # Use LogNorm for logarithmic color scale
        im = ax.imshow(fx_mesh_plot, extent=extent, origin='lower', 
                       cmap='viridis', aspect='auto', interpolation='bilinear',
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        # Fallback if no valid values
        im = ax.imshow(fx_mesh_plot, extent=extent, origin='lower', 
                       cmap='viridis', aspect='auto', interpolation='bilinear')
    
    ax.set_xlabel('$\\alpha$', fontsize=20)
    ax.set_ylabel('$k$', fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    # Remove title
    ax.set_title('', fontsize=0)
    
    # Colorbar (log scale)
    cbar = plt.colorbar(im, ax=ax, label='Objective Function (fx, log scale)')
    cbar.set_label('Objective Function (fx, log scale)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Initialize trajectory line and best point (2D only - alpha and K_therm)
    traj_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    best_point, = ax.plot([], [], 'r*', markersize=15, label='Current Best')
    
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
            # Update trajectory line (2D: alpha vs K_therm)
            traj_line.set_data(traj_up_to_now[:, 0], traj_up_to_now[:, 1])
            
            # Update best point (2D: alpha vs K_therm)
            best_point.set_data([traj_up_to_now[-1, 0]], [traj_up_to_now[-1, 1]])
            
            # Update text (fx is still in trajectory_array[:, 2])
            iter_text.set_text(f'Iteration: {idx}\n'
                             f'fx: {traj_up_to_now[-1, 2]:.4f}\n'
                             f'α: {traj_up_to_now[-1, 0]:.4f}\n'
                             f'K: {traj_up_to_now[-1, 1]:.4f}')
        
        return traj_line, best_point, iter_text
    
    # Create animation
    print(f"Generating {n_frames} frames...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=False, repeat=True)
    
    # Save as GIF
    gif_path = Path('plots/optimization/heatmap_evolution.gif')
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GIF to {gif_path}...")
    anim.save(gif_path, writer='pillow', fps=10)
    print(f"✓ GIF saved successfully!")
    
    plt.close()


if __name__ == '__main__':
    print("=" * 70)
    print("Optimization Evolution GIF Generator")
    print("=" * 70)
    create_heatmap_evolution_gif(n_frames=50, n_grid=40)
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

