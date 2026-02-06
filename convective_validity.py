import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_1samp

# Import functions from validity.py
from validity import calculate_rmse_and_correlation


def load_h_values_from_cooling():
    """
    Load h values from h_vs_voltage.csv.
    
    Returns:
    - Dictionary mapping fan voltage (int) to h value (float)
    """
    h_vs_voltage_file = Path('data/cooling/h_vs_voltage.csv')
    h_dict = {}
    
    if h_vs_voltage_file.exists():
        try:
            h_data = pd.read_csv(h_vs_voltage_file)
            for _, row in h_data.iterrows():
                voltage = int(row['voltage'])
                # Use h_after_correction column (corrected h values)
                h = float(row['h_after_correction'])
                
                # Include all voltages (including fan off = 0)
                h_dict[voltage] = h
                print(f"  Loaded h for {voltage}V fan: {h:.2f} W/(m²·K)")
        except Exception as e:
            print(f"Warning: Could not load h values from h_vs_voltage.csv: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: {h_vs_voltage_file} not found.")
        print(f"  Run cooling.py first to generate h values.")
    
    return h_dict


def find_csv_for_h_value(h_value):
    """
    Find CSV file for a given h value.
    CSV files are named like: temperature_comparison_h11p2.csv (where 11.2 -> 11p2)
    
    Parameters:
    - h_value: Convective heat transfer coefficient (W/(m²·K))
    
    Returns:
    - Path to CSV file, or None if not found
    """
    # Convert h value to filename format (replace . with p)
    h_str = f"{h_value:.1f}".replace('.', 'p')
    csv_path = Path(f'data/convective/temperature_comparison_h{h_str}.csv')
    
    if csv_path.exists():
        return csv_path
    else:
        return None


def calculate_metrics_from_csv(csv_path):
    """
    Calculate RMSE, MSE, and mean residual for all thermistors from CSV file.
    
    Parameters:
    - csv_path: Path to CSV file with temperature comparison data
    
    Returns:
    - Dictionary with positions, RMSE, MSE, and mean_residual arrays
    """
    if not csv_path.exists():
        print(f"  Warning: CSV file not found: {csv_path}")
        return None
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Get all model columns
    model_cols = [col for col in df.columns if col.startswith('T_model_')]
    model_cols = sorted(model_cols)
    
    positions = []
    rmse_values = []
    mse_values = []
    mean_residuals = []
    
    for model_col_name in model_cols:
        exp_col_name = model_col_name.replace('T_model_', 'T_exp_')
        
        # Extract thermistor ID and position from column name
        # e.g., "T_model_0_x3.0mm (°C)" -> therm_id=0, x_pos=3.0
        parts = model_col_name.split('_')
        therm_id = int(parts[2])
        # Extract position: "x3.0mm (°C)" -> "3.0"
        pos_str = parts[3].split()[0]  # Take first part before any space
        x_pos_mm = float(pos_str.replace('x', '').replace('mm', ''))
        
        T_model = df[model_col_name].values
        T_exp = df[exp_col_name].values
        
        # Calculate metrics
        rmse, _ = calculate_rmse_and_correlation(T_model, T_exp)
        
        # Calculate MSE
        valid_mask = np.isfinite(T_model) & np.isfinite(T_exp)
        if np.sum(valid_mask) > 0:
            T_model_valid = T_model[valid_mask]
            T_exp_valid = T_exp[valid_mask]
            mse = np.mean((T_model_valid - T_exp_valid)**2)
            residuals = T_model_valid - T_exp_valid
            # Use absolute value of mean residual
            mean_residual = np.abs(np.mean(residuals))
        else:
            mse = np.nan
            mean_residual = np.nan
        
        positions.append(x_pos_mm)
        rmse_values.append(rmse)
        mse_values.append(mse)
        mean_residuals.append(mean_residual)
    
    return {
        'positions': np.array(positions),
        'rmse': np.array(rmse_values),
        'mse': np.array(mse_values),
        'mean_residual': np.array(mean_residuals)
    }


def plot_3d_metrics_bar_chart(results_dict, save_path=None):
    """
    Plot 2D heatmaps of RMSE, MSE, and mean residual vs position and convective coefficient.
    Each metric is shown as a separate heatmap with color gradient indicating value.
    
    Parameters:
    - results_dict: Dictionary mapping h_value to metrics dict (from calculate_metrics_from_csv)
    - save_path: Path to save the plot
    """
    # Extract all h values and sort them
    h_values = sorted(results_dict.keys())
    
    # Get positions from first h value (should be same for all)
    first_h = h_values[0]
    positions = results_dict[first_h]['positions']
    
    # Create meshgrid for heatmap
    H_mesh, X_mesh = np.meshgrid(h_values, positions)
    
    # Prepare data arrays for each metric
    metrics = ['rmse', 'mse', 'mean_residual']
    metric_titles = ['RMSE', 'MSE', 'Residual']
    colorbar_units = ['°C', '°C²', '°C']
    
    # Create 2D arrays for each metric: shape (n_positions, n_h_values)
    metric_arrays = {}
    for metric in metrics:
        metric_array = np.zeros((len(positions), len(h_values)))
        for h_idx, h_val in enumerate(h_values):
            metrics_data = results_dict[h_val]
            metric_array[:, h_idx] = metrics_data[metric]
        metric_arrays[metric] = metric_array
    
    # Create figure with 3 subplots (one for each metric)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Find global min and max for each metric for consistent color scales
    vmin_dict = {}
    vmax_dict = {}
    for metric in metrics:
        data = metric_arrays[metric]
        valid_data = data[np.isfinite(data)]
        if len(valid_data) > 0:
            vmin_dict[metric] = np.nanmin(valid_data)
            vmax_dict[metric] = np.nanmax(valid_data)
        else:
            vmin_dict[metric] = 0
            vmax_dict[metric] = 1
    
    # Find global min and max across all metrics for shared colorbar
    all_data = np.concatenate([metric_arrays[m][np.isfinite(metric_arrays[m])] for m in metrics])
    vmin_global = np.nanmin(all_data)
    vmax_global = np.nanmax(all_data)
    
    # Plot each metric as a heatmap
    im_list = []
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]
        
        # Create heatmap with shared color scale
        im = ax.pcolormesh(H_mesh, X_mesh, metric_arrays[metric], 
                          cmap='viridis', shading='auto',
                          vmin=vmin_global, vmax=vmax_global)
        im_list.append(im)
        
        # Set labels - only on first graph (RMSE)
        if idx == 0:
            ax.set_xlabel('Convective Coefficient $h$ (W/(m²·K))', fontsize=20)
            ax.set_ylabel('Position along rod (mm)', fontsize=20)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
        else:
            # Remove axis labels for other graphs
            ax.set_xlabel('')
            ax.set_ylabel('')
            # Remove y-axis ticks for non-first graphs
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=16)
        
        ax.set_title(title, fontsize=18, fontweight='bold')
        
        # Print statistics for residual
        if metric == 'mean_residual':
            data = metric_arrays[metric]
            valid_data = data[np.isfinite(data)]
            if len(valid_data) > 0:
                print(f"\nResidual statistics:")
                print(f"  Min: {np.nanmin(valid_data):.6f} °C")
                print(f"  Max: {np.nanmax(valid_data):.6f} °C")
                print(f"  Mean: {np.nanmean(valid_data):.6f} °C")
                print(f"  Median: {np.nanmedian(valid_data):.6f} °C")
                print(f"  Std: {np.nanstd(valid_data):.6f} °C")
    
    # Add single shared colorbar on the right
    plt.tight_layout(rect=[0, 0, 0.88, 1])  # Leave space for colorbar
    cbar = fig.colorbar(im_list[0], ax=axes, shrink=0.8, aspect=20, pad=0.05)
    cbar.set_label('°C or °C²', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    # Save plot
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n2D heatmap metrics chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_threshold_testing_table(results_dict, save_path=None, u_measurement=0.1, alpha=0.05):
    """
    Generate a validation table similar to validity.py, aggregating metrics across all h values and positions.
    
    Parameters:
    - results_dict: Dictionary mapping h_value to metrics dict (from calculate_metrics_from_csv)
    - save_path: Path to save the table image
    - u_measurement: Measurement uncertainty in °C (default: 0.1°C for thermistors)
    - alpha: Significance level for statistical tests (default: 0.05)
    """
    # Collect all residuals and model/experimental data from all CSV files
    all_residuals = []
    all_T_model = []
    all_T_exp = []
    
    for h_value in sorted(results_dict.keys()):
        csv_path = find_csv_for_h_value(h_value)
        if csv_path is None or not csv_path.exists():
            continue
        
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
                # Take absolute value of all residuals
                residuals = np.abs(residuals)
                
                all_residuals.extend(residuals.tolist())
                all_T_model.extend(T_model_valid.tolist())
                all_T_exp.extend(T_exp_valid.tolist())
    
    if len(all_residuals) == 0:
        print("Error: No valid data found for threshold testing!")
        return
    
    all_residuals = np.array(all_residuals)
    all_T_model = np.array(all_T_model)
    all_T_exp = np.array(all_T_exp)
    n_points = len(all_residuals)
    
    # Calculate overall metrics
    # RMSE
    rmse = np.sqrt(np.mean(all_residuals**2))
    # Calculate uncertainty: standard error of RMSE (approximate)
    # RMSE uncertainty ≈ std(residuals^2) / (2 * RMSE * sqrt(n))
    rmse_uncertainty = np.std(all_residuals**2) / (2 * rmse * np.sqrt(n_points)) if rmse > 0 else 0.0
    
    # MSE
    mse = np.mean(all_residuals**2)
    # MSE uncertainty: standard error of MSE
    mse_uncertainty = np.std(all_residuals**2) / np.sqrt(n_points)
    
    # Mean Residual (already absolute values)
    mean_residual = np.mean(all_residuals)
    std_residual = np.std(all_residuals, ddof=1)
    mean_residual_uncertainty = std_residual / np.sqrt(n_points)
    
    # Pearson correlation
    if len(all_T_model) > 1 and np.std(all_T_model) > 0 and np.std(all_T_exp) > 0:
        from scipy.stats import pearsonr
        pearson_r, _ = pearsonr(all_T_model, all_T_exp)
        # Uncertainty: approximate standard error of Pearson r
        # SE(r) ≈ (1 - r^2) / sqrt(n - 2)
        pearson_r_uncertainty = (1 - pearson_r**2) / np.sqrt(n_points - 2) if n_points > 2 else 0.0
    else:
        pearson_r = np.nan
        pearson_r_uncertainty = np.nan
    
    # T-test: Test if mean residual is significantly different from zero
    # Since we're using absolute values, test against 0 (mean of absolute residuals should be > 0)
    if n_points >= 3 and std_residual > 0:
        t_stat, t_pvalue = ttest_1samp(all_residuals, 0.0)
    else:
        t_stat = np.nan
        t_pvalue = np.nan
    
    # Thresholds
    rmse_threshold = 2.0 * u_measurement
    mse_threshold = (2.0 * u_measurement)**2
    mean_residual_threshold = 1.5 * u_measurement
    pearson_r_threshold = 0.900
    
    # Validity checks
    rmse_valid = np.round(rmse, 3) <= np.round(rmse_threshold, 3)
    mse_valid = np.round(mse, 3) <= np.round(mse_threshold, 3)
    # Mean residual is already absolute, so no need for abs() here
    mean_residual_valid = np.round(mean_residual, 3) <= np.round(mean_residual_threshold, 3)
    t_test_valid = t_pvalue >= 0.05 if not np.isnan(t_pvalue) else False
    pearson_r_valid = pearson_r >= pearson_r_threshold if not np.isnan(pearson_r) else False
    
    # Format values with uncertainties
    def format_with_uncertainty(value, uncertainty, decimals=3):
        """Format value ± uncertainty."""
        if np.isnan(value) or np.isnan(uncertainty):
            return f'{value:.3f}'
        return f'{value:.3f} ± {uncertainty:.3f}'
    
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
        print(f"\nThreshold testing table saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main function to generate 3D bar chart analysis."""
    # Load h values from cooling data
    print("Loading h values from cooling data...")
    h_dict = load_h_values_from_cooling()
    
    if not h_dict:
        print("Error: Could not load h values. Please run cooling.py first.")
        return
    
    print(f"\nFound {len(h_dict)} h values")
    
    # Calculate metrics for each h value from CSV files
    results_dict = {}
    
    for voltage, h_value in sorted(h_dict.items()):
        # Skip fan off (voltage = 0) if you only want fan voltages
        # Uncomment the next line if you want to skip fan off:
        # if voltage == 0:
        #     continue
        
        print(f"\nProcessing h={h_value:.2f} W/(m²·K) (voltage={voltage}V)...")
        
        csv_path = find_csv_for_h_value(h_value)
        if csv_path is None:
            print(f"  Warning: CSV file not found for h={h_value:.2f}. Run convective_plot.py first to generate data.")
            continue
        
        metrics = calculate_metrics_from_csv(csv_path)
        if metrics is not None:
            results_dict[h_value] = metrics
            print(f"  Calculated metrics for {len(metrics['positions'])} thermistors")
        else:
            print(f"  Failed to calculate metrics for h={h_value:.2f}")
    
    if not results_dict:
        print("Error: No valid results calculated!")
        print("  Please run convective_plot.py first to generate comparison CSV files.")
        return
    
    print(f"\nSuccessfully calculated metrics for {len(results_dict)} h values")
    
    # Create 3D bar chart
    print("\nGenerating 3D bar chart...")
    save_path = Path('plots/validity/metrics_3d_bar_chart.png')
    plot_3d_metrics_bar_chart(results_dict, save_path=save_path)
    
    # Generate threshold testing table
    print("\nGenerating threshold testing table...")
    table_save_path = Path('plots/validity/threshold_testing_table.png')
    plot_threshold_testing_table(results_dict, save_path=table_save_path, 
                                u_measurement=0.1, alpha=0.05)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
