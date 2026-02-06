import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from scipy.optimize import curve_fit


def load_dataset(path):
	"""Load thermal dataset from CSV file."""
	data = pd.read_csv(path, header=3)
	timestamp = pd.to_numeric(data.iloc[:, 0], errors='coerce').to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()
	
	# Remove any NaN values that might have been introduced
	valid_mask = ~np.isnan(timestamp)
	timestamp = timestamp[valid_mask]
	thermistor_temperatures = thermistor_temperatures[valid_mask, :]
	
	return timestamp, thermistor_temperatures


def extract_voltage(label):
	"""Extract voltage number from label for sorting. Returns 0 for non-voltage labels."""
	match = re.search(r'(\d+)V', label)
	if match:
		return int(match.group(1))
	return 0  # For 'brass' or other non-voltage labels


def exponential_decay_model(t, T_inf, T0, tau):
	"""
	Exponential decay model for cooling: T(t) = T_inf + (T0 - T_inf) * exp(-t/τ)
	
	Parameters:
	- t: time (s)
	- T_inf: ambient/final temperature (°C)
	- T0: initial temperature (°C)
	- tau: time constant (s)
	"""
	return T_inf + (T0 - T_inf) * np.exp(-t / tau)


def fit_exponential_decay(t, T):
	"""
	Fit exponential decay curve to cooling data.
	
	Returns:
	- T_inf: ambient/final temperature (°C)
	- T0: initial temperature (°C)
	- tau: time constant (s)
	- tau_uncertainty: uncertainty in tau (s)
	"""
	# Initial guesses
	T_inf0 = np.mean(T[-len(T)//10:])  # Use last 10% for T_inf estimate
	T0_guess = T[0]
	tau0 = (t[-1] - t[0]) / 3.0  # Rough estimate
	
	try:
		popt, pcov = curve_fit(exponential_decay_model, t, T, 
		                      p0=[T_inf0, T0_guess, tau0],
		                      maxfev=10000)
		T_inf, T0, tau = popt
		tau_uncertainty = np.sqrt(pcov[2, 2]) if np.isfinite(pcov[2, 2]) else np.nan
		return T_inf, T0, tau, tau_uncertainty
	except Exception as e:
		print(f"  Fit failed: {e}")
		return np.nan, np.nan, np.nan, np.nan


def calculate_h_from_tau(tau, rho, c, V, A):
	"""
	Calculate convective heat transfer coefficient from time constant.
	
	Using: τ = (ρcV) / (hA)  =>  h = (ρcV) / (τA)
	
	Parameters:
	- tau: time constant (s)
	- rho: density (kg/m³)
	- c: specific heat capacity (J/(kg·K))
	- V: volume (m³)
	- A: surface area (m²)
	
	Returns:
	- h: convective heat transfer coefficient (W/(m²·K))
	"""
	if np.isnan(tau) or tau <= 0:
		return np.nan
	return (rho * c * V) / (tau * A)


def plot_thermistor_0_multiple(datasets, save_path=None):
	"""
	Plot thermistor 0 temperature vs time for multiple datasets with exponential fits.
	
	Parameters:
	- datasets: List of tuples (timestamp, thermistor_0, label) for each dataset
	- save_path: Path to save the plot (if None, plot is displayed)
	"""
	# Brass rod parameters from heatpump.py
	rho_brass = 8520.0  # kg/m³
	c_brass = 380.0  # J/(kg·K)
	radius_brass = 0.015  # m (1.5 cm)
	L_brass = 0.041  # m (length of brass cylinder)
	
	# Calculate volume and surface area for convection
	# Note: One end of the brass rod is in contact with the Peltier plate (covered in grease),
	# so it does NOT contribute to convective heat transfer. Only lateral area + one end-cap.
	V_brass = np.pi * radius_brass**2 * L_brass  # m³
	A_convection = (2 * np.pi * radius_brass * L_brass) + (np.pi * radius_brass**2)  # m² (lateral + one end-cap)
	
	fig, ax = plt.subplots(figsize=(12, 8))
	
	# Sort datasets by voltage in descending order (12V, 10V, 8V, 6V, 4V, then others like brass)
	datasets_sorted = sorted(datasets, key=lambda x: extract_voltage(x[2]), reverse=True)
	
	# Define colors for different curves
	colors = plt.cm.tab10(np.linspace(0, 1, len(datasets_sorted)))
	
	# Store fit results
	fit_results = []
	
	for i, (timestamp, thermistor_0, label) in enumerate(datasets_sorted):
		# Fit exponential decay to ALL available data (no time filtering)
		if len(timestamp) > 0:
			# Fit exponential decay using all data
			T_inf, T0, tau, tau_unc = fit_exponential_decay(timestamp, thermistor_0)
			
			# Calculate h from tau
			h = calculate_h_from_tau(tau, rho_brass, c_brass, V_brass, A_convection)
			
			# Store results
			fit_results.append({
				'label': label,
				'tau': tau,
				'tau_uncertainty': tau_unc,
				'h': h,
				'T_inf': T_inf,
				'T0': T0
			})
			
			# Plot data only between 50 and 300 seconds (for visualization)
			mask_plot = (timestamp >= 50) & (timestamp <= 300)
			timestamp_plot = timestamp[mask_plot]
			thermistor_0_plot = thermistor_0[mask_plot]
			
			if len(timestamp_plot) > 0:
				# Plot data
				ax.plot(timestamp_plot, thermistor_0_plot, '-', linewidth=1.5, 
				        label=label, color=colors[i], alpha=0.8)
				
				# Plot fitted curve if fit was successful (only in 50-300s range)
				if not np.isnan(tau):
					t_fit = np.linspace(50, 300, 200)
					T_fit = exponential_decay_model(t_fit, T_inf, T0, tau)
					ax.plot(t_fit, T_fit, '--', linewidth=1.5, color=colors[i], alpha=0.6)
	
	ax.set_xlabel('Time (s)', fontsize=20)
	ax.set_ylabel('Temperature (°C)', fontsize=20)
	ax.tick_params(axis='x', labelsize=16)
	ax.tick_params(axis='y', labelsize=16)
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', fontsize=14)
	ax.set_xlim(50, 300)  # Set x-axis limits to 50-300 seconds
	plt.tight_layout()
	
	if save_path:
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Plot saved to: {save_path}")
		plt.close()
	else:
		plt.show()
	
	return fit_results


def plot_single_fit_example(timestamp, thermistor_0, label, T_inf, T0, tau, tau_unc, h, save_path=None):
	"""
	Plot a detailed fitting result for one example curve.
	
	Parameters:
	- timestamp: Time data (all available data, fit was performed on this)
	- thermistor_0: Temperature data (all available data, fit was performed on this)
	- label: Label for the dataset
	- T_inf: Fitted ambient temperature (from fit on all data)
	- T0: Fitted initial temperature (from fit on all data)
	- tau: Fitted time constant (from fit on all data)
	- tau_unc: Uncertainty in tau
	- h: Calculated convective heat transfer coefficient
	- save_path: Path to save the plot (if None, plot is displayed)
	"""
	# Filter data for plotting (50-300s range only)
	mask_plot = (timestamp >= 50) & (timestamp <= 300)
	timestamp_plot = timestamp[mask_plot]
	thermistor_0_plot = thermistor_0[mask_plot]
	
	if len(timestamp_plot) == 0:
		print(f"No data in range 50-300s for {label}. Cannot create example plot.")
		return
	
	# Create figure with two subplots: main fit and residuals
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
	
	# Plot 1: Main fit (only showing 50-300s range)
	ax1.plot(timestamp_plot, thermistor_0_plot, 'bo', markersize=4, 
	         label='Experimental data', alpha=0.6)
	
	# Plot fitted curve (only in 50-300s range for visualization)
	t_fit = np.linspace(50, 300, 200)
	T_fit = exponential_decay_model(t_fit, T_inf, T0, tau)
	ax1.plot(t_fit, T_fit, 'r-', linewidth=2, label='Exponential fit', alpha=0.8)
	
	# Add fit parameters as text box
	fit_text = f'Fit Parameters:\n'
	fit_text += f'T_inf = {T_inf:.2f} °C\n'
	fit_text += f'T₀ = {T0:.2f} °C\n'
	fit_text += f'τ = {tau:.2f} ± {tau_unc:.2f} s\n'
	fit_text += f'h = {h:.2f} W/(m²·K)'
	
	ax1.text(0.02, 0.98, fit_text, transform=ax1.transAxes,
	         fontsize=14, verticalalignment='top',
	         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
	
	ax1.set_xlabel('Time (s)', fontsize=20)
	ax1.set_ylabel('Temperature (°C)', fontsize=20)
	ax1.tick_params(axis='x', labelsize=16)
	ax1.tick_params(axis='y', labelsize=16)
	ax1.set_title(f'Exponential Decay Fit: {label} (Fit on all data, plot shows 50-300s)', fontsize=18, fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc='best', fontsize=14)
	ax1.set_xlim(50, 300)
	
	# Plot 2: Residuals (only showing 50-300s range)
	residuals_plot = thermistor_0_plot - exponential_decay_model(timestamp_plot, T_inf, T0, tau)
	ax2.plot(timestamp_plot, residuals_plot, 'ko', markersize=3, alpha=0.6)
	ax2.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
	ax2.set_xlabel('Time (s)', fontsize=20)
	ax2.set_ylabel('Residual (°C)', fontsize=20)
	ax2.tick_params(axis='x', labelsize=16)
	ax2.tick_params(axis='y', labelsize=16)
	ax2.set_title('Residuals (Experimental - Fit) in 50-300s range', fontsize=18, fontweight='bold')
	ax2.grid(True, alpha=0.3)
	ax2.set_xlim(50, 300)
	
	# Add residual statistics (calculated from plotted range)
	mean_residual = np.mean(residuals_plot)
	std_residual = np.std(residuals_plot)
	residual_text = f'Mean: {mean_residual:.4f} °C\nStd: {std_residual:.4f} °C'
	ax2.text(0.02, 0.98, residual_text, transform=ax2.transAxes,
	         fontsize=14, verticalalignment='top',
	         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
	
	plt.tight_layout()
	
	if save_path:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Example fit plot saved to: {save_path}")
		plt.close()
	else:
		plt.show()


def generate_h_approx_vs_voltage_table(fit_results, save_path=None):
	"""
	Generate a table image showing approximated h (convective heat transfer coefficient) vs voltage.
	Approximation: h_approx = 11.2 + (h_voltage - h_fan_off), where fan off h is assumed to be 11.2.
	
	Parameters:
	- fit_results: List of dictionaries with fit results (from plot_thermistor_0_multiple)
	- save_path: Path to save the table image
	"""
	import matplotlib
	matplotlib.use('Agg')
	
	# Format values with uncertainties (1 decimal place)
	def format_with_uncertainty(value, uncertainty, decimals=1):
		"""Format value ± uncertainty."""
		if np.isnan(value) or np.isnan(uncertainty):
			return f'{value:.{decimals}f}'
		return f'{value:.{decimals}f} ± {uncertainty:.{decimals}f}'
	
	# Find fan off h value
	h_fan_off = None
	h_fan_off_unc = None
	for result in fit_results:
		if 'fan off' in result['label'].lower():
			h_fan_off = result['h']
			tau = result['tau']
			tau_unc = result['tau_uncertainty']
			# Calculate uncertainty in h using error propagation
			if not (np.isnan(tau) or np.isnan(tau_unc) or tau == 0):
				h_fan_off_unc = h_fan_off * (tau_unc / tau)
			else:
				h_fan_off_unc = np.nan
			break
	
	if h_fan_off is None or np.isnan(h_fan_off):
		print("Warning: Could not find fan off h value. Using measured values directly.")
		h_fan_off = 0
		h_fan_off_unc = 0
	
	# Extract voltage and calculate approximated h for each result
	table_data = []
	for result in fit_results:
		label = result['label']
		voltage = extract_voltage(label)
		h_measured = result['h']
		tau = result['tau']
		tau_unc = result['tau_uncertainty']
		
		# Calculate uncertainty in measured h using error propagation
		# h = (ρcV) / (τA), so Δh/h = Δτ/τ
		if not (np.isnan(tau) or np.isnan(tau_unc) or tau == 0):
			h_measured_unc = h_measured * (tau_unc / tau)
		else:
			h_measured_unc = np.nan
		
		# Skip if h is invalid
		if not np.isnan(h_measured) and h_measured > 0:
			# Calculate approximated h: h_approx = 11.2 + (h_voltage - h_fan_off)
			h_approx = 11.2 + (h_measured - h_fan_off)
			
			# Uncertainty propagation: h_approx = 11.2 + (h_voltage - h_fan_off)
			# Since 11.2 is a constant, uncertainty comes from the difference
			# Δh_approx = sqrt(Δh_voltage^2 + Δh_fan_off^2)
			if not (np.isnan(h_measured_unc) or np.isnan(h_fan_off_unc)):
				h_approx_unc = np.sqrt(h_measured_unc**2 + h_fan_off_unc**2)
			else:
				h_approx_unc = np.nan
			
			table_data.append({
				'voltage': voltage,
				'label': label,
				'h_before_correction': h_measured,
				'h_before_correction_unc': h_measured_unc,
				'h_after_correction': h_approx,
				'h_after_correction_unc': h_approx_unc
			})
	
	# Sort by voltage (descending)
	table_data.sort(key=lambda x: x['voltage'], reverse=True)
	
	# Prepare table data
	data = [['Voltage (V)', r'$\mathbf{h_{raw}}$ (W/(m²·K))', r'$\mathbf{h_{convective}}$ (W/(m²·K))']]
	
	for item in table_data:
		voltage_str = f"{item['voltage']}" if item['voltage'] > 0 else item['label']
		h_before_str = format_with_uncertainty(item['h_before_correction'], item['h_before_correction_unc'])
		h_after_str = format_with_uncertainty(item['h_after_correction'], item['h_after_correction_unc'])
		data.append([voltage_str, h_before_str, h_after_str])
	
	# Create figure
	fig, ax = plt.subplots(figsize=(6, max(2, len(data) * 0.3)))
	ax.axis('tight')
	ax.axis('off')
	
	# Create table
	table = ax.table(cellText=data[1:], colLabels=data[0], 
	                cellLoc='center', loc='center',
	                colWidths=[0.3, 0.35, 0.35])
	
	# Style the table
	table.auto_set_font_size(False)
	table.set_fontsize(12)
	table.scale(1, 2.0)
	
	# Header styling
	for i in range(3):
		table[(0, i)].set_facecolor('#4A90E2')
		table[(0, i)].set_text_props(weight='bold', color='white', size=9)
	
	# Save
	if save_path:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"  h vs voltage table saved to: {save_path}")
	else:
		plt.show()
	
	plt.close()


def main():
	"""Main function to plot thermistor 0 data from all cooling data files."""
	# Find all CSV files in data/cooling folder (exclude fit results CSV)
	cooling_dir = Path('data/cooling')
	all_csv_files = sorted(cooling_dir.glob('*.csv'))
	csv_files = [f for f in all_csv_files if f.name != 'cooling_fit_results.csv']
	
	if not csv_files:
		print(f"No CSV files found in {cooling_dir}")
		return
	
	print(f"Found {len(csv_files)} CSV files in {cooling_dir}")
	
	# Load all datasets
	datasets = []
	for filepath in csv_files:
		try:
			timestamp, thermistor_temperatures = load_dataset(filepath)
			thermistor_0 = thermistor_temperatures[:, 0]
			
			# Create label from filename (remove .csv and path)
			label = filepath.stem.replace('_cooling', '').replace('_', ' ')
			# Replace 'brass' with 'fan off'
			if 'brass' in label.lower():
				label = 'fan off'
			
			datasets.append((timestamp, thermistor_0, label))
			
			# Print summary statistics for each file
			print(f"\nData Summary for {filepath.name}:")
			print(f"  Time range: {timestamp[0]:.2f} s to {timestamp[-1]:.2f} s")
			print(f"  Duration: {timestamp[-1] - timestamp[0]:.2f} s")
			print(f"  Number of data points: {len(timestamp)}")
			print(f"  Temperature range: {thermistor_0.min():.2f} °C to {thermistor_0.max():.2f} °C")
			print(f"  Initial temperature: {thermistor_0[0]:.2f} °C")
			print(f"  Final temperature: {thermistor_0[-1]:.2f} °C")
		except Exception as e:
			print(f"Error loading {filepath}: {e}")
			continue
	
	if not datasets:
		print("No valid datasets loaded.")
		return
	
	# Plot all curves and get fit results
	plot_path = Path('plots/cooling/thermistor_0_temperature.png')
	fit_results = plot_thermistor_0_multiple(datasets, save_path=plot_path)
	
	print(f"\n✓ Successfully plotted {len(datasets)} cooling curves")
	
	# Print fit results
	print("\n=== Exponential Decay Fit Results ===")
	print(f"Brass rod parameters:")
	print(f"  Density (ρ): {8520.0:.0f} kg/m³")
	print(f"  Specific heat (c): {380.0:.0f} J/(kg·K)")
	print(f"  Volume (V): {np.pi * 0.015**2 * 0.041:.6e} m³")
	# Surface area for convection: lateral + one end-cap (other end is in contact with Peltier plate)
	A_convection_calc = (2 * np.pi * 0.015 * 0.041) + (np.pi * 0.015**2)
	print(f"  Surface area for convection (A): {A_convection_calc:.6f} m² (lateral + one end-cap)")
	print(f"\nUsing: τ = (ρcV) / (hA)  =>  h = (ρcV) / (τA)")
	print()
	
	for result in fit_results:
		label = result['label']
		tau = result['tau']
		tau_unc = result['tau_uncertainty']
		h = result['h']
		T_inf = result['T_inf']
		T0 = result['T0']
		
		if not np.isnan(tau):
			print(f"{label}:")
			print(f"  τ = {tau:.2f} ± {tau_unc:.2f} s")
			print(f"  h = {h:.2f} W/(m²·K)")
			print(f"  T_inf = {T_inf:.2f} °C")
			print(f"  T0 = {T0:.2f} °C")
			print()
		else:
			print(f"{label}: Fit failed")
			print()
	
	# Save results to CSV
	if fit_results:
		results_dir = Path('data/cooling')
		results_dir.mkdir(parents=True, exist_ok=True)
		results_path = results_dir / 'cooling_fit_results.csv'
		
		results_df = pd.DataFrame(fit_results)
		results_df.to_csv(results_path, index=False)
		print(f"Fit results saved to: {results_path}")
		
		# Find fan off h value
		h_fan_off = None
		h_fan_off_unc = None
		for result in fit_results:
			if 'fan off' in result['label'].lower():
				h_fan_off = result['h']
				tau = result['tau']
				tau_unc = result['tau_uncertainty']
				# Calculate uncertainty in h using error propagation
				if not (np.isnan(tau) or np.isnan(tau_unc) or tau == 0):
					h_fan_off_unc = h_fan_off * (tau_unc / tau)
				else:
					h_fan_off_unc = np.nan
				break
		
		if h_fan_off is None or np.isnan(h_fan_off):
			print("Warning: Could not find fan off h value. Using measured values directly.")
			h_fan_off = 0  # Will use measured values as-is
			h_fan_off_unc = 0
		
		# Save h vs voltage CSV with approximated h values
		# Approximation: h_approx = 11.2 + (h_voltage - h_fan_off)
		h_data = []
		for result in fit_results:
			label = result['label']
			voltage = extract_voltage(label)
			h_measured = result['h']
			tau = result['tau']
			tau_unc = result['tau_uncertainty']
			
			# Calculate uncertainty in measured h using error propagation
			# h = (ρcV) / (τA), so Δh/h = Δτ/τ
			if not (np.isnan(tau) or np.isnan(tau_unc) or tau == 0):
				h_measured_unc = h_measured * (tau_unc / tau)
			else:
				h_measured_unc = np.nan
			
			# Skip if h is invalid
			if not np.isnan(h_measured) and h_measured > 0:
				# Calculate approximated h: h_approx = 11.2 + (h_voltage - h_fan_off)
				h_approx = 11.2 + (h_measured - h_fan_off)
				
				# Uncertainty propagation: h_approx = 11.2 + (h_voltage - h_fan_off)
				# Since 11.2 is a constant, uncertainty comes from the difference
				# Δh_approx = sqrt(Δh_voltage^2 + Δh_fan_off^2)
				if not (np.isnan(h_measured_unc) or np.isnan(h_fan_off_unc)):
					h_approx_unc = np.sqrt(h_measured_unc**2 + h_fan_off_unc**2)
				else:
					h_approx_unc = np.nan
				
				h_data.append({
					'voltage': voltage,
					'label': label,
					'h_before_correction': h_measured,
					'h_before_correction_uncertainty': h_measured_unc,
					'h_after_correction': h_approx,  # Save approximated h value
					'h_after_correction_uncertainty': h_approx_unc
				})
		
		# Sort by voltage (descending)
		h_data.sort(key=lambda x: x['voltage'], reverse=True)
		
		if h_data:
			h_df = pd.DataFrame(h_data)
			h_csv_path = results_dir / 'h_vs_voltage.csv'
			h_df.to_csv(h_csv_path, index=False)
			print(f"h vs voltage data (approximated) saved to: {h_csv_path}")
	
	# Generate h vs voltage table with approximated values
	if fit_results:
		table_path = Path('plots/cooling/h_vs_voltage_table.png')
		generate_h_approx_vs_voltage_table(fit_results, save_path=table_path)
	
	# Generate example fit plot (use the first valid fit result, preferably 12V fan)
	example_result = None
	for result in fit_results:
		if not np.isnan(result['tau']) and result['tau'] > 0:
			# Prefer 12V fan if available, otherwise use first valid result
			if '12v' in result['label'].lower() or example_result is None:
				example_result = result
				if '12v' in result['label'].lower():
					break
	
	if example_result:
		# Find the corresponding dataset
		for timestamp, thermistor_0, label in datasets:
			if label == example_result['label']:
				example_plot_path = Path('plots/cooling/example_fit_result.png')
				plot_single_fit_example(
					timestamp, thermistor_0, label,
					example_result['T_inf'], example_result['T0'],
					example_result['tau'], example_result['tau_uncertainty'],
					example_result['h'],
					save_path=str(example_plot_path)
				)
				break


if __name__ == '__main__':
	main()
