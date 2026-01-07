import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path


def load_dataset(path):
	"""Load thermal dataset from CSV file."""
	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	output_voltage = data.iloc[:, 1].to_numpy()
	output_current = data.iloc[:, 2].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()

	with open(path, 'r') as f:
		file_content = f.read()
	comments_match = re.search(r"Comments: (.*)$", file_content, re.MULTILINE)
	comments = comments_match[1] if comments_match else "No comments"

	return timestamp, output_voltage, output_current, thermistor_temperatures, comments


def plot_dataset(timestamp, output_voltage, output_current, thermistor_temperatures, 
                 comments, filename, save_path):
	"""Create comprehensive plots for thermal dataset."""
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle(f'{filename}\n{comments}', fontsize=12, wrap=True)
	
	# Plot 1: Voltage vs Time
	axes[0, 0].plot(timestamp, output_voltage, 'b-', linewidth=1.5)
	axes[0, 0].set_xlabel('Time (s)')
	axes[0, 0].set_ylabel('Voltage (V)')
	axes[0, 0].set_title('Output Voltage')
	axes[0, 0].grid(True, alpha=0.3)
	
	# Plot 2: Current vs Time
	axes[0, 1].plot(timestamp, output_current, 'r-', linewidth=1.5)
	axes[0, 1].set_xlabel('Time (s)')
	axes[0, 1].set_ylabel('Current (A)')
	axes[0, 1].set_title('Output Current')
	axes[0, 1].grid(True, alpha=0.3)
	
	# Plot 3: All Thermistor Temperatures vs Time
	num_thermistors = thermistor_temperatures.shape[1]
	colors = plt.cm.tab10(np.linspace(0, 1, num_thermistors))
	for i in range(num_thermistors):
		axes[1, 0].plot(timestamp, thermistor_temperatures[:, i], 
		                color=colors[i], label=f'Thermistor {i}', linewidth=1.5)
	axes[1, 0].set_xlabel('Time (s)')
	axes[1, 0].set_ylabel('Temperature (°C)')
	axes[1, 0].set_title('Thermistor Temperatures')
	axes[1, 0].legend(loc='best', fontsize=8)
	axes[1, 0].grid(True, alpha=0.3)
	
	# Plot 4: Power vs Time
	power = output_voltage * output_current
	axes[1, 1].plot(timestamp, power, 'g-', linewidth=1.5)
	axes[1, 1].set_xlabel('Time (s)')
	axes[1, 1].set_ylabel('Power (W)')
	axes[1, 1].set_title('Power (V × I)')
	axes[1, 1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	
	# Save plot
	plot_filename = Path(filename).stem + '.png'
	plot_path = os.path.join(save_path, plot_filename)
	plt.savefig(plot_path, dpi=300, bbox_inches='tight')
	plt.close()


def plot_all_session1_data():
	"""Load and plot all datasets from session1."""
	session1_dir = Path('data/session1')
	plots_dir = Path('plots/session1')
	
	# Create plots directory if it doesn't exist
	plots_dir.mkdir(parents=True, exist_ok=True)
	
	# Get all CSV files in session1
	csv_files = sorted(session1_dir.glob('*.csv'))
	
	if not csv_files:
		print(f"No CSV files found in {session1_dir}")
		return
	
	print(f"Found {len(csv_files)} CSV files. Processing...")
	
	for csv_file in csv_files:
		try:
			print(f"Processing {csv_file.name}...")
			timestamp, output_voltage, output_current, thermistor_temperatures, comments = (
				load_dataset(str(csv_file))
			)
			plot_dataset(timestamp, output_voltage, output_current, 
			            thermistor_temperatures, comments, csv_file.name, str(plots_dir))
			print(f"  ✓ Saved plot for {csv_file.name}")
		except Exception as e:
			print(f"  ✗ Error processing {csv_file.name}: {e}")
	
	print(f"\nAll plots saved to {plots_dir}")


if __name__ == '__main__':
	plot_all_session1_data()