import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(path):
	"""Load thermal dataset from CSV file."""
	data = pd.read_csv(path, header=3)
	timestamp = data.iloc[:, 0].to_numpy()
	thermistor_temperatures = data.iloc[:, 3:].to_numpy()
	
	return timestamp, thermistor_temperatures


def plot_thermistor_0(timestamp, thermistor_0):
	"""
	Plot thermistor 0 temperature vs time.
	
	Parameters:
	- timestamp: Time data (s)
	- thermistor_0: Thermistor 0 temperature data (°C)
	"""
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.plot(timestamp, thermistor_0, 'b-', linewidth=1.5, label='Thermistor 0')
	ax.set_xlabel('Time (s)', fontsize=12)
	ax.set_ylabel('Temperature (°C)', fontsize=12)
	ax.set_title('Thermistor 0 Temperature vs Time', fontsize=14, fontweight='bold')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best', fontsize=11)
	plt.tight_layout()
	plt.show()


def main():
	"""Main function to plot thermistor 0 data."""
	# Load data
	filepath = 'data/session6/brass_cooling.csv'
	timestamp, thermistor_temperatures = load_dataset(filepath)
	
	# Extract thermistor 0 data (first thermistor column)
	thermistor_0 = thermistor_temperatures[:, 0]
	
	# Plot thermistor 0 temperature vs time
	plot_thermistor_0(timestamp, thermistor_0)
	
	# Print summary statistics
	print(f"\nData Summary for {filepath}:")
	print(f"  Time range: {timestamp[0]:.2f} s to {timestamp[-1]:.2f} s")
	print(f"  Duration: {timestamp[-1] - timestamp[0]:.2f} s")
	print(f"  Number of data points: {len(timestamp)}")
	print(f"\nThermistor 0 Statistics:")
	print(f"  Temperature range: {thermistor_0.min():.2f} °C to {thermistor_0.max():.2f} °C")
	print(f"  Average temperature: {np.mean(thermistor_0):.2f} °C")
	print(f"  Initial temperature: {thermistor_0[0]:.2f} °C")
	print(f"  Final temperature: {thermistor_0[-1]:.2f} °C")


if __name__ == '__main__':
	main()

