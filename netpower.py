import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Load power and time data from CSV
csv_path = 'data/session6/brass_7V_10s.csv'
data = pd.read_csv(csv_path, header=3)

# Extract time, voltage, and current data
time = data.iloc[:, 0].to_numpy()  # timestamp/s
voltage = data.iloc[:, 1].to_numpy()  # voltage/V
current = data.iloc[:, 2].to_numpy()  # current/A

# Calculate power as voltage × current
power_raw = voltage * current

# Adjust power so that its mean is zero
power_mean = np.mean(power_raw)
power = power_raw - power_mean

print(f"Loaded {len(time)} data points")
print(f"Time range: {time[0]:.2f} s to {time[-1]:.2f} s")
print(f"Original power range: {power_raw.min():.4f} W to {power_raw.max():.4f} W")
print(f"Power mean: {power_mean:.4f} W")
print(f"Adjusted power range: {power.min():.4f} W to {power.max():.4f} W (mean-adjusted to zero)")

# Detect the period of the oscillation using FFT
# Find the dominant frequency
dt = np.mean(np.diff(time))  # Average time step
N = len(time)
frequencies = np.fft.fftfreq(N, dt)
fft_values = np.fft.fft(power)

# Find the dominant frequency (excluding DC component)
positive_freq_idx = np.where(frequencies > 0)[0]
dominant_freq_idx = positive_freq_idx[np.argmax(np.abs(fft_values[positive_freq_idx]))]
dominant_frequency = frequencies[dominant_freq_idx]
period = 1.0 / dominant_frequency

print(f"\nDetected oscillation period: {period:.3f} s")
print(f"Dominant frequency: {dominant_frequency:.4f} Hz")

# Use only the middle 90% of the dataset
time_range = time[-1] - time[0]
time_margin = 0.05 * time_range  # 5% margin on each side
time_start = time[0] + time_margin
time_end = time[-1] - time_margin

print(f"\nUsing middle 90% of dataset:")
print(f"  Full range: {time[0]:.2f} s to {time[-1]:.2f} s")
print(f"  Analysis range: {time_start:.2f} s to {time_end:.2f} s")

# Find zero crossings in the entire dataset
zero_crossings_all = []
for i in range(len(power) - 1):
	if (power[i] >= 0 and power[i+1] < 0) or (power[i] < 0 and power[i+1] >= 0):
		# Linear interpolation to find exact zero crossing time
		t_zero = time[i] + (time[i+1] - time[i]) * (0 - power[i]) / (power[i+1] - power[i])
		zero_crossings_all.append(t_zero)

# Filter to only include zero crossings in the middle 90% range
zero_crossings = [t for t in zero_crossings_all if time_start <= t <= time_end]

print(f"  Zero crossings in analysis range: {len(zero_crossings)}")

# Calculate net power per 2 cycles
# Two cycles are defined by five zero crossings: crossing i, i+1, i+2, i+3, i+4
# This represents two full oscillations: zero -> peak -> zero -> opposite peak -> zero -> peak -> zero -> opposite peak -> zero
# Net power = integral of power over two complete cycles (positive area - negative area)
net_power_per_cycle = []
cycle_info = []  # Store cycle start/end times for shading

if len(zero_crossings) >= 5:
	# Use five consecutive zero crossings to define each 2-cycle period
	# n=1: crossings 0, 1, 2, 3, 4 (from crossing 0 to crossing 4) - 2 cycles
	# n=2: crossings 4, 5, 6, 7, 8 (from crossing 4 to crossing 8) - 2 cycles
	# n=3: crossings 8, 9, 10, 11, 12 (from crossing 8 to crossing 12) - 2 cycles
	# etc.
	# Each period spans from crossing i to crossing i+4 (five crossings total = 2 cycles)
	for i in range(0, len(zero_crossings) - 4, 4):  # Step by 4: i=0,4,8,12,...
		cycle_start = zero_crossings[i]
		cycle_end = zero_crossings[i + 4]  # Two cycles: from crossing i to crossing i+4
		
		# Find indices within this 2-cycle period (from crossing i to crossing i+4)
		mask = (time >= cycle_start) & (time <= cycle_end)
		cycle_time = time[mask]
		cycle_power = power[mask]
		
		if len(cycle_time) > 1:
			# Calculate positive and negative areas separately
			positive_area = np.trapz(np.maximum(cycle_power, 0), cycle_time)
			negative_area = np.trapz(np.minimum(cycle_power, 0), cycle_time)
			# Net power = positive area - abs(negative area)
			net_power = positive_area - np.abs(negative_area)
			net_power_per_cycle.append(net_power)
			cycle_info.append((cycle_start, cycle_end, i))  # Store for plotting
			
			cycle_num = (i // 4) + 1
			print(f"\nPeriod {cycle_num} (2 cycles: crossings {i}, {i+1}, {i+2}, {i+3}, {i+4}):")
			print(f"  Period: {cycle_end - cycle_start:.3f} s")
			print(f"  Net power: {net_power:.6f} J (positive area - abs(negative area))")
	
	# Calculate average net power per 2 cycles
	if len(net_power_per_cycle) > 0:
		avg_net_power = np.mean(net_power_per_cycle)
		std_net_power = np.std(net_power_per_cycle)
		print(f"\nAverage net power per 2 cycles: {avg_net_power:.6f} ± {std_net_power:.6f} J")
		print(f"Number of complete 2-cycle periods: {len(net_power_per_cycle)}")
else:
	print("\nNot enough zero crossings detected. Using FFT-based period estimation.")
	
	# Use FFT period to calculate net power
	num_cycles = int((time[-1] - time[0]) / period)
	print(f"Estimated number of cycles: {num_cycles}")
	
	for i in range(num_cycles):
		cycle_start = time[0] + i * period
		cycle_end = time[0] + (i + 1) * period
		
		# Find indices within this cycle
		mask = (time >= cycle_start) & (time < cycle_end)
		cycle_time = time[mask]
		cycle_power = power[mask]
		
		if len(cycle_time) > 1:
			# Calculate positive and negative areas separately
			positive_area = np.trapz(np.maximum(cycle_power, 0), cycle_time)
			negative_area = np.trapz(np.minimum(cycle_power, 0), cycle_time)
			# Net power = positive area - abs(negative area)
			net_power = positive_area - np.abs(negative_area)
			net_power_per_cycle.append(net_power)
			
			if i < 5:  # Print first 5 periods
				print(f"\nPeriod {i+1} (2 cycles):")
				print(f"  Period: {2 * period:.3f} s")
				print(f"  Net power: {net_power:.6f} J (positive area - abs(negative area))")
	
	if len(net_power_per_cycle) > 0:
		avg_net_power = np.mean(net_power_per_cycle)
		std_net_power = np.std(net_power_per_cycle)
		print(f"\nAverage net power per 2 cycles: {avg_net_power:.6f} ± {std_net_power:.6f} J")
		print(f"Total number of 2-cycle periods analyzed: {len(net_power_per_cycle)}")

# Convert to numpy array and create cycle numbers
net_power_per_cycle = np.array(net_power_per_cycle)
cycle_numbers = np.arange(1, len(net_power_per_cycle) + 1)  # n = 1, 2, 3, ...

# Create plots directory
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)
netvoltage_plots_dir = plots_dir / 'netvoltage'
netvoltage_plots_dir.mkdir(exist_ok=True)

# Plot 1: Original power with zero crossings and shaded integrated areas
if len(zero_crossings) > 0:
	first_crossing = zero_crossings[0]
	last_crossing = zero_crossings[-1]  # End at last zero crossing
	
	# Get data for the full range
	mask_full = (time >= first_crossing) & (time <= last_crossing)
	time_full = time[mask_full]
	power_full = power[mask_full]
	
	# Create single plot
	fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
	ax1.plot(time_full, power_full, 'b-', linewidth=1.5, label='Power', alpha=0.7)
	
	# Shade the integrated area for each cycle
	first_cycle = True
	for cycle_start, cycle_end, crossing_idx in cycle_info:
		# Get data for this cycle
		mask_cycle = (time >= cycle_start) & (time <= cycle_end)
		time_cycle = time[mask_cycle]
		power_cycle = power[mask_cycle]
		
		if len(time_cycle) > 1:
			# Shade the area under the curve
			ax1.fill_between(time_cycle, 0, power_cycle, alpha=0.3, color='green', 
			                where=(power_cycle >= 0), label='Positive area' if first_cycle else '')
			ax1.fill_between(time_cycle, 0, power_cycle, alpha=0.3, color='red', 
			                where=(power_cycle < 0), label='Negative area' if first_cycle else '')
			first_cycle = False
	
	# Mark zero crossings with thin crosses
	zero_crossings_power = np.zeros(len(zero_crossings))
	ax1.plot(zero_crossings, zero_crossings_power, 'r+', markersize=6, 
	        markeredgewidth=1, label='Zero crossings', zorder=5)
	
	ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
	ax1.set_xlabel('Time (s)', fontsize=12)
	ax1.set_ylabel('Power (W)', fontsize=12)
	ax1.set_title(f'Power with Integrated Areas: {first_crossing:.1f}s - {last_crossing:.1f}s', 
	              fontsize=14, fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc='best', fontsize=11)
	ax1.set_xlim(first_crossing, last_crossing)
	
	plt.tight_layout()
	plot_path1 = netvoltage_plots_dir / 'power_with_zero_crossings.png'
	plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
	print(f"\nPlot 1 saved to: {plot_path1}")
	plt.close()
else:
	print("\nNo zero crossings found, skipping power plot.")

# Plot 2: Net power vs cycle number
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.plot(cycle_numbers, net_power_per_cycle, 'b+-', markersize=8, markeredgewidth=1.5, 
         linewidth=1.5, label='Net power per cycle')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Cycle Number (n)', fontsize=12)
ax2.set_ylabel('Net Power (J)', fontsize=12)
ax2.set_title('Net Power per 2 Cycles vs Period Number', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=11)

plt.tight_layout()
plot_path2 = netvoltage_plots_dir / 'netpower_vs_cycle.png'
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {plot_path2}")
plt.close()

# Save data to CSV
data_df = pd.DataFrame({
	'Cycle_Number (n)': cycle_numbers,
	'Net_Power (J)': net_power_per_cycle
})
csv_path = Path('data/netflux/netpower_per_cycle.csv')
csv_path.parent.mkdir(parents=True, exist_ok=True)
data_df.to_csv(csv_path, index=False)
print(f"Net power data saved to: {csv_path}")

# Plot 3: Voltage and Current overlaid
fig3, ax3 = plt.subplots(1, 1, figsize=(14, 6))

# Plot voltage on left y-axis
ax3.plot(time, voltage, 'b-', linewidth=1.5, label='Voltage', alpha=0.7)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_ylabel('Voltage (V)', fontsize=12, color='b')
ax3.tick_params(axis='y', labelcolor='b')
ax3.grid(True, alpha=0.3)

# Create second y-axis for current
ax3_current = ax3.twinx()
ax3_current.plot(time, current, 'r-', linewidth=1.5, label='Current', alpha=0.7)
ax3_current.set_ylabel('Current (A)', fontsize=12, color='r')
ax3_current.tick_params(axis='y', labelcolor='r')

# Set title
ax3.set_title('Voltage and Current vs Time', fontsize=14, fontweight='bold')

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_current.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)

plt.tight_layout()
plot_path3 = netvoltage_plots_dir / 'voltage_and_current.png'
plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
print(f"Plot 3 saved to: {plot_path3}")
plt.close()

# Plot 4: Net flux and net power vs cycle number (combined)
# Load net flux data from CSV
netflux_csv_path = Path('data/netflux/netflux_per_cycle.csv')
if netflux_csv_path.exists():
	netflux_data = pd.read_csv(netflux_csv_path)
	netflux_cycle_numbers = netflux_data['Cycle_Number (n)'].to_numpy()
	netflux_per_cycle = netflux_data['Net_Flux (J)'].to_numpy()
	
	# Create figure with dual y-axes
	fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
	
	# Plot net flux on left y-axis
	ax4.plot(netflux_cycle_numbers, netflux_per_cycle, 'b+', markersize=8, markeredgewidth=1.5,
	         label='Net energy per cycle (cold plate)', linestyle='None')
	ax4.set_xlabel('Cycle Number (n)', fontsize=12)
	ax4.set_ylabel('Net Energy (J)', fontsize=12, color='b')
	ax4.tick_params(axis='y', labelcolor='b')
	ax4.grid(True, alpha=0.3)
	ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
	
	# Create second y-axis for net power
	ax4_power = ax4.twinx()
	ax4_power.plot(cycle_numbers, net_power_per_cycle, 'r+', markersize=8, markeredgewidth=1.5,
	               label='Net energy per cycle (input energy)', linestyle='None')
	ax4_power.set_ylabel('Net Energy (J)', fontsize=12, color='r')
	ax4_power.tick_params(axis='y', labelcolor='r')
	
	# Set title
	ax4.set_title('Net Flux and Net Power per Cycle vs Cycle Number', fontsize=14, fontweight='bold')
	
	# Combine legends
	lines1, labels1 = ax4.get_legend_handles_labels()
	lines2, labels2 = ax4_power.get_legend_handles_labels()
	ax4.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11)
	
	plt.tight_layout()
	plot_path4 = netvoltage_plots_dir / 'netflux_and_netpower_vs_cycle.png'
	plt.savefig(plot_path4, dpi=300, bbox_inches='tight')
	print(f"Plot 4 saved to: {plot_path4}")
	plt.close()
else:
	print(f"\nWarning: {netflux_csv_path} not found. Skipping combined net flux/net power plot.")
	print("Please run netflux.py first to generate the net flux data.")

