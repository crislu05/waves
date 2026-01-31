import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Load Qc flux and time data from CSV
csv_path = 'data/netflux/heatpump_data.csv'
data = pd.read_csv(csv_path)

# Extract time and Qc data
time = data['Time (s)'].to_numpy()
Qc = data['Qc (W)'].to_numpy()

print(f"Loaded {len(time)} data points")
print(f"Time range: {time[0]:.2f} s to {time[-1]:.2f} s")
print(f"Qc range: {Qc.min():.4f} W to {Qc.max():.4f} W")

# Detect the period of the oscillation using FFT
# Find the dominant frequency
dt = np.mean(np.diff(time))  # Average time step
N = len(time)
frequencies = np.fft.fftfreq(N, dt)
fft_values = np.fft.fft(Qc)

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
for i in range(len(Qc) - 1):
	if (Qc[i] >= 0 and Qc[i+1] < 0) or (Qc[i] < 0 and Qc[i+1] >= 0):
		# Linear interpolation to find exact zero crossing time
		t_zero = time[i] + (time[i+1] - time[i]) * (0 - Qc[i]) / (Qc[i+1] - Qc[i])
		zero_crossings_all.append(t_zero)

# Filter to only include zero crossings in the middle 90% range
zero_crossings = [t for t in zero_crossings_all if time_start <= t <= time_end]

print(f"  Zero crossings in analysis range: {len(zero_crossings)}")

# Calculate net flux per cycle
# A cycle is defined by three zero crossings: crossing i, i+1, i+2
# This represents a full oscillation: zero -> peak -> zero -> opposite peak -> zero
# Net flux = integral of Qc over one complete cycle (positive area - negative area)
net_flux_per_cycle = []
cycle_info = []  # Store cycle start/end times for shading

if len(zero_crossings) >= 3:
	# Use three consecutive zero crossings to define each cycle
	# Cycle 1: crossings 0, 1, 2 (from crossing 0 to crossing 2)
	# Cycle 2: crossings 2, 3, 4 (from crossing 2 to crossing 4)
	# Cycle 3: crossings 4, 5, 6 (from crossing 4 to crossing 6)
	# etc.
	# Each cycle spans from crossing i to crossing i+2 (three crossings total)
	for i in range(0, len(zero_crossings) - 2, 2):  # Step by 2: i=0,2,4,6,...
		cycle_start = zero_crossings[i]
		cycle_end = zero_crossings[i + 2]  # Full cycle: from crossing i to crossing i+2
		
		# Find indices within this cycle (from crossing i to crossing i+2)
		mask = (time >= cycle_start) & (time <= cycle_end)
		cycle_time = time[mask]
		cycle_Qc = Qc[mask]
		
		if len(cycle_time) > 1:
			# Calculate positive and negative areas separately
			positive_area = np.trapz(np.maximum(cycle_Qc, 0), cycle_time)
			negative_area = np.trapz(np.minimum(cycle_Qc, 0), cycle_time)
			# Net flux = positive area - abs(negative area)
			net_flux = positive_area - np.abs(negative_area)
			net_flux_per_cycle.append(net_flux)
			cycle_info.append((cycle_start, cycle_end, i))  # Store for plotting
			
			cycle_num = (i // 2) + 1
			print(f"\nCycle {cycle_num} (crossings {i}, {i+1}, {i+2}):")
			print(f"  Period: {cycle_end - cycle_start:.3f} s")
			print(f"  Net flux: {net_flux:.6f} J (positive area - abs(negative area))")
	
	# Calculate average net flux per cycle
	if len(net_flux_per_cycle) > 0:
		avg_net_flux = np.mean(net_flux_per_cycle)
		std_net_flux = np.std(net_flux_per_cycle)
		print(f"\nAverage net flux per cycle: {avg_net_flux:.6f} ± {std_net_flux:.6f} J")
		print(f"Number of complete cycles: {len(net_flux_per_cycle)}")
else:
	print("\nNot enough zero crossings detected. Using FFT-based period estimation.")
	
	# Use FFT period to calculate net flux
	num_cycles = int((time[-1] - time[0]) / period)
	print(f"Estimated number of cycles: {num_cycles}")
	
	for i in range(num_cycles):
		cycle_start = time[0] + i * period
		cycle_end = time[0] + (i + 1) * period
		
		# Find indices within this cycle
		mask = (time >= cycle_start) & (time < cycle_end)
		cycle_time = time[mask]
		cycle_Qc = Qc[mask]
		
		if len(cycle_time) > 1:
			# Calculate positive and negative areas separately
			positive_area = np.trapz(np.maximum(cycle_Qc, 0), cycle_time)
			negative_area = np.trapz(np.minimum(cycle_Qc, 0), cycle_time)
			# Net flux = positive area - abs(negative area)
			net_flux = positive_area - np.abs(negative_area)
			net_flux_per_cycle.append(net_flux)
			
			if i < 5:  # Print first 5 cycles
				print(f"\nCycle {i+1}:")
				print(f"  Period: {period:.3f} s")
				print(f"  Net flux: {net_flux:.6f} J (positive area - abs(negative area))")
	
	if len(net_flux_per_cycle) > 0:
		avg_net_flux = np.mean(net_flux_per_cycle)
		std_net_flux = np.std(net_flux_per_cycle)
		print(f"\nAverage net flux per cycle: {avg_net_flux:.6f} ± {std_net_flux:.6f} J")
		print(f"Total number of cycles analyzed: {len(net_flux_per_cycle)}")

# Convert to numpy array and create cycle numbers
net_flux_per_cycle = np.array(net_flux_per_cycle)
cycle_numbers = np.arange(1, len(net_flux_per_cycle) + 1)  # n = 1, 2, 3, ...

# Create plots directory
from pathlib import Path
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)
netflux_plots_dir = plots_dir / 'netflux'
netflux_plots_dir.mkdir(exist_ok=True)

# Plot 1: Original flux with zero crossings and shaded integrated areas
if len(zero_crossings) > 0:
	first_crossing = zero_crossings[0]
	last_crossing = zero_crossings[-1]  # End at last zero crossing
	
	# Get data for the full range
	mask_full = (time >= first_crossing) & (time <= last_crossing)
	time_full = time[mask_full]
	Qc_full = Qc[mask_full]
	
	# Create single plot
	fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
	ax1.plot(time_full, Qc_full, 'b-', linewidth=1.5, label='Qc (Cold plate heat flux)', alpha=0.7)
	
	# Shade the integrated area for each cycle
	first_cycle = True
	for cycle_start, cycle_end, crossing_idx in cycle_info:
		# Get data for this cycle
		mask_cycle = (time >= cycle_start) & (time <= cycle_end)
		time_cycle = time[mask_cycle]
		Qc_cycle = Qc[mask_cycle]
		
		if len(time_cycle) > 1:
			# Shade the area under the curve
			ax1.fill_between(time_cycle, 0, Qc_cycle, alpha=0.3, color='green', 
			                where=(Qc_cycle >= 0), label='Positive area' if first_cycle else '')
			ax1.fill_between(time_cycle, 0, Qc_cycle, alpha=0.3, color='red', 
			                where=(Qc_cycle < 0), label='Negative area' if first_cycle else '')
			first_cycle = False
	
	# Mark zero crossings with thin crosses
	zero_crossings_Qc = np.zeros(len(zero_crossings))
	ax1.plot(zero_crossings, zero_crossings_Qc, 'r+', markersize=6, 
	        markeredgewidth=1, label='Zero crossings', zorder=5)
	
	ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
	ax1.set_xlabel('Time (s)', fontsize=12)
	ax1.set_ylabel('Heat Flux Qc (W)', fontsize=12)
	ax1.set_title(f'Qc Flux with Integrated Areas: {first_crossing:.1f}s - {last_crossing:.1f}s', 
	              fontsize=14, fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.legend(loc='best', fontsize=11)
	ax1.set_xlim(first_crossing, last_crossing)
	
	plt.tight_layout()
	plot_path1 = netflux_plots_dir / 'Qc_flux_with_zero_crossings.png'
	plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
	print(f"\nPlot 1 saved to: {plot_path1}")
	plt.close()
else:
	print("\nNo zero crossings found, skipping Qc flux plot.")

# Plot 2: Net flux vs cycle number
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.plot(cycle_numbers, net_flux_per_cycle, 'b+', markersize=8, markeredgewidth=1.5, 
         label='Net flux per cycle', linestyle='None')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('Cycle Number (n)', fontsize=12)
ax2.set_ylabel('Net Flux (J)', fontsize=12)
ax2.set_title('Net Flux per Cycle vs Cycle Number', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', fontsize=11)

plt.tight_layout()
plot_path2 = netflux_plots_dir / 'netflux_vs_cycle.png'
plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
print(f"Plot 2 saved to: {plot_path2}")
plt.close()

# Save data to CSV
data_df = pd.DataFrame({
	'Cycle_Number (n)': cycle_numbers,
	'Net_Flux (J)': net_flux_per_cycle
})
csv_path = Path('data/netflux/netflux_per_cycle.csv')
data_df.to_csv(csv_path, index=False)
print(f"Net flux data saved to: {csv_path}")

