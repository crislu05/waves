import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def generate_parameter_table(save_path=None):
	"""
	Generate a table image showing all physical parameters used in the model.
	
	Parameters:
	- save_path: Path to save the table image
	"""
	import matplotlib
	matplotlib.use('Agg')
	
	# Load optimized parameters if available
	params_file = Path('data/netflux/optimized_parameters.json')
	alpha = None
	K_therm = None
	R_elec = None
	
	if params_file.exists():
		with open(params_file, 'r') as f:
			optimized_params = json.load(f)
		alpha = optimized_params.get('alpha', None)
		K_therm = optimized_params.get('K_therm', None)
		R_elec = optimized_params.get('R_elec', None)
	
	# Helper function to format to 3 significant figures without scientific notation
	def format_sig_figs(value, unit=''):
		"""Format value to 3 significant figures without scientific notation."""
		if isinstance(value, (int, float)):
			if value == 0:
				return f'0 {unit}'.strip()
			
			# Calculate number of decimal places needed for 3 significant figures
			# Find the order of magnitude
			if abs(value) >= 1:
				# For numbers >= 1, find how many digits before decimal
				order = int(np.floor(np.log10(abs(value))))
				# We want 3 significant figures total
				# If order is 2 (e.g., 8520), we want 8520 (no decimals)
				# If order is 1 (e.g., 38.0), we want 38.0 (1 decimal)
				# If order is 0 (e.g., 3.97), we want 3.97 (2 decimals)
				if order >= 2:
					# Large numbers: show as integer if possible
					if value == int(value):
						return f'{int(value)} {unit}'.strip()
					else:
						decimals = max(0, 2 - order)
						return f'{value:.{decimals}f} {unit}'.strip()
				else:
					decimals = max(0, 2 - order)
					return f'{value:.{decimals}f} {unit}'.strip()
			else:
				# For numbers < 1, find first non-zero digit position
				order = int(np.floor(np.log10(abs(value))))
				# We need 2 - order decimal places to show 3 sig figs
				decimals = max(0, 2 - order)
				return f'{value:.{decimals}f} {unit}'.strip()
		return value
	
	# Define all parameters with mathematical symbols and proper subscripts
	parameters = [
		{
			'parameter': r'$\alpha$',
			'description': 'Seebeck Coefficient',
			'value': 'Determined by PDE Parameter Optimiser'
		},
		{
			'parameter': r'$K$',
			'description': 'Peltier Thermal Conductance',
			'value': 'Determined by PDE Parameter Optimiser'
		},
		{
			'parameter': r'$R$',
			'description': 'Peltier Electrical Resistance',
			'value': 'Determined by PDE Parameter Optimiser'
		},
		{
			'parameter': r'$T_{\infty}$',
			'description': 'Ambient Temperature',
			'value': format_sig_figs(298.15, 'K')
		},
		{
			'parameter': r'$\rho_{ceramic}$',
			'description': 'Al₂O₃ Density',
			'value': format_sig_figs(3970.0, 'kg/m³')
		},
		{
			'parameter': r'$c_{ceramic}$',
			'description': 'Al₂O₃ Specific Heat',
			'value': format_sig_figs(775.0, 'J/(kg·K)')
		},
		{
			'parameter': r'$C_{hot}$',
			'description': 'Hot side Heat Capacity',
			'value': format_sig_figs(300.0, 'J/K')
		},
		{
			'parameter': r'$\rho_{brass}$',
			'description': 'Brass Density',
			'value': format_sig_figs(8520.0, 'kg/m³')
		},
		{
			'parameter': r'$c_{brass}$',
			'description': 'Brass Specific Heat',
			'value': format_sig_figs(380.0, 'J/(kg·K)')
		},
		{
			'parameter': r'$k_{brass}$',
			'description': 'Brass Thermal Conductivity',
			'value': format_sig_figs(109.0, 'W/(m·K)')
		},
		{
			'parameter': r'$L$',
			'description': 'Brass Rod Length',
			'value': format_sig_figs(0.041 * 1000, 'mm')  # Convert to mm
		},
		{
			'parameter': r'$r$',
			'description': 'Plate/Rod Radius',
			'value': format_sig_figs(0.015 * 1000, 'mm')  # Convert to mm
		},
		{
			'parameter': r'$t_{plate}$',
			'description': 'Ceramic Plate Thickness',
			'value': format_sig_figs(0.002 * 1000, 'mm')  # Convert to mm
		},
		{
			'parameter': r'$t_{grease}$',
			'description': 'Grease Layer Thickness',
			'value': format_sig_figs(0.0001 * 1000, 'mm')  # Convert to mm
		},
		{
			'parameter': r'$k_{grease}$',
			'description': 'Grease Thermal Conductivity',
			'value': format_sig_figs(1.0, 'W/(m·K)')
		},
		{
			'parameter': r'$h_{hot}$',
			'description': 'Convective co-efficient (fan)',
			'value': format_sig_figs(200.0, 'W/(m²·K)')
		},
		{
			'parameter': r'$A_{hot}$',
			'description': 'Heat Sink Surface Area',
			'value': format_sig_figs(0.156 * 10000, 'cm²')  # Convert to cm²
		}
	]
	
	# Prepare table data
	data = [['Parameter', 'Definition', 'Value']]
	
	for param in parameters:
		data.append([param['parameter'], param['description'], param['value']])
	
	# Create figure
	fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.25)))
	ax.axis('tight')
	ax.axis('off')
	
	# Create table
	table = ax.table(cellText=data[1:], colLabels=data[0], 
	                cellLoc='left', loc='center',
	                colWidths=[0.2, 0.4, 0.4])
	
	# Style the table
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	table.scale(1, 2.0)
	
	# Enable LaTeX rendering for parameter column
	plt.rcParams['text.usetex'] = False  # Use matplotlib's mathtext instead
	
	# Header styling
	for i in range(3):
		table[(0, i)].set_facecolor('#4A90E2')
		table[(0, i)].set_text_props(weight='bold', color='white', size=10)
	
	# Row styling - alternate row colors for readability
	# Identify which parameters are assumptions (not PDE-optimized, not measured)
	# Assumptions: T_∞, ρ_ceramic, c_ceramic, C_hot, h_hot, A_hot, t_plate, t_grease, k_grease
	assumption_params = [r'$T_{\infty}$', r'$\rho_{ceramic}$', r'$c_{ceramic}$', r'$C_{hot}$', 
	                    r'$h_{hot}$', r'$A_{hot}$', r'$t_{plate}$', r'$t_{grease}$', r'$k_{grease}$']
	
	for i in range(1, len(data)):
		param_symbol = data[i][0]  # Get the parameter symbol from first column
		for j in range(3):
			# First 3 data rows (indices 1, 2, 3) are PDE-optimized parameters
			if i <= 3:
				table[(i, j)].set_facecolor('#FFB74D')  # Orange for PDE-optimized (scientific standard)
			# Check if this parameter is an assumption
			elif param_symbol in assumption_params:
				table[(i, j)].set_facecolor('#4DB6AC')  # Teal for assumptions (scientific standard)
			else:
				# Measured/material property parameters
				table[(i, j)].set_facecolor('#CE93D8')  # Purple for measured/material properties (scientific standard)
	
	# Save
	if save_path:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"Parameter table saved to: {save_path}")
		plt.close()
	else:
		plt.show()
	
	plt.close()


def main():
	"""Main function to generate parameter table."""
	save_path = Path('plots/parameter_table.png')
	generate_parameter_table(save_path=save_path)


if __name__ == '__main__':
	main()

