#!/usr/bin/env python3
"""Generate image table from transient tau comparison results."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read CSV
csv_path = Path('plots/transient/transient_tau_results.csv')
if not csv_path.exists():
    csv_path = Path('data/transient/transient_tau_results.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Tau results CSV not found. Expected at: plots/transient/transient_tau_results.csv")

df = pd.read_csv(csv_path)

# Format values with uncertainties
def format_with_uncertainty(value, uncertainty, decimals=1):
    """Format value ± uncertainty."""
    if np.isnan(value) or np.isnan(uncertainty):
        return f'{value:.1f}'
    return f'{value:.1f} ± {uncertainty:.1f}'

# Prepare table data
# Validity test: |tau_exp - tau_model| <= k * sqrt(unc_exp^2 + unc_model^2)
# Using k_factor = 2.0 for ~95% confidence interval
k_factor = 2.0

data = [['Thermistor ID', 'Experimental τ (s)', 'Modeling τ (s)', 'Difference (s)', 'Combined Uncertainty (s)', 'Valid']]

for _, row in df.iterrows():
    therm_id = int(row['thermistor_id'])
    tau_exp = row['experimental_tau']
    tau_exp_unc = row['experimental_tau_uncertainty']
    tau_model = row['modeling_tau']
    tau_model_unc = row['modeling_tau_uncertainty']
    
    # Calculate difference
    diff = abs(tau_exp - tau_model)
    
    # Calculate combined uncertainty
    combined_unc = np.sqrt(tau_exp_unc**2 + tau_model_unc**2)
    
    # Validity test: difference <= k * combined_uncertainty
    is_valid = diff <= (k_factor * combined_unc) if not (np.isnan(diff) or np.isnan(combined_unc)) else False
    
    # Format values
    exp_str = format_with_uncertainty(tau_exp, tau_exp_unc)
    model_str = format_with_uncertainty(tau_model, tau_model_unc)
    diff_str = f'{diff:.1f}'
    unc_str = f'{combined_unc:.1f}'
    valid_str = '✓' if is_valid else '✗'
    
    data.append([f'T{therm_id}', exp_str, model_str, diff_str, unc_str, valid_str])

# Create figure
fig, ax = plt.subplots(figsize=(14, max(6, len(data) * 0.5)))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=data[1:], colLabels=data[0], 
                cellLoc='center', loc='center',
                colWidths=[0.12, 0.22, 0.22, 0.15, 0.19, 0.10])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.0)

# Color code validity cells
for i in range(1, len(data)):
    # Validity column (last column, index 5)
    if data[i][5] == '✓':
        table[(i, 5)].set_facecolor('#90EE90')  # Light green
        table[(i, 5)].set_text_props(weight='bold')
    elif data[i][5] == '✗':
        table[(i, 5)].set_facecolor('#FFB6C1')  # Light pink
        table[(i, 5)].set_text_props(weight='bold')

# Header styling
for i in range(6):
    table[(0, i)].set_facecolor('#4A90E2')
    table[(0, i)].set_text_props(weight='bold', color='white', size=12)

# Save
save_path = Path('plots/transient/tau_comparison_table.png')
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Table saved to: {save_path}')
plt.close()


