#!/usr/bin/env python3
"""Generate image table from threshold testing results (averaged row)."""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read CSV
csv_path = Path('plots/data/comparison/threshold_testing_results.csv')
if not csv_path.exists():
    csv_path = Path('data/comparison/threshold_testing_results.csv')
df = pd.read_csv(csv_path)
avg_row = df.iloc[-1]  # Last row (average)

# Extract key parameters for table (RMSE, MSE, Mean Residual, T-test, Pearson r)
# Use consistent decimal places: 3 decimal places for all values
# T-test validity: p-value >= 0.05 means not significantly different from zero
t_test_valid = avg_row['t_pvalue'] >= 0.05 if not np.isnan(avg_row['t_pvalue']) else False
# Pearson r validity: typically want r >= 0.9 or 0.95 for good correlation
pearson_r_threshold = 0.900  # Common threshold for good correlation
pearson_r_valid = avg_row['pearson_r'] >= pearson_r_threshold if not np.isnan(avg_row['pearson_r']) else False

# Format values with uncertainties
def format_with_uncertainty(value, uncertainty, decimals=3):
    """Format value ± uncertainty."""
    if np.isnan(value) or np.isnan(uncertainty):
        return f'{value:.3f}'
    return f'{value:.3f} ± {uncertainty:.3f}'

data = [
    ['Parameter', 'Value', 'Constraint', 'Valid'],
    ['RMSE (°C)', format_with_uncertainty(avg_row["rmse"], avg_row.get("rmse_uncertainty", 0.0)), f'≤ {avg_row["rmse_threshold"]:.3f}', '✓' if avg_row['rmse_valid'] else '✗'],
    ['MSE (°C²)', format_with_uncertainty(avg_row["mse"], avg_row.get("mse_uncertainty", 0.0)), f'≤ {avg_row["mse_threshold"]:.3f}', '✓' if avg_row['mse_valid'] else '✗'],
    ['Mean Residual (°C)', format_with_uncertainty(avg_row["mean_residual"], avg_row.get("mean_residual_uncertainty", 0.0)), f'≤ {avg_row["mean_residual_threshold"]:.3f}', '✓' if avg_row['mean_residual_valid'] else '✗'],
    ['T-test of Mean Residual (p-value)', f'{avg_row["t_pvalue"]:.3f}', f'≥ 0.050', '✓' if t_test_valid else '✗'],
    ['Pearson r', format_with_uncertainty(avg_row["pearson_r"], avg_row.get("pearson_r_uncertainty", 0.0)), f'≥ {pearson_r_threshold:.3f}', '✓' if pearson_r_valid else '✗'],
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

# No title - removed as requested

# Save
save_path = Path('plots/validity/threshold_testing_table.png')
save_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Table saved to: {save_path}')
plt.close()

