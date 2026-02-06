import matplotlib.pyplot as plt
from pathlib import Path

def generate_current_equation_image():
    """
    Generate an image of the current equation: I(t) = (V(t) - α(T_h - T_c)) / R
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Render the equation using LaTeX
    equation = r'$I(t) = \frac{V(t) - \alpha(T_h - T_c)}{R}$'
    
    # Display the equation
    ax.text(0.5, 0.5, equation, fontsize=24, ha='center', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/current_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Equation image saved to: {save_path}")
    plt.close()


def generate_cold_plate_equation_image():
    """
    Generate an image of the cold plate equation with expanded Q_c term.
    Based on heatpump.py: C_c * dT_c/dt = Qc_peltier + q_interface
    where Qc_peltier = αIT_c + (1/2)I²R + K(T_h - T_c)
    and q_interface = q'' * A_contact = (T_brass[0] - T_c) / R''_grease * A_contact
    
    Note: From heatpump.py line 46, the sign is +K(Th - Tc), not -K(Th - Tc)
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Label
    label = r'Cold Plate:'
    
    # Main equation (expanded form with q_interface)
    # From heatpump.py line 133: dTc_dt = (Qc_peltier + q_interface) / C_cold_plate
    # From heatpump.py line 46: Qc_peltier = αIT_c + (1/2)I²R + K(Th - Tc)
    # From heatpump.py line 130: q_interface = q'' * A_contact
    
    # Equation with q_interface as the last term
    equation = (r'$C_c \frac{dT_c}{dt} = \alpha I T_c + \frac{1}{2} I^2 R + K(T_h - T_c) + q_{interface}$')
    
    # Display label and equation
    ax.text(0.02, 0.5, label, fontsize=18, ha='left', va='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.15, 0.5, equation, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/cold_plate_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Cold plate equation image saved to: {save_path}")
    plt.close()


def generate_hot_plate_equation_image():
    """
    Generate an image of the hot plate equation.
    Based on heatpump.py: C_h * dT_h/dt = Qh_peltier + h_hot * A_hot * (T_inf - T_h)
    where Qh_peltier = -αIT_h + (1/2)I²R - K(T_h - T_c)
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Label
    label = r'Hot Plate:'
    
    # Main equation
    # From heatpump.py line 134: dTh_dt = (Qh_peltier + h_hot * A_hot * (T_inf - Th)) / C_hot_plate
    # From heatpump.py line 67: Qh_peltier = -αIT_h + (1/2)I²R - K(Th - Tc)
    
    # Equation with dT_h/dt notation and R instead of R_elec
    equation = (r'$C_h \frac{dT_h}{dt} = -\alpha I T_h + \frac{1}{2} I^2 R - K(T_h - T_c) + h A (T_\infty - T_h)$')
    
    # Display label and equation
    ax.text(0.02, 0.5, label, fontsize=18, ha='left', va='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.15, 0.5, equation, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/hot_plate_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Hot plate equation image saved to: {save_path}")
    plt.close()


def generate_heat_equation_image():
    """
    Generate an image of the 1D heat equation.
    ∂T/∂t = D * (∂²T/∂x²)
    where D is the thermal diffusivity (replacing α)
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Main equation with D instead of α
    equation = r'$\frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial x^2}$'
    
    # Display the equation
    ax.text(0.5, 0.5, equation, fontsize=24, ha='center', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/heat_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Heat equation image saved to: {save_path}")
    plt.close()


def generate_q_interface_equation_image():
    """
    Generate an image of the q_interface equation.
    Based on heatpump.py: q_interface = q'' * A_contact
    where q'' = (T_brass[0] - T_c) / R''_grease
    So: q_interface = (T(0,t) - T_c) / R''_g * A
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(14, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Label
    label = r'Interface:'
    
    # Main equation for q_interface
    # From heatpump.py line 130: q_interface = q_double_prime * A_contact
    # From heatpump.py line 127: q_double_prime = (T_brass[0] - Tc) / R_grease_specific
    # Combined: q_interface = (T(0,t) - T_c) / R''_g * A
    
    equation = r'$q_{interface} = \frac{(T(0,t) - T_c) A}{R^{\prime\prime}_g}$'
    
    # Display label and equation
    ax.text(0.02, 0.5, label, fontsize=18, ha='left', va='center',
            transform=ax.transAxes, fontweight='bold')
    ax.text(0.15, 0.5, equation, fontsize=20, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/q_interface_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"q_interface equation image saved to: {save_path}")
    plt.close()


def generate_boundary_conditions_image():
    """
    Generate an image of the heat equation with boundary conditions.
    Format: Heat equation on left, boundary conditions in curly brace on right.
    Based on heatpump.py boundary conditions at x=0 and x=L
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Main heat equation
    heat_eq = r'$\frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial x^2}$'
    
    # Boundary conditions - using manual positioning with separate elements
    bc1 = r'$\left.\frac{\partial T}{\partial x}\right|_{x=L} = 0$'
    bc2 = r'$-k_{brass} \left.\frac{\partial T}{\partial x}\right|_{x=0} = q_{interface}$'
    
    # Position heat equation on left
    ax.text(0.1, 0.5, heat_eq, fontsize=20, ha='left', va='center',
            transform=ax.transAxes)
    
    # Position curly brace and boundary conditions on right, closer together
    # Draw curly brace
    ax.text(0.52, 0.5, r'$\{$', fontsize=36, ha='center', va='center',
            transform=ax.transAxes)
    
    # Position boundary conditions, closer to the brace
    ax.text(0.58, 0.6, bc1, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.58, 0.4, bc2, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/boundary_conditions.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Boundary conditions image saved to: {save_path}")
    plt.close()


def generate_objective_function_image():
    """
    Generate an image of the objective function used in optimization.
    Based on optimse.py: f(α, K_therm, R_elec) = 0.7 × MSE + 0.3 × correlation_error × Var(experimental)
    Format: Curly brace on left, equations stacked vertically with less spacing
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Main objective function
    objective_eq = (r'$f(\alpha, K, R) = 0.7 \times \text{MSE} + 0.3 \times (1 - \rho) \times \text{Var}(T_{exp})$')
    
    mse_eq = r'$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (T_{model}(t_i) - T_{exp}(t_i))^2$'
    
    rho_eq = (r'$\rho = \text{corr}(T_{model} - \bar{T}_{model}, T_{exp} - \bar{T}_{exp})$')
    
    # Position curly brace on left
    ax.text(0.05, 0.5, r'$\{$', fontsize=40, ha='center', va='center',
            transform=ax.transAxes)
    
    # Position equations stacked vertically with less spacing
    ax.text(0.15, 0.65, objective_eq, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.15, 0.45, mse_eq, fontsize=16, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.15, 0.25, rho_eq, fontsize=16, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/objective_function.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Objective function equation image saved to: {save_path}")
    plt.close()


def generate_parameter_bounds_image():
    """
    Generate an image showing the constrained region (bounds) for optimization parameters.
    Based on optimse.py bounds: α ∈ [0.01, 0.06], K ∈ [0.1, 2.0], R ∈ [1.0, 5.0]
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Parameter bounds
    alpha_bound = r'$0.01 \leq \alpha \leq 0.06$'
    K_bound = r'$0.1 \leq K \leq 2.0$'
    R_bound = r'$1.0 \leq R \leq 5.0$'
    
    # Position curly brace on left
    ax.text(0.05, 0.5, r'$\{$', fontsize=40, ha='center', va='center',
            transform=ax.transAxes)
    
    # Position bounds stacked vertically with less spacing
    ax.text(0.15, 0.6, alpha_bound, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.15, 0.4, K_bound, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.15, 0.2, R_bound, fontsize=18, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/parameter_bounds.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Parameter bounds equation image saved to: {save_path}")
    plt.close()


def generate_modified_heat_equation_image():
    """
    Generate an image of the modified heat equation with convection.
    Based on convection.py: ∂T/∂t = D * ∂²T/∂x² - (2h / (ρ * c_p * r)) * (T - T_∞)
    where the convection term accounts for heat loss from the rod surface
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Modified heat equation with convection
    # From convection.py line 129: ∂T/∂t = α * ∂²T/∂x² - (hP / (ρ * c_p * A)) * (T - T_∞)
    # For cylindrical rod: P = 2πr, A = πr²
    # So: hP / (ρ * c_p * A) = 2h / (ρ * c_p * r)
    # Using specific notation: ρ_brass and c_brass
    
    modified_eq = (r'$\frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial x^2} - \frac{2h}{\rho_{brass} c_{brass} r} (T - T_\infty)$')
    
    # Display the equation
    ax.text(0.5, 0.5, modified_eq, fontsize=22, ha='center', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/modified_heat_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Modified heat equation image saved to: {save_path}")
    plt.close()


def generate_newtons_cooling_law_image():
    """
    Generate an image of Newton's Law of Cooling used in cooling.py.
    T(t) = T_inf + (T0 - T_inf) * exp(-t/τ)
    where τ = (ρ_brass * c_brass * r) / (2h) for a long cylindrical rod
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Main cooling law equation
    cooling_eq = r'$T(t) = T_\infty + (T_0 - T_\infty) e^{-t/\tau}$'
    
    # Time constant equation (simplified form for long cylindrical rod)
    tau_eq = r'$\tau = \frac{\rho_{brass} c_{brass} r}{2h}$'
    
    # Position equations with curly brace on left
    ax.text(0.05, 0.5, r'$\{$', fontsize=40, ha='center', va='center',
            transform=ax.transAxes)
    
    # Position equations stacked vertically with less spacing
    ax.text(0.15, 0.6, cooling_eq, fontsize=20, ha='left', va='center',
            transform=ax.transAxes)
    ax.text(0.15, 0.3, tau_eq, fontsize=20, ha='left', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/newtons_cooling_law.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Newton's cooling law equation image saved to: {save_path}")
    plt.close()


def generate_h_correction_equation_image():
    """
    Generate an image showing the h correction formula.
    Based on cooling.py: h_corrected = h_0 + (h_raw - h_0raw)
    where h_0 is the assumed baseline (11.2 W/(m²·K))
    """
    # Create figure with transparent background
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Correction equation
    correction_eq = r'$h_{corrected} = h_0 + (h_{raw} - h_{0,raw})$'
    
    # Display the equation
    ax.text(0.5, 0.5, correction_eq, fontsize=22, ha='center', va='center',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save to plots/equations folder
    save_path = Path('plots/equations/h_correction_equation.png')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    print(f"h correction equation image saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    generate_current_equation_image()
    generate_cold_plate_equation_image()
    generate_hot_plate_equation_image()
    generate_heat_equation_image()
    generate_q_interface_equation_image()
    generate_boundary_conditions_image()
    generate_objective_function_image()
    generate_parameter_bounds_image()
    generate_modified_heat_equation_image()
    generate_newtons_cooling_law_image()
    generate_h_correction_equation_image()

