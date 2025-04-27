import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
L = 1e-3           # Thickness of electrode (m)
R = 8.314          # Gas constant (J/mol-K)
F = 96485          # Faraday constant (C/mol)
T = 298            # Temperature (K)
D = 1e-14          # Diffusion coefficient (m^2/s)
i_app = 2.0        # Applied current (A)
C_max = 15000      # Maximum concentration (mol/m^3)
R_p = 1e-6         # Radius of the particle (m)
kappa = 1e-5       # Conductivity of electrolyte (S/m)

# Model equations
def p2d_model(t, y):
    c_s, c_e, eta = np.split(y, 3)

    # Diffusion equation in solid phase
    dc_s = D * (np.roll(c_s, -1) - 2 * c_s + np.roll(c_s, 1)) / R_p**2
    
    # Electrolyte concentration change (Nernst-Planck equation)
    dc_e = -i_app / (F * C_max * L)
    
    # Butler-Volmer equation for the reaction kinetics at the surface
    eta_surface = eta[0] - 0.5  # Simplified overpotential
    return np.concatenate([dc_s, dc_e, eta_surface])

# Initial condition
c_s_init = np.ones(100) * 1000     # Initial concentration in solid phase
c_e_init = np.ones(100) * 1000     # Initial concentration in electrolyte
eta_init = np.zeros(100)           # Initial potential

y_init = np.concatenate([c_s_init, c_e_init, eta_init])

# Time span
t_span = (0, 1000)  # Simulation time
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve ODE system
sol = solve_ivp(p2d_model, t_span, y_init, t_eval=t_eval, method='RK45')

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(t_eval, sol.y[:100, :].T)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/m³)')
plt.title('P2D Model - Battery Solid Phase Concentration')
plt.grid()
plt.show()
