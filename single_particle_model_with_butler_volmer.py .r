import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
D = 1e-14              # Diffusion coefficient (m^2/s)
R = 5e-6               # Particle radius (m)
c0 = 1000              # Initial concentration (mol/m^3)
c_max = 25000          # Maximum concentration (mol/m^3)
i_app = 1.5            # Applied current density (A/m^2)
F = 96485              # Faraday constant (C/mol)
T = 298                # Temperature (K)
R_gas = 8.314          # Gas constant (J/mol/K)
alpha = 0.5            # Charge transfer coefficient
k0 = 1e-11             # Reaction rate constant (m/s)
U0 = 4.2               # Open Circuit Potential (V)

# Spatial discretization
N = 50
r = np.linspace(0, R, N)
dr = r[1] - r[0]

# Helper function for Laplacian in spherical coords
def laplacian(c):
    lap = np.zeros_like(c)
    lap[1:-1] = (c[2:] - 2*c[1:-1] + c[:-2]) / dr**2 + (2/r[1:-1])*(c[2:] - c[:-2])/(2*dr)
    lap[0] = 0  # center symmetry
    lap[-1] = 0  # will be handled by surface BC
    return lap

# Surface Butler-Volmer reaction
def butler_volmer(cs):
    eta = U0 - (R_gas*T)/(alpha*F)*np.log((c_max - cs)/cs)
    j = k0 * ((c_max - cs)**alpha) * (cs**alpha) * (np.exp(-alpha*F*eta/(R_gas*T)) - np.exp((1-alpha)*F*eta/(R_gas*T)))
    return j

# ODE system
def dcdt(t, c):
    dc = D * laplacian(c)
    j_surf = butler_volmer(c[-1])
    dc[-1] = -j_surf*F/D  # Flux boundary condition at surface
    return dc

# Initial condition
c_init = np.ones(N) * c0

# Time span
t_span = (0, 1000)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Solve
sol = solve_ivp(dcdt, t_span, c_init, t_eval=t_eval, method='RK45')

# Plot results
plt.figure(figsize=(8,5))
for idx in [0,100,200,300,400]:
    plt.plot(r*1e6, sol.y[:,idx], label=f"t={sol.t[idx]:.1f}s")
plt.xlabel('Radius (microns)')
plt.ylabel('Concentration (mol/m³)')
plt.title('Single Particle Model with Butler-Volmer')
plt.legend()
plt.grid()
plt.show()
