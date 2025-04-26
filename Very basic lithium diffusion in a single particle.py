import numpy as np
import matplotlib.pyplot as plt

# Parameters
D = 1e-14         # Diffusion coefficient (m^2/s)
r_max = 5e-6      # Particle radius (m)
N = 50            # Number of points in radius
dr = r_max / N
dt = 0.5          # Time step (s)
t_total = 1000    # Total simulation time (s)

# Arrays
r = np.linspace(0, r_max, N)
c = np.zeros(N) + 1000  # Initial concentration
c_new = np.zeros(N)
t = np.arange(0, t_total, dt)

# Boundary conditions
c_surface = 500  # Assume surface lithium concentration drops suddenly

# Simulation
for time in t:
    for i in range(1, N-1):
        d2c_dr2 = (c[i+1] - 2*c[i] + c[i-1]) / dr**2
        dc_dr = (c[i+1] - c[i-1]) / (2*dr)
        c_new[i] = c[i] + dt * D * (d2c_dr2 + 2/r[i]*dc_dr)
    c_new[0] = c_new[1]        # Symmetry at center
    c_new[-1] = c_surface      # Set surface concentration
    c = c_new.copy()

# Plot
plt.plot(r*1e6, c)
plt.title('Single Particle Model - Concentration Profile')
plt.xlabel('Radius (microns)')
plt.ylabel('Concentration (mol/m3)')
plt.grid()
plt.show()
