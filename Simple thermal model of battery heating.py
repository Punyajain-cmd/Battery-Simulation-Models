import numpy as np
import matplotlib.pyplot as plt

# Parameters
mass = 0.05          # Battery mass (kg)
Cp = 900             # Specific heat (J/kg.K)
Rth = 1.5            # Thermal resistance (K/W)
Tamb = 25            # Ambient temperature (°C)
I = 5.0              # Current (A)
Rint = 0.02          # Internal resistance (Ohm)
dt = 1               # Time step (s)
t_total = 3600       # 1 hour

# Arrays
t = np.arange(0, t_total, dt)
T = np.zeros_like(t)
T[0] = Tamb

# Simulation
for i in range(1, len(t)):
    heat_gen = I**2 * Rint
    dT = (heat_gen - (T[i-1] - Tamb)/Rth) * dt / (mass*Cp)
    T[i] = T[i-1] + dT

# Plot
plt.plot(t/60, T)
plt.title('Battery Thermal Model - Temperature Rise')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.grid()
plt.show()
