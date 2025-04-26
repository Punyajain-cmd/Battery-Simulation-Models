import numpy as np
import matplotlib.pyplot as plt

# Parameters
R = 2.0         # Ohms
C = 5000.0      # Farads
V0 = 0          # Initial voltage (V)
V_source = 5.0  # Source voltage (V)
dt = 1.0        # Time step (s)
t_total = 10000 # Total time (s)

# Time array
t = np.arange(0, t_total, dt)
V = np.zeros_like(t)

# Simulation (charging)
for i in range(1, len(t)):
    dV = (V_source - V[i-1]) / (R * C) * dt
    V[i] = V[i-1] + dV

# Plot
plt.figure(figsize=(8,5))
plt.plot(t/3600, V)
plt.title('RC Circuit Battery Charging')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
