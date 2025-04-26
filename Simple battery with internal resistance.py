import numpy as np
import matplotlib.pyplot as plt

# Parameters
Voc = 4.2      # Open circuit voltage (V)
Rint = 0.05    # Internal resistance (Ohm)
I = 2.0        # Constant discharge current (A)
dt = 1.0       # Time step (s)
t_total = 3600 # 1 hour

# Time array
t = np.arange(0, t_total, dt)
V = np.zeros_like(t)

for i in range(len(t)):
    V[i] = Voc - I * Rint

# Plot
plt.plot(t/60, V)
plt.title('Rint Battery Model - Constant Current Discharge')
plt.xlabel('Time (minutes)')
plt.ylabel('Terminal Voltage (V)')
plt.grid()
plt.show()
