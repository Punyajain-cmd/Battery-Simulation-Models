import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_series = 4   # 4 cells in series
n_parallel = 2 # 2 cells in parallel
cell_capacity = 2.5  # Ah
cell_voltage = 3.7   # V
cell_Rint = 0.05     # Ohm

dt = 1.0
t_total = 3600
t = np.arange(0, t_total, dt)

# Current profile
I_total = np.ones_like(t) * 5.0  # 5A total draw

# Compute per-cell current
I_cell = I_total / n_parallel

# Terminal voltage over time
V_pack = np.zeros_like(t)

for i in range(len(t)):
    V_cell = cell_voltage - I_cell[i]*cell_Rint
    V_pack[i] = V_cell * n_series

# Plot
plt.plot(t/60, V_pack)
plt.title('Battery Pack Voltage During Discharge')
plt.xlabel('Time (minutes)')
plt.ylabel('Pack Voltage (V)')
plt.grid()
plt.show()
