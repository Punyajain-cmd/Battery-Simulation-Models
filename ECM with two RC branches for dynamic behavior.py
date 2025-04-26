import numpy as np
import matplotlib.pyplot as plt

# Parameters
Voc = 4.2
R0 = 0.01
R1 = 0.015
C1 = 2000
R2 = 0.02
C2 = 1000
I = 1.0
dt = 1.0
t_total = 3600

# Time array
t = np.arange(0, t_total, dt)
V_rc1 = np.zeros_like(t)
V_rc2 = np.zeros_like(t)
V = np.zeros_like(t)

for i in range(1, len(t)):
    dV1 = (-V_rc1[i-1] + I*R1) / (R1*C1) * dt
    V_rc1[i] = V_rc1[i-1] + dV1
    dV2 = (-V_rc2[i-1] + I*R2) / (R2*C2) * dt
    V_rc2[i] = V_rc2[i-1] + dV2
    V[i] = Voc - I*R0 - V_rc1[i] - V_rc2[i]

# Plot
plt.plot(t/60, V)
plt.title('Dual Polarization ECM - Battery Terminal Voltage')
plt.xlabel('Time (minutes)')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
