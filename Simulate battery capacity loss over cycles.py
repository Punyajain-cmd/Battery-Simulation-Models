import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_capacity = 3.0    # Ah
fade_per_cycle = 0.0005   # Capacity loss per cycle
n_cycles = 1000           # Total number of cycles

# Arrays
cycles = np.arange(0, n_cycles+1)
capacity = initial_capacity * (1 - fade_per_cycle) ** cycles

# Plot
plt.figure(figsize=(8,5))
plt.plot(cycles, capacity)
plt.title('Battery Capacity Fade Over Cycles')
plt.xlabel('Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.grid()
plt.show()
