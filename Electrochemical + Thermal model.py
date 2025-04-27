# Parameters (Extended)
mass = 0.1             # Battery mass (kg)
Cp = 900               # Specific heat (J/kg-K)
R_th = 1.5             # Thermal resistance (K/W)
T_ambient = 298        # Ambient temperature (K)
I = 5                  # Current (A)
V = 3.7                # Voltage (V)
Q_gen = 0.3            # Heat generation rate (W)
time_end = 3600        # Simulation time (s)

# Thermal & Electrochemical Coupling
def thermal_model(t, T):
    dTdt = (Q_gen - (T - T_ambient)/R_th) / (mass * Cp)
    return dTdt

# Time array
time = np.arange(0, time_end, 1)

# Initial temperature
T_init = 298

# Solve for temperature evolution
sol = solve_ivp(thermal_model, [0, time_end], [T_init], t_eval=time)

# Plot results
plt.plot(time, sol.y[0])
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Thermal Model Coupled with Electrochemical Reaction')
plt.grid()
plt.show()
