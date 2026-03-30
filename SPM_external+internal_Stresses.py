import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ===================================================================
# PART 1: PRE-CALCULATE AND AVERAGE THE 3D EXTERNAL STRESS
# ===================================================================

# --- Material and geometric properties for the vibrating plate ---
E_plate = 210e9
nu_plate = 0.3
h = 0.01
D_flexural = E_plate * h**3 / (12 * (1 - nu_plate**2))
a = 1.0

# --- Time properties for the vibration ---
p0 = 1000
f = 1
omega = 2 * np.pi * f
t_max = 3600
Nt = 100
time_numpy = np.linspace(0, t_max, Nt)

# --- Grid setup for the plate ---
N = 50
x = np.linspace(0, a, N)
y = np.linspace(0, a, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

# --- Precompute time-dependent stress fields ---
print("Step 1: Pre-calculating 3D stress field...")
sigma_x_surface_list = []
sigma_y_surface_list = []

for t in time_numpy:
    p_t = p0 * np.cos(omega * t)
    w = (p_t * a**4 / (D_flexural * np.pi**6)) * (np.sin(np.pi * X / a) * np.sin(np.pi * Y / a))
    w_xx = np.gradient(np.gradient(w, dx, axis=1), dx, axis=1)
    w_yy = np.gradient(np.gradient(w, dx, axis=0), dx, axis=0)
    Mx = -D_flexural * (w_xx + nu_plate * w_yy)
    My = -D_flexural * (nu_plate * w_xx + w_yy)
    sigma_x_surface = (6 * Mx / h**2)
    sigma_y_surface = (6 * My / h**2)
    sigma_x_surface_list.append(sigma_x_surface)
    sigma_y_surface_list.append(sigma_y_surface)

# --- Averaging Step ---
sigma_h_surface_t = (np.array(sigma_x_surface_list) + np.array(sigma_y_surface_list)) / 3
avg_hydrostatic_stress_t = np.mean(sigma_h_surface_t, axis=(1, 2))
print("Step 2: Averaged stress calculated. Ready for PyBaMM.")

# ===================================================================
# PART 2: SETUP AND RUN THE 1D PYBAMM BATTERY MODEL
# ===================================================================

model = pybamm.BaseModel("Coupled SPM")
c_e_n = pybamm.Variable("Concentration of Negative particle", domain="negative particle")
c_e_p = pybamm.Variable("Concentration of Positive particle", domain="positive particle")

# --- Parameters ---
F = pybamm.Scalar(96485.33289)
R_const = pybamm.Scalar(8.314)
T = pybamm.Scalar(298.15)
I_app = pybamm.Scalar(5.0)

# Negative Electrode
D_n = pybamm.Scalar(3.9e-14)
R_n = pybamm.Scalar(10e-6)
c_n_max = pybamm.Scalar(3.1e4)
c_e0_n = pybamm.Scalar(2.5e4)
eps_s_n = pybamm.Scalar(0.6)
L_n = pybamm.Scalar(50e-6)
Omega_n = pybamm.Scalar(3.497e-6)

# Positive Electrode
D_p = pybamm.Scalar(1.2e-14)
R_p = pybamm.Scalar(10e-6)
c_p_max = pybamm.Scalar(5.0e4)
c_e0_p = pybamm.Scalar(2.4e4)
eps_s_p = pybamm.Scalar(0.5)
L_p = pybamm.Scalar(50e-6)
Omega_p = pybamm.Scalar(3.497e-6)

# --- Coupling Step using pybamm.Interpolant ---
external_stress_function = pybamm.Interpolant(
    time_numpy, avg_hydrostatic_stress_t, pybamm.Time(),
    name="External Stress [Pa]"
)

# --- Define Electrochemical Expressions and Equations ---
c_surf_n = pybamm.surf(c_e_n)
a_s_n = 3 * eps_s_n / R_n
j_n = I_app / (a_s_n * L_n)

c_surf_p = pybamm.surf(c_e_p)
a_s_p = 3 * eps_s_p / R_p
j_p = -I_app / (a_s_p * L_p) # Corrected sign for discharge

flux_n = -D_n * pybamm.grad(c_e_n) + (D_n * Omega_n * c_e_n / (R_const * T)) * external_stress_function
dc_e_n_dt = -pybamm.div(flux_n)

flux_p = -D_p * pybamm.grad(c_e_p) + (D_p * Omega_p * c_e_p / (R_const * T)) * external_stress_function
dc_e_p_dt = -pybamm.div(flux_p)

model.rhs = {c_e_n: dc_e_n_dt, c_e_p: dc_e_p_dt}

model.initial_conditions = {c_e_n: c_e0_n, c_e_p: c_e0_p}
grad_surf_n = -j_n / (F * D_n)
grad_surf_p = -j_p / (F * D_p)
model.boundary_conditions = {
    c_e_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (grad_surf_n, "Neumann")},
    c_e_p: {"left": (pybamm.Scalar(0), "Neumann"), "right": (grad_surf_p, "Neumann")},
}

model.variables = {
    "Concentration of Negative particle": c_e_n,
    "Concentration of Positive particle": c_e_p,
    "External Stress [Pa]": external_stress_function,
}

# --- Setup and Solve ---
print("Step 3: Running PyBaMM simulation with coupled stress...")
geometry = {
    "negative particle": {"r_n": {"min": pybamm.Scalar(0), "max": R_n}},
    "positive particle": {"r_p": {"min": pybamm.Scalar(0), "max": R_p}},
}
submesh_types = { "negative particle": pybamm.Uniform1DSubMesh, "positive particle": pybamm.Uniform1DSubMesh }
var_pts = {"r_n": 20, "r_p": 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = { "negative particle": pybamm.FiniteVolume(), "positive particle": pybamm.FiniteVolume() }
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

solver = pybamm.ScipySolver()
solution = solver.solve(model, time_numpy)
print("Simulation complete.")

# --- Plotting ---
# CORRECTED CODE
if solution is not None:
    solution.plot([
        "Concentration of Negative particle",
    "Concentration of Positive particle",
    "External Stress [Pa]"
])

c_e_n_sol = solution["Concentration of Negative particle"].evaluate()
c_e_p_sol = solution["Concentration of Positive particle"].evaluate()

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Custom Simulation Plots")

axs[0,0].set_title("Negative Electrode Surface Concentration vs Time")
axs[0,0].plot(solution.t, c_e_n_sol(solution.t, r=R_n.value))
axs[0,0].set_xlabel("Time [s]")
axs[0,0].set_ylabel("Concentration [mol/m$^3$]")
axs[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axs[0,1].set_title("Negative Electrode Concentration at t = 1000 s")
r_n_vals = np.linspace(0, R_n.value, 100)
axs[0,1].plot(r_n_vals, c_e_n_sol(1000, r=r_n_vals))
axs[0,1].set_xlabel("Particle Radius, r_n [m]")
axs[0,1].set_ylabel("Concentration [mol/m$^3$]")
axs[0,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axs[1,0].set_title("Positive Electrode Surface Concentration vs Time")
axs[1,0].plot(solution.t, c_e_p_sol(solution.t, r=R_p.value))
axs[1,0].set_xlabel("Time [s]")
axs[1,0].set_ylabel("Concentration [mol/m$^3$]")
axs[1,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

axs[1,1].set_title("Positive Electrode Concentration at t = 1000 s")
r_p_vals = np.linspace(0, R_p.value, 100)
axs[1,1].plot(r_p_vals, c_e_p_sol(1000, r=r_p_vals))
axs[1,1].set_xlabel("Particle Radius, r_p [m]")
axs[1,1].set_ylabel("Concentration [mol/m$^3$]")
axs[1,1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.show()