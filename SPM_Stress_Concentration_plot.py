import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Base model
model = pybamm.BaseModel("SPM with Mechanical Stress")

# Define variables
c_e = pybamm.Variable("Negative particle concentration", domain="negative particle")
sigma = pybamm.Variable("Mechanical Stress", domain="negative particle")

# Parameters
D = 3.9e-14         # Diffusion coefficient [m^2/s]
R = 1e-5            # Particle radius [m]
j = 1.4             # Reaction flux [A/m^2]
F = 96485.33289     # Faraday constant [C/mol]
c_e0 = 2.5e4        # Initial concentration [mol/m^3]
K = 1e9             # Bulk modulus [Pa]
Omega = 3e-6        # Partial molar volume [m^3/mol]
T = 298             # Temperature [K]
from scipy.constants import k as kB  # Boltzmann constant

# Define missing parameters for Q calculation
eps_s_n = 0.6    # Example value for solid phase volume fraction in negative electrode
L_n = 10e-9     # Example value for negative electrode thickness (in meters)
c_n_max = 3.6e4  # Example value for max concentration (mol/m^3)
c_n_min = 0      # Example value for min concentration (mol/m^3)

# Modified flux with stress coupling
N = -D * (pybamm.grad(c_e) + (Omega / (kB * T)) * pybamm.grad(sigma))

# PDE definitions
dc_edt = -pybamm.div(N)
dsigma_dt = K * Omega * dc_edt

model.rhs = {
    c_e: dc_edt,
    sigma: dsigma_dt
}

# Initial conditions
model.initial_conditions = {
    c_e: pybamm.Scalar(c_e0),
    sigma: pybamm.Scalar(0)
}

# Boundary conditions
model.boundary_conditions = {
    c_e: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(-j / (F * D)), "Neumann"),
    },
    sigma: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# Variables for output
model.variables = {
    "Negative particle concentration": c_e,
    "Mechanical Stress": sigma,
    "Flux": N,
    "Rate of Change of Concentration": dc_edt,
    "Rate of Change of Stress": dsigma_dt,
    "Discharge capacity [A.h]": F * eps_s_n * L_n * (c_n_max - c_e),
    "Total lithium in negative electrode [mol]": F * eps_s_n * L_n * (c_n_max - c_e),
    "Total lithium capacity [A.h]": F * eps_s_n * L_n * (c_n_max - c_e) / (c_n_max - c_n_min)
}

# Geometry
geometry = {
    "negative particle": {
        "r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R)}
    }
}

# Mesh and discretisation
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {"r_n": 40}
spatial_methods = {"negative particle": pybamm.FiniteVolume()}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve the model
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)


# If solution is a list, get the first Solution object
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution. Please check the model setup and solver configuration.")

# Extract and plot concentration
c_e_sol = solution["Negative particle concentration"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_e_sol(solution.t, r=R))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Concentration at surface (r = R) [mol/m^3]")

r_vals = np.linspace(0, R, 1000)
ax2.plot(r_vals, c_e_sol(1000, r=r_vals))
ax2.set_xlabel("r [m]")
ax2.set_ylabel("Concentration at t = 1000 s [mol/m^3]")

plt.tight_layout()
plt.show()


solution.plot(output_variables=[
    "Total lithium in negative electrode [mol]",
    "Total lithium capacity [A.h]",
    "Discharge capacity [A.h]",
])

