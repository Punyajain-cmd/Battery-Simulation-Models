import pybamm
import numpy as np
import matplotlib.pyplot as plt

# Create base SPM model
model = pybamm.BaseModel(name="SPM with Mechanical Stress")

# Domains
domain_n = "negative particle"
domain_p = "positive particle"

# Parameters
param = pybamm.ParameterValues("Chen2020")

# Update variable definitions with physical limits
c_s_n = pybamm.Variable(
    "Negative particle concentration", 
    domain=domain_n
)

c_s_p = pybamm.Variable(
    "Positive particle concentration",
    domain=domain_p
)


def radial_stress(c_s, E, nu, r, c_ref, name):
    c_avg = pybamm.Integral(c_s * r**2, r) / pybamm.Integral(r**2, r)
    delta_c = (c_s - c_avg)/c_ref
    sigma_rr = 2 * E * Omega / (3 * (1 - nu)) * delta_c
    sigma_tt = E * Omega / (3 * (1 - nu)) * delta_c
    return {
        f"{name} radial stress [Pa]": sigma_rr,
        f"{name} tangential stress [Pa]": sigma_tt,
    }



# Mechanics parameters
Omega = pybamm.Scalar(1e-7)
E_n = pybamm.Scalar(100e9)
E_p = pybamm.Scalar(90e9)
nu_n = pybamm.Scalar(0.3)
nu_p = pybamm.Scalar(0.3)

# Governing equations: Fickian diffusion
r_n = pybamm.SpatialVariable("r_n", domain=domain_n)
r_p = pybamm.SpatialVariable("r_p", domain=domain_p)


c_ref_p = param["Initial concentration in positive electrode [mol.m-3]"]

c_e0 = 2.5e4  # mol/m^3


model.initial_conditions = {
    c_s_n: c_e0 ,  
    c_s_p: c_ref_p 
}

# Stress variables
stress_vars_n = radial_stress(c_s_n, E_n, nu_n, r_n, c_e0, "Negative")
stress_vars_p = radial_stress(c_s_p, E_p, nu_p, r_p, c_ref_p, "Positive")
model.variables.update(stress_vars_n)
model.variables.update(stress_vars_p)

D_s_n = param["Negative particle diffusivity [m2.s-1]"]
D_s_p = param["Positive particle diffusivity [m2.s-1]"]


D_eff = D_s_n * (1 - 0.01 * stress_vars_n["Negative radial stress [Pa]"])


R_n = param["Negative particle radius [m]"]
R_p = param["Positive particle radius [m]"]

# Proper current handling
# Calculate current density based on electrode area
A = param["Electrode width [m]"] * param["Electrode height [m]"]
i_typical = 1e3  # 1 A/m²
t = pybamm.t  # Define t as the PyBaMM time variable
I = i_typical * A * (1 + 0.5 * pybamm.sin(2 * np.pi * t))
param.update({"Current function [A]": I})



N_s_n = -D_eff * pybamm.grad(c_s_n)
N_s_p = -D_s_p * pybamm.grad(c_s_p)

diffusion_n = -pybamm.div(N_s_n)
diffusion_p = -pybamm.div(N_s_p)

model.rhs = {
    c_s_n: diffusion_n,
    c_s_p: diffusion_p,
}



# Corrected boundary conditions using current density relationship
j_n = I / (param["Negative electrode active material volume fraction"] * A)
j_p = -I / (param["Positive electrode active material volume fraction"] * A)

model.boundary_conditions = {
    c_s_n: {"left": (0, "Neumann"), "right": (j_n/(D_s_n*param["Faraday constant [C.mol-1]"]), "Neumann")},
    c_s_p: {"left": (0, "Neumann"), "right": (-j_p/(D_s_p*param["Faraday constant [C.mol-1]"]), "Neumann")},
}



param.process_model(model)  # Substitute all parameters with numerical values

from pybamm import standard_spatial_vars as sv

r_n = sv.r_n
r_p = sv.r_p


# Geometry
geometry = {
    domain_n: {r_n: {"min": pybamm.Scalar(0), "max": R_n}},
    domain_p: {r_p: {"min": pybamm.Scalar(0), "max": R_p}},
}

# Mesh
spatial_methods = {"negative particle": pybamm.FiniteVolume(), "positive particle": pybamm.FiniteVolume()}
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh, "positive particle": pybamm.Uniform1DSubMesh}
var_pts = {"r_n": 80, "r_p": 80}


mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)

# Manual discretisation
disc.process_model(model)
# Enhanced numerical settings

# Stricter solver settings
solver = pybamm.CasadiSolver(
    rtol=1e-8, 
    atol=1e-10,
    root_method="lm",
    root_tol=1e-6
)
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

# Plot using QuickPlot
plot = pybamm.QuickPlot(
    solution,
    ["Negative radial stress [Pa]", "Negative tangential stress [Pa]",
     "Positive radial stress [Pa]", "Positive tangential stress [Pa]"]
)
plot.dynamic_plot()



# If solution is a list, get the first Solution object
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution. Please check the model setup and solver configuration.")

# Extract solution
c_e_sol = solution["Negative particle concentration"]


# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_e_sol(solution.t, r=R_n))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Concentration at surface (r = R_n) [mol/m³]")

r_vals = np.linspace(0, R_n, 100)
ax2.plot(r_vals, c_e_sol(1000, r=r_vals))
ax2.set_xlabel("r")
ax2.set_ylabel("Concentration at t = 1000 s [mol/m³]")

plt.tight_layout()
plt.show()
