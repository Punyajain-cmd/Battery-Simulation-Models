import pybamm
import matplotlib.pyplot as plt
import numpy as np

# Load parameters
params = pybamm.ParameterValues("Chen2020")

# Define the model
model = pybamm.BaseModel()
model.param = params


# Variables
c_s_n = pybamm.Variable(
    "Concentration in negative particle",
    domain="negative particle",
    auxiliary_domains={"secondary": "negative electrode"},
)
c_s_p = pybamm.Variable(
    "Concentration in positive particle",
    domain="positive particle",
    auxiliary_domains={"secondary": "positive electrode"},
)



# Parameters from Chen2020-compatible names
D_n = pybamm.Parameter("Negative electrode diffusivity [m2.s-1]")
D_p = pybamm.Parameter("Positive electrode diffusivity [m2.s-1]")
R_n = pybamm.Parameter("Radius of negative electrode particle [m]")
R_p = pybamm.Parameter("Radius of positive electrode particle [m]")
I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
F = pybamm.Parameter("Faraday constant [C.mol-1]")
c_n0 = pybamm.Parameter("Initial concentration in negative electrode [mol.m-3]")
c_p0 = pybamm.Parameter("Initial concentration in positive electrode [mol.m-3]")
eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
eps_s_p = pybamm.Parameter("Negative electrode active material volume fraction")
L_n = pybamm.Parameter("Thickness of negative electrode [m]")
L_p = pybamm.Parameter("Thickness of positive electrode [m]")
A = pybamm.Parameter("Electrode width [m]") * pybamm.Parameter("Electrode height [m]")
I = 1.0 + 0.5 * pybamm.sin(2 * np.pi * pybamm.t / 3600)  # For variation
# Max concentrations
c_n_max = pybamm.Parameter("Maximum concentration in negative electrode [mol.m-3]")
c_p_max = pybamm.Parameter("Maximum concentration in positive electrode [mol.m-3]")
Q = pybamm.Variable("Discharge capacity [A.h]")

c_n = c_s_n/c_n_max
c_p = c_s_p/c_p_max

# Geometry-dependent variables
a_n = 3 * eps_s_n / R_n
a_p = 3 * eps_s_p / R_p
j_n = I / (F * A * a_n * L_n)
j_p = -I / (F * A * a_p * L_p)
j_n_o = (c_n * (1 - c_n))**0.5
j_p_o = (c_p * (1 - c_p))**0.5

# PDEs for diffusion
N_n = -D_n * pybamm.grad(c_s_n)
N_p = -D_p * pybamm.grad(c_s_p)
dc_ndt = -pybamm.div(N_n)
dc_pdt = -pybamm.div(N_p)
dQdt = I/3600

model.rhs = {
    c_s_n: dc_ndt,
    c_s_p: dc_pdt,
    Q : dQdt
}

model.initial_conditions = {
    c_s_n: c_n0,
    c_s_p: c_p0,
}

model.boundary_conditions = {
    c_s_n: {"left": (pybamm.Scalar(0), "Neumann"), "right": (-2*I/F*a_n*L_n*R_n, "Neumann")},
    c_s_p: {"left": (pybamm.Scalar(0), "Neumann"), "right": (2*I/F*a_p*L_p*R_p, "Neumann")},
}

# Surface concentrations
c_s_surf_n = pybamm.surf(c_s_n)
c_s_surf_p = pybamm.surf(c_s_p)# Max concentrations



sto_surf_n = c_s_surf_n / params["Maximum concentration in negative electrode [mol.m-3]"]
sto_surf_p = c_s_surf_p / params["Maximum concentration in positive electrode [mol.m-3]"]

U_n = (
    1.9793 * np.exp(-39.3631 * sto_surf_n)
    + 0.2482
    - 0.0909 * np.tanh(29.8538 * (sto_surf_n - 0.1234))
    - 0.04478 * np.tanh(14.9159 * (sto_surf_n - 0.2769))
    - 0.0205 * np.tanh(30.4444 * (sto_surf_n - 0.6103))
)
U_p = (
    1.9793 * np.exp(-39.3631 * sto_surf_p)
    + 0.2482
    - 0.0909 * np.tanh(29.8538 * (sto_surf_p - 0.1234))
    - 0.04478 * np.tanh(14.9159 * (sto_surf_p - 0.2769))
    - 0.0205 * np.tanh(30.4444 * (sto_surf_p - 0.6103))
)


model.events += [
        pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.01,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_n),
            ),
            pybamm.Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_surf_p) - 0.01,
            ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_p),
            ),
        ]


# Constants and temperature
R = pybamm.constants.R
T = pybamm.Parameter("Ambient temperature [K]")

# Overpotentials
eta_n = (2 * R * T / F) * pybamm.arcsinh(I / (2 * j_n_o*a_n*L_n))
eta_p = (2 * R * T / F) * pybamm.arcsinh(I / (2 * j_p_o*a_p*L_p))


# Broadcast scalar or particle-domain variables to electrode domains

phi_s_n = pybamm.Scalar(0)
phi_e = -eta_n - U_n  # broadcasted to "negative electrode"
phi_e_brodcast = pybamm.SecondaryBroadcast(phi_e, "negative electrode")
phi_s_p = eta_p + phi_e_brodcast + U_p

V = pybamm.x_average(phi_s_p)
# V = pybamm.boundary_value(phi_s_p, "right")  # optional


num_cells = pybamm.Parameter(
            "Number of cells connected in series to make a battery"
        )

whole_cell = ["negative electrode", "separator", "positive electrode"]


model.variables = {
            "Time [s]": pybamm.t,
            "Discharge capacity [A.h]": Q,
            "X-averaged negative particle concentration [mol.m-3]": c_s_n,
            "Negative particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                params["Initial concentration in electrolyte [mol.m-3]"], whole_cell
            ),
            "X-averaged positive particle concentration [mol.m-3]": c_s_p,
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Current variable [A]": I,  # for compatibility with pybamm.Experiment
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Voltage [V]": V,
            "Battery voltage [V]": V * num_cells,
        }


# Add missing or custom parameters
params.update({
    "Negative electrode diffusivity [m2.s-1]": 3.9e-14,
    "Positive electrode diffusivity [m2.s-1]": 1e-13,
    "Negative electrode reaction rate [m.s-1]": 1e-3,
    "Positive electrode reaction rate [m.s-1]": 1e-3,
    "Maximum concentration in negative electrode [mol.m-3]": 30500,
    "Maximum concentration in positive electrode [mol.m-3]": 51500,
    "Radius of negative electrode particle [m]": 1e-5,
    "Radius of positive electrode particle [m]": 1e-5,
    "Thickness of negative electrode [m]": 50e-6,
    "Thickness of positive electrode [m]": 50e-6,
    "Electrode width [m]": 0.1,
    "Electrode height [m]": 0.1,
    "Initial concentration in negative electrode [mol.m-3]": 0.8 * 30500,
    "Initial concentration in positive electrode [mol.m-3]": 0.2 * 51500,
}, check_already_exists=False)

# Geometry
r_n = pybamm.SpatialVariable("r_n", domain="negative particle")
r_p = pybamm.SpatialVariable("r_p", domain="positive particle")
geometry = {
    "negative particle": {r_n: {"min": pybamm.Scalar(0), "max": R_n}},
    "positive particle": {r_p: {"min": pybamm.Scalar(0), "max": R_p}},
}
params.process_geometry(geometry)

# Mesh and discretisation
submesh_types = {
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
}
var_pts = {"r_n": 20, "r_p": 20}
spatial_methods = {
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume(),
}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
params.process_model(model)

disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

# Plotting
if isinstance(solution, list):
    solution = solution[0]

if solution is None:
    raise RuntimeError("Solver did not return a solution.")

V_sol = solution["Terminal voltage [V]"]
c_n_surf_sol = solution["Surface concentration in negative particle"]
c_p_surf_sol = solution["Surface concentration in positive particle"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
ax1.plot(solution.t, V_sol(solution.t))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Voltage [V]")

ax2.plot(solution.t, c_n_surf_sol(solution.t), label="c_n_surf")
ax2.plot(solution.t, c_p_surf_sol(solution.t), label="c_p_surf")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Surface concentrations")
ax2.legend()

plt.tight_layout()
plt.show()
