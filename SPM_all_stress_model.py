import pybamm
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# MODEL CONFIGURATION
# -----------------------------
model = pybamm.lithium_ion.SPM(
    options={
        #"SEI": "ec reaction limited",
        #"SEI porosity change": "true",
        #"lithium plating": "reversible",
        #"loss of active material": "stress-driven",
        #"thermal": "lumped",
    }
)

# -----------------------------
# PARAMETER UPDATES (Reniers 2019)
# -----------------------------
param = model.default_parameter_values
param.update({
    "SEI kinetic rate constant [m.s-1]": 1e-13,
    "Activation energy for SEI kinetics [J.mol-1]": 130000,
    "SEI diffusion coefficient [m2.s-1]": 1.125e-14,
    "Activation energy for SEI diffusion [J.mol-1]": 20000,
    "Reference LAM stress [Pa]": 1e8,
    "Activation energy for lithium plating [J.mol-1]": 201400,
    "Initial temperature [K]": 298.15,
    "Typical plated lithium concentration [mol.m-3]": 10,
    "Initial plated lithium concentration [mol.m-3]": 0.1,
    "Lithium metal partial molar volume [m3.mol-1]": 1.3e-05,
    "Exchange-current density for stripping [A.m-2]": 2.25e-10,
    "Exchange-current density for plating [A.m-2]": 2.25e-10,
    "Lithium plating transfer coefficient": 0.5,

    # LAM mechanical params
    "Negative electrode Young's modulus [Pa]": 1e11,
    "Negative electrode porosity": 0.8,          # if needed (default ~0.3)
    "Negative electrode critical stress [Pa]": 375e6,
    "Negative electrode partial molar volume [m3.mol-1]": 3.1e-6,
    "Negative electrode reference concentration for free of deformation [mol.m-3]": 30000,
    "Negative electrode LAM constant proportional term [s-1]": 8e-7,
    "Negative electrode LAM constant exponential term": 1.25,
}, check_already_exists=False)

# -----------------------------
# EXPERIMENT DEFINITION
# -----------------------------
experiment = pybamm.Experiment([
    "Charge at 0.5 C until 4.2 V",
    "Hold at 4.2 V until C/20",
    "Rest for 30 minutes",
    "Discharge at 0.5 C until 2.5 V",
    "Rest for 30 minutes"
] * 500)




# Simulation
solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)
sim = pybamm.Simulation(model, parameter_values=param, experiment=experiment, solver=solver)
sim.solve(model)
# Plot capacity fade over cycles
sim.plot(output_variables=[
    "Total lithium capacity [A.h]",
    #"Loss of active material in negative electrode [%]",
    #"Positive SEI thickness [m]",
    #"Negative SEI thickness [m]",
    #"X-averaged positive electrode resistance [Ohm.m2]",
    #"Positive lithium plating concentration [mol.m-3]",
    #"Negative lithium plating concentration [mol.m-3]",
    #"Cell temperature [K]",
    #"Voltage [V]",
    #"Current [A]",
    #"Discharge capacity [A.h]",
    #"Negative electrode porosity",
    #"X-averaged negative particle stress [Pa]",
])
