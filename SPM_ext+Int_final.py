import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model = pybamm.BaseModel("Single Particle Model (Negative + Positive)")

# --------------------------
# Variables and spatial vars
# --------------------------
c_e_n = pybamm.Variable("Concentration of Negative electrode", domain="negative particle")
c_e_p = pybamm.Variable("Concentration of Positive electrode", domain="positive particle")

r_neg = pybamm.SpatialVariable("r_n", domain="negative particle")
r_pos = pybamm.SpatialVariable("r_p", domain="positive particle")



# --------------------------
# Physical parameters
# --------------------------
D_n = 3.9e-12      # m^2/s
R_n = 1e-3         # m
c_e0_n = 2.5e4     # mol/m^3
eps_s_n = 0.6
L_n = 10e-7
c_n_max = 3.6e4

D_p = 1.2e-14      # m^2/s
R_p = 1e-5         # m
c_e0_p = 2.4e2     # mol/m^3
eps_s_p = 0.5
L_p = 10e-9
c_p_max = 3.5e2

F = 96485.33289    # C/mol
R_const = 8.314    # J/mol·K
T = 298.15         # K
alpha = 0.5

k_p = 1e-11
k_n = 1e-11

E = 210e9
nu = 0.3
Omega = 3.497e-6

theta = (Omega / (R_const * T)) * ((2 * E) / 9 / (1 - nu))

# --------------------------
# Surface values (use boundary_value)
# --------------------------
c_surf_p = pybamm.boundary_value(c_e_p, "right")
c_surf_n = pybamm.boundary_value(c_e_n, "right")

j0_p = k_p * F * c_surf_p**alpha * (c_p_max - c_surf_p)**alpha
j0_n = k_n * F * c_surf_n**alpha * (c_n_max - c_surf_n)**alpha

I_app = pybamm.Scalar(1.0)

a_s_p = 3 * eps_s_p / R_p
a_s_n = 3 * eps_s_n / R_n

j_p = -I_app / (a_s_p * F * L_p)
j_n = I_app / (a_s_n * F * L_n)

eta_p = (R_const * T) / (alpha * F) * pybamm.arcsinh(j_p / (2 * j0_p))
eta_n = (R_const * T) / (alpha * F) * pybamm.arcsinh(j_n / (2 * j0_n))

def U_p(c): return pybamm.Scalar(4.2) - 0.1 * c / c_p_max
def U_n(c): return pybamm.Scalar(0.1) + 0.1 * c / c_n_max



# --------------------------
# External plate stress precomputation (unchanged)
# --------------------------

Lx, Ly, Lz = 0.2, 0.1, 0.05   # meters
nx, ny, nz = 10, 10, 10
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
z = np.linspace(0, Lz, nz)

# Material properties
E = 210e9       # Pa
nu = 0.3
lam = (E*nu) / ((1+nu)*(1-2*nu))   # Lamé constant λ
mu  = E / (2*(1+nu))               # Lamé constant μ

solver_mode = "SURROGATE"

# -----------------------------
# Example Force Field
# -----------------------------
def force_field(x, y, z):
    Fx = 1e6 * np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.sin(np.pi*z/Lz)
    Fy, Fz = 0.0, 0.0
    return np.array([Fx, Fy, Fz])


# -----------------------------
# Higher-order central difference (4th order)
# -----------------------------
def derivative(func, x, y, z, h, direction):
    if direction == 0:
        return (-func(x+2*h, y, z) + 8*func(x+h, y, z) - 8*func(x-h, y, z) + func(x-2*h, y, z)) / (12*h)
    elif direction == 1:
        return (-func(x, y+2*h, z) + 8*func(x, y+h, z) - 8*func(x, y-h, z) + func(x, y-2*h, z)) / (12*h)
    elif direction == 2:
        return (-func(x, y, z+2*h) + 8*func(x, y, z+h) - 8*func(x, y, z-h) + func(x, y, z-2*h)) / (12*h)


# -----------------------------
# Surrogate displacement field
# -----------------------------
def displacement_SURR(x, y, z):
    alpha = 1e-6
    ux = alpha * np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.sin(np.pi*z/Lz)
    uy = alpha * 0.5 * np.cos(np.pi*x/Lx) * np.sin(np.pi*y/Ly) * np.sin(np.pi*z/Lz)
    uz = alpha * 0.3 * np.sin(np.pi*x/Lx) * np.cos(np.pi*y/Ly) * np.sin(np.pi*z/Lz)
    return np.array([ux, uy, uz])



# -----------------------------
# Strain tensor (using analytical derivatives)
# -----------------------------
def strain_tensor_SURR(x, y, z, h):
    eps = np.zeros((3,3))
    alpha = 1e-6
    pi = np.pi
    dux_dx = alpha * (pi/Lx) * np.cos(pi*x/Lx) * np.sin(pi*y/Ly) * np.sin(pi*z/Lz)
    dux_dy = alpha * np.sin(pi*x/Lx) * (pi/Ly) * np.cos(pi*y/Ly) * np.sin(pi*z/Lz)
    dux_dz = alpha * np.sin(pi*x/Lx) * np.sin(pi*y/Ly) * (pi/Lz) * np.cos(pi*z/Lz)
    duy_dx = alpha * 0.5 * (-pi/Lx) * np.sin(pi*x/Lx) * np.sin(pi*y/Ly) * np.sin(pi*z/Lz)
    duy_dy = alpha * 0.5 * np.cos(pi*x/Lx) * (pi/Ly) * np.cos(pi*y/Ly) * np.sin(pi*z/Lz)
    duy_dz = alpha * 0.5 * np.cos(pi*x/Lx) * np.sin(pi*y/Ly) * (pi/Lz) * np.cos(pi*z/Lz)
    duz_dx = alpha * 0.3 * (pi/Lx) * np.cos(pi*x/Lx) * np.cos(pi*y/Ly) * np.sin(pi*z/Lz)
    duz_dy = alpha * 0.3 * np.sin(pi*x/Lx) * (-pi/Ly) * np.sin(pi*y/Ly) * np.sin(pi*z/Lz)
    duz_dz = alpha * 0.3 * np.sin(pi*x/Lx) * np.cos(pi*y/Ly) * (pi/Lz) * np.cos(pi*z/Lz)

    eps[0,0] = dux_dx
    eps[1,1] = duy_dy
    eps[2,2] = duz_dz
    eps[0,1] = 0.5*(dux_dy + duy_dx)
    eps[0,2] = 0.5*(dux_dz + duz_dx)
    eps[1,2] = 0.5*(duy_dz + duz_dy)
    eps = eps + eps.T - np.diag(np.diag(eps))
    return eps



# -----------------------------
# Stress tensor (Hooke’s law)
# -----------------------------
def stress_tensor_SURR(x, y, z, h):
    eps = strain_tensor_SURR(x, y, z, h)
    tr_eps = np.trace(eps)
    return lam*tr_eps*np.eye(3) + 2*mu*eps


# -----------------------------
# Equilibrium residual check
# -----------------------------
def equilibrium_residual(x, y, z, h):
    sigma = stress_tensor_SURR(x,y,z,h)
    div_sigma = np.zeros(3)
    for i in range(3):
        for j in range(3):
            sigma_comp = lambda X,Y,Z: stress_tensor_SURR(X,Y,Z,h)[i,j]
            if j == 0:
                div_sigma[i] += derivative(sigma_comp, x,y,z,h,0)
            elif j == 1:
                div_sigma[i] += derivative(sigma_comp, x,y,z,h,1)
            elif j == 2:
                div_sigma[i] += derivative(sigma_comp, x,y,z,h,2)
    f = force_field(x,y,z)
    return -div_sigma - f




# -----------------------------
# Main computation
# -----------------------------
results = []
h = x[1]-x[0]

σ1_field = np.zeros((nx,ny,nz))
σ2_field = np.zeros((nx,ny,nz))
σ3_field = np.zeros((nx,ny,nz))
σvm_field = np.zeros((nx,ny,nz))

for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            sigma = stress_tensor_SURR(x[i], y[j], z[k], h)
            vals, _ = np.linalg.eigh(sigma)
            stresses = np.sort(vals)[::-1]
            σ1_field[i,j,k], σ2_field[i,j,k], σ3_field[i,j,k] = stresses
            vm_stress = np.sqrt(0.5*((stresses[0]-stresses[1])**2 +
                                     (stresses[1]-stresses[2])**2 +
                                     (stresses[2]-stresses[0])**2))
            σvm_field[i,j,k] = vm_stress
            eps = strain_tensor_SURR(x[i], y[j], z[k], h)
            energy_density = 0.5*np.sum(sigma*eps)
            res = equilibrium_residual(x[i], y[j], z[k], h)
            res_norm = np.linalg.norm(res)
            results.append([x[i],y[j],z[k],*stresses,np.mean(stresses),vm_stress,energy_density,res_norm])


sigma_h = (σ1_field + σ2_field + σ3_field) / 3.0
# df_sph["σ_h"] = sigma_h

# Reshape into 3D grid
sigma_h_field = sigma_h.reshape((nx, ny, nz))

# Gradient of σ_h
grad_sigma_h = np.gradient(sigma_h_field, x, y, z, edge_order=2)
grad_sigma_h_mag = np.sqrt(
    grad_sigma_h[0]**2 + grad_sigma_h[1]**2 + grad_sigma_h[2]**2
)
#print(grad_sigma_h)
# Divergence (Laplacian)
div_sigma_h = (
    np.gradient(grad_sigma_h[0], x, axis=0, edge_order=2) +
    np.gradient(grad_sigma_h[1], y, axis=1, edge_order=2) +
    np.gradient(grad_sigma_h[2], z, axis=2, edge_order=2)
)

#print(div_sigma_h)
print("\n✅ Computed σ_h, its gradient, and divergence")



#df_sph.to_csv("principal_stresses_spherical.csv", index=False)
print("✅ Updated 'principal_stresses_spherical.csv' with σ_h, |∇σ_h|, and ∇·(∇σ_h)")




# --------------------------
# Stresses from concentration (symbolic)
# --------------------------
c_tilda_n = c_e_n - c_e0_n
int_0_to_Rn = pybamm.Integral(r_neg**2 * c_tilda_n, r_neg)     
int_0_to_r_n = pybamm.Integral(r_neg**2 * c_tilda_n, r_neg)     # partial integral (same symbol used)

sigma_r_n = (2 * Omega * E) / (3 * (1 - nu)) * (
    (1 / r_neg**3) * int_0_to_r_n - (1 / r_neg**3) * int_0_to_Rn
)
sigma_t_n = (Omega * E) / (3 * (1 - nu)) * (
    (2 / r_neg**3) * int_0_to_Rn + (1 / r_neg**3) * int_0_to_r_n - c_tilda_n
)
sigma_h_n = (Omega * E) / (9 * (1 - nu)) * ((3 / r_neg**3) * int_0_to_Rn - c_tilda_n)

c_tilda_p = c_e_p - c_e0_p
int_0_to_Rp = pybamm.Integral(r_pos**2 * c_tilda_p, r_pos)
int_0_to_r_p = pybamm.Integral(r_pos**2 * c_tilda_p, r_pos)

sigma_r_p = (2 * Omega * E) / (3 * (1 - nu)) * (
    (1 / r_pos**3) * int_0_to_r_p - (1 / r_pos**3) * int_0_to_Rp
)
sigma_t_p = (Omega * E) / (3 * (1 - nu)) * (
    (2 / r_pos**3) * int_0_to_Rp + (1 / r_pos**3) * int_0_to_r_p - c_tilda_p
)
sigma_h_p = (Omega * E) / (9 * (1 - nu)) * ((3 / r_pos**3) * int_0_to_Rp - c_tilda_p)

sigma_h_n_int = sigma_h_n
sigma_h_p_int = sigma_h_p

# Build time-dependent external hydrostatic stress interpolant

sigma_h_n_surf_total = pybamm.boundary_value(sigma_h_n, "right") + pybamm.Scalar(sigma_h_field[0,0,0])
sigma_h_p_surf_total = pybamm.boundary_value(sigma_h_p, "right") + pybamm.Scalar(sigma_h_field[0,0,0])

U_p_eff = U_p(c_surf_p) - (Omega / F) * sigma_h_p_surf_total
U_n_eff = U_n(c_surf_n) - (Omega / F) * sigma_h_n_surf_total

voltage = U_p_eff - U_n_eff + eta_p - eta_n

# --------------------------
# Geometry, mesh, discretisation (set var_pts consistently)
# --------------------------
geometry = {
    "negative particle": {"r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R_n)}},
    "positive particle": {"r_p": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R_p)}},
}

submesh_types = {
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
}

var_pts = {"r_n": 21, "r_p": 21}   # consistent number of cell centres
spatial_methods = {
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume(),
}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)

# --------------------------



def polynomial_interpolant(variable, spatial_var, disc, N=5):
    """
    Interpolates a PyBaMM variable from cell centers to edges using
    a high-degree polynomial fit. PyBaMM-compatible.
    """
    mesh = disc.mesh[spatial_var.domain[0]]
    nodes = mesh.nodes[0]
    edges = mesh.edges[0]

    # Process and evaluate initial conditions properly
    y0 = disc.process_initial_conditions(model.initial_conditions)
    y0_eval = y0.evaluate()

    # Evaluate the variable at t=0 and y=y0
    y = variable.evaluate(t=0, y=y0_eval)

    deg = min(N, len(nodes) - 1)
    coeffs = np.polyfit(nodes, y[:len(nodes)], deg=deg)
    y_edges = np.polyval(coeffs, edges)

    interpolant = pybamm.Interpolant(edges, y_edges, spatial_var)
    return interpolant






# Flux definitions: use PrimaryBroadcast to map center -> edges
# --------------------------
# Broadcast centre concentrations to edges (symbolic, safe)
# Interpolate to edges (not needed for now)
c_e_n_edge = polynomial_interpolant(c_e_n, r_neg, disc, N=5)
c_e_p_edge = polynomial_interpolant(c_e_p, r_pos, disc, N=5)
# Fluxes (all terms now live on edges)
J_n = D_n * pybamm.grad(c_e_n) - (D_n * Omega * c_e_n_edge / (R_const * T)) * pybamm.grad(sigma_h_n_int + pybamm.Scalar(sigma_h_field[0,0,0]))
J_p = D_p * pybamm.grad(c_e_p) - (D_p * Omega * c_e_p_edge / (R_const * T)) * pybamm.grad(sigma_h_p_int + pybamm.Scalar(sigma_h_field[0,0,0]))

# Back to centres
dc_e_n_dt = -pybamm.div(J_n)
dc_e_p_dt = -pybamm.div(J_p)

model.rhs = {c_e_n: dc_e_n_dt,
             c_e_p: dc_e_p_dt}

# Discharge capacities (unchanged)
Q_n = pybamm.Scalar(F * eps_s_n * L_n) * (pybamm.Scalar(c_n_max) - c_e_n)
Q_p = pybamm.Scalar(F * eps_s_p * L_p) * (pybamm.Scalar(c_p_max) - c_e_p)

# Initial conditions
model.initial_conditions = {
    c_e_n: pybamm.PrimaryBroadcast(c_e0_n, "negative particle"),
    c_e_p: pybamm.PrimaryBroadcast(c_e0_p, "positive particle"),
}

# Boundary conditions (use boundary_value for c_surf)
grad_surf_n = -j_n / (F * D_n * (1 + theta * c_surf_n))
grad_surf_p = -j_p / (F * D_p * (1 + theta * c_surf_p))

model.boundary_conditions = {
    c_e_n: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (grad_surf_n, "Neumann")
    },
    c_e_p: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (grad_surf_p, "Neumann")
    }
}

# Variables to output
model.variables = {
    "Concentration of Negative electrode": c_e_n,
    "Rate of change of concentration in negative electrode": dc_e_n_dt,
    "Discharge capacity (negative) [A.h]": Q_n,
    "Total lithium in negative electrode [mol]": pybamm.Scalar(F * eps_s_n * L_n) * (pybamm.Scalar(c_n_max) - c_e_n),
    "Total lithium capacity (negative) [A.h]": pybamm.Scalar(F * eps_s_n * L_n * c_n_max),
    "Negative electrode radial stress": sigma_r_n,
    "Negative electrode tangential stress": sigma_t_n,
    "Negative electrode hydrostatic stress": sigma_h_n,
    "External Hydrostatic Stress": sigma_h,

    "Concentration of Positive electrode": c_e_p,
    "Rate of change of concentration in positive electrode": dc_e_p_dt,
    "Discharge capacity (positive) [A.h]": Q_p,
    "Total lithium in positive electrode [mol]": pybamm.Scalar(F * eps_s_p * L_p) * (pybamm.Scalar(c_p_max) - c_e_p),
    "Total lithium capacity (positive) [A.h]": pybamm.Scalar(F * eps_s_p * L_p * c_p_max),
    "Positive electrode overpotential [V]": eta_p,
    "Negative electrode overpotential [V]": eta_n,
    "Positive electrode radial stress": sigma_r_p,
    "Positive electrode tangential stress": sigma_t_p,
    "Positive electrode hydrostatic stress": sigma_h_p,
    "Terminal voltage [V]": voltage,
}

# --------------------------
# Discretise and solve
# --------------------------
model_disc = disc.process_model(model)
solver = pybamm.ScipySolver()

t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

# extract and plot (same pattern as before)
if isinstance(solution, list):
    solution = solution[0]
if solution is None:
    raise RuntimeError("Solver did not return a solution.")

c_e_n_sol = solution["Concentration of Negative electrode"]
c_e_p_sol = solution["Concentration of Positive electrode"]

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0,0].set_title("Negative Electrode Surface Concentration vs Time")
axs[0,0].plot(solution.t, c_e_n_sol(solution.t, r=R_n))
axs[0,0].set_xlabel("Time [s]"); axs[0,0].set_ylabel("Concentration at surface (r = R) [mol/m^3]")

axs[0,1].set_title("Negative Electrode Concentration at t = 1000 s")
r_n_vals = np.linspace(0, R_n, 100)
axs[0,1].plot(r_n_vals, c_e_n_sol(1000, r=r_n_vals))
axs[0,1].set_xlabel("r_n"); axs[0,1].set_ylabel("Concentration at t = 1000 s [mol/m^3]")

axs[1,0].set_title("Positive Electrode Surface Concentration vs Time")
axs[1,0].plot(solution.t, c_e_p_sol(solution.t, r=R_p))
axs[1,0].set_xlabel("Time [s]"); axs[1,0].set_ylabel("Concentration at surface (r = R) [mol/m^3]")

axs[1,1].set_title("Positive Electrode Concentration at t = 1000 s")
r_p_vals = np.linspace(0, R_p, 100)
axs[1,1].plot(r_p_vals, c_e_p_sol(1000, r=r_p_vals))
axs[1,1].set_xlabel("r_p"); axs[1,1].set_ylabel("Concentration at t = 1000 s [mol/m^3]")

plt.tight_layout()
plt.show()

# Plot extra variables
solution.plot(output_variables=[
    "Total lithium in negative electrode [mol]",
    "Total lithium capacity (negative) [A.h]",
    "Discharge capacity (negative) [A.h]",
    "Total lithium in positive electrode [mol]",
    "Total lithium capacity (positive) [A.h]",
    "Discharge capacity (positive) [A.h]",
    "Terminal voltage [V]",
    "Positive electrode overpotential [V]",
    "Negative electrode overpotential [V]"]
)
