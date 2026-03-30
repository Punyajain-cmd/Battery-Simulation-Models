import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel("Single Particle Model (Negative + Positive)")

# Define variables
c_e_n = pybamm.Variable("Concentration of Negative electrode", domain="negative particle")
c_e_p = pybamm.Variable("Concentration of Positive electrode", domain="positive particle")


# 1. Define spatial variable
r_neg = pybamm.SpatialVariable("r_n", domain="negative particle")
r_pos = pybamm.SpatialVariable("r_p", domain="positive particle")

# Parameters for negative electrode
D_n = 3.9e-12      # m^2/s
R_n = 1e-3         # m
c_e0_n = 2.5e4     # mol/m^3
eps_s_n = 0.6
L_n = 10e-7
c_n_max = 3.6e4



# Parameters for positive electrode
D_p = 1.2e-14      # m^2/s
R_p = 1e-5         # m
c_e0_p = 2.4e2     # mol/m^3
eps_s_p = 0.5
L_p = 10e-9
c_p_max = 3.5e2



# Constants
F = 96485.33289    # C/mol
R_const = 8.314  # J/mol·K
T = 298.15       # K
alpha = 0.5      # typical value

k_p = 1e-11      # example reaction rate constants (adjust)
k_n = 1e-11


E = 210e9                       
nu = 0.3                          
Omega = 3.497e-6      

theta = (Omega / (R_const * T)) * ((2 * E) / 9 / (1 - nu))                # θ = (Ω/RT)[(2E)/9(1−ν)]


c_surf_p = pybamm.boundary_value(c_e_p, "right")
c_surf_n = pybamm.boundary_value(c_e_n, "right")

j0_p = k_p * F * c_surf_p**alpha * (c_p_max - c_surf_p)**alpha
j0_n = k_n * F * c_surf_n**alpha * (c_n_max - c_surf_n)**alpha

I_app = pybamm.Scalar(1.0)  # Applied current in A

# surface area per unit volume (approximate): a_s = 3 * eps_s / R
a_s_p = 3 * eps_s_p / R_p
a_s_n = 3 * eps_s_n / R_n

j_p = -I_app / (a_s_p * F * L_p)
j_n = I_app / (a_s_n * F * L_n)

eta_p = (R_const * T) / (alpha * F) * pybamm.arcsinh(j_p / (2 * j0_p))
eta_n = (R_const * T) / (alpha * F) * pybamm.arcsinh(j_n / (2 * j0_n))

def U_p(c): return pybamm.Scalar(4.2) - 0.1 * c / c_p_max
def U_n(c): return pybamm.Scalar(0.1) + 0.1 * c / c_n_max

voltage = U_p(c_surf_p) - U_n(c_surf_n) + eta_p - eta_n





# --- Material and geometric properties ---
h = 0.1        # Plate thickness (m)
D = E * h**3 / (12 * (1 - nu**2))  # Flexural rigidity
a = 1.0         # Plate length (square plate: a x a, in meters)

# --- Time properties ---
p0 = 1000       # Pressure amplitude (Pa)
f = 1           # Frequency in Hz
omega = 2 * np.pi * f
t_max = 2       # Total simulation time (s)
Nt = 10        # Number of time steps
time = np.linspace(0, t_max, Nt)

# --- Grid setup ---
N = 10
x = np.linspace(0, a, N)
y = np.linspace(0, a, N)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]
Mz = 100  # Number of points along plate thickness
z_values = np.linspace(-h/2, h/2, Mz)  # from bottom face to top face

# --- Precompute all time-dependent fields ---
σx_list = []
σy_list = []
σx_x_list = []
σx_y_list = []
σy_x_list = []
σy_y_list = []


Mx_list = []
My_list = []


for t in time:
    
    sigma_x_stack = []  # shape: (z, y, x)
    sigma_y_stack = []
    sigma_x_x_stack = []
    sigma_x_y_stack = []
    sigma_y_x_stack = []
    sigma_y_y_stack = []



    for z_val in z_values:
    

        p_t = p0 * np.cos(omega * t)
        w = (p_t * a*4 / (D * np.pi*6)) * (
        np.sin(np.pi * X / a) * np.sin(np.pi * Y / a)
        - (1 / 9) * np.sin(3 * np.pi * X / a) * np.sin(np.pi * Y / a)
        - (1 / 9) * np.sin(np.pi * X / a) * np.sin(3 * np.pi * Y / a)
        )

        w_xx = np.gradient(np.gradient(w, dx, axis=1), dx, axis=1)
        w_yy = np.gradient(np.gradient(w, dy, axis=0), dy, axis=0)
    
        Mx = -D * (w_xx + nu * w_yy)
        My = -D * (nu * w_xx + w_yy)
        sigma_x = (12 * Mx / h**3) * z_val
        sigma_y = (12 * My / h**3) * z_val

    # Spatial gradients
        sigma_x_x = np.gradient(sigma_x, dx, axis=1)
        sigma_x_y = np.gradient(sigma_x, dy, axis=0)
        sigma_y_x = np.gradient(sigma_y, dx, axis=1)
        sigma_y_y = np.gradient(sigma_y, dy, axis=0)

        sigma_x_stack.append(sigma_x)
        sigma_y_stack.append(sigma_y)
        sigma_x_x_stack.append(sigma_x_x)
        sigma_x_y_stack.append(sigma_x_y)
        sigma_y_x_stack.append(sigma_y_x)
        sigma_y_y_stack.append(sigma_y_y)
            # After inner z-loop, stack and append to time-wise lists
            
    σx_list.append(np.array(sigma_x_stack))      # (Mz, N, N)
    σy_list.append(np.array(sigma_y_stack))
    σx_x_list.append(np.array(sigma_x_x_stack))
    σx_y_list.append(np.array(sigma_x_y_stack))
    σy_x_list.append(np.array(sigma_y_x_stack))
    σy_y_list.append(np.array(sigma_y_y_stack))



# Convert lists to 4D arrays
σx_all = np.array(σx_list)        # shape: (Nt, Mz, N, N)
σy_all = np.array(σy_list)
σx_x_all = np.array(σx_x_list)
σx_y_all = np.array(σx_y_list)
σy_x_all = np.array(σy_x_list)
σy_y_all = np.array(σy_y_list)



# --- 1. Define r (magnitude) and avoid divide-by-zero ---
r = np.sqrt(X**2 + Y**2 + z_values[:, np.newaxis, np.newaxis]**2)
r[r == 0] = 1e-12
 # prevent division by zero at origin

# --- 2. Compute radial projection of Cartesian stress gradients ---
# For σx
grad_sigma_x_r = (X/r) * σx_x_all + (Y/r) * σx_y_all
# For σy
grad_sigma_y_r = (X/r) * σy_x_all + (Y/r) * σy_y_all


# --- 3. Add both gradients ---
grad_cartesian_r_total = grad_sigma_x_r + grad_sigma_y_r

# Use np.gradient: returns same shape as input
dfx_dx = np.gradient(grad_sigma_x_r, dx, axis=3)
dfy_dy = np.gradient(grad_sigma_y_r, dy, axis=2)

# Total divergence
divergence = dfx_dx + dfy_dy  # shape: (Nt, Mz, Nx, Ny)

#for 1D
grad_r_data = grad_cartesian_r_total[:, -1, N//2, N//2]
divergence_data = divergence[:, -1, N//2, N//2] 




# Negative electrode stresses


c_tilda_n = c_e_n - c_e0_n                                           # ∼c = c - c₀
# --- Define integrals with c_tilda ---
# Full volume integral over [0, r₀] of r^2 * c_tilda(r)
int_0_to_Rn = pybamm.Integral(r_neg**2 * c_tilda_n, r_neg)

# Partial integral over [0, r] of r^2 * c_tilda(r)
int_0_to_r_n = pybamm.Integral(r_neg**2 * c_tilda_n, r_neg)


# --- Equation 22: Radial stress ---
sigma_r_n = (2 * Omega * E) / (3 * (1 - nu)) * (
    (1 / r_neg**3) * int_0_to_r_n - (1 / r_neg**3) * int_0_to_Rn
)

# --- Equation 23: Hoop stress ---
sigma_t_n = (Omega * E) / (3 * (1 - nu)) * (
    (2 / r_neg**3) * int_0_to_Rn + (1 / r_neg**3) * int_0_to_r_n - c_tilda_n
)

# --- Equation 25: Hydrostatic stress ---
sigma_h_n = (Omega * E) / (9 * (1 - nu)) * ((3 / r_neg**3) * int_0_to_Rn - c_tilda_n)




# Positive electrode stresses

c_tilda_p = c_e_p - c_e0_p                                           # ∼c = c - c₀
# --- Define integrals with c_tilda ---
# Full volume integral over [0, r₀] of r^2 * c_tilda(r)
int_0_to_Rp = pybamm.Integral(r_pos**2 * c_tilda_p, r_pos)

# Partial integral over [0, r] of r^2 * c_tilda(r)
int_0_to_r_p = pybamm.Integral(r_pos**2 * c_tilda_p, r_pos)


# --- Equation 22: Radial stress ---
sigma_r_p = (2 * Omega * E) / (3 * (1 - nu)) * (
    (1 / r_pos**3) * int_0_to_r_p - (1 / r_pos**3) * int_0_to_Rp
)

# --- Equation 23: Hoop stress ---
sigma_t_p = (Omega * E) / (3 * (1 - nu)) * (
    (2 / r_pos**3) * int_0_to_Rp + (1 / r_pos**3) * int_0_to_r_p - c_tilda_p
)

# --- Equation 25: Hydrostatic stress ---
sigma_h_p = (Omega * E) / (9 * (1 - nu)) * ((3 / r_pos**3) * int_0_to_Rp - c_tilda_p)


sigma_h_n_int = sigma_h_n
sigma_h_p_int = sigma_h_p

sigma_h_ext_all = (σx_all + σy_all) / 3.0
sigma_h_ext_series = sigma_h_ext_all[:, -1, N//2, N//2]
sigma_h_ext = pybamm.Interpolant(time, sigma_h_ext_series, pybamm.t, name="External Hydrostatic Stress")

sigma_h_n_surf_total = pybamm.boundary_value(sigma_h_n, "right") + sigma_h_ext
sigma_h_p_surf_total = pybamm.boundary_value(sigma_h_p, "right") + sigma_h_ext

# The stress affects the open-circuit potential (OCP)
# U_stress = - (Ω/F) * σ_h
U_p_eff = U_p(c_surf_p) - (Omega / F) * sigma_h_p_surf_total
U_n_eff = U_n(c_surf_n) - (Omega / F) * sigma_h_n_surf_total

# Redefine the terminal voltage with the stress-modified OCPs
voltage = U_p_eff - U_n_eff + eta_p - eta_n



# Assume r_n is your spatial variable (cell centers) and N points
r_n = pybamm.SpatialVariable("r_n", domain="negative particle")
r_p = pybamm.SpatialVariable("r_p", domain="positive particle")



def polynomial_interpolant(geometry, submesh_types, var_pts, variable, spatial_var, N=5):
    """
    Build a polynomial interpolant of a PyBaMM variable to match grad shape.
    Extends variable to N+1 points (from N).
    """
    # Get discretisation mesh for the spatial variable
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    nodes = mesh.nodes[0]  # cell centers
    edges = mesh.edges[0]  # edges (len = len(nodes)+1)

    # Create numpy interpolation function (polynomial fit)
    # This works at "discretisation-time"
    def interp_fun(x, y):
        coeffs = np.polyfit(x, y, deg=min(N, len(x)-1))
        return np.polyval(coeffs, edges)

    disc.process_model(model)
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    nodes = mesh.nodes[0]  # cell centers
    edges = mesh.edges[0]  # edges (len = len(nodes)+1)
    interpolant = pybamm.Interpolant(nodes, interp_fun(nodes, np.ones_like(nodes)), spatial_var)
    return interpolant * variable / variable  # keep symbolic dependency


J_n = D_n * pybamm.grad(c_e_n) - (D_n * Omega * c_e_n_edge / (R_const * T)) * pybamm.grad(sigma_h_n_int + sigma_h_ext)
J_p = D_p * pybamm.grad(c_e_p) - (D_p * Omega * c_e_p_edge / (R_const * T)) * pybamm.grad(sigma_h_p_int + sigma_h_ext)


# Ensure sigma_h terms are symbols defined on the particle domain (so grad(...) -> edges)
# (sigma_h_n_int / sigma_h_p_int are your internal hydrostatic stress symbols)
# sigma_h_ext should be an Interpolant or symbol defined on time (broadcast safe)


# Back to cell centres
dc_e_n_dt = -pybamm.div(J_n)
dc_e_p_dt = -pybamm.div(J_p)

model.rhs = {c_e_n: dc_e_n_dt,
             c_e_p: dc_e_p_dt}



# Discharge capacities
Q_n = pybamm.Scalar(F * eps_s_n * L_n) * (pybamm.Scalar(c_n_max) - c_e_n)
Q_p = pybamm.Scalar(F * eps_s_p * L_p) * (pybamm.Scalar(c_p_max) - c_e_p)


# Initial conditions
model.initial_conditions = {
    c_e_n: pybamm.PrimaryBroadcast(c_e0_n, "negative particle"),
    c_e_p: pybamm.PrimaryBroadcast(c_e0_p, "positive particle"),
}


# --- Boundary conditions also need to be updated ---
# The boundary flux 'j' depends on the overpotential, which is now stress-dependent.
# We must use the new voltage definition to correctly link the flux to the applied current.
# The simplest way to ensure this coupling is to define the boundary flux 'j'
# implicitly through the model variables.

# 7. Boundary conditions (Eq. 27)
# J = -D(1 + θc) ∂c/∂r = i_n / F at r = r_0
grad_surf_n = -j_n / (F * D_n * (1 + theta * c_surf_n))
grad_surf_p = -j_p / (F * D_p * (1 + theta * c_surf_p))

model.boundary_conditions = {
    c_e_n: {
        "left": (pybamm.Scalar(0), "Neumann"),  # symmetry at r = 0
        "right": (grad_surf_n, "Neumann")          # imposed current flux at r = r_0
    },
    c_e_p: {
        "left": (pybamm.Scalar(0), "Neumann"),  # symmetry at r = 0
        "right": (grad_surf_p, "Neumann")          # imposed current flux at r = r_0
    }
}




# Model variables
model.variables = {
    # Negative electrode
    "Concentration of Negative electrode": c_e_n,
    # "Flux of Lithium ions in Negative electrode": dc_e_n_dr,
    "Rate of change of concentration in negative electrode": dc_e_n_dt,
    "Discharge capacity (negative) [A.h]": Q_n,
    "Total lithium in negative electrode [mol]": pybamm.Scalar(F * eps_s_n * L_n) * (pybamm.Scalar(c_n_max) - c_e_n),
    "Total lithium capacity (negative) [A.h]": pybamm.Scalar(F * eps_s_n * L_n * c_n_max),
    "Negative electrode radial stress": sigma_r_n,
    "Negative electrode tangential stress": sigma_t_n,
    "Negative electrode hydrostatic stress": sigma_h_n,
    "External Hydrostatic Stress": sigma_h_ext,
    # Positive electrode
    "Concentration of Positive electrode": c_e_p,
    # "Flux of Lithium ions in Positive electrode": dc_e_p_dr,
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

# Geometry
geometry = {
    "negative particle": {"r_n": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R_n)}},
    "positive particle": {"r_p": {"min": pybamm.Scalar(0), "max": pybamm.Scalar(R_p)}},
}

# --- Mesh & discretisation ---
submesh_types = {
    "negative particle": pybamm.Uniform1DSubMesh,
    "positive particle": pybamm.Uniform1DSubMesh,
}

# Make sure both domains have consistent number of discretisation points:
# (choose Nx so the finite-volume submesh yields consistent node counts)
var_pts = {"r_n": 21, "r_p": 21}   # use same values for both domains

spatial_methods = {
    "negative particle": pybamm.FiniteVolume(),
    "positive particle": pybamm.FiniteVolume(),
}

mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model)

# Take the discretised variable
c_e_n_disc = model.variables["Concentration of Negative particle"]
c_e_p_disc = model.variables["Concentration of Positive particle"]

# Interpolate to edges
c_e_n_edge = polynomial_interpolant(c_e_n_disc, r_n, disc, N=5)
c_e_p_edge = polynomial_interpolant(c_e_p_disc, r_p, disc, N=5)

# Solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 100)
solution = solver.solve(model, t)

# Check and extract
if isinstance(solution, list):
    solution = solution[0]
if solution is None:
    raise RuntimeError("Solver did not return a solution.")

# Extract variables
c_e_n_sol = solution["Concentration of Negative electrode"]
c_e_p_sol = solution["Concentration of Positive electrode"]



fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0,0].set_title("Negative Electrode Surface Concentration vs Time")
axs[0,0].plot(solution.t, c_e_n_sol(solution.t, r=R_n))
axs[0,0].set_xlabel("Time [s]")
axs[0,0].set_ylabel("Concentration at surface (r = R) [mol/m^3]")

axs[0,1].set_title("Negative Electrode Concentration at t = 1000 s")
r_n_vals = np.linspace(0, R_n, 100)
axs[0,1].plot(r_neg, c_e_n_sol(1000, r=r_n_vals))
axs[0,1].set_xlabel("r_n")
axs[0,1].set_ylabel("Concentration at t = 1000 s [mol/m^3]")


axs[1,0].set_title("Positive Electrode Surface Concentration vs Time")
axs[1,0].plot(solution.t, c_e_p_sol(solution.t, r=R_p))
axs[1,0].set_xlabel("Time [s]")
axs[1,0].set_ylabel("Concentration at surface (r = R) [mol/m^3]")


axs[1,1].set_title("Positive Electrode Concentration at t = 1000 s")
r_p_vals = np.linspace(0, R_p, 100)
axs[1,1].plot(r_p_vals, c_e_p_sol(1000, r=r_p_vals))
axs[1,1].set_xlabel("r_p")
axs[1,1].set_ylabel("Concentration at t = 1000 s [mol/m^3]")



plt.tight_layout()
plt.show()




# Plot extra variables from solution
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
