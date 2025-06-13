# This code is taken from a chapter in The FEniCS Book:
#   https://fenicsproject.org/book/),
# which is available for free online. That chapter an implementation of a model
# setup from van Keken et al (1997):
#   https://doi.org/10.1029/97JB01353).
# I took the thermomechanical parts and removed the chemistry.

import argparse
import numpy as np
from numpy import pi as π
try:
    import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
from petsc4py import PETSc
import firedrake
from firedrake import (
    Constant,
    sqrt,
    exp,
    min_value,
    max_value,
    jump,
    avg,
    inner,
    sym,
    grad,
    div,
    dx,
    ds,
    dS,
)
import irksome
from irksome import Dt


# Get command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--output-filename", default="mantle.h5")
parser.add_argument("--log-filename", default="mantle.log")
parser.add_argument("--solution-method", choices=["split", "monolithic"])
parser.add_argument("--num-cells", type=int, default=32)
parser.add_argument("--temperature-basis", choices=["cg", "dg"])
parser.add_argument("--temperature-degree", type=int, default=1)
parser.add_argument("--cfl-fraction", type=float, default=1.0)
parser.add_argument("--final-time", type=float, default=0.25)
args = parser.parse_args()

solution_method = args.solution_method
num_cells = args.num_cells
basis = args.temperature_basis.upper()
degree = args.temperature_degree

# Make the mesh and some function spaces
Lx, Ly = Constant(2.0), Constant(1.0)
num_cells_x = int(float(Lx / Ly)) * num_cells
mesh = firedrake.RectangleMesh(
    num_cells_x, num_cells, float(Lx), float(Ly), diagonal="crossed"
)

pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
element = firedrake.FiniteElement(basis, "triangle", degree)
temperature_space = firedrake.FunctionSpace(mesh, element)

if solution_method == "split":
    Z = velocity_space * pressure_space
elif solution_method == "monolithic":
    Z = velocity_space * pressure_space * temperature_space


# Create the initial temperature field; the expression is heinous but if you
# plot it you'll see what it's supposed to do.
def clamp(z, zmin, zmax):
    return min_value(Constant(zmax), max_value(Constant(zmin), z))

def switch(z):
    return exp(z) / (exp(z) + exp(-z))

Ra = Constant(1e6)
x = firedrake.SpatialCoordinate(mesh)

def initial_temperature(x):
    ϵ = Constant(1 / num_cells_x)
    q = Lx**(7 / 3) / (1 + Lx**4)**(2 / 3) * (Ra / (2 * np.sqrt(π)))**(2/3)
    Q = 2 * sqrt(Lx / (π * q))
    T_u = 0.5 * switch((1 - x[1]) / 2 * sqrt(q / (x[0] + ϵ)))
    T_l = 1 - 0.5 * switch(x[1] / 2 * sqrt(q / (Lx - x[0] + ϵ)))
    T_r = 0.5 + Q / (2 * np.sqrt(π)) * sqrt(q / (x[1] + 1)) * exp(-x[0]**2 * q / (4 * x[1] + 4))
    T_s = 0.5 - Q / (2 * np.sqrt(π)) * sqrt(q / (2 - x[1])) * exp(-(Lx - x[0])**2 * q / (8 - 4 * x[1]))
    return T_u + T_l + T_r + T_s - Constant(1.5)

T_in = firedrake.Function(temperature_space)
T_in.interpolate(clamp(initial_temperature(x), 0, 1))

# Create the momentum and energy conservation equations
if solution_method == "split":
    z = firedrake.Function(Z)
    u, p = firedrake.split(z)
    v, q = firedrake.TestFunctions(Z)

    T = firedrake.Function(temperature_space)
    φ = firedrake.TestFunction(temperature_space)
    T.assign(T_in)
elif solution_method == "monolithic":
    z = firedrake.Function(Z)
    u, p, T = firedrake.split(z)
    v, q, φ = firedrake.TestFunctions(Z)

    z.sub(2).assign(T_in)


μ = Constant(1)
def ε(u):
    return sym(grad(u))

τ = 2 * μ * ε(u)
g = firedrake.as_vector((0, -1))
f = -Ra * T * g
F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx

ρ, c, k = Constant(1), Constant(1), Constant(1)
F_energy = (ρ * c * Dt(T) * ϕ + inner(-ρ * c * T * u + k * grad(T), grad(φ))) * dx
if args.temperature_basis == "dg":
    n = firedrake.FacetNormal(mesh)
    h = firedrake.CellSize(mesh)

    θ = π / 4
    α = 1 / 2
    γ = Constant(2 * degree * (degree + 1) / α**2 / (np.sin(θ) * np.tan(θ / 2)))

    u_n = max_value(0, inner(u, n))
    G_conv_facet_flux = ρ * c * jump(T * u_n) * jump(φ) * dS
    G_conv_influx = ρ * c * T_in * min_value(0, inner(u, n)) * φ * ds
    G_conv_outflux = ρ * c * T * u_n * φ * ds

    dT_dn = inner(grad(T), n)
    dφ_dn = inner(grad(φ), n)
    G_diff_facet_flux = -k * (jump(dT_dn) * jump(φ) + jump(T) * jump(dφ_dn)) * dS
    G_diff_facet_penalty = γ * avg(k) / avg(h) * jump(T) * jump(φ) * dS
    G_diff_boundary_flux = -k * dφ_dn * (T - T_in) * ds((3, 4))
    G_diff_boundary_penalty = γ * k / h * (T - T_in) * φ * ds((3, 4))

    F_energy += (
        G_conv_facet_flux
        #+ G_conv_influx
        #+ G_conv_outflux
        + G_diff_facet_flux
        + G_diff_facet_penalty
        + G_diff_boundary_flux
        + G_diff_boundary_penalty
    )


# Set up the boundary conditions
velocity_bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")
if solution_method == "split":
    lower_bc = firedrake.DirichletBC(temperature_space, 1, [3])
    upper_bc = firedrake.DirichletBC(temperature_space, 0, [4])
elif solution_method == "monolithic":
    lower_bc = firedrake.DirichletBC(Z.sub(2), 1, [3])
    upper_bc = firedrake.DirichletBC(Z.sub(2), 0, [4])

temperature_bcs = [lower_bc, upper_bc]


# Set up some solvers
basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
params = {
    "solver_parameters": {
        "snes_monitor": ":" + args.log_filename,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

method = irksome.BackwardEuler()
t = Constant(0.0)
dt = Constant(1e3)

if solution_method == "split":
    stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, z, velocity_bc)

    nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), basis])
    params["nullspace"] = nullspace
    stokes_solver = firedrake.NonlinearVariationalSolver(stokes_problem, **params)
    temperature_solver = irksome.TimeStepper(
        F_energy, method, t, dt, T, bcs=temperature_bcs
    )
elif solution_method == "monolithic":
    bcs = [velocity_bc] + temperature_bcs
    F_temp_init = (T - T_in) * φ * dx
    F_initial = F_momentum + F_temp_init
    stokes_problem = firedrake.NonlinearVariationalProblem(F_initial, z, bcs)
    nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), basis, Z.sub(2)])
    stokes_solver = firedrake.NonlinearVariationalSolver(
        stokes_problem, **params, nullspace=nullspace
    )

    F = F_momentum + F_energy
    solver = irksome.TimeStepper(
        F, method, t, dt, z, bcs=bcs, **params, nullspace=[(1, basis)]
    )

stokes_solver.solve()


# Pick the timestep based on the cell size and max velocity
δx = mesh.cell_sizes.dat.data_ro[:].min()
umax = z.sub(0).dat.data_ro[:].max()
dt.assign(args.cfl_fraction * δx / umax)


# And the timestepping loop.
final_time = args.final_time
num_steps = int(final_time / float(dt))
iterator = range(num_steps) if not has_tqdm else tqdm.trange(num_steps)

with firedrake.CheckpointFile(args.output_filename, "w") as output_file:
    output_file.save_mesh(mesh)

    if solution_method == "split":
        u, p = z.subfunctions
    elif solution_method == "monolithic":
        u, p, T = z.subfunctions
    output_file.save_function(T, name="temperature", idx=0)
    output_file.save_function(u, name="velocity", idx=0)
    output_file.save_function(p, name="pressure", idx=0)

    try:
        for step in iterator:
            if solution_method == "split":
                temperature_solver.advance()
                stokes_solver.solve()
                u, p = z.subfunctions
            elif solution_method == "monolithic":
                solver.advance()
                u, p, T = z.subfunctions

            output_file.save_function(T, name="temperature", idx=step + 1)
            output_file.save_function(u, name="velocity", idx=step + 1)
            output_file.save_function(p, name="pressure", idx=step + 1)
    except firedrake.ConvergenceError as error:
        output_file.h5pyfile.attrs["num_steps"] = step
        print(error)
    else:
        output_file.h5pyfile.attrs["num_steps"] = num_steps
