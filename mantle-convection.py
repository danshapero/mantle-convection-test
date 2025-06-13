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
    Constant, sqrt, exp, min_value, max_value, inner, sym, grad, div, dx, ds,
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
basis = args.temperature_basis
degree = args.temperature_degree

# Make the mesh and some function spaces
Lx, Ly = Constant(2.0), Constant(1.0)
num_cells_x = int(float(Lx / Ly)) * num_cells
mesh = firedrake.RectangleMesh(
    num_cells_x, num_cells, float(Lx), float(Ly), diagonal="crossed"
)

pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
Z_1 = velocity_space * pressure_space

temperature_space = firedrake.FunctionSpace(mesh, basis.upper(), degree)
if basis == "cg":
    Z_2 = temperature_space
else:
    flux_space = firedrake.FunctionSpace(mesh, "RT", degree + 1)
    Z_2 = temperature_space * flux_space

if solution_method == "split":
    Z = velocity_space * pressure_space
    W = Z_2
elif solution_method == "monolithic":
    Z = Z_1 * Z_2


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


# Some annoying control flow to deal with all the different configurations
z = firedrake.Function(Z)
if solution_method == "split":
    u, p = firedrake.split(z)
    v, q = firedrake.TestFunctions(Z)

    w = firedrake.Function(W)
    w.sub(0).assign(T_in)
    if basis == "cg":
        T = w
        φ = firedrake.TestFunction(W)
    elif basis == "dg":
        T, F = firedrake.split(w)
        φ, G = firedrake.TestFunctions(W)
elif solution_method == "monolithic":
    z = firedrake.Function(Z)
    z.sub(2).assign(T_in)

    if basis == "cg":
        u, p, T = firedrake.split(z)
        v, q, φ = firedrake.TestFunctions(Z)
    elif basis == "dg":
        u, p, T, F = firedrake.split(z)
        v, q, φ, G = firedrake.TestFunctions(Z)


# Form the momentum balance equation
μ = Constant(1)
ε = lambda u: sym(grad(u))
τ = 2 * μ * ε(u)
g = firedrake.as_vector((0, -1))
f = -Ra * T * g
F_momentum = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx


# Form the energy balance equation
ρ, c, k = Constant(1), Constant(1), Constant(1)
T_lower, T_upper = Constant(1.0), Constant(0.0)
lower_ids, upper_ids = (3,), (4,)
if basis == "cg":
    F_temperature = (
        ρ * c * Dt(T) * ϕ - inner(ρ * c * T * u - k * grad(T), grad(φ))
    ) * dx
elif basis == "dg":
    F_conservation = (ρ * c * Dt(T) + div(F)) * φ * dx
    F_constitutive = (T * div(G) - inner(F - ρ * c * T * u, G) / k) * dx

    n = firedrake.FacetNormal(mesh)
    F_boundary = (
        T_lower * inner(G, n) * ds(lower_ids) + T_upper * inner(G, n) * ds(upper_ids)
    )

    F_temperature = F_conservation + F_constitutive - F_boundary


# Set up the boundary conditions
velocity_bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")
if basis == "cg":
    if solution_method == "split":
        space = W
    elif solution_method == "monolithic":
        space = Z.sub(2)

    lower_bc = firedrake.DirichletBC(space, T_lower, lower_ids)
    upper_bc = firedrake.DirichletBC(space, T_upper, upper_ids)

    temperature_bcs = [lower_bc, upper_bc]
elif basis == "dg":
    if solution_method == "split":
        space = W.sub(1)
    elif solution_method == "monolithic":
        space = Z.sub(3)

    side_ids = [1, 2]
    flux_bc = firedrake.DirichletBC(space, Constant((0, 0)), side_ids)

    temperature_bcs = [flux_bc]


# Set up some solvers
const_fns = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
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

    nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), const_fns])
    params["nullspace"] = nullspace
    stokes_solver = firedrake.NonlinearVariationalSolver(stokes_problem, **params)

    temperature_solver = irksome.TimeStepper(
        F_temperature, method, t, dt, w, bcs=temperature_bcs
    )
elif solution_method == "monolithic":
    bcs = [velocity_bc] + temperature_bcs
    F_temp_init = (T - T_in) * φ * dx
    if basis == "dg":
        F_temp_init += F_constitutive - F_boundary
    F_initial = F_momentum + F_temp_init
    stokes_problem = firedrake.NonlinearVariationalProblem(F_initial, z, bcs)

    bases = [Z.sub(0), const_fns, Z.sub(2)]
    if basis == "dg":
        bases.append(Z.sub(3))
    nullspace = firedrake.MixedVectorSpaceBasis(Z, bases)
    stokes_solver = firedrake.NonlinearVariationalSolver(
        stokes_problem, **params, nullspace=nullspace
    )

    F = F_momentum + F_temperature
    solver = irksome.TimeStepper(
        F, method, t, dt, z, bcs=bcs, **params, nullspace=[(1, const_fns)]
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

    output_file.save_function(z.sub(0), name="velocity", idx=0)
    output_file.save_function(z.sub(1), name="pressure", idx=0)
    if solution_method == "split":
        T = w.sub(0)
    elif solution_method == "monolithic":
        T = z.sub(2)
    output_file.save_function(T, name="temperature", idx=0)

    try:
        for step in iterator:
            if solution_method == "split":
                temperature_solver.advance()
                stokes_solver.solve()
                u, p = z.subfunctions
                T = w.subfunctions[0]
            elif solution_method == "monolithic":
                solver.advance()
                u, p, T = z.subfunctions[:3]

            output_file.save_function(T, name="temperature", idx=step + 1)
            output_file.save_function(u, name="velocity", idx=step + 1)
            output_file.save_function(p, name="pressure", idx=step + 1)
    except firedrake.ConvergenceError as error:
        output_file.h5pyfile.attrs["num_steps"] = step
        print(error)
    else:
        output_file.h5pyfile.attrs["num_steps"] = num_steps
