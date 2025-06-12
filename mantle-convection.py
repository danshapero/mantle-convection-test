# This code is taken from a chapter in The FEniCS Book:
#   https://fenicsproject.org/book/),
# which is available for free online. That chapter an implementation of a model
# setup from van Keken et al (1997):
#   https://doi.org/10.1029/97JB01353).
# I took the thermomechanical parts and removed the chemistry.

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
    Constant, sqrt, exp, min_value, max_value, inner, sym, grad, div, dx
)
import irksome
from irksome import Dt


# Get command-line options
options = PETSc.Options()
output_filename = options.getString("output", "mantle-convection.h5")
num_cells = options.getInt("num-cells", 32)
degree = options.getInt("degree", 1)


# Make the mesh
Lx, Ly = Constant(2.0), Constant(1.0)
num_cells_x = int(float(Lx / Ly)) * num_cells
mesh = firedrake.RectangleMesh(
    num_cells_x, num_cells, float(Lx), float(Ly), diagonal="crossed"
)


# Create the initial temperature field; the expression is heinous but if you
# plot it you'll see what it's supposed to do.
def clamp(z, zmin, zmax):
    return min_value(Constant(zmax), max_value(Constant(zmin), z))

def switch(z):
    return exp(z) / (exp(z) + exp(-z))

Ra = Constant(1e6)

ϵ = Constant(1 / num_cells_x)
x = firedrake.SpatialCoordinate(mesh)

q = Lx**(7 / 3) / (1 + Lx**4)**(2 / 3) * (Ra / (2 * np.sqrt(π)))**(2/3)
Q = 2 * sqrt(Lx / (π * q))
T_u = 0.5 * switch((1 - x[1]) / 2 * sqrt(q / (x[0] + ϵ)))
T_l = 1 - 0.5 * switch(x[1] / 2 * sqrt(q / (Lx - x[0] + ϵ)))
T_r = 0.5 + Q / (2 * np.sqrt(π)) * sqrt(q / (x[1] + 1)) * exp(-x[0]**2 * q / (4 * x[1] + 4))
T_s = 0.5 - Q / (2 * np.sqrt(π)) * sqrt(q / (2 - x[1])) * exp(-(Lx - x[0])**2 * q / (8 - 4 * x[1]))
expr = T_u + T_l + T_r + T_s - Constant(1.5)

temperature_space = firedrake.FunctionSpace(mesh, "CG", degree)
T_0 = firedrake.Function(temperature_space).interpolate(clamp(expr, 0, 1))
T = T_0.copy(deepcopy=True)


# Set up the momentum balance equation
pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
Z = velocity_space * pressure_space

z = firedrake.Function(Z)
u, p = firedrake.split(z)

μ = Constant(1)
def ε(u):
    return sym(grad(u))

v, q = firedrake.TestFunctions(z.function_space())

τ = 2 * μ * ε(u)
g = firedrake.as_vector((0, -1))
f = -Ra * T * g
F = (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx

bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")

basis = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), basis])

stokes_problem = firedrake.NonlinearVariationalProblem(F, z, bc)
parameters = {
    "nullspace": nullspace,
    "solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
stokes_solver = firedrake.NonlinearVariationalSolver(stokes_problem, **parameters)
stokes_solver.solve()


# Set up the energy balance equation
ρ, c, k = Constant(1), Constant(1), Constant(1)
δx = mesh.cell_sizes.dat.data_ro[:].min()
umax = z.sub(0).dat.data_ro[:].max()
δt = Constant(δx / umax)

ϕ = firedrake.TestFunction(temperature_space)
F_convective = -ρ * c * T * inner(u, grad(ϕ)) * dx
F_diffusive = k * inner(grad(T), grad(ϕ)) * dx
F = ρ * c * Dt(T) * ϕ * dx + F_convective + F_diffusive

lower_bc = firedrake.DirichletBC(temperature_space, 1, [3])
upper_bc = firedrake.DirichletBC(temperature_space, 0, [4])
bcs = [lower_bc, upper_bc]

method = irksome.BackwardEuler()
temperature_solver = irksome.TimeStepper(F, method, Constant(0.0), δt, T, bcs=bcs)


# And the timestepping loop.
final_time = 0.25
num_steps = int(final_time / float(δt))
iterator = range(num_steps) if not has_tqdm else tqdm.trange(num_steps)

with firedrake.CheckpointFile(output_filename, "w") as output_file:
    output_file.h5pyfile.attrs["final_time"] = final_time
    output_file.h5pyfile.attrs["num_steps"] = num_steps

    output_file.save_mesh(mesh)

    u, p = z.subfunctions
    output_file.save_function(T, name="temperature", idx=0)
    output_file.save_function(u, name="velocity", idx=0)
    output_file.save_function(p, name="pressure", idx=0)

    for step in iterator:
        temperature_solver.advance()
        stokes_solver.solve()

        u, p = z.subfunctions
        output_file.save_function(T, name="temperature", idx=step + 1)
        output_file.save_function(u, name="velocity", idx=step + 1)
        output_file.save_function(p, name="pressure", idx=step + 1)
