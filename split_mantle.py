import argparse
import firedrake
from firedrake import Constant
from irksome import BackwardEuler, TimeStepper
import mantle


# Get command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--output-filename", type=str, default="split.h5")
parser.add_argument("--num-cells", type=int, default=32)
parser.add_argument("--temperature-degree", type=int, default=1)
parser.add_argument("--cfl-fraction", type=float, default=1.0)
parser.add_argument("--final-time", type=float, default=0.25)
args = parser.parse_args()

# Make the mesh and some function spaces
lx, ly = 2.0, 1.0
num_cells = args.num_cells
num_cells_x = int(lx / ly) * num_cells
mesh = firedrake.RectangleMesh(num_cells_x, num_cells, lx, ly, diagonal="crossed")

# Make osme function spaces
pressure_space = firedrake.FunctionSpace(mesh, "CG", 1)
velocity_space = firedrake.VectorFunctionSpace(mesh, "CG", 2)
temperature_space = firedrake.FunctionSpace(mesh, "CG", args.temperature_degree)
Z = velocity_space * pressure_space

# Make some fields and initialize the temperature field
z = firedrake.Function(Z)
T = firedrake.Function(temperature_space)
x = firedrake.SpatialCoordinate(mesh)
ra = mantle.default_parameters["rayleigh_number"]
T.interpolate(mantle.initial_temperature(x, num_cells_x, lx, ra))

# Form the PDEs we wish to solve
u, p = firedrake.split(z)
v, q = firedrake.TestFunctions(Z)
φ = firedrake.TestFunction(temperature_space)

F_momentum = mantle.form_momentum_eqn(u, p, T, v, q, **mantle.default_parameters)
F_energy = mantle.form_energy_eqn(T, u, φ, **mantle.default_parameters)

# Make some boundary conditions
velocity_bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")

T_lower, T_upper = Constant(1.0), Constant(0.0)
lower_ids, upper_ids = [3], [4]
lower_bc = firedrake.DirichletBC(temperature_space, T_lower, lower_ids)
upper_bc = firedrake.DirichletBC(temperature_space, T_upper, upper_ids)
temperature_bcs = [lower_bc, upper_bc]

# Make solvers
params = {
    "solver_parameters": {
        "snes_monitor": None,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

stokes_problem = firedrake.NonlinearVariationalProblem(F_momentum, z, velocity_bc)
const_fns = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), const_fns])
stokes_solver = firedrake.NonlinearVariationalSolver(
    stokes_problem, **params, nullspace=nullspace
)
stokes_solver.solve()

method = BackwardEuler()
t = Constant(0.0)
dt = Constant(1e3)
δx = mesh.cell_sizes.dat.data_ro[:].min()
umax = z.sub(0).dat.data_ro[:].max()
dt.assign(args.cfl_fraction * δx / umax)

temperature_solver = TimeStepper(F_energy, method, t, dt, T, bcs=temperature_bcs)

# The solution loop
final_time = args.final_time
num_steps = int(final_time / float(dt))

with firedrake.CheckpointFile(args.output_filename, "w") as output_file:
    output_file.save_mesh(mesh)

    u, p = z.subfunctions
    output_file.save_function(T, name="temperature", idx=0)
    output_file.save_function(u, name="velocity", idx=0)
    output_file.save_function(p, name="pressure", idx=0)

    try:
        for step in range(num_steps):
            temperature_solver.advance()
            stokes_solver.solve()
            u, p = z.subfunctions

            output_file.save_function(T, name="temperature", idx=step + 1)
            output_file.save_function(u, name="velocity", idx=step + 1)
            output_file.save_function(p, name="pressure", idx=step + 1)
    except firedrake.ConvergenceError as error:
        output_file.h5pyfile.attrs["num_steps"] = step
        print(error)
    else:
        output_file.h5pyfile.attrs["num_steps"] = num_steps
