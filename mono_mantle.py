import argparse
import firedrake
from firedrake import Constant, dx
from irksome import BackwardEuler, TimeStepper
import mantle


# Get command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--output-filename", type=str, default="mono.h5")
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
Z = velocity_space * pressure_space * temperature_space

# Make some fields and initialize the temperature field
z = firedrake.Function(Z)
T_in = firedrake.Function(temperature_space)
x = firedrake.SpatialCoordinate(mesh)
ra = mantle.default_parameters["rayleigh_number"]
T_in.interpolate(mantle.initial_temperature(x, num_cells_x, lx, ra))
z.sub(2).assign(T_in)

# Form the PDEs we wish to solve
u, p, T = firedrake.split(z)
v, q, φ = firedrake.TestFunctions(Z)

F_momentum = mantle.form_momentum_eqn(u, p, T, v, q, **mantle.default_parameters)
F_energy = mantle.form_energy_eqn(T, u, φ, **mantle.default_parameters)

# Make some boundary conditions
velocity_bc = firedrake.DirichletBC(Z.sub(0), Constant((0, 0)), "on_boundary")

T_lower, T_upper = Constant(1.0), Constant(0.0)
lower_ids, upper_ids = [3], [4]
lower_bc = firedrake.DirichletBC(Z.sub(2), T_lower, lower_ids)
upper_bc = firedrake.DirichletBC(Z.sub(2), T_upper, upper_ids)
bcs = [velocity_bc, lower_bc, upper_bc]

# Make solvers
const_fns = firedrake.VectorSpaceBasis(constant=True, comm=firedrake.COMM_WORLD)
nullspace = firedrake.MixedVectorSpaceBasis(Z, [Z.sub(0), const_fns, Z.sub(2)])
params = {
    "solver_parameters": {
        "snes_monitor": None,
        "snes_linesearch_monitor": None,
        "snes_linesearch_type": "l2",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

F_temp_init = (T - T_in) * φ * dx
F_initial = F_momentum + F_temp_init
stokes_problem = firedrake.NonlinearVariationalProblem(F_initial, z, velocity_bc)
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

F = F_momentum + F_energy
solver = TimeStepper(
    F, method, t, dt, z, bcs=bcs, **params, nullspace=[(1, const_fns)]
)

# The solution loop
final_time = args.final_time
num_steps = int(final_time / float(dt))
with firedrake.CheckpointFile(args.output_filename, "w") as output_file:
    output_file.save_mesh(mesh)

    u, p, T = z.subfunctions
    output_file.save_function(T, name="temperature", idx=0)
    output_file.save_function(u, name="velocity", idx=0)
    output_file.save_function(p, name="pressure", idx=0)

    try:
        for step in range(num_steps):
            solver.advance()
            u, p, T = z.subfunctions

            output_file.save_function(T, name="temperature", idx=step + 1)
            output_file.save_function(u, name="velocity", idx=step + 1)
            output_file.save_function(p, name="pressure", idx=step + 1)
    except firedrake.ConvergenceError as error:
        output_file.h5pyfile.attrs["num_steps"] = step
        print(error)
        print(f"Failed at step #{step}/{num_steps}")
    else:
        output_file.h5pyfile.attrs["num_steps"] = num_steps
