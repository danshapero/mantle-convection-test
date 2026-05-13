import argparse
from petsc4py import PETSc
import firedrake
from firedrake import Constant, dx
from irksome import BackwardEuler, TimeStepper, getForm
from irksome.tools import get_stage_space, getNullspace
import mantle

Print = firedrake.PETSc.Sys.Print

# Get command-line options
parser = argparse.ArgumentParser()
parser.add_argument("--output-filename", type=str, default="output/mono.h5")
parser.add_argument("--num-cells", type=int, default=32)
parser.add_argument("--temperature-degree", type=int, default=1)
parser.add_argument("--cfl-fraction", type=float, default=1.0)
parser.add_argument("--final-time", type=float, default=0.25)
parser.add_argument("--show-args", action="store_true")
parser.add_argument("--progress", action="store_true")
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
stokes_parameters = {
    "snes_monitor": None,
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_factor_shift_type": "nonzero",
}

F_temp_init = (T - T_in) * φ * dx
F_initial = F_momentum + F_temp_init
stokes_problem = firedrake.NonlinearVariationalProblem(F_initial, z, velocity_bc)
stokes_solver = firedrake.NonlinearVariationalSolver(
    stokes_problem, nullspace=nullspace,
    solver_parameters=stokes_parameters,
    options_prefix="stokes",
)
stokes_solver.solve()

u, p, T = z.subfunctions

method = BackwardEuler()
t = Constant(0.0)
dt = Constant(1e3)

speed = firedrake.interpolate(firedrake.sqrt(firedrake.dot(u, u)), u.sub(0).function_space())

with mesh.cell_sizes.dat.vec_ro as vec:
    δx = vec.min()[1]
with firedrake.assemble(speed).dat.vec_ro as vec:
    Print(f"{vec.max()[1]= :.2e} . {vec.mean()= :.2e}")
    umax = vec.max()[1]
dt.assign(args.cfl_fraction * δx / umax)

F = F_momentum + F_energy

W = get_stage_space(Z, method.num_stages)
w = firedrake.Function(W)
G, gbc = getForm(F, method, t, dt, z, w, bcs=bcs)
gnullspace = getNullspace(Z, W, method.num_stages, [(1, const_fns)])


r = firedrake.Cofunction(W.dual())
def callback(X, F):
    with r.dat.vec_wo as v:
        F.copy(v)

    error_norms = []
    for r_i in r.subfunctions:
        with r_i.dat.vec_ro as R_i:
            error_norms.append(R_i.norm())

    PETSc.Sys.Print(f"    > Component norms: |u|={error_norms[0]:.6e} . |p|={error_norms[1]:.6e} . |T|={error_norms[2]:.6e}")

mono_params = {
    "snes_type": "newtonls",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "pc_factor_shift_type": "nonzero",

    "snes_type": "python",
    "snes_python_type": "firedrake.FieldsplitSNES",
}

problem = firedrake.NonlinearVariationalProblem(G, w, bcs=gbc)
solver = firedrake.NonlinearVariationalSolver(
    problem, **params, nullspace=gnullspace, options_prefix="",
    post_function_callback=callback
)

# The solution loop
final_time = args.final_time
num_steps = int(final_time / float(dt))

nranks = mesh.comm.size
Print(f"{Z.dim()= :>6d} . {Z.dim()/nranks= :>6.0f}")
Print(f"{umax= :.2e} . nt= {num_steps:>4d} . tend= {final_time:.2e} . dt= {float(dt):.2e}")
Print()

with firedrake.CheckpointFile(args.output_filename, "w") as output_file:
    output_file.save_mesh(mesh)

    output_file.save_function(T, name="temperature", idx=0)
    output_file.save_function(u, name="velocity", idx=0)
    output_file.save_function(p, name="pressure", idx=0)

    num_fields = len(Z)

    try:
        steps_iter = range(num_steps)
        if args.progress:
            firedrake.ProgressBar.width = 20
            steps_iter = firedrake.ProgressBar().iter(steps_iter)

        for step in steps_iter:
            if not args.progress:
                Print(f"\n=== Timestep {step:>4d}/{num_steps} ===")

            solver.solve()

            for stage_index in range(method.num_stages):
                for field_index in range(num_fields):
                    stage = w.dat.data_ro[num_fields * stage_index + field_index][:]
                    coeff = method.b[stage_index]
                    z.dat.data[field_index][:] += float(dt) * coeff * stage

            with firedrake.assemble(speed).dat.vec_ro as vec:
                Print(f"{vec.max()[1]= :.2e} . {vec.mean()= :.2e}")

            output_file.save_function(T, name="temperature", idx=step + 1)
            output_file.save_function(u, name="velocity", idx=step + 1)
            output_file.save_function(p, name="pressure", idx=step + 1)
    except firedrake.ConvergenceError as error:
        print(error)
        print(f"Failed at step #{step}/{num_steps}")
    finally:
        output_file.h5pyfile.attrs["num_steps"] = step + 1
