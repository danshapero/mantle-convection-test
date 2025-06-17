import argparse
try:
    import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from petsc4py import PETSc
import firedrake

parser = argparse.ArgumentParser()
parser.add_argument("--input-filename")
parser.add_argument("--output-filename")
parser.add_argument("--framerate", type=int, default=30)
args = parser.parse_args()

with firedrake.CheckpointFile(args.input_filename, "r") as input_file:
    mesh = input_file.load_mesh()
    num_steps = input_file.h5pyfile.attrs["num_steps"]

    us, Ts = [], []
    for step in range(num_steps + 1):
        us.append(input_file.load_function(mesh, name="velocity", idx=step))
        Ts.append(input_file.load_function(mesh, name="temperature", idx=step))

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
for ax in axes:
    ax.set_aspect("equal")
    ax.set_axis_off()

fn_plotter = firedrake.FunctionPlotter(mesh, num_sample_points=4)
kw = {"vmin": 0, "vmax": 1, "cmap": "inferno"}
colors = firedrake.tripcolor(Ts[0], axes=axes[0], num_sample_points=4, **kw)

X = mesh.coordinates.dat.data_ro
V = mesh.coordinates.function_space()
v = us[0].copy(deepcopy=True)
interpolator = firedrake.Interpolate(v, V)
u_X = firedrake.assemble(interpolator)
u_vals = u_X.dat.data_ro
arrows = firedrake.quiver(us[0], axes=axes[1], cmap="inferno")

def animate(fields):
    u, T = fields

    colors.set_array(fn_plotter(T))

    v.assign(u)
    u_X.assign(firedrake.assemble(interpolator))
    u_vals = u_X.dat.data_ro
    arrows.set_UVC(*(u_vals.T))

fields = list(zip(us, Ts))
iterator = fields if not has_tqdm else tqdm.tqdm(fields)
animation = FuncAnimation(fig, animate, iterator, interval=1e3 / args.framerate)
animation.save(args.output_filename)
