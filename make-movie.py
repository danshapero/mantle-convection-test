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

    Ts = []
    for step in range(num_steps + 1):
        Ts.append(input_file.load_function(mesh, name="temperature", idx=step))

fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

fn_plotter = firedrake.FunctionPlotter(mesh, num_sample_points=4)
kw = {"vmin": 0, "vmax": 1, "cmap": "inferno"}
colors = firedrake.tripcolor(Ts[0], axes=ax, num_sample_points=4, **kw)

def animate(T):
    colors.set_array(fn_plotter(T))

iterator = Ts if not has_tqdm else tqdm.tqdm(Ts)
animation = FuncAnimation(fig, animate, iterator, interval=1e3 / args.framerate)
animation.save(args.output_filename)
