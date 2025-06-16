import numpy as np
from numpy import pi as π
import firedrake
from firedrake import (
    Constant, sqrt, exp, min_value, max_value, inner, sym, grad, div, dx
)
from irksome import Dt


default_parameters = {
    "density": 1.0,
    "heat_capacity": 1.0,
    "thermal_conductivity": 1.0,
    "viscosity": 1.0,
    "rayleigh_number": 1e6,
}


def clamp(z, zmin, zmax):
    return min_value(Constant(zmax), max_value(Constant(zmin), z))


def switch(z):
    return exp(z) / (exp(z) + exp(-z))


def initial_temperature(x, nx, lx, ra):
    Lx, Ra = Constant(lx), Constant(ra)
    δ = Constant(1 / nx)
    q = Lx**(7 / 3) / (1 + Lx**4)**(2 / 3) * (Ra / (2 * np.sqrt(π)))**(2/3)
    Q = 2 * sqrt(Lx / (π * q))
    T_u = 0.5 * switch((1 - x[1]) / 2 * sqrt(q / (x[0] + δ)))
    T_l = 1 - 0.5 * switch(x[1] / 2 * sqrt(q / (Lx - x[0] + δ)))
    T_r = 0.5 + Q / (2 * np.sqrt(π)) * sqrt(q / (x[1] + 1)) * exp(-x[0]**2 * q / (4 * x[1] + 4))
    T_s = 0.5 - Q / (2 * np.sqrt(π)) * sqrt(q / (2 - x[1])) * exp(-(Lx - x[0])**2 * q / (8 - 4 * x[1]))
    return clamp(T_u + T_l + T_r + T_s - Constant(1.5), 0, 1)


def form_momentum_eqn(u, p, T, v, q, **parameters):
    μ = Constant(parameters["viscosity"])
    Ra = Constant(parameters["rayleigh_number"])
    ε = lambda u: sym(grad(u))
    τ = 2 * μ * ε(u)
    g = Constant((0, -1))
    f = -Ra * T * g
    return (inner(τ, ε(v)) - q * div(u) - p * div(v) - inner(f, v)) * dx


def form_energy_eqn(T, u, φ, **parameters):
    ρ = Constant(parameters["density"])
    c = Constant(parameters["heat_capacity"])
    k = Constant(parameters["thermal_conductivity"])
    return (ρ * c * Dt(T) * ϕ - inner(ρ * c * T * u - k * grad(T), grad(φ))) * dx

