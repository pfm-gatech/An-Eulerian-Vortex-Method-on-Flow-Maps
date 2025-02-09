from taichi_utils import *
import math
from hyperparameters import *
# 3D specific
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: central position
# unit_x, unit_y: the direction
@ti.kernel
def add_vortex_ring(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: int, num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            if smokef[i,j,k][3] != 1.0:
                smokef[i, j, k][color] = 1.0
            smokef[i,j,k][3] = 1.0
        

def init_vorts_leapfrog(X, u):
    add_vortex_ring(w = 2.e-2, rad = 0.21, delta = 0.0168, c = ti.Vector([0.5,0.5,0.23]),
            unit_x = ti.Vector([1.,0.,0.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

    add_vortex_ring(w = 2.e-2, rad = 0.21, delta = 0.0168, c = ti.Vector([0.5,0.5,0.36125]),
            unit_x = ti.Vector([1.,0.,0.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)


def init_vorts_headon(X, u):
    add_vortex_ring(w = 2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.1,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

    add_vortex_ring(w = -2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.4,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

def init_vorts_oblique(X, u):
    x_offset = 0.18
    z_offset = 0.18
    size = 0.15
    add_vortex_ring(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5-z_offset]),
        unit_x = ti.Vector([-0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, num_samples = 500)
    add_vortex_ring(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5+z_offset]),
        unit_x = ti.Vector([0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, num_samples = 500)