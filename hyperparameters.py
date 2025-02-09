# some hyperparameters
import taichi as ti
from taichi_utils import *

case = 1


@ti.kernel
def moving_paddle_boundary_mask(boundary_mask: ti.template(), boundary_vel: ti.template(), _t: float): # if is boundary then 1, if not boundary then 0
    t = _t * 0.5 # scale t

    tmp_dt = 0.01
    pos_left = 0.5
    pos_right = 1.7
    COM = ti.Vector([lerp(pos_left, pos_right, 0.5*(1+ti.cos(t))), 0.5, 0.5])
    COM_next = ti.Vector([lerp(pos_left, pos_right, 0.5*(1+ti.cos(t+tmp_dt))), 0.5, 0.5])
    body2world = ti.math.rot_yaw_pitch_roll(0., 0., 1.5 * (t))
    world2body = body2world.inverse()
    body2world_next = ti.math.rot_yaw_pitch_roll(0., 0., 1.5 * (t+tmp_dt))
    world2body_next = body2world_next.inverse()

    # single sided
    x_width = 0.025
    y_width = 0.27
    z_width = 0.27
    for i,j,k in boundary_mask:
        pos_world = ti.Vector([i+0.5,j+0.5,k+0.5]) / res_y
        pos = pos_world - COM
        pos = (world2body @ ti.Vector([pos.x, pos.y, pos.z, 1])).xyz # this is the pos in body frame
        if -x_width < pos.x < x_width and -y_width < pos.y < y_width and -z_width < pos.z < z_width:
            boundary_mask[i,j,k] = 1
            pos_world_next = (body2world_next @ ti.Vector([pos.x, pos.y, pos.z, 1])).xyz + COM_next
            boundary_vel[i,j,k] = (pos_world_next - pos_world)/tmp_dt
        else:
            boundary_mask[i,j,k] = 0
            boundary_vel[i,j,k] *= 0

@ti.kernel
def no_bond(boundary_mask: ti.template(), boundary_vel: ti.template(), _t: float):
    pass


if case == 0:
    res_x = 256
    res_y = 128
    res_z = 128
    dx = 1./res_y
    inv_dx = res_y
    visualize_dt = 0.05
    reinit_every = 8
    CFL = 0.5
    from_frame = 0
    total_frames = 600
    gen_boundary_mask = moving_paddle_boundary_mask
    exp_name = "3D_moving_paddle" + "_reinit_" + str(reinit_every)  

elif case == 1:
    res_x = 128
    res_y = 128
    res_z = 256
    dx = 1./res_y
    inv_dx = res_y
    visualize_dt = 0.05
    reinit_every = 20
    CFL = 0.5
    from_frame = 0
    total_frames = 800
    gen_boundary_mask = no_bond
    exp_name = "3D_leapfrog" + "_reinit_" + str(reinit_every)  