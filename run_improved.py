#
from hyperparameters import *
from taichi_utils import *
from mgpcg_w2v import *
from init_conditions import *
from io_utils import *
import sys


u_l_w = 0.
u_r_w = 0.
v_t_w = 0.
v_b_w = 0.
w_a_w = 0.
w_c_w = 0.

ti.init(arch=ti.cuda, device_memory_GB=8.0, debug=False)
boundary_mask = ti.field(ti.i32, shape=(res_x, res_y, res_z))
boundary_vel = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
boundary_types_1 = ti.Matrix(
    [[1, 1], [1, 1], [1, 1]], ti.i32
)
boundary_types_2 = ti.Matrix(
    [[2, 2], [2, 2], [2, 2]], ti.i32
)
solver_w2v = MGPCG_3_w2v(
    boundary_types=boundary_types_1, boundary_mask = boundary_mask, boundary_vel = boundary_vel,
    u_l_w = u_l_w, u_r_w = u_r_w, v_t_w = v_t_w, v_b_w = v_b_w, w_a_w = w_a_w, w_c_w = w_c_w,
    N=[res_x, res_y, res_z], N_together=[3 * res_x, res_y, res_z], base_level=3, dx=dx
)

# undeformed coordinates (cell center and faces)
X = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z))
X_y = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z))
X_z = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z + 1))
X_x_e = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z + 1))
X_y_e = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z + 1))
X_z_e = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y + 1, res_z))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)
x_coords_func_edge(X_x_e, dx)
y_coords_func_edge(X_y_e, dx)
z_coords_func_edge(X_z_e, dx)

# back flow map
T_x = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x, res_y + 1, res_z + 1))  # d_psi / d_x (On Edge)
T_y = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x + 1, res_y, res_z + 1))  # d_psi / d_y (On Edge)
T_z = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x + 1, res_y + 1, res_z))  # d_psi / d_z (On Edge)
psi_x = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z + 1))
psi_y = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z + 1))
psi_z = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y + 1, res_z))

# fwrd flow map
F_x = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x, res_y + 1, res_z + 1))  # d_phi / d_x (On Edge)
F_y = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x + 1, res_y, res_z + 1))  # d_phi / d_y (On Edge)
F_z = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x + 1, res_y + 1, res_z))  # d_phi / d_z (On Edge)
phi_x = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z + 1))
phi_y = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z + 1))
phi_z = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y + 1, res_z))

# velocity storage
u = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))  # curl of u
u_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
u_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
u_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))

# some helper storage for u
tmp_u_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
tmp_u_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
tmp_u_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))

u_x_buffer = [ti.field(ti.f32, shape=(res_x+1, res_y, res_z)) for _ in range(reinit_every - 1)]
u_y_buffer = [ti.field(ti.f32, shape=(res_x, res_y+1, res_z)) for _ in range(reinit_every - 1)]
u_z_buffer = [ti.field(ti.f32, shape=(res_x, res_y, res_z+1)) for _ in range(reinit_every - 1)]

# vorticity storage
w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))
init_w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
init_w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
init_w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))
err_w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
err_w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
err_w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))
tmp_w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
tmp_w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
tmp_w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))

# CFL related
max_speed = ti.field(ti.f32, shape=())
dts = np.zeros(reinit_every)

@ti.kernel
def mask_by_boundary(field: ti.template()):
    for I in ti.grouped(field):
        if boundary_mask[I] > 0:
            field[I] *= 0

@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.0e-3  # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i + 1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j + 1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k + 1])
        speed = ti.sqrt(u**2 + v**2 + w**2)
        ti.atomic_max(max_speed[None], speed)

@ti.kernel
def reset_to_identity(
    psi_x: ti.template(),
    psi_y: ti.template(),
    psi_z: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
):
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x_e[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y_e[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z_e[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Matrix.identity(n=3, dt=ti.f32)

def backtrack_psi_grid(curr_step):
    reset_to_identity(psi_x, psi_y, psi_z, F_x, F_y, F_z)
    RK4_grid_graduT_psiF(psi_x, F_x, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid_graduT_psiF(psi_y, F_y, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid_graduT_psiF(psi_z, F_z, u_x, u_y, u_z, dts[curr_step].item())
    for step in reversed(range(curr_step)):
        RK4_grid(psi_x, F_x, step)
        RK4_grid(psi_y, F_y, step)
        RK4_grid(psi_z, F_z, step)

def RK4_grid(psi_x, T_x, step):
    copy_to(u_x_buffer[step], tmp_u_x)
    copy_to(u_y_buffer[step], tmp_u_y)
    copy_to(u_z_buffer[step], tmp_u_z)
    RK4_grid_graduT_psiF(psi_x, T_x, tmp_u_x, tmp_u_y, tmp_u_z, dts[step].item())

def march_phi_grid(curr_step):
    RK4_grid_graduT_phiT(phi_x, T_x, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid_graduT_phiT(phi_y, T_y, u_x, u_y, u_z, dts[curr_step].item())
    RK4_grid_graduT_phiT(phi_z, T_z, u_x, u_y, u_z, dts[curr_step].item())

@ti.func
def interp_u_MAC_grad(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p = interp_grad_2(u_x, p, inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p, grad_u_y_p = interp_grad_2(u_y, p, inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p, grad_u_z_p = interp_grad_2(u_z, p, inv_dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.rows(
        [grad_u_x_p, grad_u_y_p, grad_u_z_p]
    )

@ti.func
def interp_w_MAC(w_x, w_y, w_z, p, dx):
    w_x_p = interp_2(w_x, p, inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
    w_y_p = interp_2(w_y, p, inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
    w_z_p = interp_2(w_z, p, inv_dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
    return ti.Vector([w_x_p, w_y_p, w_z_p])


@ti.kernel
def RK4_grid_graduT_psiF(
    psi_x: ti.template(),
    T_x: ti.template(),
    u_x0: ti.template(),
    u_y0: ti.template(),
    u_z0: ti.template(),
    dt: ti.f32,
):

    # neg_dt = -1 * dt  # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = T_x[I] @ grad_u_at_psi  # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] - 0.5 * dt * u1  # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = T_x1 @ grad_u_at_psi  # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] - 0.5 * dt * u2  # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * dt * dT_x_dt2
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = T_x2 @ grad_u_at_psi # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] - 1.0 * dt * u3
        T_x3 = T_x[I] + 1.0 * dt * dT_x_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = T_x3 @ grad_u_at_psi  # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] - dt * 1.0 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + dt * 1.0 / 6 * (
            dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4
        )  # advance full


@ti.kernel
def RK4_grid_graduT_phiT(
    psi_x: ti.template(),
    T_x: ti.template(),
    u_x0: ti.template(),
    u_y0: ti.template(),
    u_z0: ti.template(),
    dt: ti.f32,
):

    # neg_dt = -1 * dt  # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = T_x[I] @ grad_u_at_psi  # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * dt * u1  # advance 0.5 steps
        T_x1 = T_x[I] - 0.5 * dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = T_x1 @ grad_u_at_psi  # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * dt * u2  # advance 0.5 again
        T_x2 = T_x[I] - 0.5 * dt * dT_x_dt2
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = T_x2 @ grad_u_at_psi # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * dt * u3
        T_x3 = T_x[I] - 1.0 * dt * dT_x_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = T_x3 @ grad_u_at_psi  # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + dt * 1.0 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] - dt * 1.0 / 6 * (
            dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4
        )  # advance full

@ti.kernel
def advect_w_notrans(
    w_x0: ti.template(), w_y0: ti.template(), w_z0: ti.template(),
    w_x1: ti.template(), w_y1: ti.template(), w_z1: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
    dx: ti.f32,
):
    # x velocity
    for I in ti.grouped(w_x1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_x[I], dx)
        w_x1[I] = (T_x[I] @ w_at_psi)[0]
    # y velocity
    for I in ti.grouped(w_y1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_y[I], dx)
        w_y1[I] = (T_y[I] @ w_at_psi)[1]
    # z velocity
    for I in ti.grouped(w_z1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_z[I], dx)
        w_z1[I] = (T_z[I] @ w_at_psi)[2]


# main function
def main():
    logsdir = os.path.join("logs", exp_name)
    os.makedirs(logsdir, exist_ok=True)
    remove_everything_in(logsdir)
    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)
    # initial condition
    total_t = 0
    gen_boundary_mask(boundary_mask, boundary_vel, total_t)

    u_x.fill(0.)
    u_y.fill(0.)
    u_z.fill(0.)
    if case == 0: # moving paddle
        pass
    elif case == 1: # leapfrog
        init_vorts_leapfrog(X, u)
        split_central_vector(u, u_x, u_y, u_z)

    # for visualization
    get_central_vector(u_x, u_y, u_z, u)
    curl(u, w, inv_dx)
    w_numpy = w.to_numpy()
    w_norm = np.linalg.norm(w_numpy, axis=-1)
    write_vtk(w_norm, vtkdir, from_frame, "vorticity")

    sub_t = 0.0 
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0
    i, j = -1, -1
    while True:
        i += 1
        j += 1
        if i == 0:
            reset_to_identity(phi_x, phi_y, phi_z, T_x, T_y, T_z)
            apply_bc_w(u_x, u_y, u_z, init_w_x, init_w_y, init_w_z, boundary_mask, boundary_vel, inv_dx)
            copy_to(init_w_x, w_x)
            copy_to(init_w_y, w_y)
            copy_to(init_w_z, w_z)
        print("[Simulate] Running step: ", i, " / substep: ", j)

        if (j == reinit_every):
            print(
                "[Simulate] Reinit flow map at substep: ",
                j,
                "with total reinit num", 
                num_reinits,
            )
            reset_to_identity(phi_x, phi_y, phi_z, T_x, T_y, T_z)
            apply_bc_w(u_x, u_y, u_z, init_w_x, init_w_y, init_w_z, boundary_mask, boundary_vel, inv_dx)
            copy_to(init_w_x, w_x)
            copy_to(init_w_y, w_y)
            copy_to(init_w_z, w_z)
            num_reinits += 1
            j = 0

        # determine dt
        calc_max_speed(u_x, u_y, u_z)  # saved to max_speed[None]
        curr_dt = CFL * dx / max_speed[None]
        if sub_t + curr_dt >= visualize_dt:
            curr_dt = visualize_dt - sub_t
            sub_t = 0.0
            frame_idx += 1
            output_frame = True
        else:
            sub_t += curr_dt
            output_frame = False
        dts[j] = curr_dt
        total_t += 0.5 * curr_dt
        # done dt    
        gen_boundary_mask(boundary_mask, boundary_vel, total_t)
        # start midpoint
        reset_to_identity(psi_x, psi_y, psi_z, F_x, F_y, F_z)

        RK4_grid_graduT_psiF(psi_x, F_x, u_x, u_y, u_z, 0.5 * curr_dt)
        RK4_grid_graduT_psiF(psi_y, F_y, u_x, u_y, u_z, 0.5 * curr_dt)
        RK4_grid_graduT_psiF(psi_z, F_z, u_x, u_y, u_z, 0.5 * curr_dt)

        copy_to(w_x, tmp_w_x)
        copy_to(w_y, tmp_w_y)
        copy_to(w_z, tmp_w_z)
        advect_w_notrans(
            tmp_w_x, tmp_w_y, tmp_w_z,
            w_x, w_y, w_z,
            F_x, F_y, F_z,
            psi_x, psi_y, psi_z,
            dx,
        )

        solver_w2v.Poisson_w2v(u_x, u_y, u_z, w_x, w_y, w_z)
        
        copy_to(u_x, tmp_u_x)
        copy_to(u_y, tmp_u_y)
        copy_to(u_z, tmp_u_z)

        backtrack_psi_grid(j)
        march_phi_grid(j)

        if (not j == reinit_every - 1):
            copy_to(u_x, u_x_buffer[j])
            copy_to(u_y, u_y_buffer[j])
            copy_to(u_z, u_z_buffer[j])

        advect_w_notrans(
            init_w_x, init_w_y, init_w_z, w_x, w_y, w_z,
            F_x, F_y, F_z, psi_x, psi_y, psi_z,
            dx,
        )
        # # Begin BFECC
        advect_w_notrans(
            w_x, w_y, w_z, err_w_x, err_w_y, err_w_z,
            T_x, T_y, T_z, phi_x, phi_y, phi_z,
            dx,
        )
        add_fields(err_w_x, init_w_x, err_w_x, -1.0)
        add_fields(err_w_y, init_w_y, err_w_y, -1.0)
        add_fields(err_w_z, init_w_z, err_w_z, -1.0)
        scale_field(err_w_x, 0.5, err_w_x)  # halve error
        scale_field(err_w_y, 0.5, err_w_y)
        scale_field(err_w_z, 0.5, err_w_z)
        advect_w_notrans(
            err_w_x, err_w_y, err_w_z, tmp_w_x, tmp_w_y, tmp_w_z,
            F_x, F_y, F_z, psi_x, psi_y, psi_z,
            dx,
        )
        add_fields(w_x, tmp_w_x, err_w_x, -1.0)
        add_fields(w_y, tmp_w_y, err_w_y, -1.0)
        add_fields(w_z, tmp_w_z, err_w_z, -1.0)

        copy_to(err_w_x, w_x)
        copy_to(err_w_y, w_y)
        copy_to(err_w_z, w_z)

        solver_w2v.Poisson_w2v(u_x, u_y, u_z, w_x, w_y, w_z)
        apply_bc_w(u_x, u_y, u_z, w_x, w_y, w_z, boundary_mask, boundary_vel, inv_dx)

        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush=True)

        if output_frame:
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, inv_dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis=-1)
            write_vtk(w_norm, vtkdir, frame_idx, "vorticity")

            print(
                "[Simulate] Finished frame: ",
                frame_idx,
                " in ",
                i - last_output_substep,
                "substeps \n\n",
            )
            last_output_substep = i

            if frame_idx >= total_frames:
                break


if __name__ == "__main__":
    main()