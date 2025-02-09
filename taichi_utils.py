import taichi as ti

eps = 1.e-6
data_type = ti.f32


@ti.kernel
def copy_to3(source: ti.template(), dest1: ti.template(), dest2: ti.template(), dest3: ti.template()):
    for I in ti.grouped(source):
        dest1[I] = source[I].x
        dest2[I] = source[I].y
        dest3[I] = source[I].z


@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    for I in ti.grouped(source):
        dest[I] = source[I]

@ti.kernel
def scale_field(a: ti.template(), alpha: ti.f32, result: ti.template()):
    for I in ti.grouped(result):
        result[I] = alpha * a[I]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: ti.f32):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0., 1.0]
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: ti.f32):
    for I in ti.grouped(pf):
        pf[I] = (I+0.5) * dx

@ti.kernel
def x_coords_func(pf: ti.template(), dx: ti.f32):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k + 0.5]) * dx

@ti.kernel
def y_coords_func(pf: ti.template(), dx: ti.f32):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k + 0.5]) * dx

@ti.kernel
def z_coords_func(pf: ti.template(), dx: ti.f32):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j + 0.5, k]) * dx

@ti.kernel
def x_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k]) * dx

@ti.kernel
def y_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k]) * dx

@ti.kernel
def z_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j, k + 0.5]) * dx


@ti.func
def sample(qf: ti.template(), u: ti.f32, v: ti.f32, w: ti.f32):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    k = ti.max(0, ti.min(int(w), w_dim-1))
    return qf[i, j, k]

@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), inv_dx: ti.f32):
    inv_dist = 0.5 * inv_dx
    for i, j, k in cf:
        vr = sample(vf, i+1, j, k)
        vl = sample(vf, i-1, j, k)
        vt = sample(vf, i, j+1, k)
        vb = sample(vf, i, j-1, k)
        vc = sample(vf, i, j, k+1)
        va = sample(vf, i, j, k-1)

        d_vx_dz = inv_dist * (vc.x - va.x)
        d_vx_dy = inv_dist * (vt.x - vb.x)
        
        d_vy_dx = inv_dist * (vr.y - vl.y)
        d_vy_dz = inv_dist * (vc.y - va.y)

        d_vz_dx = inv_dist * (vr.z - vl.z)
        d_vz_dy = inv_dist * (vt.z - vb.z)

        cf[i,j,k][0] = d_vz_dy - d_vy_dz
        cf[i,j,k][1] = d_vx_dz - d_vz_dx
        cf[i,j,k][2] = d_vy_dx - d_vx_dy

@ti.kernel
def curl_f2e_x(u_z: ti.template(), u_y: ti.template(), w_x: ti.template(), inv_dist: ti.f32):
    for i, j, k in w_x:
        vt = sample(u_z, i, j, k)
        vb = sample(u_z, i, j - 1, k)
        vc = sample(u_y, i, j, k)
        va = sample(u_y, i, j, k - 1)

        w_x[i, j, k] = ((vt - vb) - (vc - va)) * inv_dist

@ti.kernel
def curl_f2e_y(u_x: ti.template(), u_z: ti.template(), w_y: ti.template(), inv_dist: ti.f32):
    for i, j, k in w_y:
        vc = sample(u_x, i, j, k)
        va = sample(u_x, i, j, k - 1)
        vr = sample(u_z, i, j, k)
        vl = sample(u_z, i - 1, j, k)

        w_y[i, j, k] = ((vc - va) - (vr - vl)) * inv_dist

@ti.kernel
def curl_f2e_z(u_y: ti.template(), u_x: ti.template(), w_z: ti.template(), inv_dist: ti.f32):
    for i, j, k in w_z:
        vr = sample(u_y, i, j, k)
        vl = sample(u_y, i - 1, j, k)
        vt = sample(u_x, i, j, k)
        vb = sample(u_x, i, j - 1, k)

        w_z[i, j, k] = ((vr - vl) - (vt - vb)) * inv_dist


# limit (modify) u with u1, write to u2
@ti.kernel
def BFECC_limiter(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_l = sample(u1, i - 1, j, k)
        u1_r = sample(u1, i + 1, j, k)
        u1_b = sample(u1, i, j - 1, k)
        u1_t = sample(u1, i, j + 1, k)
        u1_a = sample(u1, i, j, k - 1)
        u1_c = sample(u1, i, j, k + 1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)

@ti.kernel
def calculate_visc(w_n:ti.template(),T_n:ti.template()):
    for i,j,k in w_n:
        w_n[i,j,k] = T_n[i,j,k] @ w_n[i,j,k]


@ti.kernel
def laplace_node(wn:ti.template(), laplace_wn:ti.template(), dx:float):
    for i,j,k in laplace_wn:
        laplace_wn[i,j,k] = -(6.0 * wn[i, j, k] -
                sample(wn, i - 1, j, k) +
                sample(wn, i + 1, j, k) +
                sample(wn, i, j - 1, k) +
                sample(wn, i, j + 1, k) +
                sample(wn, i, j, k - 1) +
                sample(wn, i, j, k + 1))/dx/dx

@ti.kernel
def get_node_vector_sameshape(wn:ti.template(), wx:ti.template(), wy:ti.template(), wz:ti.template()):
    for i,j,k in wn:
        wn[i,j,k] = 1/6.0 * ((sample(wx, i-1, j, k) + sample(wx, i, j, k)) 
        +  (sample(wy, i, j-1, k) + sample(wy, i, j, k))
        +  (sample(wz, i, j, k-1) + sample(wz, i, j, k)))

@ti.kernel
def get_node_vector(wn:ti.template(), wx:ti.template(), wy:ti.template(), wz:ti.template()):
    for i,j,k in wn:
        wn[i,j,k].x = 0.5 * (sample(wx, i-1, j, k) + sample(wx, i, j, k))
        wn[i,j,k].y = 0.5 * (sample(wy, i, j-1, k) + sample(wy, i, j, k))
        wn[i,j,k].z = 0.5 * (sample(wz, i, j, k-1) + sample(wz, i, j, k))

@ti.kernel
def split_node_vector(wn:ti.template(), wx:ti.template(), wy:ti.template(), wz:ti.template()):
    for i, j, k in wx:
        r = sample(wn, i+1, j, k)
        l = sample(wn, i, j, k)
        wx[i,j,k] = 0.5 * (r.x + l.x)
    for i, j, k in wy:
        t = sample(wn, i, j+1, k)
        b = sample(wn, i, j, k)
        wy[i,j,k] = 0.5 * (t.y + b.y)
    for i, j, k in wz:
        c = sample(wn, i, j, k+1)
        a = sample(wn, i, j, k)
        wz[i,j,k] = 0.5 * (c.z + a.z)

@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i,j,k].x = 0.5 * (vx[i+1, j, k] + vx[i, j, k])
        vc[i,j,k].y = 0.5 * (vy[i, j+1, k] + vy[i, j, k])
        vc[i,j,k].z = 0.5 * (vz[i, j, k+1] + vz[i, j, k])

@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i-1, j, k)
        vx[i,j,k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j-1, k)
        vy[i,j,k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k-1)
        vz[i,j,k] = 0.5 * (c.z + a.z)
    

def diffuse_field_implicit(field_temp, field, coe):
    copy_to(field, field_temp)
    for it in range(20):
        GS(field, field_temp, coe)
    copy_to(field_temp, field)

@ti.kernel
def GS(field:ti.template(), field_temp:ti.template(), coe:float):
    for i, j, k in field_temp:
        if (i + j + k)%2==0:
            field_temp[i, j, k] = (field[i, j, k] + coe * (
                                sample(field_temp, i - 1, j, k) +
                                sample(field_temp, i + 1, j, k) +
                                sample(field_temp, i, j - 1, k) +
                                sample(field_temp, i, j + 1, k) +
                                sample(field_temp, i, j, k - 1) +
                                sample(field_temp, i, j, k + 1)
                        )) / (1.0 + 6.0 * coe)
    for i, j, k in field_temp:
        if (i + j + k)%2==1:
            field_temp[i, j, k] = (field[i, j, k] + coe * (
                                sample(field_temp, i - 1, j, k) +
                                sample(field_temp, i + 1, j, k) +
                                sample(field_temp, i, j - 1, k) +
                                sample(field_temp, i, j + 1, k) +
                                sample(field_temp, i, j, k - 1) +
                                sample(field_temp, i, j, k + 1)
                        )) / (1.0 + 6.0 * coe)

# # # # interpolation kernels # # # # 
@ti.kernel
def interp_f2e(
    ff_x: ti.template(),
    ff_y: ti.template(),
    ff_z: ti.template(),
    ef_x: ti.template(),
    ef_y: ti.template(),
    ef_z: ti.template(),
):
    for i, j, k in ef_x:
        ef_x[i, j, k] = (
            sample(ff_x, i, j, k)
            + sample(ff_x, i, j, k-1)
            + sample(ff_x, i, j-1, k)
            + sample(ff_x, i, j-1, k-1)
            + sample(ff_x, i+1, j, k)
            + sample(ff_x, i+1, j, k-1)
            + sample(ff_x, i+1, j-1, k)
            + sample(ff_x, i+1, j-1, k-1)
        ) * 0.125

    for i, j, k in ef_y:
        ef_y[i, j, k] = (
            sample(ff_y, i, j, k)
            + sample(ff_y, i-1, j, k)
            + sample(ff_y, i, j, k-1)
            + sample(ff_y, i-1, j, k-1)
            + sample(ff_y, i, j+1, k)
            + sample(ff_y, i-1, j+1, k)
            + sample(ff_y, i, j+1, k-1)
            + sample(ff_y, i-1, j+1, k-1)
        ) * 0.125

    for i, j, k in ef_z:
        ef_z[i, j, k] = (
            sample(ff_z, i, j, k)
            + sample(ff_z, i, j-1, k)
            + sample(ff_z, i-1, j, k)
            + sample(ff_z, i-1, j-1, k)
            + sample(ff_z, i, j, k+1)
            + sample(ff_z, i, j-1, k+1)
            + sample(ff_z, i-1, j, k+1)
            + sample(ff_z, i-1, j-1, k+1)
        ) * 0.125

@ti.kernel
def interp_e2f(
    ef_x: ti.template(),
    ef_y: ti.template(),
    ef_z: ti.template(),
    ff_x: ti.template(),
    ff_y: ti.template(),
    ff_z: ti.template(),
):
    for i, j, k in ff_x:
        ff_x[i, j, k] = (
            sample(ef_x, i, j, k)
            + sample(ef_x, i, j+1, k)
            + sample(ef_x, i, j, k+1)
            + sample(ef_x, i, j+1, k+1)
            + sample(ef_x, i-1, j, k)
            + sample(ef_x, i-1, j+1, k)
            + sample(ef_x, i-1, j, k+1)
            + sample(ef_x, i-1, j+1, k+1)
        ) * 0.125

    for i, j, k in ff_y:
        ff_y[i, j, k] = (
            sample(ef_y, i, j, k)
            + sample(ef_y, i+1, j, k)
            + sample(ef_y, i, j, k+1)
            + sample(ef_y, i+1, j, k+1)
            + sample(ef_y, i, j-1, k)
            + sample(ef_y, i+1, j-1, k)
            + sample(ef_y, i, j-1, k+1)
            + sample(ef_y, i+1, j-1, k+1)
        ) * 0.125

    for i, j, k in ff_z:
        ff_z[i, j, k] = (
            sample(ef_z, i, j, k)
            + sample(ef_z, i, j+1, k)
            + sample(ef_z, i+1, j, k)
            + sample(ef_z, i+1, j+1, k)
            + sample(ef_z, i, j, k-1)
            + sample(ef_z, i, j+1, k-1)
            + sample(ef_z, i+1, j, k-1)
            + sample(ef_z, i+1, j+1, k-1)
        ) * 0.125

# linear
@ti.func
def N_1(x):
    return 1.0-ti.abs(x)
    
@ti.func
def dN_1(x):
    result = -1.0
    if x < 0.:
        result = 1.0
    return result

@ti.func
def interp_grad_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = ti.cast(0., dtype = data_type)
    partial_y = ti.cast(0., dtype = data_type)
    partial_z = ti.cast(0., dtype = data_type)
    interped = ti.cast(0., dtype = data_type)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i))
                partial_y += inv_dx * (value * N_1(x_p_x_i) * dN_1(y_p_y_i) * N_1(z_p_z_i))
                partial_z += inv_dx * (value * N_1(x_p_x_i) * N_1(y_p_y_i) * dN_1(z_p_z_i))
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])

@ti.func
def interp_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0. * sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped

@ti.func
def sample_min_max_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    mini = sample(vf, iu, iv, iw)
    maxi = sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                value = sample(vf, iu + i, iv + j, iw + k)
                mini = ti.min(mini, value)
                maxi = ti.max(maxi, value)

    return mini, maxi

@ti.func
def interp_2(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = ti.cast(0., dtype = data_type)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped

@ti.func
def interp_2_v(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = ti.Vector([ti.cast(0., dtype = data_type), ti.cast(0., dtype = data_type), ti.cast(0., dtype = data_type)])

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped

# quadratic
@ti.func
def N_2(x):
    result = ti.cast(0., dtype = data_type)
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = ti.cast(0., dtype = data_type)
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def interp_grad_2(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = ti.cast(0., dtype = data_type)
    partial_y = ti.cast(0., dtype = data_type)
    partial_z = ti.cast(0., dtype = data_type)
    interped = ti.cast(0., dtype = data_type)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
                partial_y += inv_dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
                partial_z += inv_dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])


@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

@ti.func
def judge_inside_w_boud_x(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i, j-1, k] + boundary_mask[i, j, k-1] + boundary_mask[i, j-1, k-1]
    return ret

@ti.func
def judge_inside_w_boud_y(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i-1, j, k] + boundary_mask[i-1, j, k-1] + boundary_mask[i, j, k-1]
    return ret

@ti.func
def judge_inside_w_boud_z(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i-1, j, k] + boundary_mask[i, j-1, k] + boundary_mask[i-1, j-1, k]
    return ret

@ti.kernel
def apply_bc_w(
    u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
    w_x: ti.template(), w_y: ti.template(), w_z: ti.template(),
    boundary_mask: ti.template(), boundary_vel: ti.template(), inv_dx: ti.f32):
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        vt = vb = vc = va = ti.cast(0., dtype = data_type)
        # 4 corners
        if (k == 0 and j == 0) or (k == 0 and j == v_dim - 1) or (k == w_dim - 1 and j == 0) or (k == w_dim - 1 and j == v_dim - 1):
            w_x[i, j, k] = ti.cast(0., dtype = data_type)
            continue

        if j == 0 or j == v_dim - 1 or k == 0 or k == w_dim - 1:
            continue
        
        num_solid_neighbors = judge_inside_w_boud_x(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_x[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # back has wall
            if boundary_mask[i, j-1, k-1] > 0 and boundary_mask[i, j, k-1]>0:
                va = boundary_vel[i, j-1, k-1][1] + boundary_vel[i, j, k-1][1] - u_y[i, j, k]
            else:
                va = u_y[i, j, k-1]

            # frd has wall
            if boundary_mask[i, j-1, k] > 0 and boundary_mask[i, j, k]>0:
                vc = boundary_vel[i, j-1, k][1] + boundary_vel[i, j, k][1] - u_y[i, j, k-1]
            else:
                vc = u_y[i, j, k]

            # bottom has wall
            if boundary_mask[i, j-1, k] > 0 and boundary_mask[i, j-1, k-1]>0:
                vb = boundary_vel[i, j-1, k][2] + boundary_vel[i, j-1, k-1][2] - u_z[i, j, k]
            else:
                vb = u_z[i, j-1, k]

            # top has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j, k-1]>0:
                vt = boundary_vel[i, j, k][2] + boundary_vel[i, j, k-1][2] - u_z[i, j-1, k]
            else:
                vt = u_z[i, j, k]

            w_x[i, j, k] = ((vt - vb) - (vc - va)) * inv_dx


    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        vc = va = vr = vl = ti.cast(0., dtype = data_type)
        # 4 corners
        if (k == 0 and i == 0) or (k == 0 and i == u_dim - 1) or (k == w_dim - 1 and i == 0) or (k == w_dim - 1 and i == u_dim - 1):
            w_y[i, j, k] = ti.cast(0., dtype = data_type)
            continue

        if i == 0 or i == u_dim - 1 or k == 0 or k == w_dim - 1:
            continue
        
        num_solid_neighbors = judge_inside_w_boud_y(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_y[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # frd
            if boundary_mask[i, j, k] > 0 and boundary_mask[i-1, j, k]>0:
                vc = boundary_vel[i, j, k][0] + boundary_vel[i-1, j, k][0] - u_x[i, j, k-1]
            else:
                vc = u_x[i, j, k]

            # back has wall
            if boundary_mask[i, j, k-1] > 0 and boundary_mask[i-1, j, k-1]>0:
                va = boundary_vel[i, j, k-1][0] + boundary_vel[i-1, j, k-1][0] - u_x[i, j, k]
            else:
                va = u_x[i, j, k-1]

            # right has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j, k-1]>0:
                vr = boundary_vel[i, j, k][2] + boundary_vel[i, j, k-1][2] - u_z[i-1, j, k]
            else:
                vr = u_z[i, j, k]

            # left has wall
            if boundary_mask[i-1, j, k] > 0 and boundary_mask[i-1, j, k-1]>0:
                vl = boundary_vel[i-1, j, k][2] + boundary_vel[i-1, j, k-1][2] - u_z[i, j, k]
            else:
                vl = u_z[i - 1, j, k]

            w_y[i, j, k] = ((vc - va) - (vr - vl)) * inv_dx

    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        vr = vl = vt = vb = ti.cast(0., dtype = data_type)
        # 4 corners
        if (j == 0 and i == 0) or (j == 0 and i == u_dim - 1) or (j == v_dim - 1 and i == 0) or (j == v_dim - 1 and i == u_dim - 1):
            w_z[i, j, k] = ti.cast(0., dtype = data_type)
            continue

        if i == 0 or i == u_dim - 1 or j == 0 or j == v_dim - 1:
            continue
        
        num_solid_neighbors = judge_inside_w_boud_z(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_z[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # right has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j-1, k]>0:
                vr = boundary_vel[i, j, k][1] + boundary_vel[i, j-1, k][1] - u_y[i-1, j, k]
            else:
                vr = u_y[i, j, k]

            # left has wall
            if boundary_mask[i-1, j, k] > 0 and boundary_mask[i-1, j-1, k]>0:
                vl = boundary_vel[i-1, j, k][1] + boundary_vel[i-1, j-1, k][1] - u_y[i, j, k]
            else:
                vl = u_y[i-1, j, k]

            # top has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i-1, j, k]>0:
                vt = boundary_vel[i, j, k][0] + boundary_vel[i-1, j, k][0] - u_x[i, j - 1, k]
            else:
                vt = u_x[i, j, k]

            # bottom has wall
            if boundary_mask[i-1, j-1, k] > 0 and boundary_mask[i, j-1, k]>0:
                vb = boundary_vel[i-1, j-1, k][0] + boundary_vel[i, j-1, k][0] - u_x[i, j, k]
            else:
                vb = u_x[i, j - 1, k]

            w_z[i, j, k] = ((vr - vl) - (vt - vb)) * inv_dx