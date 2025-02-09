import math
import time

from taichi_utils import *
from hyperparameters import *


@ti.data_oriented
class MGPCG_w2v:
    """
    Grid-based MGPCG solver for the possion equation.

    .. note::

        This solver only runs on CPU and CUDA backends since it requires the
        ``pointer`` SNode.
    """

    def __init__(self, boundary_types, boundary_mask, boundary_vel,  N, N_together, dim=2, base_level=3, dx=1.0, real=float):
        """
        :parameter dim: Dimensionality of the fields.
        :parameter N: Grid resolutions.
        :parameter n_mg_levels: Number of multigrid levels.
        """

        # grid parameters
        self.use_multigrid = True

        self.N = N
        self.N_together = N_together
        self.n_mg_levels = int(math.log2(min(N_together))) - base_level + 1
        self.pre_and_post_smoothing = 4
        self.bottom_smoothing = 50
        self.dim = dim
        self.real = real
        self.dx = dx
        self.inv_dx = 1.0 / dx

        # setup sparse simulation data arrays
        self.r = [
            ti.field(dtype=self.real) for _ in range(self.n_mg_levels)
        ]  # residual
        self.bm = [boundary_mask] + [ti.field(dtype=ti.i32)
                  for _ in range(self.n_mg_levels - 1)]  # boundary_mask
        self.bm_v = boundary_vel
        self.z = [
            ti.field(dtype=self.real) for _ in range(self.n_mg_levels)
        ]  # M^-1 self.r
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        self.sum = ti.field(dtype=self.real)  # storage for reductions
        self.sum_double = ti.field(dtype = ti.f64)
        self.r_mean = ti.field(dtype=self.real)  # storage for avg of r
        self.num_entries = math.prod(self.N)

        indices = ti.ijk if self.dim == 3 else ti.ij

        shape_list = []
        for n in self.N_together:
            if n % 4 > 0:
                shape_list.append(n // 4 + 1)
            else:
                shape_list.append(n // 4)
        self.grid = (
            ti.root.pointer(indices, shape_list)
            .dense(indices, 4)
            .place(self.x, self.p, self.Ap)
        )
        # print(shape_list)

        for l in range(self.n_mg_levels):
            shape_list = []
            for n in self.N_together:
                if n % (4 * 2**l) > 0:
                    shape_list.append(n // (4 * 2**l) + 1)
                else:
                    shape_list.append(n // (4 * 2**l))
            self.grid = (
                ti.root.pointer(indices, shape_list)
                .dense(indices, 4)
                .place(self.r[l], self.z[l])
            )

        for l in range(1, self.n_mg_levels):
            shape_list = []
            for n in self.N:
                if n % (4 * 2**l) > 0:
                    shape_list.append(n // (4 * 2**l) + 1)
                else:
                    shape_list.append(n // (4 * 2**l))
            ti.root.dense(indices,
                            shape_list).dense(
                                indices, 4).place(self.bm[l])

        ti.root.place(self.alpha, self.beta, self.sum, self.r_mean, self.sum_double)

        self.boundary_types = boundary_types

    @ti.func
    def init_r(self, I, r_I):
        self.r[0][I] = r_I
        self.z[0][I] = 0
        self.Ap[I] = 0
        self.p[I] = 0
        self.x[I] = 0

    @ti.kernel
    def init(self, r: ti.template(), k: ti.template()):
        """
        Set up the solver for $\nabla^2 x = k r$, a scaled Poisson problem.
        :parameter k: (scalar) A scaling factor of the right-hand side.
        :parameter r: (ti.field) Unscaled right-hand side.
        """
        dims = self.N
        for I in ti.grouped(ti.ndrange(*self.N_together)):
            ret = ti.cast(0.0, self.real)
            if I[0]!= self.N[0] - 1 \
            and (not (I[1] == self.N[1] - 1 and I[0] > self.N[0] - 1 and I[0] <= 2*self.N[0] - 1)) \
            and (not (I[2] == self.N[2] - 1 and I[0] > 2*self.N[0] - 1)):
                original_I = ti.Vector([I[0]%dims[0], I[1], I[2]])
                
                x_y_unit = ti.Vector([1, 0, 0])
                # 256 < i < 512
                if I[0] > self.N[0] - 1 and I[0] <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                # 512 < i
                elif I[0] > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])

                for i in ti.static(range(self.dim)):
                    offset = ti.Vector.unit(self.dim, i)
                    if x_y_unit[i] == 1:
                        if original_I[i] == dims[i] - 2 and i == 0:
                            ret += self.u_r_w
                        elif original_I[i] == dims[i] - 2 and i == 1:
                            ret += self.v_t_w
                        elif original_I[i] == dims[i] - 2 and i == 2:
                            ret += self.w_c_w

                        elif original_I[i] < dims[i] - 2: # 253
                            if self.bm[0][original_I + 2 * offset] > 0:
                                ret += self.bm_v[original_I + 2 * offset][i]
                        # add left if has left

                        if original_I[i] == 0 and i == 0:
                            ret += self.u_l_w
                        elif original_I[i] == 0 and i == 1:
                            ret += self.v_b_w
                        elif original_I[i] == 0 and i == 2:
                            ret += self.w_a_w

                        elif original_I[i] > 0:
                            if self.bm[0][original_I - offset] > 0:
                                ret += self.bm_v[original_I - offset][i]
                    else:
                        if original_I[i] < dims[i] - 1:
                            if (self.bm[0][original_I+offset] > 0):
                                ret -= self.bm_v[original_I + offset][i]
                            if (self.bm[0][original_I+offset+x_y_unit] > 0):
                                ret += self.bm_v[original_I+offset+x_y_unit][i]
                        # add left if has left
                        if original_I[i] > 0:
                            if (self.bm[0][original_I-offset] > 0):
                                ret += self.bm_v[original_I-offset][i]
                            if (self.bm[0][original_I-offset+x_y_unit] > 0):
                                ret -= self.bm_v[original_I-offset+x_y_unit][i]

            self.init_r(I, r[I] * k + ret)

    @ti.kernel
    def get_result(self, x: ti.template(), y: ti.template(), z:ti.template()):
        """
        Get the solution field.

        :parameter x: (ti.field) The field to store the solution
        """
        for i, j, k in ti.ndrange(*self.N):
            x[i + 1, j, k] = self.x[i, j, k]
            y[i, j + 1, k] = self.x[i + self.N[0], j, k]
            z[i, j, k + 1] = self.x[i + 2 * self.N[0], j, k] 


    @ti.func
    def neighbor_sum(self, x, I, bm):
        dims = (x.shape[0]//self.dim, x.shape[1], x.shape[2]) # 64, 64
        original_I = ti.Vector([I[0]%dims[0], I[1], I[2]])
        ret = ti.cast(0.0, self.real)
        x_y_unit = ti.Vector([1, 0, 0])
        if I[0] > dims[0] - 1 and I[0] <= 2 * dims[0] - 1:
            x_y_unit = ti.Vector([0, 1, 0])
        elif I[0] > 2 * dims[0] - 1:
            x_y_unit = ti.Vector([0, 0, 1])

        for i in ti.static(range(self.dim)):
            offset = ti.Vector.unit(self.dim, i)
            if x_y_unit[i] == 1:
                if original_I[i] < dims[i] - 2:
                    if bm[original_I + 2 * offset] <=0:
                        ret += x[I + offset]
                if original_I[i] > 0:
                    if bm[original_I - offset] <=0:
                        ret += x[I - offset]
            else:
                if original_I[i] < dims[i] - 1:
                    if ((bm[original_I+offset] <= 0) and (bm[original_I+offset+x_y_unit] <= 0)):
                        ret += x[I + offset]
                    elif ((bm[original_I+offset] <= 0) and (bm[original_I+offset+x_y_unit] > 0)):
                        ret -= x[ti.Vector([original_I[0] + i * dims[0], I[1], I[2]])]
                    elif ((bm[original_I+offset] > 0) and (bm[original_I+offset+x_y_unit] <= 0)):
                        ret += x[ti.Vector([original_I[0] + i * dims[0], I[1], I[2]]) + x_y_unit]

                if original_I[i] > 0:
                    if ((bm[original_I-offset] <= 0) and (bm[original_I-offset+x_y_unit] <= 0)):
                        ret += x[I - offset]
                    elif ((bm[original_I-offset] <= 0) and (bm[original_I-offset+x_y_unit] > 0)):
                        ret += x[ti.Vector([original_I[0] + i * dims[0], I[1], I[2]]) - offset]
                    elif ((bm[original_I-offset] > 0) and (bm[original_I-offset+x_y_unit] <= 0)):
                        ret -= x[ti.Vector([original_I[0] + i * dims[0], I[1], I[2]]) + x_y_unit - offset]

        return ret

    @ti.func
    def num_fluid_neighbors(self, x, I, bm):
        dims = (x.shape[0]//self.dim, x.shape[1], x.shape[2])
        original_I = ti.Vector([I[0]%dims[0], I[1], I[2]])
        num = 2.0 * self.dim
        x_y_unit = ti.Vector([1, 0, 0])
        if I[0] > dims[0] - 1 and I[0] <= 2 * dims[0] - 1:
            x_y_unit = ti.Vector([0, 1, 0])
        elif I[0] > 2 * dims[0] - 1:
            x_y_unit = ti.Vector([0, 0, 1])

        for i in ti.static(range(self.dim)):
            if x_y_unit[i] == 0:
                offset = ti.Vector.unit(self.dim, i)
                # bottom/left has wall
                if original_I[i] <= 0:
                    num -= 1.0
                elif bm[original_I - offset] > 0 or bm[original_I - offset + x_y_unit] > 0:
                    num -= 1.0
                # top/right has wall
                if original_I[i] >= dims[i] - 1:
                    num -= 1.0
                elif bm[original_I + offset] > 0 or bm[original_I + offset + x_y_unit] > 0:
                    num -= 1.0                   
        return num
    @ti.kernel
    def downsample_bm(self, bm_fine: ti.template(), bm_coarse: ti.template()):
        for I in ti.grouped(bm_coarse):
            I2 = I * 2
            all_solid = 1
            range_d = 2 * ti.Vector.one(ti.i32, self.dim)
            for J in ti.grouped(ti.ndrange(*range_d)):
                if bm_fine[I2 + J] <= 0:
                    all_solid = 0
            
            bm_coarse[I] = all_solid

    @ti.kernel
    def compute_Ap(self):
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):
                    multiplier = self.num_fluid_neighbors(self.p, ti.Vector([i, j, k]), self.bm[0])
                    self.Ap[i, j, k] = multiplier * self.p[i, j, k] - self.neighbor_sum(self.p, ti.Vector([i, j, k]), self.bm[0])

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):        
                    self.sum[None] += p[i, j, k] * q[i, j, k]

    @ti.kernel
    def reduce_double(self, p: ti.template(), q: ti.template()):
        self.sum_double[None] = 0.
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):        
                    self.sum_double[None] += ti.cast(p[i, j, k], ti.f64)  * ti.cast(q[i, j, k], ti.f64)
    @ti.kernel
    def update_x(self):
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):
                    self.x[i, j, k] += self.alpha[None] * self.p[i, j, k]

    @ti.kernel
    def update_r(self):
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):
                    self.r[0][i, j, k] -= self.alpha[None] * self.Ap[i, j, k]

    @ti.kernel
    def update_p(self):
        for i, j, k in ti.ndrange(*self.N_together):
            if i!= self.N[0] - 1 \
            and (not (j == self.N[1] - 1 and i > self.N[0] - 1 and i <= 2*self.N[0] - 1)) \
            and (not (k == self.N[2] - 1 and i > 2*self.N[0] - 1)):
                x_y_unit = ti.Vector([1, 0, 0])
                if i > self.N[0] - 1 and i <= 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 1, 0])
                elif i > 2 * self.N[0] - 1:
                    x_y_unit = ti.Vector([0, 0, 1])
                i_original = i%self.N[0]
                if (self.bm[0][i_original, j, k] <= 0 and self.bm[0][ti.Vector([i_original, j, k]) + x_y_unit] <= 0):
                    self.p[i, j, k] = self.z[0][i, j, k] + self.beta[None] * self.p[i, j, k]

    @ti.kernel
    def restrict(self, l: ti.template()):
        u_dim, v_dim, w_dim = self.r[l].shape
        original_dim_0 = (u_dim//self.dim)
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            x_y_unit = ti.Vector([1, 0, 0])
            if i > original_dim_0 - 1 and i <= 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 1, 0])
            elif i > 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 0, 1])

            if (i!= u_dim//self.dim - 1 and (not (j == v_dim - 1 and i > u_dim//self.dim - 1 and i <= 2*original_dim_0 - 1)) and (not (k == w_dim - 1 and i > 2 * original_dim_0 - 1))):
                i_original_dim_0 = i%original_dim_0
                if self.bm[l][i_original_dim_0, j, k] <= 0:
                    if self.bm[l][ti.Vector([i_original_dim_0, j, k]) + x_y_unit] <= 0:
                        multiplier = self.num_fluid_neighbors(self.z[l], ti.Vector([i, j, k]), self.bm[l])
                        res = self.r[l][i, j, k] - (
                            multiplier * self.z[l][i, j, k] - self.neighbor_sum(self.z[l], ti.Vector([i, j, k]), self.bm[l])
                        )
                        self.r[l + 1][ti.Vector([i, j, k]) // 2] += res * 1.0 / (self.dim - 1.0)
               
    @ti.kernel
    def prolongate(self, l: ti.template()):
        u_dim, v_dim, w_dim = self.z[l].shape
        original_dim_0 = (u_dim//self.dim)
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            x_y_unit = ti.Vector([1, 0, 0])
            if i > original_dim_0 - 1 and i <= 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 1, 0])
            elif i > 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 0, 1])
            
            if (i!= u_dim//self.dim - 1 and (not (j == v_dim - 1 and i > u_dim//self.dim - 1 and i <= 2*original_dim_0 - 1)) and (not (k == w_dim - 1 and i > 2 * original_dim_0 - 1))):
                i_original_dim_0 = i%original_dim_0
                if self.bm[l][i_original_dim_0, j, k] <= 0:
                    if self.bm[l][ti.Vector([i_original_dim_0, j, k]) + x_y_unit] <= 0:
                        self.z[l][i, j, k] += self.z[l + 1][ti.Vector([i, j, k]) // 2]

    @ti.kernel
    def smooth(self, l: ti.template(), phase: ti.template()):
        # phase = red/black Gauss-Seidel phase
        # print(self.r[l].shape, self.z[l].shape)
        u_dim, v_dim, w_dim = self.r[l].shape
        original_dim_0 = (u_dim//self.dim)
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            x_y_unit = ti.Vector([1, 0, 0])
            if i > original_dim_0 - 1 and i <= 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 1, 0])
            elif i > 2 * original_dim_0 - 1:
                x_y_unit = ti.Vector([0, 0, 1])

            if (i + j + k) & 1 == phase:
                if (i!= u_dim//self.dim - 1 and (not (j == v_dim - 1 and i > u_dim//self.dim - 1 and i <= 2*original_dim_0 - 1)) and (not (k == w_dim - 1 and i > 2 * original_dim_0 - 1))):
                    i_original_dim_0 = i%original_dim_0
                    if self.bm[l][i_original_dim_0, j, k] <= 0:
                        if self.bm[l][ti.Vector([i_original_dim_0, j, k]) + x_y_unit] <= 0:
                            multiplier = self.num_fluid_neighbors(self.z[l], ti.Vector([i, j, k]), self.bm[l])
                            self.z[l][i, j, k] = (
                                self.r[l][i, j, k] + self.neighbor_sum(self.z[l], ti.Vector([i, j, k]), self.bm[l])
                            ) / multiplier

    def apply_preconditioner(self):
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 0)
                self.smooth(l, 1)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1, 0)
            self.smooth(self.n_mg_levels - 1, 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing):
                self.smooth(l, 1)
                self.smooth(l, 0)

    def solve(self, max_iters=-1, eps=1e-12, tol=1e-12, verbose=False):
        """
        Solve a Poisson problem.

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :parameter eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        """
        # self.r = b - Ax = b    since self.x = 0
        # self.p = self.r = self.r + 0 self.p

        for l in range(1, self.n_mg_levels):
            self.downsample_bm(self.bm[l - 1], self.bm[l])
            # self.downsample_bm_v(self.bm_v[l - 1], self.bm_v[l])

        if self.use_multigrid:
            self.apply_preconditioner()
        else:
            self.z[0].copy_from(self.r[0]) # z = r

        # self.set_boundary(self.x)
        self.update_p()  # p = r

        self.reduce(self.z[0], self.r[0]) #rTr
        old_zTr = self.sum[None]

        # Conjugate gradients
        it = 0
        start_t = time.time()
        while max_iters == -1 or it < max_iters:
            # self.alpha = rTr / pTAp
            self.compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]
            self.alpha[None] = old_zTr / (pAp + eps)

            # self.x = self.x + self.alpha self.p
            self.update_x()

            # self.r = self.r - self.alpha self.Ap
            self.update_r()

            # check for convergence
            self.reduce(self.r[0], self.r[0])
            rTr = self.sum[None]
            # print(rTr)
            if verbose:
                print(f"iter {it}, |residual|_2={math.sqrt(rTr)}")

            if rTr < tol:
                end_t = time.time()
                print(
                    "[MGPCG] Converged at iter: ",
                    it,
                    " with final error: ",
                    math.sqrt(rTr),
                    " using time: ",
                    end_t - start_t,
                )
                return

            # self.z = M^-1 self.r
            if self.use_multigrid:
                self.apply_preconditioner()
            else:
                self.z[0].copy_from(self.r[0]) # z = r_k+1

            # self.beta = new_rTr / old_rTr
            self.reduce(self.z[0], self.r[0])
            new_zTr = self.sum[None]
            self.beta[None] = new_zTr / (old_zTr + eps)

            # self.p = self.z + self.beta self.p
            self.update_p()
            old_zTr = new_zTr

            it += 1

        end_t = time.time()
        print(
            "[MGPCG] Return without converging at iter: ",
            it,
            " with final error: ",
            math.sqrt(rTr),
            " using time: ",
            end_t - start_t,
        )


class MGPCG_3_w2v(MGPCG_w2v):
    def __init__(self, boundary_types, boundary_mask, boundary_vel, N, N_together,
                u_l_w, u_r_w, v_t_w, v_b_w, w_a_w, w_c_w, base_level=3, dx=1.0, real=ti.f32):

        super().__init__(boundary_types, boundary_mask, boundary_vel, N, N_together, dim=3, base_level=base_level, real=real, dx = dx)
        self.w_curl = ti.field(self.real, shape=N_together) # 256 + 256, 256
        self.boundary_types = boundary_types
        self.boundary_mask = boundary_mask
        self.boundary_vel = boundary_vel
        self.dx = dx
        self.u_l_w = u_l_w
        self.u_r_w = u_r_w
        self.v_t_w = v_t_w
        self.v_b_w = v_b_w
        self.w_a_w = w_a_w
        self.w_c_w = w_c_w
    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            if i == 0:
                u_x[i, j, k] = self.u_l_w
            elif  i == u_dim - 1:
                u_x[i, j, k] = self.u_r_w
            else:
                if self.boundary_mask[i-1, j, k] > 0:
                    u_x[i, j, k] = self.boundary_vel[i-1, j, k][0]
                if self.boundary_mask[i, j, k] > 0:
                    u_x[i, j, k] = self.boundary_vel[i, j, k][0]
        
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            if j == 0:
                u_y[i, j, k] = self.v_b_w
            elif j == v_dim - 1:
                u_y[i, j, k] = self.v_t_w
            else:
                if self.boundary_mask[i, j-1, k] > 0:
                    u_y[i, j, k] = self.boundary_vel[i, j-1, k][1]
                if self.boundary_mask[i, j, k] > 0:
                    u_y[i, j, k] = self.boundary_vel[i, j, k][1]

        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim):
            if k == 0:
                u_z[i, j, k] = self.w_a_w
            elif k == w_dim - 1:
                u_z[i, j, k] = self.w_c_w
            else:
                if self.boundary_mask[i, j, k-1] > 0:
                    u_z[i, j, k] = self.boundary_vel[i, j, k-1][2]
                if self.boundary_mask[i, j, k] > 0:
                    u_z[i, j, k] = self.boundary_vel[i ,j, k][2]

    @ti.kernel
    def curl_e2f_x(self, w_z: ti.template(), w_y: ti.template()):
        u_dim, v_dim, w_dim = self.N
        for i, j, k in ti.ndrange(u_dim - 1, v_dim, w_dim):
            wz_t = sample(w_z, i + 1, j + 1, k)
            wz_b = sample(w_z, i + 1, j, k)
            wy_a = sample(w_y, i + 1, j, k)
            wy_c = sample(w_y, i + 1, j, k + 1)
            self.w_curl[i, j, k] = ((wz_t - wz_b) - (wy_c - wy_a))

            # if bottom has wall
            if j == 0:
                self.w_curl[i, j, k] += wz_b
            elif self.bm[0][i, j - 1, k] > 0 or self.bm[0][i + 1, j - 1, k] > 0:
                self.w_curl[i, j, k] += wz_b

            # if top has wall
            if j == v_dim - 1:
                self.w_curl[i, j, k] -= wz_t
            elif self.bm[0][i, j + 1, k] > 0 or self.bm[0][i + 1, j + 1, k] > 0:
                self.w_curl[i, j, k] -= wz_t

            # if back has wall
            if k == 0:
                self.w_curl[i, j, k] -= wy_a
            elif self.bm[0][i, j, k - 1] > 0 or self.bm[0][i + 1, j, k - 1] > 0:
                self.w_curl[i, j, k] -= wy_a

            # if frd has wall
            if k == w_dim - 1:
                self.w_curl[i, j, k] += wy_c
            elif self.bm[0][i, j, k + 1] > 0 or self.bm[0][i + 1, j, k + 1] > 0:
                self.w_curl[i, j, k] += wy_c

    @ti.kernel
    def curl_e2f_y(self, w_x: ti.template(), w_z: ti.template()):
        u_dim, v_dim, w_dim = self.N
        for i, j, k in ti.ndrange(u_dim, v_dim - 1, w_dim):
            wx_c = sample(w_x, i, j + 1, k + 1)
            wx_a = sample(w_x, i, j + 1, k)
            wz_r = sample(w_z, i + 1, j + 1, k)
            wz_l = sample(w_z, i, j + 1, k)
            self.w_curl[i + u_dim, j, k] = ((wx_c - wx_a) - (wz_r - wz_l))

            # if left has wall
            if i == 0:
                self.w_curl[i + u_dim, j, k] -= wz_l
            elif self.bm[0][i - 1, j, k] > 0 or self.bm[0][i - 1, j + 1, k] > 0:
                self.w_curl[i + u_dim, j, k] -= wz_l

            # if right has wall
            if i == u_dim - 1:
                self.w_curl[i + u_dim, j, k] += wz_r
            elif self.bm[0][i + 1, j, k] > 0 or self.bm[0][i + 1, j + 1, k] > 0:
                self.w_curl[i + u_dim, j, k] += wz_r
            
            # if back has wall
            if k == 0:
                self.w_curl[i + u_dim, j, k] += wx_a
            elif self.bm[0][i, j, k - 1] > 0 or self.bm[0][i, j + 1, k - 1] > 0:
                self.w_curl[i + u_dim, j, k] += wx_a
            
            # if frd has wall
            if k == w_dim - 1:
                self.w_curl[i + u_dim, j, k] -= wx_c
            elif self.bm[0][i, j, k + 1] > 0 or self.bm[0][i, j + 1, k + 1] > 0:
                self.w_curl[i + u_dim, j, k] -= wx_c


    @ti.kernel
    def curl_e2f_z(self, w_y: ti.template(), w_x: ti.template()):
        u_dim, v_dim, w_dim = self.N
        for i, j, k in ti.ndrange(u_dim, v_dim, w_dim - 1):
            wy_r = sample(w_y, i + 1, j, k + 1)
            wy_l = sample(w_y, i, j, k + 1)
            wx_t = sample(w_x, i, j + 1, k + 1)
            wx_b = sample(w_x, i, j, k + 1)
            self.w_curl[i + 2*u_dim, j, k] = ((wy_r - wy_l) - (wx_t - wx_b))

            # if left has wall
            if i == 0:
                self.w_curl[i + 2*u_dim, j, k] += wy_l
            elif self.bm[0][i - 1, j, k] > 0 or self.bm[0][i - 1, j, k + 1] > 0:
                self.w_curl[i + 2*u_dim, j, k] += wy_l

            # if right has wall
            if i == u_dim - 1:
                self.w_curl[i + 2*u_dim, j, k] -= wy_r
            elif self.bm[0][i + 1, j, k] > 0 or self.bm[0][i + 1, j, k + 1] > 0:
                self.w_curl[i + 2*u_dim, j, k] -= wy_r
            
            # if bottom has wall
            if j == 0:
                self.w_curl[i + 2*u_dim, j, k] -= wx_b
            elif self.bm[0][i, j - 1, k] > 0 or self.bm[0][i, j - 1, k + 1] > 0:
                self.w_curl[i + 2*u_dim, j, k] -= wx_b
            
            # if top has wall
            if j == v_dim - 1:
                self.w_curl[i + 2*u_dim, j, k] += wx_t
            elif self.bm[0][i, j + 1, k] > 0 or self.bm[0][i, j + 1, k + 1] > 0:
                self.w_curl[i + 2*u_dim, j, k] += wx_t

    def solve_velocity_MGPCG(self, u_x, u_y, u_z, verbose):
        self.init(self.w_curl, self.dx)
        self.solve(max_iters=5000, verbose=verbose, tol=1.0e-12)
        self.get_result(u_x, u_y, u_z)

    def Poisson_w2v(self, u_x, u_y, u_z, w_x, w_y, w_z, verbose=False):
        # self.apply_bc(w_x, w_y, w_z)
        # compute curl of omega, get 3 rhs, the curl is stored in w_curl_xyz
        self.curl_e2f_x(w_z, w_y)
        self.curl_e2f_y(w_x, w_z)
        self.curl_e2f_z(w_y, w_x)
        self.solve_velocity_MGPCG(u_x, u_y, u_z, verbose=verbose)
        self.apply_bc(u_x, u_y, u_z)