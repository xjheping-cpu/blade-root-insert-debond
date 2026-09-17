# -*- coding: utf-8 -*-
"""
M3：叶根预埋螺套连接的"分层"三维扇形块有限元模型。

本模块是 sector3d.Sector3D 的子类，**不修改**已通过验证的父模型，只做三件事：

  1. 分层几何。父模型把预埋螺套简化成"钢 + 一圈折减剪切模量的层压"。这里把真实
     构造自内向外显式分成五层（距螺套轴线的距离 d，单位 mm）：

         d < d_bore/2         螺孔（M42 小径）。盲孔底以内为空孔 MAT_VOID，
                              盲孔底以外为实心钢 MAT_STEEL。
         d_bore/2 <= d < D_ins/2   钢衬套 MAT_STEEL，42CrMoA，E=210 GPa，nu=0.3
         再往外 0.5 mm       过渡层（树脂富集）。太薄不单独划网格，
                              照搬父模型 G_eff 的柔度串联法等效（见 _classify）。
         过渡层以外 6 mm     玻纤束缠绕层 MAT_WRAP  【推断参数】
         45.0  <= d < 52.0    拉挤 GFRP 块 MAT_BLOCK 【推断参数】
         d >= 52.0            叶根层压 MAT_LAM（沿用父模型参数）

     为什么必须显式分开：He 等 2025 的实测失效面不在钢/胶界面，而在玻纤束缠绕层
     与拉挤块/层压之间。没有这两层，就没法把脱粘放在正确的半径上。

  2. 界面脱粘 set_debond()。把跨越界面半径的那一圈单元的三个剪切模量降到 1e-4 倍，
     法向（拉压）刚度保留。**这是"剪切失效等效"，不是接触单元、也不是内聚力单元**：
     它不能承受法向张开/闭合的非线性，也不给出界面的能量释放率，只表达"这一圈面
     上不再传递剪力"。与父模型 G_eff 的做法同源，只是把附加柔度推到极限。

  3. 三维场导出。把单元轴向应变、von Mises 应力、最大剪应力、结点位移、内表面
     (r=0) 轴向应变按 (nx_e, ny, nr_e) 的规整网格存成 out/sector_layered.npz。

坐标约定（与父模型一致）
    x  轴向，0 = 叶根端面/法兰面，900 = 远端夹支
    y  周向展开坐标，周期，螺栓节距 P_PITCH
    r  径向厚度坐标，0 = 叶根内表面（光纤所在），t_wall = 外表面；螺套轴线在 r = r_inner_off
    拉为正；长度 mm，力 N，应力 MPa。

三维渲染用的真实空间映射（同样写在 out/README_field.txt 里）
    真实半径   R     = R_IN + r            (mm)   # R_IN = 叶根内表面半径
    周向角     theta = y / R_bc             (rad)  # R_bc = 螺栓圆半径
    轴向       X     = x                    (mm)
    笛卡尔     (X, R*sin(theta), R*cos(theta))
    这样可以把 npz 里的任何 (nx_e, ny, nr_e) 场直接铺到真实叶根圆筒上。

用法
    python sector3d_layered.py             # 跑全部 7 个工况并写 npz + README
    python sector3d_layered.py --quick     # 粗网格冒烟测试
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from sector3d import (                      # noqa: E402  （父模型，只读不改）
    Sector3D, C_ortho, C_iso, C_lam, _B_at, _x_grid,
    P_PITCH, T_WALL, X_MAX, L_INS, R_INS_C, D_INS, D_BORE, L_ENG,
    E_X, E_Y, E_R, G_XY, G_XR, G_YR, NU_XY, NU_XR, NU_YR,
    E_ST, NU_ST, E_VOID, NU_VOID, G_ADH, T_ADH, R_IN,
    K_S, FS_INSTALL, MAT_LAM, MAT_STEEL, MAT_VOID, XCHK,
)

# ------------------------------------------------------------------ 新增材料
MAT_WRAP, MAT_BLOCK = 3, 4
MAT_NAMES = np.array(['lam', 'steel', 'void', 'wrap', 'block'])

# 分层半径（到螺套轴线的距离 d）
D_BORE_R = 0.5 * D_BORE          # 18.75  螺孔
D_STEEL_O = 0.5 * D_INS          # 钢衬套外半径，随几何配置
# 以下三层厚度为工艺推断值，与机型无关，按增量叠在钢体外径上【推断】
D_TRANS_O = D_STEEL_O + 0.5      # 过渡层外半径（0.5 mm 树脂富集，不单独划网格）
D_WRAP_O = D_STEEL_O + 6.5       # 缠绕层外半径  → 界面 B
D_BLOCK_O = D_STEEL_O + 13.5     # 拉挤块外半径  → 界面 C
X_BORE = 130.0                   # 螺孔深度：x < 130 空孔，x >= 130 实心钢

# ---- 【推断参数】玻纤束缠绕层（缠绕玻纤/环氧）
E_WRAP_X, E_WRAP_H, G_WRAP, NU_WRAP = 22000.0, 20000.0, 5500.0, 0.35
# ---- 【推断参数】拉挤 GFRP 块
E_BLK_X, E_BLK_T, G_BLK, NU_BLK = 45000.0, 14000.0, 4500.0, 0.30

# 脱粘：剪切模量折减到原值的 DEBOND_GFAC 倍
DEBOND_GFAC = 1.0e-4

# 各材料的"名义"剪切模量（脱粘折减的基准）
G_OF_MAT = np.array([G_XY, E_ST / (2.0 * (1.0 + NU_ST)),
                     E_VOID / (2.0 * (1.0 + NU_VOID)), G_WRAP, G_BLK])

# 工况用的载荷
FA_RATED = 159.0e3               # 额定挥舞单柱外载 N
FS_TARGET = 420.0e3              # 安装预紧轴力 N

INTERFACE_D = {'B': D_WRAP_O, 'C': D_BLOCK_O}
INTERFACE_TXT = {'B': '缠绕层/拉挤块 (d=45.0)', 'C': '拉挤块/层压 (d=52.0)'}


# ====================================================================== 本构
def C_wrap(G=None):
    """玻纤束缠绕层。轴向 E_x、环向/径向 E_h，三个剪切模量相同。"""
    g = G_WRAP if G is None else float(G)
    return C_ortho(E_WRAP_X, E_WRAP_H, E_WRAP_H,
                   NU_WRAP, NU_WRAP, NU_WRAP, g, g, g)


def C_block(G=None):
    """拉挤 GFRP 块。轴向 E_x 高、横向 E_t 低。"""
    g = G_BLK if G is None else float(G)
    return C_ortho(E_BLK_X, E_BLK_T, E_BLK_T,
                   NU_BLK, NU_BLK, NU_BLK, g, g, g)


def C_iso_G(E, nu, G):
    """各向同性材料但剪切模量被人为替换（脱粘用）。"""
    return C_ortho(E, E, E, nu, nu, nu, G, G, G)


# ================================================================== 稀疏直接解
# 分层以后自由度从父模型的 9 万涨到 28 万。实测同一个矩阵：
#     scipy 的 SuperLU (MMD_AT_PLUS_A)  1704 s，填入 1.42e9 非零元（15.9 GB）
#     MKL PARDISO (嵌套剖分, 16 线程)     23 s，约 6 GB
# 所以优先用 PARDISO；装不上就退回 SuperLU（结果一样，只是要等 28 分钟一次）。
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
try:
    from pypardiso.pardiso_wrapper import PyPardisoSolver as _PyPardisoSolver
    HAVE_PARDISO = True
except Exception:                                                  # noqa: BLE001
    _PyPardisoSolver = None
    HAVE_PARDISO = False


class SparseLU:
    """稀疏直接解的统一外壳，只需要 .solve(b)（b 可以是一维或多右端）。"""

    backend = 'pardiso' if HAVE_PARDISO else 'superlu'

    def __init__(self, A):
        if HAVE_PARDISO:
            self.A = A.tocsr()
            self.A.sort_indices()
            self.ps = _PyPardisoSolver()
            self.ps.set_iparm(8, 2)            # iparm(8)：最多 2 步迭代精化（1 基编号）
            self.ps.factorize(self.A)
        else:
            self.lu = spla.splu(A.tocsc(), permc_spec='COLAMD',
                                diag_pivot_thresh=0.0,
                                options=dict(SymmetricMode=True))

    def solve(self, b):
        if HAVE_PARDISO:
            return self.ps.solve(self.A, np.ascontiguousarray(b, dtype=float))
        return self.lu.solve(b)

    def free(self):
        if HAVE_PARDISO:
            try:
                self.ps.free_memory(everything=True)
            except Exception:                                      # noqa: BLE001
                pass
            self.A = None
        else:
            self.lu = None


# ================================================================== 主模型
class SectorLayered(Sector3D):
    """分层三维扇形块模型。

    与父模型 Sector3D 的差别只有材料分区、径向网格和界面脱粘；网格拓扑、周期条件、
    端面接触主动集、螺柱秩一刚度、Woodbury 求解、后处理接口全部沿用父类。

    实现上的一个约定：父类 __init__ 的调用次序是
        …设 x/y/r 网格与 n_node/ndof… → _classify() → _build_eldof() → _assemble()
    所以本类在 _classify() 的第一行重建径向网格（_rebuild_radial），随后的
    _build_eldof / _assemble / _face_and_stud 自然拿到新的 nr_n / n_node / ndof。
    这样就不必复制父类 __init__ 的代码，也就不会和父类漂移。

    参数
        m_y     每个 cell 的周向单元数（任务要求 20 → dy = 4.26 mm）
        n_cell  周向建模的 cell 数（任务要求 5，控制规模）
        layers  (n_lam, n_block, n_wrap, n_steel, n_bore)
                各层在径向的单元数；内外侧对称使用。默认 (2, 2, 4, 4, 4)。
    """

    def __init__(self, m_y=20, n_cell=5, layers=(2, 2, 4, 4, 4),
                 with_inserts=True, hoop='global', verbose=True, **kw):
        self._layers = tuple(int(v) for v in layers)
        self._debonds = []
        self._coo_rc = None
        self._Mcache = {}
        super().__init__(m_y=m_y, r_div=self._layers, n_cell=n_cell,
                         with_inserts=with_inserts, hoop=hoop, verbose=verbose, **kw)

    # -------------------------------------------------------------- 径向网格
    def _rebuild_radial(self):
        """按分层界面重建径向结点，使每一层的边界都落在结点面上。

        r = R_INS_C -+ d。内侧界面依次是拉挤块 / 缠绕层 / 钢衬套 / 螺孔，
        外侧依次是螺孔 / 钢衬套 / 缠绕层 / 拉挤块。最外那道界面距外表面
        只剩零点几毫米层压皮，单独划一层单元会出现畸形单元，
        因此并入最外一层拉挤块单元（形心判据下它本来也会被判成拉挤块）。
        过渡层（钢衬套外那 0.5 mm）同样不单独划网格（见类文档）。
        """
        nl, nb, nw, ns, nc = self._layers
        r_blk_i = R_INS_C - D_BLOCK_O
        r_wrp_i = R_INS_C - D_WRAP_O
        r_stl_i = R_INS_C - D_STEEL_O
        r_bor_i = R_INS_C - D_BORE_R
        r_bor_o = R_INS_C + D_BORE_R
        r_stl_o = R_INS_C + D_STEEL_O
        r_wrp_o = R_INS_C + D_WRAP_O
        segs = [(0.0, r_blk_i, nl), (r_blk_i, r_wrp_i, nb), (r_wrp_i, r_stl_i, nw),
                (r_stl_i, r_bor_i, ns), (r_bor_i, r_bor_o, nc), (r_bor_o, r_stl_o, ns),
                (r_stl_o, r_wrp_o, nw), (r_wrp_o, T_WALL, nb)]
        self.r_seg = segs
        r = [0.0]
        for a, b, n in segs:
            r.extend(np.linspace(a, b, n + 1)[1:])
        self.r = np.array(r, float)
        self.nr_n = len(self.r)
        self.nr_e = self.nr_n - 1
        self.dr = np.diff(self.r)
        self.rc = 0.5 * (self.r[:-1] + self.r[1:])
        self.n_node = self.nx_n * self.ny * self.nr_n
        self.n_elem = self.nx_e * self.ny * self.nr_e
        self.i_hoop = 3 * self.n_node
        self.ndof = self.i_hoop + self.n_hoop

    # -------------------------------------------------------------- 材料判定
    def _classify(self):
        self._rebuild_radial()

        # ---- 周向：到最近螺套轴线的距离（单元形心）
        d = (self.yc - 0.5 * P_PITCH) % P_PITCH
        self.dy_c = np.minimum(d, P_PITCH - d)
        self.cell_of_col = np.floor(self.yc / P_PITCH).astype(int) % self.n_cell
        dr_c = self.rc - R_INS_C
        self.dr_c = dr_c
        dist = np.hypot(self.dy_c[:, None], dr_c[None, :])         # (ny, nr_e)
        self.dist = dist

        # ---- 单元在 (y,r) 截面里的最小/最大轴线距（脱粘环的"跨越"判据）
        t = np.linspace(0.0, 1.0, 9)
        ys = self.y[:, None] + t[None, :] * self.dy
        aa = (ys - 0.5 * P_PITCH) % P_PITCH
        aa = np.minimum(aa, P_PITCH - aa)
        a_lo, a_hi = aa.min(axis=1), aa.max(axis=1)
        r0, r1 = self.r[:-1] - R_INS_C, self.r[1:] - R_INS_C
        dr_lo = np.where(r0 * r1 <= 0.0, 0.0, np.minimum(np.abs(r0), np.abs(r1)))
        dr_hi = np.maximum(np.abs(r0), np.abs(r1))
        self.d_cmin = np.hypot(a_lo[:, None], dr_lo[None, :])
        self.d_cmax = np.hypot(a_hi[:, None], dr_hi[None, :])

        # ---- 截面材料（形心判据）。螺孔先当钢，随后按 x 改成空孔。
        m2 = np.where(dist < D_STEEL_O, MAT_STEEL,
                      np.where(dist < D_WRAP_O, MAT_WRAP,
                               np.where(dist < D_BLOCK_O, MAT_BLOCK, MAT_LAM)))
        m2 = m2.astype(np.int8)
        core = dist < D_BORE_R
        self.in_out = dist < D_STEEL_O            # 钢外圆以内（父类 interface_slip 用）
        self.in_bore = core
        self.in_x = self.xc < L_INS

        if not self.with_inserts:
            m2 = np.full_like(m2, MAT_LAM)
            self.in_out = np.zeros_like(self.in_out)
            self.in_bore = np.zeros_like(self.in_bore)
            core = np.zeros_like(core)

        mat = np.where(self.in_x[:, None, None],
                       np.broadcast_to(m2[None, :, :],
                                       (self.nx_e, self.ny, self.nr_e)),
                       np.int8(MAT_LAM)).astype(np.int8)
        if self.with_inserts:
            void3 = (self.in_x & (self.xc < X_BORE))[:, None, None] & core[None, :, :]
            mat[void3] = MAT_VOID
            self.n_void = int(void3.sum())
        else:
            self.n_void = 0
        self.mat = mat
        self.deb = np.zeros(mat.shape, bool)      # 脱粘单元掩码

        # ---- 阶梯化截面积（cell 0，x<130 的横截面）
        cell0 = np.flatnonzero(self.cell_of_col == 0)
        A = np.broadcast_to(self.dy * self.dr[None, :], (len(cell0), self.nr_e))
        self.A_mesh = {}
        ix_bore = int(np.flatnonzero(self.xc < X_BORE)[-1]) if self.with_inserts else 0
        mcs = mat[ix_bore][cell0] if self.with_inserts else \
            np.full((len(cell0), self.nr_e), MAT_LAM, np.int8)
        for m in range(5):
            self.A_mesh[int(m)] = float(A[mcs == m].sum())
        self.A_steel_mesh = self.A_mesh[MAT_STEEL]
        self.A_bore_mesh = self.A_mesh[MAT_VOID]
        self.A_lam_mesh = self.A_mesh[MAT_LAM]
        self.A_exact = exact_layer_areas()

        # ---- 过渡层（0.5 mm 树脂富集）：紧贴钢外圆的一圈缠绕层单元，串联附加剪切柔度
        #      1/G_eff = 1/G_wrap + t_trans/(G_trans*h_e)，h_e = 单元沿界面法向的穿越长度
        st = self.in_out
        nb = np.zeros_like(st)
        nb[1:, :] |= st[:-1, :]
        nb[:-1, :] |= st[1:, :]
        nb[0, :] |= st[-1, :]
        nb[-1, :] |= st[0, :]                    # y 周期
        nb[:, 1:] |= st[:, :-1]
        nb[:, :-1] |= st[:, 1:]
        ring = nb & ~st
        self.ring_mask = ring

        G_eff = np.full((self.ny, self.nr_e), float(G_WRAP))
        h_ring = np.zeros((self.ny, self.nr_e))
        for iy, ir in np.argwhere(ring):
            ay, ar = self.dy_c[iy], dr_c[ir]
            nn = float(np.hypot(ay, ar))
            if nn < 1e-12:
                continue
            uy, ur = abs(ay) / nn, abs(ar) / nn
            h = min(self.dy / uy if uy > 1e-9 else np.inf,
                    self.dr[ir] / ur if ur > 1e-9 else np.inf)
            h_ring[iy, ir] = h
            G_eff[iy, ir] = 1.0 / (1.0 / G_WRAP + T_ADH / (G_ADH * h))
        self.G_eff, self.h_ring = G_eff, h_ring

    # ------------------------------------------------------------- 本构查表
    def _C_of(self, m, G=None):
        m = int(m)
        if m == MAT_LAM:
            return C_lam() if G is None else C_lam(Gxy=G, Gxr=G, Gyr=G)
        if m == MAT_STEEL:
            return C_iso(E_ST, NU_ST) if G is None else C_iso_G(E_ST, NU_ST, G)
        if m == MAT_VOID:
            return C_iso(E_VOID, NU_VOID) if G is None else C_iso_G(E_VOID, NU_VOID, G)
        if m == MAT_WRAP:
            return C_wrap(G)
        if m == MAT_BLOCK:
            return C_block(G)
        raise ValueError('未知材料编号 %d' % m)

    def _build_ckey(self):
        """把每个单元映射到一个本构编号 ckey，并给出本构表 Clist。

        分组依据 = (材料编号, 被覆盖的剪切模量)。被覆盖的情形有两种：
          · 过渡层等效：紧贴钢的缠绕层单元用 G_eff
          · 脱粘：跨界面的一圈单元用 DEBOND_GFAC * G_材料
        脱粘优先于过渡层。
        """
        Gov = np.full(self.mat.shape, np.nan)
        if self.with_inserts and self.ring_mask.any():
            ring3 = np.zeros(self.mat.shape, bool)
            ring3[self.in_x] = self.ring_mask[None, :, :]
            G3 = np.zeros(self.mat.shape)
            G3[self.in_x] = self.G_eff[None, :, :]
            Gov[ring3] = G3[ring3]
            self.ring3 = ring3
        else:
            self.ring3 = np.zeros(self.mat.shape, bool)
        if self.deb.any():
            Gov[self.deb] = DEBOND_GFAC * G_OF_MAT[self.mat[self.deb]]

        gkey = np.nan_to_num(np.round(Gov, 6), nan=-1.0)
        pairs = np.stack([self.mat.ravel().astype(float), gkey.ravel()], axis=1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        self.ckey = inv.reshape(self.mat.shape).astype(np.int32)
        self.Clist = [self._C_of(int(m), None if g < 0 else float(g)) for m, g in uniq]
        self.ckey_desc = [(int(m), None if g < 0 else float(g)) for m, g in uniq]

    # ------------------------------------------------------------- COO 结构
    def _build_coo_rc(self):
        """刚度阵 COO 的行列索引。它只依赖网格拓扑，与材料/脱粘无关，
        所以只建一次；每次重装配只重算数值。"""
        d = self.ELDOF.astype(np.int32)
        R = np.repeat(d, 24, axis=1).ravel()
        C = np.tile(d, (1, 24)).ravel()
        if self.hoop_free:
            nS = self.nx_e * self.nr_e
            hr = np.empty((nS, 208), np.int32)
            hc = np.empty((nS, 208), np.int32)
            ih = np.int32(self.i_hoop)
            for ix in range(self.nx_e):
                for ir in range(self.nr_e):
                    es = (ix * self.ny + self.ny - 1) * self.nr_e + ir
                    ds = self.ELDOF[es].astype(np.int32)
                    k = ix * self.nr_e + ir
                    hr[k, :96] = ih
                    hc[k, :96] = np.tile(ds, 4)
                    hr[k, 96:192] = np.tile(ds, 4)
                    hc[k, 96:192] = ih
                    hr[k, 192:] = ih
                    hc[k, 192:] = ih
            R = np.concatenate([R, hr.ravel()])
            C = np.concatenate([C, hc.ravel()])
            self._n_hoop_ent = nS * 208
        else:
            hh = (self.i_hoop + np.arange(self.n_hoop)).astype(np.int32)
            R = np.concatenate([R, hh])
            C = np.concatenate([C, hh])
            self._n_hoop_ent = self.n_hoop
        self._coo_rc = (R, C)

    # ---------------------------------------------------------------- 装配
    def _assemble(self):
        t0 = time.time()
        self._build_ckey()
        nE = self.n_elem
        nent = nE * 576 + (self.nx_e * self.nr_e * 208 if self.hoop_free else self.n_hoop)
        Vall = np.zeros(nent)
        V = Vall[:nE * 576].reshape(nE, 576)
        if self.hoop_free:
            H = Vall[nE * 576:].reshape(self.nx_e * self.nr_e, 208)
        else:
            Vall[nE * 576:] = 1.0
        W = self.WRAP_LOC
        ke_cache = {}
        for ix in range(self.nx_e):
            for ir in range(self.nr_e):
                gid = self.ckey[ix, :, ir]
                for g in np.unique(gid):
                    sel = np.flatnonzero(gid == g)
                    kk = (ix, ir, int(g))
                    ke = ke_cache.get(kk)
                    if ke is None:
                        ke = self._ke(ix, ir, self.Clist[int(g)])
                        ke_cache[kk] = ke
                    e = (ix * self.ny + sel) * self.nr_e + ir
                    V[e] = ke.ravel()
                if self.hoop_free:
                    ke = ke_cache[(ix, ir, int(gid[self.ny - 1]))]
                    k = ix * self.nr_e + ir
                    H[k, :96] = ke[W, :].ravel()
                    H[k, 96:192] = ke[:, W].T.ravel()
                    H[k, 192:] = ke[np.ix_(W, W)].ravel()
        if self._coo_rc is None:
            self._build_coo_rc()
        R, C = self._coo_rc
        self.K0 = sp.coo_matrix((Vall, (R, C)), shape=(self.ndof, self.ndof)).tocsr()
        self.K0.sum_duplicates()
        self.t_asm = time.time() - t0
        if self.verbose:
            print('[asm ] nnz=%d  groups=%d  %.1fs'
                  % (self.K0.nnz, len(self.Clist), self.t_asm))

    def _reassemble(self):
        """脱粘设置改变后重装刚度阵，并丢掉旧的 LU 因子（主动集缓存按 K0 失效）。"""
        self._fcache = {}
        self._assemble()

    # ---------------------------------------------------------------- 脱粘
    def set_debond(self, interface, x_range, cells, reassemble=True):
        """在指定界面、轴向区间、指定 cell 上设置界面脱粘（剪切失效等效）。

        interface : 'B' = 缠绕层/拉挤块界面 (d = 45.0 mm)
                    'C' = 拉挤块/层压界面   (d = 52.0 mm)
        x_range   : (x_lo, x_hi)，沿轴向的脱粘区间，mm（按单元形心判断是否落入）
        cells     : 脱粘的 cell 编号列表

        实现：把"在 (y,r) 截面里被该界面圆穿过"的那一圈单元（判据是单元四角到螺套
        轴线的最小距离 <= d_界面 <= 最大距离，因此这一圈一定是连通、不漏的）的三个
        剪切模量 G_xy / G_xr / G_yr 一起降到名义值的 1e-4 倍，法向（拉压）刚度保持
        不变。

        **这是"剪切失效等效"，不是真正的接触或内聚力单元。** 它表达的是"这一圈柱面
        上不再传递剪流"；它不能张开、不能闭合、不给能量释放率，也不会在脱粘尖端给
        出正确的应力奇异性。脱粘尖端处的应力峰值受单元尺寸控制，只能当量级读。

        多次调用会累加（并集），clear_debond() 复位。
        """
        if interface not in INTERFACE_D:
            raise ValueError("interface 必须是 'B' 或 'C'")
        d_if = INTERFACE_D[interface]
        ring = (self.d_cmin <= d_if) & (self.d_cmax >= d_if)          # (ny, nr_e)
        xlo, xhi = float(x_range[0]), float(x_range[1])
        mx = (self.xc >= xlo) & (self.xc <= xhi) & self.in_x
        cells = list(np.atleast_1d(cells).astype(int))
        mc = np.isin(self.cell_of_col, np.array(cells) % self.n_cell)
        m3 = mx[:, None, None] & (ring & mc[:, None])[None, :, :]
        self.deb |= m3
        self._debonds.append(dict(interface=interface, x_range=(xlo, xhi),
                                  cells=cells, n_elem=int(m3.sum())))
        if reassemble:
            self._reassemble()
        return int(m3.sum())

    def clear_debond(self, reassemble=True):
        had = bool(self.deb.any())
        self.deb[:] = False
        self._debonds = []
        if reassemble and had:
            self._reassemble()

    def debond_tag(self):
        if not self._debonds:
            return 'intact'
        return '+'.join('%s[%.0f,%.0f]c%s' % (d['interface'], d['x_range'][0],
                                              d['x_range'][1],
                                              ','.join(str(c) for c in d['cells']))
                        for d in self._debonds)

    # ------------------------------------------------------------- 场后处理
    def element_fields(self, u):
        """单元形心处的应变/应力张量。返回 eps, sig，形状 (nx_e, ny, nr_e, 6)。
        Voigt 次序 (xx, yy, rr, yr, xr, xy)。"""
        sh = (self.nx_e, self.ny, self.nr_e, 6)
        eps = np.zeros(sh)
        sig = np.zeros(sh)
        idx = np.arange(self.ny)
        for ix in range(self.nx_e):
            for ir in range(self.nr_e):
                B = _B_at(0.0, 0.0, 0.0, self.dx[ix], self.dy, self.dr[ir],
                          self._r0(ir))
                els = (ix * self.ny + idx) * self.nr_e + ir
                e = self._elem_disp(u, els) @ B.T                # (ny, 6)
                eps[ix, :, ir, :] = e
                gid = self.ckey[ix, :, ir]
                for g in np.unique(gid):
                    sel = np.flatnonzero(gid == g)
                    sig[ix, sel, ir, :] = e[sel] @ self.Clist[int(g)].T
        return eps, sig

    @staticmethod
    def von_mises(sig):
        sxx, syy, srr = sig[..., 0], sig[..., 1], sig[..., 2]
        tyr, txr, txy = sig[..., 3], sig[..., 4], sig[..., 5]
        return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - srr) ** 2 + (srr - sxx) ** 2)
                       + 3.0 * (tyr ** 2 + txr ** 2 + txy ** 2))

    @staticmethod
    def tau_res(sig):
        """本模块采用的"最大剪应力"定义：三个剪应力分量的合成
        tau = sqrt(txy^2 + txr^2 + tyr^2)。**不是**主应力差的一半。
        选它是因为界面失效关心的就是作用在界面上的剪流合量。"""
        return np.sqrt(sig[..., 3] ** 2 + sig[..., 4] ** 2 + sig[..., 5] ** 2)

    def surface_strain(self, u):
        """内表面 r=0 的轴向应变场，返回 (nx_e, ny)。
        三线性六面体在结点处 eps_xx 就是相邻两结点 u_x 的差商（精确）；
        这里取单元列两侧 y 结点面的平均，落在单元列中心。"""
        ux = u[3 * np.arange(self.n_node)].reshape(self.ny, self.nx_n, self.nr_n)
        d = (ux[:, 1:, 0] - ux[:, :-1, 0]) / self.dx[None, :]        # (ny, nx_e)
        s = 0.5 * (d + np.roll(d, -1, axis=0))
        return s.T

    def section_force(self, u):
        """截面轴力 N(x) = ∫ sigma_xx dA（对全部 cell 求和）。父类版本按 3 种材料
        写死了本构表，这里改成按 ckey 查 Clist。"""
        N = np.zeros(self.nx_e)
        idx = np.arange(self.ny)
        for ix in range(self.nx_e):
            tot = 0.0
            for ir in range(self.nr_e):
                B = _B_at(0.0, 0.0, 0.0, self.dx[ix], self.dy, self.dr[ir],
                          self._r0(ir))
                els = (ix * self.ny + idx) * self.nr_e + ir
                e = self._elem_disp(u, els) @ B.T
                gid = self.ckey[ix, :, ir]
                sxx = np.zeros(self.ny)
                for g in np.unique(gid):
                    sel = np.flatnonzero(gid == g)
                    sxx[sel] = e[sel] @ self.Clist[int(g)][0, :]
                tot += float(sxx.sum()) * self.dy * self.dr[ir]
            N[ix] = tot
        return N

    # --------------------------------------------------------- 分解（换求解器）
    def _factor(self, act):
        """与父类 Sector3D._factor 同逻辑，只把 scipy.splu 换成 SparseLU 外壳
        （优先 MKL PARDISO）。父类是只读的，所以这里整段重写而不是打补丁。"""
        if not hasattr(self, '_fcache'):
            self._fcache = {}
            self.t_lu = 0.0
            self.n_lu = 0
        key = act.tobytes()
        if key in self._fcache:
            return self._fcache[key]
        t0 = time.time()
        far, rbm = self._con_base()
        cface = self.face_dof[act]
        con = np.concatenate([far, rbm, cface])
        mask = np.ones(self.ndof, bool)
        mask[con] = False
        free = np.flatnonzero(mask)
        A = self.K0
        lu = SparseLU(A[free][:, free])
        U = np.zeros((self.ndof, self.n_cell))
        for j in range(self.n_cell):
            U[self.stud_dof[j], j] = self.stud_w[j]
        Z = lu.solve(U[free])
        S = dict(free=free, con=con, cface=cface, far=far, lu=lu, U=U,
                 Uf=U[free], Z=Z, A=A, act=act.copy())
        self.t_lu += time.time() - t0
        self.n_lu += 1
        if self.verbose:
            print('  [lu:%s] act=%d/%d  free=%d  %.1fs (累计 %d 次 %.1fs)'
                  % (SparseLU.backend, int(act.sum()), act.size, len(free),
                     time.time() - t0, self.n_lu, self.t_lu))
        while len(self._fcache) >= 2:
            old = self._fcache.pop(next(iter(self._fcache)))
            old['lu'].free()
        self._fcache[key] = S
        return S

    def free_factor_cache(self):
        for S in getattr(self, '_fcache', {}).values():
            S['lu'].free()
        self._fcache = {}


# ================================================================== 精确面积
def exact_layer_areas(ngy=3000, ngr=3000):
    """单个 cell 横截面（y 一个节距 × r 全壁厚）里各层的精确面积，数值积分。
    用最近螺套轴线的距离判层，与网格判据同定义，差别只在阶梯化。"""
    y = (np.arange(ngy) + 0.5) * P_PITCH / ngy
    r = (np.arange(ngr) + 0.5) * T_WALL / ngr
    a = (y - 0.5 * P_PITCH) % P_PITCH
    a = np.minimum(a, P_PITCH - a)
    d = np.hypot(a[:, None], (r - R_INS_C)[None, :])
    dA = (P_PITCH / ngy) * (T_WALL / ngr)
    out = {}
    out[MAT_VOID] = float((d < D_BORE_R).sum()) * dA
    out[MAT_STEEL] = float(((d >= D_BORE_R) & (d < D_STEEL_O)).sum()) * dA
    out[MAT_WRAP] = float(((d >= D_STEEL_O) & (d < D_WRAP_O)).sum()) * dA
    out[MAT_BLOCK] = float(((d >= D_WRAP_O) & (d < D_BLOCK_O)).sum()) * dA
    out[MAT_LAM] = float((d >= D_BLOCK_O).sum()) * dA
    return out


# ==================================================================== 校验
def verify_uniform(sigma=10.0, m_y=20, n_cell=2, layers=(2, 2, 4, 4, 4),
                   hoop='global'):
    """校验 (a)：拿掉全部螺套（整块层压），x=0 施加均布轴向端载，
    检查 eps_xx 是否等于 sigma/E_x。与父模型 verify_uniform 同法，
    只是网格换成分层模型的径向网格，用来确认新的 r 网格 + 周期条件没被做坏。

    均匀应变场对三线性六面体是可以精确表示的，所以本校验与周向 cell 数无关；
    为了省一次 30 万自由度的 LU，默认只用 n_cell=2 的窗口（m_y、径向分层与
    生产网格完全一致）。生产网格的周期性由第二项校验"完好工况各 cell 的 F_A
    必须相等"来保证。"""
    md = SectorLayered(m_y=m_y, n_cell=n_cell, layers=layers,
                       with_inserts=False, hoop=hoop, verbose=False)
    far, rbm = md._con_base()
    con = np.concatenate([far, rbm])
    mask = np.ones(md.ndof, bool)
    mask[con] = False
    free = np.flatnonzero(mask)
    At = np.zeros(md.nr_n)
    At[:-1] += 0.5 * md.dr
    At[1:] += 0.5 * md.dr
    At = At * md.dy
    f = np.zeros(md.ndof)
    f[md.face_dof] = -sigma * At[None, :]
    lu = SparseLU(md.K0[free][:, free])
    u = np.zeros(md.ndof)
    u[free] = lu.solve(f[free])
    ei, ea = md.profiles(u, md.m_y // 2)
    A_tot = md.ny * md.dy * T_WALL
    F_tot = sigma * A_tot
    eps_ref = F_tot / (E_X * A_tot)
    out = dict(eps_ref=eps_ref,
               err_inner=float(np.max(np.abs(ei / eps_ref - 1.0))),
               err_avg=float(np.max(np.abs(ea / eps_ref - 1.0))),
               F_tot=F_tot, A_tot=A_tot, ndof=md.ndof,
               R_far=float((md.K0 @ u - f)[far].sum()),
               backend=SparseLU.backend)
    lu.free()
    del lu, md
    return out


def verify_constitutive():
    """校验新增两种材料的本构矩阵：C^-1 必须还原出给定的工程常数。"""
    out = {}
    for nm, C, ref in (('wrap', C_wrap(), (E_WRAP_X, E_WRAP_H, NU_WRAP, G_WRAP)),
                       ('block', C_block(), (E_BLK_X, E_BLK_T, NU_BLK, G_BLK))):
        S = np.linalg.inv(C)
        Ex, Ey = 1.0 / S[0, 0], 1.0 / S[1, 1]
        nxy = -S[0, 1] / S[0, 0]
        G = 1.0 / S[5, 5]
        out[nm] = dict(Ex=Ex, Ey=Ey, nu_xy=nxy, G=G,
                       err=max(abs(Ex / ref[0] - 1), abs(Ey / ref[1] - 1),
                               abs(nxy / ref[2] - 1), abs(G / ref[3] - 1)),
                       spd=bool(np.linalg.eigvalsh(C).min() > 0))
    return out


# ===================================================================== 工况
CASES = OrderedDict([
    ('intact',      dict(kind='intact')),
    ('debB_200',    dict(kind='debond', iface='B',
                         x=(L_INS - 200.0, L_INS), nc=1)),
    ('debB_400',    dict(kind='debond', iface='B',
                         x=(L_INS - 400.0, L_INS), nc=1)),
    ('debB_460',    dict(kind='debond', iface='B',
                         x=(L_INS - 460.0, L_INS), nc=1)),
    ('debB_400_x3', dict(kind='debond', iface='B',
                         x=(L_INS - 400.0, L_INS), nc=3)),
    ('debC_400',    dict(kind='debond', iface='C',
                         x=(L_INS - 400.0, L_INS), nc=1)),
    ('broken',      dict(kind='broken')),
])


def run_all(m_y=20, n_cell=5, layers=(2, 2, 4, 4, 4), verbose=True):
    """跑全部工况。预紧按完好态标定一次，法兰位移 w 也按完好态标定到
    单柱 F_A = 159 kN，随后所有缺陷工况沿用同一个 w（远场载荷不变），
    这与父模型断柱工况 R4/R5 的口径一致。"""
    t0 = time.time()
    md = SectorLayered(m_y=m_y, n_cell=n_cell, layers=layers, verbose=verbose)
    JC = n_cell // 2
    iy_cl = JC * md.m_y + md.m_y // 2                 # 中心螺套轴线所在的 y 结点面

    print('>>> 标定预紧 F_M（目标 F_S = %.0f kN）' % (FS_TARGET / 1e3))
    FM, _ = md.calibrate_FM(target_FS=FS_TARGET)
    print('    F_M = %.1f kN' % (FM / 1e3))
    print('>>> 标定法兰位移 w（目标单柱 F_A = %.0f kN）' % (FA_RATED / 1e3))
    r_ref = md.solve(FM, target_FA=0.0)          # F_A=0 参考态（算螺栓载荷系数 Phi 用）
    r0 = md.solve(FM, target_FA=FA_RATED)
    w0 = r0['w']
    print('    w = %.6f mm   F_A = %.1f kN   F_S = %.1f kN'
          % (w0, r0['FA'].mean() / 1e3, r0['FS'].mean() / 1e3))

    res = OrderedDict()
    order = ['intact', 'broken', 'debB_200', 'debB_400', 'debB_460',
             'debB_400_x3', 'debC_400']
    for name in order:
        spec = CASES[name]
        ts = time.time()
        if spec['kind'] == 'intact':
            r = r0
        elif spec['kind'] == 'broken':
            r = md.solve(FM, broken=[JC], w=w0)
        else:
            # 先清掉上一工况的脱粘再设新的，只重装配一次
            md.clear_debond(reassemble=False)
            cells = [JC] if spec['nc'] == 1 else [(JC - 1) % n_cell, JC,
                                                  (JC + 1) % n_cell]
            n = md.set_debond(spec['iface'], spec['x'], cells)
            r = md.solve(FM, w=w0)
            r['n_deb_elem'] = n
        r['deb_mask'] = md.deb.copy()
        r['tag'] = md.debond_tag()
        # ---- 后处理必须在本工况的材料状态（ckey/Clist）还在的时候做完，
        #      否则下一次 set_debond 重装配后再算应力就会张冠李戴。
        _postprocess(md, r, JC, iy_cl)
        r['t_case'] = time.time() - ts
        res[name] = r
        print('  [%-12s] %-28s F_S=%8.2f kN  F_KR=%8.2f kN  F_A=%8.2f kN '
              '  open=%d  conv=%s  %.0fs'
              % (name, r['tag'], r['FS'][JC] / 1e3, r['C'][JC] / 1e3,
                 r['FA'][JC] / 1e3, r['n_open'], r['converged'], r['t_case']))
        md.free_factor_cache()
    md.clear_debond(reassemble=False)
    return dict(md=md, res=res, FM=FM, w0=w0, JC=JC, iy_cl=iy_cl,
                r_ref=r_ref, t_total=time.time() - t0)


def _postprocess(md, r, JC, iy_cl):
    """在当前材料状态下把一个工况需要的全部导出量算完。"""
    u = r['u']
    eps, sig = md.element_fields(u)
    r['eps_xx'] = eps[..., 0].astype(np.float32)
    r['svm'] = md.von_mises(sig).astype(np.float32)
    r['tau_max'] = md.tau_res(sig).astype(np.float32)
    # 脱粘环内的剪应力合量（检查"剪切失效等效"是否真的生效）
    if r['deb_mask'].any():
        r['tau_in_debond'] = float(np.abs(r['tau_max'][r['deb_mask']]).max())
    else:
        r['tau_in_debond'] = np.nan
    del eps, sig
    r['surf_eps'] = md.surface_strain(u).astype(np.float32)
    r['eps_in'] = md.profiles(u, iy_cl)[0]
    r['Ns'] = md.steel_force(u, JC)
    us, ul = md.interface_slip(u, JC)
    r['slip'] = us - ul
    r['N_x'] = md.section_force(u)
    r['u32'] = u[:3 * md.n_node].reshape(md.n_node, 3).astype(np.float32)


# ================================================================ 场导出
FIELD_README = u"""\
sector_layered.npz —— 叶根分层螺套连接三维场数据
================================================================================
由 sector3d_layered.py 生成（sector3d.Sector3D 的分层子类）。

1. 网格
--------------------------------------------------------------------------------
  x  (nx_n,)  轴向结点坐标 mm，0 = 叶根端面/法兰面
  y  (ny,)    周向展开结点坐标 mm，周期，节距 %(pitch).2f mm
  r  (nr_n,)  径向结点坐标 mm，0 = 叶根内表面（光纤位置），%(wall).0f = 外表面
  xc (nx_e,)  yc (ny,)  rc (nr_e,)   对应的单元形心坐标
  注意 y 方向结点数 = 单元数 = ny（周期，最后一列单元绕回 y=0）。

  结点编号：nid = (iy * nx_n + ix) * nr_n + ir     (ix∈[0,nx_n), iy∈[0,ny), ir∈[0,nr_n))
  单元编号：e   = (ix * ny + iy) * nr_e + ir
  所以 <case>_u[nid] 就是结点 (ix,iy,ir) 的位移 (u_x, u_y, u_r)。

2. 三维渲染：展开坐标 → 真实空间
--------------------------------------------------------------------------------
      真实半径   R      = R_IN + r           (mm)   # r=R_INS_C 处正好是螺栓圆半径
      周向角     theta  = y / R_bc            (rad)
      轴向       X      = x                   (mm)
      笛卡尔     (X, R*sin(theta), R*cos(theta))

  python 示例（把单元场铺到真实圆筒上）：
      d = np.load('sector_layered.npz')
      R  = %(r_in).1f + d['rc']
      th = d['yc'] / %(r_bc).1f
      X  = d['xc']
      # 网格 (nx_e, ny, nr_e)
      Xg = X[:, None, None] * np.ones((1, len(th), len(R)))
      Yg = R[None, None, :] * np.sin(th)[None, :, None]
      Zg = R[None, None, :] * np.cos(th)[None, :, None]
      F  = d['debB_400_eps_xx']          # 任意一个单元场

  本模型只建了 %(ncell)d 个 cell（周向 %(deg).1f 度），要画整环就按 %(pitch).2f mm 的节距平铺。

3. 材料
--------------------------------------------------------------------------------
  mat (nx_e, ny, nr_e) int8，取值查 mat_names：
      0 lam   叶根层压                d >= 52.0
      1 steel 钢衬套/实心芯           d_bore/2 <= d < D_ins/2（盲孔底以后全钢）
      2 void  空螺孔                  d < 18.75 且 x < 130
      3 wrap  玻纤束缠绕层            过渡层以外 6 mm（过渡层并入其中）
      4 block 拉挤 GFRP 块            45.0 <= d < 52.0
  d = 到最近螺套轴线的距离，轴线在 r=R_INS_C、每 P_PITCH 一根。

4. 工况与场
--------------------------------------------------------------------------------
  工况名：%(cases)s
  每个工况 <case> 有：
    <case>_eps_xx   (nx_e, ny, nr_e) float32  单元形心轴向应变（拉为正）
    <case>_svm      (nx_e, ny, nr_e) float32  单元 von Mises 应力 MPa
    <case>_tau_max  (nx_e, ny, nr_e) float32  单元剪应力合量
                     tau = sqrt(txy^2 + txr^2 + tyr^2)（不是主应力差的一半）
    <case>_u        (n_node, 3)      float32  结点位移 (u_x,u_y,u_r) mm
    <case>_surf_eps (nx_e, ny)       float32  内表面 r=0 的轴向应变（光纤读数来源）
    <case>_FS       标量  监测 cell 的螺柱轴力 N
    <case>_FKR      标量  监测 cell 的端面残余夹紧力 N
    <case>_FS_cells / <case>_FKR_cells / <case>_FA_cells  (n_cell,) 各 cell 的值
    <case>_Ns       (nx_e,)  监测 cell 的螺套钢体轴力沿 x 的分布 N
    <case>_slip     (nx_n,)  钢体与周围复合材料的横截面平均轴向位移之差 mm
    <case>_eps_in_cl(nx_e,)  中心螺套轴线所在 y 面、r=0 的轴向应变
    <case>_debond   (nx_e, ny, nr_e) bool  被判为脱粘（剪切失效）的单元
    <case>_w        标量  法兰面轴向位移 mm

5. 约定与告诫
--------------------------------------------------------------------------------
  · 脱粘用"剪切模量降到 1e-4 倍"等效，不是接触/内聚力单元：不能张开闭合，
    脱粘尖端的应力峰值受单元尺寸控制，只能当量级读。
  · 应变单位是无量纲（乘 1e6 得微应变），应力单位 MPa，位移 mm，力 N。
  · 预紧、外载：F_M 按 F_S=420 kN 标定，法兰位移按完好态单柱 F_A=159 kN 标定，
    所有缺陷工况沿用同一个法兰位移（远场载荷不变）。
"""


def write_npz(R, path, fields=True):
    md, res, JC = R['md'], R['res'], R['JC']
    d = OrderedDict()
    d['x'] = md.x
    d['y'] = md.y
    d['r'] = md.r
    d['xc'] = md.xc
    d['yc'] = md.yc
    d['rc'] = md.rc
    d['mat'] = md.mat
    d['mat_names'] = MAT_NAMES
    d['n_cell'] = md.n_cell
    d['m_y'] = md.m_y
    d['nx_n'] = md.nx_n
    d['ny'] = md.ny
    d['nr_n'] = md.nr_n
    d['n_node'] = md.n_node
    d['n_elem'] = md.n_elem
    d['ndof'] = md.ndof
    d['pitch'] = P_PITCH
    d['R_in'] = R_IN
    d['R_bolt'] = 1600.0
    d['JC'] = JC
    d['iy_cl'] = R['iy_cl']
    d['F_M'] = R['FM']
    d['w0'] = R['w0']
    d['FA_target'] = FA_RATED
    d['FS_target'] = FS_TARGET
    d['layer_d'] = np.array([D_BORE_R, D_STEEL_O, D_TRANS_O, D_WRAP_O, D_BLOCK_O])
    d['case_names'] = np.array(list(res.keys()))
    for nm, r in res.items():
        if fields:
            d['%s_eps_xx' % nm] = r['eps_xx']
            d['%s_svm' % nm] = r['svm']
            d['%s_tau_max' % nm] = r['tau_max']
        d['%s_u' % nm] = r['u32']
        d['%s_surf_eps' % nm] = r['surf_eps']
        d['%s_FS' % nm] = float(r['FS'][JC])
        d['%s_FKR' % nm] = float(r['C'][JC])
        d['%s_FS_cells' % nm] = r['FS']
        d['%s_FKR_cells' % nm] = r['C']
        d['%s_FA_cells' % nm] = r['FA']
        d['%s_Ns' % nm] = r['Ns']
        d['%s_slip' % nm] = r['slip']
        d['%s_eps_in_cl' % nm] = r['eps_in']
        d['%s_N_x' % nm] = r['N_x']
        d['%s_debond' % nm] = r['deb_mask']
        d['%s_w' % nm] = float(r['w'])
        d['%s_converged' % nm] = bool(r['converged'])
    np.savez_compressed(path, **d)
    return d


def write_readme(R, path):
    txt = FIELD_README % dict(
        pitch=P_PITCH, wall=T_WALL, r_in=R_IN, r_bc=R_IN + R_INS_C,
        ncell=R['md'].n_cell,
        deg=np.degrees(R['md'].n_cell * P_PITCH / (R_IN + R_INS_C)),
        cases=', '.join(R['res'].keys()))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    return txt


# ==================================================================== 报告
def report(R, va, vc):
    md, res, JC = R['md'], R['res'], R['JC']
    L = '=' * 92
    out = []
    P = out.append
    P(L)
    P('叶根预埋螺套连接 · 分层三维扇形块模型 (sector3d_layered.py)')
    P('生成时间 %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    P(L)

    # --------------------------------------------------------------- 一、网格
    P('')
    P('一、分层网格')
    P(L)
    P('  轴向 x：%d 个单元（沿用父模型分级网格），x_max = %.0f mm' % (md.nx_e, X_MAX))
    P('  周向 y：m_y = %d，dy = %.4f mm，共 %d 个单元 / %d 个 cell（周向 %.1f 度）'
      % (md.m_y, md.dy, md.ny, md.n_cell,
         np.degrees(md.n_cell * P_PITCH / 1600.0)))
    P('  径向 r：%d 个单元，结点 r = %s'
      % (md.nr_e, np.array2string(md.r, precision=2, max_line_width=200)))
    P('  单元 %d，结点 %d，自由度 %d（含 1 个环向周期跳变 Δ）'
      % (md.n_elem, md.n_node, md.ndof))
    P('  刚度阵非零元 %d；装配 %.1f s；LU 分解 %d 次共 %.1f s'
      % (md.K0.nnz, md.t_asm, md.n_lu, md.t_lu))
    P('')
    P('  各层的径向分辨（沿螺套轴线所在的 y 列）：')
    names = {MAT_LAM: '层压', MAT_STEEL: '钢', MAT_VOID: '空孔',
             MAT_WRAP: '缠绕层', MAT_BLOCK: '拉挤块'}
    for (a, b, n), lbl in zip(md.r_seg, ['内层压', '拉挤块(内)', '缠绕层(内)',
                                         '钢(内)', '螺孔/芯', '钢(外)',
                                         '缠绕层(外)', '拉挤块(外)+外皮']):
        P('    %-16s r = %6.2f ~ %6.2f mm   %d 个单元   dr = %.3f mm'
          % (lbl, a, b, n, (b - a) / n))
    P('')
    P('  阶梯化截面积（单 cell 横截面，x<130 处；网格按单元形心判层）：')
    P('    %-8s %14s %14s %10s' % ('层', '网格 [mm^2]', '精确 [mm^2]', '误差'))
    tot_m = tot_e = 0.0
    for m in (MAT_VOID, MAT_STEEL, MAT_WRAP, MAT_BLOCK, MAT_LAM):
        am, ae = md.A_mesh[m], md.A_exact[m]
        tot_m += am
        tot_e += ae
        P('    %-8s %14.1f %14.1f %9.2f%%'
          % (names[m], am, ae, 100 * (am / ae - 1) if ae > 0 else np.nan))
    P('    %-8s %14.1f %14.1f %9.2f%%' % ('合计', tot_m, tot_e,
                                          100 * (tot_m / tot_e - 1)))
    P('')
    P('  几何提示：缠绕层外径 Ø%.0f、拉挤块外径 Ø%.0f 都大于螺栓节距 %.2f mm，'
      % (2 * D_WRAP_O, 2 * D_BLOCK_O, P_PITCH))
    P('  所以相邻螺套的缠绕层/拉挤块在周向互相搭接。模型按"到最近螺套轴线的距离"')
    P('  判层，结果这两层在周向连成带状（真实构造里拉挤块本来也是彼此顶紧的楔块）。')
    P('  后果：单个 cell 的界面脱粘不是一个封闭的圆环，它在 cell 边界处敞口，')
    P('  载荷可以绕过脱粘区、经相邻 cell 完好的界面传出去。见第四节。')
    P('')
    P('  过渡层（0.5 mm 树脂富集）：不单独划网格，照搬父模型 G_eff 的柔度串联法，')
    P('  把紧贴钢外圆的一圈缠绕层单元（%d 个/横截面）的三个剪切模量改为'
      % int(md.ring_mask.sum() / md.n_cell))
    P('  1/G_eff = 1/G_wrap + t/(G_trans*h_e)，h_e = %.2f~%.2f mm，'
      % (md.h_ring[md.ring_mask].min(), md.h_ring[md.ring_mask].max()))
    P('  得到 G_eff = %.0f~%.0f MPa（原 G_wrap = %.0f MPa）。'
      % (md.G_eff[md.ring_mask].min(), md.G_eff[md.ring_mask].max(), G_WRAP))

    # --------------------------------------------------------------- 二、校验
    P('')
    P('二、校验')
    P(L)
    P('  (a) 新增本构矩阵的自洽性（C 求逆还原工程常数）：')
    for nm, v in vc.items():
        P('      %-6s E_x=%8.1f  E_h=%8.1f  nu=%.4f  G=%8.1f   最大相对偏差 %.2e  正定=%s'
          % (nm, v['Ex'], v['Ey'], v['nu_xy'], v['G'], v['err'], v['spd']))
    P('')
    P('  (b) 单轴拉伸（拿掉全部螺套，x=0 均布端载 sigma=10 MPa，DOF=%d）：' % va['ndof'])
    P('      理论 eps_xx = F/(E_x*A) = %.6e' % va['eps_ref'])
    P('      有限元最大相对偏差：内表面 %.3e，厚度平均 %.3e   %s'
      % (va['err_inner'], va['err_avg'],
         '→ 通过 (<1%)' if max(va['err_inner'], va['err_avg']) < 0.01 else '→ 不通过'))
    P('      远端支反力 %.1f N vs 施加 %.1f N，差 %.2e'
      % (va['R_far'], va['F_tot'], abs(va['R_far'] - va['F_tot'])))
    P('')
    ri = res['intact']
    fa = ri['FA']
    dev = float(np.max(np.abs(fa / fa.mean() - 1.0)))
    P('  (c) 完好工况各 cell 的 F_A 均匀性（周期性是否被做坏）：')
    P('      ' + '  '.join('cell%d=%.3f kN' % (j, fa[j] / 1e3) for j in range(md.n_cell)))
    P('      均值 %.3f kN，最大相对偏差 %.3e   %s'
      % (fa.mean() / 1e3, dev, '→ 通过 (<1%)' if dev < 0.01 else '→ 不通过'))
    P('')
    P('  (d) 整体力平衡 R_far = sum(F_S) - sum(C)：')
    P('      %-12s %14s %14s %14s %12s' % ('工况', 'sum F_S [kN]', 'sum C [kN]',
                                           'R_far [kN]', '不平衡 [N]'))
    for nm, r in res.items():
        P('      %-12s %14.3f %14.3f %14.3f %12.3e'
          % (nm, r['FS'].sum() / 1e3, r['C'].sum() / 1e3, r['R_far'] / 1e3,
             r['R_far'] - (r['FS'].sum() - r['C'].sum())))
    P('')
    P('  (e) 截面轴力守恒：x > 120 mm 不再有外力注入，N(x) 应等于 sum(F_A)。')
    P('      %-12s %14s' % ('工况', 'sum F_A [kN]')
      + ''.join('%12s' % ('N(%.0f)' % x) for x in XCHK))
    for nm, r in res.items():
        if 'N_x' not in r:
            continue
        row = '      %-12s %14.3f' % (nm, r['FA'].sum() / 1e3)
        for x in XCHK:
            row += '%12.3f' % (np.interp(x, md.xc, r['N_x']) / 1e3)
        P(row)
    P('')
    P('  (f) 脱粘等效是否真的切断了剪力：脱粘圈单元内的剪应力合量峰值，')
    P('      与完好工况同一圈单元的峰值对照。')
    P('      %-12s %10s %14s %16s %10s' % ('工况', '圈内单元数', 'tau [MPa]',
                                           '完好态同圈 [MPa]', '残余比'))
    for nm, r in res.items():
        if not r['deb_mask'].any():
            continue
        ref = float(np.abs(res['intact']['tau_max'][r['deb_mask']]).max())
        P('      %-12s %10d %14.4f %16.4f %9.2f%%'
          % (nm, int(r['deb_mask'].sum()), r['tau_in_debond'], ref,
             100 * r['tau_in_debond'] / max(ref, 1e-30)))

    # --------------------------------------------------------------- 三、工况
    P('')
    P('三、七个工况的关键数字（监测 cell = %d，预紧 F_M=%.1f kN，法兰位移 w=%.6f mm）'
      % (JC, R['FM'] / 1e3, R['w0']))
    P(L)
    e0 = res['intact']['eps_in']
    P('  %-12s %10s %10s %10s %12s %10s %12s'
      % ('工况', 'F_S[kN]', 'F_KR[kN]', 'F_A[kN]', 'eps_in峰[ue]', 'x峰[mm]',
         'Δeps峰[ue]'))
    for nm, r in res.items():
        e = r['eps_in']
        k = int(np.argmax(np.abs(e)))
        de = e - e0
        kd = int(np.argmax(np.abs(de)))
        P('  %-12s %10.2f %10.2f %10.2f %12.1f %10.0f %12.1f'
          % (nm, r['FS'][JC] / 1e3, r['C'][JC] / 1e3, r['FA'][JC] / 1e3,
             e[k] * 1e6, md.xc[k], de[kd] * 1e6))
    P('')
    P('  内表面 r=0（中心螺套轴线所在 y 面）轴向应变沿 x [微应变]：')
    xp = np.array([5., 20., 50., 100., 150., 200., 300., L_INS - 90,
                   L_INS - 40, L_INS - 10, L_INS + 30, 600., 800.])
    P('  %-12s' % 'x[mm]' + ''.join('%9.0f' % v for v in xp))
    for nm, r in res.items():
        P('  %-12s' % nm + ''.join('%9.1f' % (np.interp(v, md.xc, r['eps_in']) * 1e6)
                                   for v in xp))
    P('')
    P('  相对完好工况的应变变化量 Δeps [微应变]：')
    P('  %-12s' % 'x[mm]' + ''.join('%9.0f' % v for v in xp))
    for nm, r in res.items():
        if nm == 'intact':
            continue
        P('  %-12s' % nm + ''.join('%9.1f' % (np.interp(v, md.xc, r['eps_in'] - e0) * 1e6)
                                   for v in xp))
    P('')
    P('  螺套钢体轴力 N_s(x) [kN]：')
    P('  %-12s' % 'x[mm]' + ''.join('%9.0f' % v for v in xp[:11]))
    for nm, r in res.items():
        P('  %-12s' % nm + ''.join('%9.1f' % (np.interp(v, md.xc, r['Ns']) / 1e3)
                                   for v in xp[:11]))
    P('')
    P('  界面滑移（钢体 - 周围复合材料的横截面平均轴向位移）[µm]：')
    P('  %-12s' % 'x[mm]' + ''.join('%9.0f' % v for v in xp[:11]))
    for nm, r in res.items():
        P('  %-12s' % nm + ''.join('%9.2f' % (np.interp(v, md.x, r['slip']) * 1e3)
                                   for v in xp[:11]))

    # ---------------------------------------------------------- 四、B vs C
    P('')
    P('四、界面 B 与界面 C 哪个在内表面应变上更显著')
    P(L)
    dB = res['debB_400']['eps_in'] - e0
    dC = res['debC_400']['eps_in'] - e0
    P('  同为单 cell、400 mm 脱粘：')
    P('    界面 B（缠绕层/拉挤块，d=45）：Δeps 峰值 %+.1f ue @ x=%.0f mm，'
      % (dB[np.argmax(np.abs(dB))] * 1e6, md.xc[np.argmax(np.abs(dB))]))
    P('                                   ΔF_S = %+.2f kN，ΔF_KR = %+.2f kN'
      % ((res['debB_400']['FS'][JC] - res['intact']['FS'][JC]) / 1e3,
         (res['debB_400']['C'][JC] - res['intact']['C'][JC]) / 1e3))
    P('    界面 C（拉挤块/层压，d=52）  ：Δeps 峰值 %+.1f ue @ x=%.0f mm，'
      % (dC[np.argmax(np.abs(dC))] * 1e6, md.xc[np.argmax(np.abs(dC))]))
    P('                                   ΔF_S = %+.2f kN，ΔF_KR = %+.2f kN'
      % ((res['debC_400']['FS'][JC] - res['intact']['FS'][JC]) / 1e3,
         (res['debC_400']['C'][JC] - res['intact']['C'][JC]) / 1e3))
    rb = abs(dB[np.argmax(np.abs(dB))])
    rc = abs(dC[np.argmax(np.abs(dC))])
    P('    比值 |ΔepsC| / |ΔepsB| = %.2f' % (rc / max(rb, 1e-30)))
    P('')
    if rc > rb:
        P('  结论：界面 C（拉挤块/层压）更显著，应当作为监测判据的标定对象。')
        P('  机理：C 在 d=52，离内表面 (r=0) 只有 5.5 mm，它一旦失效，螺套-缠绕层-')
        P('  拉挤块这一整块芯体就只能靠端部把力顶进层压，内表面正好骑在被卸载的那')
        P('  5.5 mm 皮层上，所以应变重分布看得最清楚。')
    else:
        P('  结论：界面 B（缠绕层/拉挤块）更显著，应当作为监测判据的标定对象。')
        P('  机理：B 切断的是钢-缠绕层芯体向外的第一条剪切路径，被隔离的芯体刚度')
        P('  （E_钢=210 GPa）与外侧差得最多，剪流重分布幅度最大；C 切断时，内侧仍')
        P('  留着 E_轴=45 GPa 的拉挤块与芯体连成一体，等效截面变化小，内表面反应弱。')
    P('  另外两个都受"周向搭接"的影响：d=%.1f 与 d=%.1f 两个圆都大于节距 %.2f mm，'
      % (D_WRAP_O, D_BLOCK_O, P_PITCH))
    P('  脱粘圈在 cell 边界处敞口，载荷能经相邻 cell 完好的界面绕过去，所以单 cell')
    P('  脱粘的信号被系统性压低。debB_400_x3（相邻 3 个 cell 同时脱粘）就是用来量')
    P('  这条旁路有多大的。')

    # ---------------------------------------------------------- 五、2D 对照
    P('')
    P('五、与二维模型 root_model.RootFE 的量级对照')
    P(L)
    if R.get('d2') is not None:
        d2 = R['d2']
        P('  %-28s %14s %14s %10s' % ('量', '三维分层 M3', '二维 M1', '相对差'))
        for lbl, a, b in d2['rows']:
            P('  %-28s %14.4f %14.4f %9.1f%%'
              % (lbl, a, b, 100 * (a / b - 1) if abs(b) > 1e-12 else np.nan))
    else:
        P('  （本次运行未跑二维对照）')

    # ---------------------------------------------------------- 六、推断参数
    P('')
    P('六、本模块采用的【推断】参数清单')
    P(L)
    P('  玻纤束缠绕层 MAT_WRAP  E_轴=%.0f MPa  E_环=E_径=%.0f MPa  G=%.0f MPa  nu=%.2f'
      % (E_WRAP_X, E_WRAP_H, G_WRAP, NU_WRAP))
    P('  拉挤 GFRP 块 MAT_BLOCK E_轴=%.0f MPa  E_横=%.0f MPa    G=%.0f MPa  nu=%.2f'
      % (E_BLK_X, E_BLK_T, G_BLK, NU_BLK))
    P('  过渡层（树脂富集）     t=%.1f mm  G=%.0f MPa（沿用父模型胶层常数）'
      % (T_ADH, G_ADH))
    P('  螺孔深度               x < %.0f mm 为空孔，其余实心钢（网格实际切换在 x=132 mm）'
      % X_BORE)
    P('  脱粘等效               G → %.0e × 名义值，法向刚度不变' % DEBOND_GFAC)
    P('  额定挥舞单柱外载       F_A = %.0f kN' % (FA_RATED / 1e3))
    P('  安装预紧               F_S = %.0f kN' % (FS_TARGET / 1e3))
    P('  以上除父模型已有的层压/钢/胶层参数外，全部是推断值，没有实测支撑。')
    P(L)
    return '\n'.join(out) + '\n'


# ===================================================================== 主流程
def main(quick=False, do2d=True, fields=True):
    t0 = time.time()
    try:                      # 本机控制台默认 GBK，报告里有 Ø 之类字符会炸
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                              # noqa: BLE001
        pass
    print('[solver] 稀疏直接解后端 = %s' % SparseLU.backend)
    outdir = os.path.join(HERE, 'out')
    os.makedirs(outdir, exist_ok=True)
    if quick:
        kw = dict(m_y=6, n_cell=3, layers=(1, 1, 2, 1, 1))
    else:
        kw = dict(m_y=20, n_cell=5, layers=(2, 2, 4, 4, 4))

    print('>>> 校验 (a) 本构自洽性')
    vc = verify_constitutive()
    for nm, v in vc.items():
        print('    %-6s 还原误差 %.2e  正定 %s' % (nm, v['err'], v['spd']))
    if max(v['err'] for v in vc.values()) > 1e-10:
        raise RuntimeError('本构矩阵不自洽，停止')

    print('>>> 校验 (b) 单轴拉伸（无螺套）')
    va = verify_uniform(m_y=kw['m_y'], layers=kw['layers'],
                        n_cell=min(kw['n_cell'], 2))
    print('    eps_ref=%.6e  内表面误差 %.3e  厚度平均误差 %.3e'
          % (va['eps_ref'], va['err_inner'], va['err_avg']))
    if max(va['err_inner'], va['err_avg']) > 0.01:
        raise RuntimeError('单轴校验不通过（>1%），停止')

    print('>>> 建模并求解全部工况')
    R = run_all(**kw)
    md, res, JC = R['md'], R['res'], R['JC']

    fa = res['intact']['FA']
    dev = float(np.max(np.abs(fa / fa.mean() - 1.0)))
    print('    完好工况 cell 间 F_A 最大偏差 %.3e %s'
          % (dev, '(通过)' if dev < 0.01 else '(不通过!)'))
    if dev > 0.01:
        raise RuntimeError('完好工况各 cell F_A 不均匀（>1%），停止')

    if do2d:
        print('>>> 二维模型对照')
        try:
            R['d2'] = compare_2d(R)
        except Exception as exc:                                  # noqa: BLE001
            print('    二维对照失败：%r' % (exc,))
            R['d2'] = None
    else:
        R['d2'] = None

    print('>>> 写 npz')
    npz = os.path.join(outdir, 'sector_layered.npz')
    d = write_npz(R, npz, fields=fields)
    sz = os.path.getsize(npz)
    print('    %s  %.1f MB  %d 个键' % (npz, sz / 2 ** 20, len(d)))
    readme = os.path.join(outdir, 'README_field.txt')
    write_readme(R, readme)
    print('    %s' % readme)

    txt = report(R, va, vc)
    txt += '\n  npz 文件 %.2f MB，共 %d 个键：\n' % (sz / 2 ** 20, len(d))
    for i in range(0, len(d), 4):
        txt += '    ' + '  '.join('%-28s' % k for k in list(d.keys())[i:i + 4]) + '\n'
    txt += '\n  总耗时 %.1f s\n' % (time.time() - t0)
    print(txt)
    return R, txt


def compare_2d(R):
    """与二维展开壳模型 root_model.RootFE 的量级对照（同预紧、同单柱外载）。"""
    from root_model import RootFE
    md, res, JC = R['md'], R['res'], R['JC']
    fe = RootFE(n_cells=md.n_cell, m_y=4)
    FM2 = fe.calibrate_FM(FS_TARGET)

    def solve_FA(FA):
        r = fe.solve(0.0, FM2)
        r1 = fe.solve(-1e-3, FM2)
        k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3)
        w = 0.0
        for _ in range(40):
            if abs(r['FA'].mean() - FA) < max(5.0, 1e-6 * abs(FA)):
                break
            w += (FA - r['FA'].mean()) / k
            r = fe.solve(w, FM2)
        r['w'] = w
        return r

    p0 = fe.solve(0.0, FM2)
    p2 = solve_FA(FA_RATED)
    ri = res['intact']
    rr = R['r_ref']
    Phi3 = (ri['FS'].mean() - rr['FS'].mean()) /         max(ri['FA'].mean() - rr['FA'].mean(), 1e-9)
    Phi2 = (p2['FS'].mean() - p0['FS'].mean()) / \
        max(p2['FA'].mean() - p0['FA'].mean(), 1e-9)
    e3 = np.interp(200.0, md.xc, ri['eps_in']) * 1e6
    x2 = p2['x_c']
    e2 = np.interp(200.0, x2, p2['eps'][fe.iy_center[JC] % fe.ny, :]) * 1e6
    e3f = np.interp(800.0, md.xc, ri['eps_in']) * 1e6
    e2f = np.interp(800.0, x2, p2['eps'][fe.iy_center[JC] % fe.ny, :]) * 1e6
    rows = [('螺栓载荷系数 Phi', Phi3, Phi2),
            ('标定后 F_M [kN]', R['FM'] / 1e3, FM2 / 1e3),
            ('F_S [kN] (完好,159kN)', ri['FS'].mean() / 1e3, p2['FS'].mean() / 1e3),
            ('端面夹紧力 [kN]', ri['C'].mean() / 1e3,
             (p2['Cl'] + p2['Cs']).mean() / 1e3),
            ('法兰位移 w [mm]', R['w0'], p2['w']),
            ('eps(x=200) [ue]', e3, e2),
            ('eps(x=800) [ue]', e3f, e2f)]
    return dict(rows=rows, fe=fe, FM2=FM2, p2=p2)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='粗网格冒烟测试')
    ap.add_argument('--no-2d', action='store_true')
    ap.add_argument('--no-fields', action='store_true')
    a = ap.parse_args()
    main(quick=a.quick, do2d=not a.no_2d, fields=not a.no_fields)
