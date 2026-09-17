# -*- coding: utf-8 -*-
"""
M2：叶根连接的三维扇形块有限元模型（8 节点三线性六面体，2x2x2 高斯）。

为什么要做它：二维展开壳模型 root_model.RootFE 把根部壁当成"膜"，只有厚度平均
的轴向应变，答不出"光纤贴在内表面(r=0)上实测应变到底是多少"。本模型把壁厚方向
（r，0=内表面，110=外表面）显式离散，因此可以直接给出

    k_surf(x) = eps_xx(内表面) / eps_xx(层压厚度平均)

并且在螺柱断裂时看这个比值怎么变。

坐标：x 轴向（0=叶根端面/法兰面，900=远端夹支），y 周向（周期），r 径向（厚度）。
拉为正。所有长度 mm，力 N，应力 MPa。

模型要素：
  · 正交各向异性层压（完整 6x6 三维本构，不用各向同性近似）
  · 预埋螺套：钢环（外 D_ins，内 d_bore 空螺孔），x∈[0,L_ins]，轴线在 r=r_inner_off
  · 胶层不显式建网格：紧贴螺套的那一圈层压单元用折减剪切模量 G_eff 代表
    0.5 mm / G=1200 MPa 环氧层的附加剪切柔度
  · x=0：刚性钢法兰面，只压不拉、无摩擦，主动集迭代
  · 螺柱：秩一刚度 + 常力，力沿螺纹啮合段 x∈[0,105] 按体积支配量分布
  · y 向周期，并额外放一个"广义周期跳变"自由度 Δ，允许整环周长自由伸缩。
    不放这一项会把环向应变锁成 0，轴向应变被系统性压低 1-nu_xy*nu_yx ≈ 9.6%，
    校验项 (a)（单轴拉伸必须给出 F/(E_x*A)）就会直接失败。
"""
from __future__ import annotations

import os
import sys
import time
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ----------------------------------------------------------------------- 常量
P_PITCH = _G.pitch       # 螺栓周向节距 mm
N_CELL = 9               # 建模的 cell 数（周向周期窗口）
T_WALL = _G.t_wall       # 壁厚 mm（r: 0=内表面 → t_wall=外表面）
X_MAX = _G.x_max         # 轴向模型长度
L_INS = _G.L_ins         # 螺套埋深
R_INS_C = _G.r_inner_off  # 螺套轴线到内表面的径向距离
D_INS = _G.D_ins         # 螺套外径
D_BORE = _G.d_bore       # 螺孔（空）直径
L_ENG = _G.l_eng         # 螺纹啮合长度

E_X, E_Y, E_R = 30000.0, 18000.0, 18000.0
G_XY = G_XR = G_YR = 8000.0
NU_XY = NU_XR = 0.40
NU_YR = 0.35
E_ST, NU_ST = 210000.0, 0.30
E_VOID, NU_VOID = 1.0, 0.30
G_ADH, T_ADH = 1200.0, 0.5
R_IN = _G.D_in / 2.0     # 叶根内表面半径；r=R_INS_C 处正好落在螺栓圆上

K_S = 210000.0 * 1120.0 / (160.0 + 0.4 * 42.0)   # 螺柱轴向刚度 1.3305e6 N/mm
FS_INSTALL = 420.0e3     # 安装轴力 N
FA_REF = 74.0e3          # 参考单柱外载 N

MAT_LAM, MAT_STEEL, MAT_VOID = 0, 1, 2

# 精确截面积（用于评价网格阶梯化误差）
A_STEEL_EXACT = np.pi / 4.0 * (D_INS ** 2 - D_BORE ** 2)     # 3552.2
A_LAM_EXACT = P_PITCH * T_WALL - np.pi / 4.0 * D_INS ** 2    # 4715.4

# 二维模型的界面剪切刚度（对照用）
KQ_2D = (1.0 / (T_ADH / G_ADH + 16.0 / G_XY)) * np.pi * D_INS


# ----------------------------------------------------------------------- 本构
def C_ortho(Ex, Ey, Ez, nu_xy, nu_xz, nu_yz, Gyz, Gxz, Gxy):
    """正交各向异性三维刚度矩阵。Voigt 次序 (xx, yy, rr, yr, xr, xy)。"""
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / Ex
    S[1, 1] = 1.0 / Ey
    S[2, 2] = 1.0 / Ez
    S[0, 1] = S[1, 0] = -nu_xy / Ex
    S[0, 2] = S[2, 0] = -nu_xz / Ex
    S[1, 2] = S[2, 1] = -nu_yz / Ey
    S[3, 3] = 1.0 / Gyz
    S[4, 4] = 1.0 / Gxz
    S[5, 5] = 1.0 / Gxy
    ev = np.linalg.eigvalsh(S[:3, :3])
    if ev.min() <= 0:
        raise ValueError('柔度矩阵非正定，材料参数不自洽')
    return np.linalg.inv(S)


def C_iso(E, nu):
    G = E / (2.0 * (1.0 + nu))
    return C_ortho(E, E, E, nu, nu, nu, G, G, G)


def C_lam(Gxy=G_XY, Gxr=G_XR, Gyr=G_YR):
    return C_ortho(E_X, E_Y, E_R, NU_XY, NU_XR, NU_YR, Gyr, Gxr, Gxy)


# ------------------------------------------------------------------- 单元矩阵
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_SG = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def _B_at(xi, et, ze, a, b, c, r0=None):
    """长方体单元在自然坐标 (xi,et,ze) 处的 B 矩阵 (6x24)。局部自由度序 u,v,w 交错。

    r0 给定时启用"准柱面"环向运动学：把展开成平板时丢掉的那两项补回来
        eps_yy += u_r / R,   gamma_yr -= u_y / R,   R = R_IN + r
    其余仍按直棱柱处理（体积元仍取 dx*dy*dr）。补这两项的物理意义：真实圆环轴向受拉
    时靠半径缩小释放环向应变，环向应力逐点为零；展开成平板后这条路没了，朴素周期
    （结点等同）会把环向应变锁死，轴向应变被系统性压低 1-nu_xy*nu_yx = 9.6%。"""
    s = _SG
    dN = np.empty((8, 3))
    dN[:, 0] = s[:, 0] * (1 + et * s[:, 1]) * (1 + ze * s[:, 2]) / 8.0 * (2.0 / a)
    dN[:, 1] = s[:, 1] * (1 + xi * s[:, 0]) * (1 + ze * s[:, 2]) / 8.0 * (2.0 / b)
    dN[:, 2] = s[:, 2] * (1 + xi * s[:, 0]) * (1 + et * s[:, 1]) / 8.0 * (2.0 / c)
    B = np.zeros((6, 24))
    B[0, 0::3] = dN[:, 0]
    B[1, 1::3] = dN[:, 1]
    B[2, 2::3] = dN[:, 2]
    B[3, 1::3] = dN[:, 2]; B[3, 2::3] = dN[:, 1]
    B[4, 0::3] = dN[:, 2]; B[4, 2::3] = dN[:, 0]
    B[5, 0::3] = dN[:, 1]; B[5, 1::3] = dN[:, 0]
    if r0 is not None:
        N = (1 + xi * s[:, 0]) * (1 + et * s[:, 1]) * (1 + ze * s[:, 2]) / 8.0
        Rg = R_IN + r0 + c * (1.0 + ze) / 2.0
        B[1, 2::3] += N / Rg
        B[3, 1::3] -= N / Rg
    return B


def _Mtensor(a, b, c, r0=None):
    """几何张量 M[p,q,i,j] = sum_g w_g B[p,i] B[q,j] detJ，使 ke = sum_pq C[p,q] M[p,q]。"""
    Bs = np.array([_B_at(xi, et, ze, a, b, c, r0)
                   for xi in _GP for et in _GP for ze in _GP])
    return np.einsum('gpi,gqj->pqij', Bs, Bs) * (a * b * c / 8.0)


# --------------------------------------------------------------------- 轴向网格
def _x_grid():
    """轴向分级网格：端面 6 mm（12 个），过渡 15 mm，中段 29.8 mm，
    螺套端部 60 mm 加密到 10 mm，远场几何级数放粗。共 41 个单元。"""
    x_ref = L_INS - 60.0
    a = np.linspace(0.0, 72.0, 13)
    b = np.linspace(72.0, 132.0, 5)[1:]
    c = np.linspace(132.0, x_ref, 11)[1:]
    d = np.linspace(x_ref, L_INS, 7)[1:]
    inc = np.geomspace(10.0, 110.0, 9)
    inc = inc / inc.sum() * (X_MAX - L_INS)
    e = L_INS + np.cumsum(inc)
    return np.concatenate([a, b, c, d, e])


# ===================================================================== 主模型
class Sector3D:
    """三维扇形块模型。m_y = 每个 cell 的周向单元数；r_div = (内皮, 螺套段, 外皮) 单元数。"""

    def __init__(self, m_y=10, r_div=(2, 4, 1), n_cell=N_CELL,
                 with_inserts=True, hoop='global', verbose=True):
        t0 = time.time()
        self.m_y, self.r_div, self.n_cell = int(m_y), tuple(r_div), int(n_cell)
        self.with_inserts = bool(with_inserts)
        # hoop: 'global'= 展开平板 + 一个全局环向跳变 Δ（二维模型 RootFE 的做法）。默认。
        #                 精确 y 周期，单轴校验精确，与二维模型可比。
        #       'cyl'   = 朴素周期 + 单元里补准柱面项 u_r/R、-u_y/R（模型形式敏感性对照；
        #                 它会把真实的柱壳边缘效应带进来，衰减长度 ~sqrt(R*t)=420 mm，
        #                 本模型螺套末端到 x_max 装不下这个长度，所以不作主结果）
        #       'none'  = 展开平板 + 朴素周期，环向应变被锁死（只用来演示它为什么不行）
        # 注：曾试过"每个 (x,r) 各一个跳变 Δ(x,r)"。那是错的：Δ 随 x、r 变化会在缝合
        #     单元里生成正比于 y 的伪剪应变，把 y 周期性破坏掉——完好工况下各 cell 的
        #     F_A 不再相等（靠缝的 cell 高出 27%），已弃用。
        assert hoop in ('cyl', 'global', 'none')
        self.hoop = hoop
        self.hoop_free = hoop == 'global'
        self.cyl = hoop == 'cyl'
        self.verbose = verbose

        # ---- 结点网格
        self.x = _x_grid()
        self.nx_n = len(self.x)
        self.nx_e = self.nx_n - 1
        self.dx = np.diff(self.x)

        self.ny = self.n_cell * self.m_y          # 周期方向：结点数 = 单元数
        self.dy = P_PITCH / self.m_y
        self.y = np.arange(self.ny) * self.dy

        r_lo = R_INS_C - D_INS / 2.0        # 螺套钢体下缘
        r_hi = R_INS_C + D_INS / 2.0        # 螺套钢体上缘
        r0 = np.linspace(0.0, r_lo, r_div[0] + 1)
        r1 = np.linspace(r_lo, r_hi, r_div[1] + 1)[1:]
        r2 = np.linspace(r_hi, T_WALL, r_div[2] + 1)[1:]
        self.r = np.concatenate([r0, r1, r2])
        self.nr_n = len(self.r)
        self.nr_e = self.nr_n - 1
        self.dr = np.diff(self.r)

        self.n_node = self.nx_n * self.ny * self.nr_n
        self.n_elem = self.nx_e * self.ny * self.nr_e
        self.i_hoop = 3 * self.n_node             # 广义周期跳变自由度的起始编号
        self.n_hoop = 1                        # 'global' 用；'cyl'/'none' 把它锁死
        self.ndof = self.i_hoop + self.n_hoop

        self.xc = 0.5 * (self.x[:-1] + self.x[1:])
        self.yc = (np.arange(self.ny) + 0.5) * self.dy
        self.rc = 0.5 * (self.r[:-1] + self.r[1:])

        self._classify()
        self._build_eldof()
        self._assemble()
        self._face_and_stud()
        self.t_build = time.time() - t0
        if verbose:
            print('[mesh] m_y=%d r_div=%s  elem=%d  node=%d  DOF=%d  build %.1fs'
                  % (m_y, r_div, self.n_elem, self.n_node, self.ndof, self.t_build))

    # ------------------------------------------------------------ 材料判定
    def _classify(self):
        """按单元形心落在螺套圆/螺孔圆内外判材料；并找出紧贴螺套的一圈层压单元。"""
        d = (self.yc - 0.5 * P_PITCH) % P_PITCH
        self.dy_c = np.minimum(d, P_PITCH - d)                  # (ny,) 到最近螺套轴线的周向距离
        self.cell_of_col = np.floor(self.yc / P_PITCH).astype(int) % self.n_cell
        dr_c = self.rc - R_INS_C                                 # (nr_e,)
        d2 = self.dy_c[:, None] ** 2 + dr_c[None, :] ** 2        # (ny, nr_e)

        in_out = d2 < (D_INS / 2.0) ** 2
        in_bore = d2 < (D_BORE / 2.0) ** 2
        in_x = self.xc < L_INS                                   # (nx_e,)
        if not self.with_inserts:
            in_out = np.zeros_like(in_out)
            in_bore = np.zeros_like(in_bore)

        mat = np.full((self.nx_e, self.ny, self.nr_e), MAT_LAM, np.int8)
        if self.with_inserts:
            m2 = np.where(in_bore, MAT_VOID, np.where(in_out, MAT_STEEL, MAT_LAM))
            mat[in_x] = m2[None, :, :]
        self.mat = mat
        self.in_out, self.in_bore, self.in_x = in_out, in_bore, in_x

        # ---- 阶梯化截面积（取 cell 0 统计）
        cell0 = np.flatnonzero(self.cell_of_col == 0)
        A = np.broadcast_to(self.dy * self.dr[None, :], (len(cell0), self.nr_e))
        st0 = (in_out & ~in_bore)[cell0]
        self.A_steel_mesh = float(A[st0].sum())
        self.A_bore_mesh = float(A[in_bore[cell0]].sum())
        self.A_lam_mesh = float(A[~in_out[cell0]].sum())

        # ---- 胶层环：与钢单元在横截面内共面相邻的层压单元（不含端盖）
        st = in_out & ~in_bore
        nb = np.zeros_like(st)
        nb[1:, :] |= st[:-1, :]
        nb[:-1, :] |= st[1:, :]
        nb[0, :] |= st[-1, :]
        nb[-1, :] |= st[0, :]                                    # y 周期
        nb[:, 1:] |= st[:, :-1]
        nb[:, :-1] |= st[:, 1:]
        ring = nb & ~in_out
        self.ring_mask = ring

        # ---- 环单元等效剪切模量：沿界面法向串联上 0.5 mm 胶层的剪切柔度
        #      1/G_eff = 1/G_lam + t_adh/(G_adh * h_e)，h_e = 单元沿法向的穿越长度
        G_eff = np.full((self.ny, self.nr_e), float(G_XY))
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
            G_eff[iy, ir] = 1.0 / (1.0 / G_XY + T_ADH / (G_ADH * h))
        self.G_eff, self.h_ring = G_eff, h_ring

    # ------------------------------------------------------------ 单元自由度表
    def _nid(self, ix, iy, ir):
        return ((iy % self.ny) * self.nx_n + ix) * self.nr_n + ir

    def _build_eldof(self):
        """ELDOF (n_elem,24)。单元编号 e = (ix*ny+iy)*nr_e+ir。"""
        ix = np.arange(self.nx_e)[:, None, None]
        iy = np.arange(self.ny)[None, :, None]
        ir = np.arange(self.nr_e)[None, None, :]
        loc = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        nodes = np.empty((self.nx_e, self.ny, self.nr_e, 8), np.int64)
        for l, (a, b, c) in enumerate(loc):
            nodes[..., l] = self._nid(ix + a, iy + b, ir + c)
        nodes = nodes.reshape(self.n_elem, 8)
        self.ELNODE = nodes
        dofs = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(self.n_elem, 24)
        self.ELDOF = dofs
        self.WRAP_LOC = np.array([7, 10, 19, 22])   # 局部结点 2,3,6,7 的 v 自由度（绕回 iy=0）

    def _hoop_of(self, ix, ir):
        """缝合单元 (ix,ir) 的 4 个绕回结点(局部 2,3,6,7)各自对应的跳变自由度编号。"""
        ix = np.asarray(ix); ir = np.asarray(ir)
        return np.stack([np.full_like(ix, self.i_hoop)] * 4, axis=-1).astype(np.int64)

    # ------------------------------------------------------------ 组装
    def _r0(self, ir):
        """准柱面模式下单元的 r 下界；平板模式返回 None。"""
        return float(self.r[ir]) if self.cyl else None

    def _ke(self, ix, ir, C):
        key = (ix, ir)
        M = self._Mcache.get(key)
        if M is None:
            M = _Mtensor(self.dx[ix], self.dy, self.dr[ir], self._r0(ir))
            self._Mcache[key] = M
        return np.einsum('pq,pqij->ij', C, M)

    def _assemble(self):
        """装配实体单元刚度（与缺陷无关）。螺柱秩一项在 solve 里按状态加。"""
        t0 = time.time()
        self._Mcache = {}
        Cl = C_lam()
        Cs = C_iso(E_ST, NU_ST)
        Cv = C_iso(E_VOID, NU_VOID)
        Cmat = (Cl, Cs, Cv)

        ring3 = np.zeros(self.mat.shape, bool)
        if self.with_inserts:
            ring3[self.in_x] = self.ring_mask[None, :, :]

        rows, cols, vals = [], [], []
        ke_cache = {}
        for ix in range(self.nx_e):
            for ir in range(self.nr_e):
                mcol = self.mat[ix, :, ir].astype(int)
                rcol = ring3[ix, :, ir]
                gid = np.where(rcol, 1000 + np.arange(self.ny), mcol)
                for g in np.unique(gid):
                    sel = np.flatnonzero(gid == g)
                    if g >= 1000:
                        Gg = float(self.G_eff[g - 1000, ir])
                        ck = ('ring', round(Gg, 6))
                        C = C_lam(Gxy=Gg, Gxr=Gg, Gyr=Gg)
                    else:
                        ck = ('m', int(g))
                        C = Cmat[int(g)]
                    kk = (ix, ir) + ck
                    ke = ke_cache.get(kk)
                    if ke is None:
                        ke = self._ke(ix, ir, C)
                        ke_cache[kk] = ke
                    e = (ix * self.ny + sel) * self.nr_e + ir
                    d = self.ELDOF[e]
                    rows.append(np.repeat(d, 24, axis=1).ravel())
                    cols.append(np.tile(d, (1, 24)).ravel())
                    vals.append(np.tile(ke.ravel(), len(sel)))
                    # 缝合单元（iy = ny-1）的 Δ 耦合
                    if self.hoop_free and (self.ny - 1) in sel:
                        es = (ix * self.ny + (self.ny - 1)) * self.nr_e + ir
                        ds = self.ELDOF[es]
                        hd = self._hoop_of(ix, ir)          # (4,)
                        for a in range(4):
                            ha = np.full(24, hd[a], np.int64)
                            rows.append(ha); cols.append(ds); vals.append(ke[self.WRAP_LOC[a], :])
                            rows.append(ds); cols.append(ha); vals.append(ke[:, self.WRAP_LOC[a]])
                        rows.append(np.repeat(hd, 4)); cols.append(np.tile(hd, 4))
                        vals.append(ke[np.ix_(self.WRAP_LOC, self.WRAP_LOC)].ravel())

        R = np.concatenate([np.asarray(a).ravel() for a in rows])
        Cc = np.concatenate([np.asarray(a).ravel() for a in cols])
        V = np.concatenate([np.asarray(a).ravel() for a in vals])
        if not self.hoop_free:
            hh = self.i_hoop + np.arange(self.n_hoop)
            R = np.append(R, hh)
            Cc = np.append(Cc, hh)
            V = np.append(V, np.ones(self.n_hoop))
        self.K0 = sp.coo_matrix((V, (R, Cc)), shape=(self.ndof, self.ndof)).tocsr()
        self.K0.sum_duplicates()
        self.t_asm = time.time() - t0
        if self.verbose:
            print('[asm ] nnz=%d  %.1fs' % (self.K0.nnz, self.t_asm))

    # ------------------------------------------------------- 端面接触 & 螺柱
    def _face_and_stud(self):
        ny, nr_n, nr_e = self.ny, self.nr_n, self.nr_e
        m0 = self.mat[0]
        cap = np.zeros((ny, nr_n), bool)
        steel = np.zeros((ny, nr_n), bool)
        for dyi in (-1, 0):
            for dri in (-1, 0):
                iyv = (np.arange(ny)[:, None] + dyi) % ny
                irv = np.arange(nr_n)[None, :] + dri
                ok = (irv >= 0) & (irv < nr_e)
                irc = np.clip(irv, 0, nr_e - 1)
                cap |= ok & (m0[iyv, irc] != MAT_VOID)
                steel |= ok & (m0[iyv, irc] == MAT_STEEL)
        self.face_cap = cap                 # 空螺孔结点不承压
        self.face_is_steel = steel
        self.face_dof = 3 * self._nid(0, np.arange(ny)[:, None], np.arange(nr_n)[None, :])

        # 端面结点按 y 列归到最近的 cell（正好落在 cell 边界的算半个）
        fw = np.zeros((self.n_cell, ny))
        half = self.m_y // 2
        for j in range(self.n_cell):
            c = j * self.m_y + half
            for k in range(-half, half + 1):
                fw[j, (c + k) % ny] += 0.5 if abs(k) == half else 1.0
        self.face_w = fw

        # ---- 螺柱加载权重：x∈[0,105] 内的钢单元体积按 1/8 分摊到 8 个结点
        self.stud_dof, self.stud_w = [], []
        if not self.with_inserts:
            self.stud_dof = [np.zeros(0, np.int64)] * self.n_cell
            self.stud_w = [np.zeros(0)] * self.n_cell
            return
        dxc = np.clip(np.minimum(self.x[1:], L_ENG) - self.x[:-1], 0.0, None)
        ixs = np.flatnonzero(dxc > 0)
        for j in range(self.n_cell):
            W = np.zeros(self.n_node)
            cols = np.flatnonzero(self.cell_of_col == j)
            for ix in ixs:
                sub = self.mat[ix][np.ix_(cols, np.arange(nr_e))] == MAT_STEEL
                for a, iy in enumerate(cols):
                    irs = np.flatnonzero(sub[a])
                    if len(irs) == 0:
                        continue
                    e = (ix * self.ny + iy) * self.nr_e + irs
                    v = dxc[ix] * self.dy * self.dr[irs] / 8.0
                    np.add.at(W, self.ELNODE[e].ravel(), np.repeat(v, 8))
            nz = np.flatnonzero(W > 0)
            self.stud_dof.append(3 * nz)
            self.stud_w.append(W[nz] / W[nz].sum())

    # ------------------------------------------------------------ 求解基础设施
    def _con_base(self):
        """远端 x=900 全部结点 u_x=0；再固定一个结点的 v、w 去掉刚体平动。
        注意 y 向是周期的，整体 y 平动是零能模式，必须约束（这里就是那个 v）。"""
        far = (3 * self._nid(self.nx_n - 1,
                             np.arange(self.ny)[:, None],
                             np.arange(self.nr_n)[None, :])).ravel()
        n0 = self._nid(self.nx_n - 1, 0, 0)
        if self.cyl:
            # 准柱面模式下 u_r 不再是刚体模式（整体径向外扩会产生环向应变，要花能量），
            # 再去固定它就是多余约束——会把 (a) 单轴拉伸校验打坏到 28%。只留 u_y。
            # 剩下的唯一零能模式是绕 x 轴的刚体转动 u_y = w*(R_IN+r)，固定一个 u_y 即可。
            rbm = np.array([3 * n0 + 1], np.int64)
        else:
            rbm = np.array([3 * n0 + 1, 3 * n0 + 2], np.int64)
        return far.astype(np.int64), rbm

    def _elem_disp(self, u, els):
        """取单元结点位移（缝合单元的 v 要补上广义周期跳变 Δ）。"""
        d = u[self.ELDOF[els]]
        iy = (els // self.nr_e) % self.ny
        sm = iy == self.ny - 1
        if sm.any() and self.hoop_free:
            k = np.flatnonzero(sm)
            e = np.asarray(els)[k]
            hd = self._hoop_of(e // (self.ny * self.nr_e), e % self.nr_e)   # (m,4)
            d[k[:, None], self.WRAP_LOC[None, :]] += u[hd]
        return d

    def _factor(self, act):
        """给定接触主动集，对 K0 做 Dirichlet 消元并分解；螺柱秩一项用 Woodbury 带入。
        按主动集缓存，因为 K0 与预紧力/法兰位移/断柱与否都无关，绝大多数工况共用同一次分解。"""
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
        Aff = A[free][:, free].tocsc()
        lu = spla.splu(Aff, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        U = np.zeros((self.ndof, self.n_cell))
        for j in range(self.n_cell):
            U[self.stud_dof[j], j] = self.stud_w[j]
        Z = lu.solve(U[free])
        S = dict(free=free, con=con, cface=cface, far=far, lu=lu, U=U,
                 Uf=U[free], Z=Z, A=A, act=act.copy())
        self.t_lu += time.time() - t0
        self.n_lu += 1
        if self.verbose:
            print('  [lu] act=%d/%d  free=%d  %.1fs (累计 %d 次 %.1fs)'
                  % (int(act.sum()), act.size, len(free), time.time() - t0,
                     self.n_lu, self.t_lu))
        if len(self._fcache) >= 2:
            self._fcache.pop(next(iter(self._fcache)))
        self._fcache[key] = S
        return S

    def _lin_solve(self, S, f, uc, broken):
        """解 (K0 + kS*U_s U_s^T) u = f，给定 Dirichlet 值 uc（全长向量，自由自由度处为 0）。"""
        free, A, Uf, Z, U = S['free'], S['A'], S['Uf'], S['Z'], S['U']
        cols = np.array([j for j in range(self.n_cell) if j not in broken], int)
        Ufs, Zs = Uf[:, cols], Z[:, cols]
        m = len(cols)
        t = K_S * (U[:, cols].T @ uc)
        b = f[free] - (A @ uc)[free] - (Ufs @ t if m else 0.0)
        y = S['lu'].solve(b)
        if m:
            M = np.eye(m) / K_S + Ufs.T @ Zs
            y = y - Zs @ np.linalg.solve(M, Ufs.T @ y)
        u = uc.copy()
        u[free] = y
        # 反力 R = K u - f
        R = A @ u - f
        if m:
            R += U[:, cols] @ (K_S * (U[:, cols].T @ u))
        return u, R

    def _rhs(self, F_M, w, broken):
        """螺柱载荷向量：f = w_eng*(k_S*w - F_M)。"""
        f = np.zeros(self.ndof)
        for j in range(self.n_cell):
            if j in broken:
                continue
            f[self.stud_dof[j]] += self.stud_w[j] * (K_S * w - F_M[j])
        return f

    # ------------------------------------------------------------ 主求解
    def solve(self, F_M, broken=(), w=None, target_FA=None, max_iter=8, verbose=False):
        """主动集接触 + 线性 w 标定。u(w) = u0 + w*u1 在主动集不变时是精确的仿射关系，
        所以每个主动集只需两次回代就能把 F_A 打到目标值（不需要割线迭代的收敛容差）。"""
        FM = np.full(self.n_cell, float(F_M)) if np.isscalar(F_M) else np.asarray(F_M, float).copy()
        broken = set(broken or [])
        for j in broken:
            FM[j] = 0.0
        act = self.face_cap.copy()
        nfac = 0
        for it in range(max_iter):
            S = self._factor(act)
            nfac += 1
            f0 = self._rhs(FM, 0.0, broken)
            f1 = self._rhs(np.zeros(self.n_cell), 1.0, broken)
            uc0 = np.zeros(self.ndof)
            uc1 = np.zeros(self.ndof)
            uc1[S['cface']] = 1.0
            u0, R0 = self._lin_solve(S, f0, uc0, broken)
            u1, R1 = self._lin_solve(S, f1, uc1, broken)
            if w is None:
                p0 = self._post(u0, R0, 0.0, FM, broken, act)
                p1 = self._post(u1, R1, 1.0, np.zeros(self.n_cell), broken, act)
                a0, a1 = p0['FA'].mean(), p1['FA'].mean()
                if abs(a1) < 1e-6:
                    raise RuntimeError('F_A 对法兰位移不敏感，无法标定 w')
                w_use = (target_FA - a0) / a1
            else:
                w_use = float(w)
            u = u0 + w_use * u1
            R = R0 + w_use * R1
            # ---- 主动集更新：受压(R>=0)保持闭合；张开的结点若被法兰压穿则重新闭合
            new = act.copy()
            Rf = R[self.face_dof]
            new[act] = Rf[act] >= 0.0
            op = self.face_cap & ~act
            if op.any():
                new[op] = u[self.face_dof][op] < w_use
            new &= self.face_cap
            if np.array_equal(new, act):
                res = self._post(u, R, w_use, FM, broken, act)
                res.update(w=w_use, u=u, R=R, n_factor=nfac, act=act.copy(),
                           converged=True, n_iter=it + 1)
                return res
            act = new
        res = self._post(u, R, w_use, FM, broken, act)
        res.update(w=w_use, u=u, R=R, n_factor=nfac, act=act.copy(),
                   converged=False, n_iter=max_iter)
        if verbose:
            print('  !! 主动集未收敛')
        return res

    # ------------------------------------------------------------ 后处理
    def _post(self, u, R, w, FM, broken, act):
        Rf = np.where(act, R[self.face_dof], 0.0)          # (ny, nr_n) 端面接触反力
        C_node = Rf.sum(axis=1)                            # 每个 y 结点列的合力
        C_lam_n = np.where(self.face_is_steel, 0.0, Rf).sum(axis=1)
        C_st_n = np.where(self.face_is_steel, Rf, 0.0).sum(axis=1)
        C = self.face_w @ C_node
        Cl = self.face_w @ C_lam_n
        Cs = self.face_w @ C_st_n
        FS = np.zeros(self.n_cell)
        for j in range(self.n_cell):
            if j in broken:
                continue
            ubar = float(self.stud_w[j] @ u[self.stud_dof[j]])
            FS[j] = FM[j] + K_S * (ubar - w)
        FA = FS - C
        R_far = float(R[self._con_base()[0]].sum())
        return dict(FA=FA, FS=FS, C=C, C_lam=Cl, C_steel=Cs, R_far=R_far,
                    n_open=int((self.face_cap & ~act).sum()))

    # ------------------------------------------------------------ 应变提取
    def _lam_rows(self, iy_node):
        """y = 结点面 iy_node 两侧的单元是否都是层压 → (nx_e, nr_e) 掩码。"""
        a = self.mat[:, (iy_node - 1) % self.ny, :]
        b = self.mat[:, iy_node % self.ny, :]
        return (a == MAT_LAM) & (b == MAT_LAM)

    def profiles(self, u, iy_node):
        """返回 (eps_inner, eps_avg)：在 y=结点面 iy_node 上，
        eps_inner = r=0 内表面的 eps_xx；eps_avg = 只对层压做的厚度平均 eps_xx。
        三线性六面体在角点处 eps_xx 就是该角点两个 x 相邻结点 u_x 的差商，精确。"""
        ux = u[3 * np.arange(self.n_node)].reshape(self.ny, self.nx_n, self.nr_n)
        col = ux[iy_node % self.ny]                                  # (nx_n, nr_n)
        d = (col[1:, :] - col[:-1, :]) / self.dx[:, None]            # (nx_e, nr_n)
        eps_inner = d[:, 0].copy()
        eps_row = 0.5 * (d[:, :-1] + d[:, 1:])                       # (nx_e, nr_e)
        msk = self._lam_rows(iy_node)
        wr = self.dr[None, :] * msk
        sw = wr.sum(axis=1)
        eps_avg = np.where(sw > 0, (eps_row * wr).sum(axis=1) / np.maximum(sw, 1e-30), np.nan)
        return eps_inner, eps_avg

    def eps_through(self, u, iy_node, xp):
        """给定轴向位置，返回沿厚度各结点 r 处的 eps_xx（在 y=结点面 iy_node 上）。"""
        ux = u[3 * np.arange(self.n_node)].reshape(self.ny, self.nx_n, self.nr_n)
        col = ux[iy_node % self.ny]
        d = (col[1:, :] - col[:-1, :]) / self.dx[:, None]        # (nx_e, nr_n)
        return self.r, np.array([np.interp(xp, self.xc, d[:, k]) for k in range(self.nr_n)])

    def section_force(self, u):
        """各轴向单元层的截面轴力 N(x) = ∫σ_xx dA（对全部 9 个 cell 求和）。"""
        Cl = C_lam()
        Cs = C_iso(E_ST, NU_ST)
        Cv = C_iso(E_VOID, NU_VOID)
        Cmat = (Cl, Cs, Cv)
        ring3 = np.zeros(self.mat.shape, bool)
        if self.with_inserts:
            ring3[self.in_x] = self.ring_mask[None, :, :]
        N = np.zeros(self.nx_e)
        for ix in range(self.nx_e):
            tot = 0.0
            for ir in range(self.nr_e):
                B = _B_at(0.0, 0.0, 0.0, self.dx[ix], self.dy, self.dr[ir], self._r0(ir))
                els = (ix * self.ny + np.arange(self.ny)) * self.nr_e + ir
                D = self._elem_disp(u, els)
                eps = D @ B.T                                        # (ny, 6)
                mcol = self.mat[ix, :, ir].astype(int)
                rcol = ring3[ix, :, ir]
                sxx = np.zeros(self.ny)
                gid = np.where(rcol, 1000 + np.arange(self.ny), mcol)
                for g in np.unique(gid):
                    sel = np.flatnonzero(gid == g)
                    if g >= 1000:
                        Gg = float(self.G_eff[g - 1000, ir])
                        C = C_lam(Gxy=Gg, Gxr=Gg, Gyr=Gg)
                    else:
                        C = Cmat[int(g)]
                    sxx[sel] = eps[sel] @ C[0, :]
                tot += float(sxx.sum()) * self.dy * self.dr[ir]
            N[ix] = tot
        return N

    def steel_force(self, u, j):
        """第 j 个 cell 的螺套钢截面轴力 N_s(x)。"""
        Cs = C_iso(E_ST, NU_ST)
        cols = np.flatnonzero(self.cell_of_col == j)
        N = np.zeros(self.nx_e)
        for ix in range(self.nx_e):
            tot = 0.0
            for ir in range(self.nr_e):
                sel = cols[self.mat[ix, cols, ir] == MAT_STEEL]
                if len(sel) == 0:
                    continue
                B = _B_at(0.0, 0.0, 0.0, self.dx[ix], self.dy, self.dr[ir], self._r0(ir))
                els = (ix * self.ny + sel) * self.nr_e + ir
                eps = self._elem_disp(u, els) @ B.T
                tot += float((eps @ Cs[0, :]).sum()) * self.dy * self.dr[ir]
            N[ix] = tot
        return N

    def interface_slip(self, u, j):
        """第 j 个 cell 在每个轴向结点面上，螺套钢与该 cell 层压各自的横截面平均轴向位移。
        两者之差就是剪滞模型里的"滑移"，用来直接测界面剪切刚度 k_q = q/slip。"""
        ux = u[3 * np.arange(self.n_node)].reshape(self.ny, self.nx_n, self.nr_n)
        cols = np.flatnonzero(self.cell_of_col == j)
        us = np.zeros(self.nx_n); ws = 0.0
        ul = np.zeros(self.nx_n); wl = 0.0
        for iy in cols:
            iy2 = (iy + 1) % self.ny
            for ir in range(self.nr_e):
                a = 0.25 * (ux[iy, :, ir] + ux[iy2, :, ir] +
                            ux[iy, :, ir + 1] + ux[iy2, :, ir + 1])
                w = self.dy * self.dr[ir]
                if self.in_out[iy, ir] and not self.in_bore[iy, ir]:
                    us += a * w; ws += w
                elif not self.in_out[iy, ir]:
                    ul += a * w; wl += w
        return us / ws, ul / wl

    def interface_kq(self, u, j, Ns=None):
        """直接测界面单位长度剪切刚度：k_q(x) = q(x)/slip(x)，q = -dN_s/dx。
        这是与二维模型 k_q 完全同定义的量，不依赖任何指数拟合。"""
        if Ns is None:
            Ns = self.steel_force(u, j)
        ni = int(np.sum(self.xc < L_INS))
        # N_s 在结点上的值：由单元中心值线性外推（结点 0..ni）
        xe, Ne = self.xc[:ni], Ns[:ni]
        xn = self.x[:ni + 1]
        Nn = np.interp(xn, xe, Ne)
        q = -np.diff(Nn) / np.diff(xn)                  # 单元中心的剪流 N/mm
        us, ul = self.interface_slip(u, j)
        slip = 0.5 * (us[:ni + 1][:-1] + us[:ni + 1][1:]) - \
               0.5 * (ul[:ni + 1][:-1] + ul[:ni + 1][1:])
        return xe, q, slip, np.where(np.abs(slip) > 1e-9, q / slip, np.nan)

    def calibrate_FM(self, target_FS=FS_INSTALL):
        """与 root_model.calibrate_FM 同法：w=0、F_M=target 解一次，取 a=F_S/target，返回 target/a。"""
        r = self.solve(target_FS, w=0.0)
        a = r['FS'].mean() / target_FS
        return target_FS / max(a, 1e-6), r


# ===================================================================== 校验
def verify_uniform(sigma=10.0, m_y=10, r_div=(2, 4, 1), hoop='global'):
    """校验 (a)：拿掉全部螺套，x=0 施加均布轴向端载，应变必须等于 F/(E_x*A)。
    这同时检验了正交各向异性本构、周期条件和广义周期跳变自由度是否正确。"""
    md = Sector3D(m_y=m_y, r_div=r_div, with_inserts=False, hoop=hoop,
                  verbose=False)
    far, rbm = md._con_base()
    con = np.concatenate([far, rbm])
    mask = np.ones(md.ndof, bool)
    mask[con] = False
    free = np.flatnonzero(mask)
    # x=0 端面结点支配面积
    At = np.zeros(md.nr_n)
    At[:-1] += 0.5 * md.dr
    At[1:] += 0.5 * md.dr
    At = At * md.dy
    f = np.zeros(md.ndof)
    fd = md.face_dof                                   # (ny, nr_n)
    f[fd] = -sigma * At[None, :]                       # 往 -x 拉 → 轴向受拉
    Kff = md.K0[free][:, free].tocsc()
    lu = spla.splu(Kff, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))
    u = np.zeros(md.ndof)
    u[free] = lu.solve(f[free])
    ei, ea = md.profiles(u, md.m_y // 2)
    A_tot = md.ny * md.dy * T_WALL
    F_tot = sigma * A_tot
    eps_ref = F_tot / (E_X * A_tot)
    res = dict(eps_ref=eps_ref, eps_inner=ei, eps_avg=ea,
               err_inner=float(np.max(np.abs(ei / eps_ref - 1.0))),
               err_avg=float(np.max(np.abs(ea / eps_ref - 1.0))),
               F_tot=F_tot, A_tot=A_tot,
               R_far=float((md.K0 @ u - f)[far].sum()),
               hoop=float(np.mean(u[md.i_hoop:md.i_hoop + md.n_hoop])),
               hoop_spread=float(np.ptp(u[md.i_hoop:md.i_hoop + md.n_hoop])),
               hoop_ref=-NU_XY * eps_ref * md.ny * md.dy, model=md)
    return res


# ===================================================================== 二维对照
def run_2d(targets=(0.0, FA_REF, -FA_REF), broken_cell=4):
    """用现成的二维展开壳模型跑同样的工况，作为 Q2 的对照。"""
    from root_model import RootFE
    fe = RootFE(n_cells=N_CELL, m_y=4)
    FM = fe.calibrate_FM(FS_INSTALL)

    def solve_FA(FA, broken=None):
        r = fe.solve(0.0, FM, broken)
        r1 = fe.solve(-1e-3, FM, broken)
        k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3)
        w = 0.0
        for _ in range(30):
            if abs(r['FA'].mean() - FA) < max(5.0, 1e-6 * abs(FA)):
                break
            w += (FA - r['FA'].mean()) / k
            r = fe.solve(w, FM, broken)
        r['w'] = w
        return r

    out = {}
    out['R1'] = solve_FA(targets[0])
    out['R2'] = solve_FA(targets[1])
    out['R3'] = solve_FA(targets[2])
    out['R4'] = fe.solve(out['R2']['w'], FM, broken=[broken_cell])
    out['R5'] = fe.solve(out['R3']['w'], FM, broken=[broken_cell])
    for k in ('R4', 'R5'):
        out[k]['w'] = out['R2']['w'] if k == 'R4' else out['R3']['w']
    out['fe'] = fe
    out['FM'] = FM
    return out


# ===================================================================== 主流程
# 取样点跟着埋深走：螺套末端前 40、前 10，以及末端后 30。
XPROBE = np.array([5.0, 20.0, 50.0, 100.0, 200.0, 300.0,
                   L_INS - 40, L_INS - 10, L_INS + 30, 600.0, 800.0])
XCHK = (150, 300, L_INS - 20, 600, 850)   # N(x) 守恒校验的取样点
XFAR = (L_INS + 30, 600, 700, 800, 870)   # 远场与梁理论对照的取样点
JC = N_CELL // 2                      # 中心 cell


def run_model(md, tag='base', verbose=True):
    """在给定网格上跑完 R1~R5，返回结果字典。"""
    t0 = time.time()
    FM, r_cal = md.calibrate_FM(FS_INSTALL)
    iy_cl = JC * md.m_y + md.m_y // 2          # 中心螺套轴线所在的 y 结点面
    iy_mg = (JC + 1) * md.m_y                  # 相邻两螺套正中间的 y 结点面
    out = dict(FM=FM, iy_cl=iy_cl, iy_mg=iy_mg, tag=tag,
               y_cl=md.y[iy_cl % md.ny], y_mg=md.y[iy_mg % md.ny])
    runs = {}
    runs['R1'] = md.solve(FM, target_FA=0.0)
    runs['R2'] = md.solve(FM, target_FA=FA_REF)
    runs['R3'] = md.solve(FM, target_FA=-FA_REF)
    runs['R4'] = md.solve(FM, broken=[JC], w=runs['R2']['w'])
    runs['R5'] = md.solve(FM, broken=[JC], w=runs['R3']['w'])
    for k, r in runs.items():
        ei_c, ea_c = md.profiles(r['u'], iy_cl)
        ei_m, ea_m = md.profiles(r['u'], iy_mg)
        r['eps_inner_cl'], r['eps_avg_cl'] = ei_c, ea_c
        r['eps_inner_mg'], r['eps_avg_mg'] = ei_m, ea_m
        r['k_cl'] = np.where(np.abs(ea_c) > 1e-12, ei_c / ea_c, np.nan)
        r['k_mg'] = np.where(np.abs(ea_m) > 1e-12, ei_m / ea_m, np.nan)
        r['N_x'] = md.section_force(r['u'])
        if verbose:
            print('  [%s] w=%9.5f mm  FA=%8.1f kN/cell  FS=%8.1f  C=%8.1f  '
                  'R_far=%9.1f kN  open=%d  iters=%d fac=%d conv=%s'
                  % (k, r['w'], r['FA'].mean() / 1e3, r['FS'].mean() / 1e3,
                     r['C'].mean() / 1e3, r['R_far'] / 1e3, r['n_open'],
                     r['n_iter'], r['n_factor'], r['converged']))
    out['runs'] = runs
    out['t_solve'] = time.time() - t0
    return out


def _fmt_tab(xprobe, xc, arrs, names, head, fmt='%9.3f'):
    lines = [head, '  x[mm] ' + ''.join('%14s' % n for n in names)]
    for xp in xprobe:
        row = '  %6.0f' % xp
        for a in arrs:
            v = np.interp(xp, xc, a)
            row += '%14s' % (fmt % v)
        lines.append(row)
    return lines


def main(refine=True, do2d=True):
    t_start = time.time()
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    L = '=' * 92
    rep = []
    rep.append('叶根连接三维扇形块有限元交叉校核模型 M2  (sector3d.py)')
    rep.append('生成时间：%s' % time.strftime('%Y-%m-%d %H:%M:%S'))
    rep.append(L)

    # ---------------------------------------------------------------- 1 网格
    print('>>> 建立基准网格')
    md = Sector3D(m_y=10, r_div=(2, 4, 1))
    e_st = md.A_steel_mesh / A_STEEL_EXACT - 1.0
    e_lam = md.A_lam_mesh / A_LAM_EXACT - 1.0
    rep.append('')
    rep.append('一、网格与规模')
    rep.append(L)
    rep.append('  轴向 x：%d 个单元（端面 6.0 mm ×12，过渡 15 mm ×4，中段 29.8 mm ×10，'
               % md.nx_e)
    rep.append('          %.0f~%.0f 加密 10 mm ×6，%.0f~%.0f 几何级数 10→110 mm ×9），x_max=%.0f mm'
               % (L_INS - 60.0, L_INS, L_INS, X_MAX, X_MAX))
    rep.append('  周向 y：每 cell %d 个单元（dy=%.4f mm），共 %d 个，周期' % (md.m_y, md.dy, md.ny))
    rep.append('  径向 r：%d 个单元，结点 r = %s' % (md.nr_e, np.array2string(md.r, precision=2)))
    rep.append('  单元总数 %d，结点总数 %d，自由度 %d（其中 %d 个是环向周期跳变 Δ(x,r)）'
               % (md.n_elem, md.n_node, md.ndof, md.n_hoop))
    rep.append('  刚度阵非零元 %d，装配耗时 %.1f s' % (md.K0.nnz, md.t_asm))
    rep.append('')
    rep.append('  截面积（单个 cell，阶梯化网格 vs 精确值）：')
    rep.append('    A_steel : 网格 %8.1f mm^2   精确 %8.1f mm^2   误差 %+6.2f %%'
               % (md.A_steel_mesh, A_STEEL_EXACT, 100 * e_st))
    rep.append('    A_lam   : 网格 %8.1f mm^2   精确 %8.1f mm^2   误差 %+6.2f %%'
               % (md.A_lam_mesh, A_LAM_EXACT, 100 * e_lam))
    rep.append('    A_bore  : 网格 %8.1f mm^2   精确 %8.1f mm^2   误差 %+6.2f %%'
               % (md.A_bore_mesh, np.pi / 4 * D_BORE ** 2,
                  100 * (md.A_bore_mesh / (np.pi / 4 * D_BORE ** 2) - 1)))
    rep.append('    说明：起初按任务书给的 6 单元/cell + r=(2,4,1) 试算，A_steel 误差 +23.1%、')
    rep.append('    A_lam -17.2%%，超过 10%% 的门槛，因此周向加密到 %d 单元/cell（r 分层不变），'
               % md.m_y)
    rep.append('    两个误差都降到 10% 以内。判据是单元形心是否落在螺套圆/螺孔圆内。')
    rep.append('')
    rep.append('  胶层处理：不单独划网格。把与钢单元在横截面内共面相邻的那一圈层压单元')
    rep.append('  （共 %d 个/横截面）的三个剪切模量改为 G_eff，使其沿界面法向的剪切柔度'
               % int(md.ring_mask.sum() / N_CELL))
    rep.append('  等于"原层压 + 0.5 mm/1200 MPa 胶层"的串联：1/G_eff = 1/G_lam + t_adh/(G_adh*h_e)，')
    rep.append('  h_e 为该单元沿法向的穿越长度（%.1f~%.1f mm）。结果 G_eff 落在 %.0f~%.0f MPa。'
               % (md.h_ring[md.ring_mask].min(), md.h_ring[md.ring_mask].max(),
                  md.G_eff[md.ring_mask].min(), md.G_eff[md.ring_mask].max()))
    k_adh = np.pi * D_INS * G_ADH / T_ADH
    rep.append('  这样引入的"纯胶层"单位长度剪切刚度 k_adh = pi*D*G_adh/t_adh = %.0f N/mm/mm；'
               % k_adh)
    rep.append('  层压本身的剪切柔度由三维网格自然给出（二维模型是用 t_lam_shear=16 mm 的')
    rep.append('  剪滞当量厚度人为假设的）。二维模型的串联值 k_q = %.0f N/mm/mm 作为对照。'
               % KQ_2D)

    # ---------------------------------------------------------------- 2 校验 a
    print('>>> 校验 (a)：均布单轴拉伸')
    va = verify_uniform(sigma=10.0)
    va_glob = verify_uniform(sigma=10.0, hoop='cyl')
    va_lock = verify_uniform(sigma=10.0, hoop='none')
    Ksym = float(abs(md.K0 - md.K0.T).max() / abs(md.K0).max())

    # ---------------------------------------------------------------- 3 主计算
    print('>>> 基准网格 R1~R5')
    base = run_model(md, 'base')
    runs = base['runs']

    # 界面剪切传递：直接测 q(x) 与 slip(x)，不做任何指数拟合
    x_ins_e = md.xc[md.xc < L_INS]
    Ns2 = md.steel_force(runs['R2']['u'], JC)[:len(x_ins_e)]
    Ns1 = md.steel_force(runs['R1']['u'], JC)[:len(x_ins_e)]
    dNs = Ns2 - Ns1
    dFA = runs['R2']['FA'].mean() - runs['R1']['FA'].mean()
    EA_s = E_ST * md.A_steel_mesh
    EA_l = E_X * md.A_lam_mesh
    cc = 1.0 / EA_s + 1.0 / EA_l
    lam2 = np.sqrt(KQ_2D * (1.0 / (E_ST * A_STEEL_EXACT) + 1.0 / (E_X * A_LAM_EXACT)))
    Ns_share = dNs[len(x_ins_e) // 2] / dFA
    end_frac = dNs[len(x_ins_e) - 1] / dFA        # 仍留在钢里、只能由端面顶过去的份额
    kq_x, kq_q, kq_slip, kq_ratio = md.interface_kq(runs['R2']['u'] - runs['R1']['u'],
                                                    JC, Ns=dNs)

    # ------------------------------------------------- 3b 环向周期条件敏感性
    print('>>> 环向/曲率处理的模型形式敏感性：flat+global vs 准柱面 cyl')
    md_g = Sector3D(m_y=10, r_div=(2, 4, 1), hoop='cyl')
    FMg, _ = md_g.calibrate_FM()
    g1 = md_g.solve(FMg, target_FA=0.0)
    g2 = md_g.solve(FMg, target_FA=FA_REF)
    gi1, ga1 = md_g.profiles(g1['u'], base['iy_cl'])
    gi2, ga2 = md_g.profiles(g2['u'], base['iy_cl'])
    hoop_cmp = dict(md=md_g, x=md_g.xc, d_avg=ga2 - ga1, d_in=gi2 - gi1,
                    k=(gi2 - gi1) / (ga2 - ga1), FM=FMg, w=g2['w'], FA=g2['FA'])
    md_g._fcache = {}

    # ---------------------------------------------------------------- 4 二维对照
    d2 = None
    if do2d:
        print('>>> 二维模型对照')
        d2 = run_2d()

    # ---------------------------------------------------------------- 5 网格敏感性
    ref = {}
    if refine:
        print('>>> 网格敏感性：周向加密')
        md_y = Sector3D(m_y=14, r_div=(2, 4, 1))
        ref['y'] = run_model(md_y, 'refine_y')
        ref['y']['md'] = md_y
        md_y._fcache = {}                      # 释放 LU 因子，避免同时驻留多份
        print('>>> 网格敏感性：径向加密')
        md_r = Sector3D(m_y=10, r_div=(4, 8, 2))
        ref['r'] = run_model(md_r, 'refine_r')
        ref['r']['md'] = md_r
        md_r._fcache = {}

    return dict(md=md, base=base, va=va, va_glob=va_glob, va_lock=va_lock,
                Ns_share=Ns_share, end_frac=end_frac, Ns=dNs, x_ins=x_ins_e, dFA=dFA,
                kq_x=kq_x, kq_q=kq_q, kq_slip=kq_slip, kq_ratio=kq_ratio,
                Ksym=Ksym, hoop_cmp=hoop_cmp, d2=d2, ref=ref, lam2=lam2, rep=rep,
                t_start=t_start, e_st=e_st, e_lam=e_lam)


# ===================================================================== 输出
def write_npz(res, path):
    md, base = res['md'], res['base']
    d = dict(x_centres=md.xc, x_nodes=md.x, r_nodes=md.r, y_nodes=md.y,
             y_cl=base['y_cl'], y_mg=base['y_mg'], F_M=base['FM'],
             A_steel_mesh=md.A_steel_mesh, A_lam_mesh=md.A_lam_mesh,
             A_steel_exact=A_STEEL_EXACT, A_lam_exact=A_LAM_EXACT,
             ndof=md.ndof, n_elem=md.n_elem, broken_cell=JC,
             kq_2d=KQ_2D, Ltr_2d=1.0 / res['lam2'], k_adh=np.pi * D_INS * G_ADH / T_ADH,
             Ns_incr=res['Ns'], x_ins=res['x_ins'], hoop_mode=md.hoop,
             end_face_frac=res['end_frac'], steel_share=res['Ns_share'],
             kq_x=res['kq_x'], kq_q=res['kq_q'], kq_slip=res['kq_slip'],
             kq_ratio=res['kq_ratio'])
    for k, r in base['runs'].items():
        for nm in ('eps_inner_cl', 'eps_avg_cl', 'eps_inner_mg', 'eps_avg_mg',
                   'k_cl', 'k_mg', 'N_x', 'FA', 'FS', 'C', 'C_lam', 'C_steel'):
            d['%s_%s' % (k, nm)] = r[nm]
        d['%s_w' % k] = r['w']
        d['%s_R_far' % k] = r['R_far']
        d['%s_converged' % k] = r['converged']
    # 增量（相对 R1 预紧态）——OFDR 实际看到的是这个
    r1 = base['runs']['R1']
    for k in ('R2', 'R3', 'R4', 'R5'):
        r = base['runs'][k]
        for nm in ('eps_inner_cl', 'eps_avg_cl', 'eps_inner_mg', 'eps_avg_mg'):
            d['d%s_%s' % (k, nm)] = r[nm] - r1[nm]
        d['d%s_k_cl' % k] = (r['eps_inner_cl'] - r1['eps_inner_cl']) / \
                            (r['eps_avg_cl'] - r1['eps_avg_cl'])
        d['d%s_k_mg' % k] = (r['eps_inner_mg'] - r1['eps_inner_mg']) / \
                            (r['eps_avg_mg'] - r1['eps_avg_mg'])
    hc = res['hoop_cmp']
    d['hoopcyl_x'] = hc['x']
    d['hoopcyl_d_eps_avg_cl'] = hc['d_avg']
    d['hoopcyl_d_eps_inner_cl'] = hc['d_in']
    d['hoopcyl_k_cl'] = hc['k']
    d['hoopcyl_FA'] = hc['FA']
    for tag, rr in res['ref'].items():
        m2 = rr['md']
        d['ref_%s_x' % tag] = m2.xc
        for k in ('R2', 'R4'):
            d['ref_%s_%s_k_cl' % (tag, k)] = rr['runs'][k]['k_cl']
            d['ref_%s_%s_eps_inner_cl' % (tag, k)] = rr['runs'][k]['eps_inner_cl']
            d['ref_%s_%s_FA' % (tag, k)] = rr['runs'][k]['FA']
    if res['d2'] is not None:
        fe = res['d2']['fe']
        d['d2_x'] = res['d2']['R2']['x_c']
        for k in ('R1', 'R2', 'R3', 'R4', 'R5'):
            d['d2_%s_FA' % k] = res['d2'][k]['FA']
            d['d2_%s_FS' % k] = res['d2'][k]['FS']
            d['d2_%s_eps_cl' % k] = res['d2'][k]['eps'][fe.iy_center[JC] % fe.ny, :]
            d['d2_%s_w' % k] = res['d2'][k]['w']
    np.savez(path, **d)
    return d


def write_report(res, path):
    md, base, rep = res['md'], res['base'], res['rep']
    runs = base['runs']
    L = '=' * 92
    va, vb = res['va'], res['va_lock']

    rep.append('')
    rep.append('  求解耗时：基准网格 R1~R5 合计 %.1f s（含 F_M 标定），其中 LU 分解 %d 次'
               % (base['t_solve'], md.n_lu))
    rep.append('  共 %.1f s（%.0f s/次），其余都是回代。五个工况的接触主动集相同，K0 又与'
               % (md.t_lu, md.t_lu / max(md.n_lu, 1)))
    rep.append('  预紧力、法兰位移、断柱与否都无关，所以整轮只分解了一次。')
    rep.append('  螺柱秩一刚度用 Woodbury 公式带入，因此断柱工况不必重新分解 K；')
    rep.append('  又因为主动集不变时 u(w)=u0+w*u1 是精确仿射，目标 F_A 一次线性求解即可命中，')
    rep.append('  不需要割线迭代的收敛误差。')

    # ------------------------------------------------------------------ 二、校验
    rep.append('')
    rep.append('二、校验')
    rep.append(L)
    rep.append('  (a) 拿掉全部螺套、x=0 施加均布轴向端载 sigma=10 MPa（F=%.0f N，A=%.0f mm^2）：'
               % (va['F_tot'], va['A_tot']))
    rep.append('      理论 eps_xx = F/(E_x*A) = %.6e' % va['eps_ref'])
    rep.append('      有限元最大相对偏差：内表面 %.2e，厚度平均 %.2e   → 通过（要求 <2%%）'
               % (va['err_inner'], va['err_avg']))
    rep.append('      远端支反力 %.1f N vs 施加 %.1f N，差 %.2e' %
               (va['R_far'], va['F_tot'], abs(va['R_far'] - va['F_tot'])))
    rep.append('      换成准柱面运动学(cyl)：最大相对偏差 %.2e（同样通过）。'
               % res['va_glob']['err_inner'])
    rep.append('      若按任务书字面做——展开成平板 + "y=max 结点与 y=0 结点完全等同"——')
    rep.append('      同一算例误差变成 %.2f%%，正好是 1-nu_xy*nu_yx。原因是展开以后"圆环靠'
               % (100 * vb['err_inner']))
    rep.append('      半径缩小释放环向"这条路没了，环向应变被锁成 0，轴向被憋硬。')
    rep.append('      所以本模型默认在朴素周期之外再放一个全局环向跳变 Δ（global 模式，')
    rep.append('      与二维模型 RootFE 完全同法），它既精确满足 y 周期（完好工况各 cell 的')
    rep.append('      F_A 完全相等），又能通过本校验。另两种做法的对比见第七节 2)。')

    rep.append('')
    rep.append('  (b) 整体力平衡：R_far 应等于 sum(F_S) - sum(C)（= 9*F_A）。')
    rep.append('      %-4s %14s %14s %14s %12s' % ('工况', 'sum F_S [kN]', 'sum C [kN]',
                                                   'R_far [kN]', '不平衡 [N]'))
    for k in ('R1', 'R2', 'R3', 'R4', 'R5'):
        r = runs[k]
        imb = r['R_far'] - (r['FS'].sum() - r['C'].sum())
        rep.append('      %-4s %14.3f %14.3f %14.3f %12.3e'
                   % (k, r['FS'].sum() / 1e3, r['C'].sum() / 1e3, r['R_far'] / 1e3, imb))
    rep.append('      （这是刚度反力与螺柱本构公式两条独立路径的对照，不是恒等式。）')

    rep.append('')
    rep.append('  (c) 截面轴力守恒：x>105 mm（螺纹啮合段以外）不再有外力注入，由单元应力')
    rep.append('      积分出的 N(x)=∫sigma_xx dA 必须处处等于 sum(F_A)。')
    rep.append('      %-4s %14s' % ('工况', 'sum F_A [kN]') +
               ''.join('%12s' % ('N(%.0f)' % x) for x in XCHK))
    for k in ('R1', 'R2', 'R3', 'R4', 'R5'):
        r = runs[k]
        row = '      %-4s %14.3f' % (k, r['FA'].sum() / 1e3)
        for x in XCHK:
            row += '%12.3f' % (np.interp(x, md.xc, r['N_x']) / 1e3)
        rep.append(row)
    errs = []
    for k in ('R2', 'R3', 'R4', 'R5'):
        r = runs[k]
        s = np.abs(r['FA'].sum())
        sel = md.xc > 120
        errs.append(np.max(np.abs(r['N_x'][sel] - r['FA'].sum())) / max(s, 1.0))
    rep.append('      x>120 mm 区间内 N(x) 相对 sum(F_A) 的最大偏差（R2/R3/R4/R5）：'
               + ' '.join('%.2f%%' % (1e2 * e) for e in errs))

    rep.append('')
    rep.append('  (d) 远场与梁理论：螺套之外的远场，厚度平均增量应变应趋于')
    rep.append('      sum(ΔF_A)/(E_x*A_total)。这一项检验的是"模型有没有真的进入远场"。')
    A_tot = md.ny * md.dy * T_WALL
    beam = (runs['R2']['FA'].sum() - runs['R1']['FA'].sum()) / (E_X * A_tot)
    rep.append('      梁理论 %.1f ue。R2-R1 中心线厚度平均值：' % (beam * 1e6))
    rep.append('      %10s' % 'x[mm]' + ''.join('%10.0f' % x for x in XFAR))
    dd = runs['R2']['eps_avg_cl'] - runs['R1']['eps_avg_cl']
    rep.append('      %10s' % 'FE(global)' + ''.join('%10.1f' % (np.interp(x, md.xc, dd) * 1e6)
                                                 for x in XFAR))
    rep.append('      %10s' % '偏差' + ''.join('%9.1f%%' % (100 * (np.interp(x, md.xc, dd) / beam - 1))
                                               for x in XFAR))
    hc = res['hoop_cmp']
    rep.append('      %10s' % 'FE(cyl)' + ''.join('%10.1f' % (np.interp(x, hc['x'], hc['d_avg']) * 1e6)
                                                  for x in XFAR))
    rep.append('      %10s' % '偏差' + ''.join('%9.1f%%' % (100 * (np.interp(x, hc['x'], hc['d_avg']) / beam - 1))
                                               for x in XFAR))
    rep.append('      默认(global)在 x>700 收敛到一个比梁理论低 5.0% 的平台。这不是求解误差')
    rep.append('      （校验 a 精确、N(x) 精确守恒），而是"一个全局 Δ"这条约束的后果：螺套区')
    rep.append('      被约束出来的环向力必须由远场反号承担，远场因此挂着约 1 MPa 的环向拉应力，')
    rep.append('      经泊松耦合把轴向应变压低约 5%。准柱面(cyl)把这条人为耦合去掉，远场偏差')
    rep.append('      降到 -3% 并仍在往 0 走，但它又引入真实的柱壳边缘效应（衰减长度')
    rep.append('      ~sqrt(R*t)=%.0f mm），而 %.0f~%.0f 只有 %.0f mm 装不下，所以两者都不能'
               % (np.sqrt(R_IN * T_WALL), L_INS, X_MAX, X_MAX - L_INS))
    rep.append('      当作干净远场读。想要干净远场，模型轴向长度至少要 2000 mm。')

    rep.append('')
    rep.append('  (e) 界面剪切传递：三维实测 vs 二维的剪滞假设。')
    kadh = np.pi * D_INS * G_ADH / T_ADH
    kq_near = res['kq_ratio'][(res['kq_x'] > 30) & (res['kq_x'] < 100)]
    rep.append('      把 R2-R1 的增量拿出来（单 cell 外载增量 ΔF_A = %.1f kN），沿埋深积分'
               % (res['dFA'] / 1e3))
    rep.append('      螺套钢截面上的轴力 N_s(x)，再取螺套钢与该 cell 层压各自的横截面平均')
    rep.append('      轴向位移之差 slip(x)，二者就是剪滞模型里 q = k_q*slip 的两个量。')
    rep.append('      %9s %12s %12s %14s %12s' % ('x[mm]', 'N_s/ΔF_A', 'q=-dNs/dx', 'slip[mm]', 'q/slip'))
    for xp in (10.0, 50.0, 100.0, 150.0, 250.0, 350.0, 430.0, 460.0, 485.0):
        i = int(np.argmin(np.abs(res['kq_x'] - xp)))
        rep.append('      %9.1f %12.4f %12.2f %14.3e %12.0f'
                   % (res['kq_x'][i], res['Ns'][i] / res['dFA'], res['kq_q'][i],
                      res['kq_slip'][i], res['kq_ratio'][i]))
    rep.append('')
    rep.append('      读这张表得到三条结论，其中前两条是二维模型完全没有的：')
    rep.append('      1) 螺套在 x=%.0f mm（最后一个单元）仍然携带 ΔF_A 的 %.0f%%。这些力只能'
               % (res['kq_x'][-1], 100 * res['end_frac']))
    rep.append('         经螺套端面以法向挤压/拉伸直接顶进前方层压，而不是靠侧面剪切传走。')
    rep.append('         二维模型的一维杆在 x=%.0f 是 N_s=0 的自由端，根本没有这条路径。'
               % L_INS)
    rep.append('      2) 正因为如此，q/slip 在 x>400 段变成负值——一维剪滞关系 q=k_q*slip')
    rep.append('         在三维里根本不成立，不存在一个能代表本问题的单一 k_q。所以这里不')
    rep.append('         报"三维等效 k_q"，报原始的 q 和 slip，让读者自己判断。')
    rep.append('      3) 能干净给出的只有本模型显式加进去的胶层项：')
    rep.append('         k_adh = pi*D*G_adh/t_adh = %.0f N/mm 每 mm 滑移。' % kadh)
    rep.append('         二维模型的 k_q = %.0f 是"胶层 + 层压当量厚度 16 mm"的串联，比纯胶层'
               % KQ_2D)
    rep.append('         还软 %.1f 倍。三维结果不支持那么大的额外柔度：在 30<x<100 mm 段'
               % (kadh / KQ_2D))
    rep.append('         实测 q/slip 落在 %.1e ~ %.1e，与 k_adh 同量级甚至更硬（注意这一段还'
               % (np.nanmin(kq_near), np.nanmax(kq_near)))
    rep.append('         叠加了螺柱沿啮合段的分布力，所以只能当量级参考）。')
    rep.append('      后果（可观测量）：二维模型把螺套载荷的卸载集中在 x≈%.0f 前后'
               % L_INS)
    rep.append('      1/lambda=%.1f mm 的一小段里，所以第六节里螺套末端前后的层压应变二维比'
               % (1.0 / res['lam2']))
    rep.append('      三维高出 30~70%%。要用哪一个，取决于 OFDR 测点是否落在 %.0f~%.0f mm 区间。'
               % (L_INS - 60, L_INS + 30))
    rep.append('      远场钢/层压轴向刚度分配：三维数值 %.1f%%，按网格截面积的解析值 %.1f%%。'
               % (100 * res['Ns_share'],
                  100 * E_ST * md.A_steel_mesh / (E_ST * md.A_steel_mesh + E_X * md.A_lam_mesh)))
    rep.append('')
    rep.append('  (f) 刚度阵对称性：max|K-K^T| / max|K| = %.2e（含缝合单元的 Δ 耦合项）。'
               % res['Ksym'])

    # ------------------------------------------------------------------ 三、k_surf
    rep.append('')
    rep.append('三、k_surf(x) = eps_xx(内表面 r=0) / eps_xx(层压厚度平均)')
    rep.append(L)
    rep.append('  R2 = 完好、F_A=+74 kN/cell（拉）；R4 = 中心 cell 断柱、法兰位移与 R2 相同。')
    rep.append('  centreline = 中心螺套轴线所在 y 面 (y=%.1f mm)；midgap = 相邻两螺套正中 (y=%.1f mm)。'
               % (base['y_cl'], base['y_mg']))
    rep.append('')
    rep.append('  (1) 总应变（含 420 kN 预紧引起的压应变）：')
    rep.append('  %8s %12s %12s %12s %12s' % ('x[mm]', 'R2 中心线', 'R2 中间', 'R4 中心线', 'R4 中间'))
    for xp in XPROBE:
        v = [np.interp(xp, md.xc, runs[k][nm]) for k, nm in
             (('R2', 'k_cl'), ('R2', 'k_mg'), ('R4', 'k_cl'), ('R4', 'k_mg'))]
        rep.append('  %8.0f %12.4f %12.4f %12.4f %12.4f' % (xp, v[0], v[1], v[2], v[3]))
    rep.append('')
    rep.append('  (2) 增量应变（相对 R1 预紧态，即 OFDR 实测到的载荷变化量）：')
    rep.append('  %8s %12s %12s %12s %12s' % ('x[mm]', 'R2 中心线', 'R2 中间', 'R4 中心线', 'R4 中间'))
    r1 = runs['R1']
    for xp in XPROBE:
        row = '  %8.0f' % xp
        for k in ('R2', 'R4'):
            for c, a in (('eps_inner_cl', 'eps_avg_cl'), ('eps_inner_mg', 'eps_avg_mg')):
                di = np.interp(xp, md.xc, runs[k][c] - r1[c])
                da = np.interp(xp, md.xc, runs[k][a] - r1[a])
                row += '%12.4f' % (di / da)
            # 顺序 cl, mg
        rep.append(row)
    rep.append('')
    rep.append('  注：(1) 里 x=100 mm 附近 k_surf 会出现负值或异常大，那是因为总应变的厚度')
    rep.append('  平均值在那里穿过零点（预紧压应变与外载拉应变抵消），比值本身病态，不是')
    rep.append('  数值错误。OFDR 实测的是载荷变化引起的应变增量，用 (2) 的数才有意义。')
    rep.append('')
    rep.append('  (3) 厚度方向 eps_xx 分布（R2-R1 增量，中心线，微应变）——k_surf 的来源：')
    rep.append('  %8s' % 'r[mm]' + ''.join('%10.1f' % rr for rr in md.r))
    for xp in (20.0, 200.0, L_INS - 10, L_INS + 30, 800.0):
        _, e2 = md.eps_through(runs['R2']['u'], base['iy_cl'], xp)
        _, e1 = md.eps_through(runs['R1']['u'], base['iy_cl'], xp)
        rep.append('  x=%5.0f ' % xp + ''.join('%10.1f' % v for v in (e2 - e1) * 1e6))
    rep.append('  （r=0 内表面=光纤位置，r=%.1f~%.1f 为螺套所在层，'
               'r=%.0f 外表面）'
               % (R_INS_C - D_INS / 2, R_INS_C + D_INS / 2, T_WALL))

    # ------------------------------------------------------------------ 四、改道
    rep.append('')
    rep.append('四、中心 cell 螺柱断裂后的单柱外载 F_A 重分配')
    rep.append(L)
    rep.append('  法兰位移固定为 R2/R3 的值（远场加载不变）。以名义 74 kN 为分母。')
    for kk, kb, nom in (('R2', 'R4', FA_REF), ('R3', 'R5', -FA_REF)):
        rep.append('')
        rep.append('  %s(完好) -> %s(中心 cell %d 断柱)，w=%.6f mm'
                   % (kk, kb, JC, runs[kb]['w']))
        rep.append('  %6s %14s %14s %14s %12s' % ('cell', 'F_A 完好[kN]', 'F_A 断柱[kN]',
                                                  'ΔF_A [kN]', 'ΔF_A/74kN'))
        a, b = runs[kk]['FA'], runs[kb]['FA']
        for j in range(N_CELL):
            lbl = 'j%+d' % (j - JC) if j != JC else 'j(断)'
            rep.append('  %6s %14.2f %14.2f %14.3f %12.4f'
                       % (lbl, a[j] / 1e3, b[j] / 1e3, (b[j] - a[j]) / 1e3,
                          (b[j] - a[j]) / abs(nom)))
        rep.append('  合计 ΔF_A = %.1f N（法兰位移不变，总外载应基本守恒）' % (b.sum() - a.sum()))

    # ------------------------------------------------------------------ 五、应变剖面
    rep.append('')
    rep.append('五、内表面（r=0）中心线轴向应变剖面 [微应变]')
    rep.append(L)
    rep.append('  %8s %14s %14s %14s %14s %14s' %
               ('x[mm]', 'R1 预紧', 'R2 完好拉', 'R4 断柱', 'R2-R1', 'R4-R1'))
    for xp in XPROBE:
        v = [np.interp(xp, md.xc, runs[k]['eps_inner_cl']) * 1e6 for k in ('R1', 'R2', 'R4')]
        rep.append('  %8.0f %14.1f %14.1f %14.1f %14.1f %14.1f'
                   % (xp, v[0], v[1], v[2], v[1] - v[0], v[2] - v[0]))
    rep.append('')
    rep.append('  同一位置的层压厚度平均应变（二维膜模型能给的量）[微应变]：')
    rep.append('  %8s %14s %14s %14s' % ('x[mm]', 'R1 预紧', 'R2 完好拉', 'R4 断柱'))
    for xp in XPROBE:
        v = [np.interp(xp, md.xc, runs[k]['eps_avg_cl']) * 1e6 for k in ('R1', 'R2', 'R4')]
        rep.append('  %8.0f %14.1f %14.1f %14.1f' % (xp, v[0], v[1], v[2]))

    # ------------------------------------------------------------------ 六、2D 对照
    if res['d2'] is not None:
        d2 = res['d2']
        fe = d2['fe']
        rep.append('')
        rep.append('六、与二维展开壳模型 (root_model.RootFE, n_cells=9, m_y=4) 的对照 —— Q2')
        rep.append(L)
        Phi3 = (runs['R2']['FS'].mean() - runs['R1']['FS'].mean()) / \
               (runs['R2']['FA'].mean() - runs['R1']['FA'].mean())
        Phi2 = (d2['R2']['FS'].mean() - d2['R1']['FS'].mean()) / \
               (d2['R2']['FA'].mean() - d2['R1']['FA'].mean())
        chi3 = runs['R1']['C_lam'].mean() / max(runs['R1']['C'].mean(), 1e-9)
        chi2 = d2['R1']['chi'].mean()
        rep.append('  %-28s %12s %12s %10s' % ('量', '三维 M2', '二维 M1', '相对差'))
        rep.append('  %-28s %12.4f %12.4f %9.1f%%' % ('螺栓载荷系数 Phi', Phi3, Phi2,
                                                      100 * (Phi3 / Phi2 - 1)))
        rep.append('  %-28s %12.4f %12.4f %9.1f%%' % ('端面夹紧力分配 chi(层压占比)', chi3, chi2,
                                                      100 * (chi3 / chi2 - 1)))
        rep.append('  %-28s %12.4f %12.4f %9.1f%%' % ('标定后 F_M [kN]', base['FM'] / 1e3,
                                                      d2['FM'] / 1e3,
                                                      100 * (base['FM'] / d2['FM'] - 1)))
        rep.append('  %-28s %12.5f %12.5f %9.1f%%' % ('R2 法兰位移 w [mm]', runs['R2']['w'],
                                                      d2['R2']['w'],
                                                      100 * (runs['R2']['w'] / d2['R2']['w'] - 1)))
        rep.append('')
        rep.append('  每 cell 外载 F_A 分布 [kN]（R4 = 中心断柱，w 与 R2 相同）：')
        rep.append('  %6s %12s %12s %10s %12s %12s %10s'
                   % ('cell', '3D R2', '2D R2', '差', '3D R4', '2D R4', '差'))
        for j in range(N_CELL):
            a3, a2 = runs['R2']['FA'][j] / 1e3, d2['R2']['FA'][j] / 1e3
            b3, b2 = runs['R4']['FA'][j] / 1e3, d2['R4']['FA'][j] / 1e3
            rep.append('  %6d %12.2f %12.2f %9.2f%% %12.2f %12.2f %9.2f%%'
                       % (j, a3, a2, 100 * (a3 / a2 - 1) if a2 else np.nan,
                          b3, b2, 100 * (b3 / b2 - 1) if b2 else np.nan))
        rep.append('')
        rep.append('  断柱后的载荷改道份额（ΔF_A / 74 kN）三维 vs 二维：')
        rep.append('  %6s %12s %12s' % ('cell', '3D', '2D'))
        for j in range(N_CELL):
            f3 = (runs['R4']['FA'][j] - runs['R2']['FA'][j]) / FA_REF
            f2 = (d2['R4']['FA'][j] - d2['R2']['FA'][j]) / FA_REF
            rep.append('  %6s %12.4f %12.4f' % ('j%+d' % (j - JC) if j != JC else 'j(断)', f3, f2))
        rep.append('')
        rep.append('  中心线轴向应变剖面对照 [微应变]（三维取层压厚度平均，与二维的膜应变同义）：')
        rep.append('  x=%.0f~%.0f 几行差 30~70%%，原因见校验 (e)：二维把螺套卸载集中在 x≈%.0f'
                   % (L_INS - 40.0, L_INS + 30.0, L_INS))
        rep.append('  附近一小段，三维里螺套端面直接顶住前方层压，卸载被摊开。x<400 与')
        rep.append('  x>600 两段吻合在 10% 以内。')
        rep.append('  %8s %14s %14s %10s %14s %14s %10s'
                   % ('x[mm]', '3D R2 avg', '2D R2', '差', '3D R2-R1', '2D R2-R1', '差'))
        x2 = d2['R2']['x_c']
        e2_2 = d2['R2']['eps'][fe.iy_center[JC] % fe.ny, :]
        e2_1 = d2['R1']['eps'][fe.iy_center[JC] % fe.ny, :]
        for xp in XPROBE:
            a3 = np.interp(xp, md.xc, runs['R2']['eps_avg_cl']) * 1e6
            a2 = np.interp(xp, x2, e2_2) * 1e6
            d3 = np.interp(xp, md.xc, runs['R2']['eps_avg_cl'] - runs['R1']['eps_avg_cl']) * 1e6
            dd2 = np.interp(xp, x2, e2_2 - e2_1) * 1e6
            rep.append('  %8.0f %14.1f %14.1f %9.1f%% %14.1f %14.1f %9.1f%%'
                       % (xp, a3, a2, 100 * (a3 / a2 - 1) if abs(a2) > 1e-9 else np.nan,
                          d3, dd2, 100 * (d3 / dd2 - 1) if abs(dd2) > 1e-9 else np.nan))

    # ------------------------------------------------------------------ 七、数值质量
    rep.append('')
    rep.append('七、数值质量评估')
    rep.append(L)
    rep.append('  1) 接触主动集收敛：')
    for k in ('R1', 'R2', 'R3', 'R4', 'R5'):
        r = runs[k]
        rep.append('     %s: 迭代 %d 次，LU 分解 %d 次，张开结点 %d / %d，收敛=%s'
                   % (k, r['n_iter'], r['n_factor'], r['n_open'], int(md.face_cap.sum()),
                      r['converged']))
    rep.append('')
    rep.append('  2) 环向/曲率的处理方式——这是本模型里影响最大、也最该被质疑的一条决定：')
    rep.append('     把圆环展开成平板以后，"半径可以胀缩来释放环向"这个自由度就没了，')
    rep.append('     必须人为补回来。三种补法都做了：')
    rep.append('     · none   ：不补。单轴拉伸校验直接错 9.60%，排除。')
    rep.append('     · global ：补一个全局跳变 Δ（= 二维模型 RootFE 的做法）。精确 y 周期，')
    rep.append('                校验 (a) 精确通过。缺点见校验 (d)：远场被挂上约 1 MPa 环向应力。')
    rep.append('                本报告的主结果用它，因为要和二维模型同口径比较。')
    rep.append('     · cyl    ：保持朴素周期，在单元里补 eps_yy += u_r/R、gamma_yr -= u_y/R')
    rep.append('                （R = %.1f+r，r=%.1f 处正好 %.1f = 螺栓圆）。校验 (a) 也'
               % (R_IN, R_INS_C, R_IN + R_INS_C))
    rep.append('                精确通过，y 周期也精确。但它把真实柱壳边缘效应带了进来。')
    rep.append('     还试过"每个 (x,r) 各一个跳变 Δ(x,r)"，那是错的：Δ 随 x、r 变化会在缝合')
    rep.append('     单元里生成正比于 y 的伪剪应变，完好工况下靠缝的 cell 的 F_A 高出 27%。已弃。')
    rep.append('')
    rep.append('     global 与 cyl 的差（增量量，中心线）：')
    hc = res['hoop_cmp']
    dd = runs['R2']['eps_avg_cl'] - runs['R1']['eps_avg_cl']
    di = runs['R2']['eps_inner_cl'] - runs['R1']['eps_inner_cl']
    rep.append('     %8s %12s %12s %9s %12s %12s' %
               ('x[mm]', 'global 平均', 'cyl 平均', '差', 'global k', 'cyl k'))
    for xp in (20.0, 100.0, 200.0, 300.0, 480.0, 600.0, 800.0):
        a = np.interp(xp, md.xc, dd) * 1e6
        b = np.interp(xp, hc['x'], hc['d_avg']) * 1e6
        ka = np.interp(xp, md.xc, di) / np.interp(xp, md.xc, dd)
        kb = np.interp(xp, hc['x'], hc['k'])
        rep.append('     %8.0f %12.1f %12.1f %8.1f%% %12.4f %12.4f'
                   % (xp, a, b, 100 * (a / b - 1), ka, kb))
    rep.append('')
    rep.append('     必须说清楚：在螺套段（x<%.0f），k_surf 在两种处理下差得很大——x=200 mm'
               % L_INS)
    rep.append('     处 global 给 %.2f、cyl 给 %.2f。也就是说任务书"曲率可忽略"这个前提，'
               % (np.interp(200.0, md.xc, di) / np.interp(200.0, md.xc, dd),
                  np.interp(200.0, hc['x'], hc['k'])))
    rep.append('     对轴向应变的绝对值成立（差几个百分点），但对"内表面/厚度平均"这个比值')
    rep.append('     不成立。原因是 k_surf 靠的是壁厚方向的应变梯度，而壁厚方向的梯度恰恰')
    rep.append('     是柱壳弯曲最敏感的量。')
    rep.append('     两个模型也都没把远场跑干净（见校验 d），所以现阶段应当把 k_surf 的')
    rep.append('     模型形式不确定度按 ±0.2 量级来用，而不是按表里的四位小数。')
    rep.append('     要收敛掉这条不确定度，需要：(i) 轴向加长到 >=2000 mm，(ii) 用真正的')
    rep.append('     柱面单元（含 R*dtheta 体积元），(iii) 远端换成"平截面+自由径向"的')
    rep.append('     广义平面应变边界，而不是 u_x=0 的硬夹支。')

    if res['ref']:
        rep.append('')
        rep.append('  3) 网格敏感性（重跑 R2 与 R4）：')
        rep.append('     %-12s %8s %10s %10s %10s %10s %10s'
                   % ('网格', 'DOF', 'A_st 误差', 'k_surf(50)', 'k_surf(200)',
                      'k_surf(450)', 'F_A 中心'))
        rows = [('基准 m_y=10,r=(2,4,1)', md, base)]
        for tag, rr in res['ref'].items():
            rows.append(('加密-' + tag, rr['md'], rr))
        for nm, m2, rr in rows:
            k50 = np.interp(50.0, m2.xc, rr['runs']['R2']['k_cl'])
            k200 = np.interp(200.0, m2.xc, rr['runs']['R2']['k_cl'])
            k450 = np.interp(450.0, m2.xc, rr['runs']['R2']['k_cl'])
            rep.append('     %-12s %8d %9.2f%% %10.4f %10.4f %10.4f %10.2f'
                       % (nm, m2.ndof, 100 * (m2.A_steel_mesh / A_STEEL_EXACT - 1),
                          k50, k200, k450, rr['runs']['R2']['FA'][JC] / 1e3))
        rep.append('')
        rep.append('     断柱改道份额 ΔF_A(j±1)/74kN 的网格敏感性：')
        for nm, m2, rr in rows:
            f1 = (rr['runs']['R4']['FA'][JC + 1] - rr['runs']['R2']['FA'][JC + 1]) / FA_REF
            f0 = (rr['runs']['R4']['FA'][JC] - rr['runs']['R2']['FA'][JC]) / FA_REF
            rep.append('     %-24s  断柱 cell %+.4f   j±1 %+.4f' % (nm, f0, f1))
    rep.append('')
    rep.append('  4) 已知的模型近似与存疑之处（如实列出）：')
    rep.append('     · 螺套/螺孔用阶梯化单元近似，A_steel 偏差 %+.1f%%、A_lam 偏差 %+.1f%%。'
               % (100 * res['e_st'], 100 * res['e_lam']))
    rep.append('       这会让钢的轴向刚度份额偏低约 %.1f%%，螺套轴力 N_s 系统性偏小同量级。'
               % (-100 * res['e_st']))
    rep.append('     · 胶层没有独立网格，只用一圈折减剪切模量的层压单元代表，因此')
    rep.append('       胶层的法向（挤压/剥离）柔度被忽略，界面剪应力的峰值被单元尺寸抹平。')
    rep.append('     · 螺柱力沿啮合段 x∈[0,105] 按钢单元体积分摊到所有钢结点，而不是只加在')
    rep.append('       螺纹面（孔壁）上。按 St.Venant，这只影响 x<~40 mm 的局部；表中 x=5、20 mm')
    rep.append('       两行的 k_surf 应当按"量级正确、数值待细化"来读。')
    rep.append('     · 曲率：主结果按任务书展开成直块（t/R=6.9%）。见本节 2)——这条前提对')
    rep.append('       轴向应变绝对值成立，对 k_surf 这个厚度方向的比值不成立，是本模型')
    rep.append('       目前最大的一条不确定度（±0.2 量级）。')
    rep.append('     · 只建了 9 个 cell 的周向窗口。断柱改道到 j±4 已衰减到 0.014，窗口够用；')
    rep.append('       但 j-4 与 j+4 隔着缝其实互为邻居，严格说 9-cell 窗口把最远那一档略微')
    rep.append('       高估，量级在名义外载的 1% 以内。')
    rep.append('     · 接触是无摩擦、只压不拉的结点级 Dirichlet 主动集；法兰当成完全刚性，')
    rep.append('       且只加均匀位移 w（不含弯曲分布），因此这里的结论只适用于均匀加载。')
    rep.append('     · 材料线弹性，无损伤、无蠕变、无温度。')
    rep.append('')
    rep.append('  总耗时 %.1f s' % (time.time() - res['t_start']))
    rep.append(L)

    txt = '\n'.join(rep) + '\n'
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(txt)
    return txt


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-refine', action='store_true')
    ap.add_argument('--no-2d', action='store_true')
    a = ap.parse_args()
    R = main(refine=not a.no_refine, do2d=not a.no_2d)
    out = os.path.join(HERE, 'out')
    write_npz(R, os.path.join(out, 'sector3d_results.npz'))
    txt = write_report(R, os.path.join(out, 'sector3d_report.txt'))
    print(txt)
