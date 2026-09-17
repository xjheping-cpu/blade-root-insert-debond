# -*- coding: utf-8 -*-
"""
M1′／M2′：单个预埋螺套的轴对称有限元模型，界面采用双线性内聚力本构。

与前一代模型（root_model.py）的根本区别：
  旧：界面为线弹性分布剪切弹簧，脱粘长度由人工设定；
  新：界面为内聚力单元，含损伤起始与软化，脱粘长度由模型解出。
因此本模型可以回答"脱粘扩展是否稳定"——这是建模重规划列出的首要待定问题。

几何（径向自内向外）：
  r < r_bore                螺纹孔（空）
  r_bore ~ r_steel          钢衬套 42CrMoA
  r_steel ~ r_wrap          玻纤束缠绕层
  r_wrap  ~ R_out           叶根层压
内聚力界面置于 r = r_wrap（玻纤束缠绕层与层压之间），依据 He 等 2025 实测的失效面位置。

加载：子部件拉拔构型。螺柱力沿螺纹啮合段施加于钢衬套；
      反力由层压端面外环承受（对应真实结构中的法兰承压）。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


# ----------------------------------------------------------------- 参数
@dataclass(frozen=True)
class Geo:
    r_bore: float = _G.d_bore / 2          # 螺纹小径/2
    r_steel: float = _G.D_ins / 2          # 螺套外半径
    r_wrap: float = _G.D_ins / 2 + 6.0     # 玻纤束缠绕层外半径【推断】
    R_out: float = 100.0       # 模型外半径（子部件试件）
    L_ins: float = _G.L_ins    # 螺套埋深
    l_eng: float = _G.l_eng    # 螺纹啮合长度
    L_tot: float = 700.0       # 模型轴向长度
    r_react: float = 62.0      # 端面反力环内半径


@dataclass(frozen=True)
class Mat:
    # 层压：横观各向同性，轴向为 z
    E_a: float = 30.0e3
    E_t: float = 15.0e3
    nu_a: float = 0.30
    nu_t: float = 0.40
    G_a: float = 5.0e3
    # 玻纤束缠绕层（环向/螺旋缠绕，轴向偏软、环向偏硬）
    Ew_a: float = 22.0e3
    Ew_t: float = 20.0e3
    Gw_a: float = 5.5e3
    # 钢
    E_s: float = 210.0e3
    nu_s: float = 0.30


@dataclass(frozen=True)
class Coh:
    """双线性牵引-分离。参数为文献量级值，须经 DCB/ENF 试验标定。"""
    K: float = 1.0e5           # 罚刚度 N/mm^3
    tn0: float = 12.0          # 法向强度 MPa
    ts0: float = 20.0          # 切向强度 MPa
    GIc: float = 0.6           # I 型断裂能 N/mm
    GIIc: float = 1.8          # II 型断裂能 N/mm
    eta: float = 1.45          # B-K 指数


# ----------------------------------------------------------------- 网格
def graded(a, b, n, dense_a=False, dense_b=False, ratio=4.0):
    """单调网格，可在一端或两端加密。"""
    t = np.linspace(0, 1, n + 1)
    if dense_a and dense_b:
        t = 0.5 * (1 - np.cos(np.pi * t))
    elif dense_a:
        t = t ** ratio
    elif dense_b:
        t = 1 - (1 - t) ** ratio
    return a + (b - a) * t


def build_grid(g: Geo):
    r = np.unique(np.concatenate([
        graded(0.0, g.r_bore, 3),
        graded(g.r_bore, g.r_steel, 7),
        graded(g.r_steel, g.r_wrap, 4),
        graded(g.r_wrap, 60.0, 8, dense_a=True, ratio=2.0),
        graded(60.0, g.R_out, 7),
    ]))
    x = np.unique(np.concatenate([
        graded(0.0, 60.0, 16, dense_a=True, ratio=1.6),
        graded(60.0, 420.0, 24),
        graded(420.0, g.L_ins, 10, dense_b=True, ratio=2.0),
        graded(g.L_ins, 560.0, 10, dense_a=True, ratio=2.0),
        graded(560.0, g.L_tot, 8),
    ]))
    return r, x


# ----------------------------------------------------------------- 材料矩阵
def D_axisym(Ea, Et, nua, nut, Ga):
    """轴对称正交各向异性 D 矩阵，应变序 (εr, εz, εθ, γrz)。"""
    S = np.zeros((4, 4))
    S[0, 0] = 1.0 / Et
    S[1, 1] = 1.0 / Ea
    S[2, 2] = 1.0 / Et
    S[0, 1] = S[1, 0] = -nua / Ea
    S[1, 2] = S[2, 1] = -nua / Ea
    S[0, 2] = S[2, 0] = -nut / Et
    S[3, 3] = 1.0 / Ga
    return np.linalg.inv(S)


def D_iso(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((4, 4))
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[3, 3] = mu
    return D


# ----------------------------------------------------------------- 模型
class AxiCZM:
    def __init__(self, geo=Geo(), mat=Mat(), coh=Coh(), x_coh_end=None):
        self.g, self.m, self.c = geo, mat, coh
        self.r, self.x = build_grid(geo)
        self.nr, self.nx = len(self.r), len(self.x)
        self.x_coh_end = geo.L_ins + 30.0 if x_coh_end is None else x_coh_end

        # 内聚力界面所在的径向节点行
        self.i_coh = int(np.argmin(np.abs(self.r - geo.r_wrap)))
        assert abs(self.r[self.i_coh] - geo.r_wrap) < 1e-6

        self._build_nodes()
        self._build_elements()
        self._assemble_bulk()
        self._build_cohesive()

    # -------------------------------------------------- 节点（界面处复制）
    def _build_nodes(self):
        nr, nx = self.nr, self.nx
        self.nid = np.arange(nr * nx).reshape(nr, nx)          # 主节点
        self.n_main = nr * nx
        # 界面上、且 x <= x_coh_end 的节点需要复制一份给外侧
        self.coh_mask = self.x <= self.x_coh_end + 1e-9
        self.n_coh = int(self.coh_mask.sum())
        self.dup = -np.ones(nx, int)
        self.dup[self.coh_mask] = self.n_main + np.arange(self.n_coh)
        self.n_node = self.n_main + self.n_coh
        self.ndof = 2 * self.n_node
        # 节点坐标
        R, X = np.meshgrid(self.r, self.x, indexing='ij')
        self.R = np.concatenate([R.ravel(), np.full(self.n_coh, self.g.r_wrap)])
        self.X = np.concatenate([X.ravel(), self.x[self.coh_mask]])

    def node_of(self, ir, ix, outer=False):
        """取节点号。outer=True 时，若位于复制界面上则返回复制节点。"""
        if outer and ir == self.i_coh and self.dup[ix] >= 0:
            return self.dup[ix]
        return self.nid[ir, ix]

    # -------------------------------------------------- 单元与材料
    def _build_elements(self):
        g, m = self.g, self.m
        D_lam = D_axisym(m.E_a, m.E_t, m.nu_a, m.nu_t, m.G_a)
        D_wrap = D_axisym(m.Ew_a, m.Ew_t, m.nu_a, m.nu_t, m.Gw_a)
        D_st = D_iso(m.E_s, m.nu_s)
        D_void = D_iso(1.0, 0.3)
        self.elems, self.Dmap, self.etag = [], [], []
        rc = 0.5 * (self.r[:-1] + self.r[1:])
        xc = 0.5 * (self.x[:-1] + self.x[1:])
        for i in range(self.nr - 1):
            for k in range(self.nx - 1):
                rr, xx = rc[i], xc[k]
                if rr < g.r_bore:
                    D, tag = D_void, 0
                elif rr < g.r_steel and xx < g.L_ins:
                    D, tag = D_st, 1
                elif rr < g.r_wrap and xx < self.x_coh_end:
                    D, tag = D_wrap, 2
                elif rr < g.r_steel and xx >= g.L_ins:
                    D, tag = D_lam, 3          # 螺套末端之后为层压
                else:
                    D, tag = D_lam, 3
                outer = rr > g.r_wrap
                n = [self.node_of(i, k, outer), self.node_of(i + 1, k, outer),
                     self.node_of(i + 1, k + 1, outer), self.node_of(i, k + 1, outer)]
                self.elems.append(n); self.Dmap.append(D); self.etag.append(tag)
        self.elems = np.array(self.elems)
        self.etag = np.array(self.etag)

    # -------------------------------------------------- 体单元装配
    def _assemble_bulk(self):
        rows, cols, vals = [], [], []
        gp = np.array([-1, 1]) / np.sqrt(3.0)
        for e, (nd, D) in enumerate(zip(self.elems, self.Dmap)):
            rr = self.R[nd]; xx = self.X[nd]
            ke = np.zeros((8, 8))
            for xi in gp:
                for et in gp:
                    N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                                         (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
                    dNx = 0.25 * np.array([-(1 - et), (1 - et), (1 + et), -(1 + et)])
                    dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                    J = np.array([[dNx @ rr, dNx @ xx], [dNe @ rr, dNe @ xx]])
                    detJ = np.linalg.det(J)
                    if detJ <= 0:
                        continue
                    dN = np.linalg.solve(J, np.vstack([dNx, dNe]))   # (2,4): d/dr, d/dz
                    rg = float(N @ rr)
                    B = np.zeros((4, 8))
                    B[0, 0::2] = dN[0]            # εr = du_r/dr
                    B[1, 1::2] = dN[1]            # εz = du_z/dz
                    B[2, 0::2] = N / max(rg, 1e-9)  # εθ = u_r/r
                    B[3, 0::2] = dN[1]            # γrz
                    B[3, 1::2] = dN[0]
                    ke += B.T @ D @ B * detJ * 2 * np.pi * rg
            dofs = np.stack([2 * nd, 2 * nd + 1], 1).ravel()
            rows.append(np.repeat(dofs, 8)); cols.append(np.tile(dofs, 8))
            vals.append(ke.ravel())
        self.Kb = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.ndof, self.ndof)).tocsr()

    # -------------------------------------------------- 内聚力界面
    def _build_cohesive(self):
        ix = np.flatnonzero(self.coh_mask)
        self.coh_ix = ix
        xin = self.x[ix]
        tri = np.zeros(len(ix))
        tri[:-1] += 0.5 * np.diff(xin)
        tri[1:] += 0.5 * np.diff(xin)
        self.coh_area = 2 * np.pi * self.g.r_wrap * tri      # 每个节点对的贡献面积 mm²
        self.coh_x = xin
        self.dof_in = np.stack([2 * self.nid[self.i_coh, ix],
                                2 * self.nid[self.i_coh, ix] + 1], 1)
        self.dof_out = np.stack([2 * self.dup[ix], 2 * self.dup[ix] + 1], 1)
        self.dmg = np.zeros(len(ix))

    def _coh_state(self, u):
        """由位移求分离量、损伤与切线刚度。"""
        c = self.c
        dn = u[self.dof_out[:, 0]] - u[self.dof_in[:, 0]]     # 径向张开
        ds = u[self.dof_out[:, 1]] - u[self.dof_in[:, 1]]     # 轴向滑移
        dnp = np.maximum(dn, 0.0)
        lam = np.sqrt(dnp ** 2 + ds ** 2)
        # 混合比
        beta = np.divide(np.abs(ds), np.maximum(dnp, 1e-12))
        GII_frac = beta ** 2 / (1.0 + beta ** 2)
        # 起始有效分离（二次应力准则）
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = (dnp / (c.tn0 / c.K)) ** 2 + (np.abs(ds) / (c.ts0 / c.K)) ** 2
            d_m0 = np.where(inv > 0, lam / np.sqrt(np.maximum(inv, 1e-30)), c.tn0 / c.K)
        d_m0 = np.clip(d_m0, 1e-9, None)
        Gc = c.GIc + (c.GIIc - c.GIc) * GII_frac ** c.eta
        d_mf = np.maximum(2.0 * Gc / (c.K * d_m0), d_m0 * 1.001)
        d_new = np.where(lam <= d_m0, 0.0,
                         np.clip(d_mf * (lam - d_m0) / np.maximum(lam * (d_mf - d_m0), 1e-30),
                                 0.0, 1.0))
        d = np.maximum(self.dmg, d_new)          # 不可逆
        return dn, ds, d

    def _coh_matrix(self, d, dn):
        """内聚力的割线刚度（压缩不退化）。"""
        c = self.c
        kn = np.where(dn >= 0, (1 - d) * c.K, c.K) * self.coh_area
        ks = (1 - d) * c.K * self.coh_area
        rows, cols, vals = [], [], []
        for comp, k in ((0, kn), (1, ks)):
            a = self.dof_in[:, comp]; b = self.dof_out[:, comp]
            rows.append(np.concatenate([a, a, b, b]))
            cols.append(np.concatenate([a, b, a, b]))
            vals.append(np.concatenate([k, -k, -k, k]))
        return sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.ndof, self.ndof)).tocsr()

    # -------------------------------------------------- 边界与求解
    def _bc(self):
        g = self.g
        load_ix = np.flatnonzero(self.x <= g.l_eng)
        ir_load = np.flatnonzero((self.r >= g.r_bore - 1e-9) & (self.r <= g.r_steel + 1e-9))
        self.load_dof = np.unique(
            np.array([2 * self.nid[i, k] + 1 for i in ir_load for k in load_ix]))
        react = np.flatnonzero(self.r >= g.r_react)
        self.fix_dof = np.array([2 * self.nid[i, 0] + 1 for i in react])
        self.fix_r = np.array([2 * self.nid[0, k] for k in range(self.nx)])  # 轴上 u_r=0

    def solve(self, u_applied, max_it=40, tol=1e-7, verbose=False):
        """位移控制求解，返回反力与状态。"""
        if not hasattr(self, 'load_dof'):
            self._bc()
        con = np.concatenate([self.load_dof, self.fix_dof, self.fix_r])
        val = np.concatenate([np.full(len(self.load_dof), -u_applied),
                              np.zeros(len(self.fix_dof) + len(self.fix_r))])
        _, keep = np.unique(con, return_index=True)
        con, val = con[keep], val[keep]
        free = np.setdiff1d(np.arange(self.ndof), con)
        u = np.zeros(self.ndof); u[con] = val
        d_prev = self.dmg.copy()
        for it in range(max_it):
            dn, ds, d = self._coh_state(u)
            K = (self.Kb + self._coh_matrix(d, dn)).tocsc()
            rhs = -(K @ u)[free]
            u_f = spla.spsolve(K[free][:, free], rhs)
            u_new = u.copy(); u_new[free] = u_f
            err = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u_new)), 1e-12)
            u = 0.5 * u + 0.5 * u_new if it < 3 else u_new
            if err < tol and it > 1:
                break
        dn, ds, d = self._coh_state(u)
        self.dmg = np.maximum(d_prev, d)
        K = (self.Kb + self._coh_matrix(self.dmg, dn)).tocsc()
        R = K @ u
        P = -float(R[self.load_dof].sum())
        if verbose:
            print(f'    iters={it+1} err={err:.2e} P={P/1e3:.1f} kN '
                  f'debond={self.debond_len():.0f} mm')
        return dict(u=u, P=P, dn=dn, ds=ds, d=self.dmg.copy(), it=it + 1, err=err)

    def debond_len(self, thr=0.99):
        """完全损伤区（d>thr）的轴向长度。界面自埋入端向端面扩展时从 x=L_ins 起算。"""
        m = self.dmg > thr
        if not m.any():
            return 0.0
        return float(self.coh_x[m].max() - self.coh_x[m].min())

    def debond_front(self, thr=0.99):
        m = self.dmg > thr
        return float(self.coh_x[m].min()) if m.any() else float('nan')

    # -------------------------------------------------- 应力后处理
    def stress(self, u):
        """单元中心应力（σr, σz, σθ, τrz）与 von Mises。"""
        out = np.zeros((len(self.elems), 5))
        for e, (nd, D) in enumerate(zip(self.elems, self.Dmap)):
            rr = self.R[nd]; xx = self.X[nd]
            N = 0.25 * np.ones(4)
            dNx = 0.25 * np.array([-1, 1, 1, -1]); dNe = 0.25 * np.array([-1, -1, 1, 1])
            J = np.array([[dNx @ rr, dNx @ xx], [dNe @ rr, dNe @ xx]])
            if np.linalg.det(J) <= 0:
                continue
            dN = np.linalg.solve(J, np.vstack([dNx, dNe]))
            rg = float(N @ rr)
            B = np.zeros((4, 8))
            B[0, 0::2] = dN[0]; B[1, 1::2] = dN[1]
            B[2, 0::2] = N / max(rg, 1e-9)
            B[3, 0::2] = dN[1]; B[3, 1::2] = dN[0]
            dofs = np.stack([2 * nd, 2 * nd + 1], 1).ravel()
            s = D @ (B @ u[dofs])
            vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 +
                                (s[2] - s[0]) ** 2) + 3 * s[3] ** 2)
            out[e] = [s[0], s[1], s[2], s[3], vm]
        return out

    @property
    def elem_centers(self):
        rc = np.array([self.R[nd].mean() for nd in self.elems])
        xc = np.array([self.X[nd].mean() for nd in self.elems])
        return rc, xc


if __name__ == '__main__':
    import time
    t0 = time.time()
    M = AxiCZM()
    print(f'节点 {M.n_node}，自由度 {M.ndof}，单元 {len(M.elems)}，'
          f'内聚力节点对 {len(M.coh_x)}，装配 {time.time()-t0:.1f}s')
    r = M.solve(0.05, verbose=True)
    print(f'试算完成 {time.time()-t0:.1f}s')
