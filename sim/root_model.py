# -*- coding: utf-8 -*-
"""
M1：叶根连接的二维展开壳 + 螺套杆有限元模型。

对象：采用预埋螺套连接的叶根。几何自 config/ 下的配置读入，见 geometry.py。
把圆环根部沿周向展开成平面（x 轴向、y 周向，y 方向周期），
用变厚度正交各向异性膜单元表示层压，用一维杆 + 分布剪切弹簧表示每个预埋螺套，
用单侧接触（法兰刚性面）+ 预紧螺柱表示 x=0 的连接界面。

这个模型能自然给出的量（都不是输入，是解出来的）：
  · 螺栓载荷系数 Φ = dF_S/dF_A
  · 端面夹紧力在层压端面与螺套端面之间的分配比 χ
  · 传力长度 1/λ 与界面剪应力沿埋深的分布
  · 某一根螺柱断裂／某一个螺套脱粘时，载荷如何沿周向改道，
    以及根部舱内表面（光纤所在面）的轴向应变场怎么变

坐标与符号：
  x：轴向，x=0 为叶根端面（法兰面），+x 指向叶尖
  y：周向展开坐标，y ∈ [0, 2πR)，周期
  u：x 向位移；v：y 向位移；拉为正
  F_A：单个螺栓位置上的外载（拉为正）；F_S：螺柱轴力；F_M：预紧力
  C_l / C_s：法兰压在层压端面／螺套端面上的接触力（压为正）
  F_A = F_S - C_l - C_s
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------- 参数
# 几何由 geometry.Geom 统一提供，默认值取自 config/ 下的配置文件。
# 换机型只改配置，本文件不动。
from geometry import Geom      # noqa: E402  （放在 import 区之后便于阅读）


@dataclass(frozen=True)
class Mat:
    """材料。层压为叶根三轴玻纤，工程值；螺套 42CrMoA。"""
    E_lam_x: float = 30.0e3      # 轴向模量 MPa
    E_lam_y: float = 18.0e3      # 周向模量
    G_lam: float = 8.0e3         # 面内剪切
    nu_xy: float = 0.40
    E_steel: float = 210.0e3
    G_adh: float = 1200.0        # 胶层剪切模量
    t_adh: float = 0.5           # 胶层厚度 mm
    t_lam_shear: float = 16.0    # 界面剪切传递的层压等效厚度（剪滞参数）


@dataclass(frozen=True)
class Stud:
    """螺柱 M42×4.5，10.9 级。"""
    d: float = 42.0
    A_s: float = 1120.0          # GB/T 16823.1 应力截面积
    Rp02: float = 940.0          # 10.9 级
    l_clamp: float = 160.0       # [推断] 夹紧长度（叶片法兰 + 变桨轴承内圈）
    E: float = 210.0e3

    @property
    def F_02(self) -> float:
        return self.Rp02 * self.A_s / 1000.0     # kN

    @property
    def k_S(self) -> float:
        """螺柱轴向刚度 N/mm（含 0.4d 旋合当量长度）。"""
        return self.E * self.A_s / (self.l_clamp + 0.4 * self.d)


# ----------------------------------------------------------------------------- 网格
def _x_grid(g: Geom) -> np.ndarray:
    """轴向分级网格：端面附近 6 mm，螺套中段 18.5 mm，远场 25.6 mm。"""
    a = np.linspace(0.0, 120.0, 21)
    b = np.linspace(120.0, g.L_ins, 21)[1:]
    c = np.linspace(g.L_ins, g.x_max, 17)[1:]
    return np.concatenate([a, b, c])


def _t_axial(dy: np.ndarray, g: Geom) -> np.ndarray:
    """螺套段内，层压的等效厚度 = 壁厚 - 螺套圆在该 y 处的弦长（面积精确）。"""
    r = g.D_ins / 2.0
    chord = 2.0 * np.sqrt(np.clip(r ** 2 - dy ** 2, 0.0, None))
    return g.t_wall - chord


# ----------------------------------------------------------------------------- 单元
def _ke_rect(a: float, b: float, D: np.ndarray) -> np.ndarray:
    """矩形四节点双线性平面应力单元，单位厚度刚度（8×8）。节点序：(0,0)(a,0)(a,b)(0,b)。"""
    gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    ke = np.zeros((8, 8))
    for xi in gp:
        for eta in gp:
            dN_dxi = np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)]) / 4.0
            dN_de = np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]) / 4.0
            dN_dx = dN_dxi * (2.0 / a)
            dN_dy = dN_de * (2.0 / b)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dx
            B[1, 1::2] = dN_dy
            B[2, 0::2] = dN_dy
            B[2, 1::2] = dN_dx
            ke += B.T @ D @ B * (a * b / 4.0)
    return ke


def _D_plane_stress(m: Mat) -> np.ndarray:
    nu_yx = m.nu_xy * m.E_lam_y / m.E_lam_x
    den = 1.0 - m.nu_xy * nu_yx
    return np.array([
        [m.E_lam_x / den, nu_yx * m.E_lam_x / den, 0.0],
        [nu_yx * m.E_lam_x / den, m.E_lam_y / den, 0.0],
        [0.0, 0.0, m.G_lam],
    ])


# ----------------------------------------------------------------------------- 主模型
class RootFE:
    """叶根二维展开有限元模型。n_cells 缺省为整环螺套数，也可取较小值（窗口，周向周期）。"""

    def __init__(self, n_cells: int | None = None, m_y: int = 4,
                 geom: Geom = Geom(), mat: Mat = Mat(), stud: Stud = Stud()):
        self.g, self.m, self.st = geom, mat, stud
        # 缺省取整环：随几何配置走，不写死
        self.n_cells = int(geom.n_bolt if n_cells is None else n_cells)
        self.m_y = int(m_y)

        self.x = _x_grid(geom)
        self.nx = len(self.x)
        self.dx = np.diff(self.x)
        self.ny = self.n_cells * self.m_y          # 周向节点数（周期，节点=单元数）
        self.dy = geom.pitch / self.m_y
        self.y = np.arange(self.ny) * self.dy

        self.n_ins_node = int(np.sum(self.x <= geom.L_ins + 1e-9))
        self.x_ins = self.x[: self.n_ins_node]

        self.n_pnode = self.ny * self.nx
        self.ndof_plate = 2 * self.n_pnode
        # 最后一个自由度是周向"广义周期"跳变 Δ：允许整环周长在泊松作用下变化。
        # 不加这一项会把环向应变锁成 0，轴向应变被系统性压低 (1-ν_xy·ν_yx) ≈ 9.6%。
        self.i_hoop = self.ndof_plate + self.n_cells * self.n_ins_node
        self.ndof = self.i_hoop + 1

        # 每个螺套中心落在节点列 iy = j*m_y + m_y//2
        self.iy_center = np.arange(self.n_cells) * self.m_y + self.m_y // 2

        self._build_thickness()
        self._k_q = self._shear_lag_k()
        self._assemble_linear()

    # ---------------------------------------------------------------- 几何/材料场
    def _build_thickness(self):
        """每个单元的等效层压厚度 t_e[iy, ix]。"""
        g = self.g
        # 单元 y 中心到最近螺套中心的距离
        yc = (np.arange(self.ny) + 0.5) * self.dy
        dyc = np.minimum((yc % g.pitch), g.pitch - (yc % g.pitch))
        # 单元内 21 点平均（面积精确）
        s = np.linspace(-0.5, 0.5, 21) * self.dy
        t_ins = np.array([np.mean(_t_axial(np.abs(d + s), g)) for d in dyc])
        t = np.empty((self.ny, self.nx - 1))
        xc = 0.5 * (self.x[:-1] + self.x[1:])
        for ix, xx in enumerate(xc):
            t[:, ix] = t_ins if xx < g.L_ins else g.t_wall
        self.t_elem = t

        # x=0 端面各节点的受压层压面积（用于接触与 χ 统计）
        ycn = np.arange(self.ny) * self.dy
        dyn = np.minimum((ycn % g.pitch), g.pitch - (ycn % g.pitch))
        t_node = _t_axial(np.minimum(dyn, g.D_ins / 2.0), g)
        self.A_face_node = t_node * self.dy

    def _shear_lag_k(self) -> float:
        """螺套—层压界面单位长度剪切刚度 k_q [N/mm 每 mm 滑移]。"""
        m = self.m
        compliance = m.t_adh / m.G_adh + m.t_lam_shear / m.G_lam
        return (1.0 / compliance) * np.pi * self.g.D_ins

    @property
    def lam_analytic(self) -> float:
        """解析剪滞衰减系数 λ [1/mm]。"""
        EA_s = self.m.E_steel * self.g.A_steel
        EA_l = self.m.E_lam_x * self.g.A_lam_cell
        return float(np.sqrt(self._k_q * (1.0 / EA_s + 1.0 / EA_l)))

    # ---------------------------------------------------------------- 组装
    def _node(self, iy, ix):
        return (iy % self.ny) * self.nx + ix

    def _assemble_linear(self):
        """装配与缺陷无关的线性部分：膜单元 + 螺套杆。剪切弹簧与螺柱在 solve 中按状态加。"""
        D = _D_plane_stress(self.m)
        rows, cols, vals = [], [], []

        ke_unit = [_ke_rect(self.dx[ix], self.dy, D) for ix in range(self.nx - 1)]
        for ix in range(self.nx - 1):
            ke = ke_unit[ix]
            n0 = self._node(np.arange(self.ny), ix)
            n1 = self._node(np.arange(self.ny), ix + 1)
            n2 = self._node(np.arange(self.ny) + 1, ix + 1)
            n3 = self._node(np.arange(self.ny) + 1, ix)
            nodes = np.stack([n0, n1, n2, n3], axis=1)               # (ny,4)
            dofs = np.stack([2 * nodes, 2 * nodes + 1], axis=2).reshape(self.ny, 8)
            t = self.t_elem[:, ix]
            r = np.repeat(dofs, 8, axis=1).ravel()
            c = np.tile(dofs, (1, 8)).ravel()
            v = (ke.ravel()[None, :] * t[:, None]).ravel()
            rows.append(r); cols.append(c); vals.append(v)

        # 缝合行单元（iy = ny-1 连回 iy = 0）的 v 位移要加上跳变 Δ
        iy = self.ny - 1
        h = self.i_hoop
        for ix in range(self.nx - 1):
            ke = ke_unit[ix] * self.t_elem[iy, ix]
            nn = [self._node(iy, ix), self._node(iy, ix + 1),
                  self._node(iy + 1, ix + 1), self._node(iy + 1, ix)]
            g8 = np.array([d for n in nn for d in (2 * n, 2 * n + 1)])
            for i in (5, 7):                       # 节点 2、3 的 v（绕回 iy=0）
                rows.append(np.full(8, h)); cols.append(g8); vals.append(ke[i, :].copy())
                rows.append(g8); cols.append(np.full(8, h)); vals.append(ke[:, i].copy())
            ssum = ke[5, 5] + ke[5, 7] + ke[7, 5] + ke[7, 7]
            rows.append(np.array([h])); cols.append(np.array([h])); vals.append(np.array([ssum]))

        # 螺套杆
        EA = self.m.E_steel * self.g.A_steel
        for j in range(self.n_cells):
            base = self.ndof_plate + j * self.n_ins_node
            for k in range(self.n_ins_node - 1):
                L = self.x_ins[k + 1] - self.x_ins[k]
                kk = EA / L
                d = np.array([base + k, base + k + 1])
                rows.append(np.repeat(d, 2)); cols.append(np.tile(d, 2))
                vals.append(np.array([kk, -kk, -kk, kk]))

        self._K_rows = np.concatenate([np.asarray(r).ravel() for r in rows])
        self._K_cols = np.concatenate([np.asarray(c).ravel() for c in cols])
        self._K_vals = np.concatenate([np.asarray(v).ravel() for v in vals])

        # 螺套节点的支配长度（剪切弹簧用）
        lt = np.zeros(self.n_ins_node)
        lt[:-1] += 0.5 * np.diff(self.x_ins)
        lt[1:] += 0.5 * np.diff(self.x_ins)
        self.trib_ins = lt

        # 螺纹啮合权重（把螺柱力沿啮合段分布）
        w = np.where(self.x_ins <= self.g.l_eng, self.trib_ins, 0.0)
        self.w_eng = w / w.sum()

        # 端面接触力按"最近螺套中心"归属到 cell（节点 4j 在两个中心等距 → 各半）
        wf = np.zeros((self.n_cells, self.ny))
        for j in range(self.n_cells):
            c = self.iy_center[j]
            for k in range(-(self.m_y // 2), self.m_y // 2 + 1):
                idx = (c + k) % self.ny
                wf[j, idx] += 0.5 if abs(k) == self.m_y // 2 else 1.0
        self.face_w = wf

    # ---------------------------------------------------------------- 求解
    def solve(self, w_flange, F_M, broken=None, debond=None, gap_s: float = 0.0,
              max_iter: int = 12, verbose: bool = False):
        """
        w_flange : (n_cells,) 或标量，法兰刚性面在每个螺栓位置的轴向位移 [mm]
        F_M      : (n_cells,) 或标量，各螺柱预紧力 [N]（断柱处会被置零）
        broken   : 断裂螺柱的 cell 序号列表
        debond   : {cell: 脱粘长度 mm}，或 {cell: (起点, 终点)} 表示任意区间脱粘
        gap_s    : 螺套端面相对叶根层压端面的内缩量 [mm]（>0 时螺套端面不直接承压）
        返回 dict：位移、每个 cell 的 F_A/F_S/C_l/C_s/状态、应变场
        """
        g = self.g
        nC = self.n_cells
        w_f = np.full(nC, float(w_flange)) if np.isscalar(w_flange) else np.asarray(w_flange, float)
        FM = np.full(nC, float(F_M)) if np.isscalar(F_M) else np.asarray(F_M, float).copy()
        broken = set(broken or [])
        debond = dict(debond or {})
        for j in broken:
            FM[j] = 0.0

        # 法兰面在每个 y 节点上的位移（cell 内线性插值）
        w_node = np.interp(self.y, np.concatenate([self.iy_center * self.dy, [self.ny * self.dy]]),
                           np.concatenate([w_f, [w_f[0]]]), period=self.ny * self.dy)

        # --- 与状态相关的刚度项
        rows = [self._K_rows]; cols = [self._K_cols]; vals = [self._K_vals]
        for j in range(nC):
            base = self.ndof_plate + j * self.n_ins_node
            d_deb = debond.get(j, 0.0)
            # 脱粘可以是标量（从端面 0 起算的长度），也可以是 (lo, hi) 区间 [mm]
            if np.isscalar(d_deb):
                m_deb = self.x_ins < d_deb
            else:
                m_deb = (self.x_ins >= d_deb[0]) & (self.x_ins <= d_deb[1])
            kq = np.where(m_deb, 0.0, self._k_q) * self.trib_ins
            p_dof = 2 * self._node(self.iy_center[j], np.arange(self.n_ins_node))
            i_dof = base + np.arange(self.n_ins_node)
            rows.append(np.concatenate([i_dof, i_dof, p_dof, p_dof]))
            cols.append(np.concatenate([i_dof, p_dof, i_dof, p_dof]))
            vals.append(np.concatenate([kq, -kq, -kq, kq]))
            # 螺柱弹性（秩一）
            if j not in broken:
                kS = self.st.k_S
                e = i_dof
                rows.append(np.repeat(e, self.n_ins_node))
                cols.append(np.tile(e, self.n_ins_node))
                vals.append((kS * np.outer(self.w_eng, self.w_eng)).ravel())

        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof)).tocsr()

        # --- 载荷向量：螺柱预紧 + 螺柱随法兰位移的项
        f = np.zeros(self.ndof)
        for j in range(nC):
            if j in broken:
                continue
            base = self.ndof_plate + j * self.n_ins_node
            i_dof = base + np.arange(self.n_ins_node)
            f[i_dof] += self.w_eng * (self.st.k_S * w_f[j] - FM[j])

        # --- 约束：远端 u=0；周向刚体 v 固定一个点；接触为 Dirichlet（主动集）
        fixed_far = 2 * self._node(np.arange(self.ny), self.nx - 1)
        fixed_v = np.array([2 * self._node(0, self.nx - 1) + 1])
        u_face = 2 * self._node(np.arange(self.ny), 0)          # 层压端面 u 自由度
        i_face = self.ndof_plate + np.arange(nC) * self.n_ins_node   # 螺套端面 u 自由度

        act_l = np.ones(self.ny, bool)
        act_s = np.ones(nC, bool)
        u = np.zeros(self.ndof)
        for it in range(max_iter):
            con_dof = np.concatenate([fixed_far, fixed_v, u_face[act_l], i_face[act_s]])
            con_val = np.concatenate([np.zeros(len(fixed_far) + 1),
                                      w_node[act_l], w_f[act_s] - gap_s])
            u = _solve_dirichlet(K, f, con_dof, con_val)
            R = K @ u - f
            # 接触反力：法兰推向 +x 为压，R>0 表示需要外力维持 → 压
            Rl = np.zeros(self.ny); Rl[act_l] = R[u_face[act_l]]
            Rs = np.zeros(nC); Rs[act_s] = R[i_face[act_s]]
            new_l = act_l.copy(); new_s = act_s.copy()
            new_l[act_l] = Rl[act_l] >= 0.0
            new_s[act_s] = Rs[act_s] >= 0.0
            # 检查已张开的是否重新侵入
            op_l = ~act_l
            if op_l.any():
                new_l[op_l] = u[u_face[op_l]] < w_node[op_l]
            op_s = ~act_s
            if op_s.any():
                new_s[op_s] = u[i_face[op_s]] < w_f[op_s] - gap_s
            if np.array_equal(new_l, act_l) and np.array_equal(new_s, act_s):
                break
            act_l, act_s = new_l, new_s
        if verbose:
            print(f"  active-set iters={it+1}, open faces={np.sum(~act_l)}/{self.ny}")

        return self._post(u, K, f, w_f, FM, broken, act_l, act_s, R)

    # ---------------------------------------------------------------- 后处理
    def _post(self, u, K, f, w_f, FM, broken, act_l, act_s, R):
        nC = self.n_cells
        u_face = 2 * self._node(np.arange(self.ny), 0)
        i_face = self.ndof_plate + np.arange(nC) * self.n_ins_node
        Cl_node = np.zeros(self.ny); Cl_node[act_l] = R[u_face[act_l]]
        Cs = np.zeros(nC); Cs[act_s] = R[i_face[act_s]]
        # 按 cell 汇总层压端面接触力
        Cl = self.face_w @ Cl_node
        # 螺柱轴力
        FS = np.zeros(nC)
        for j in range(nC):
            if j in broken:
                continue
            base = self.ndof_plate + j * self.n_ins_node
            ubar = float(self.w_eng @ u[base: base + self.n_ins_node])
            FS[j] = FM[j] + self.st.k_S * (ubar - w_f[j])
        FA = FS - Cl - Cs

        # 应变场（膜应变 ε_xx），单元中心
        U = u[: self.ndof_plate].reshape(self.n_pnode, 2)
        ux = U[:, 0].reshape(self.ny, self.nx)
        eps = np.diff(ux, axis=1) / self.dx[None, :]

        # 螺套轴力分布
        Ns = np.zeros((nC, self.n_ins_node - 1))
        EA = self.m.E_steel * self.g.A_steel
        for j in range(nC):
            base = self.ndof_plate + j * self.n_ins_node
            us = u[base: base + self.n_ins_node]
            Ns[j] = EA * np.diff(us) / np.diff(self.x_ins)

        state = np.array(['intact'] * nC, dtype=object)
        for j in broken:
            state[j] = 'broken'
        for j in range(nC):
            if j in broken:
                continue
            if not act_l[self.face_w[j] > 0].all():
                state[j] = 'open'

        return dict(u=u, eps=eps, x_c=0.5 * (self.x[:-1] + self.x[1:]), y=self.y,
                    FA=FA, FS=FS, Cl=Cl, Cs=Cs, FM=FM, Ns=Ns, state=state,
                    open_face=~act_l, chi=np.divide(Cl, Cl + Cs, out=np.zeros(nC),
                                                    where=(Cl + Cs) != 0))

    def calibrate_FM(self, target_FS: float, gap_s: float = 0.0, w: float = 0.0):
        """返回能在 F_A≈0 时得到指定安装轴力 target_FS 的 F_M 输入值。
        现场按扭矩或拉伸拧到目标轴力，端面构造（gap_s）只改变夹紧力去向，不减小安装轴力。"""
        r1 = self.solve(w, target_FS, gap_s=gap_s)
        a = r1['FS'].mean() / target_FS
        return target_FS / max(a, 1e-6)

    # ---------------------------------------------------------------- 便捷接口
    def cell_profile(self, res, j: int):
        """第 j 个螺套中心线上的轴向应变剖面（内表面光纤位置）。"""
        return res['x_c'], res['eps'][self.iy_center[j] % self.ny, :]

    def fingerprint(self, res, x_probe: float):
        """给定轴向位置，取每个螺套中心线的应变 → 整环各位的周向基线分布。"""
        ix = int(np.argmin(np.abs(res['x_c'] - x_probe)))
        return res['eps'][self.iy_center % self.ny, ix]

    def solve_for_moment(self, M_flap, M_edge=0.0, F_z=0.0, F_M=420e3,
                         broken=None, debond=None, gap_s=0.0, tol=1e-3, verbose=False, p0=None):
        """给定叶根弯矩，牛顿迭代求刚性法兰的 (w0, φ_f, φ_e)。仅整环模型有意义。"""
        g = self.g
        th = self.iy_center * self.dy / g.R_bc
        basis = np.stack([np.ones_like(th), g.R_bc * np.cos(th), g.R_bc * np.sin(th)])
        target = np.array([F_z, M_flap, M_edge])

        def resid(p):
            w = p @ basis
            r = self.solve(w, F_M, broken, debond, gap_s)
            got = np.array([r['FA'].sum(),
                            (r['FA'] * g.R_bc * np.cos(th)).sum(),
                            (r['FA'] * g.R_bc * np.sin(th)).sum()])
            return got - target, r

        p = np.zeros(3) if p0 is None else np.asarray(p0, float).copy()
        r0, res = resid(p)
        scale = np.array([1e-3, 1e-7, 1e-7])
        for it in range(8):
            if np.max(np.abs(r0) / np.maximum(np.abs(target), [1e4, 1e6, 1e6])) < tol:
                break
            J = np.zeros((3, 3))
            for k in range(3):
                dp = np.zeros(3); dp[k] = scale[k]
                rk, _ = resid(p + dp)
                J[:, k] = (rk - r0) / scale[k]
            p = p - np.linalg.solve(J, r0)
            r0, res = resid(p)
        if verbose:
            print(f"  moment solve: {it+1} iters, resid={r0}")
        res['flange'] = p
        return res


# ----------------------------------------------------------------------------- 工具
def _solve_dirichlet(K, f, con_dof, con_val):
    """带 Dirichlet 约束的稀疏求解（行列消去，保持对称）。"""
    n = K.shape[0]
    mask = np.ones(n, bool)
    mask[con_dof] = False
    free = np.flatnonzero(mask)
    uc = np.zeros(n)
    uc[con_dof] = con_val
    rhs = f[free] - (K @ uc)[free]
    Kff = K[free][:, free].tocsc()
    uf = spla.spsolve(Kff, rhs, use_umfpack=False)
    u = uc.copy()
    u[free] = uf
    return u


def analytic_shearlag(fe: RootFE, F_A: float, F_M: float, chi: float, Phi: float):
    """
    一维解析剪滞解，用于校验有限元的完好状态。Phi/chi 由有限元给出。
    边界：x=0 处 N_s = F_S - (1-chi)F_KR，N_l = -chi·F_KR；x=L 处 N_s = 0。
    """
    g, m = fe.g, fe.m
    EA_s = m.E_steel * g.A_steel
    EA_l = m.E_lam_x * g.A_lam_cell
    Cc = 1.0 / EA_s + 1.0 / EA_l
    lam = fe.lam_analytic
    L = g.L_ins
    F_KR = F_M - (1.0 - Phi) * F_A
    F_S = F_M + Phi * F_A
    Ns0 = F_S - (1.0 - chi) * F_KR
    Ns_inf = F_A / (EA_l * Cc)
    x = np.linspace(0, L, 400)
    # 两端边界层叠加
    A = (Ns0 - Ns_inf)
    B = -Ns_inf
    Ns = Ns_inf + A * np.exp(-lam * x) + B * np.exp(-lam * (L - x))
    Nl = F_A - Ns
    return x, Ns, Nl, Nl / EA_l * 1e6      # 微应变


if __name__ == '__main__':
    fe = RootFE(n_cells=9, m_y=4)
    print(f"pitch={fe.g.pitch:.2f} mm  t_wall={fe.g.t_wall:.0f} mm  "
          f"A_steel={fe.g.A_steel:.0f} A_lam={fe.g.A_lam_cell:.0f} mm^2")
    print(f"k_q={fe._k_q:.0f} N/mm/mm  1/lambda={1/fe.lam_analytic:.1f} mm")
    print(f"DOF={fe.ndof}  nodes={fe.n_pnode}  x-nodes={fe.nx}  y-nodes={fe.ny}")
    import time
    t0 = time.time()
    r = fe.solve(w_flange=0.0, F_M=420e3, verbose=True)
    print(f"preload-only solve {time.time()-t0:.2f}s  F_A={r['FA'][:3]/1e3} kN  "
          f"F_S={r['FS'][:3]/1e3}  Cl={r['Cl'][:3]/1e3}  Cs={r['Cs'][:3]/1e3}  chi={r['chi'][:3]}")
