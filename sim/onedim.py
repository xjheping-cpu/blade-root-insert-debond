# -*- coding: utf-8 -*-
"""
单 cell 一维剪滞模型（M0）。物理与 M1 相同，实现完全独立，用于交叉验证 M1。

层压杆 + 螺套杆 + 分布界面剪切弹簧 + 沿啮合段分布的螺柱力 + 两个端面单侧接触。
只有一个周向 cell，因此不含周向改道——正因为如此，它能把 M1 里"轴向传力"这一半
单独隔离出来校验。
"""
from __future__ import annotations
import numpy as np
from root_model import Geom, Mat, Stud, _x_grid


class OneDCell:
    def __init__(self, geom: Geom = Geom(), mat: Mat = Mat(), stud: Stud = Stud()):
        self.g, self.m, self.st = geom, mat, stud
        self.x = _x_grid(geom)
        self.nx = len(self.x)
        self.n_ins = int(np.sum(self.x <= geom.L_ins + 1e-9))
        self.x_ins = self.x[: self.n_ins]
        self.k_q = (1.0 / (mat.t_adh / mat.G_adh + mat.t_lam_shear / mat.G_lam)) * np.pi * geom.D_ins
        lt = np.zeros(self.n_ins)
        lt[:-1] += 0.5 * np.diff(self.x_ins)
        lt[1:] += 0.5 * np.diff(self.x_ins)
        self.trib = lt
        w = np.where(self.x_ins <= geom.l_eng, lt, 0.0)
        self.w_eng = w / w.sum()
        # 层压截面：螺套段为净截面，之后为整个间距
        xc = 0.5 * (self.x[:-1] + self.x[1:])
        self.A_lam = np.where(xc < geom.L_ins, geom.A_lam_cell, geom.pitch * geom.t_wall)

    def solve(self, w_flange: float, F_M: float, gap_s: float = 0.0, debond: float = 0.0):
        nx, ni = self.nx, self.n_ins
        n = nx + ni                       # [u_lam(0..nx-1), u_ins(0..ni-1)]
        K = np.zeros((n, n)); f = np.zeros(n)
        # 层压杆
        for k in range(nx - 1):
            kk = self.m.E_lam_x * self.A_lam[k] / (self.x[k + 1] - self.x[k])
            K[np.ix_([k, k + 1], [k, k + 1])] += kk * np.array([[1, -1], [-1, 1]])
        # 螺套杆
        EA = self.m.E_steel * self.g.A_steel
        for k in range(ni - 1):
            kk = EA / (self.x_ins[k + 1] - self.x_ins[k])
            a = nx + k
            K[np.ix_([a, a + 1], [a, a + 1])] += kk * np.array([[1, -1], [-1, 1]])
        # 界面剪切弹簧
        kq = np.where(self.x_ins < debond, 0.0, self.k_q) * self.trib
        for k in range(ni):
            a, b = nx + k, k
            K[a, a] += kq[k]; K[b, b] += kq[k]
            K[a, b] -= kq[k]; K[b, a] -= kq[k]
        # 螺柱（秩一）+ 预紧
        e = nx + np.arange(ni)
        K[np.ix_(e, e)] += self.st.k_S * np.outer(self.w_eng, self.w_eng)
        f[e] += self.w_eng * (self.st.k_S * w_flange - F_M)
        # 约束：远端层压 u=0；两个端面单侧接触
        act_l, act_s = True, True
        for _ in range(10):
            con = [nx - 1]; val = [0.0]
            if act_l: con.append(0); val.append(w_flange)
            if act_s: con.append(nx); val.append(w_flange - gap_s)
            con = np.array(con); val = np.array(val)
            free = np.setdiff1d(np.arange(n), con)
            u = np.zeros(n); u[con] = val
            u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free] - K[np.ix_(free, con)] @ val)
            R = K @ u - f
            nl = (R[0] >= 0) if act_l else (u[0] < w_flange)
            ns = (R[nx] >= 0) if act_s else (u[nx] < w_flange - gap_s)
            if (nl, ns) == (act_l, act_s):
                break
            act_l, act_s = nl, ns
        Cl = R[0] if act_l else 0.0
        Cs = R[nx] if act_s else 0.0
        FS = F_M + self.st.k_S * (self.w_eng @ u[nx:] - w_flange)
        Ns = EA * np.diff(u[nx:]) / np.diff(self.x_ins)
        eps_l = np.diff(u[:nx]) / np.diff(self.x)
        return dict(u=u, Cl=Cl, Cs=Cs, FS=FS, FA=FS - Cl - Cs, Ns=Ns, eps=eps_l,
                    x_c=0.5 * (self.x[:-1] + self.x[1:]),
                    x_ins_c=0.5 * (self.x_ins[:-1] + self.x_ins[1:]),
                    chi=Cl / (Cl + Cs) if (Cl + Cs) != 0 else 0.0)

    def solve_FA(self, F_A_target: float, F_M: float, gap_s: float = 0.0, debond: float = 0.0):
        """割线迭代求满足目标外载的法兰位移。"""
        w = 0.0
        r = self.solve(w, F_M, gap_s, debond)
        r2 = self.solve(w + 1e-3, F_M, gap_s, debond)
        k = (r2['FA'] - r['FA']) / 1e-3
        for _ in range(30):
            if abs(r['FA'] - F_A_target) < 1.0:
                break
            w += (F_A_target - r['FA']) / k
            rn = self.solve(w, F_M, gap_s, debond)
            if abs(rn['FA'] - r['FA']) > 1e-9:
                k = (rn['FA'] - r['FA']) / (w - (w - (F_A_target - r['FA']) / k)) if False else k
            r = rn
        r['w'] = w
        return r

    def calibrate_FM(self, target_FS: float, gap_s: float = 0.0):
        r = self.solve(0.0, target_FS, gap_s)
        return target_FS / max(r['FS'] / target_FS, 1e-6)


def closed_form_check(m: Mat, g: Geom, k_q: float):
    """无端面接触、螺柱力集中在 x=0 的退化情形，有闭式解，用来验证 1D 实现。"""
    EA_s = m.E_steel * g.A_steel
    EA_l = m.E_lam_x * g.A_lam_cell
    C = 1.0 / EA_s + 1.0 / EA_l
    lam = np.sqrt(k_q * C)
    return lam, 1.0 / lam
