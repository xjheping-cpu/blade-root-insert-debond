# -*- coding: utf-8 -*-
"""
M2′：预埋螺套单胞的轴对称有限元模型，分层构造 + 内聚力界面。

建模重规划要求的三项改动在本模块落地：
  A2  界面由"线弹性剪切弹簧 + 人工指定脱粘长度"改为双线性内聚力本构，
      脱粘长度成为解的一部分，因而可以回答扩展是否稳定；
  A3  螺套局部按实际构成分层：钢衬套 / 过渡层 / 玻纤束缠绕层 / 拉挤 GFRP 块 /
      叶根层压，层间分别设界面，而不是合并成单一界面；
  A4  载荷按真实连接的力流分解施加，预紧由螺柱弹簧产生，
      载荷系数 Φ 由模型解出而不是假定。

坐标：x 为轴向，x = 0 是叶根端面（与轮毂法兰接触），x 增大方向指向叶片内部；
      r 为到螺套轴线的距离。拉为正。长度 mm，力 N，应力 MPa。

力流分解（与 VDI 2230 一致，可用于自检）：
      螺柱轴力 F_S  作用于螺纹啮合段，方向 −x（把螺套往轮毂方向拽）
      端面承压 F_KR 作用于叶根端面环带，方向 +x，单侧接触、只压不拉
      远端膜力  F_A  作用于 x = L_tot 截面，方向 +x
      平衡即 F_S = F_KR + F_A，载荷系数 Φ = (F_S − F_M) / F_A。

单胞外半径按面积等效取定：π·R_out² = 节距 × 壁厚。
"""
from __future__ import annotations

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

# ------------------------------------------------------------------ 几何常量
P_PITCH = _G.pitch
T_WALL = _G.t_wall
R_CELL = float(np.sqrt(P_PITCH * T_WALL / np.pi))     # 54.63 mm

R_BORE = _G.d_bore / 2          # 螺纹小径/2
R_STEEL = _G.D_ins / 2          # 螺套外半径
# 以下三层的厚度是工艺推断值，与机型无关，按增量叠在钢体外径上【推断】
R_TRANS = R_STEEL + 0.5         # 过渡层（树脂富集）外缘，厚 0.5 mm
R_WRAP = R_STEEL + 6.5          # 玻纤束缠绕层外缘，厚 6 mm
R_BLOCK = R_STEEL + 13.5        # 拉挤 GFRP 块外缘

L_INS = _G.L_ins      # 螺套埋深
L_ENG = _G.l_eng      # 螺纹啮合长度
X_BORE = L_ENG + 25.0  # 盲孔底：啮合段之外再留 25 mm【推断】
L_TOT = 700.0         # 模型轴向长度

# 螺柱：M42，公称应力面积 1120 mm²，夹紧长度 = 法兰厚 160 + 0.4×d
K_STUD = 210000.0 * 1120.0 / (160.0 + 0.4 * 42.0)     # 1.3305e6 N/mm
FM_DEFAULT = 420.0e3

MATS = ('void', 'steel', 'trans', 'wrap', 'block', 'lam')
MAT_COL = {'void': '#FFFFFF', 'steel': '#6E7F8D', 'trans': '#C8871B',
           'wrap': '#1E7A5A', 'block': '#2F6690', 'lam': '#BFD3DE'}
MAT_CN = {'void': '螺纹孔', 'steel': '钢衬套 42CrMoA', 'trans': '过渡层',
          'wrap': '玻纤束缠绕层', 'block': '拉挤 GFRP 块', 'lam': '叶根层压'}


@dataclass(frozen=True)
class Layer:
    """轴对称正交各向异性层。轴向 = x，横向 = r 与 θ。"""
    Ea: float
    Et: float
    nu_a: float
    nu_t: float
    Ga: float


LAYERS = {
    'void':  Layer(1.0, 1.0, 0.30, 0.30, 0.4),
    'steel': Layer(210e3, 210e3, 0.30, 0.30, 80.8e3),
    'trans': Layer(3.2e3, 3.2e3, 0.36, 0.36, 1.18e3),      # 环氧【推断】
    'wrap':  Layer(22e3, 20e3, 0.32, 0.35, 5.5e3),          # 缠绕玻纤【推断】
    'block': Layer(45e3, 14e3, 0.29, 0.40, 4.5e3),          # 拉挤板【推断】
    'lam':   Layer(30e3, 15e3, 0.30, 0.40, 5.0e3),          # 三轴层压
}


@dataclass(frozen=True)
class Coh:
    """双线性牵引-分离律。全部为文献量级的【推断】值，须由 DCB / ENF 试验标定。"""
    name: str
    r: float
    K: float = 5.0e4       # 罚刚度 N/mm³
    tn0: float = 12.0      # I 型强度 MPa
    ts0: float = 20.0      # II 型强度 MPa
    GIc: float = 0.5       # I 型断裂能 N/mm
    GIIc: float = 1.5      # II 型断裂能 N/mm
    eta: float = 1.45      # Benzeggagh-Kenane 指数

    def l_cz(self, E=30e3):
        """内聚区长度估计（Hillerborg），判断网格是否够细。"""
        return 0.88 * E * self.GIIc / self.ts0 ** 2


IFACES = (
    Coh('A 钢/过渡层', R_TRANS, tn0=25.0, ts0=35.0, GIc=0.8, GIIc=2.5),
    Coh('B 缠绕层/拉挤块', R_WRAP, tn0=12.0, ts0=20.0, GIc=0.5, GIIc=1.5),
    Coh('C 拉挤块/层压', R_BLOCK, tn0=15.0, ts0=24.0, GIc=0.6, GIIc=1.8),
)


# ------------------------------------------------------------------ 网格工具
def _seg(a, b, n, bias=0.0):
    """一段单调网格。bias>0 向 a 端加密，bias<0 向 b 端加密。"""
    t = np.linspace(0.0, 1.0, n + 1)
    if bias > 0:
        t = t ** (1.0 + bias)
    elif bias < 0:
        t = 1.0 - (1.0 - t) ** (1.0 - bias)
    return a + (b - a) * t


def _cat(*segs):
    return np.unique(np.round(np.concatenate(segs), 9))


def _D(L: Layer):
    """轴对称正交各向异性 D 矩阵，应变次序 (εr, εx, εθ, γrx)。"""
    S = np.zeros((4, 4))
    S[0, 0] = S[2, 2] = 1.0 / L.Et
    S[1, 1] = 1.0 / L.Ea
    S[0, 1] = S[1, 0] = -L.nu_a / L.Ea
    S[1, 2] = S[2, 1] = -L.nu_a / L.Ea
    S[0, 2] = S[2, 0] = -L.nu_t / L.Et
    S[3, 3] = 1.0 / L.Ga
    return np.linalg.inv(S)


# ------------------------------------------------------------------ 模型
class InsertAxi:
    def __init__(self, ifaces=IFACES, nx_fac=1.0, nr_fac=1.0, verbose=True):
        t0 = time.time()
        self.ifaces = tuple(ifaces)
        self.verbose = verbose

        n = lambda k: max(1, int(round(k * nr_fac)))
        self.r = _cat(_seg(0.0, R_BORE, n(3)),
                      _seg(R_BORE, R_STEEL, n(6)),
                      _seg(R_STEEL, R_TRANS, 1),
                      _seg(R_TRANS, R_WRAP, n(3)),
                      _seg(R_WRAP, R_BLOCK, n(3)),
                      _seg(R_BLOCK, R_CELL, n(2)))
        m = lambda k: max(1, int(round(k * nx_fac)))
        self.x = _cat(_seg(0.0, 20.0, m(10)),
                      _seg(20.0, L_ENG, m(12)),
                      _seg(L_ENG, X_BORE, m(5)),
                      _seg(X_BORE, 430.0, m(26)),
                      _seg(430.0, L_INS, m(24)),
                      _seg(L_INS, 560.0, m(14)),
                      _seg(560.0, L_TOT, m(8)))
        self.nr, self.nx = len(self.r), len(self.x)

        self._build_nodes()
        self._build_elems()
        self._assemble()
        self._build_coh()
        self._build_bc()
        self._build_T()
        self.t_build = time.time() - t0
        if verbose:
            print('[mesh] 结点 %d（含界面复制 %d）  单元 %d  自由度 %d(缩减后 %d)  '
                  '装配 %.1fs' % (self.n_node, self.n_node - self.nr * self.nx,
                                len(self.el), self.ndof, self.nred, self.t_build))
            for c in self.ifaces:
                dxmin = float(np.diff(self.x[self.x <= L_INS]).min())
                print('       界面 %-14s r=%5.1f  内聚区长 %.0f mm，'
                      '尖端单元 %.1f mm → %.0f 个单元/内聚区'
                      % (c.name, c.r, c.l_cz(), dxmin, c.l_cz() / dxmin))

    # ---------------------------------------------------------- 结点
    def _build_nodes(self):
        self.nid = np.arange(self.nr * self.nx).reshape(self.nr, self.nx)
        self.n_base = self.nr * self.nx
        # 界面只在螺套埋深范围内存在；x = L_INS 处不复制，天然形成闭合的裂尖
        self.cmask = self.x < L_INS - 1e-9
        self.i_if = []
        self.dup = {}
        nxt = self.n_base
        for c in self.ifaces:
            i = int(np.argmin(np.abs(self.r - c.r)))
            assert abs(self.r[i] - c.r) < 1e-6, f'界面半径 {c.r} 不在网格线上'
            self.i_if.append(i)
            d = -np.ones(self.nx, np.int64)
            k = int(self.cmask.sum())
            d[self.cmask] = nxt + np.arange(k)
            nxt += k
            self.dup[i] = d
        self.n_node = nxt
        self.ndof = 2 * self.n_node + 1          # 末位：单胞边界的统一径向位移
        self.i_ur = 2 * self.n_node
        R, X = np.meshgrid(self.r, self.x, indexing='ij')
        self.R = np.concatenate([R.ravel()] +
                                [np.full(int(self.cmask.sum()), self.r[i])
                                 for i in self.i_if])
        self.X = np.concatenate([X.ravel()] +
                                [self.x[self.cmask] for _ in self.i_if])

    def _node(self, ir, ix, outside):
        """outside=True 表示取界面外侧那一份。"""
        if outside and ir in self.dup and self.dup[ir][ix] >= 0:
            return self.dup[ir][ix]
        return self.nid[ir, ix]

    # ---------------------------------------------------------- 单元
    def _mat_of(self, rc, xc):
        if xc >= L_INS:
            return 'lam'
        if rc < R_BORE:
            return 'void' if xc < X_BORE else 'steel'
        if rc < R_STEEL:
            return 'steel'
        if rc < R_TRANS:
            return 'trans'
        if rc < R_WRAP:
            return 'wrap'
        if rc < R_BLOCK:
            return 'block'
        return 'lam'

    def _build_elems(self):
        rc = 0.5 * (self.r[:-1] + self.r[1:])
        xc = 0.5 * (self.x[:-1] + self.x[1:])
        el, tag = [], []
        for i in range(self.nr - 1):
            for k in range(self.nx - 1):
                mt = self._mat_of(rc[i], xc[k])
                nd = []
                for di, dk in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    ir = i + di
                    nd.append(self._node(ir, k + dk, outside=(rc[i] > self.r[ir])))
                el.append(nd)
                tag.append(MATS.index(mt))
        self.el = np.array(el, np.int64)
        self.tag = np.array(tag, np.int8)
        self.rc, self.xc = rc, xc
        self.el_rc = np.array([self.R[e].mean() for e in self.el])
        self.el_xc = np.array([self.X[e].mean() for e in self.el])

    @staticmethod
    def _shape(xi, et):
        N = 0.25 * np.array([(1 - xi) * (1 - et), (1 + xi) * (1 - et),
                             (1 + xi) * (1 + et), (1 - xi) * (1 + et)])
        dNx = 0.25 * np.array([-(1 - et), (1 - et), (1 + et), -(1 + et)])
        dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        return N, dNx, dNe

    def _B(self, nd, xi, et):
        rr, xx = self.R[nd], self.X[nd]
        N, dNx, dNe = self._shape(xi, et)
        J = np.array([[dNx @ rr, dNx @ xx], [dNe @ rr, dNe @ xx]])
        dJ = np.linalg.det(J)
        if dJ <= 0:
            return None, 0.0, 0.0
        dN = np.linalg.solve(J, np.vstack([dNx, dNe]))
        rg = float(N @ rr)
        B = np.zeros((4, 8))
        B[0, 0::2] = dN[0]
        B[1, 1::2] = dN[1]
        B[2, 0::2] = N / max(rg, 1e-9)
        B[3, 0::2] = dN[1]
        B[3, 1::2] = dN[0]
        return B, dJ, rg

    def _assemble(self):
        gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
        Ds = [_D(LAYERS[m]) for m in MATS]
        rows, cols, vals = [], [], []
        for nd, tg in zip(self.el, self.tag):
            D = Ds[tg]
            ke = np.zeros((8, 8))
            for xi in gp:
                for et in gp:
                    B, dJ, rg = self._B(nd, xi, et)
                    if B is None:
                        continue
                    ke += B.T @ D @ B * dJ * 2 * np.pi * rg
            dof = np.stack([2 * nd, 2 * nd + 1], 1).ravel()
            rows.append(np.repeat(dof, 8))
            cols.append(np.tile(dof, 8))
            vals.append(ke.ravel())
        self.Kb = sp.coo_matrix((np.concatenate(vals),
                                 (np.concatenate(rows), np.concatenate(cols))),
                                shape=(self.ndof, self.ndof)).tocsr()

    # ---------------------------------------------------------- 内聚力界面
    def _build_coh(self):
        ix = np.flatnonzero(self.cmask)
        xi = self.x[ix]
        tri = np.zeros(len(ix))
        tri[:-1] += 0.5 * np.diff(xi)
        tri[1:] += 0.5 * np.diff(xi)
        tri[-1] += 0.5 * (L_INS - xi[-1])          # 尖端半格
        self.coh_x = xi
        self.coh = []
        for c, i in zip(self.ifaces, self.i_if):
            din = np.stack([2 * self.nid[i, ix], 2 * self.nid[i, ix] + 1], 1)
            dout = np.stack([2 * self.dup[i][ix], 2 * self.dup[i][ix] + 1], 1)
            self.coh.append(dict(c=c, din=din, dout=dout,
                                 A=2 * np.pi * c.r * tri,
                                 d=np.zeros(len(ix)),
                                 dt=np.zeros(len(ix)),
                                 cut=np.zeros(len(ix), bool)))

    def reset(self):
        """回到无损伤的初始状态。"""
        for h in self.coh:
            # 必须换新数组而不是原地赋零：solve() 返回的状态快照里的 d
            # 与 h['dt'] 是同一个对象，原地清零会把已经取回的结果一起抹掉。
            h['d'] = np.zeros(len(self.coh_x))      # 已收敛（已提交）的损伤
            h['dt'] = np.zeros(len(self.coh_x))     # 当前迭代的试探损伤

    def clear_cut(self):
        """取消人为切口。切口是模型设定不是状态，故 reset() 不碰它。"""
        for h in self.coh:
            h['cut'][:] = False

    def commit(self):
        """载荷步收敛后提交损伤。损伤的不可逆性只在载荷步之间成立，
        在同一步的迭代之内不成立——否则未收敛的中间迭代会把伪损伤锁死。"""
        for h in self.coh:
            h['d'] = np.maximum(h['d'], h['dt'])

    def cut(self, which, x_range):
        """人工切开一段界面（用于柔度法求能量释放率，绕过内聚力）。"""
        for h in self.coh:
            if h['c'].name.startswith(which):
                lo, hi = x_range
                h['cut'] = (self.coh_x >= lo) & (self.coh_x <= hi)

    def _coh_update(self, u):
        out = []
        for h in self.coh:
            c = h['c']
            dn = u[h['dout'][:, 0]] - u[h['din'][:, 0]]
            ds = u[h['dout'][:, 1]] - u[h['din'][:, 1]]
            dnp = np.maximum(dn, 0.0)
            lam = np.hypot(dnp, ds)
            beta2 = ds ** 2 / np.maximum(dnp ** 2, 1e-24)
            gII = beta2 / (1.0 + beta2)
            quad = (dnp * c.K / c.tn0) ** 2 + (np.abs(ds) * c.K / c.ts0) ** 2
            d0 = np.where(quad > 1e-30, lam / np.sqrt(np.maximum(quad, 1e-30)),
                          c.tn0 / c.K)
            d0 = np.clip(d0, 1e-9, None)
            Gc = c.GIc + (c.GIIc - c.GIc) * gII ** c.eta
            df = np.maximum(2.0 * Gc / (c.K * d0), d0 * 1.0001)
            dnew = np.where(lam <= d0, 0.0,
                            np.clip(df * (lam - d0) /
                                    np.maximum(lam * (df - d0), 1e-30), 0.0, 1.0))
            # 与"上一个已收敛载荷步"的损伤取大，而不是与本步迭代的运行最大值取大
            d = np.maximum(h['d'], dnew)
            d = np.where(h['cut'], 1.0, d)
            h['dt'] = d
            out.append((dn, ds, d, lam, d0, df, Gc))
        return out

    def _Kcoh(self, state):
        rows, cols, vals = [], [], []
        for h, (dn, ds, d, *_ ) in zip(self.coh, state):
            c = h['c']
            kn = np.where(dn >= 0.0, (1.0 - d) * c.K, c.K) * h['A']
            ks = (1.0 - d) * c.K * h['A']
            kn = np.maximum(kn, 1e-6 * c.K * h['A'])
            ks = np.maximum(ks, 1e-6 * c.K * h['A'])
            for j, k in ((0, kn), (1, ks)):
                a, b = h['din'][:, j], h['dout'][:, j]
                rows.append(np.concatenate([a, a, b, b]))
                cols.append(np.concatenate([a, b, a, b]))
                vals.append(np.concatenate([k, -k, -k, k]))
        return sp.coo_matrix((np.concatenate(vals),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.ndof, self.ndof)).tocsr()

    # ---------------------------------------------------------- 边界与载荷
    def _build_bc(self):
        # 螺柱：啮合段孔壁结点连到刚性地（轮毂），总刚度 K_STUD
        ithr = int(np.argmin(np.abs(self.r - R_BORE)))
        kx = np.flatnonzero(self.x <= L_ENG + 1e-9)
        w = np.zeros(len(kx))
        xs = self.x[kx]
        w[:-1] += 0.5 * np.diff(xs)
        w[1:] += 0.5 * np.diff(xs)
        self.stud_dof = 2 * self.nid[ithr, kx] + 1
        self.stud_w = w / w.sum()
        self.stud_k = self.stud_w * K_STUD

        # 端面承压环带：r ∈ [R_STEEL, R_CELL]（螺套端面下沉，不参与承压）
        ir = np.flatnonzero(self.r >= R_STEEL - 1e-9)
        rr = self.r[ir]
        a = np.zeros(len(ir))
        a[:-1] += np.pi * (0.5 * (rr[:-1] + rr[1:]) ** 2 - rr[:-1] ** 2)
        a[1:] += np.pi * (rr[1:] ** 2 - 0.5 * (rr[:-1] + rr[1:]) ** 2)
        self.face_dof = 2 * self.nid[ir, 0] + 1
        self.face_A = a

        # 远端截面：均布轴向牵引，合力 = F_A
        ir2 = np.arange(self.nr)
        rr2 = self.r
        a2 = np.zeros(self.nr)
        a2[:-1] += np.pi * (0.5 * (rr2[:-1] + rr2[1:]) ** 2 - rr2[:-1] ** 2)
        a2[1:] += np.pi * (rr2[1:] ** 2 - 0.5 * (rr2[:-1] + rr2[1:]) ** 2)
        self.far_dof = 2 * self.nid[ir2, -1] + 1
        self.far_w = a2 / a2.sum()

        # 轴上 u_r = 0
        self.axis_dof = 2 * self.nid[0, :]

    def _build_T(self):
        """把单胞外边界的 u_r 并到一个主自由度上（广义周期），并消去轴上的 u_r。"""
        col = np.arange(self.ndof)
        out_r = 2 * self.nid[-1, :]
        col[out_r] = self.i_ur
        keep = np.setdiff1d(np.arange(self.ndof),
                            np.concatenate([out_r, self.axis_dof, [self.i_ur]]))
        newcol = -np.ones(self.ndof, np.int64)
        newcol[keep] = np.arange(len(keep))
        newcol[out_r] = len(keep)
        newcol[self.i_ur] = len(keep)
        self.nred = len(keep) + 1
        row = np.flatnonzero(newcol >= 0)
        self.T = sp.coo_matrix((np.ones(len(row)), (row, newcol[row])),
                               shape=(self.ndof, self.nred)).tocsr()
        self.newcol = newcol

    # ---------------------------------------------------------- 求解
    def _step(self, F_M, F_A, u0=None, act0=None, max_out=15, max_in=40,
              tol=1e-9, relax=0.6):
        """单个载荷步：主动集（端面接触）外循环 + 内聚力割线迭代内循环。
        不提交损伤——提交由调用者在收敛后用 commit() 完成。"""
        Kstud = sp.coo_matrix((self.stud_k, (self.stud_dof, self.stud_dof)),
                              shape=(self.ndof, self.ndof)).tocsr()
        f0 = np.zeros(self.ndof)
        f0[self.stud_dof] -= self.stud_k * (F_M / K_STUD)     # 预紧：−x 方向
        f0[self.far_dof] += self.far_w * F_A                  # 远端膜力：+x
        act = np.ones(len(self.face_dof), bool) if act0 is None else act0.copy()
        u = np.zeros(self.ndof) if u0 is None else u0.copy()
        K = None
        err = np.nan
        for io in range(max_out):
            for ii in range(max_in):
                st = self._coh_update(u)
                K = (self.Kb + self._Kcoh(st) + Kstud)
                Kr = (self.T.T @ K @ self.T).tocsc()
                fr = self.T.T @ f0
                fix = np.unique(self.newcol[self.face_dof[act]])
                fix = fix[fix >= 0]
                free = np.setdiff1d(np.arange(self.nred), fix)
                ur = np.zeros(self.nred)
                ur[free] = spla.spsolve(Kr[free][:, free], fr[free])
                un = self.T @ ur
                err = np.max(np.abs(un - u)) / max(np.max(np.abs(un)), 1e-12)
                u = relax * u + (1.0 - relax) * un if ii < 2 else un
                if err < tol and ii >= 2:
                    break
            R = K @ u - f0
            Rf = R[self.face_dof]
            new = act.copy()
            new[act] = Rf[act] >= -1e-3          # 支反力必须为 +x（受压）才保持接触
            new[~act] = u[self.face_dof][~act] < -1e-12   # 穿透则重新闭合
            if np.array_equal(new, act):
                break
            act = new
        st = self._coh_update(u)
        F_S = float(np.sum(self.stud_k * (F_M / K_STUD + u[self.stud_dof])))
        F_KR = float(np.sum(R[self.face_dof][act])) if act.any() else 0.0
        return dict(u=u, F_S=F_S, F_KR=F_KR, F_A=F_A, F_M=F_M,
                    state=st, act=act.copy(), it_out=io + 1, err=err,
                    open_frac=float(1.0 - act.sum() / len(act)))

    def solve(self, F_M=FM_DEFAULT, F_A=0.0, n_pre=4, n_load=8, keep=False,
              verbose=False):
        """从无载状态出发，先增量加预紧，再增量加外载。每步收敛后提交损伤。

        增量是必须的：内聚力损伤是路径相关的，一次加满会让中间迭代
        经过并不存在的状态，把伪损伤锁进结果里。"""
        if not keep:
            self.reset()
        r = None
        u0 = act0 = None
        for k in range(1, n_pre + 1):
            r = self._step(F_M * k / n_pre, 0.0, u0, act0)
            self.commit()
            u0, act0 = r['u'], r['act']
        if F_A != 0.0:
            for k in range(1, n_load + 1):
                r = self._step(F_M, F_A * k / n_load, u0, act0)
                self.commit()
                u0, act0 = r['u'], r['act']
                if verbose:
                    print('    F_A=%7.1f kN  F_S=%7.1f  F_KR=%7.1f  dmax=%.3f'
                          % (F_A * k / n_load / 1e3, r['F_S'] / 1e3,
                             r['F_KR'] / 1e3, self.dmax()))
        return r

    def dmax(self):
        return float(max(h['d'].max() for h in self.coh))

    def calibrate_FM(self, target_FS=FM_DEFAULT, tol=200.0, it=8):
        """按目标安装轴力反解需要的螺柱预拉伸。现场是拧到目标轴力，
        不是拧到目标伸长，所以标定这一步不能省。"""
        FM = target_FS
        for _ in range(it):
            self.reset()
            r = self._step(FM, 0.0)
            self.commit()
            e = r['F_S'] - target_FS
            if abs(e) < tol:
                break
            FM -= e * 1.05
        self.reset()
        return float(FM)

    # ---------------------------------------------------------- 后处理
    def tractions(self, res):
        """各界面的法向/切向牵引力沿 x 的分布 [MPa]。"""
        out = {}
        for h, (dn, ds, d, *_ ) in zip(self.coh, res['state']):
            c = h['c']
            tn = np.where(dn >= 0, (1 - d) * c.K * dn, c.K * dn)
            ts = (1 - d) * c.K * ds
            # 一律返回副本：d 与 h['dt'] 同源，直接外传会被后续 reset() 抹掉
            out[c.name] = dict(x=self.coh_x.copy(), tn=tn, ts=ts,
                               d=np.array(d, copy=True),
                               dn=np.array(dn, copy=True),
                               ds=np.array(ds, copy=True))
        return out

    def debond(self, res, thr=0.99):
        """各界面完全损伤区的范围与长度。"""
        out = {}
        for h in self.coh:
            m = h['d'] > thr
            if m.any():
                lo, hi = float(self.coh_x[m].min()), float(self.coh_x[m].max())
                out[h['c'].name] = (lo, hi, hi - lo)
            else:
                out[h['c'].name] = (np.nan, np.nan, 0.0)
        return out

    def stresses(self, u):
        """单元中心应力 (σr, σx, σθ, τrx) 与 von Mises。"""
        Ds = [_D(LAYERS[m]) for m in MATS]
        S = np.zeros((len(self.el), 5))
        for e, (nd, tg) in enumerate(zip(self.el, self.tag)):
            B, dJ, rg = self._B(nd, 0.0, 0.0)
            if B is None:
                continue
            dof = np.stack([2 * nd, 2 * nd + 1], 1).ravel()
            s = Ds[tg] @ (B @ u[dof])
            vm = np.sqrt(0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 +
                                (s[2] - s[0]) ** 2) + 3 * s[3] ** 2)
            S[e] = [s[0], s[1], s[2], s[3], vm]
        return S

    def steel_force(self, u):
        """钢衬套截面轴力沿 x [N]。"""
        m = (self.tag == MATS.index('steel'))
        S = self.stresses(u)[:, 1]
        F = np.zeros(len(self.xc))
        for e in np.flatnonzero(m):
            i, k = divmod(e, self.nx - 1)
            a = np.pi * (self.r[i + 1] ** 2 - self.r[i] ** 2)
            F[k] += S[e] * a
        return self.xc, F

    def compliance(self, F_M=0.0, F_A=1.0e5):
        """在无预紧、仅远端拉力下测柔度，用于能量释放率的柔度法。"""
        r = self.solve(F_M=F_M, F_A=F_A)
        u = r['u']
        C = float(np.sum(self.far_w * F_A * u[self.far_dof])) / F_A ** 2
        return C, r


# ------------------------------------------------------------------ 自检
def selfcheck(M: InsertAxi, FM):
    print('\n[自检]')
    A_cell = np.pi * R_CELL ** 2
    print('  单胞面积 %.0f mm2，节距×壁厚 %.0f mm2，差 %.2f%%'
          % (A_cell, P_PITCH * T_WALL, 100 * (A_cell / (P_PITCH * T_WALL) - 1)))
    r = M.solve(F_M=FM, F_A=0.0)
    print('  仅预紧：F_S = %.1f kN（目标 %.1f），F_KR = %.1f kN，力流平衡残差 %.3f kN'
          % (r['F_S'] / 1e3, FM_DEFAULT / 1e3, r['F_KR'] / 1e3,
             (r['F_S'] - r['F_KR'] - r['F_A']) / 1e3))
    tr = M.tractions(r)
    for k, v in tr.items():
        i = int(np.argmax(np.abs(v['ts'])))
        print('  界面 %-14s 预紧下切向牵引峰值 %6.2f MPa @ x = %5.1f mm，'
              '损伤峰值 %.3f' % (k, v['ts'][i], v['x'][i], v['d'].max()))
    return r


def main():
    print('=' * 96)
    print('M2′  预埋螺套单胞轴对称分层内聚力模型')
    print('=' * 96)
    M = InsertAxi()
    FM = M.calibrate_FM(FM_DEFAULT)
    print('\n[预紧标定] 达到安装轴力 %.0f kN 需要的螺柱预拉伸对应力 F_M = %.1f kN'
          % (FM_DEFAULT / 1e3, FM / 1e3))
    selfcheck(M, FM)

    print('\n[载荷系数 Φ] 由模型解出，不作假定')
    print('  %-10s %9s %9s %9s %9s %9s' %
          ('外载 F_A', 'F_S', 'F_KR', 'Φ', '端面张开', '界面损伤'))
    FS0 = M.solve(F_M=FM, F_A=0.0)['F_S']
    for FA in (0.0, 74e3, 159e3, 250e3, 339e3, 420e3, 480e3):
        r = M.solve(F_M=FM, F_A=FA)
        phi = (r['F_S'] - FS0) / FA if FA > 0 else float('nan')
        print('  %7.0f kN %8.1f %9.1f %9.4f %8.1f%% %9.3f'
              % (FA / 1e3, r['F_S'] / 1e3, r['F_KR'] / 1e3, phi,
                 100 * r['open_frac'], M.dmax()))
    print('\n  Φ 为载荷系数，F_A,open = F_M/(1−Φ) = %.0f kN'
          % (FM / (1 - 0.14) / 1e3))


if __name__ == '__main__':
    main()
