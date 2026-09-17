# -*- coding: utf-8 -*-
"""叶根预埋螺套：三维结构与细节图（纯几何，不含任何仿真结果）。

几何自 config/ 下的配置读入（见 geometry.py）。凡不由基本尺寸直接确定、
而由工艺常识推断的尺寸，代码里用 [推断] 标出，图面上也写明"推断"，两者不混。

输出（全部写进 figs/，文件名一律 fig3d_ 前缀，不碰已有的 fig01..fig18）：
    fig3d_01_root_ring.png      叶根整环三维总览
    fig3d_02_cutaway.png        壁厚剖切（露出 3 个螺套全长）
    fig3d_03_layers.png         单个螺套的五层构造
    fig3d_04_fiber_layout.png   光纤敷设三维示意

渲染方式：mpl_toolkits.mplot3d 的 Poly3DCollection，但深度关系由本文件自己管：

  1) ax.set_proj_type('ortho')  —— 正交投影下，投影深度与"面心到视点的距离"是同
     一个排序，不会出现近大远小带来的排序反转；
  2) 一张图里所有实体面片放进 **同一个** Poly3DCollection。mplot3d 只在集合内部
     排序，跨集合按整体深度排，拆成多个集合必然出现前后穿插；
  3) 面片交给 matplotlib 之前先由 Scene._order() 按"面心到视点的欧氏距离"降序排
     好（远的先画）。matplotlib 内部的 sorted() 稳定，共面面片（端面上叠放的螺套
     截面等）因此保持我们指定的先后；
  4) ax.computed_zorder = False，线条/文字的前后由显式 zorder 决定。

比例：fit_box() 按几何真实包围盒设 set_box_aspect，不做任何拉伸；fit_axes() 再把
坐标轴矩形调成该包围盒在当前视角下的投影宽高比，使图形填满画面且不变形。
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11
rcParams['savefig.dpi'] = 200
rcParams['figure.dpi'] = 200

# ----------------------------------------------------------------- 项目色板
INK = '#12212C'
STEEL = '#2F6690'
SIGNAL = '#D94F2B'
AMBER = '#C8871B'
SAFE = '#1E7A5A'
GREY = '#5A6B77'

STEEL_L = '#4E86AD'      # 钢体剖切面（剖面惯例用浅一号）
STEEL_D = '#1F4763'      # 钢体端面 / 暗面
LAM = '#A9C4D6'          # 叶根层压
LAM_D = '#8AACC4'        # 层压暗面
LAM_F = '#C5D8E4'        # 层压亮面
BLOCK = '#7C8B96'        # 拉挤 GFRP 块
HAIR = '#C9D4DC'

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figs')

# --------------------------------------------------------------- 几何常量 mm
from geometry import Geom
_G = Geom()                        # 几何自 config/ 读入，换机型只改配置

N_BOLT = _G.n_bolt                 # 螺栓数
D_BC = _G.D_bc                     # 螺栓圆直径
R_BC = _G.R_bc
D_OUT, D_IN = _G.D_out, _G.D_in    # 叶根外径 / 内径
R_OUT, R_IN = D_OUT / 2, D_IN / 2
T_WALL = _G.t_wall
PITCH = _G.pitch                   # 螺栓节距
DTH = 2 * np.pi / N_BOLT           # 相邻螺套的圆心角
R_AX = R_BC - R_IN                 # 螺套轴线距内表面

D_INS, RHO_S = _G.D_ins, _G.D_ins / 2   # 螺套外径 / 半径
L_INS = _G.L_ins                   # 埋深
D_BORE, RHO_B = _G.d_bore, _G.d_bore / 2   # 螺纹小径孔
L_ENG = _G.l_eng                   # 螺纹啮合长度
X_BLIND = L_ENG + 25.0             # 盲孔底【推断：啮合段再进 25】
D_STUD, RHO_D = 42.0, 21.0         # 双头螺柱 M42
P_THD = 4.5                        # M42 粗牙螺距 [推断]

RHO_TR = _G.D_ins / 2 + 0.5        # 过渡层外半径【推断：钢体 + 0.5】
RHO_W = _G.D_ins / 2 + 6.0         # 缠绕层外半径【推断：钢体外径 + 6】
RHO_W_RING = PITCH / 2.0           # 整环图中按节距截断
GRV_D, GRV_P, GRV_W = 1.5, 8.0, 4.8   # 环向沟槽 深/节距/宽 [推断]

GAP_NET = PITCH - D_INS            # 相邻螺套钢体净间距
COV_IN = R_AX - RHO_W              # 缠绕层下缘到内表面
COV_S_IN = R_AX - RHO_S            # 钢体下缘到内表面

X_FIB = _G.L_ins + 30.0            # 光纤轴向覆盖上限【推断：埋深再延 30】
DX_RES = 1.3                       # OFDR 空间分辨率
P_IN = 2 * np.pi * R_IN / N_BOLT   # 内表面上的螺栓间距
C_IN = 2 * np.pi * R_IN            # 内表面周长


def _sp(v):
    """1234567 -> '1 234 567'，与图上原有写法一致。"""
    return format(int(round(v)), ',d').replace(',', ' ')
L_LOOP = 2 * np.pi * R_IN          # 9691.9 内表面绕一圈长度


# =========================================================== 三维渲染基础设施
def grid_quads(P):
    """(n+1, m+1, 3) 结点网格 → (n*m, 4, 3) 四边形面片。"""
    P = np.asarray(P, float)
    a, b = P[:-1, :-1], P[1:, :-1]
    c, d = P[1:, 1:], P[:-1, 1:]
    return np.stack([a, b, c, d], axis=2).reshape(-1, 4, 3)


def strip_quads(A, B):
    """两条等长折线 A、B（各 (n,3)）之间张成的条带。"""
    A, B = np.asarray(A, float), np.asarray(B, float)
    return np.stack([A[:-1], A[1:], B[1:], B[:-1]], axis=1)


def polyline(pts, color, lw, zorder=8, ls='solid'):
    """把一串三维点画成折线（Line3DCollection，前后由 zorder 决定）。"""
    p = np.asarray(pts, float)
    return Line3DCollection(list(np.stack([p[:-1], p[1:]], axis=1)),
                            colors=color, linewidths=lw, zorder=zorder,
                            linestyles=ls)


class Scene:
    """收集面片 → 自己做画家算法深度排序 → 一个 Poly3DCollection。"""

    def __init__(self):
        self.q, self.fc, self.ec = [], [], []

    def add(self, quads, fc, ec='none', alpha=1.0, ealpha=1.0):
        quads = np.asarray(quads, float)
        if quads.ndim == 2:
            quads = quads[None]
        if len(quads) == 0:
            return
        self.q.append(quads)
        n = len(quads)
        self.fc += [to_rgba(fc, alpha)] * n
        self.ec += [(0, 0, 0, 0) if ec == 'none' else to_rgba(ec, ealpha)] * n

    @property
    def pts(self):
        return np.concatenate(self.q, 0).reshape(-1, 3)

    def _order(self, ax, Q):
        """面心到视点的距离，降序（远 → 近）。距离在"归一化盒"里算：
        set_box_aspect 之后，屏幕上的深度就是盒坐标里的深度。"""
        lo = np.array([ax.get_xlim3d()[0], ax.get_ylim3d()[0], ax.get_zlim3d()[0]])
        hi = np.array([ax.get_xlim3d()[1], ax.get_ylim3d()[1], ax.get_zlim3d()[1]])
        ba = np.asarray(ax.get_box_aspect(), float)
        ba = ba / ba.max()
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        _, _, d = view_basis(ax)
        eye = 0.5 * ba + 60.0 * d               # 正交投影：视点放到很远
        C = Q.mean(axis=1)                      # 面心
        N = (C - lo) / span * ba
        return np.argsort(-np.linalg.norm(N - eye, axis=1))

    def commit(self, ax, lw=0.35, zorder=4):
        Q = np.concatenate(self.q, 0)
        fc, ec = np.array(self.fc), np.array(self.ec)
        k = self._order(ax, Q)
        pc = Poly3DCollection(Q[k], facecolors=fc[k], edgecolors=ec[k],
                              linewidths=lw, zorder=zorder)
        # 正交投影下 mean(投影 z) 即面心深度，与上面的判据一致
        pc.set_zsort('average')
        ax.add_collection3d(pc)
        return pc


def view_basis(ax):
    """当前视角的屏幕基：(右, 上, 指向视点)，与 proj3d._view_axes 一致。"""
    e = np.deg2rad(ax.elev)
    a = np.deg2rad(ax.azim)
    r = np.deg2rad(getattr(ax, 'roll', 0.0) or 0.0)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u = np.array([-np.sin(a), np.cos(a), 0.0])
    v = np.cross(d, u)
    return (np.cos(r) * u - np.sin(r) * v,
            np.sin(r) * u + np.cos(r) * v, d)


def roll_horiz(elev, azim, t, flip=False):
    """求 roll：使三维方向 t 在屏幕上水平。flip=True 则反向 180°。"""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u = np.array([-np.sin(a), np.cos(a), 0.0])
    v = np.cross(d, u)
    t = np.asarray(t, float)
    al = np.arctan2(float(t @ v), float(t @ u))
    return -np.rad2deg(al) + (180.0 if flip else 0.0)


def pick_roll(elev, azim, pts, prefer_up=None, prefer_right=None, w=0.30):
    """扫描 roll，取投影包围盒最接近正方形的那个。mplot3d 的 3D 视口恒为正方形
    （取轴矩形宽高的较小者），细长物体摆成对角线才能画得最大。prefer_up /
    prefer_right 只用来在几个等效解之间决定朝向，不改变比例。"""
    P = np.asarray(pts, float).reshape(-1, 3)
    lo, hi = P.min(0), P.max(0)
    C = np.array([[hi[i] if b[i] else lo[i] for i in range(3)]
                  for b in itertools.product([0, 1], repeat=3)], float)
    C = C - C.mean(0)
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u0 = np.array([-np.sin(a), np.cos(a), 0.0])
    v0 = np.cross(d, u0)
    best = (1e9, 0.0)
    for r in np.arange(0.0, 360.0, 1.0):
        rr = np.deg2rad(r)
        u = np.cos(rr) * u0 - np.sin(rr) * v0
        v = np.sin(rr) * u0 + np.cos(rr) * v0
        su, sv = C @ u, C @ v
        W = su.max() - su.min()
        H = sv.max() - sv.min()
        sc = max(W / H, H / W)
        if prefer_up is not None:
            sc -= w * float(np.asarray(prefer_up, float) @ v)
        if prefer_right is not None:
            sc -= w * float(np.asarray(prefer_right, float) @ u)
        if sc < best[0]:
            best = (sc, r)
    return best[1]


def new3d(figsize, rect=(0.0, 0.0, 1.0, 1.0), elev=20.0, azim=-124.0,
          roll=0.0, fig=None):
    if fig is None:
        fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_axes(rect, projection='3d')
    ax.set_proj_type('ortho')
    ax.computed_zorder = False
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()
    ax.patch.set_alpha(0.0)
    return fig, ax


def fit_box(ax, pts, pad=0.01):
    """按几何真实包围盒设范围与 box_aspect —— 不失真。"""
    pts = np.asarray(pts, float).reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    lo, hi = lo - pad * span, hi + pad * span
    ax.set_xlim3d(lo[0], hi[0])
    ax.set_ylim3d(lo[1], hi[1])
    ax.set_zlim3d(lo[2], hi[2])
    ax._raw_aspect = hi - lo
    ax.set_box_aspect(hi - lo)


def _corners(ax):
    lo = np.array([ax.get_xlim3d()[0], ax.get_ylim3d()[0], ax.get_zlim3d()[0]])
    hi = np.array([ax.get_xlim3d()[1], ax.get_ylim3d()[1], ax.get_zlim3d()[1]])
    return np.array([[hi[i] if b[i] else lo[i] for i in range(3)]
                     for b in itertools.product([0, 1], repeat=3)], float)


def fill3d(fig, ax, cx, cy, side_in, fill=0.95, verbose=False):
    """mplot3d 的 3D 视口恒为正方形（取轴矩形宽高的较小者），所以这里先把轴矩形
    设成边长 side_in 英寸、中心在图坐标 (cx, cy) 的正方形，再用 set_box_aspect
    的 zoom 把几何体放大到刚好填满该视口。几何体本身的比例始终由 _raw_aspect
    决定，zoom 只做整体缩放，不引入失真。"""
    fw, fh = fig.get_size_inches()
    sx, sy = side_in / fw, side_in / fh
    ax.set_position([cx - 0.5 * sx, cy - 0.5 * sy, sx, sy])
    fig.canvas.draw()
    bb = ax.get_window_extent()
    inv = ax.transData.inverted()
    c0 = np.array(inv.transform((bb.x0, bb.y0)), float)
    c1 = np.array(inv.transform((bb.x1, bb.y1)), float)
    vx, vy = np.abs(c1 - c0)
    P = np.array([p2(ax, c) for c in _corners(ax)])
    W = P[:, 0].max() - P[:, 0].min()
    H = P[:, 1].max() - P[:, 1].min()
    z = fill * min(vx / W, vy / H)
    ax.set_box_aspect(ax._raw_aspect, zoom=z)
    fig.canvas.draw()
    if verbose:
        print('    fill3d: 投影宽高比 %.2f, zoom %.2f' % (W / H, z))


def p2(ax, p):
    """三维点 → 该 3D 轴的二维数据坐标（用于精确放置标注/引线）。"""
    x, y, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
    return x, y


def lab(ax, p, s, dx, dy, ha='left', va='center', color=INK, fs=11.0,
        arrow=True, box=True, acol=None, weight='normal', lw=0.9, alpha=0.9):
    """三维锚点上挂一条引线 + 文字（文字位置用 points 偏移，不会飘）。"""
    ap = dict(arrowstyle='-', lw=lw, color=acol or color,
              shrinkA=0.0, shrinkB=1.5) if arrow else None
    bb = dict(boxstyle='round,pad=0.26', fc='white', ec='none',
              alpha=alpha) if box else None
    ax.annotate(s, xy=p2(ax, p), xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va, color=color, fontsize=fs, zorder=40,
                arrowprops=ap, bbox=bb, fontweight=weight)


def dim3(ax, p, q, s, dx=0, dy=0, color=INK, fs=10.0, ha='center', va='center'):
    """两个三维点之间的双箭头尺寸线 + 文字。"""
    a, b = p2(ax, p), p2(ax, q)
    ax.annotate('', xy=a, xytext=b, zorder=40,
                arrowprops=dict(arrowstyle='<->', lw=1.0, color=color,
                                shrinkA=0.0, shrinkB=0.0))
    mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    ax.annotate(s, xy=mid, xytext=(dx, dy), textcoords='offset points',
                ha=ha, va=va, color=color, fontsize=fs, zorder=41,
                bbox=dict(boxstyle='round,pad=0.20', fc='white', ec='none',
                          alpha=0.92))


def tbox(fig, x, y, s, fs=11.0, color=INK, ha='left', va='top',
         fc='#F4F7F9', ec=HAIR, weight='normal'):
    return fig.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, zorder=50,
                    fontweight=weight, linespacing=1.5,
                    bbox=dict(boxstyle='round,pad=0.55', fc=fc, ec=ec, lw=0.9))


def swatches(fig, x, y, rows, fs=11.0, dy=0.030, sw=0.011, sh=0.0155):
    """图例色块（图坐标）。rows = [(颜色, 文字), ...]"""
    for i, (c, t) in enumerate(rows):
        yy = y - i * dy
        fig.add_artist(Rectangle((x, yy - 0.004), sw, sh, fc=c, ec=INK,
                                 lw=0.7, transform=fig.transFigure, zorder=51))
        fig.text(x + sw + 0.008, yy + 0.0035, t, fontsize=fs, color=INK,
                 va='center', ha='left', zorder=51)


# =============================================================== 环 / 螺套映射
def wall(x, th, r, z0=0.0):
    """叶根壁上的点：x 轴向，th 周向角，r 自内表面起的径向坐标。"""
    x, th, r = np.broadcast_arrays(np.asarray(x, float), np.asarray(th, float),
                                   np.asarray(r, float))
    R = R_IN + r
    return np.stack([x, R * np.sin(th), R * np.cos(th) - z0], axis=-1)


def ins_frame(th, z0=0.0):
    c = np.array([(R_IN + R_AX) * np.sin(th), (R_IN + R_AX) * np.cos(th) - z0])
    return c, np.array([np.cos(th), -np.sin(th)]), np.array([np.sin(th),
                                                             np.cos(th)])


def ins(x, u, v, th, z0=0.0):
    """螺套局部坐标 (x 轴向, u 周向, v 径向) → 全局 xyz。真圆截面。"""
    c, eu, ev = ins_frame(th, z0)
    x, u, v = np.broadcast_arrays(np.asarray(x, float), np.asarray(u, float),
                                  np.asarray(v, float))
    return np.stack([x, c[0] + u * eu[0] + v * ev[0],
                     c[1] + u * eu[1] + v * ev[1]], axis=-1)


def cyl(xs, phis, rho, th, z0=0.0):
    """螺套同轴圆柱面：xs 轴向采样，phis 截面角（0 = 径向外）。"""
    xs = np.asarray(xs, float)
    phis = np.asarray(phis, float)
    X = xs[:, None] + 0 * phis[None, :]
    PH = phis[None, :] + 0 * xs[:, None]
    rho = np.asarray(rho, float)
    if rho.ndim == 1:
        rho = rho[:, None]
    return ins(X, rho * np.sin(PH), rho * np.cos(PH), th, z0)


def annulus(x, phis, r0, r1, th, z0=0.0):
    """垂直于轴的环面（端面 / 端盖）。r0=0 会退化成扇形三角，统一给 0.35 针孔。"""
    r0 = max(float(r0), 0.35)
    P = np.stack([ins(x, r * np.sin(phis), r * np.cos(phis), th, z0)
                  for r in (r0, r1)], axis=0)
    return grid_quads(P)


def groove_rho(x, rho=RHO_S, d=GRV_D, p=GRV_P, w=GRV_W):
    """钢衬套外表面的环向沟槽轮廓（升余弦槽 + 平台）[推断]。"""
    u = np.mod(np.asarray(x, float), p) - 0.5 * p
    g = np.where(np.abs(u) < 0.5 * w, 0.5 * (1 + np.cos(2 * np.pi * u / w)), 0.0)
    return rho - d * g


# ================================================== 图 1  叶根整环三维总览
def fig_ring():
    X1 = L_INS + 130.0
    ELEV, AZIM = 22.0, -119.0
    SQ = (0.345, 0.450, 9.30)          # 正方形视口：中心(图坐标) + 边长(英寸)
    fig = plt.figure(figsize=(13.8, 10.8), facecolor='white')
    _, ax = new3d(None, rect=(0.05, 0.05, 0.5, 0.5), elev=ELEV, azim=AZIM,
                  fig=fig)
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    thc = np.arctan2(np.cos(e) * np.sin(a), np.sin(e))   # 实体扇段正对视点
    half = np.deg2rad(26.0)
    n_sec = int(np.sum(np.abs(((np.arange(N_BOLT) * DTH - thc + np.pi)
                              % (2 * np.pi)) - np.pi) <= half))
    S = Scene()
    ph = np.linspace(0, 2 * np.pi, 49)

    # ---- 端面 x=0 的整环环面（= 与轮毂法兰的接触面）
    thf = np.linspace(0, 2 * np.pi, 721)
    S.add(grid_quads(np.stack([wall(0.0, thf, 0.0), wall(0.0, thf, T_WALL)])),
          LAM_F, ec=GREY, ealpha=0.30)

    # ---- 扇段外的其余螺套：端面截面（示意整环均布、几乎相接）
    for i in range(N_BOLT):
        th = i * DTH
        if abs(((th - thc + np.pi) % (2 * np.pi)) - np.pi) <= half:
            continue
        S.add(annulus(-0.6, ph, RHO_B, RHO_S, th), STEEL)
        S.add(annulus(-0.9, ph, 0.0, RHO_B, th), '#F0F4F6', ec='#F0F4F6')

    # ---- 实体扇段
    ths = np.linspace(thc - half, thc + half, 181)
    S.add(grid_quads(wall(np.array([[0.0], [X1]]), ths[None, :], 0.0)), LAM_D)
    for sgn in (-1, 1):
        S.add(grid_quads(wall(np.array([[0.0], [X1]]), thc + sgn * half,
                              np.array([[0.0, T_WALL]]))), LAM_D)
    S.add(grid_quads(wall(X1, np.stack([ths, ths]),
                          np.array([[0.0], [T_WALL]]))), '#8FA9BC')

    # ---- 扇段内的螺套实体（按配置的真直径、真节距、真埋深）
    i0 = int(round(thc / DTH))
    ith = [i * DTH for i in range(i0 - 12, i0 + 13)
           if abs(((i * DTH - thc + np.pi) % (2 * np.pi)) - np.pi) <= half]
    for th in ith:
        S.add(grid_quads(cyl(np.array([0.0, L_INS]), ph, RHO_S, th)), STEEL)
        S.add(annulus(L_INS, ph, 0.0, RHO_S, th), STEEL_D, ec=STEEL_D)
        S.add(annulus(-0.6, ph, RHO_B, RHO_S, th), STEEL_L)
        S.add(annulus(-0.9, ph, 0.0, RHO_B, th), '#F0F4F6', ec='#F0F4F6')
        S.add(grid_quads(cyl(np.array([0.0, L_ENG]), ph, RHO_D, th)), GREY)

    # ---- 外表面最后加（半透明，螺套透过它可见）
    S.add(grid_quads(wall(np.array([[0.0], [X1]]), ths[None, :], T_WALL)),
          LAM, alpha=0.22)

    fit_box(ax, S.pts)
    fill3d(fig, ax, *SQ)
    S.commit(ax, zorder=4)

    # ---- 螺栓圆
    ax.add_collection3d(polyline(wall(-2.5, np.linspace(0, 2 * np.pi, 361), R_AX),
                                 SIGNAL, 1.1, zorder=6, ls=(0, (6, 5))))
    fig.canvas.draw()

    # ---------------- 标注
    th_a = ith[len(ith) // 2]
    dim3(ax, wall(0.0, thc + np.pi / 2, T_WALL), wall(0.0, thc - np.pi / 2, T_WALL),
         '外径 Ø%.0f' % D_OUT, 0, 14, fs=13.5)
    lab(ax, wall(0.0, thc + np.pi * 0.80, 0.0), '内径 Ø%.0f' % D_IN, -22, -30,
        ha='right', color=INK, fs=13.5, weight='bold')
    lab(ax, wall(-2.5, thc + np.pi, R_AX), '螺栓圆 Ø%.0f（n = %d 均布）' % (D_BC, N_BOLT),
        -22, 4, ha='right', color=SIGNAL, fs=13.0, weight='bold')
    lab(ax, wall(0.0, thc + np.pi * 0.62, T_WALL * 0.5),
        '端面上示意其余 %d 个螺套截面' % (N_BOLT - n_sec), -22, 6, ha='right', color=GREY, fs=11.5)

    lab(ax, ins(-1.0, 0.0, 0.0, ith[1]), '相邻螺套节距 %.2f，净间距仅 %.2f' % (PITCH, _G.ligament),
        -30, -34, ha='right', color=INK, fs=11.5)
    dim3(ax, wall(0.0, thc - half - np.deg2rad(2), 0.0),
         wall(0.0, thc - half - np.deg2rad(2), T_WALL),
         '壁厚 %.0f' % T_WALL, -34, 16, ha='right', fs=12.0)

    lab(ax, cyl(np.array([L_INS]), np.array([np.deg2rad(150)]), RHO_S, th_a)[0, 0],
        '螺套末端 x = %.0f（埋深 %.0f）' % (L_INS, L_INS), 24, 34, ha='left', color=STEEL,
        fs=12.5, weight='bold')
    lab(ax, cyl(np.array([250.0]), np.array([np.deg2rad(30)]), RHO_S,
                ith[3])[0, 0], '预埋螺套 Ø%.0f × %.0f\n轴线在螺栓圆上（r = %.1f）' % (_G.D_ins, L_INS, R_AX),
        34, -40, ha='left', color=STEEL, fs=12.0)
    lab(ax, ins(-1.0, RHO_S, 0.0, ith[-2]), '叶根端面 x = 0\n与轮毂法兰接触面',
        30, 58, ha='left', color=INK, fs=12.0, weight='bold')

    fig.suptitle('图 3D-1  叶根整环三维总览：Ø%.0f / Ø%.0f 圆筒，'
                 '壁内周向均布 %d 个预埋螺套' % (D_OUT, D_IN, N_BOLT),
                 x=0.012, y=0.986, ha='left', fontsize=17.0,
                 fontweight='bold', color=INK)
    fig.text(0.012, 0.950,
             '几何据 config/ 的几何配置。实体扇段 %.0f°（%d 个螺套）'
             '按真尺寸绘制，其余以端面截面示意；轴向仅显示 0 ~ %.0f mm。'
             % (2 * np.degrees(half), n_sec, X1),
             fontsize=11.5, color=GREY, ha='left', va='top')

    # ---------------- 右列：1:1 节距细节
    axi = fig.add_axes([0.700, 0.645, 0.288, 0.232])
    axi.set_facecolor('#FAFCFD')
    axi.add_patch(Rectangle((-1.6 * PITCH, 0), 3.2 * PITCH, T_WALL,
                            fc=LAM, ec=GREY, lw=1.0))
    for k in (-1, 0, 1):
        axi.add_patch(Circle((k * PITCH, R_AX), RHO_W_RING, fc=SAFE, ec='none',
                             alpha=0.40))
        axi.add_patch(Circle((k * PITCH, R_AX), RHO_S, fc=STEEL, ec=INK, lw=1.0))
        axi.add_patch(Circle((k * PITCH, R_AX), RHO_B, fc='white', ec=INK, lw=0.7))
    axi.annotate('', xy=(-PITCH, 120), xytext=(0, 120),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=1.1))
    axi.text(-0.5 * PITCH, 126, '节距 %.2f' % PITCH, ha='center', fontsize=11.0,
             color=INK, fontweight='bold')
    axi.annotate('', xy=(-RHO_S, R_AX), xytext=(RHO_S, R_AX),
                 arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
    axi.text(0, R_AX - 16, 'Ø%.0f' % _G.D_ins, ha='center', fontsize=11.0, color='white',
             fontweight='bold')
    axi.annotate('净间距 %.2f' % _G.ligament, xy=(PITCH / 2, R_AX), xytext=(PITCH / 2, -34),
                 ha='center', fontsize=11.0, color=SIGNAL, fontweight='bold',
                 arrowprops=dict(arrowstyle='-|>', color=SIGNAL, lw=1.3))
    axi.text(-1.55 * PITCH, T_WALL - 6, '壁厚 %.0f' % T_WALL, fontsize=10.0, color=GREY, va='top')
    axi.set_xlim(-1.62 * PITCH, 1.62 * PITCH)
    axi.set_ylim(-48, 140)
    axi.set_aspect('equal')
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color(HAIR)
    axi.set_title('节距细节（周向展开，1:1）', fontsize=12.0, color=INK, pad=6)

    tbox(fig, 0.700, 0.598,
         ('关键事实\n'
          '  螺栓节距   π×%.0f/%d = %.2f mm\n'
          '  螺套外径   Ø%.0f\n'
          '  相邻钢体净间距   %.2f mm\n'
          '→ 两个螺套之间只剩 %.1f mm 层压，\n'
          '   不足壁厚的 1/%.1f。沿周向是一条\n'
          '   连续的弱化带，而不是一串孤立\n'
          '   的埋入体。'
          % (D_BC, N_BOLT, PITCH, _G.D_ins, _G.ligament,
             _G.ligament, T_WALL / _G.ligament)), fs=12.0)
    swatches(fig, 0.706, 0.358, [
        (STEEL, '钢衬套 42CrMoA（Ø%.0f × %.0f）' % (_G.D_ins, _G.L_ins)),
        (SAFE, '玻纤束缠绕层（见右上 1:1 细节）'),
        (GREY, '双头螺柱 M42（啮合 %.0f）' % _G.l_eng),
        (LAM, '叶根层压'),
        (SIGNAL, '螺栓圆 Ø%.0f' % D_BC),
    ], fs=11.5, dy=0.034)
    tbox(fig, 0.700, 0.175,
         ('几何冲突提示\n'
          '缠绕层名义外径 Ø%.0f > 节距 %.2f，\n'
          '二者不可能同时成立。整环各图中缠绕\n'
          '层按节距截断为 Ø%.1f 绘制，单体详图\n'
          '（图 3D-3）按 Ø%.0f 绘制。'
          % (2 * RHO_W, PITCH, PITCH, 2 * RHO_W)),
         fs=11.2, fc='#FBF2E4', ec=AMBER)
    return fig


# ================================================== 图 2  壁厚剖切
def fig_cutaway():
    X0, X1, X_CUT = 0.0, L_INS + 130.0, L_INS + 65.0
    XS = -170.0                         # 螺柱外伸端
    YW = 1.62 * PITCH                   # 周向窗口半宽（3 个节距）
    RC = R_AX                           # 剖切面取螺套轴线所在柱面
    z0 = R_IN
    ELEV, AZIM = 40.0, -119.0
    SQ = (0.293, 0.452, 7.45)

    fig = plt.figure(figsize=(13.4, 8.6), facecolor='white')
    _, ax = new3d(None, rect=(0.05, 0.05, 0.5, 0.5), elev=ELEV, azim=AZIM,
                  fig=fig)
    S = Scene()
    thi = np.array([-DTH, 0.0, DTH])

    def th_of(s):
        return np.asarray(s, float) / R_BC

    ths_w = th_of(np.linspace(-YW, YW, 221))
    ths_s = th_of(np.linspace(-YW, YW, 121))

    def band(t, u0, u1, xa, xb, col, **kw):
        S.add(grid_quads(np.stack([ins(np.array([xa, xb]), u0, 0.0, t, z0),
                                   ins(np.array([xa, xb]), u1, 0.0, t, z0)])),
              col, **kw)

    # ---- 内表面 r=0（全长）
    S.add(grid_quads(wall(np.array([[X0], [X1]]), ths_w[None, :], 0.0, z0)), LAM_D)

    # ---- 剖切面 r=R_AX：层压 + 三个螺套的纵剖面
    for t, sg in ((thi[0], -1), (thi[-1], +1)):
        band(t, sg * RHO_W_RING, sg * YW, X0, X_CUT, LAM_F)
    for t in thi:
        for a_, b_, col in ((RHO_TR, RHO_W_RING, SAFE), (RHO_S, RHO_TR, AMBER)):
            for sg in (-1, 1):
                band(t, sg * a_, sg * b_, 0.0, L_INS, col)
        for xa, xb, ri in ((0.0, L_ENG, RHO_D), (L_ENG, X_BLIND, RHO_B),
                           (X_BLIND, L_INS, 0.0)):
            if ri > 0:
                for sg in (-1, 1):
                    band(t, sg * ri, sg * RHO_S, xa, xb, STEEL_L)
            else:
                band(t, -RHO_S, RHO_S, xa, xb, STEEL_L)
        band(t, -RHO_W_RING, RHO_W_RING, L_INS, X_CUT, LAM_F)       # 末端之后

    # ---- 螺柱、盲孔半槽、孔底
    phl = np.linspace(np.pi / 2, 3 * np.pi / 2, 33)      # 下半圈
    phf = np.linspace(0, 2 * np.pi, 41)
    for t in thi:
        S.add(grid_quads(cyl(np.array([XS, 0.0]), phf, RHO_D, t, z0)), GREY)
        S.add(annulus(XS, phf, 0.0, RHO_D, t, z0), '#485762', ec='#485762')
        S.add(grid_quads(cyl(np.array([0.0, L_ENG]), phl, RHO_D, t, z0)), '#75838E')
        S.add(annulus(L_ENG, phl, 0.0, RHO_D, t, z0), '#8C9AA4', ec='#8C9AA4')
        S.add(grid_quads(cyl(np.array([L_ENG, X_BLIND]), phl, RHO_B, t, z0)),
              STEEL_D)
        S.add(annulus(X_BLIND, phl, 0.0, RHO_B, t, z0), '#2A5A7C', ec='#2A5A7C')

    # ---- x=0 端面（剖切后只剩 r<=R_AX，且被螺套下半挖掉）
    ys = np.linspace(-YW, YW, 401)
    rt = np.full_like(ys, RC)
    for t in thi:
        du = ys - R_BC * np.sin(t)          # 周向弧长差（窗口 <5°，弧弦差 <0.1）
        m = np.abs(du) < RHO_W_RING
        rt[m] = np.minimum(rt[m], RC - np.sqrt(np.maximum(
            RHO_W_RING ** 2 - du[m] ** 2, 0.0)))
    S.add(strip_quads(wall(0.0, th_of(ys), 0.0, z0),
                      wall(0.0, th_of(ys), rt, z0)), LAM_F)

    # ---- x>X_CUT 的全壁厚块 + 台阶
    S.add(grid_quads(wall(X_CUT, np.stack([ths_s, ths_s]),
                          np.array([[RC], [T_WALL]]), z0)), '#8FA9BC')
    S.add(grid_quads(wall(np.array([[X_CUT], [X1]]), ths_s[None, :], T_WALL, z0)),
          LAM)
    S.add(grid_quads(wall(X1, np.stack([ths_s, ths_s]),
                          np.array([[0.0], [T_WALL]]), z0)), '#8FA9BC')
    for sg in (-1, 1):
        S.add(grid_quads(wall(np.array([[X0], [X_CUT]]), th_of(sg * YW),
                              np.array([[0.0, RC]]), z0)), LAM_D)
        S.add(grid_quads(wall(np.array([[X_CUT], [X1]]), th_of(sg * YW),
                              np.array([[0.0, T_WALL]]), z0)), LAM_D)

    # ---- 轮毂法兰（只画窗口一侧一格，避免遮住端面）+ 螺母
    ths_f = th_of(np.linspace(-YW, -YW + 1.02 * PITCH, 41))
    for r in (0.0, T_WALL):
        S.add(grid_quads(wall(np.array([[-95.0], [0.0]]), ths_f[None, :], r, z0)),
              '#98A6AF')
    S.add(grid_quads(wall(-95.0, np.stack([ths_f, ths_f]),
                          np.array([[0.0], [T_WALL]]), z0)), '#7C8B96')
    for tt in (ths_f[0], ths_f[-1]):
        S.add(grid_quads(wall(np.array([[-95.0], [0.0]]), tt,
                              np.array([[0.0, T_WALL]]), z0)), '#68767F')
    phh = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    S.add(grid_quads(cyl(np.array([-155.0, -100.0]), phh, 33.0, thi[0], z0)),
          '#3F4D58')
    S.add(annulus(-155.0, phh, RHO_D, 33.0, thi[0], z0), '#55636D')

    fit_box(ax, S.pts)
    ax.view_init(elev=ELEV, azim=AZIM,
                 roll=pick_roll(ELEV, AZIM, S.pts, prefer_up=(0.0, 0.0, 1.0),
                                prefer_right=(1.0, 0.0, 0.0)))
    fill3d(fig, ax, *SQ)
    S.commit(ax, zorder=4)

    # ---- 剖面轮廓线：让三个螺套彼此分开、边界清楚
    for t in thi:
        for sg in (-1, 1):
            for rr, c, w in ((RHO_S, INK, 1.1), (RHO_TR, AMBER, 0.6),
                             (RHO_W_RING, SAFE, 0.9)):
                ax.add_collection3d(polyline(
                    ins(np.array([0.0, L_INS]), sg * rr, 0.0, t, z0), c, w,
                    zorder=7))
            ax.add_collection3d(polyline(
                ins(np.array([0.0, L_ENG]), sg * RHO_D, 0.0, t, z0), INK, 0.8,
                zorder=7))
            ax.add_collection3d(polyline(
                ins(np.array([L_ENG, X_BLIND]), sg * RHO_B, 0.0, t, z0), INK,
                0.8, zorder=7))
        ax.add_collection3d(polyline(
            ins(L_INS, np.linspace(-RHO_S, RHO_S, 2), 0.0, t, z0), INK, 1.1,
            zorder=7))
        # 螺纹环线（啮合段 105，节距 4.5 [推断]）
        segs = []
        for xk in np.arange(2.0, L_ENG, P_THD):
            c = cyl(np.array([xk]), phl, RHO_D + 0.4, t, z0)[0]
            segs += list(np.stack([c[:-1], c[1:]], axis=1))
        ax.add_collection3d(Line3DCollection(segs, colors='#2B3942',
                                             linewidths=0.55, zorder=7))

    # ---- 光纤：只画内表面 r=0 上真正可见的两条棱 + 端面下缘一段
    for sg in (-1, 1):
        ax.add_collection3d(polyline(
            wall(np.linspace(0.0, X_FIB, 80), th_of(sg * YW), 0.0, z0),
            SIGNAL, 3.2, zorder=9))
    ax.add_collection3d(polyline(
        wall(0.0, th_of(np.linspace(-YW, YW, 61)), 0.0, z0), SIGNAL, 3.2,
        zorder=9))
    # ---- 螺套末端站位
    ax.add_collection3d(polyline(
        wall(L_INS, th_of(np.linspace(-YW, YW, 41)), RC + 0.5, z0), INK, 1.3,
        zorder=10, ls=(0, (5, 3))))

    fig.canvas.draw()
    t0, t1, t2 = thi

    lab(ax, ins(1.0, -RHO_S, 0.0, t0, z0), '叶根端面 x = 0\n与轮毂法兰接触面',
        -16, -36, ha='right', color=INK, fs=11.5, weight='bold')
    lab(ax, ins(0.5 * L_ENG, 0.0, -RHO_D, t2, z0), 'M42 螺纹啮合段 l = %.0f' % _G.l_eng,
        18, -26, ha='left', color=AMBER, fs=11.5, weight='bold')
    lab(ax, ins(0.5 * (L_ENG + X_BLIND), 0.0, -RHO_B, t2, z0),
        '盲孔 Ø%.1f，孔底 x = %.0f' % (_G.d_bore, _G.l_eng + 25.0), 26, -64, ha='left', color=STEEL, fs=11.0)
    lab(ax, ins(385.0, 0.0, 0.0, t1, z0), '实心钢体 42CrMoA（x = %.0f ~ %.0f）' % (_G.l_eng + 25.0, L_INS),
        0, 0, ha='center', va='center', color='white', fs=11.5, arrow=False,
        box=False, weight='bold')
    lab(ax, ins(L_INS, 0.0, 0.0, t0, z0), '螺套末端 x = %.0f' % L_INS,
        -18, 22, ha='right', color=INK, fs=11.5, weight='bold')
    lab(ax, wall(0.5 * (L_INS + X_CUT), th_of(0.0), RC, z0),
        '其后层压厚度渐变过渡，无突变', -22, -66, ha='right', color=GREY,
        fs=10.5)
    lab(ax, ins(-140.0, 0.0, RHO_D, t0, z0), '双头螺柱 M42 + 螺母',
        -14, 26, ha='right', color=GREY, fs=11.0)
    lab(ax, wall(-48.0, th_of(-YW + 0.5 * PITCH), T_WALL, z0),
        '轮毂法兰（局部）', -10, 34, ha='right', color=GREY, fs=11.0)
    lab(ax, wall(190.0, th_of(-YW), 0.0, z0),
        '光纤：叶根内表面 r = 0\n轴向 x = 0 ~ %.0f，周向绕环一圈' % X_FIB,
        -14, -40,
        ha='right', color=SIGNAL, fs=11.5, weight='bold')
    lab(ax, ins(120.0, RHO_TR + 1.7, 0.0, t0, z0), '过渡层 0.5 ＋ 缠绕层',
        -14, 18, ha='right', color=SAFE, fs=10.5)

    dim3(ax, wall(X1, th_of(YW), 0.0, z0), wall(X1, th_of(YW), T_WALL, z0),
         '壁厚 %.0f' % T_WALL, 30, 6, ha='left', fs=11.5)
    dim3(ax, ins(1.0, 0.0, -R_AX, t2, z0), ins(1.0, 0.0, 0.0, t2, z0),
         '%.1f' % R_AX, 24, -2, ha='left', fs=11.0)
    dim3(ax, ins(3.0, -RHO_S, 0.0, t2, z0), ins(3.0, RHO_S, 0.0, t2, z0),
         'Ø%.0f' % _G.D_ins, 30, 14, ha='left', fs=11.0)

    fig.suptitle('图 3D-2  叶根壁厚剖切：沿 r = %.1f mm（三根螺套轴线所在柱面）'
                 '剖开，露出 3 个螺套的全长 %.0f mm' % (R_AX, L_INS),
                 x=0.010, y=0.984, ha='left', fontsize=15.0,
                 fontweight='bold', color=INK)
    fig.text(0.010, 0.944,
             '剖切范围 x = 0 ~ %.0f mm；x > %.0f 保留全壁厚，以显示螺套末端之后的'
             '层压过渡。周向窗口 = 3 个螺栓节距。' % (X_CUT, X_CUT),
             fontsize=11.0, color=GREY, ha='left', va='top')

    _xb = _G.l_eng + 25.0          # 盲孔底
    _rs = _G.D_ins / 2             # 螺套外半径
    tbox(fig, 0.605, 0.905,
         ('轴向分段（自叶根端面起算）\n'
          '  0 ~ %.0f     M42 螺纹啮合段\n'
          '  %.0f ~ %.0f   盲孔 Ø%.1f\n'
          '  %.0f ~ %.0f   实心钢体 42CrMoA\n'
          '  %.0f         螺套末端\n'
          '  %.0f ~       层压厚度渐变过渡\n'
          '  0 ~ %.0f     光纤轴向覆盖范围'
          % (_G.l_eng, _G.l_eng, _xb, _G.d_bore,
             _xb, L_INS, L_INS, L_INS, L_INS + 30.0)), fs=11.5)
    tbox(fig, 0.605, 0.620,
         ('径向位置（自内表面起算）\n'
          '  r = 0       光纤（内表面）\n'
          '  r = %.1f    螺套钢体下缘\n'
          '  r = %.1f    螺套轴线 = 螺栓圆\n'
          '  r = %.1f    螺套钢体上缘\n'
          '  r = %.0f     叶根外表面'
          % (R_AX - _rs, R_AX, R_AX + _rs, T_WALL)), fs=11.5)
    swatches(fig, 0.612, 0.355, [
        (STEEL_L, '钢体纵剖面 42CrMoA'),
        (GREY, '双头螺柱 M42'),
        (AMBER, '过渡层 0.5（树脂富集）'),
        (SAFE, '玻纤束缠绕层'),
        (LAM_F, '叶根层压（剖面）'),
        (SIGNAL, '光纤'),
    ], fs=11.5, dy=0.040)
    tbox(fig, 0.605, 0.085,
         '螺纹节距 4.5 mm 与沟槽尺寸为工艺推断值，\n非几何配置给定。',
         fs=10.8, fc='#FBF2E4', ec=AMBER)
    return fig


# ================================================== 图 3  单个螺套分层细节
def fig_layers():
    XA, XB = 0.0, 190.0
    ELEV, AZIM = 20.0, -146.0
    SQ = (0.252, 0.472, 7.60)
    fig = plt.figure(figsize=(16.0, 9.4), facecolor='white')
    _, ax = new3d(None, rect=(0.05, 0.05, 0.5, 0.5), elev=ELEV, azim=AZIM,
                  fig=fig)

    # 1/4 切除：缺口正对视点
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    phe = np.arctan2(np.cos(e) * np.sin(a), np.sin(e))
    ph0, ph1 = phe + np.deg2rad(45), phe + np.deg2rad(315)      # 保留 3/4
    ph = np.linspace(ph0, ph1, 109)
    xs = np.arange(XA, XB + 0.8, 0.8)
    th = 0.0
    S = Scene()
    rs = groove_rho(xs)                       # 带沟槽的钢体外半径

    RB = _G.r_block_out                       # 拉挤块对边半宽 [推断]
    RL = RB + 14.0                            # 层压壳对边半宽 [推断]
    r_oct = RB / np.cos(np.deg2rad(22.5))
    r_lam = RL / np.cos(np.deg2rad(22.5))
    pho = np.linspace(-np.pi, np.pi, 9) + np.pi / 8
    keep = [k for k in range(8)
            if abs(((0.5 * (pho[k] + pho[k + 1]) - phe + np.pi)
                    % (2 * np.pi)) - np.pi) > np.deg2rad(45)]

    # ---- 层 1 钢衬套 / 层 2 过渡层 / 层 3 缠绕层：同轴柱面
    S.add(grid_quads(cyl(xs, ph, rs, th)), STEEL)
    S.add(grid_quads(cyl(np.array([XA, XB]), ph, RHO_B, th)), STEEL_D)
    S.add(annulus(XB - 0.5, ph, 0.0, RHO_B, th), '#132F42', ec='#132F42')
    S.add(grid_quads(cyl(xs, ph, np.full_like(xs, RHO_TR), th)), AMBER)
    S.add(grid_quads(cyl(xs, ph, np.full_like(xs, RHO_W), th)), SAFE)
    # ---- 层 4 拉挤块 / 层 5 层压：八棱柱
    for r_, col in ((r_oct, BLOCK), (r_lam, LAM)):
        for k in keep:
            S.add(grid_quads(np.stack([
                ins(np.array([XA, XB]), r_ * np.sin(pho[k]),
                    r_ * np.cos(pho[k]), th),
                ins(np.array([XA, XB]), r_ * np.sin(pho[k + 1]),
                    r_ * np.cos(pho[k + 1]), th)])), col)

    # ---- 1/4 缺口的两个径向剖面：五层条带
    for phc in (ph0, ph1):
        u, v = np.sin(phc), np.cos(phc)
        A = ins(xs, RHO_B * u, RHO_B * v, th)
        B = ins(xs, rs * u, rs * v, th)
        S.add(strip_quads(A, B), STEEL_L)
        S.add(strip_quads(B, ins(xs, RHO_TR * u, RHO_TR * v, th)), AMBER)
        for r0, r1, col in ((RHO_TR, RHO_W, SAFE), (RHO_W, r_oct, BLOCK),
                            (r_oct, r_lam, LAM_F)):
            S.add(grid_quads(np.stack([
                ins(np.array([XA, XB]), r0 * u, r0 * v, th),
                ins(np.array([XA, XB]), r1 * u, r1 * v, th)])), col)

    # ---- x=0 端面：五个同心环（最能看清层序）。螺纹孔不封口。
    for r0, r1, col in ((RHO_B, RHO_S, STEEL_L), (RHO_S, RHO_TR, AMBER),
                        (RHO_TR, RHO_W, SAFE), (RHO_W, r_oct, BLOCK),
                        (r_oct, r_lam, LAM_F)):
        S.add(annulus(-0.5, ph, r0, r1, th), col, ec=col)
    # ---- 远端面
    for r0, r1, col in ((RHO_B, RHO_S, STEEL_D), (RHO_S, RHO_TR, '#9E6B15'),
                        (RHO_TR, RHO_W, '#176046'), (RHO_W, r_oct, '#697782'),
                        (r_oct, r_lam, LAM_D)):
        S.add(annulus(XB, ph, r0, r1, th), col)

    fit_box(ax, S.pts)
    fill3d(fig, ax, *SQ)
    S.commit(ax, zorder=4)

    # ---- 端面各界面的圆：强调 A / B / C
    for r_, c, w in ((RHO_B, INK, 1.0), (RHO_TR, INK, 0.8), (r_lam, INK, 1.0),
                     (RHO_S, AMBER, 1.8), (RHO_W, SIGNAL, 2.0),
                     (r_oct, SIGNAL, 2.0)):
        ax.add_collection3d(polyline(
            ins(-1.2, r_ * np.sin(ph), r_ * np.cos(ph), th), c, w, zorder=8))
    fig.canvas.draw()

    # ---- 层号 ①~⑤ 标在端面环上
    for num, r_, ang in (('①', 0.5 * (RHO_B + RHO_S), 74),
                         ('②', 0.5 * (RHO_S + RHO_TR), 96),
                         ('③', 0.5 * (RHO_TR + RHO_W), 117),
                         ('④', 0.5 * (RHO_W + r_oct), 140),
                         ('⑤', 0.5 * (r_oct + r_lam), 162)):
        pn = phe + np.deg2rad(ang)
        lab(ax, ins(-1.4, r_ * np.sin(pn), r_ * np.cos(pn), th), num, 0, 0,
            ha='center', va='center', color=INK if num == '⑤' else 'white',
            fs=13.0, arrow=False, box=False, weight='bold')

    # ---- 界面 A / B / C（锚在端面上，短引线，互不交叉）
    pa = phe + np.deg2rad(-120)
    lab(ax, ins(-1.2, RHO_S * np.sin(pa), RHO_S * np.cos(pa), th),
        '界面 A  钢 / 过渡层', -16, -14, ha='right', color=INK, fs=13.0,
        weight='bold', acol=AMBER, lw=1.5)
    pb = phe + np.deg2rad(-152)
    lab(ax, ins(-1.2, RHO_W * np.sin(pb), RHO_W * np.cos(pb), th),
        '界面 B  缠绕层 / 拉挤块', -16, 26, ha='right', color=INK, fs=13.0,
        weight='bold', acol=SIGNAL, lw=1.5)
    pc = phe + np.deg2rad(-178)
    lab(ax, ins(-1.2, r_oct * np.sin(pc), r_oct * np.cos(pc), th),
        '界面 C  拉挤块 / 层压', -16, 64, ha='right', color=INK, fs=13.0,
        weight='bold', acol=SIGNAL, lw=1.5)
    lab(ax, ins(0.45 * XB, 0.0, -RHO_B, th), '内螺纹孔 Ø37.5（M42 小径）',
        8, -56, ha='left', color=STEEL, fs=12.0)
    pg = ph1 + np.deg2rad(24)
    lab(ax, ins(0.34 * XB, rs[80] * np.sin(pg), rs[80] * np.cos(pg), th),
        '钢衬套外表面环向沟槽\n槽深 1.5 / 节距 8（推断）→ 右下放大', 20, 36,
        ha='left', color=STEEL, fs=11.5)
    lab(ax, ins(0.82 * XB, r_lam * np.sin(pho[keep[0]] + 0.2),
                r_lam * np.cos(pho[keep[0]] + 0.2), th),
        '⑤ 叶根层压', 16, 18, ha='left', color=GREY, fs=12.0)

    fig.suptitle('图 3D-3  单个预埋螺套的五层构造（1/4 切除，轴向截取 x = 0 ~ 190 mm）',
                 x=0.008, y=0.985, ha='left', fontsize=16.5,
                 fontweight='bold', color=INK)
    fig.text(0.008, 0.947,
             '自内向外：钢衬套 → 过渡层 → 玻纤束缠绕层 → 拉挤 GFRP 块 → 叶根层压。'
             '左端同心环与 1/4 缺口的两个径向剖面同时给出层序。',
             fontsize=11.5, color=GREY, ha='left', va='top')

    # ------------------------------------------------ 右上：层序图例表
    axl = fig.add_axes([0.545, 0.470, 0.448, 0.425])
    axl.set_xlim(0, 1); axl.set_ylim(0, 1); axl.axis('off')
    axl.add_patch(Rectangle((0, 0), 1, 1, fc='#F7FAFB', ec=HAIR, lw=1.0))
    axl.text(0.030, 0.962, '分层构造（自内向外）', fontsize=13.5,
             fontweight='bold', color=INK, va='top')
    axl.text(0.100, 0.872, '层名 / 材料', fontsize=10.5, color=GREY, va='top')
    axl.text(0.560, 0.872, '径向范围', fontsize=10.5, color=GREY, va='top')
    axl.text(0.820, 0.872, '厚度 mm', fontsize=10.5, color=GREY, va='top')
    axl.plot([0.028, 0.972], [0.845, 0.845], color=HAIR, lw=1.0)
    rows = [
        ('①', STEEL, '钢衬套', '42CrMoA，外表面环向沟槽',
         'Ø%.1f → Ø%.0f' % (_G.d_bore, _G.D_ins),
         '壁厚 %.2f' % ((_G.D_ins - _G.d_bore) / 2)),
        ('②', AMBER, '过渡层（树脂富集）', '环氧树脂',
         'Ø%.0f → Ø%.0f' % (_G.D_ins, _G.D_ins + 1), '0.5'),
        ('③', SAFE, '玻纤束缠绕层', '玻纤 / 环氧',
         'Ø%.0f → Ø%.0f' % (_G.D_ins + 1, _G.D_ins + 12), '5.5（名义约 6）'),
        ('④', BLOCK, '拉挤 GFRP 块', '拉挤玻纤型材，棱柱形',
         '≈ □%.0f（推断）' % (_G.D_ins + 27), '约 7.5（推断）'),
        ('⑤', LAM, '叶根层压', '三轴 / 单轴玻纤铺层',
         '至 Ø%.0f 外表面' % D_OUT, '与整环连续'),
    ]
    y = 0.762
    for num, col, name, mat, rng, thk in rows:
        axl.add_patch(Rectangle((0.030, y - 0.032), 0.052, 0.068, fc=col,
                                ec=INK, lw=0.7))
        axl.text(0.056, y + 0.002, num, fontsize=10.5, color='white',
                 ha='center', va='center', fontweight='bold')
        axl.text(0.100, y + 0.030, name, fontsize=12.5, color=INK, va='center')
        axl.text(0.100, y - 0.026, mat, fontsize=10.2, color=GREY, va='center')
        axl.text(0.560, y + 0.002, rng, fontsize=11.5, color=INK, va='center')
        axl.text(0.820, y + 0.002, thk, fontsize=11.0, color=INK, va='center')
        y -= 0.140
    axl.plot([0.028, 0.972], [0.132, 0.132], color=HAIR, lw=1.0)
    axl.text(0.030, 0.108,
             '界面 A = ①/②（钢 / 过渡层）　　界面 B = ③/④　　界面 C = ④/⑤',
             fontsize=11.5, color=INK, va='top')
    axl.text(0.030, 0.060,
             '文献（He 等 2025）实测的拉拔失效面出现在界面 B / C 一带，'
             '不在界面 A。',
             fontsize=11.5, color=SIGNAL, va='top', fontweight='bold')

    # ------------------------------------------------ 右下：沟槽剖面放大
    axg = fig.add_axes([0.545, 0.078, 0.448, 0.320])
    axg.set_facecolor('#FAFCFD')
    xg = np.linspace(0, 27.0, 900)
    rg = groove_rho(xg)
    axg.fill_between(xg, 30.0, rg, color=STEEL, ec=INK, lw=1.1, zorder=3)
    axg.fill_between(xg, rg, rg + 0.5, color=AMBER, ec='none', zorder=4)
    axg.fill_between(xg, rg + 0.5, RHO_W, color=SAFE, ec='none', alpha=0.9,
                     zorder=2)
    axg.axhline(RHO_W, color=INK, lw=1.0, zorder=5)
    axg.fill_between(xg, RHO_W, 48.4, color=BLOCK, zorder=2)
    xa0 = 0.5 * GRV_P
    axg.annotate('', xy=(xa0, 47.3), xytext=(xa0 + GRV_P, 47.3), zorder=6,
                 arrowprops=dict(arrowstyle='<->', color='white', lw=1.2))
    axg.text(xa0 + 0.5 * GRV_P, 47.55, '节距 8.0（推断）', ha='center',
             fontsize=10.8, color='white', zorder=6, va='bottom')
    axg.annotate('', xy=(xa0, RHO_S), xytext=(xa0, RHO_S - GRV_D), zorder=6,
                 arrowprops=dict(arrowstyle='<->', color=SIGNAL, lw=1.4))
    axg.text(xa0 + 0.7, RHO_S - 0.75, '槽深 1.5（推断）', ha='left',
             fontsize=10.8, color=SIGNAL, va='center', zorder=6,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none',
                       alpha=0.82))
    axg.text(0.7, 33.0, '① 钢衬套 42CrMoA', fontsize=11.5, color='white',
             zorder=6, fontweight='bold')
    axg.text(26.3, 39.8, '② 过渡层 0.5', fontsize=10.5, color=AMBER,
             zorder=6, fontweight='bold', ha='right', va='bottom')
    axg.text(0.7, 42.2, '③ 玻纤束缠绕层（随槽形贴合）', fontsize=11.5,
             color='white', zorder=6, fontweight='bold')
    axg.text(0.7, 46.2, '④ 拉挤 GFRP 块', fontsize=11.5, color='white',
             zorder=6, fontweight='bold')
    axg.set_xlim(0, 27.0); axg.set_ylim(30.0, 48.4)
    axg.set_aspect('equal')
    axg.set_xlabel('轴向 x [mm]', fontsize=10.5, color=GREY, labelpad=2)
    axg.set_ylabel('半径 ρ [mm]', fontsize=10.5, color=GREY, labelpad=2)
    axg.tick_params(labelsize=9.5, colors=GREY)
    for sp in axg.spines.values():
        sp.set_color(HAIR)
    axg.set_title('钢衬套外表面环向沟槽剖面（局部放大，1:1）',
                  fontsize=12.5, color=INK, pad=6)

    # ---- 左下：轴向位置索引条
    axk = fig.add_axes([0.075, 0.030, 0.330, 0.038])
    axk.add_patch(Rectangle((0, 0), L_INS, 1, fc=LAM, ec=GREY, lw=0.9))
    axk.add_patch(Rectangle((XA, 0), XB - XA, 1, fc=STEEL, ec=INK, lw=1.0))
    axk.text(0.5 * (XA + XB), 0.5, '本图范围', ha='center', va='center',
             color='white', fontsize=10.5, fontweight='bold')
    axk.text(L_INS + 10, 0.5, '螺套全长 %.0f' % L_INS, ha='left', va='center',
             color=GREY, fontsize=10.5)
    axk.text(-10, 0.5, 'x = 0', ha='right', va='center', color=GREY, fontsize=10.5)
    axk.set_xlim(-80, 660); axk.set_ylim(0, 1)
    axk.axis('off')
    return fig


# ================================================== 图 4  光纤敷设三维示意
def fig_fiber():
    X0, X1 = -35.0, 650.0
    ELEV, AZIM = 26.0, -60.0
    SQ = (0.272, 0.452, 7.20)
    fig = plt.figure(figsize=(13.6, 9.0), facecolor='white')
    _, ax = new3d(None, rect=(0.05, 0.05, 0.5, 0.5), elev=ELEV, azim=AZIM,
                  fig=fig)
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    # 内表面正对视点，再偏 18° 以保留立体感
    thc = np.arctan2(-np.cos(e) * np.sin(a), -np.sin(e)) + np.deg2rad(18.0)
    half = np.deg2rad(30.0)                                # 60° 扇段
    S = Scene()
    ths = np.linspace(thc - half, thc + half, 241)

    # ---- 壁体与螺套（远）；半透明内表面最后画（近）
    S.add(grid_quads(wall(np.array([[X0], [X1]]), ths[None, :], T_WALL)),
          LAM, alpha=0.35)
    for sgn in (-1, 1):
        S.add(grid_quads(wall(np.array([[X0], [X1]]), thc + sgn * half,
                              np.array([[0.0, T_WALL]]))), LAM_D)
    for xk in (X0, X1):
        S.add(grid_quads(wall(xk, np.stack([ths, ths]),
                              np.array([[0.0], [T_WALL]]))), '#8FA9BC')

    ph = np.linspace(0, 2 * np.pi, 41)
    i0 = int(round(thc / DTH))
    ith = [i * DTH for i in range(i0 - 13, i0 + 14)
           if abs(((i * DTH - thc + np.pi) % (2 * np.pi)) - np.pi) <= half]
    for th in ith:
        S.add(grid_quads(cyl(np.array([0.0, L_INS]), ph, RHO_W_RING, th)),
              SAFE, alpha=0.32)
        S.add(grid_quads(cyl(np.array([0.0, L_INS]), ph, RHO_S, th)), STEEL)
        S.add(annulus(L_INS, ph, 0.0, RHO_S, th), STEEL_D, ec=STEEL_D)
        S.add(annulus(0.0, ph, RHO_B, RHO_S, th), STEEL_L)
    # 半透明内表面（光纤贴在它上面）
    S.add(grid_quads(wall(np.array([[X0], [X1]]), ths[None, :], 0.0)),
          '#EAF0F4', alpha=0.50)

    fit_box(ax, S.pts)
    ax.view_init(elev=ELEV, azim=AZIM,
                 roll=pick_roll(ELEV, AZIM, S.pts,
                                prefer_up=(1.0, 0.0, 0.0)))
    fill3d(fig, ax, *SQ)
    S.commit(ax, zorder=4)

    # ---- 光纤走线：内表面上 4 个周向圈 + 端部轴向连接段
    x_loops = [60.0, 210.0, 360.0, L_INS]
    thl = np.linspace(thc - half, thc + half, 241)
    for xk in x_loops:
        ax.add_collection3d(polyline(wall(xk, thl, 0.4), SIGNAL, 3.6, zorder=10))
    th_jog = thc - half + np.deg2rad(1.8)
    for xa, xb in zip(x_loops[:-1], x_loops[1:]):
        ax.add_collection3d(polyline(wall(np.linspace(xa, xb, 30), th_jog, 0.4),
                                     SIGNAL, 2.6, zorder=10))
    for xk in (0.0, X_FIB):
        ax.add_collection3d(polyline(wall(xk, thl, 0.2), SIGNAL, 1.3, zorder=9,
                                     ls=(0, (6, 4))))

    # ---- 1.3 mm 读数点：中间一圈上打一小段真实间距的点
    th_m = thc + np.deg2rad(8.0)
    arc = np.arange(48) * DX_RES
    pm = wall(210.0, th_m + arc / R_IN, 1.0)
    ax.scatter(pm[:, 0], pm[:, 1], pm[:, 2], s=7.0, c=INK, depthshade=False,
               zorder=12)

    fig.canvas.draw()
    t0 = ith[len(ith) // 2]

    lab(ax, wall(280.0, thc + np.deg2rad(15), 0.0),
        '光纤敷设在叶根内表面 r = 0\n周向绕环一圈 = %s mm' % _sp(C_IN), 0, 46,
        ha='center', color=SIGNAL, fs=12.5, weight='bold')
    lab(ax, pm[24], '空间分辨率 %.1f mm\n每 %.1f mm 一个独立读数' % (DX_RES, DX_RES), 6, -52,
        ha='center', color=INK, fs=11.0)
    lab(ax, cyl(np.array([300.0]), np.array([np.pi]), RHO_S, t0)[0, 0],
        '预埋螺套（透过层压示意）\n轴线在 r = %.1f' % R_AX, -18, -46, ha='right',
        color=STEEL, fs=11.5)
    lab(ax, wall(L_INS, thc + half - np.deg2rad(2), 0.0), '螺套末端 x = %.0f' % L_INS,
        16, -14, ha='left', color=INK, fs=11.0)
    lab(ax, wall(X_FIB, thc + half - np.deg2rad(2), 0.0), '光纤覆盖上限 x = %.0f' % X_FIB,
        16, 20, ha='left', color=SIGNAL, fs=11.0)
    for xk, nm in zip(x_loops, ['第 %d 圈 x = %.0f' % (k + 1, xv)
                                for k, xv in enumerate(x_loops)]):
        lab(ax, wall(xk, thc - half, 0.4), nm, -12, 0, ha='right',
            color=SIGNAL, fs=10.5, lw=0.7)
    dim3(ax, wall(X0, thc + half, 0.0), wall(X0, thc + half, T_WALL),
         '壁厚 %.0f' % T_WALL, 26, 6, ha='left', fs=11.0)
    dim3(ax, ins(X0 + 3, 0.0, -R_AX, ith[2]), ins(X0 + 3, 0.0, 0.0, ith[2]),
         '螺套轴线 r = %.1f' % R_AX, -26, -16, ha='right', fs=11.0)

    fig.suptitle('图 3D-4  光纤敷设三维示意：内表面走线与壁内螺套的空间关系',
                 x=0.008, y=0.985, ha='left', fontsize=16.5,
                 fontweight='bold', color=INK)
    fig.text(0.008, 0.947,
             '显示 60° 扇段（%.0f 个螺套），自叶根内部看向内表面。内表面画成半透明，'
             % round(N_BOLT / 6.0) +
             '壁内螺套透过层压可见；\n光纤（橙色）贴在内表面上，不进入壁厚。',
             fontsize=11.0, color=GREY, ha='left', va='top')

    tbox(fig, 0.590, 0.900,
         ('光纤与螺套的径向关系\n'
          '  光纤             r = 0\n'
          '  缠绕层下缘       r = %.1f\n'
          '  钢体下缘         r = %.1f\n'
          '  螺套轴线         r = %.1f = 螺栓圆\n'
          '  叶根外表面       r = %.0f\n'
          '→ 光纤与螺套之间只隔 %.0f ~ %.0f mm\n'
          '   层压，中间没有第二个结构界面。'
          % (R_AX - RHO_W, R_AX - _G.D_ins / 2, R_AX, T_WALL,
             R_AX - RHO_W, R_AX - _G.D_ins / 2)), fs=11.5)
    tbox(fig, 0.590, 0.585,
         ('分辨率与覆盖\n'
          '  空间分辨率              %.1f mm\n'
          '  内表面绕一圈            %s mm\n'
          '  一圈的读数数            %s 点\n'
          '  内表面上的螺栓间距      %.2f mm\n'
          '  每个螺栓对应            %.0f 个读数\n'
          '  轴向覆盖                x = 0 ~ %.0f\n'
          '  四圈总光纤长度          约 %.0f m'
          % (DX_RES, _sp(C_IN), _sp(C_IN / DX_RES), P_IN,
             P_IN / DX_RES, L_INS + 30.0, 4 * C_IN / 1000.0)), fs=11.5)

    # ---- 右下：1.3 mm 读数间距（1:1）
    axr = fig.add_axes([0.596, 0.080, 0.330, 0.175])
    axr.set_facecolor('#FAFCFD')
    n = int(P_IN / DX_RES) + 1
    xr = np.arange(n) * DX_RES
    axr.plot([xr[0] - 2, xr[-1] + 2], [0, 0], color=SIGNAL, lw=3.6,
             solid_capstyle='butt')
    axr.plot(xr, np.zeros(n), 'o', ms=3.8, color=INK, zorder=4)
    axr.annotate('', xy=(0, 0.42), xytext=(DX_RES, 0.42),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=1.0))
    axr.text(DX_RES * 0.5, 0.52, '1.3 mm', ha='center', fontsize=10.5, color=INK)
    axr.annotate('', xy=(0, -0.50), xytext=(P_IN, -0.50),
                 arrowprops=dict(arrowstyle='<->', color=STEEL, lw=1.4))
    axr.text(P_IN * 0.5, -0.72,
             '内表面上一个螺栓间距 %.2f mm = %.0f 个读数'
             % (P_IN, P_IN / DX_RES),
             ha='center', fontsize=10.5, color=STEEL, va='top')
    axr.set_xlim(-3, n * DX_RES + 3)
    axr.set_ylim(-1.15, 1.00)
    axr.set_xticks([]); axr.set_yticks([])
    for sp in axr.spines.values():
        sp.set_color(HAIR)
    axr.set_title('沿光纤的读数间距（1:1）', fontsize=12.0, color=INK, pad=5)
    return fig


# ====================================================================== main
def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white',
                pad_inches=0.16)
    plt.close(fig)
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        print('  %-28s  %d x %d px' % (name, w, h))
    except Exception as exc:                        # pragma: no cover
        print('  %-28s  (PIL failed: %s)' % (name, exc))


def main():
    if not os.path.isdir(FIGS):
        raise SystemExit('找不到输出目录: %s' % FIGS)
    print('root insert 3D geometry figures ->', FIGS)
    save(fig_ring(), 'fig3d_01_root_ring.png')
    save(fig_cutaway(), 'fig3d_02_cutaway.png')
    save(fig_layers(), 'fig3d_03_layers.png')
    save(fig_fiber(), 'fig3d_04_fiber_layout.png')
    print('done.')


if __name__ == '__main__':
    main()
