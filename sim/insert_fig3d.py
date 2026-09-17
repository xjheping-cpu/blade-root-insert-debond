# -*- coding: utf-8 -*-
"""
把 M2′ 轴对称模型的应力场绕螺套轴线旋成三维，剖开显示。

轴对称模型的解本身就是三维的（场量与周向角无关），旋成三维不是插值猜测，
而是把同一个解按它本来的对称性画出来。剖切面与外表面上的颜色都是计算值。

自带一个正交投影 + 画家算法的渲染器：面片按面心到视点的距离排序，
逐面绘制，并按面法向做简单明暗，避免 mplot3d 的前后穿插错误。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import insert_study  # noqa: F401
from insert_study import set_elastic
from geometry import Geom
_G = Geom()            # 几何自 config/ 读入，换机型只改配置
from insert_axi import (InsertAxi, IFACES, MATS, MAT_COL, MAT_CN,
                        L_INS, L_ENG, X_BORE, R_CELL, R_BORE, R_STEEL,
                        R_TRANS, R_WRAP, R_BLOCK, FM_DEFAULT, P_PITCH, T_WALL)

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10.5
rcParams['savefig.dpi'] = 200
rcParams['axes.grid'] = False

INK = '#12212C'; STEEL = '#2F6690'; SIGNAL = '#D94F2B'
AMBER = '#C8871B'; SAFE = '#1E7A5A'; GREY = '#5A6B77'
FIG = 'figs/'
FA_OPER = 159e3
R_IN = _G.D_in / 2     # 叶根内表面半径
R_BC = _G.R_bc         # 螺栓圆半径


# ------------------------------------------------------------------ 渲染器
class Ortho:
    """正交投影 + 画家算法。az 为绕竖轴方位角，el 为仰角，均为度。"""

    def __init__(self, az=-58.0, el=20.0, light=(0.4, -0.75, 0.53)):
        a, e = np.radians(az), np.radians(el)
        self.d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        up0 = np.array([0.0, 0.0, 1.0])
        r = np.cross(up0, self.d)
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.d, self.right)
        L = np.asarray(light, float)
        self.light = L / np.linalg.norm(L)

    def to2d(self, P):
        P = np.asarray(P, float)
        return np.stack([P @ self.right, P @ self.up], -1)

    def depth(self, P):
        return np.asarray(P, float) @ self.d

    def draw(self, ax, faces, vals, cmap, norm, edge=None, lw=0.0,
             shade=True, alpha=1.0):
        """faces: (n,4,3)；vals: (n,) 或 None（None 时用 edge 颜色填充）。"""
        F = np.asarray(faces, float)
        c = F.mean(1)
        order = np.argsort(self.depth(c))          # 远的先画
        F, c = F[order], c[order]
        v = None if vals is None else np.asarray(vals)[order]
        n = np.cross(F[:, 1] - F[:, 0], F[:, 2] - F[:, 0])
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(nn, 1e-12)
        sh = 0.58 + 0.42 * np.abs(n @ self.light) if shade else np.ones(len(F))
        if v is None:
            base = np.tile(np.array(matplotlib.colors.to_rgb(edge)), (len(F), 1))
        else:
            base = plt.get_cmap(cmap)(norm(v))[:, :3]
        rgb = np.clip(base * sh[:, None], 0, 1)
        pc = PolyCollection(self.to2d(F), facecolors=rgb, alpha=alpha,
                            edgecolors='none' if lw == 0 else edge, linewidths=lw)
        ax.add_collection(pc)
        return self.to2d(F)


# ------------------------------------------------------------------ 面片生成
X_VIEW = 560.0          # 只画到这里，远端 560~700 与结论无关


def _xe(M):
    return np.flatnonzero(0.5 * (M.x[:-1] + M.x[1:]) <= X_VIEW)


def revolve_faces(M, val, phi0, phi1, nphi=40):
    """外表面（r = R_CELL）的旋转面片，颜色取最外一层单元的值。"""
    nr_e, nx_e = M.nr - 1, M.nx - 1
    ph = np.linspace(phi0, phi1, nphi + 1)
    F, V = [], []
    for k in _xe(M):
        x0, x1 = M.x[k], M.x[k + 1]
        v = val[k + (nr_e - 1) * nx_e]
        for j in range(nphi):
            p0, p1 = ph[j], ph[j + 1]
            F.append([[x0, R_CELL * np.cos(p0), R_CELL * np.sin(p0)],
                      [x1, R_CELL * np.cos(p0), R_CELL * np.sin(p0)],
                      [x1, R_CELL * np.cos(p1), R_CELL * np.sin(p1)],
                      [x0, R_CELL * np.cos(p1), R_CELL * np.sin(p1)]])
            V.append(v)
    return np.array(F), np.array(V)


def section_faces(M, val, sign=1.0, only=None):
    """剖切面（y = 0 平面）上的单元面片。sign=+1 取 +z 半，-1 取 -z 半。
    only='comp' 只取复合材料层，only='steel' 只取钢。"""
    F, V = [], []
    void, steel = MATS.index('void'), MATS.index('steel')
    xc = 0.5 * (M.x[:-1] + M.x[1:])
    for e, nd in enumerate(M.el):
        t = M.tag[e]
        if t == void:
            continue
        if only == 'comp' and t == steel:
            continue
        if only == 'steel' and t != steel:
            continue
        if M.el_xc[e] > X_VIEW:
            continue
        xx, rr = M.X[nd], M.R[nd]
        F.append(np.stack([xx, np.zeros(4), sign * rr], 1))
        V.append(val[e])
    return np.array(F), np.array(V)


def endcap_faces(M, val, phi0, phi1, nphi=40):
    """x = 0 端面的承压环带（螺套端面下沉，不含 r < R_STEEL）。"""
    ir = np.flatnonzero(M.r >= R_STEEL - 1e-9)
    ph = np.linspace(phi0, phi1, nphi + 1)
    nx_e = M.nx - 1
    F, V = [], []
    for a, b in zip(ir[:-1], ir[1:]):
        r0, r1 = M.r[a], M.r[b]
        v = val[0 + a * nx_e]
        for j in range(nphi):
            p0, p1 = ph[j], ph[j + 1]
            F.append([[0, r0 * np.cos(p0), r0 * np.sin(p0)],
                      [0, r1 * np.cos(p0), r1 * np.sin(p0)],
                      [0, r1 * np.cos(p1), r1 * np.sin(p1)],
                      [0, r0 * np.cos(p1), r0 * np.sin(p1)]])
            V.append(v)
    return np.array(F), np.array(V)


def tube_faces(radius, x, val, phi0=-np.pi / 2, phi1=np.pi / 2, nphi=40):
    """一个圆柱面（如界面 B）上的面片，val 沿 x 给定。默认只画朝向视点的半圈。"""
    ph = np.linspace(phi0, phi1, nphi + 1)
    F, V = [], []
    for k in range(len(x) - 1):
        v = 0.5 * (val[k] + val[k + 1])
        for j in range(nphi):
            p0, p1 = ph[j], ph[j + 1]
            F.append([[x[k], radius * np.cos(p0), radius * np.sin(p0)],
                      [x[k + 1], radius * np.cos(p0), radius * np.sin(p0)],
                      [x[k + 1], radius * np.cos(p1), radius * np.sin(p1)],
                      [x[k], radius * np.cos(p1), radius * np.sin(p1)]])
            V.append(v)
    return np.array(F), np.array(V)


# ------------------------------------------------------------------ 面板
def panel_cut(ax, cam, M, val, cmap, norm, title, deb=None, tag=''):
    """半剖的三维单胞。剖面取 y = 0 平面（正对视点），远侧保留外表面。
    钢衬套画成灰色实体：它的应力量级比复合材料高一个数量级，
    同色标会把复合材料里的变化全部压平。"""
    Fo, Vo = revolve_faces(M, val, -np.pi / 2, np.pi / 2)
    Fe, Ve = endcap_faces(M, val, -np.pi / 2, np.pi / 2)
    Fs, Vs = [], []
    for sg in (1.0, -1.0):
        a, b = section_faces(M, val, sg, only='comp')
        if len(a):
            Fs.append(a); Vs.append(b)
    F = np.concatenate([Fo, Fe] + Fs)
    V = np.concatenate([Vo, Ve] + Vs)
    cam.draw(ax, F, V, cmap, norm)
    Fst = [section_faces(M, val, sg, only='steel')[0] for sg in (1.0, -1.0)]
    Fst = np.concatenate([a for a in Fst if len(a)])
    cam.draw(ax, Fst, None, None, None, edge='#7A8894')
    for s in (1, -1):
        m = M.x <= L_INS
        p = cam.to2d(np.stack([M.x[m], np.zeros(m.sum()),
                               s * np.full(m.sum(), R_WRAP)], 1))
        ax.plot(p[:, 0], p[:, 1], color='#FFFFFF', lw=0.9, alpha=0.9)
        if deb is not None:
            q = M.x[(M.x >= L_INS - deb) & (M.x <= L_INS)]
            pd = cam.to2d(np.stack([q, np.zeros(len(q)),
                                    s * np.full(len(q), R_WRAP)], 1))
            ax.plot(pd[:, 0], pd[:, 1], color=SIGNAL, lw=2.8)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    ax.set_aspect('equal'); ax.axis('off')
    ax.autoscale_view()
    if tag:
        ax.text(0.985, 0.90, tag, transform=ax.transAxes, fontsize=9.5,
                color=SIGNAL, ha='right')


def main():
    print('三维面应力图件：')
    M = InsertAxi(verbose=False)
    FM = M.calibrate_FM(FM_DEFAULT)

    cases = [('完好', None), ('界面 B 脱粘 200 mm', 200.0),
             ('界面 B 脱粘 400 mm', 400.0)]
    S, TR = [], []
    set_elastic(M, True)
    for nm, a in cases:
        M.reset(); M.clear_cut()
        if a:
            M.cut('B', (L_INS - a, L_INS))
        r = M.solve(F_M=FM, F_A=FA_OPER, n_pre=1, n_load=1)
        S.append(M.stresses(r['u']))
        TR.append(M.tractions(r)['B 缠绕层/拉挤块'])
    set_elastic(M, False); M.reset(); M.clear_cut()

    comp = (M.tag != MATS.index('void')) & (M.tag != MATS.index('steel')) \
        & (M.el_xc <= X_VIEW)
    cam = Ortho(az=-62, el=22)

    # ---------------------------------------------------------- fig24
    fig, axs = plt.subplots(3, 2, figsize=(15.5, 8.8))
    vm = float(np.nanpercentile(S[0][comp, 4], 99.0))
    nrm = Normalize(0, vm)
    dmax = max(float(np.nanpercentile(np.abs(S[k][comp, 4] - S[0][comp, 4]), 99.0))
               for k in (1, 2))
    nrm2 = Normalize(-dmax, dmax)
    for i, (nm, a) in enumerate(cases):
        panel_cut(axs[i, 0], cam, M, S[i][:, 4], 'inferno_r', nrm,
                  '(%s) %s　von Mises 应力' % ('abc'[i], nm), deb=a,
                  tag='红线为脱粘段' if a else '')
        if i == 0:
            panel_cut(axs[i, 1], cam, M, S[i][:, 3], 'coolwarm',
                      Normalize(-vm / 3, vm / 3), '(d) 完好　剪应力 τ_rx')
        else:
            panel_cut(axs[i, 1], cam, M, S[i][:, 4] - S[0][:, 4], 'coolwarm',
                      nrm2, '(%s) %s　von Mises 相对完好的变化'
                      % ('ef'[i - 1], nm), deb=a)
    cax = fig.add_axes([0.09, 0.045, 0.32, 0.016])
    plt.colorbar(ScalarMappable(nrm, 'inferno_r'), cax=cax,
                 orientation='horizontal',
                 label='复合材料层 von Mises 应力  [MPa]（钢衬套以灰色实体表示）')
    cax2 = fig.add_axes([0.58, 0.045, 0.32, 0.016])
    plt.colorbar(ScalarMappable(nrm2, 'coolwarm'), cax=cax2,
                 orientation='horizontal', label='Δ von Mises  [MPa]')
    fig.suptitle('图 24　预埋螺套单胞的三维应力场（额定挥舞，单柱外载 %.0f kN）。'
                 '轴对称解绕螺套轴线旋出后沿轴向平面半剖，'
                 '剖面与外表面上的颜色都是计算值。' % (FA_OPER / 1e3),
                 fontsize=12.5, fontweight='bold', y=0.97)
    fig.tight_layout(rect=(0, 0.085, 1, 0.955))
    fig.savefig(FIG + 'fig24_stress3d.png', bbox_inches='tight',
                facecolor='white', pad_inches=0.18)
    plt.close(fig)
    print('  fig24_stress3d.png')

    # ---------------------------------------------------------- fig25
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    # 色标按 x > 5 mm 的分布定，端面第一格的尖峰是端面位移间断造成的奇异值，
    # 用它定色标会把整根管子压成一个颜色。
    mm0 = (TR[0]['x'] > 5.0) & (TR[0]['x'] <= X_VIEW)
    tmax = float(np.percentile(np.abs(TR[0]['ts'][mm0]), 99.0))
    nb = Normalize(0, tmax)
    for i2, (nm, a) in enumerate(cases):
        ax = axs[i2]
        mm = TR[i2]['x'] <= X_VIEW
        F, V = tube_faces(R_WRAP, TR[i2]['x'][mm], np.abs(TR[i2]['ts'][mm]))
        cam.draw(ax, F, np.clip(V, 0, tmax), 'viridis', nb)
        if a:
            q = TR[i2]['x'][(TR[i2]['x'] >= L_INS - a) & mm]
            F2, _ = tube_faces(R_WRAP * 1.015, q, np.zeros(len(q)))
            cam.draw(ax, F2, None, None, None, edge=SIGNAL, shade=False,
                     alpha=0.42)
            pf = cam.to2d(np.array([[L_INS - a, R_WRAP * np.cos(t),
                                     R_WRAP * np.sin(t)]
                                    for t in np.linspace(-np.pi / 2, np.pi / 2, 40)]))
            ax.plot(pf[:, 0], pf[:, 1], color=SIGNAL, lw=2.4)
            ax.text(0.5, 0.86, '红色 = 已脱粘 %.0f mm，裂纹前缘在 x = %.0f mm'
                    % (a, L_INS - a), transform=ax.transAxes, fontsize=10,
                    color=SIGNAL, fontweight='bold', ha='center')
        # 轴向刻度
        for xv in (0, 100, 200, 300, 400, L_INS):
            q = cam.to2d(np.array([[xv, 0, -R_WRAP], [xv, 0, -R_WRAP * 1.5]]))
            ax.plot(q[:, 0], q[:, 1], color=GREY, lw=0.9)
            ax.text(q[1, 0], q[1, 1] - 2, '%d' % xv, fontsize=8.5,
                    color=GREY, ha='center', va='top')
        q = cam.to2d(np.array([[0, 0, -R_WRAP * 1.5], [L_INS, 0, -R_WRAP * 1.5]]))
        ax.plot(q[:, 0], q[:, 1], color=GREY, lw=0.9)

        ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view()
        ax.set_title('(%s) %s' % ('abc'[i2], nm), loc='left', fontsize=12,
                     fontweight='bold')
    cax = fig.add_axes([0.33, 0.05, 0.34, 0.022])
    cb = plt.colorbar(ScalarMappable(nb, 'viridis'), cax=cax,
                      orientation='horizontal',
                      label='界面 B 切向牵引 |τ|  [MPa]（超过 %.1f MPa 的端面尖峰已截顶）'
                            % tmax)
    fig.suptitle('图 25　界面 B（玻纤束缠绕层 / 拉挤 GFRP 块）圆柱面上的切向牵引。'
                 '刻度为轴向 x [mm]，0 = 叶根端面、%.0f = 螺套埋入端。'
                 '脱粘段上牵引归零，载荷被挤向剩余的粘接段与裂纹前缘。'
                 % L_INS,
                 fontsize=12.5, fontweight='bold', y=0.97)
    fig.savefig(FIG + 'fig25_iface3d.png', bbox_inches='tight',
                facecolor='white', pad_inches=0.18)
    plt.close(fig)
    print('  fig25_iface3d.png')


if __name__ == '__main__':
    main()
