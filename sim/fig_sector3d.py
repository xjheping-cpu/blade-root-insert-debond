# -*- coding: utf-8 -*-
"""
分层三维扇区模型（sector3d_layered）的结果渲染：真实曲率的叶根壁块，
场量画在内表面、外表面与两个剖切面上。

与 fig24/fig25 的分工：那两张来自轴对称单胞模型，周向没有变化；
本图来自真三维模型，周向变化是它独有的内容——相邻螺套之间怎么改道、
光纤贴的内表面上测得的是什么，只有这个模型能回答。
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from insert_fig3d import Ortho
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10.5
rcParams['savefig.dpi'] = 200
rcParams['axes.grid'] = False

INK = '#12212C'; STEEL = '#2F6690'; SIGNAL = '#D94F2B'
AMBER = '#C8871B'; SAFE = '#1E7A5A'; GREY = '#5A6B77'
FIG = 'figs/'
D = np.load('out/sector_layered.npz', allow_pickle=True)
R_IN = float(D['R_in']); R_BC = float(D['R_bolt'])
X, Y, R = D['x'], D['y'], D['r']
XC, YC, RC = D['xc'], D['yc'], D['rc']
DY = float(D['pitch']) / int(D['m_y'])
X_VIEW = 620.0


def pt(x, y, r):
    """(轴向, 周向弧长, 壁厚方向) → 真实三维坐标。周向角 θ = y / R_bolt。"""
    th = (y - Y.mean()) / R_BC
    rr = R_IN + r
    # 径向取反，使 r = 0 的内表面朝上：光纤贴在内表面，图要给读者看的是那一面
    return np.array([x, rr * np.sin(th), R_IN + 55.0 - rr * np.cos(th)])


def surf_faces(vals, ir, xs=None):
    """给定壁厚层 ir 的一张曲面（内表面 ir=0，外表面 ir=-1）。"""
    F, V = [], []
    ie = np.flatnonzero(XC <= X_VIEW)
    rv = R[0] if ir == 0 else R[-1]
    for i in ie:
        for j in range(len(YC)):
            y0, y1 = Y[j], Y[j] + DY
            F.append([pt(X[i], y0, rv), pt(X[i + 1], y0, rv),
                      pt(X[i + 1], y1, rv), pt(X[i], y1, rv)])
            V.append(vals[i, j, 0 if ir == 0 else -1])
    return np.array(F), np.array(V)


def cut_faces(vals, jside):
    """周向两端的剖切面。jside=0 取 y 最小侧，-1 取 y 最大侧。"""
    yv = Y[0] if jside == 0 else Y[-1] + DY
    j = 0 if jside == 0 else len(YC) - 1
    F, V = [], []
    for i in np.flatnonzero(XC <= X_VIEW):
        for k in range(len(RC)):
            F.append([pt(X[i], yv, R[k]), pt(X[i + 1], yv, R[k]),
                      pt(X[i + 1], yv, R[k + 1]), pt(X[i], yv, R[k + 1])])
            V.append(vals[i, j, k])
    return np.array(F), np.array(V)


def end_faces(vals, ix):
    """x 端面。ix=-1 时取视野边界处的截面，避免画出一块脱离本体的面片。"""
    if ix == 0:
        xv, i = X[0], 0
    else:
        i = int(np.flatnonzero(XC <= X_VIEW)[-1])
        xv = X[i + 1]
    F, V = [], []
    for j in range(len(YC)):
        y0, y1 = Y[j], Y[j] + DY
        for k in range(len(RC)):
            F.append([pt(xv, y0, R[k]), pt(xv, y1, R[k]),
                      pt(xv, y1, R[k + 1]), pt(xv, y0, R[k + 1])])
            V.append(vals[i, j, k])
    return np.array(F), np.array(V)


def panel(ax, cam, vals, cmap, norm, title, mark_cells=(), deb_x=None):
    F, V = [], []
    for f, v in (surf_faces(vals, 0), surf_faces(vals, -1),
                 cut_faces(vals, 0), cut_faces(vals, -1),
                 end_faces(vals, 0), end_faces(vals, -1)):
        F.append(f); V.append(v)
    cam.draw(ax, np.concatenate(F), np.concatenate(V), cmap, norm)
    # 螺套在内表面上的投影：宽度等于螺套外径
    _rs = _G.D_ins / 2
    xs0 = np.linspace(0.0, _G.L_ins, 30)
    for jc in mark_cells:
        yc0 = (jc + 0.5) * float(D['pitch'])
        for dy in (-_rs, _rs):
            q = cam.to2d(np.array([pt(xv, yc0 + dy, R[0] + 0.5) for xv in xs0]))
            ax.plot(q[:, 0], q[:, 1], color='#FFFFFF', lw=0.8, alpha=0.75)
        q = cam.to2d(np.array([pt(_G.L_ins, yc0 + t, R[0] + 0.5)
                               for t in np.linspace(-_rs, _rs, 12)]))
        ax.plot(q[:, 0], q[:, 1], color='#FFFFFF', lw=0.8, alpha=0.75)
    if deb_x is not None:
        for jc in mark_cells:
            yc0 = (jc + 0.5) * float(D['pitch'])
            xs = np.linspace(deb_x, _G.L_ins, 20)
            for dy in (-_rs, _rs):
                q = cam.to2d(np.array([pt(xv, yc0 + dy, R[0] + 0.9)
                                       for xv in xs]))
                ax.plot(q[:, 0], q[:, 1], color=SIGNAL, lw=2.2)
    ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
    ax.set_aspect('equal'); ax.axis('off'); ax.autoscale_view()


def main():
    print('分层三维扇区图件：')
    cam = Ortho(az=-66, el=26)
    base = D['intact_eps_xx']
    JC = int(D['JC'])
    NC = int(D['n_cell'])
    ALL = tuple(range(NC))
    cases = [('debB_400', '界面 B 脱粘 400 mm（单个螺套）', (JC,), 90.0),
             ('debB_400_x3', '界面 B 脱粘 400 mm（3 个相邻螺套）',
              (JC - 1, JC, JC + 1), 90.0),
             ('debC_400', '界面 C 脱粘 400 mm（单个螺套）', (JC,), 90.0),
             ('broken', '螺柱断裂（单个位置）', (JC,), None)]

    fig, axs = plt.subplots(3, 2, figsize=(15.5, 11.2))
    nrm = Normalize(*np.percentile(base[..., 0] * 1e6, [1, 99]))
    panel(axs[0, 0], cam, base * 1e6, 'viridis', nrm,
          '(a) 完好　轴向应变 ε_xx', mark_cells=ALL)
    plt.colorbar(ScalarMappable(nrm, 'viridis'), ax=axs[0, 0],
                 fraction=0.022, pad=0.01, label='µε')
    sv = D['intact_svm']
    nsv = Normalize(0, float(np.percentile(sv, 99)))
    panel(axs[0, 1], cam, sv, 'inferno_r', nsv,
          '(b) 完好　von Mises 应力', mark_cells=ALL)
    plt.colorbar(ScalarMappable(nsv, 'inferno_r'), ax=axs[0, 1],
                 fraction=0.022, pad=0.01, label='MPa')

    # 色标按内表面定：面板讲的是光纤实测的量，用整个壁厚的分位数会把它压平
    lim = max(float(np.abs((D[c[0] + '_surf_eps'] - D['intact_surf_eps'])
                           * 1e6).max()) for c in cases) * 1.05
    n2 = Normalize(-lim, lim)
    for k, (nm, ttl, cells, dx) in enumerate(cases):
        ax = axs[1 + k // 2, k % 2]
        dd = (D[nm + '_eps_xx'] - base) * 1e6
        panel(ax, cam, dd, 'coolwarm', n2,
              '(%s) %s' % ('cdef'[k], ttl), mark_cells=cells, deb_x=dx)
        pk = np.abs((D[nm + '_surf_eps'] - D['intact_surf_eps']) * 1e6)
        i, j = np.unravel_index(np.argmax(pk), pk.shape)
        ax.text(0.02, 0.06, '内表面 |Δε| 峰值 %.0f µε，在 x = %.0f mm'
                % (pk[i, j], XC[i]), transform=ax.transAxes, fontsize=10,
                color=SIGNAL, fontweight='bold')
    cax = fig.add_axes([0.36, 0.052, 0.30, 0.014])
    plt.colorbar(ScalarMappable(n2, 'coolwarm'), cax=cax,
                 orientation='horizontal', label='Δ轴向应变（相对完好）  [µε]')
    fig.suptitle('图 26　分层三维扇区模型：%d 个螺套的叶根壁块（真实曲率，'
                 '周向 %.0f mm、壁厚 %.0f mm，%s 单元 / %s 自由度）。'
                 '视角自叶根内部看内表面，光纤即贴在这一面。'
                 '白色带为螺套在内表面上的投影（宽 %.0f mm），红线为脱粘段。'
                 '额定挥舞，单柱外载 %.0f kN。'
                 % (NC, NC * float(D['pitch']), _G.t_wall,
                    format(int(D['n_elem']), ','),
                    format(int(D['ndof']), ','), _G.D_ins,
                    float(D['FA_target']) / 1e3),
                 fontsize=12.5, fontweight='bold', y=0.965)
    fig.tight_layout(rect=(0, 0.085, 1, 0.95))
    fig.savefig(FIG + 'fig26_sector3d.png', bbox_inches='tight',
                facecolor='white', pad_inches=0.18)
    plt.close(fig)
    print('  fig26_sector3d.png')

    # ------------------------------------------------ fig27 光纤读数正演
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    a = ax[0]
    jcl = int(D['iy_cl'])
    a.plot(XC, D['intact_surf_eps'][:, jcl] * 1e6, color=INK, lw=2.4,
           label='完好')
    cs = {'debB_200': (STEEL, '界面 B 脱粘 200 mm'),
          'debB_400': (AMBER, '界面 B 脱粘 400 mm'),
          'debC_400': (SAFE, '界面 C 脱粘 400 mm'),
          'debB_400_x3': (SIGNAL, '界面 B 脱粘 400 mm × 3 相邻')}
    for nm, (c, lb) in cs.items():
        a.plot(XC, D[nm + '_surf_eps'][:, jcl] * 1e6, color=c, lw=2.0, label=lb)
    a.axvspan(0, _G.L_ins, color=GREY, alpha=0.07)
    a.text(150, -10, '螺套埋深范围', fontsize=10, color=GREY)
    a.set_xlim(0, X_VIEW); a.set_xlabel('轴向 x  [mm]')
    a.set_ylabel('内表面轴向应变  [µε]')
    a.set_title('(a) 光纤所在的内表面上，应变沿轴向的分布', loc='left',
                fontsize=12, fontweight='bold')
    a.legend(fontsize=9.5, loc='upper left')
    a.grid(alpha=0.2)

    b = ax[1]
    for nm, (c, lb) in cs.items():
        b.plot(XC, (D[nm + '_surf_eps'][:, jcl] - D['intact_surf_eps'][:, jcl])
               * 1e6, color=c, lw=2.2, label=lb)
    b.plot(XC, (D['broken_surf_eps'][:, jcl] - D['intact_surf_eps'][:, jcl])
           * 1e6, color='#7B3FA0', lw=2.2, ls='--', label='螺柱断裂')
    b.axhline(13.2, color=SIGNAL, ls=':', lw=1.6)
    b.axhline(-13.2, color=SIGNAL, ls=':', lw=1.6)
    b.text(612, 18, '光纤 4σ 判据 ±13.2 µε', fontsize=10, color=SIGNAL,
           ha='right')
    b.axvline(_G.L_ins, color=GREY, lw=1.2, ls='--')
    b.annotate('螺套埋入端 x = %.0f' % _G.L_ins, xy=(_G.L_ins, 62),
               xytext=(300, 66),
               fontsize=9.5, color=GREY,
               arrowprops=dict(arrowstyle='->', color=GREY, lw=1.0))
    b.set_xlim(0, X_VIEW); b.set_xlabel('轴向 x  [mm]')
    b.set_ylabel('Δ应变（相对完好）  [µε]')
    b.set_title('(b) 光纤实测的量：与基线之差', loc='left',
                fontsize=12, fontweight='bold')
    b.legend(fontsize=9.5, loc='lower left', ncol=2, framealpha=0.95)
    b.grid(alpha=0.2)
    fig.suptitle('图 27　三维模型正演的光纤读数。脱粘的信号峰出现在螺套埋入端之后（x ≈ 500 mm），'
                 '螺柱断裂的信号峰出现在端面一侧（x < 100 mm），两者位置不同，可据此区分。',
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG + 'fig27_fiber_fwd.png', bbox_inches='tight',
                facecolor='white', pad_inches=0.16)
    plt.close(fig)
    print('  fig27_fiber_fwd.png')


if __name__ == '__main__':
    main()
