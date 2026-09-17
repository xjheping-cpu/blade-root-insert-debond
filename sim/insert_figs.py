# -*- coding: utf-8 -*-
"""M2′ 轴对称分层内聚力模型的图件：fig20 起编号，避免与既有图件冲突。"""
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
from matplotlib.patches import Rectangle, Patch

import insert_study  # noqa: F401 —— 装上弹性模式补丁
from insert_study import set_elastic
from insert_axi import (InsertAxi, IFACES, MATS, MAT_COL, MAT_CN, LAYERS,
                        L_INS, L_ENG, X_BORE, R_CELL, R_BORE, R_STEEL,
                        R_TRANS, R_WRAP, R_BLOCK, FM_DEFAULT, K_STUD)

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10.5
rcParams['savefig.dpi'] = 200
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.20
rcParams['axes.edgecolor'] = '#8FA3B8'

INK = '#12212C'; STEEL = '#2F6690'; SIGNAL = '#D94F2B'
AMBER = '#C8871B'; SAFE = '#1E7A5A'; GREY = '#5A6B77'
FIG = 'figs/'
FA_GRAV, FA_OPER, FA_EXT = 74e3, 159e3, 339e3


def save(f, n):
    f.savefig(FIG + n, bbox_inches='tight', facecolor='white', pad_inches=0.16)
    plt.close(f)
    print('  ' + n)


def polys(M, mirror=True):
    """单元四边形顶点，(x, r) 坐标；mirror=True 时同时给出镜像的下半。"""
    P = [np.stack([M.X[nd], M.R[nd]], 1) for nd in M.el]
    if mirror:
        P = P + [np.stack([M.X[nd], -M.R[nd]], 1) for nd in M.el]
    return P


# ------------------------------------------------------------------ fig20 几何
def fig_geom(M):
    fig, ax = plt.subplots(2, 1, figsize=(15, 8.2),
                           gridspec_kw=dict(height_ratios=[1.45, 1]))
    a = ax[0]
    cols = [MAT_COL[MATS[t]] for t in M.tag]
    pc = PolyCollection(polys(M), facecolors=cols + cols,
                        edgecolors='#33414C', linewidths=0.18)
    a.add_collection(pc)
    a.set_xlim(-25, 720); a.set_ylim(-62, 62)
    a.set_aspect('equal')
    a.set_xlabel('轴向 x  [mm]　（x = 0 为叶根端面，与轮毂法兰接触）')
    a.set_ylabel('到螺套轴线的距离 r  [mm]')
    a.set_title('(a) 单胞的分层构造与网格　单元 %d，自由度 %d'
                % (len(M.el), M.ndof), loc='left', fontsize=12.5,
                fontweight='bold')
    a.plot([0, 0], [-R_CELL, R_CELL], color=INK, lw=2.6)
    a.annotate('轮毂法兰承压面', xy=(0, -R_CELL + 4), xytext=(90, -46),
               fontsize=10, color=INK,
               arrowprops=dict(arrowstyle='->', color=INK, lw=1.1))
    a.annotate('螺柱啮合段 %g mm' % L_ENG, xy=(L_ENG / 2, 24), xytext=(170, 44),
               fontsize=10, color=SIGNAL,
               arrowprops=dict(arrowstyle='->', color=SIGNAL, lw=1.1))
    a.annotate('螺套埋入端 x = %g' % L_INS, xy=(L_INS, 30), xytext=(540, 48),
               fontsize=10, color=STEEL,
               arrowprops=dict(arrowstyle='->', color=STEEL, lw=1.1))
    a.legend(handles=[Patch(fc=MAT_COL[m], ec='#33414C', label=MAT_CN[m])
                      for m in MATS if m != 'void'],
             loc='upper center', bbox_to_anchor=(0.5, -0.24), fontsize=10,
             ncol=5, frameon=False)
    a.grid(False)

    b = ax[1]
    b.set_xlim(-6, 130); b.set_ylim(0, R_CELL + 3)
    bnds = [(0, R_BORE, 'void'), (R_BORE, R_STEEL, 'steel'),
            (R_STEEL, R_TRANS, 'trans'), (R_TRANS, R_WRAP, 'wrap'),
            (R_WRAP, R_BLOCK, 'block'), (R_BLOCK, R_CELL, 'lam')]
    for lo, hi, m in bnds:
        b.add_patch(Rectangle((0, lo), 120, hi - lo, fc=MAT_COL[m],
                              ec='#33414C', lw=0.8))
        L = LAYERS[m]
        txt = MAT_CN[m] if m == 'void' else \
            '%s　E_轴 %.0f GPa，厚 %.1f mm' % (MAT_CN[m], L.Ea / 1e3, hi - lo)
        b.text(124, 0.5 * (lo + hi), txt, va='center', fontsize=10, color=INK)
    for c in IFACES:
        b.plot([0, 120], [c.r, c.r], color=SIGNAL, lw=2.0)
        b.text(-4, c.r, c.name.split()[0], ha='right', va='center',
               fontsize=11, fontweight='bold', color=SIGNAL)
    b.set_ylabel('r  [mm]')
    b.set_xticks([])
    b.set_title('(b) 径向分层与三个内聚力界面　'
                '（文献 He 等 2025 实测失效面在界面 B/C 一带，不在 A）',
                loc='left', fontsize=12.5, fontweight='bold')
    b.grid(False)
    for s in ('top', 'right', 'bottom'):
        b.spines[s].set_visible(False)
    save(fig, 'fig20_layers.png')


# ------------------------------------------------------------------ fig21 应力场
def fig_fields(M, FM):
    cases = [('完好', None), ('界面 B 脱粘 200 mm', 200.0),
             ('界面 B 脱粘 400 mm', 400.0)]
    dat = []
    set_elastic(M, True)
    for nm, a in cases:
        M.reset(); M.clear_cut()
        if a:
            M.cut('B', (L_INS - a, L_INS))
        r = M.solve(F_M=FM, F_A=FA_OPER, n_pre=1, n_load=1)
        dat.append((nm, a, M.stresses(r['u'])))
    set_elastic(M, False); M.reset(); M.clear_cut()

    base = dat[0][2]
    keep = M.tag != MATS.index('void')
    fig, axs = plt.subplots(3, 2, figsize=(15.5, 10.4))
    P = polys(M, mirror=True)
    n = len(M.el)

    def draw(ax, val, vmin, vmax, cmap, title, unit):
        v = np.where(keep, val, np.nan)
        pc = PolyCollection(P, array=np.concatenate([v, v]), cmap=cmap,
                            edgecolors='none')
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(-10, L_INS + 130); ax.set_ylim(-R_CELL - 2, R_CELL + 2)
        ax.set_aspect('equal'); ax.grid(False)
        ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold')
        cb = plt.colorbar(pc, ax=ax, fraction=0.028, pad=0.012)
        cb.ax.tick_params(labelsize=8.5)
        cb.set_label(unit, fontsize=9)
        ax.plot([0, 0], [-R_CELL, R_CELL], color=INK, lw=1.8)
        for c in IFACES:
            ax.plot([0, L_INS], [c.r, c.r], color='#FFFFFF', lw=0.5, alpha=0.7)
            ax.plot([0, L_INS], [-c.r, -c.r], color='#FFFFFF', lw=0.5, alpha=0.7)
        return pc

    vmax_vm = float(np.nanpercentile(np.abs(base[keep, 4]), 99.5))
    vmax_t = float(np.nanpercentile(np.abs(base[keep, 3]), 99.5))
    for i, (nm, a, S) in enumerate(dat):
        draw(axs[i, 0], S[:, 4], 0, vmax_vm, 'inferno_r',
             '(%s) %s　von Mises 应力' % ('abc'[i], nm), 'MPa')
        d = S[:, 4] - base[:, 4]
        lim = max(float(np.nanpercentile(np.abs(np.where(keep, d, np.nan)), 99.5)), 1e-6)
        if i == 0:
            draw(axs[i, 1], S[:, 3], -vmax_t, vmax_t, 'coolwarm',
                 '(d) 完好　剪应力 τ_rx', 'MPa')
        else:
            draw(axs[i, 1], d, -lim, lim, 'coolwarm',
                 '(%s) %s　von Mises 相对完好的变化' % ('_ef'[i], nm), 'MPa')
        if a:
            for s in (1, -1):
                axs[i, 0].plot([L_INS - a, L_INS], [s * R_WRAP, s * R_WRAP],
                               color=SIGNAL, lw=2.4)
                axs[i, 1].plot([L_INS - a, L_INS], [s * R_WRAP, s * R_WRAP],
                               color=SIGNAL, lw=2.4)
    axs[-1, 0].set_xlabel('轴向 x  [mm]')
    axs[-1, 1].set_xlabel('轴向 x  [mm]')
    fig.suptitle('图 21　额定挥舞载荷（单柱外载 %.0f kN）下的应力场：'
                 '完好与两种脱粘状态　红线为脱粘段'
                 % (FA_OPER / 1e3), fontsize=13, fontweight='bold', y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save(fig, 'fig21_fields.png')


# ------------------------------------------------------------------ fig22 界面牵引
def fig_tractions(M, FM):
    d = np.load('out/iface_reconcile.npz', allow_pickle=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cs = {'仅预紧': INK, '重力摆振': STEEL, '额定挥舞': AMBER, '极限': SIGNAL}

    a = ax[0]
    for nm, c in cs.items():
        a.plot(d['x3'], d['thr_' + nm], color=c, lw=2.0, label=nm)
    a.axhline(-IFACES[1].ts0, color=SIGNAL, ls='--', lw=1.3)
    a.text(250, -IFACES[1].ts0 + 0.8, '界面 B 切向强度 %.0f MPa' % IFACES[1].ts0,
           fontsize=9.5, color=SIGNAL)
    a.set_xlabel('轴向 x  [mm]'); a.set_ylabel('切向牵引 τ  [MPa]')
    a.set_title('(a) 轴对称分层模型：界面 B', loc='left',
                fontsize=12, fontweight='bold')
    a.legend(fontsize=9)
    a.set_xlim(0, L_INS)

    b = ax[1]
    for nm, c in cs.items():
        b.plot(d['x2'], d['two_' + nm], color=c, lw=2.0, label=nm)
    b.set_xlabel('轴向 x  [mm]'); b.set_ylabel('名义剪应力 τ  [MPa]')
    b.set_title('(b) 二维环模型：同一界面', loc='left',
                fontsize=12, fontweight='bold')
    b.legend(fontsize=9)
    b.set_xlim(0, L_INS)

    c = ax[2]
    c.plot(d['x3'], d['thr_增量'], color=SIGNAL, lw=2.2, label='轴对称分层模型')
    c.plot(d['x2'], d['two_增量'], color=STEEL, lw=2.2, label='二维环模型')
    c.axhline(0, color=GREY, lw=0.9)
    c.set_xlabel('轴向 x  [mm]'); c.set_ylabel('Δτ（额定 − 预紧）  [MPa]')
    c.set_title('(c) 外载引起的增量：两模型分歧所在', loc='left',
                fontsize=12, fontweight='bold')
    c.legend(fontsize=9)
    c.set_xlim(0, L_INS)
    fig.suptitle('图 22　界面剪切沿轴向的分布。总量峰值两模型一致在端面；'
                 '增量分布不同，源于螺柱载荷引入方式的差别，须由子部件试验裁决。',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, 'fig22_traction.png')


# ------------------------------------------------------------------ fig23 稳定性
def fig_stability():
    d = np.load('out/insert_life.npz')

    def smooth(v, k=3):
        # 轻度滑动平均。锯齿来自端面接触主动集在不同 a 之间切换，非物理。
        w = np.ones(k) / k
        return np.convolve(np.pad(v, (k // 2, k // 2), mode='edge'), w, 'valid')

    a = d['a']
    GA, GB = smooth(d['G_A']), smooth(d['G_B'])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    p = ax[0]
    p.semilogy(a, np.maximum(GA, 1e-6), color=STEEL, lw=2.2, label='界面 A 钢/过渡层')
    p.semilogy(a, np.maximum(GB, 1e-6), color=SIGNAL, lw=2.4, label='界面 B 缠绕层/拉挤块')
    i = int(np.argmin(GB[2:-6])) + 2
    p.plot(a[i], GB[i], 'o', color=SIGNAL, ms=8)
    p.annotate('驱动力谷底 a = %.0f mm' % a[i], xy=(a[i], GB[i]),
               xytext=(70, GB[i] * 2.6), fontsize=10, color=SIGNAL,
               arrowprops=dict(arrowstyle='->', color=SIGNAL, lw=1.1))
    p.axvspan(L_INS - 110, L_INS, color=SIGNAL, alpha=0.08)
    p.text(386, GB.min() * 2.2, '末段 G 急升', fontsize=10.5, color=SIGNAL,
           fontweight='bold', ha='left', rotation=90)
    p.set_xlabel('脱粘长度 a  [mm]（自埋入端起算）')
    p.set_ylabel('能量释放率 G  [N/mm]')
    p.set_title('(a) 驱动力 G(a)：先降后升', loc='left',
                fontsize=12, fontweight='bold')
    p.legend(fontsize=9.5, loc='lower left', framealpha=0.95)
    p.set_xlim(0, L_INS - 10)
    p.set_ylim(GB.min() * 0.45, max(GA.max(), GB.max()) * 3.2)

    q = ax[1]
    for m, c, ls in ((4, STEEL, '-'), (6, AMBER, '--'), (8, SIGNAL, ':')):
        q.plot(a, 100 * d['fr%d' % m], color=c, lw=2.2, ls=ls, label='m = %d' % m)
    q.axhline(50, color=GREY, lw=1.0, ls='--')
    q.axhline(10, color=SIGNAL, lw=1.0, ls='--')
    q.text(10, 12, '只剩 10%', fontsize=9.5, color=SIGNAL)
    q.axvspan(0, 200, color=SAFE, alpha=0.10)
    q.text(18, 62, '预警必须发生在这一段', fontsize=10.5, color=SAFE,
           fontweight='bold')
    q.set_xlabel('已脱粘长度 a  [mm]')
    q.set_ylabel('剩余疲劳寿命占比  [%]')
    q.set_title('(b) 剩余寿命：与未知系数 C 无关', loc='left',
                fontsize=12, fontweight='bold')
    q.legend(fontsize=9.5, title='Paris 指数', title_fontsize=9)
    q.set_xlim(0, L_INS - 10); q.set_ylim(0, 105)

    s = np.load('out/insert_axi_study.npz')
    r = ax[2]
    aa, FS, FKR = s['G_B_deep_a'], s['G_B_deep_FS'], s['G_B_deep_FKR']
    r.plot(aa, 100 * (FS / FS[0] - 1), 'o-', color=STEEL, lw=2.2, ms=5,
           label='螺柱轴力 F_S')
    r.plot(aa, 100 * (FKR / FKR[0] - 1), 's-', color=AMBER, lw=2.2, ms=5,
           label='端面残余夹紧力 F_KR')
    r.axhline(-5, color=SIGNAL, ls='--', lw=1.4)
    r.text(10, -4.6, '常规复紧检查的可分辨限 ±5%', fontsize=9.5, color=SIGNAL)
    r.set_xlabel('脱粘长度 a  [mm]')
    r.set_ylabel('相对完好的变化  [%]')
    r.set_title('(c) 常规检查可检出吗', loc='left', fontsize=12, fontweight='bold')
    r.legend(fontsize=9.5, loc='lower left')
    r.set_xlim(0, L_INS - 10); r.set_ylim(-6, 1.5)
    fig.suptitle('图 23　脱粘扩展的稳定性与可观测性。'
                 '扩展在前 60% 埋深内减速，随后驱动力急升、剩余寿命塌缩；'
                 '而常规检查在全程均无法检出。', fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, 'fig23_stability.png')


def main():
    print('M2′ 图件：')
    M = InsertAxi(verbose=False)
    FM = M.calibrate_FM(FM_DEFAULT)
    for f, args in ((fig_geom, (M,)), (fig_fields, (M, FM)),
                    (fig_tractions, (M, FM)), (fig_stability, ())):
        try:
            f(*args)
        except Exception as e:
            import traceback
            print('  !! %s: %s' % (f.__name__, e))
            traceback.print_exc()


if __name__ == '__main__':
    main()
