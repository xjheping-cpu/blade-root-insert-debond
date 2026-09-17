# -*- coding: utf-8 -*-
"""拔出通道的图件。"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 9
rcParams['figure.dpi'] = 150; rcParams['savefig.dpi'] = 150
rcParams['axes.grid'] = True; rcParams['grid.alpha'] = 0.25; rcParams['grid.linewidth'] = 0.5

C_INK='#1C2530'; C_BLUE='#2B5D8C'; C_RED='#D2452F'; C_AMBER='#D08A1E'
C_GREEN='#2E8B57'; C_GREY='#8FA3B8'
FIG = 'figs/'

from root_model import RootFE
from pullout2 import ishear, insert_force, solve_w, FS_INSTALL, FA_GRAV, FA_OPER, FA_EXT

TAU_ULT = (15.0, 25.0); TAU_DEG = (8.0, 12.0)


def save(f, n):
    f.tight_layout(); f.savefig(FIG + n, bbox_inches='tight', facecolor='white')
    plt.close(f); print('  ' + n)


def fig_mech():
    fe = RootFE(n_cells=25, m_y=4); j = 12
    FM = fe.calibrate_FM(FS_INSTALL)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    for FA, c, lab in ((0.0, C_GREY, '仅预紧 420 kN'), (FA_GRAV, C_GREEN, '+重力 74 kN'),
                       (FA_OPER, C_BLUE, '+额定 159 kN'), (FA_EXT, C_RED, '+极端 339 kN')):
        _, r = solve_w(fe, FM, FA)
        x, t = ishear(fe, r, j)
        ax[0].plot(x, np.abs(t), color=c, lw=1.7, label=lab)
    ax[0].axhspan(TAU_ULT[0], TAU_ULT[1], color=C_GREEN, alpha=0.18)
    ax[0].axhspan(TAU_DEG[0], TAU_DEG[1], color=C_RED, alpha=0.14)
    ax[0].text(20, 20, '完好界面强度 15~25 MPa', fontsize=7.5, color=C_GREEN)
    ax[0].text(20, 9.3, '缺陷界面 8~12 MPa', fontsize=7.5, color=C_RED)
    ax[0].set_xlabel('沿螺套埋深 x [mm]'); ax[0].set_ylabel('界面剪应力 |τ| [MPa]')
    ax[0].set_title('(a) 界面剪应力：峰值在埋入端，不在端面', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5, loc='upper left')

    _, r = solve_w(fe, FM, FA_EXT)
    xc, Ns = insert_force(fe, r, j)
    ax[1].plot(xc, Ns / 1e3, color=C_RED, lw=2)
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].axvspan(0, 105, color=C_AMBER, alpha=0.15)
    ax[1].text(8, 260, '螺纹啮合段\n0~105 mm', fontsize=8, color=C_AMBER)
    ax[1].annotate('端面接触把大部分\n螺柱力直接顶回法兰', xy=(5, -90), xytext=(120, -60),
                   fontsize=7.5, color=C_INK,
                   arrowprops=dict(arrowstyle='->', lw=0.8, color=C_INK))
    ax[1].annotate('剩余 ≈150 kN 在\n埋入端交给层压',
                   xy=(_G.L_ins - 10, 60), xytext=(280, 150),
                   fontsize=7.5, color=C_RED,
                   arrowprops=dict(arrowstyle='->', lw=0.8, color=C_RED))
    ax[1].set_xlabel('x [mm]'); ax[1].set_ylabel('螺套轴力 $N_s$ [kN]')
    ax[1].set_title('(b) 载荷在螺套里怎么走（极端工况）', loc='left', fontsize=10)

    gaps = [0.0, 0.02, 0.05, 0.10, 0.20]
    p_face, p_deep, chis = [], [], []
    for gp in gaps:
        FMg = fe.calibrate_FM(FS_INSTALL, gap_s=gp)
        _, r0 = solve_w(fe, FMg, 0.0, gap_s=gp)
        _, re = solve_w(fe, FMg, FA_EXT, gap_s=gp)
        x, t = ishear(fe, re, j)
        p_face.append(np.abs(t[x <= 150]).max()); p_deep.append(np.abs(t[x >= 400]).max())
        chis.append(r0['chi'].mean())
    ax[2].plot(gaps, p_deep, 'o-', color=C_RED, lw=1.8, label='埋入端峰值')
    ax[2].plot(gaps, p_face, 's-', color=C_BLUE, lw=1.8, label='端面段峰值')
    ax[2].axhspan(TAU_ULT[0], TAU_ULT[1], color=C_GREEN, alpha=0.18)
    ax[2].set_xlabel('螺套端面内缩量 gap$_s$ [mm]'); ax[2].set_ylabel('界面剪应力峰值 [MPa]')
    ax[2].set_title('(c) 不论端面怎么造，埋入端始终是控制位置', loc='left', fontsize=9.5)
    ax[2].legend(fontsize=8)
    save(fig, 'fig13_pullout_mech.png')


def fig_growth():
    cap = np.load('out/pullout_capacity.npy')
    sig = np.load('out/pullout_signal.npy')
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ax[0].plot(cap[:, 0], cap[:, 4], 'o-', color=C_BLUE, lw=2, label='塑性承载上界')
    ax[0].axhline(FA_EXT / 1e3, color=C_RED, lw=1.5, ls='--', label='极端载荷 339 kN')
    ax[0].axhline(FA_OPER / 1e3, color=C_AMBER, lw=1.2, ls=':', label='额定 159 kN')
    dcrit = np.interp(FA_EXT / 1e3, cap[::-1, 4], cap[::-1, 0])
    ax[0].axvline(dcrit, color=C_RED, lw=1, ls='-.')
    ax[0].text(dcrit - 190, 1600, f'承载力降到极端载荷\n脱粘 {dcrit:.0f} mm', fontsize=8, color=C_RED)
    ax[0].set_xlabel('脱粘长度 [mm]'); ax[0].set_ylabel('剩余承载力 [kN]')
    ax[0].set_title('(a) 脱粘 420 mm 之前承载力都够', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5)

    ax[1].plot(cap[:, 0], cap[:, 3], 'o-', color=C_RED, lw=1.8, label='极端工况峰值 τ')
    ax[1].plot(cap[:, 0], cap[:, 2], 's-', color=C_BLUE, lw=1.6, label='额定工况峰值 τ')
    ax[1].axhspan(TAU_ULT[0], TAU_ULT[1], color=C_GREEN, alpha=0.18)
    ax[1].set_xlabel('脱粘长度 [mm]'); ax[1].set_ylabel('剩余粘接段的峰值 τ [MPa]')
    ax[1].set_title('(b) 峰值自相似：扩展过程长期稳定', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5)

    d = sig[:, 0]; dip = np.abs(sig[:, 2]); face = np.abs(sig[:, 1])
    ax[2].semilogy(d, np.maximum(dip, 1), 'o-', color=C_RED, lw=1.8, label='凹陷幅值')
    ax[2].semilogy(d, np.maximum(face, 1), 's-', color=C_BLUE, lw=1.6, label='端面特征 Δ')
    ax[2].axhline(4 * 3.3, color=C_GREEN, lw=1.4, ls='--', label='模式B 5年 4σ = 13 µε')
    ax[2].axhline(6 * 3.3, color=C_AMBER, lw=1.2, ls=':', label='6σ 稳健线 = 20 µε')
    ax[2].axvline(dcrit, color=C_RED, lw=1, ls='-.')
    ax[2].text(dcrit - 150, 2.5, '承载力开始掉', fontsize=8, color=C_RED)
    ax[2].set_xlabel('脱粘长度 [mm]'); ax[2].set_ylabel('OFDR 信号 [µε]')
    ax[2].set_title('(c) 20~30 mm 就能测到，420 mm 才失效', loc='left', fontsize=10)
    ax[2].legend(fontsize=7)
    save(fig, 'fig14_pullout_growth.png')


def fig_cascade():
    fe2 = RootFE(n_cells=41, m_y=4); FM = fe2.calibrate_FM(FS_INSTALL)
    jj, tot = 20, FA_EXT * 41
    ks = [0, 1, 2, 3, 5, 8, 12]
    FAn, TAU, MAR = [], [], []
    for k in ks:
        deb = ({(jj + i) % 41: _G.L_ins
                for i in range(-(k // 2), k - k // 2)} if k else {})
        r = fe2.solve(0.0, FM, debond=deb); r1 = fe2.solve(-1e-3, FM, debond=deb)
        kk = (r1['FA'].sum() - r['FA'].sum()) / (-1e-3); w = 0.0
        for _ in range(15):
            if abs(r['FA'].sum() - tot) < 1e3:
                break
            w += (tot - r['FA'].sum()) / kk
            r = fe2.solve(w, FM, debond=deb)
        nb = (jj - k // 2 - 1) % 41
        x, t = ishear(fe2, r, nb)
        FAn.append(r['FA'][nb] / 1e3); TAU.append(np.abs(t).max())
        MAR.append(np.mean(TAU_ULT) * np.pi * _G.D_ins * _G.L_ins / 1e3
                   / (r['FA'][nb] / 1e3))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ax[0].plot(ks, np.array(FAn) / FAn[0], 'o-', color=C_RED, lw=1.8, label='邻位外载')
    ax[0].plot(ks, np.array(TAU) / TAU[0], 's-', color=C_BLUE, lw=1.8, label='邻位界面峰值 τ')
    ax[0].set_xlabel('已拔出的相邻螺套个数'); ax[0].set_ylabel('相对完好状态')
    ax[0].set_title('(a) 载荷涨 45%，界面应力只涨 4%', loc='left', fontsize=10)
    ax[0].legend(fontsize=8)
    ax[0].annotate('多出来的载荷沿长度\n进入层压，没堆到埋入端', xy=(8, 1.05), xytext=(2.2, 1.25),
                   fontsize=7.5, color=C_INK,
                   arrowprops=dict(arrowstyle='->', lw=0.8, color=C_INK))

    ax[1].plot(ks, MAR, 'o-', color=C_GREEN, lw=2, label='完好界面 15~25 MPa')
    ax[1].plot(ks, np.array(MAR) * np.mean(TAU_DEG) / np.mean(TAU_ULT), 's-',
               color=C_AMBER, lw=1.8, label='缺陷界面 8~12 MPa')
    ax[1].plot(ks, np.array(MAR) * 5.0 / np.mean(TAU_ULT), '^-', color=C_RED, lw=1.8,
               label='已脱粘 50% 的缺陷界面')
    ax[1].axhline(1.0, color='k', lw=1.2, ls='--')
    ax[1].text(6, 1.15, '裕度 = 1：拔出', fontsize=8)
    ax[1].set_yscale('log'); ax[1].set_xlabel('已拔出的相邻螺套个数')
    ax[1].set_ylabel('紧邻完好位的静拔出裕度')
    ax[1].set_title('(b) 只有界面本身已劣化才会失稳', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5)

    m = 6.0
    ov = np.array(TAU) / TAU[0]
    ax[2].plot(ks, ov ** m, 'o-', color=C_BLUE, lw=1.8, label='本模型（τ 只涨 4%）')
    ax[2].plot(ks, (np.array(FAn) / FAn[0]) ** m, 's--', color=C_GREY, lw=1.5,
               label='若按载荷比例估（错）')
    ax[2].set_yscale('log'); ax[2].set_xlabel('已拔出的相邻螺套个数')
    ax[2].set_ylabel('邻位界面疲劳扩展速率放大倍数')
    ax[2].set_title('(c) 级联不是"拉链式"，而是慢速的', loc='left', fontsize=9.5)
    ax[2].legend(fontsize=7.5)
    save(fig, 'fig15_cascade.png')


def fig_warning():
    fig, ax = plt.subplots(1, 2, figsize=(12.0, 3.6),
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    a = ax[0]
    stages = [(0, 40, '孕育：界面微损伤\n（制造缺陷 / 极端事件啃噬）', C_GREY),
              (40, 90, 'OFDR 可测\n脱粘 30~40 mm 起', C_GREEN),
              (90, 97, '承载力开始下降\n脱粘 >420 mm', C_AMBER),
              (97, 100, '拔出\n→开裂→坠落', C_RED)]
    for x0, x1, lab, c in stages:
        a.add_patch(Rectangle((x0, 0.3), x1 - x0, 0.4, fc=c, alpha=0.35, ec=c, lw=1.2))
        a.text((x0 + x1) / 2, 0.5, lab, ha='center', va='center', fontsize=8.5)
    a.annotate('', xy=(40, 0.18), xytext=(97, 0.18),
               arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=2))
    a.text(68, 0.10, '预警窗口 = 脱粘全程的 90%', ha='center', fontsize=10,
           color=C_GREEN, fontweight='bold')
    a.set_xlim(0, 108); a.set_ylim(0, 1.05); a.axis('off')
    a.set_title('(a) 拔出通道的时间轴与预警窗口', loc='left', fontsize=10.5)

    b = ax[1]
    items = ['界面剪应力峰值位置\n（埋入端 x≈%.0f）' % _G.L_ins,
             '可测脱粘长度\n30~40 mm',
             '承载力开始掉\n420 mm', '级联放大\n仅 1.3 倍', '预警窗口\n≈90% 寿命']
    vals = [1, 1, 1, 1, 1]
    cols = [C_BLUE, C_GREEN, C_AMBER, C_BLUE, C_GREEN]
    b.barh(range(5), vals, color=cols, alpha=0.75)
    for i, s in enumerate(items):
        b.text(0.03, i, s, va='center', fontsize=8.5, color='white', fontweight='bold')
    b.set_xlim(0, 1); b.set_yticks([]); b.set_xticks([]); b.grid(False)
    for sp in b.spines.values():
        sp.set_visible(False)
    b.set_title('(b) 五个决定性数字', loc='left', fontsize=10.5)
    save(fig, 'fig16_warning.png')


if __name__ == '__main__':
    print('拔出通道图件：')
    for f in (fig_mech, fig_growth, fig_cascade, fig_warning):
        try:
            f()
        except Exception as e:
            print(f'  !! {f.__name__}: {type(e).__name__}: {e}')
