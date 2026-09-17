# -*- coding: utf-8 -*-
"""建模重规划用图：观测量对比、模型层次。"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, FancyBboxPatch
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 11
rcParams['savefig.dpi'] = 200
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.22
rcParams['axes.edgecolor'] = '#8FA3B8'

INK = '#12212C'; STEEL = '#2F6690'; SIGNAL = '#D94F2B'
AMBER = '#C8871B'; SAFE = '#1E7A5A'; GREY = '#5A6B77'; HAIR = '#DDE4E8'
FIG = 'figs/'


def save(f, n):
    f.savefig(FIG + n, bbox_inches='tight', facecolor='white', pad_inches=0.15)
    plt.close(f); print('  ' + n)


def fig_observable():
    """两种观测量随脱粘长度的响应对比。"""
    _le = _G.L_ins
    d = np.array([0, 30, 60, 100, 150, 200, 250, 300, 350,
                  _le - 90, _le - 50, _le - 20])
    fib = np.array([0, 430, 457, 410, 335, 292, 244, 248, 250, 316, 370, 324])
    gap = np.zeros_like(d, dtype=float)          # 单位置：全程 <0.1 µm
    cap = (20.0 * np.pi * _G.D_ins * (_G.L_ins - d) / 1e3 / 339.0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    a.semilogy(d, np.maximum(fib, 1), 'o-', color=SIGNAL, lw=2.4, ms=7,
               label='本方案：光纤 Δ 应变 |Δε|')
    a.axhline(13.2, color=SAFE, lw=1.6, ls='--', label='光纤 4σ 判据 13.2 µε')
    a.semilogy(d, np.maximum(gap * 1000, 0.02), 's-', color=GREY, lw=2.2, ms=7,
               label='间隙位移法：端面相对位移（单位置）')
    a.axhline(100, color=INK, lw=1.6, ls=':', label='间隙位移分辨率 0.1 mm = 100 µm')
    a.fill_between([20, _le - 20], 13.2, 2000, color=SAFE, alpha=0.07)
    a.text(120, 700, '光纤可检出区间', color=SAFE, fontsize=11.5, fontweight='bold')
    a.set_ylim(0.02, 2500)
    a.set_xlabel('螺套脱粘长度 [mm]（埋深 %.0f mm）' % _G.L_ins)
    a.set_ylabel('观测量（µε 或 µm，对数轴）')
    a.set_title('(a) 两种观测量对同一损伤的响应', loc='left', fontsize=12.5, fontweight='bold')
    a.legend(fontsize=9.5, loc='lower right')

    b = ax[1]
    ks = np.array([1, 3, 5, 5, 9, 9, 15])
    dl = np.array([300, 300, 300, 400, 400, 460, 460])
    gp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 16.8, 21.4])
    fb = np.array([535, 617, 755, 850, 1029, 1421, 1637])
    lbl = [f'{k}位\n{d}mm' for k, d in zip(ks, dl)]
    xx = np.arange(len(ks))
    b.bar(xx - 0.2, gp, 0.4, color=GREY, label='端面相对位移 [µm]')
    b.bar(xx + 0.2, fb / 20, 0.4, color=SIGNAL, alpha=0.85, label='光纤 |Δε| / 20 [µε]')
    b.axhline(100, color=INK, lw=1.6, ls=':')
    b.text(0.1, 106, '间隙分辨率 0.1 mm', fontsize=10, color=INK)
    b.set_xticks(xx); b.set_xticklabels(lbl, fontsize=9)
    b.set_ylabel('观测量')
    b.set_title('(b) 多位同时脱粘（批次性缺陷）', loc='left', fontsize=12.5, fontweight='bold')
    b.legend(fontsize=9.5, loc='upper left')
    b.set_ylim(0, 120)
    save(fig, 'fig17_observable.png')


def fig_hierarchy():
    """模型层次与文献方法对照。"""
    fig, ax = plt.subplots(figsize=(14, 6.8))
    ax.axis('off'); ax.grid(False)
    ax.set_xlim(0, 100); ax.set_ylim(-2, 70)

    levels = [
        (52, 'M4　整环 / 全叶片', '壳-实体混合，线弹性', '载荷分配、周向改道、基线分布正演', STEEL, 'M1 二维环（保留，降级为快速代理）'),
        (39, 'M3　扇区子模型', '3D 实体 + 子模型切割边界', '内表面应变场、光纤读数正演', STEEL, 'M2 三维扇区（保留，需重建几何与网格）'),
        (26, 'M2′　螺套局部 CZM 模型', '3D 实体 + 双线性内聚力单元', '脱粘起裂与扩展、稳定性判定', SIGNAL, '【新建】文献标准做法，本项目缺失'),
        (13, 'M1′　子部件试件模型', '1:1 复现试验件与夹具', '与试验直接对标、参数反演', SIGNAL, '【新建】He 等 2025 的验证路径'),
        (0, 'M0　一维剪滞', '解析 + 一维有限元', '量级校核、互校', SAFE, '保留不变'),
    ]
    for y, name, how, out, col, note in levels:
        ax.add_patch(FancyBboxPatch((2, y + 1), 44, 9.4, boxstyle='round,pad=0.3',
                                    fc=col, alpha=0.12, ec=col, lw=1.8))
        ax.text(4, y + 7.6, name, fontsize=13, fontweight='bold', color=col)
        ax.text(4, y + 4.9, how, fontsize=10.5, color=INK)
        ax.text(4, y + 2.6, '输出：' + out, fontsize=10, color=GREY)
        ax.text(49, y + 6.2, note, fontsize=11, color=col if '新建' in note else GREY,
                fontweight='bold' if '新建' in note else 'normal')
        if y > 0:
            ax.annotate('', xy=(24, y + 0.8), xytext=(24, y - 1.6),
                        arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.4))
    ax.text(26, 11.4, '参数反演', fontsize=9.5, color=GREY)
    ax.text(26, 24.4, '界面本构传递', fontsize=9.5, color=GREY)
    ax.text(26, 37.4, '子模型边界', fontsize=9.5, color=GREY)
    ax.text(26, 50.4, '载荷与刚度', fontsize=9.5, color=GREY)
    ax.text(49, 65.5, '与现有模型的关系', fontsize=12.5, fontweight='bold', color=INK)
    ax.text(2, 65.5, '模型层次（自下而上标定，自上而下传载）', fontsize=12.5,
            fontweight='bold', color=INK)
    save(fig, 'fig18_hierarchy.png')


if __name__ == '__main__':
    print('建模重规划图件：')
    for f in (fig_observable, fig_hierarchy):
        try:
            f()
        except Exception as e:
            print(f'  !! {f.__name__}: {type(e).__name__}: {e}')
