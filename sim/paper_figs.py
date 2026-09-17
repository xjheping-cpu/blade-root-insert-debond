# -*- coding: utf-8 -*-
"""English figures for the journal manuscript.

Same underlying results as the Chinese report figures, replotted with English
labels and journal-appropriate styling (serif labels, no decorative colour,
single- or double-column widths).

Output: paper/figs/fig01..fig08.{png,pdf}
"""
from __future__ import annotations

import os
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
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 9
rcParams['axes.labelsize'] = 9
rcParams['axes.titlesize'] = 9.5
rcParams['xtick.labelsize'] = 8.5
rcParams['ytick.labelsize'] = 8.5
rcParams['legend.fontsize'] = 8
rcParams['axes.linewidth'] = 0.8
rcParams['savefig.dpi'] = 400
rcParams['figure.dpi'] = 120

INK = '#1a1a1a'
C_STEEL = '#2F6690'
C_RED = '#B3301A'
C_AMBER = '#C8871B'
C_GREEN = '#1E7A5A'
C_GREY = '#6a6a6a'
LAM = '#DCE6EC'
FLANGE = '#C3CCD3'

OUT = 'paper/figs/'
os.makedirs(OUT, exist_ok=True)

# single / double column widths for a typical two-column journal, in inches
W1, W2 = 3.35, 6.9

D3 = np.load('out/sector_layered.npz', allow_pickle=True)
LIFE = np.load('out/insert_life.npz')
JCL = int(D3['iy_cl'])
BASE = D3['intact_surf_eps'][:, JCL]
XC = D3['xc']
CRIT = 13.2          # criterion line, 4 sigma of the differential repeatability
REP = 3.0            # differential repeatability, +-3 ue

# insert geometry, mm
THREAD = 'M42'
L_INS, R_BORE, R_STEEL = _G.L_ins, _G.d_bore / 2, _G.D_ins / 2
X_VIEW = L_INS + 130.0          # 轴向视野上限
# 过渡层/缠绕层/拉挤块的半径按螺套外径顺次外扩，比例取自构造分层
R_TRANS, R_WRAP, R_BLOCK = R_STEEL + 0.5, R_STEEL + 6.5, R_STEEL + 13.5
PITCH, WALL = _G.pitch, _G.t_wall


def save(fig, name):
    fig.savefig(OUT + name + '.png', bbox_inches='tight', pad_inches=0.02,
                facecolor='white')
    fig.savefig(OUT + name + '.pdf', bbox_inches='tight', pad_inches=0.02,
                facecolor='white')
    plt.close(fig)
    print('  ' + name)


def fiber(case):
    """Differential fibre reading along the insert centreline, in microstrain."""
    return (D3[case + '_surf_eps'][:, JCL] - BASE) * 1e6


# ---------------------------------------------------------------- Fig 1
def fig01_problem():
    """(a) ring geometry and the ligament between neighbouring inserts;
       (b) longitudinal section with the three bond interfaces."""
    fig = plt.figure(figsize=(W2, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.75], wspace=0.22)

    # (a) circumferential arrangement
    ax = fig.add_subplot(gs[0])
    R = _G.R_bc                                  # bolt circle radius, mm
    th = np.linspace(-0.16, 0.16, 400)
    for rr, lw in ((R - 55, 0.8), (R + 55, 0.8)):
        ax.plot(rr * np.sin(th), rr * np.cos(th), color=INK, lw=lw)
    for k in (-2, -1, 0, 1, 2):
        t = k * PITCH / R
        cx, cy = R * np.sin(t), R * np.cos(t)
        ci = plt.Circle((cx, cy), R_STEEL, fc=C_STEEL, ec=INK, lw=0.7,
                        alpha=0.9)
        ax.add_patch(ci)
        ax.add_patch(plt.Circle((cx, cy), 18.75, fc='white', ec=INK, lw=0.5))
    t1, t2 = 0.0, PITCH / R
    ax.annotate('', xy=(R * np.sin(t1), R * np.cos(t1)),
                xytext=(R * np.sin(t2), R * np.cos(t2)),
                arrowprops=dict(arrowstyle='<->', color=C_RED, lw=0.9))
    ax.text(R * np.sin(t2 / 2) + 4, R * np.cos(t2 / 2) + 46,
            'pitch %.1f' % PITCH, fontsize=7.5, color=C_RED, ha='center')
    ax.annotate('8.2 mm ligament', xy=(R * np.sin(t2 / 2), R * np.cos(t2 / 2) - 2),
                xytext=(-190, 1485), fontsize=7.5, color=C_RED,
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))
    ax.text(0, 1408, r'%d inserts, %s$\times$%.0f, $\varnothing$%.0f' % (_G.n_bolt, THREAD, _G.L_ins, _G.D_ins),
            fontsize=7.5, ha='center', color=INK)
    ax.set_xlim(-260, 260); ax.set_ylim(1395, 1680)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('(a) blade root ring, detail', loc='left', fontsize=9)

    # (b) longitudinal section. The radial direction is exaggerated: at true
    # scale an L_INS x WALL section leaves no room for the interface lines.
    ax = fig.add_subplot(gs[1])
    ax.add_patch(Rectangle((0, 0), 600, WALL, fc=LAM, ec=INK, lw=0.7))
    ax.add_patch(Rectangle((-90, -14), 70, WALL + 28, fc=FLANGE, ec=INK, lw=0.7))
    for r0, r1, col in ((R_WRAP, R_BLOCK, '#BFD3DE'),
                        (R_TRANS, R_WRAP, '#9CC3B4')):
        for sgn in (1, -1):
            lo = WALL / 2 + (r0 if sgn > 0 else -r1)
            ax.add_patch(Rectangle((0, lo), L_INS, r1 - r0, fc=col, ec='none'))
    ax.add_patch(Rectangle((0, WALL / 2 - R_STEEL), L_INS, 2 * R_STEEL,
                           fc=C_STEEL, ec=INK, lw=0.7, alpha=0.9))
    # blind bore to x = 130; the M42 stud engages the first 105 mm of thread
    ax.add_patch(Rectangle((0, WALL / 2 - R_BORE), 130, 2 * R_BORE,
                           fc='#5B7384', ec=INK, lw=0.5))
    ax.add_patch(Rectangle((-95, WALL / 2 - 21), 200, 42, fc='#8d97a0',
                           ec=INK, lw=0.6))
    for sgn in (1, -1):
        for r, c in ((R_STEEL, C_RED), (R_WRAP, C_AMBER), (R_BLOCK, C_GREEN)):
            ax.plot([0, L_INS], [WALL / 2 + sgn * r] * 2, color=c, lw=1.2)
    for c, nm in ((C_RED, 'A  steel / resin-rich'),
                  (C_AMBER, 'B  wrap / block'),
                  (C_GREEN, 'C  block / laminate')):
        ax.plot([], [], color=c, lw=1.2, label=nm)
    ax.text(245, WALL / 2, 'steel insert', fontsize=7.5, color='white',
            ha='center', va='center')
    ax.annotate('', xy=(L_INS, WALL + 27), xytext=(330, WALL + 27),
                arrowprops=dict(arrowstyle='<-', color=C_RED, lw=1.0))
    ax.text(325, WALL + 27, 'debond grows this way ', fontsize=7.5,
            color=C_RED, ha='right', va='center')
    ax.plot([0, 600], [-8, -8], color=C_RED, lw=1.8)
    ax.text(300, -14, 'distributed optical fibre on the inner surface',
            fontsize=7.5, color=C_RED, ha='center', va='top')
    ax.annotate('', xy=(-96, WALL / 2), xytext=(-175, WALL / 2),
                arrowprops=dict(arrowstyle='->', color=INK, lw=1.2))
    ax.text(-178, WALL / 2, 'blade\nload', fontsize=7.5, ha='right',
            va='center', color=INK)
    ax.text(4, WALL + 6, r'$x=0$', fontsize=7.5)
    ax.text(L_INS, WALL + 6, r'$x=%.0f$' % L_INS, fontsize=7.5, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False,
              fontsize=7.2, ncol=3, handlelength=1.3, columnspacing=1.1)
    ax.set_xlim(-200, X_VIEW); ax.set_ylim(-30, WALL + 42)
    ax.axis('off')
    ax.set_title('(b) longitudinal section through one insert', loc='left',
                 fontsize=9)
    save(fig, 'fig01_problem')


# ---------------------------------------------------------------- Fig 2
def fig02_models():
    """(a) radial stack of M1; (b) material map of the M3 sector in the x-r plane."""
    fig = plt.figure(figsize=(W2, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.85], wspace=0.2)

    ax = fig.add_subplot(gs[0])
    layers = [(0, R_BORE, 'bore / stud', 'white'),
              (R_BORE, R_STEEL, 'steel insert', C_STEEL),
              (R_STEEL, R_TRANS, 'resin-rich 0.5', C_AMBER),
              (R_TRANS, R_WRAP, 'glass wrap', '#9CC3B4'),
              (R_WRAP, R_BLOCK, 'pultruded block', '#BFD3DE'),
              (R_BLOCK, 54.63, 'root laminate', LAM)]
    # the 0.5 mm resin-rich layer is thinner than its own label, so the labels
    # are put on a ladder and joined to the layer by a leader line
    label_y = [9, 28, 40.5, 46.0, 51.0, 55.5]
    for (r0, r1, nm, col), ly in zip(layers, label_y):
        ax.add_patch(Rectangle((0, r0), 1, r1 - r0, fc=col, ec=INK, lw=0.6))
        ax.plot([1.0, 1.12], [(r0 + r1) / 2, ly], color=C_GREY, lw=0.5)
        ax.text(1.16, ly, '%s, %.2f–%.2f mm' % (nm, r0, r1),
                fontsize=6.8, va='center')
    for r, c, nm in ((R_STEEL, C_RED, 'A'), (R_WRAP, C_AMBER, 'B'),
                     (R_BLOCK, C_GREEN, 'C')):
        ax.plot([0, 1], [r, r], color=c, lw=1.6)
        ax.text(-0.12, r, nm, fontsize=8, color=c, ha='right', va='center',
                fontweight='bold')
    ax.text(-0.12, 28, 'cohesive\ninterfaces', fontsize=7, color=INK,
            ha='right', va='center')
    ax.set_xlim(-0.75, 3.0); ax.set_ylim(0, 57)
    ax.axis('off')
    ax.set_title('(a) M1 unit cell, radial build-up', loc='left', fontsize=9)

    ax = fig.add_subplot(gs[1])
    mat = D3['mat']
    xc, rc = D3['xc'], D3['rc']
    sl = mat[:, JCL, :].T
    # mat_names order in the npz is lam, steel, void, wrap, block
    cmap = matplotlib.colors.ListedColormap(
        [LAM, C_STEEL, 'white', '#9CC3B4', '#BFD3DE'][:int(sl.max()) + 1])
    ax.pcolormesh(xc, rc, sl, cmap=cmap, edgecolors='#9aa5ad', linewidth=0.08)
    ax.set_xlabel('axial coordinate $x$  [mm]')
    ax.set_ylabel('radial position\nfrom inner surface  [mm]')
    ax.set_xlim(0, X_VIEW)
    ax.text(0.98, 0.08, '106 600 elements, 340 201 DOF',
            transform=ax.transAxes, ha='right', fontsize=7.5, color=INK)
    ax.set_title('(b) M3 layered sector, material map on the cell centreline',
                 loc='left', fontsize=9)
    save(fig, 'fig02_models')


# ---------------------------------------------------------------- Fig 3
def fig03_blindspot():
    """The central result: what the stud-based checks read, against remaining life."""
    a = LIFE['a']
    fs = (LIFE['FS_B'] / LIFE['FS_B'][0] - 1) * 100
    fk = (LIFE['FKR_B'] / LIFE['FKR_B'][0] - 1) * 100

    fig, ax = plt.subplots(figsize=(W2, 2.8))
    ax.axhspan(-5, 5, color=C_GREY, alpha=0.13, zorder=0)
    ax.text(8, 4.3, 'field-resolvable band of a re-torque check, $\\pm$5 %',
            fontsize=7.5, color=C_GREY, va='top')
    ax.plot(a, fs, color=C_STEEL, lw=1.5, label='stud axial force $F_S$')
    ax.plot(a, fk, color=C_AMBER, lw=1.5, ls='--',
            label='residual clamp force at the end face $F_{KR}$')
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel('debond length $a$ from the buried end  [mm]')
    ax.set_ylabel('relative change of the\nstud-based measurands  [%]')
    ax.set_ylim(-6.5, 5.6)
    ax.set_xlim(0, 475)

    ax2 = ax.twinx()
    ax2.fill_between(a, LIFE['fr4'] * 100, LIFE['fr8'] * 100, color=C_GREEN,
                     alpha=0.16, lw=0)
    ax2.plot(a, LIFE['fr6'] * 100, color=C_GREEN, lw=1.8,
             label='remaining fatigue life ($m=6$; band: $m=4\\dots8$)')
    ax2.set_ylabel('remaining fatigue life  [%]', color=C_GREEN)
    ax2.tick_params(axis='y', colors=C_GREEN)
    ax2.set_ylim(-8, 112)

    ax.annotate('$-0.12$ %\nat $a=400$ mm (82 % of embedment)',
                xy=(400, fs[np.argmin(np.abs(a - 400))]), xytext=(238, -4.9),
                fontsize=7.8, color=C_STEEL, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_STEEL, lw=0.8))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower left', frameon=False,
              bbox_to_anchor=(0.015, 0.50))
    ax.grid(alpha=0.15, lw=0.5)
    save(fig, 'fig03_blindspot')


# ---------------------------------------------------------------- Fig 4
def fig04_margins():
    """Load-carrying margin and remaining fatigue life are exhausted at
       different debond lengths."""
    a = LIFE['a']
    cap0, ult = 2771.0, 339.0                    # kN, from the M1 study
    cap = cap0 * (L_INS - a) / L_INS
    a50 = float(np.interp(50, LIFE['fr6'][::-1] * 100, a[::-1]))
    acr = float(LIFE['crit'][1])

    fig, ax = plt.subplots(figsize=(W2, 2.7))
    ax.plot(a, LIFE['fr6'] * 100, color=C_GREEN, lw=1.8,
            label='remaining fatigue life')
    ax.set_xlabel('debond length $a$  [mm]')
    ax.set_ylabel('remaining fatigue life  [%]', color=C_GREEN)
    ax.tick_params(axis='y', colors=C_GREEN)
    ax.set_ylim(-4, WALL - 2); ax.set_xlim(0, L_INS)

    ax2 = ax.twinx()
    ax2.plot(a, cap, color=C_STEEL, lw=1.5, ls='--',
             label='remaining interface capacity')
    ax2.axhline(ult, color=C_RED, lw=1.0, ls=':')
    ax2.text(12, ult + 70, 'ultimate demand per stud', fontsize=7.5, color=C_RED)
    ax2.set_ylabel('interface capacity  [kN]', color=C_STEEL)
    ax2.tick_params(axis='y', colors=C_STEEL)
    ax2.set_ylim(-100, 2900)

    ax.axvspan(a50, acr, color=C_AMBER, alpha=0.15, lw=0)
    for xv, c, txt in ((a50, C_GREEN, 'half the fatigue life spent\n$a=295$ mm (60 %)'),
                       (acr, C_STEEL, 'capacity exhausted\n$a=430$ mm (88 %)')):
        ax.axvline(xv, color=c, lw=0.9, ls='-.')
    ax.annotate('half the fatigue life spent\n$a = 295$ mm (60 % of embedment)',
                xy=(a50, 50), xytext=(150, 66), fontsize=7.8, color=C_GREEN,
                ha='center', arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=0.8))
    ax.annotate('capacity exhausted, pull-out\n$a = 430$ mm (88 %)',
                xy=(acr, 4), xytext=(398, 30), fontsize=7.8, color=C_STEEL,
                ha='center', arrowprops=dict(arrowstyle='->', color=C_STEEL, lw=0.8))
    ax.text((a50 + acr) / 2, 88, '135 mm', fontsize=8, color=C_AMBER,
            ha='center', fontweight='bold')
    ax.annotate('', xy=(a50, 84), xytext=(acr, 84),
                arrowprops=dict(arrowstyle='<->', color=C_AMBER, lw=1.0))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', frameon=False,
              bbox_to_anchor=(0.995, 0.74))
    ax.grid(alpha=0.15, lw=0.5)
    save(fig, 'fig04_margins')


# ---------------------------------------------------------------- Fig 5
def fig05_forward():
    """Forward-modelled inner-surface strain field and the fibre reading."""
    cases = [('intact', 'intact'), ('debB_200', r'$a=200$ mm'),
             ('debB_400', r'$a=400$ mm'), ('debB_460', r'$a=460$ mm')]
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.0),
                             gridspec_kw=dict(height_ratios=[1.0, 1.15],
                                              hspace=0.42, wspace=0.16))
    lim = max(np.abs((D3[c + '_surf_eps'] - D3['intact_surf_eps']).max() * 1e6)
              for c, _ in cases[1:])
    yc = D3['yc'] - 213.0
    for k, (c, lab) in enumerate(cases):
        f2 = (D3[c + '_surf_eps'] - D3['intact_surf_eps']) * 1e6
        ax = axes[0, k]
        m = ax.pcolormesh(XC, yc, f2.T, cmap='RdBu_r', vmin=-lim, vmax=lim,
                          shading='auto', rasterized=True)
        ax.set_xlim(0, X_VIEW); ax.set_ylim(-130, 130)
        ax.set_title(lab, fontsize=8.5)
        if k == 0:
            ax.set_ylabel('circumferential\noffset  [mm]')
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.5)
        ax.set_xticks([0, 300, 600])

        bx = axes[1, k]
        y = fiber(c) if c != 'intact' else np.zeros_like(XC)
        bx.axhspan(-CRIT, CRIT, color=C_GREY, alpha=0.16, lw=0)
        bx.plot(XC, y, color=C_RED, lw=1.2)
        bx.set_xlim(0, X_VIEW); bx.set_ylim(-115, 100)
        bx.set_xlabel('$x$  [mm]', fontsize=8)
        bx.tick_params(labelsize=7.5)
        bx.set_xticks([0, 300, 600])
        if k == 0:
            bx.set_ylabel('fibre reading\n$\\Delta\\varepsilon$  [$\\mu\\varepsilon$]')
            bx.text(20, -100, 'criterion line $\\pm$13.2', fontsize=7,
                    color=C_GREY)
        else:
            bx.set_yticklabels([])
        bx.grid(alpha=0.15, lw=0.4)
    cb = fig.colorbar(m, ax=axes[0, :].tolist(), fraction=0.02, pad=0.012)
    cb.set_label('$\\Delta\\varepsilon_{xx}$  [$\\mu\\varepsilon$]', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    save(fig, 'fig05_forward')


# ---------------------------------------------------------------- Fig 6
def fig06_criterion():
    """Two-segment criterion: the buried-end trough saturates, the face
       segment does not."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5),
                             gridspec_kw=dict(width_ratios=[1.55, 1.0],
                                              wspace=0.28))
    ax = axes[0]
    ax.axhspan(-CRIT, CRIT, color=C_GREY, alpha=0.16, lw=0)
    for c, lab, col, ls in (('debB_200', r'$a=200$ mm', C_GREEN, '-'),
                            ('debB_400', r'$a=400$ mm', C_AMBER, '--'),
                            ('debB_460', r'$a=460$ mm', C_RED, '-.')):
        ax.plot(XC, fiber(c), color=col, lw=1.3, ls=ls, label=lab)
    ax.axvspan(0, 110, color=C_STEEL, alpha=0.09, lw=0)
    ax.axvspan(L_INS - 20, L_INS + 70, color=C_RED, alpha=0.09, lw=0)
    ax.text(55, 86, 'face segment', fontsize=7.5, color=C_STEEL, ha='center')
    ax.text(515, 86, 'buried end', fontsize=7.5, color=C_RED, ha='center')
    ax.set_xlabel('$x$  [mm]')
    ax.set_ylabel('$\\Delta\\varepsilon$  [$\\mu\\varepsilon$]')
    ax.set_xlim(0, X_VIEW); ax.set_ylim(-88, 100)
    ax.legend(loc='lower left', frameon=False, ncol=1)
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_title('(a) differential fibre reading', loc='left', fontsize=9)

    ax = axes[1]
    aa = [200, 400, 460]
    trough = [abs(fiber('debB_%d' % v)[XC > L_INS - 20].min()) for v in aa]
    face = [abs(fiber('debB_%d' % v)[XC < 110]).max() for v in aa]
    xx = np.arange(3)
    ax.bar(xx - 0.19, trough, 0.36, color=C_RED, label='buried-end trough')
    ax.bar(xx + 0.19, face, 0.36, color=C_STEEL, label='face segment')
    ax.axhline(CRIT, color=INK, lw=0.8, ls=':')
    ax.text(2.42, CRIT * 1.25, 'criterion', fontsize=7, ha='right')
    ax.set_yscale('log')
    ax.set_xticks(xx); ax.set_xticklabels(['200', '400', '460'])
    ax.set_xlabel('debond length $a$  [mm]')
    ax.set_ylabel('$|\\Delta\\varepsilon|$  [$\\mu\\varepsilon$]')
    ax.set_ylim(0.5, 400)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('(b) amplitude of the two segments', loc='left', fontsize=9)
    save(fig, 'fig06_criterion')


# ---------------------------------------------------------------- Fig 7
def fig07_discriminate():
    """A broken stud and an insert debond are separated in space."""
    fig, ax = plt.subplots(figsize=(W1 * 1.55, 2.3))
    ax.axhspan(-CRIT, CRIT, color=C_GREY, alpha=0.16, lw=0)
    ax.plot(XC, fiber('broken'), color=C_STEEL, lw=1.4, label='broken stud')
    ax.plot(XC, fiber('debB_400'), color=C_RED, lw=1.4, ls='--',
            label='insert debond, $a=400$ mm')
    i1 = int(np.argmax(fiber('broken')))
    i2 = int(np.argmin(fiber('debB_400')))
    ax.annotate('', xy=(XC[i1], 84), xytext=(XC[i2], 84),
                arrowprops=dict(arrowstyle='<->', color=INK, lw=0.9))
    ax.text((XC[i1] + XC[i2]) / 2, 88, '%.0f mm apart' % (XC[i2] - XC[i1]),
            fontsize=7.8, ha='center')
    ax.text(0.985, 0.06, 'spatial resolution 1.3 mm', transform=ax.transAxes,
            fontsize=7.5, ha='right', color=C_GREY)
    ax.set_xlabel('$x$  [mm]')
    ax.set_ylabel('$\\Delta\\varepsilon$  [$\\mu\\varepsilon$]')
    ax.set_xlim(0, X_VIEW); ax.set_ylim(-80, 104)
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=0.15, lw=0.5)
    save(fig, 'fig07_discriminate')


# ---------------------------------------------------------------- Fig 8
def fig08_coverage():
    """Circumferential decay: one pitch away the signal is already gone."""
    ix = int(np.argmin(fiber('debB_400')))
    yc = D3['yc'] - 213.0
    fig, ax = plt.subplots(figsize=(W1 * 1.55, 2.3))
    for c, lab, col, ls in (('debB_400', 'single insert, $a=400$ mm', C_RED, '-'),
                            ('debB_400_x3', 'three adjacent inserts, $a=400$ mm',
                             C_STEEL, '--')):
        prof = (D3[c + '_surf_eps'][ix, :] - D3['intact_surf_eps'][ix, :]) * 1e6
        ax.plot(yc, prof, color=col, lw=1.4, ls=ls, label=lab)
    ax.axhspan(-CRIT, CRIT, color=C_GREY, alpha=0.16, lw=0)
    for k in (1, 2):
        for sgn in (1, -1):
            ax.axvline(sgn * k * PITCH, color=INK, lw=0.5, ls=':')
    ax.text(PITCH, 14, ' 1 pitch', fontsize=7.2, va='bottom')
    ax.text(2 * PITCH, 14, ' 2 pitches', fontsize=7.2, va='bottom')
    ax.annotate('$-2.4\\,\\mu\\varepsilon$ at the neighbouring bolt',
                xy=(PITCH, -2.4), xytext=(150, -62), fontsize=7.5, color=C_RED,
                ha='center', arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))
    ax.set_xlabel('circumferential offset from the defect centre  [mm]')
    ax.set_ylabel('$\\Delta\\varepsilon$  [$\\mu\\varepsilon$]')
    ax.set_xlim(-213, 213); ax.set_ylim(-112, 42)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.15, lw=0.5)
    save(fig, 'fig08_coverage')


def main():
    print('English figures for the manuscript:')
    for f in (fig01_problem, fig02_models, fig03_blindspot, fig04_margins,
              fig05_forward, fig06_criterion, fig07_discriminate, fig08_coverage):
        try:
            f()
        except Exception as e:
            import traceback
            print('  !! %s: %s' % (f.__name__, e))
            traceback.print_exc()


if __name__ == '__main__':
    main()
