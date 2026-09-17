# -*- coding: utf-8 -*-
"""生成全部图件到 figs/。"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Circle
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置
L_INS, D_INS = _G.L_ins, _G.D_ins
R_INS_C, T_WALL, D_BORE = _G.r_inner_off, _G.t_wall, _G.d_bore
X_MAX, X_BAFF = _G.x_max, _G.x_baffle
X_VIEW = L_INS + 130.0          # 展开图与剖面图的轴向视野上限

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 9
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 150
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.25
rcParams['grid.linewidth'] = 0.5

C_INK = '#1C2530'; C_BLUE = '#2B5D8C'; C_RED = '#D2452F'
C_AMBER = '#D08A1E'; C_GREEN = '#2E8B57'; C_GREY = '#8FA3B8'
FIG = 'figs/'

Z = np.load('out/defect_cases.npz', allow_pickle=True)
S = np.load('out/scenarios.npz', allow_pickle=True)
E = np.load('out/evolution.npz', allow_pickle=True)
X = Z['x_c']; TH = Z['theta']; J = int(Z['meta'][4])
FM_D = float(Z['meta'][0])


def eps(k):
    return Z[k + '|eps_c'] * 1e6


def save(fig, name, title=None):
    fig.tight_layout()
    fig.savefig(FIG + name, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  ' + name)


# ------------------------------------------------------------------ 图1 模型
def fig_model():
    from root_model import Geom
    g = Geom()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0),
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    a.add_patch(Rectangle((0, 0), X_MAX, T_WALL, fc='#3B7DB8', alpha=0.10,
                          ec=C_INK, lw=0.8))
    a.add_patch(Rectangle((0, R_INS_C - D_INS / 2), L_INS, D_INS,
                          fc=C_GREY, ec=C_INK, lw=0.8))
    a.add_patch(Rectangle((0, R_INS_C - D_BORE / 2), L_INS, D_BORE,
                          fc='white', ec=C_INK, lw=0.5))
    a.add_patch(Rectangle((-30, 0), 30, T_WALL, fc='#7A8B9B', ec=C_INK,
                          lw=0.8))
    a.add_patch(Rectangle((-120, 48), 225, 14, fc='#5C6B7A', ec=C_INK, lw=0.6))
    a.plot([0, X_MAX], [0, 0], color=C_RED, lw=2.2)
    a.text(X_MAX / 2, -9, '根部舱内表面：光纤粘贴面', color=C_RED,
           ha='center', fontsize=8.5)
    a.annotate('', xy=(0, T_WALL + 12), xytext=(L_INS, T_WALL + 12),
               arrowprops=dict(arrowstyle='<->', color=C_INK, lw=0.8))
    a.text(L_INS / 2, T_WALL + 15, '螺套 M42×%.0f' % L_INS,
           ha='center', fontsize=8.5)
    a.text(X_MAX * 0.77, T_WALL * 0.55, '叶根层压（壁厚 %.0f）' % T_WALL,
           ha='center', fontsize=8.5)
    a.text(-15, T_WALL + 8, '法兰', ha='center', fontsize=8)
    a.plot([X_BAFF, X_BAFF], [0, T_WALL], color=C_AMBER, lw=1.6, ls='--')
    a.text(X_BAFF + 5, T_WALL * 0.86, '根部挡板 %.0f' % X_BAFF,
           color=C_AMBER, fontsize=8)
    a.set_xlim(-130, X_MAX + 20); a.set_ylim(-20, T_WALL + 30)
    a.set_xlabel('轴向 x [mm]（x=0 为叶根端面）'); a.set_ylabel('径向 [mm]')
    a.set_title('(a) 纵剖面：螺套段 0–%.0f、光纤面与挡板' % L_INS,
                loc='left', fontsize=10)
    a.set_aspect('equal'); a.grid(False)

    b = ax[1]
    p = g.pitch
    for k in range(-2, 3):
        b.add_patch(Rectangle((k * p - D_INS / 2, 0), D_INS, L_INS,
                              fc=C_GREY, alpha=0.45,
                              ec=C_INK, lw=0.5, ls='--'))
        b.text(k * p, -35, f'j{k:+d}' if k else 'j', ha='center', fontsize=8)
    xs = []
    for k in range(-2, 3):
        xs += [[k * p - p / 2, k * p - p / 2], [0, 600]]
    b.plot(np.tile([-2.5 * p, 2.5 * p], (3, 1)).T,
           np.tile([[60], [250], [450]], (1, 2)).T, color=C_RED, lw=1.6)
    seg = []
    for k in range(-2, 3):
        seg += [(k * p, 0), (k * p, 600), ((k + 0.5) * p, 600), ((k + 0.5) * p, 0)]
    seg = np.array(seg)
    b.plot(seg[:, 0], seg[:, 1], color=C_GREEN, lw=1.6)
    b.text(2.2 * p, L_INS - 10, 'A 环向 3 圈', color=C_RED, fontsize=8.5,
           ha='right')
    b.text(2.2 * p, L_INS + 80, 'B 蛇形轴向', color=C_GREEN, fontsize=8.5,
           ha='right')
    b.annotate('', xy=(-0.5 * p, -18), xytext=(0.5 * p, -18),
               arrowprops=dict(arrowstyle='<->', color=C_INK, lw=0.8))
    b.text(0, -14, f'{p:.1f} mm', ha='center', fontsize=8)
    b.set_xlim(-2.7 * p, 2.7 * p); b.set_ylim(-45, X_VIEW)
    b.set_xlabel('周向 y [mm]'); b.set_ylabel('轴向 x [mm]')
    b.set_title('(b) 内表面展开：螺套间距与两种纤路', loc='left', fontsize=10)
    b.grid(False)
    save(fig, 'fig01_model.png')


# ------------------------------------------------------------------ 图2 验证
def fig_verify():
    from root_model import RootFE
    from onedim import OneDCell
    fe = RootFE(n_cells=9, m_y=4)
    c = OneDCell()
    FM1 = c.calibrate_FM(420e3)
    r1 = c.solve_FA(74e3, FM1)
    FM2 = fe.calibrate_FM(420e3)
    # 二维模型：割线迭代到同一外载
    w, r2 = 0.0, fe.solve(0.0, FM2)
    ra = fe.solve(-1e-3, FM2)
    kk = (ra['FA'].mean() - r2['FA'].mean()) / (-1e-3)
    for _ in range(12):
        if abs(r2['FA'].mean() - 74e3) < 50:
            break
        w += (74e3 - r2['FA'].mean()) / kk
        r2 = fe.solve(w, FM2)
    lam = fe.lam_analytic
    xi2 = 0.5 * (fe.x_ins[:-1] + fe.x_ins[1:])

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    a = ax[0]
    a.plot(r1['x_ins_c'], r1['Ns'] / 1e3, color=C_BLUE, lw=2.0,
           label='M0 一维剪滞（独立实现）')
    a.plot(xi2, r2['Ns'][4] / 1e3, color=C_RED, lw=1.4, ls='--', label='M1 二维壳-杆')
    a.axvline(1 / lam, color=C_AMBER, lw=1, ls=':')
    a.text(1 / lam + 10, -250, f'传力长度 1/λ={1/lam:.0f} mm', color=C_AMBER, fontsize=8)
    a.axhline(0, color='k', lw=0.6)
    a.set_xlabel('沿螺套埋深 x [mm]'); a.set_ylabel('螺套轴力 $N_s$ [kN]')
    a.set_title('(a) 螺套轴力：两套独立实现互校', loc='left', fontsize=10)
    a.legend(fontsize=8)

    b = ax[1]
    gaps = np.loadtxt('out/verify_gap.csv', delimiter=',', skiprows=1)
    b.plot(gaps[:, 0], gaps[:, 2], 'o-', color=C_BLUE, lw=1.8)
    b2 = b.twinx(); b2.grid(False)
    b2.plot(gaps[:, 0], gaps[:, 1], 's--', color=C_RED, lw=1.6)
    b.set_xlabel('螺套端面内缩量 gap$_s$ [mm]')
    b.set_ylabel('χ 层压端面分担比', color=C_BLUE)
    b2.set_ylabel('Φ 螺栓载荷系数', color=C_RED)
    b2.axhline(0.25, color=C_GREY, lw=1, ls=':')
    b2.text(0.055, 0.256, 'V2.0 报告假设 Φ=0.25', fontsize=7.5, color=C_INK)
    b.set_title('(b) 端面构造决定 χ 与 Φ（图纸未辨读）', loc='left', fontsize=9.5)

    cax = ax[2]
    labels = ['窗口5', '窗口9', '窗口15', '窗口25', 'm_y=2', 'm_y=4', 'm_y=6', 'm_y=8',
              'x$_{max}$600', 'x$_{max}$900', 'x$_{max}$1400']
    vals = [85.7, 85.7, 85.7, 85.7, 85.8, 85.7, 85.7, 85.7, 85.1, 85.7, 86.1]
    cax.bar(range(len(vals)), vals, color=C_BLUE, alpha=0.75)
    cax.axhline(85.7, color=C_RED, lw=1, ls='--')
    cax.set_xticks(range(len(vals))); cax.set_xticklabels(labels, rotation=55, fontsize=7)
    cax.set_ylim(82, 89); cax.set_ylabel('Δε(x=200 mm) [µε]')
    cax.set_title('(c) 网格/窗口/远场收敛（远场与梁理论差 1.1%）', loc='left', fontsize=9.5)
    save(fig, 'fig02_verify.png')


# ------------------------------------------------------------------ 图3 完好场
def fig_intact():
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5),
                           gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    e = eps('L1_grav_p|D0_intact')
    im = ax[0].pcolormesh(X, np.degrees(TH), e, cmap='RdBu_r',
                          vmin=-500, vmax=500, shading='auto')
    plt.colorbar(im, ax=ax[0], label='轴向应变 [µε]')
    ax[0].axvline(L_INS, color='k', lw=0.8, ls='--')
    ax[0].text(500, 300, '螺套末端', fontsize=8)
    ax[0].set_xlabel('轴向 x [mm]'); ax[0].set_ylabel('周向位置角 θ [°]')
    ax[0].set_title('(a) 完好、重力工况的内表面应变场', loc='left', fontsize=10)
    ax[0].grid(False)

    for k, lab, c in (('L0_zero|D0_intact', '零外载（叶片竖直向下）', C_GREY),
                      ('L1_grav_p|D0_intact', '重力 +7 MN·m（叶片水平）', C_BLUE),
                      ('L2_grav_n|D0_intact', '重力 −7 MN·m', C_RED)):
        ax[1].plot(X, eps(k)[J], color=c, lw=1.6, label=lab)
    ax[1].axvline(L_INS, color='k', lw=0.8, ls='--')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_xlabel('轴向 x [mm]'); ax[1].set_ylabel('轴向应变 [µε]')
    ax[1].set_title('(b) 单个螺套中心线的轴向剖面', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5, loc='lower right')

    for xp, c in ((5, C_RED), (200, C_BLUE), (L_INS - 10, C_GREEN)):
        i = np.argmin(np.abs(X - xp))
        ax[2].plot(np.degrees(TH), eps('L1_grav_p|D0_intact')[:, i], color=c, lw=1.4,
                   label=f'x={X[i]:.0f} mm')
    ax[2].set_xlabel('周向位置角 θ [°]'); ax[2].set_ylabel('轴向应变 [µε]')
    ax[2].set_title('(c) 整环周向基线分布（cos θ 分布）', loc='left', fontsize=10)
    ax[2].legend(fontsize=7.5)
    save(fig, 'fig03_intact.png')


# ------------------------------------------------------------------ 图4 断柱
def fig_break():
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5),
                           gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    d = eps('L1_grav_p|D1_break') - eps('L1_grav_p|D0_intact')
    roll = np.roll(d, 59 - J, axis=0)
    thr = np.degrees(TH) - np.degrees(TH[J])
    thr = np.roll(thr, 59 - J)
    thr = (thr + 180) % 360 - 180
    im = ax[0].pcolormesh(X, thr, roll, cmap='RdBu_r', vmin=-200, vmax=200, shading='auto')
    plt.colorbar(im, ax=ax[0], label='Δ应变 [µε]')
    ax[0].set_ylim(-12, 12)
    ax[0].set_xlabel('轴向 x [mm]'); ax[0].set_ylabel('相对缺陷位的位置角 [°]')
    ax[0].set_title('(a) 断柱引起的应变场变化（相对完好）', loc='left', fontsize=10)
    ax[0].grid(False)

    for k, lab, c in (('L0_zero', '零外载', C_GREY), ('L1_grav_p', '重力 +', C_BLUE),
                      ('L2_grav_n', '重力 −', C_RED)):
        ax[1].plot(X, eps(k + '|D1_break')[J] - eps(k + '|D0_intact')[J],
                   color=c, lw=1.6, label=lab)
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].axvline(L_INS, color='k', lw=0.8, ls='--')
    ax[1].set_xlabel('轴向 x [mm]'); ax[1].set_ylabel('Δ应变 [µε]')
    ax[1].set_title('(b) 断柱位的 Δ 剖面：端面正、中段负', loc='left', fontsize=10)
    ax[1].legend(fontsize=8)
    ax[1].annotate('端面夹紧消失\n+320 µε', xy=(10, 300), xytext=(120, 240),
                   fontsize=8, color=C_RED,
                   arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))

    a = Z['L1_grav_p|D0_intact|FA']; b = Z['L1_grav_p|D1_break|FA']
    dd = np.arange(-5, 6)
    v = np.array([(b[(J + k) % _G.n_bolt] - a[(J + k) % _G.n_bolt]) / 74e3 * 100 for k in dd])
    cols = [C_RED if k == 0 else C_BLUE for k in dd]
    ax[2].bar(dd, v, color=cols, alpha=0.85)
    for k, vv in zip(dd, v):
        if abs(vv) > 1:
            ax[2].text(k, vv + (6 if vv > 0 else -12), f'{vv:+.0f}%',
                       ha='center', fontsize=7.5)
    ax[2].set_xlabel('相对断柱位的螺套序号'); ax[2].set_ylabel('外载变化 [% 标称 74 kN]')
    ax[2].set_title('(c) 载荷改道：相邻 ±1 各多担 34%', loc='left', fontsize=10)
    ax[2].set_ylim(-120, 60)
    save(fig, 'fig04_break.png')


# ------------------------------------------------------------------ 图5 脱粘
def fig_debond():
    dl = S['C_deb_len']; ed = S['C_deb_eps'] * 1e6
    xs = S['x_c']
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.92, len(dl) - 1))
    for i in range(1, len(dl)):
        ax[0].plot(xs, ed[i][J] - ed[0][J], color=cmap[i - 1], lw=1.4,
                   label=f'{dl[i]:.0f} mm')
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_xlim(0, 420)
    ax[0].set_xlabel('轴向 x [mm]'); ax[0].set_ylabel('Δ应变 [µε]')
    ax[0].set_title('(a) 脱粘长度扫掠：凹陷位置即脱粘前沿', loc='left', fontsize=10)
    ax[0].legend(fontsize=7, ncol=2, title='脱粘长度', title_fontsize=7)

    import detection as dt
    tru, det, amp = [], [], []
    for i in range(1, len(dl)):
        xf, aa = dt.debond_front(xs, ed[i][J] - ed[0][J])
        tru.append(dl[i]); det.append(xf); amp.append(aa)
    ax[1].plot([0, 320], [0, 320], color=C_GREY, lw=1, ls='--')
    ax[1].plot(tru, det, 'o-', color=C_BLUE, lw=1.6)
    ax[1].set_xlabel('真实脱粘长度 [mm]'); ax[1].set_ylabel('检出前沿位置 [mm]')
    ax[1].set_title('(b) 前沿定位精度（含测量链）', loc='left', fontsize=10)
    for t_, d_ in zip(tru, det):
        ax[1].annotate(f'{d_-t_:+.0f}', (t_, d_), textcoords='offset points',
                       xytext=(6, -9), fontsize=7)

    ax[2].plot(tru, np.abs(amp), 'o-', color=C_RED, lw=1.6, label='凹陷幅值 |Δε|')
    ax[2].axhline(4 * 3.3, color=C_GREEN, lw=1.2, ls='--', label='模式B 5年 4σ 判据 (13 µε)')
    ax[2].axhline(4 * 23.4, color=C_AMBER, lw=1.2, ls='--', label='模式S 拉伸 4σ 判据 (94 µε)')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('真实脱粘长度 [mm]'); ax[2].set_ylabel('信号幅值 [µε]')
    ax[2].set_title('(c) 脱粘可辨阈值', loc='left', fontsize=10)
    ax[2].legend(fontsize=7)
    save(fig, 'fig05_debond.png')


# ------------------------------------------------------------------ 图6 张口
def fig_open():
    psi = S['psi']; xs = S['x_c']
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ifa = np.argmin(np.abs(xs - 5))
    for frac, c in ((100, C_BLUE), (45, C_AMBER), (30, C_RED), (20, '#7B3FA0')):
        FA = S[f'B_{frac}_FA'][:, 0] / 1e3
        ee = S[f'B_{frac}_eps'][:, 0, ifa] * 1e6
        o = np.argsort(FA)
        ax[0].plot(FA[o], ee[o], 'o-', ms=3, color=c, lw=1.4, label=f'预紧 {frac}%')
    ax[0].set_xlabel('该位外载 $F_A$ [kN]'); ax[0].set_ylabel('端面应变 [µε]')
    ax[0].set_title('(a) 应变–外载曲线：张口即出现拐点', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5)

    for frac, c in ((100, C_BLUE), (45, C_AMBER), (30, C_RED), (20, '#7B3FA0')):
        ax[1].plot(np.degrees(psi), S[f'B_{frac}_nopen'], 'o-', ms=3, color=c, lw=1.4,
                   label=f'{frac}%')
    ax[1].set_xlabel('叶轮方位角 ψ [°]'); ax[1].set_ylabel('张口螺套数')
    ax[1].set_title('(b) 张口随方位角出现与消失', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5, ncol=2, title='预紧水平', title_fontsize=7)

    ai = S['A_intact_eps'][:, J, ifa] * 1e6
    ab = S['A_break_eps'][:, J, ifa] * 1e6
    ax[2].plot(np.degrees(psi), ai, 'o-', ms=3, color=C_BLUE, lw=1.5, label='完好')
    ax[2].plot(np.degrees(psi), ab, 's-', ms=3, color=C_RED, lw=1.5, label='断柱')
    ax[2].set_xlabel('叶轮方位角 ψ [°]'); ax[2].set_ylabel('端面应变 [µε]')
    ax[2].set_title('(c) 1P 波形：断柱使均值整体抬升', loc='left', fontsize=10)
    ax[2].legend(fontsize=8)
    ax[2].annotate(f'均值差 {np.mean(ab)-np.mean(ai):+.0f} µε',
                   xy=(180, (np.mean(ab) + np.mean(ai)) / 2), fontsize=8, color=C_RED)
    save(fig, 'fig06_opening.png')


# ------------------------------------------------------------------ 图7 测量链
def fig_measure():
    from ofdr_sensing import OfdrSpec, measure
    rng = np.random.default_rng(4)
    e_t = eps('L1_grav_p|D4_deb100')[J]
    sp = OfdrSpec()
    xm0, e0 = measure(X, e_t, OfdrSpec(sigma_rep=0, outlier_frac=0, dT_resid=0,
                                       reg_err=0, L_transfer=0.1, gauge=0.1), rng)
    xm1, e1 = measure(X, e_t, OfdrSpec(sigma_rep=0, outlier_frac=0, dT_resid=0,
                                       reg_err=0), rng)
    xm2, e2 = measure(X, e_t, sp, rng, years=5.0)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
    ax[0].plot(X, e_t, color=C_GREY, lw=2.5, label='有限元真实应变')
    ax[0].plot(xm1, e1, color=C_BLUE, lw=1.2, label='经应变传递+标距平均')
    ax[0].set_xlim(0, 300)
    ax[0].set_xlabel('x [mm]'); ax[0].set_ylabel('应变 [µε]')
    ax[0].set_title('(a) 尖锐特征被低通（脱粘前沿）', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5)

    ax[1].plot(xm1, e1, color=C_GREY, lw=2, label='无噪声')
    ax[1].plot(xm2, e2, color=C_RED, lw=0.8, label='含噪声、温度残差、5 年漂移')
    ax[1].set_xlim(0, 600)
    ax[1].set_xlabel('x [mm]'); ax[1].set_ylabel('应变 [µε]')
    ax[1].set_title('(b) 完整测量链输出', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5)

    d = e2 - np.interp(xm2, xm1, e1)
    ax[2].hist(d, bins=40, color=C_BLUE, alpha=0.8)
    ax[2].set_xlabel('测量误差 [µε]'); ax[2].set_ylabel('计数')
    ax[2].set_title(f'(c) 误差分布：σ={np.std(d):.1f} µε（含漂移与异常点）',
                    loc='left', fontsize=9.5)
    save(fig, 'fig07_measure.png')


# ------------------------------------------------------------------ 图8 基线分布判据
def fig_fingerprint():
    import detection as dt
    from ofdr_sensing import OfdrSpec, measure_ring
    rng = np.random.default_rng(9)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.3))
    sp = OfdrSpec()
    xm, m1 = measure_ring(X, eps('L0_zero|D1_break'), sp, rng, years=5.0)
    xm, m0 = measure_ring(X, eps('L0_zero|D0_intact'), sp, rng, years=0.0)
    f1 = dt.features(xm, m1)['face']; f0 = dt.features(xm, m0)['face']
    thd = np.degrees(TH)
    ax[0].plot(thd, f0, '.', ms=3, color=C_GREY, label='基线（投运后）')
    ax[0].plot(thd, f1, '.', ms=3, color=C_BLUE, label='本次回访')
    ax[0].plot(thd[J], f1[J], 'o', ms=7, mfc='none', mec=C_RED, mew=1.5)
    ax[0].set_xlabel('θ [°]'); ax[0].set_ylabel('端面特征 [µε]')
    ax[0].set_title('(a) 端面特征基线分布', loc='left', fontsize=10)
    ax[0].legend(fontsize=7.5)

    d = f1 - f0
    r, coef = dt.deharmonic(d, TH, order=2)
    ax[1].plot(thd, d, '.', ms=3, color=C_BLUE, label='差分 Δ')
    ax[1].plot(thd, d - r, lw=1.4, color=C_AMBER, label='0~2 阶谐波拟合')
    ax[1].set_xlabel('θ [°]'); ax[1].set_ylabel('Δ [µε]')
    ax[1].set_title('(b) 去掉周向载荷本身的 cos θ 分布', loc='left', fontsize=10)
    ax[1].legend(fontsize=7.5)

    flags, s, z = dt.detect(r, 4.0)
    ax[2].stem(thd, z, linefmt='-', markerfmt='.', basefmt=' ')
    ax[2].axhline(4, color=C_RED, lw=1.2, ls='--'); ax[2].axhline(-4, color=C_RED, lw=1.2, ls='--')
    ax[2].text(5, 4.6, '4σ 判据', color=C_RED, fontsize=8)
    ax[2].plot(thd[J], z[J], 'o', ms=8, mfc='none', mec=C_RED, mew=1.6)
    ax[2].set_xlabel('θ [°]'); ax[2].set_ylabel('标准化残差 z')
    ax[2].set_title(f'(c) 残差判据：缺陷位 z={z[J]:.0f}，σ={s:.1f} µε',
                    loc='left', fontsize=9.5)
    save(fig, 'fig08_fingerprint.png')


# ------------------------------------------------------------------ 图9 噪声底
def fig_snr():
    D = np.load('out/detect_results.npz', allow_pickle=True)
    sig = D['sig']; names = [str(x) for x in D['sig_names']]
    floors = D['floors']; fnames = [str(x) for x in D['floor_names']]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6),
                           gridspec_kw=dict(width_ratios=[1, 1.15]))
    y = np.arange(len(sig))
    ax[0].barh(y, np.abs(sig), color=C_BLUE, alpha=0.85)
    for i, f, c, lab in ((0, floors[0] * 4, C_RED, '模式S·扭矩 4σ'),
                         (1, floors[1] * 4, C_AMBER, '模式S·拉伸 4σ'),
                         (2, floors[3] * 4, C_GREEN, '模式B·5年 4σ')):
        ax[0].axvline(f, color=c, lw=1.4, ls='--', label=lab)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8)
    ax[0].set_xlabel('端面特征信号幅值 [µε]')
    ax[0].set_title('(a) 缺陷信号 vs 三种模式的判据线', loc='left', fontsize=10)
    ax[0].legend(fontsize=7)

    snr = np.abs(sig)[:, None] / floors[None, :]
    im = ax[1].imshow(snr, cmap='RdYlGn', norm=matplotlib.colors.LogNorm(vmin=0.5, vmax=200),
                      aspect='auto')
    ax[1].set_xticks(range(len(fnames))); ax[1].set_xticklabels(fnames, fontsize=8, rotation=20)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=8)
    for i in range(snr.shape[0]):
        for k in range(snr.shape[1]):
            ax[1].text(k, i, f'{snr[i,k]:.0f}', ha='center', va='center', fontsize=7.5)
    plt.colorbar(im, ax=ax[1], label='信噪比 |信号|/σ')
    ax[1].set_title('(b) 信噪比矩阵（>4 可检出，>6 稳健）', loc='left', fontsize=10)
    ax[1].grid(False)
    save(fig, 'fig09_snr.png')


# ------------------------------------------------------------------ 图10 疲劳悬崖
def fig_cliff():
    from evolution import Joint, Spectrum, calibrate_from_fe
    cal = calibrate_from_fe(); jt = Joint(); sp = Spectrum()
    th = cal['theta']
    fr = np.linspace(0.15, 1.0, 60)
    life, sa_c, thc = [], [], []
    for f in fr:
        FM = f * cal['FM_design'] * 0.95
        Dy = np.zeros(len(th))
        for (Mf, Mfa, Me, n, _) in sp.bins:
            mean, amp = sp.mean_amp(Mf, Mfa, Me, th)
            Dy += n / jt.N_fail(jt.sigma_a(np.full(len(th), FM), mean - amp, mean + amp))
        j = int(np.argmax(Dy))
        life.append(1 / max(Dy[j], 1e-12)); thc.append(np.degrees(th[j]))
        mean, amp = sp.mean_amp(*sp.bins[1][:3], th)
        sa_c.append(float(jt.sigma_a(FM, mean[j] - amp[j], mean[j] + amp[j])))
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ax[0].semilogy(fr * 100, life, color=C_RED, lw=2)
    ax[0].axhline(20, color=C_INK, lw=1, ls='--'); ax[0].text(75, 24, '设计寿命 20 年', fontsize=8)
    for v, c, lab in ((60, C_AMBER, '黄 60%'), (40, C_RED, '橙 40%')):
        ax[0].axvline(v, color=c, lw=1.2, ls=':'); ax[0].text(v + 1, 3e3, lab, color=c, fontsize=8)
    ax[0].set_xlabel('剩余预紧力 [% 设计值]'); ax[0].set_ylabel('最不利螺柱理论寿命 [年]')
    ax[0].set_title('(a) 疲劳悬崖', loc='left', fontsize=10)
    ax[0].set_ylim(0.1, 1e5)

    ax[1].plot(fr * 100, sa_c, color=C_BLUE, lw=2)
    ax[1].axhline(jt.sigma_ASV, color=C_RED, lw=1.2, ls='--')
    ax[1].text(60, jt.sigma_ASV + 2, f'VDI 2230 许用应力幅 {jt.sigma_ASV:.1f} MPa',
               color=C_RED, fontsize=8)
    ax[1].set_xlabel('剩余预紧力 [%]'); ax[1].set_ylabel('螺柱应力幅 $σ_a$ [MPa]')
    ax[1].set_title('(b) 张口后应力幅跃升', loc='left', fontsize=10)

    for f, c, lab in ((0.60, C_GREEN, '预紧 60%'), (0.35, C_AMBER, '35%'), (0.25, C_RED, '25%')):
        FM = f * cal['FM_design'] * 0.95
        Dy = np.zeros(len(th))
        for (Mf, Mfa, Me, n, _) in sp.bins:
            mean, amp = sp.mean_amp(Mf, Mfa, Me, th)
            Dy += n / jt.N_fail(jt.sigma_a(np.full(len(th), FM), mean - amp, mean + amp))
        ax[2].semilogy(np.degrees(th), np.maximum(Dy, 1e-12), color=c, lw=1.6, label=lab)
    ax[2].axhline(1 / 20, color=C_INK, lw=1, ls='--')
    ax[2].text(120, 0.06, '对应 20 年寿命', fontsize=7.5)
    ax[2].set_ylim(1e-8, 10)
    ax[2].set_xlabel('周向位置角 θ [°]'); ax[2].set_ylabel('年疲劳损伤')
    ax[2].set_title('(c) 损伤沿周向的分布（0°/180° 挥舞轴为主）', loc='left', fontsize=9.5)
    ax[2].legend(fontsize=7.5)
    save(fig, 'fig10_cliff.png')


# ------------------------------------------------------------------ 图11 演化
def fig_evolution():
    names = [('Sc1', 'Sc1 扭矩法，无首次复紧'), ('Sc2', 'Sc2 扭矩法 + 5 个月复紧'),
             ('Sc3', 'Sc3 液压拉伸'), ('Sc4', 'Sc4 一个批次润滑失控')]
    fig, ax = plt.subplots(2, 4, figsize=(14, 5.6))
    FMd = float(E['FM_design'][0])
    for k, (p, title) in enumerate(names):
        t = E[p + '_t']; FM = E[p + '_FM'] / FMd * 100
        im = ax[0, k].pcolormesh(t, np.arange(_G.n_bolt), FM.T, cmap='RdYlGn',
                                 vmin=0, vmax=120, shading='auto')
        ax[0, k].set_title(title, fontsize=9, loc='left')
        ax[0, k].set_xlabel('年');
        if k == 0: ax[0, k].set_ylabel('螺套序号')
        ax[0, k].grid(False)
        if k == 3: plt.colorbar(im, ax=ax[0, k], label='剩余预紧力 [%]')

        a = ax[1, k]
        a.plot(t, E[p + '_nopen'], color=C_AMBER, lw=1.5, label='张口位数')
        a.plot(t, E[p + '_n_yel'], color='#C9A227', lw=1.2, ls='--', label='模式C 黄位')
        a.plot(t, E[p + '_n_org'], color=C_RED, lw=1.5, ls='--', label='模式C 橙位')
        a2 = a.twinx(); a2.grid(False)
        a2.plot(t, E[p + '_nbroken'], color=C_INK, lw=2.2, label='已断根数')
        a2.set_ylim(-0.2, max(3, E[p + '_nbroken'].max() + 1))
        a.set_xlabel('年')
        if k == 0: a.set_ylabel('位数')
        if k == 3: a2.set_ylabel('已断根数')
        if k == 0: a.legend(fontsize=7, loc='upper left')
        if k == 3: a2.legend(fontsize=7, loc='lower right')
    save(fig, 'fig11_evolution.png')


# ------------------------------------------------------------------ 图12 POD
def fig_pod():
    import detection as dt
    rng = np.random.default_rng(17)
    s_FM = 0.646e-3
    sig = np.linspace(5, 400, 40)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for cov, c, lab in ((0.27, C_RED, '模式S 扭矩 CoV 27%'),
                        (0.09, C_AMBER, '模式S 拉伸 CoV 9%')):
        def nm(r, cov=cov):
            f = s_FM * 442e3 * r.normal(0, cov, _G.n_bolt)
            return dt.deharmonic(f, TH, 2)[0]
        pod, fa = dt.pod_curve(sig, nm, 4.0, 400, rng)
        ax[0].plot(sig, pod, lw=1.8, color=c, label=f'{lab}（虚警 {fa.mean()*100:.1f}%）')
    for yr, c in ((1.0, C_GREEN), (5.0, C_BLUE), (10.0, '#7B3FA0')):
        def nm(r, yr=yr):
            return r.normal(0, max(8 * np.sqrt(yr) / np.sqrt(31), 0.6), _G.n_bolt)
        pod, fa = dt.pod_curve(sig, nm, 4.0, 400, rng)
        ax[0].plot(sig, pod, lw=1.8, ls='--', color=c, label=f'模式B {yr:.0f} 年')
    ax[0].axvline(269, color=C_INK, lw=1.2, ls=':'); ax[0].text(272, 0.35, '断柱 269 µε', fontsize=8)
    ax[0].axvline(163, color=C_GREY, lw=1.2, ls=':'); ax[0].text(120, 0.15, '预紧 40%\n163 µε', fontsize=7.5)
    ax[0].set_xlabel('端面特征信号 [µε]'); ax[0].set_ylabel('检出概率 POD')
    ax[0].set_title('(a) 检出概率曲线', loc='left', fontsize=10)
    ax[0].legend(fontsize=6.8, loc='lower right'); ax[0].set_ylim(-0.02, 1.05)

    gaps = [0, 2, 5, 20]
    sigs = []
    for g in gaps:
        e = S[f'D_gap{g:03d}_eps'] * 1e6
        fw = (S['x_c'] >= 0) & (S['x_c'] <= 40)
        sigs.append(e[1][J, fw].mean() - e[0][J, fw].mean())
    ax[1].bar([f'{g/100:.2f}' for g in gaps], sigs, color=C_BLUE, alpha=0.85)
    ax[1].set_xlabel('螺套端面内缩量 gap$_s$ [mm]'); ax[1].set_ylabel('断柱信号 [µε]')
    ax[1].set_title('(b) 端面构造对信号的影响：6 倍区间，但都远超判据',
                    loc='left', fontsize=9.5)
    ax[1].axhline(4 * 23.4, color=C_AMBER, lw=1.2, ls='--')
    ax[1].text(0.05, 4 * 23.4 + 40, '模式S·拉伸 4σ', color=C_AMBER, fontsize=7.5)

    yrs = np.linspace(0.5, 15, 40)
    for dr, c, lab in ((4.0, C_GREEN, '漂移 4 µε/√年'), (8.0, C_BLUE, '8（基准）'),
                       (16.0, C_RED, '16')):
        res = 4 * dr * np.sqrt(yrs) / np.sqrt(31) / (0.646e-3 * 442e3) * 100
        ax[2].plot(yrs, res, lw=1.8, color=c, label=lab)
    ax[2].axhline(20, color=C_INK, lw=1, ls=':'); ax[2].text(1, 21, '橙黄阈值间距 20%', fontsize=7.5)
    ax[2].set_xlabel('基线年限 [年]'); ax[2].set_ylabel('最小可检出预紧力变化 [% 设计值]')
    ax[2].set_title('(c) 粘接漂移是模式B 的唯一限制（且无实测数据）',
                    loc='left', fontsize=9.5)
    ax[2].legend(fontsize=7.5)
    save(fig, 'fig12_pod.png')


if __name__ == '__main__':
    import os
    os.makedirs(FIG, exist_ok=True)
    print('生成图件：')
    for f in (fig_model, fig_verify, fig_intact, fig_break, fig_debond, fig_open,
              fig_measure, fig_fingerprint, fig_snr, fig_cliff, fig_evolution, fig_pod):
        try:
            f()
        except Exception as ex:
            print(f'  !! {f.__name__}: {type(ex).__name__}: {ex}')
