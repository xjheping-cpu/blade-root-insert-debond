# -*- coding: utf-8 -*-
"""
故障演化 + 检测时间线。回答那个真正的问题：不是"断了能不能看见"，
而是"断之前多久能看见"，以及"看见了能不能判对是什么"。

四个场景：
  Sc1 扭矩法安装（CoV 27%），无首次复紧
  Sc2 扭矩法安装 + 投运 5 个月首次复紧
  Sc3 液压拉伸安装（CoV 9%）
  Sc4 一个批次润滑失控（20 根按 40% 预紧装上）
每年做一次回访检测，两种模式各判一次。
"""
from __future__ import annotations
import numpy as np
from evolution import Sim, calibrate_from_fe, Joint
import detection as dt
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

CAL = calibrate_from_fe()
TH = CAL['theta']
S_FM = abs(CAL['s_FM'])          # µε/N


def _drift_walk(t, drift, rng):
    """粘接漂移：随机游走，使 t 年后的标准差为 drift·√t。"""
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    step = drift * np.sqrt(dt)
    return np.cumsum(rng.normal(0, step, (len(t), _G.n_bolt)), axis=0)


def detect_timeline(hist, cal, mode='B', cov=0.27, drift=8.0, k=4.0, rng=None,
                    t_base=0.5, sigma_floor=2.0):
    """
    对每个巡检时刻跑一次判据。
    基线取在 t_base（默认 0.5 年，即首次复紧之后）——投运头几个月的压陷与蠕变
    在各螺柱间是不一样的（松弛量正比于各自的预紧力），若把基线放在投运当天，
    这部分真实但无害的差异会直接触发判据。这是一条操作性结论。
    """
    rng = rng or np.random.default_rng(11)
    t = hist['t']; eps = hist['eps_face']
    ib = int(np.argmin(np.abs(t - t_base)))
    dw = _drift_walk(t, drift, rng)            # 粘接漂移是随机游走，不是逐次独立
    base = eps[ib] + dw[ib]
    flags, zmax = [], []
    for i, ti in enumerate(t):
        if mode == 'B':
            d = eps[i] + dw[i] - base
        else:                                  # 模式 S：无基线，只有当次
            d = eps[i] + dw[i] * 0.3
        r, _ = dt.deharmonic(d, TH, order=2)
        s = max(dt.robust_sigma(r), sigma_floor)
        f, _, z = dt.detect(r, k, sigma=s)
        f[:ib + 1] if False else None
        flags.append(int(f.sum())); zmax.append(float(np.max(np.abs(z))))
    flags = np.array(flags); flags[:ib + 1] = 0     # 基线之前不判
    return flags, np.array(zmax)


def track_preload(hist, cal, drift=8.0, rng=None, t_base=0.5, FM_known_cov=0.09):
    """
    模式 C：标定跟踪。投运时（或首次复紧后）用已知安装轴力给每一位做一次标定，
    此后把端面应变的变化直接换算成该位的预紧力。这是三种模式里唯一能给出
    "这一根现在还剩百分之几"的，也是唯一能对接 V2.0 分级判据的。
    前提：安装轴力必须是已知的——液压拉伸有记录，或超声抽检定标后外推。
    """
    rng = rng or np.random.default_rng(13)
    t = hist['t']; eps = hist['eps_face']; FM = hist['FM']
    ib = int(np.argmin(np.abs(t - t_base)))
    # 标定：已知该时刻的真实预紧力（带记录不确定度），并记下当时的端面应变
    FM_cal = FM[ib] * rng.normal(1.0, FM_known_cov, _G.n_bolt)
    dw = _drift_walk(t, drift, rng)
    e_cal = eps[ib] + dw[ib]
    est = [FM_cal + (eps[i] + dw[i] - e_cal) / cal['s_FM'] for i in range(len(t))]
    return np.array(est), ib


def first_sustained(t, flags, n=2):
    """连续 n 次巡检都报警才算首次报警，避免单点虚警。"""
    for i in range(len(flags) - n + 1):
        if np.all(flags[i:i + n] > 0):
            return t[i]
    return np.nan


def run_one(name, sim: Sim, cov):
    h = sim.run(CAL)
    t = h['t']
    nb = h['nbroken']
    FMmin = h['FM'].min(axis=1) / CAL['FM_design']
    t_first_break = t[np.argmax(nb > 0)] if nb.max() > 0 else np.nan
    fB, zB = detect_timeline(h, CAL, 'B', cov)
    fS, zS = detect_timeline(h, CAL, 'S', cov)
    est, ib = track_preload(h, CAL, FM_known_cov=(0.09 if cov < 0.15 else 0.27))
    fr = est / CAL['FM_design']
    n_org = (fr < 0.40).sum(axis=1); n_yel = (fr < 0.60).sum(axis=1)
    fC = (n_org > 0).astype(int); fC[:ib + 1] = 0
    tB = first_sustained(t, fB); tS = first_sustained(t, fS); tC = first_sustained(t, fC)
    lead = t_first_break - tC if np.isfinite(t_first_break) and np.isfinite(tC) else np.nan
    print(f"  {name:32s} 真实最低预紧 {FMmin[-1]*100:3.0f}%  20年断裂 {nb[-1]:2d} 根  "
          f"首断 {t_first_break:5.1f}  |  首警 S={tS:5.1f} B={tB:5.1f} C={tC:5.1f} 年  "
          f"C 提前量 {lead:5.1f} 年  20年末 黄{n_yel[-1]:3d}/橙{n_org[-1]:3d} 位")
    return dict(name=name, h=h, fB=fB, fS=fS, fC=fC, est=est, n_org=n_org, n_yel=n_yel,
                t_break=t_first_break, tB=tB, tS=tS, tC=tC)


def main():
    print("=" * 104)
    print("零、标定常数（全部来自 M1 有限元，不是假设）")
    print("=" * 104)
    print(f"  端面特征灵敏度 ∂ε/∂F_M = {CAL['s_FM']*1e3:.3f} µε/kN，"
          f"∂ε/∂F_A = {CAL['s_FA']*1e3:.3f} µε/kN")
    print(f"  断柱端面信号 {CAL['sig_break']:+.0f} µε；改道核 ±1 位 {CAL['kernel'][1]*100:.1f}%")
    jt = Joint()
    print(f"  σ_ASV(M42,10.9) = {jt.sigma_ASV:.1f} MPa；张口载荷 = F_M/(1-Φ) = 1.082 F_M")

    print()
    print("=" * 104)
    print("一、疲劳悬崖：应力幅与寿命对预紧力的依赖（额定工况 F_A = 159±53 kN）")
    print("=" * 104)
    print(f"  {'预紧力':>8} {'张口载荷':>10} {'是否张口':>8} {'σ_a':>8} {'N_f':>12} {'理论寿命':>10}")
    for f in (1.0, 0.8, 0.6, 0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
        FM = f * CAL['FM_design'] * 0.95
        sa = float(jt.sigma_a(FM, 106e3, 212e3))
        N = float(jt.N_fail(sa))
        life = N / 4.0e6
        op = "是" if 212e3 > FM / (1 - jt.Phi) else "否"
        print(f"  {f*100:6.0f}% {FM/(1-jt.Phi)/1e3:8.0f} kN {op:>8} {sa:6.1f} MPa "
              f"{N:11.2e} {life:8.1f} 年")

    print()
    print("=" * 104)
    print("二、四个安装/维护场景的 20 年演化与检测时间线")
    print("=" * 104)
    res = []
    res.append(run_one("Sc1 扭矩法，无首次复紧", Sim(install_cov=0.27, retorque_at=0.0, seed=3), 0.27))
    res.append(run_one("Sc2 扭矩法 + 5 个月首次复紧", Sim(install_cov=0.27, retorque_at=0.42, seed=3), 0.27))
    res.append(run_one("Sc3 液压拉伸，无首次复紧", Sim(install_cov=0.09, retorque_at=0.0, seed=3), 0.09))
    res.append(run_one("Sc4 一个批次润滑失控 (20根@40%)", Sim(install_cov=0.09, retorque_at=0.0,
                                                      bad_batch=(40, 20, 0.40), seed=3), 0.09))

    print()
    print("=" * 104)
    print("三、级联：Sc1 中断裂根数随时间")
    print("=" * 104)
    h = res[0]['h']; t = h['t']
    idx = [np.argmin(np.abs(t - v)) for v in (1, 2, 3, 5, 8, 12, 16, 20)]
    print("  年份       " + "".join(f"{t[i]:8.0f}" for i in idx))
    print("  断裂根数   " + "".join(f"{h['nbroken'][i]:8d}" for i in idx))
    print("  张口根数   " + "".join(f"{h['nopen'][i]:8d}" for i in idx))
    print("  最低预紧%  " + "".join(f"{h['FM'][i].min()/CAL['FM_design']*100:8.0f}" for i in idx))
    print("  模式B报警  " + "".join(f"{res[0]['fB'][i]:8d}" for i in idx))
    print("  模式C 橙位 " + "".join(f"{res[0]['n_org'][i]:8d}" for i in idx))
    print("  模式C 黄位 " + "".join(f"{res[0]['n_yel'][i]:8d}" for i in idx))

    print()
    print("=" * 104)
    print("四、模式 C 的预紧力估计精度（真实值 vs 估计值，Sc1 第 10 年）")
    print("=" * 104)
    r0 = res[0]; i10 = np.argmin(np.abs(r0['h']['t'] - 10))
    true = r0['h']['FM'][i10] / CAL['FM_design'] * 100
    estp = r0['est'][i10] / CAL['FM_design'] * 100
    err = estp - true
    ok = ~np.isclose(true, 0)
    print(f"  估计误差：均值 {err[ok].mean():+.1f}%，标准差 {err[ok].std():.1f}%，"
          f"最大 {np.abs(err[ok]).max():.1f}%（设计预紧力的百分数）")
    o = np.argsort(true)[:8]
    print("  最弱 8 位：真实 " + " ".join(f"{true[k]:5.0f}" for k in o))
    print("            估计 " + " ".join(f"{estp[k]:5.0f}" for k in o))

    print()
    print("=" * 104)
    print("五、标定方式对模式 C 绝对精度的影响（Sc1，第 10 年）")
    print("=" * 104)
    print(f"  {'标定方式':30s} {'安装轴力已知度':>14} {'预紧力估计误差 std':>18} {'能否对接分级判据':>16}")
    for tag, cov_k in (('仅扭矩规程（无记录）', 0.27),
                       ('液压拉伸并记录施加值', 0.09),
                       ('拧紧前后各测一次基线分布 + 超声标定斜率', 0.02)):
        e2, ib2 = track_preload(r0['h'], CAL, FM_known_cov=cov_k,
                                rng=np.random.default_rng(21))
        er = (e2[i10] - r0['h']['FM'][i10]) / CAL['FM_design'] * 100
        okm = r0['h']['FM'][i10] > 0
        v = er[okm].std()
        verdict = '可以' if v < 8 else ('勉强' if v < 15 else '不可以')
        print(f"  {tag:30s} {cov_k*100:12.0f}% {v:16.1f}% {verdict:>16}")
    print("\n  注：OFDR 贡献的是变化量，精度由粘接漂移决定（5 年 ±3%、10 年 ±4%）；"
          "绝对量的精度由安装时对轴力的已知度决定。")
    print("  拧紧前后各测一次基线分布之所以有效，是因为松开状态就是每一位天然的零点，"
          "整环一次全得；超声抽检只能给其中几根。")

    np.savez_compressed('out/evolution.npz',
                        **{f"{r['name'][:3]}_{k}": np.asarray(v)
                           for r in res for k, v in
                           (('t', r['h']['t']), ('FM', r['h']['FM']), ('D', r['h']['D']),
                            ('nbroken', r['h']['nbroken']), ('nopen', r['h']['nopen']),
                            ('eps', r['h']['eps_face']), ('fB', r['fB']), ('fS', r['fS']),
                            ('fC', r['fC']), ('est', r['est']),
                            ('n_org', r['n_org']), ('n_yel', r['n_yel']))},
                        theta=TH, FM_design=np.array([CAL['FM_design']]))
    print("\n已存 out/evolution.npz")


if __name__ == '__main__':
    main()
