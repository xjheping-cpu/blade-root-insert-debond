# -*- coding: utf-8 -*-
"""
故障与缺陷的产生过程仿真（集总环模型，参数全部从 M1 有限元标定）。

链条：安装预紧力离散 → 压陷与蠕变松弛 →（是否首次复紧）→ 接头张口 →
      螺柱应力幅跃升 → 疲劳损伤累积 → 断裂 → 载荷改道 → 相邻螺柱加速 → 级联

同时在每个巡检时刻生成 OFDR 实测的端面基线分布，交给 detection.py 判据，
得到"什么时候能发现"的时间线。这才是这份仿真真正要回答的问题：
不是"能不能看见断柱"（断了当然可检出），而是"能不能在断之前看见"。
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

N_BOLT = _G.n_bolt
R_BC = _G.R_bc
A_S = 1120.0              # M42 应力截面积 mm²
RP02 = 940.0              # 10.9 级
F_02 = RP02 * A_S         # N
D_NOM = 42.0


# ------------------------------------------------------------------ 从 M1 标定
def calibrate_from_fe(npz='out/defect_cases.npz'):
    """从有限元结果提取集总模型需要的常数，避免任何凭空假设。"""
    z = np.load(npz, allow_pickle=True)
    x = z['x_c']; j = int(z['meta'][4])
    face = (x >= 0.0) & (x <= 40.0)

    def fm(key):
        return z[key + '|eps_c'][:, face].mean(axis=1) * 1e6      # µε，每 cell

    e0 = fm('L0_zero|D0_intact')          # 零外载、满预紧
    eb = fm('L0_zero|D1_break')           # 零外载、断柱（该位无预紧）
    e1 = fm('L1_grav_p|D0_intact')        # 重力载荷、满预紧
    ew = fm('L0_zero|D3_weak40')          # 零外载、该位预紧 40%
    FM0 = z['meta'][0]
    # 端面应变对预紧力的灵敏度（用 40% 工况，避开断柱时相邻改道的干扰）
    dFM = -0.60 * FM0
    s_FM = (ew[j] - e0[j]) / dFM                       # µε/N
    # 端面应变对外载的灵敏度
    FA1 = z['L1_grav_p|D0_intact|FA']
    s_FA = (e1[j] - e0[j]) / FA1[j]                    # µε/N
    # 断柱时的端面信号（含相邻改道）
    sig_break = eb[j] - e0[j]
    # 载荷改道核（有限元精确值）
    dFA = z['L1_grav_p|D1_break|FA'] - FA1
    ker = np.roll(dFA, -j)                             # 以缺陷位为 0
    ker = ker / abs(ker[0])                            # 归一化到 -1
    return dict(s_FM=s_FM, s_FA=s_FA, sig_break=float(sig_break), kernel=ker,
                FM_design=float(FM0), eps0=float(e0[j]),
                scatter_ref=e0, theta=z['theta'], x=x)


# ------------------------------------------------------------------ 接头
@dataclass
class Joint:
    Phi: float = 0.076              # M1 解出（螺套端面承压构造）
    A_s: float = A_S
    sigma_ASV: float = 0.85 * (150.0 / D_NOM + 45.0)     # VDI 2230，M42 → 41.3 MPa
    m1: float = 3.0                 # 拐点以上斜率
    m2: float = 5.0                 # Haibach 延长
    N_ref: float = 2.0e6

    def F_S(self, F_M, F_A):
        """含张口的非线性螺柱轴力。"""
        F_open = F_M / (1.0 - self.Phi)
        return np.where(F_A < F_open, F_M + self.Phi * F_A, np.maximum(F_A, 0.0))

    def sigma_a(self, F_M, F_A_min, F_A_max):
        return (self.F_S(F_M, F_A_max) - self.F_S(F_M, F_A_min)) / (2.0 * self.A_s)

    def N_fail(self, sa):
        sa = np.maximum(sa, 1e-6)
        return np.where(sa >= self.sigma_ASV,
                        self.N_ref * (self.sigma_ASV / sa) ** self.m1,
                        self.N_ref * (self.sigma_ASV / sa) ** self.m2)


# ------------------------------------------------------------------ 载荷谱
@dataclass
class Spectrum:
    """
    简化载荷谱。对第 j 根螺柱，一个转周内的外载可写成
        F_A(ψ) = k·M_f·cosθ_j + k·M_fa·cos(ψ+φ)·cosθ_j + k·M_e·cosψ·sinθ_j
    即"均值 + 挥舞波动 + 重力摆振"。挥舞波动是宽带的、与重力 1P 不同相，
    工程上按正交合成取幅值：
        F_A,mean = k·M_f·cosθ      F_A,amp = k·sqrt((M_fa·cosθ)² + (M_e·sinθ)²)
    分档给出每年循环数。注意这是简化谱，绝对寿命只有量级意义；
    本研究要的是寿命对预紧力的相对敏感性，那一条对谱形不敏感。
    """
    # (挥舞均值, 挥舞波动幅值, 重力摆振幅值, 每年循环数, 说明)
    bins: tuple = (
        (15.0e9, 2.0e9, 7.0e9, 4.0e6, '1P 正常运行'),
        (15.0e9, 5.0e9, 7.0e9, 2.0e5, '湍流大幅'),
        (20.0e9, 7.0e9, 7.0e9, 1.0e4, '阵风'),
        (26.0e9, 8.0e9, 7.0e9, 3.0e2, '强阵风'),
        (32.0e9, 9.0e9, 7.0e9, 2.0e1, '极端'),
    )

    @staticmethod
    def k():
        return 2.0 / (N_BOLT * R_BC)

    def mean_amp(self, M_f, M_fa, M_e, theta):
        k = self.k()
        mean = k * M_f * np.cos(theta)
        amp = k * np.sqrt((M_fa * np.cos(theta)) ** 2 + (M_e * np.sin(theta)) ** 2)
        return mean, amp


# ------------------------------------------------------------------ 主仿真
@dataclass
class Sim:
    years: float = 20.0
    dt: float = 1.0 / 12.0
    install_cov: float = 0.27        # 扭矩法安装离散；液压拉伸取 0.09
    retorque_at: float = 0.0         # 首次复紧时刻（年），0 = 不做
    bad_batch: tuple = ()            # (起始 cell, 个数, 预紧力系数)
    seed: int = 3
    Phi: float = 0.076
    relax_emb: float = 0.03
    relax_creep: float = 0.035

    def run(self, cal):
        rng = np.random.default_rng(self.seed)
        th = cal['theta']
        jt = Joint(Phi=self.Phi)
        sp = Spectrum()
        FM_design = cal['FM_design']

        FM0 = FM_design * rng.normal(1.0, self.install_cov, N_BOLT).clip(0.18, 1.8)
        if self.bad_batch:
            s, n, f = self.bad_batch
            FM0[s:s + n] = FM_design * f * rng.normal(1.0, 0.06, n)
        FM_ref = FM0.copy()          # 松弛的参照（复紧后会更新）
        t_ref = np.zeros(N_BOLT)

        D = np.zeros(N_BOLT)
        broken = np.zeros(N_BOLT, bool)
        t = 0.0
        hist = dict(t=[], FM=[], D=[], nbroken=[], nopen=[], eps_face=[], sa_max=[])
        events = []

        while t < self.years:
            # --- 松弛
            days = np.maximum((t - t_ref) * 365.0, 1e-3)
            relax = self.relax_emb + self.relax_creep * np.log10(1.0 + days)
            FM = FM_ref * np.clip(1.0 - relax, 0.05, 1.0)
            FM[broken] = 0.0

            # --- 载荷改道（叠加每个已断位的核）
            extra = np.zeros(N_BOLT)
            for j in np.flatnonzero(broken):
                extra += np.roll(cal['kernel'], j)

            # --- 损伤累积
            n_open = 0
            sa_max = 0.0
            for _i_bin, (Mf, Mfa, Me, ncyc, _) in enumerate(sp.bins):
                mean, amp = sp.mean_amp(Mf, Mfa, Me, th)
                mean = mean * (1.0 + extra); amp = amp * (1.0 + extra)
                sa = jt.sigma_a(FM, mean - amp, mean + amp)
                sa = np.where(broken, 0.0, sa)
                D += (ncyc * self.dt) / jt.N_fail(sa)
                if _i_bin == 0:      # 只统计正常运行档的张口——决定高周疲劳的是它
                    n_open = int(np.sum((mean + amp) > FM / (1 - jt.Phi)))
                sa_max = max(sa_max, float(np.max(sa)))

            # --- 断裂判定
            newly = np.flatnonzero((D >= 1.0) & ~broken)
            for j in newly:
                broken[j] = True
                events.append((t, 'break', int(j)))

            # --- OFDR 观测量：端面原始应变分布基线
            FA_nom = sp.mean_amp(*sp.bins[0][:3], th)[0] * (1 + extra)
            eps = cal['eps0'] + cal['s_FM'] * (FM - FM_design) + cal['s_FA'] * FA_nom
            eps[broken] = cal['eps0'] + cal['sig_break']

            hist['t'].append(t); hist['FM'].append(FM.copy()); hist['D'].append(D.copy())
            hist['nbroken'].append(int(broken.sum())); hist['nopen'].append(n_open)
            hist['eps_face'].append(eps.copy()); hist['sa_max'].append(sa_max)

            # --- 首次复紧
            if self.retorque_at > 0 and t <= self.retorque_at < t + self.dt:
                FM_ref = np.where(broken, 0.0, FM_design * rng.normal(1.0, self.install_cov * 0.7, N_BOLT).clip(0.18, 1.8))
                t_ref = np.full(N_BOLT, t)
                events.append((t, 'retorque', -1))

            t += self.dt

        for k in ('t', 'nbroken', 'nopen', 'sa_max'):
            hist[k] = np.array(hist[k])
        for k in ('FM', 'D', 'eps_face'):
            hist[k] = np.array(hist[k])
        hist['events'] = events
        hist['theta'] = th
        return hist


if __name__ == '__main__':
    cal = calibrate_from_fe()
    print(f"标定：ε_face(满预紧,零载) = {cal['eps0']:.0f} µε")
    print(f"      ∂ε/∂F_M = {cal['s_FM']*1e3:.3f} µε/kN   "
          f"∂ε/∂F_A = {cal['s_FA']*1e3:.3f} µε/kN")
    print(f"      断柱端面信号 = {cal['sig_break']:+.0f} µε")
    print(f"      改道核（相邻 0..4）= {np.round(cal['kernel'][:5], 3)}")
    jt = Joint()
    print(f"      σ_ASV(M42) = {jt.sigma_ASV:.1f} MPa")
    for f in (1.0, 0.6, 0.4, 0.3):
        FM = f * 420e3
        print(f"      预紧 {f*100:3.0f}% = {FM/1e3:5.0f} kN → 张口载荷 "
              f"{FM/(1-jt.Phi)/1e3:5.0f} kN；"
              f"σ_a(±74kN)={jt.sigma_a(FM,-74e3,74e3):5.1f} MPa；"
              f"σ_a(159±53kN)={jt.sigma_a(FM,106e3,212e3):5.1f} MPa")
