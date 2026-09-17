# -*- coding: utf-8 -*-
"""
检测与判据。三种工作模式，噪声底完全不同，必须分开算：

  模式 S（单次回访、无历史基线）：靠整环各个螺套互为参照做空间检测。
      噪声底 = 安装预紧力离散（不是仪器噪声）。
  模式 B（对投运基线差分）：预紧力离散被基线抵消。
      噪声底 = 粘接漂移 + 温度残差 + 重复性 + 配准误差。
  模式 O（在线）：单个 1P 周期内的波形特征，温度在周期内是常数，天然解耦。
      能看到张口这种只在阵风里出现的瞬态。

判据链：特征提取 → 去低阶谐波（周向载荷分布本身是 cos θ）→ 稳健阈值（MAD）
        → 匹配滤波（用有限元模板区分断柱 / 脱粘 / 预紧力衰减）
"""
from __future__ import annotations
import numpy as np
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


# 特征窗口（mm）：端面区反映夹紧状态，中段反映载荷分担
WIN_FACE = (0.0, 40.0)
WIN_MID = (150.0, 450.0)
WIN_END = (_G.L_ins - 50.0, _G.L_ins)


def _win_mean(x, e, win):
    m = (x >= win[0]) & (x <= win[1])
    return np.asarray(e)[..., m].mean(axis=-1)


def features(x, eps_cells):
    """每个螺套一组特征。eps_cells: (n_cell, nx)，µε。"""
    e = np.asarray(eps_cells, float)
    return dict(face=_win_mean(x, e, WIN_FACE),
                mid=_win_mean(x, e, WIN_MID),
                end=_win_mean(x, e, WIN_END))


def deharmonic(f, theta, order=2):
    """去掉 0~order 阶周向谐波：叶根弯矩本身就是 cos θ 分布，不去掉会淹没局部信号。
    用稳健（迭代重加权）拟合，避免缺陷本身把基准拉偏。"""
    f = np.asarray(f, float)
    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    A = np.stack(cols, axis=1)
    w = np.ones_like(f)
    for _ in range(5):
        W = w[:, None]
        coef, *_ = np.linalg.lstsq(A * W, f * w, rcond=None)
        r = f - A @ coef
        s = robust_sigma(r)
        w = 1.0 / (1.0 + (r / (3.0 * max(s, 1e-9))) ** 2)
    return f - A @ coef, coef


def robust_sigma(r):
    return 1.4826 * np.median(np.abs(np.asarray(r) - np.median(r)))


def detect(resid, k=4.0, sigma=None):
    """返回 (flags, sigma, z)。k=4 对应单点虚警率约 3e-5，全环一次测量约 0.4% 虚警。"""
    s = robust_sigma(resid) if sigma is None else sigma
    z = np.asarray(resid) / max(s, 1e-9)
    return np.abs(z) > k, s, z


def build_templates(x, eps_intact, eps_cases, j_def):
    """用有限元结果做模板：每种缺陷在缺陷位上的轴向 Δ 剖面，归一化。"""
    T = {}
    for name, e in eps_cases.items():
        d = np.asarray(e)[j_def] - np.asarray(eps_intact)[j_def]
        n = np.linalg.norm(d)
        T[name] = d / n if n > 0 else d
    return T


def matched_filter(x, d_profile, templates):
    """把一条 Δ 剖面投影到各模板上，返回 {名称: 相关系数} 与最佳匹配。"""
    d = np.asarray(d_profile, float)
    nd = np.linalg.norm(d)
    if nd == 0:
        return {k: 0.0 for k in templates}, None, 0.0
    sc = {k: float(np.dot(d, t) / (nd * np.linalg.norm(t))) for k, t in templates.items()}
    best = max(sc, key=lambda k: sc[k])
    return sc, best, sc[best]


def half_wave_asym(eps_tension, eps_compression):
    """半波不对称：拉伸半周与压缩半周的应变之和（完好接头两者对称相消）。"""
    return np.asarray(eps_tension) + np.asarray(eps_compression)


def waveform_kink(FA, eps, frac=0.5):
    """
    张口拐点：接头闭合时 ε 随 F_A 线性；张口后斜率突变。
    用前 frac 段（低载）拟合直线，报告高载段偏离该直线的最大值与拐点处的 F_A。
    """
    FA = np.asarray(FA, float); e = np.asarray(eps, float)
    o = np.argsort(FA); FA, e = FA[o], e[o]
    n = max(4, int(len(FA) * frac))
    p = np.polyfit(FA[:n], e[:n], 1)
    dev = e - np.polyval(p, FA)
    i = int(np.argmax(np.abs(dev)))
    # 拐点：偏离首次超过 3 倍低载段残差
    s = np.std(e[:n] - np.polyval(p, FA[:n]))
    over = np.flatnonzero(np.abs(dev) > 3 * max(s, 1e-6))
    F_kink = FA[over[0]] if len(over) else np.nan
    return dict(dev_max=float(dev[i]), F_kink=float(F_kink), slope_lo=float(p[0]), dev=dev, FA=FA)


def rise_point(x, eps, lo=0.15, hi=0.85):
    """
    脱粘定位：完好时传力在 x≈0 就开始，脱粘后起升点内移。
    取剖面从最小值上升到 (lo..hi) 区间的位置，用插值给出亚采样精度。
    """
    e = np.asarray(eps, float)
    i0 = int(np.argmin(e[: max(3, len(e) // 3)]))
    seg = e[i0:]
    lo_v = seg.min() + lo * (seg.max() - seg.min())
    hi_v = seg.min() + hi * (seg.max() - seg.min())
    def cross(v):
        k = np.flatnonzero(seg >= v)
        if len(k) == 0:
            return np.nan
        k = k[0]
        if k == 0:
            return x[i0]
        x0, x1 = x[i0 + k - 1], x[i0 + k]
        y0, y1 = seg[k - 1], seg[k]
        return x0 + (v - y0) * (x1 - x0) / max(y1 - y0, 1e-9)
    return cross(lo_v), cross(hi_v)


def debond_front(x, d_profile):
    """脱粘前沿：Δ 剖面上那个尖锐的负向凹陷位置。"""
    d = np.asarray(d_profile, float)
    m = x <= 400.0
    i = int(np.argmin(d[m]))
    return float(x[m][i]), float(d[m][i])


# --------------------------------------------------------------------- POD
def pod_montecarlo(sig_face, noise_model, k=4.0, n_mc=2000, rng=None):
    """
    给定缺陷在 face 特征上的信号幅值 sig_face（µε）与噪声模型（返回整环长度残差的函数），
    蒙特卡洛估计检出概率与虚警率。
    """
    rng = rng or np.random.default_rng(0)
    n_hit = 0
    n_fa = 0
    for _ in range(n_mc):
        r0 = noise_model(rng)                      # 无缺陷
        f0, s0, _ = detect(r0, k)
        n_fa += int(f0.any())
        r1 = r0.copy(); r1[len(r1) // 2] += sig_face
        f1, s1, z1 = detect(r1, k)
        n_hit += int(f1[len(r1) // 2])
    return n_hit / n_mc, n_fa / n_mc


def pod_curve(sig_list, noise_model, k=4.0, n_mc=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    pod, fa = [], []
    for s in sig_list:
        p, f = pod_montecarlo(s, noise_model, k, n_mc, rng)
        pod.append(p); fa.append(f)
    return np.array(pod), np.array(fa)
