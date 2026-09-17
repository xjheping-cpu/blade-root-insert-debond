# -*- coding: utf-8 -*-
"""
OFDR 测量模型：把有限元算出的"真实"基体应变场，变成解调仪实际会输出的数据。

依次施加五个环节，每一个都对应一条已核实的指标或一条已知的物理限制：
  1 应变传递    光纤—胶层—基体的剪滞，使尖锐特征被低通（特征长度 2~5 mm）
  2 标距平均    OFDR 输出的是标距内的平均值，50 m 档 1.3 mm / 100 m 档 2.6 mm
  3 空间采样    按标距间距采样，并含回访之间的配准误差（沿纤位置对不齐）
  4 噪声        重复性 (2σ) ±2~5 µε（本包既有口径）+ 少量异常点
  5 环境        温度视应变 13 µε/°C（贴复材）经参考纤补偿后的残差 + 粘接长期漂移

注意口径：系统精度 ±25~30 µε 是绝对量指标，本模型里不进入差分误差预算；
差分用法看的是重复性与漂移。这与《OFDR钢塔应用分析》§2.3 的处理一致。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class OfdrSpec:
    gauge: float = 1.3            # 标距 mm（50 m 档）
    pitch: float = 1.3            # 采样间距 mm
    sigma_rep: float = 1.5        # 重复性 1σ，µε（对应 2σ ≈ ±3 µε）
    outlier_frac: float = 0.004   # 异常点比例（DLR WiValdi 报告里 5.2 mm 内跳变 >70 µε 的那类）
    outlier_amp: float = 60.0     # 异常点幅值 µε
    L_transfer: float = 3.0       # 应变传递特征长度 mm
    dT_resid: float = 0.5         # 参考纤补偿后的温度残差 °C
    temp_sens: float = 13.0       # µε/°C，贴复合材料
    drift_rate: float = 8.0       # 粘接长期漂移 µε/年（1σ，随机游走）
    reg_err: float = 1.0          # 回访配准误差 1σ，mm


def _kernel(dx: float, L: float, gauge: float):
    """应变传递（指数核）与标距平均（矩形核）的合成卷积核。"""
    n = max(3, int(np.ceil(4 * L / dx)) | 1)
    t = (np.arange(n) - n // 2) * dx
    k1 = np.exp(-np.abs(t) / L)
    k1 /= k1.sum()
    m = max(1, int(round(gauge / dx)))
    k2 = np.ones(m) / m
    k = np.convolve(k1, k2)
    return k / k.sum()


def measure(x_true, eps_true, spec: OfdrSpec, rng=None,
            years: float = 0.0, drift_field=None, dT=None, apply_noise=True):
    """
    x_true, eps_true : 有限元给出的轴向坐标与真实应变（µε），eps_true 可为 (..., nx)
    返回 (x_meas, eps_meas)，eps_meas 形状与 eps_true 前置维一致
    """
    rng = rng or np.random.default_rng()
    eps = np.atleast_2d(np.asarray(eps_true, float))
    # 均匀重采样到细网格再卷积
    dx = min(spec.pitch, spec.gauge) / 4.0
    xf = np.arange(x_true[0], x_true[-1], dx)
    ef = np.stack([np.interp(xf, x_true, e) for e in eps])
    k = _kernel(dx, spec.L_transfer, spec.gauge)
    ef = np.stack([np.convolve(e, k, mode='same') for e in ef])
    # 采样（含配准误差）
    xm = np.arange(x_true[0], x_true[-1] - spec.gauge, spec.pitch)
    shift = rng.normal(0.0, spec.reg_err) if (apply_noise and spec.reg_err > 0) else 0.0
    em = np.stack([np.interp(xm + shift, xf, e) for e in ef])
    if apply_noise:
        em = em + rng.normal(0.0, spec.sigma_rep, em.shape)
        if spec.outlier_frac > 0:
            m = rng.random(em.shape) < spec.outlier_frac
            em = em + m * rng.normal(0.0, spec.outlier_amp, em.shape)
        # 温度残差：沿纤缓变（低阶多项式）
        if dT is None:
            dT = rng.normal(0.0, spec.dT_resid)
        prof = 1.0 + 0.3 * np.cos(np.pi * (xm - xm[0]) / (xm[-1] - xm[0]))
        em = em + dT * spec.temp_sens * prof[None, :]
        # 粘接漂移：随位置的随机游走，幅值随时间根号增长
        if years > 0:
            if drift_field is None:
                w = rng.normal(0.0, 1.0, (em.shape[0], em.shape[1]))
                w = np.cumsum(w, axis=1) / np.sqrt(em.shape[1])
                drift_field = w / max(np.std(w), 1e-9)
            em = em + spec.drift_rate * np.sqrt(years) * drift_field
    return xm, (em[0] if np.ndim(eps_true) == 1 else em)


def measure_ring(x_true, eps_cells, spec: OfdrSpec, rng=None, years=0.0, dT=None,
                 apply_noise=True):
    """整环：eps_cells 形状 (n_bolt, nx)，µε。返回 (x_meas, (n_bolt, nm))。"""
    return measure(x_true, eps_cells, spec, rng, years, None, dT, apply_noise)


def fiber_length_budget(n_bolt=None, x_span=600.0, pitch=None, n_ring=3, R=None):
    """两种纤路的长度账，用于核对是否落在解调仪单通道量程内。

    n_bolt / pitch / R 缺省时取几何配置里的值。
    """
    from geometry import Geom
    g = Geom()
    n_bolt = g.n_bolt if n_bolt is None else n_bolt
    pitch = g.pitch if pitch is None else pitch
    R = g.R_bc if R is None else R
    serp = n_bolt * (x_span + pitch * 0.9) / 1000.0     # 蛇形：每 cell 一段 + 折返
    ring = n_ring * 2 * np.pi * R / 1000.0
    return dict(serpentine_m=serp, serpentine_per_channel_m=serp / 2,
                rings_m=ring, total_m=serp + ring)


if __name__ == '__main__':
    import numpy as np
    print(fiber_length_budget())
    x = np.linspace(0, 600, 200)
    e = 300 * np.exp(-x / 40) - 50
    xm, em = measure(x, e, OfdrSpec(), rng=np.random.default_rng(0))
    print(f"真实 x=0 {e[0]:.0f} µε → 测得 {em[0]:.0f} µε；采样点数 {len(xm)}")
