# -*- coding: utf-8 -*-
"""
检测研究：把有限元的真实场经测量模型变成数据，跑判据，给出
  · 三种模式各自的噪声底（这三个噪声底完全不同，混在一起谈就会得出错误结论）
  · 检出概率 POD 与虚警率
  · 缺陷分类（断柱 / 脱粘 / 预紧力衰减）的可分性
输出 out/detect_results.npz 与屏幕表格。
"""
from __future__ import annotations
import numpy as np
from ofdr_sensing import OfdrSpec, measure_ring, fiber_length_budget
import detection as dt
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

Z = np.load('out/defect_cases.npz', allow_pickle=True)
X = Z['x_c']; TH = Z['theta']; J = int(Z['meta'][4]); FM_D = Z['meta'][0]
SPEC = OfdrSpec()


def eps(key):
    return Z[key + '|eps_c'] * 1e6      # (n_bolt, nx) µε


def face_feat(x, e):
    return dt.features(x, e)['face']


# --------------------------------------------------------------------- 噪声底
def floor_mode_S(cov, n_mc=4000, rng=None):
    """模式 S：无基线，靠整环各位互为参照。噪声底 = 安装预紧力离散。"""
    rng = rng or np.random.default_rng(1)
    s_FM = -0.646e-3                      # µε/N，evolution.calibrate 给出
    out = []
    for _ in range(n_mc):
        dFM = FM_D * rng.normal(0, cov, _G.n_bolt)
        f = s_FM * dFM
        r, _ = dt.deharmonic(f, TH, order=2)
        out.append(dt.robust_sigma(r))
    return float(np.mean(out))


def floor_mode_B(years, n_mc=2000, rng=None, drift=None):
    """模式 B：对投运基线差分。预紧力离散被抵消，剩粘接漂移+温度残差+重复性+配准。"""
    rng = rng or np.random.default_rng(2)
    sp = OfdrSpec() if drift is None else OfdrSpec(drift_rate=drift)
    e_int = eps('L0_zero|D0_intact')
    out = []
    for _ in range(n_mc // 40):
        xm, m0 = measure_ring(X, e_int, sp, rng, years=0.0)
        xm, m1 = measure_ring(X, e_int, sp, rng, years=years)
        d = face_feat(xm, m1) - face_feat(xm, m0)
        r, _ = dt.deharmonic(d, TH, order=2)
        out.append(dt.robust_sigma(r))
    return float(np.mean(out))


def main():
    print("=" * 100)
    print("零、纤路长度账（是否落在解调仪单通道量程内）")
    print("=" * 100)
    b = fiber_length_budget()
    print(f"  蛇形轴向 {b['serpentine_m']:.0f} m（分两通道，每通道 {b['serpentine_per_channel_m']:.0f} m）"
          f" + 环向三圈 {b['rings_m']:.0f} m")
    print(f"  每通道 {b['serpentine_per_channel_m']:.0f} m → 落在 50 m 档（采样 1.3 mm）")

    print()
    print("=" * 100)
    print("一、缺陷信号幅值（face 特征 = x∈[0,40] mm 的平均轴向应变）")
    print("=" * 100)
    base = face_feat(X, eps('L0_zero|D0_intact'))
    cases = [('断柱', 'L0_zero|D1_break'), ('预紧 40%', 'L0_zero|D3_weak40'),
             ('脱粘 50mm', 'L1_grav_p|D4_deb50'), ('脱粘 100mm', 'L1_grav_p|D4_deb100'),
             ('脱粘 200mm', 'L1_grav_p|D4_deb200'), ('脱粘 300mm', 'L1_grav_p|D4_deb300')]
    sig = {}
    for name, key in cases:
        b0 = face_feat(X, eps(key.split('|')[0] + '|D0_intact'))
        f = face_feat(X, eps(key))
        r0, _ = dt.deharmonic(b0, TH); r1, _ = dt.deharmonic(f, TH)
        sig[name] = float(r1[J] - r0[J])
        print(f"  {name:12s} face 残差 {sig[name]:+8.1f} µε")

    print()
    print("=" * 100)
    print("二、三种模式的噪声底（µε，1σ，去 2 阶谐波后的稳健标准差）")
    print("=" * 100)
    fS_t = floor_mode_S(0.27); fS_p = floor_mode_S(0.09)
    print(f"  模式 S 无基线 · 扭矩法安装 (CoV 27%)   σ = {fS_t:6.1f}")
    print(f"  模式 S 无基线 · 液压拉伸   (CoV  9%)   σ = {fS_p:6.1f}")
    for y in (0.0, 1.0, 5.0, 10.0):
        print(f"  模式 B 基线差分 · {y:4.0f} 年后               σ = {floor_mode_B(y):6.1f}")
    print(f"  模式 B · 假设粘接漂移减半 (4 µε/√年)、5 年   σ = {floor_mode_B(5.0, drift=4.0):6.1f}")
    print(f"  模式 B · 假设粘接漂移加倍 (16 µε/√年)、5 年  σ = {floor_mode_B(5.0, drift=16.0):6.1f}")

    print()
    print("=" * 100)
    print("三、信噪比与检出判断（阈值 k=4σ，单次测量全环的虚警率约 0.4%）")
    print("=" * 100)
    floors = {'S·扭矩': fS_t, 'S·拉伸': fS_p, 'B·1年': floor_mode_B(1.0),
              'B·5年': floor_mode_B(5.0), 'B·10年': floor_mode_B(10.0)}
    print(f"  {'缺陷':14s}" + "".join(f"{k:>10s}" for k in floors))
    for name, s in sig.items():
        row = "".join(f"{abs(s)/v:10.1f}" for v in floors.values())
        print(f"  {name:14s}" + row)
    print(f"\n  （表内为 |信号|/σ；>4 才算可检出，>6 才算稳健）")

    print()
    print("=" * 100)
    print("四、预紧力衰减的可检出量（模式 B）")
    print("=" * 100)
    s_FM = 0.646e-3
    for y in (1.0, 5.0, 10.0):
        f = floor_mode_B(y)
        pct = 4.0 * f / (s_FM * FM_D) * 100
        print(f"  基线 {y:4.0f} 年后：最小可检出预紧力变化 = {4*f:5.1f} µε = "
              f"{pct:4.1f}% 设计预紧力（4σ 判据）")

    print()
    print("=" * 100)
    print("五、缺陷分类：用完整轴向剖面做匹配滤波")
    print("=" * 100)
    tmpl = {}
    for nm, key, bkey in (('断柱', 'L1_grav_p|D1_break', 'L1_grav_p|D0_intact'),
                          ('预紧衰减', 'L0_zero|D3_weak40', 'L0_zero|D0_intact'),
                          ('脱粘100', 'L1_grav_p|D4_deb100', 'L1_grav_p|D0_intact'),
                          ('脱粘300', 'L1_grav_p|D4_deb300', 'L1_grav_p|D0_intact')):
        d = eps(key)[J] - eps(bkey)[J]
        tmpl[nm] = d / np.linalg.norm(d)
    print(f"  {'真实缺陷':12s}" + "".join(f"{k:>10s}" for k in tmpl) + "   判定")
    rng = np.random.default_rng(5)
    conf = {}
    for nm, key, bkey in (('断柱', 'L1_grav_p|D1_break', 'L1_grav_p|D0_intact'),
                          ('预紧衰减', 'L0_zero|D3_weak40', 'L0_zero|D0_intact'),
                          ('脱粘 50', 'L1_grav_p|D4_deb50', 'L1_grav_p|D0_intact'),
                          ('脱粘100', 'L1_grav_p|D4_deb100', 'L1_grav_p|D0_intact'),
                          ('脱粘200', 'L1_grav_p|D4_deb200', 'L1_grav_p|D0_intact'),
                          ('脱粘300', 'L1_grav_p|D4_deb300', 'L1_grav_p|D0_intact')):
        xm, m1 = measure_ring(X, eps(key), SPEC, rng, years=1.0)
        xm, m0 = measure_ring(X, eps(bkey), SPEC, rng, years=0.0)
        d = m1[J] - m0[J]
        # 模板重采样到测量网格
        T = {k: np.interp(xm, X, v) for k, v in tmpl.items()}
        sc, best, val = dt.matched_filter(xm, d, T)
        print(f"  {nm:12s}" + "".join(f"{sc[k]:10.2f}" for k in tmpl) + f"   → {best}")
        conf[nm] = (best, val)

    print()
    print("=" * 100)
    print("六、脱粘前沿定位")
    print("=" * 100)
    b0 = eps('L1_grav_p|D0_intact')[J]
    for d_mm, key in ((50, 'D4_deb50'), (100, 'D4_deb100'), (200, 'D4_deb200'), (300, 'D4_deb300')):
        xm, m1 = measure_ring(X, eps('L1_grav_p|' + key), SPEC, rng, years=0.5)
        xm, m0 = measure_ring(X, eps('L1_grav_p|D0_intact'), SPEC, rng, years=0.0)
        xf, amp = dt.debond_front(xm, m1[J] - m0[J])
        print(f"  真实脱粘 {d_mm:3d} mm → 检出前沿 {xf:6.1f} mm（误差 {xf-d_mm:+5.1f} mm），"
              f"凹陷幅值 {amp:7.1f} µε")

    np.savez('out/detect_results.npz', sig=np.array(list(sig.values())),
             sig_names=np.array(list(sig.keys())),
             floors=np.array(list(floors.values())),
             floor_names=np.array(list(floors.keys())))
    print("\n已存 out/detect_results.npz")


if __name__ == '__main__':
    main()
