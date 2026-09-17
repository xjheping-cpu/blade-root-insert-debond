# -*- coding: utf-8 -*-
"""
补充工况：
  A 方位角扫掠（弦向轴 j=29，1P 幅值最大位）：完好 vs 断柱 → 1P 波形判据
  B 方位角扫掠（挥舞轴 j=0，均值最大位）：不同预紧水平 → 接头张口的拐点判据
  C 细脱粘长度（10~300 mm）→ 脱粘可辨阈值
  D 端面构造 gap_s 敏感性（第一不确定性）
  E 安装预紧力离散的实现（无基线空间检测的噪声底）

求解策略：每个方位角先用完好工况按目标弯矩解出刚性法兰的三个自由度，再把同一组
法兰位移施加到各缺陷工况上。物理上这是对的——一根螺柱断掉只改变接头刚度的 1/n，
叶片承受的气动与重力载荷不会因此改变；数值上省掉每个缺陷工况的牛顿迭代。
"""
from __future__ import annotations
import time
import numpy as np
from root_model import RootFE
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置

FS = 420.0e3
M_GRAV = 7.0e9
M_FLAP = 15.0e9
J_EDGE, J_FLAP = 29, 0
NAZ = 16


def main():
    t0 = time.time()
    fe = RootFE(n_cells=_G.n_bolt, m_y=4)
    FM0 = fe.calibrate_FM(FS)
    FMv = np.full(_G.n_bolt, FM0)
    th = fe.iy_center * fe.dy / fe.g.R_bc
    basis = np.stack([np.ones_like(th), fe.g.R_bc * np.cos(th), fe.g.R_bc * np.sin(th)])
    out = {}
    psi = np.linspace(0, 2 * np.pi, NAZ, endpoint=False)

    def ec(r):
        return r['eps'][fe.iy_center % fe.ny, :].astype(np.float32)

    # 先把每个方位角的法兰位移标定出来（完好工况，目标弯矩）
    print("标定各方位角的法兰位移…", flush=True)
    P = []
    p = None
    for ps in psi:
        r = fe.solve_for_moment(M_flap=M_FLAP, M_edge=M_GRAV * np.cos(ps), F_M=FMv, p0=p)
        p = r['flange']; P.append(p.copy())
    P = np.array(P)
    print(f"  done {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- A 弦向轴波形
    print("A 方位角扫掠（弦向轴 j=29）", flush=True)
    for tag, kw in (('intact', {}), ('break', dict(broken=[J_EDGE]))):
        E, F = [], []
        for k in range(NAZ):
            r = fe.solve(P[k] @ basis, FMv, **kw)
            E.append(ec(r)); F.append(r['FA'])
        out[f'A_{tag}_eps'] = np.array(E); out[f'A_{tag}_FA'] = np.array(F)
    print(f"  done {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- B 张口
    print("B 方位角扫掠（挥舞轴 j=0），不同预紧水平", flush=True)
    for frac in (1.00, 0.45, 0.30, 0.20):
        FMw = FMv.copy(); FMw[J_FLAP] = FM0 * frac
        E, F, NO = [], [], []
        for k in range(NAZ):
            r = fe.solve(P[k] @ basis, FMw)
            E.append(ec(r)); F.append(r['FA'])
            NO.append(int((r['state'] == 'open').sum()))
        out[f'B_{int(frac*100)}_eps'] = np.array(E)
        out[f'B_{int(frac*100)}_FA'] = np.array(F)
        out[f'B_{int(frac*100)}_nopen'] = np.array(NO)
        print(f"  预紧 {frac*100:3.0f}%: 各方位角张口 cell 数 {NO}", flush=True)
    print(f"  done {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- C 细脱粘
    print("C 脱粘长度扫掠（重力工况）", flush=True)
    rg = fe.solve_for_moment(M_flap=0.0, M_edge=M_GRAV, F_M=FMv)
    wg = rg['flange'] @ basis
    debs = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300]
    E = []
    for d in debs:
        kw = dict(debond={J_EDGE: float(d)}) if d else {}
        E.append(ec(fe.solve(wg, FMv, **kw)))
    out['C_deb_eps'] = np.array(E); out['C_deb_len'] = np.array(debs, float)
    print(f"  done {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- D gap 敏感性
    print("D 端面构造 gap_s 敏感性", flush=True)
    for gap in (0.0, 0.02, 0.05, 0.20):
        FMg = fe.calibrate_FM(FS, gap_s=gap)
        rr = fe.solve_for_moment(M_flap=0.0, M_edge=M_GRAV, F_M=np.full(_G.n_bolt, FMg), gap_s=gap)
        wgg = rr['flange'] @ basis
        a = ec(fe.solve(wgg, np.full(_G.n_bolt, FMg), gap_s=gap))
        b = ec(fe.solve(wgg, np.full(_G.n_bolt, FMg), broken=[J_EDGE], gap_s=gap))
        out[f'D_gap{int(gap*100):03d}_eps'] = np.array([a, b])
        fw = (rr['x_c'] >= 0) & (rr['x_c'] <= 40)
        print(f"  gap={gap:.2f} mm: 断柱 face 信号 "
              f"{(b[J_EDGE, fw].mean()-a[J_EDGE, fw].mean())*1e6:+.0f} µε", flush=True)

    # ---------------------------------------------------------------- E 预紧离散
    print("E 安装预紧力离散", flush=True)
    rng = np.random.default_rng(7)
    for tag, cov in (('torque', 0.27), ('tensioner', 0.09)):
        FMs = FM0 * rng.normal(1.0, cov, _G.n_bolt).clip(0.35, 1.6)
        r = fe.solve(wg, FMs)
        out[f'E_{tag}_eps'] = ec(r); out[f'E_{tag}_FM'] = FMs
        fw = (r['x_c'] >= 0) & (r['x_c'] <= 40)
        print(f"  {tag} (CoV={cov}): face 特征离散 std="
              f"{np.std(out[f'E_{tag}_eps'][:, fw].mean(axis=1))*1e6:.0f} µε", flush=True)

    out['x_c'] = rg['x_c']; out['psi'] = psi; out['theta'] = th
    np.savez_compressed('out/scenarios.npz', **out)
    print(f"总耗时 {time.time()-t0:.0f}s → out/scenarios.npz", flush=True)


if __name__ == '__main__':
    main()
