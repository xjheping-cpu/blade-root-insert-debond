# -*- coding: utf-8 -*-
"""
模型校验（V&V）。三件事：
  一、M1（二维展开壳-杆）与 M0（一维剪滞，独立实现）在同一物理下互校；
  二、M0 在退化情形下与闭式剪滞解互校；
  三、M1 的网格、窗口宽度、远场边界距离收敛性，以及远场应变与梁理论对照。
再顺带把 Φ、χ、传力长度这三个 V2.0 报告只能假设的量提取出来。
"""
from __future__ import annotations
import time
import numpy as np
from root_model import RootFE, Geom, Mat, Stud
from onedim import OneDCell, closed_form_check

FS_INSTALL = 420.0e3      # 安装轴力示例 N（0.4 F_0.2）
FA_REF = 74.0e3           # 摆振 1P 幅值对应的单柱附加轴力 N


def solve_FA_2d(fe: RootFE, FA, FM_in, gap_s=0.0, broken=None, debond=None):
    """二维模型：割线迭代求满足目标平均外载的法兰位移。"""
    w, r = 0.0, None
    r = fe.solve(0.0, FM_in, broken, debond, gap_s)
    r1 = fe.solve(-1e-3, FM_in, broken, debond, gap_s)
    k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3)
    for _ in range(20):
        if abs(r['FA'].mean() - FA) < max(20.0, 1e-5 * abs(FA)):
            break
        w += (FA - r['FA'].mean()) / k
        r = fe.solve(w, FM_in, broken, debond, gap_s)
    r['w'] = w
    return r


def main():
    L = "=" * 100
    print(L); print("一、M0（一维剪滞）自校验：退化情形与闭式解"); print(L)
    c1 = OneDCell()
    lam, Ltr = closed_form_check(c1.m, c1.g, c1.k_q)
    print(f"  闭式：k_q={c1.k_q:.0f} N/mm/mm  λ={lam:.5f} /mm  传力长度 1/λ={Ltr:.1f} mm")
    # 退化：无端面接触（gap 很大）、螺柱只作用在第一个节点
    c2 = OneDCell()
    c2.w_eng = np.zeros_like(c2.w_eng); c2.w_eng[0] = 1.0
    r = c2.solve(w_flange=-0.05, F_M=0.0, gap_s=10.0)
    Ns = r['Ns']; xi = r['x_ins_c']
    Ns_inf = Ns[len(Ns) // 2]
    tgt = Ns_inf + (Ns[0] - Ns_inf) / np.e
    Ltr_num = np.interp(tgt, Ns[:len(Ns)//2][::-1], xi[:len(Ns)//2][::-1])
    print(f"  数值：N_s 从端部衰减到 1/e 的长度 = {Ltr_num:.1f} mm  → 与闭式偏差 "
          f"{abs(Ltr_num-Ltr)/Ltr*100:.1f}%")
    EA_s = c1.m.E_steel * c1.g.A_steel
    EA_l = c1.m.E_lam_x * c1.g.A_lam_cell
    share = EA_s / (EA_s + EA_l)
    print(f"  远场钢/层压刚度分配：闭式 {share*100:.1f}%  数值 {Ns_inf/r['FA']*100:.1f}%")

    print(); print(L); print("二、M1 与 M0 互校（同一物理，独立实现）"); print(L)
    fe = RootFE(n_cells=9, m_y=4)
    print(f"  {'gap_s':>7} | {'':>4}  Φ(M1)   Φ(M0) | {'':>3}χ(M1)   χ(M0) | "
          f"N_s(x=10) M1/M0 [kN] | ε_lam(x=200) M1/M0 [µε]")
    for gap in (0.0, 0.05, 0.30):
        FM1 = fe.calibrate_FM(FS_INSTALL, gap_s=gap)
        a0 = solve_FA_2d(fe, 0.0, FM1, gap_s=gap)
        a1 = solve_FA_2d(fe, FA_REF, FM1, gap_s=gap)
        Phi1 = (a1['FS'].mean() - a0['FS'].mean()) / (a1['FA'].mean() - a0['FA'].mean())
        FM0 = c1.calibrate_FM(FS_INSTALL, gap_s=gap)
        b0 = c1.solve_FA(0.0, FM0, gap_s=gap)
        b1 = c1.solve_FA(FA_REF, FM0, gap_s=gap)
        Phi0 = (b1['FS'] - b0['FS']) / (b1['FA'] - b0['FA'])
        j = fe.n_cells // 2
        i2 = np.argmin(np.abs(0.5 * (fe.x_ins[:-1] + fe.x_ins[1:]) - 10.0))
        i3 = np.argmin(np.abs(a1['x_c'] - 200.0))
        # M1 的 cell 平均层压应变（按厚度加权），才能和 M0 的一维量比
        wgt = fe.t_elem[:, i3] * fe.dy
        sl = (fe.face_w[j] > 0)
        eps1 = np.average(a1['eps'][sl, i3], weights=wgt[sl])
        print(f"  {gap:7.2f} | {Phi1:9.4f} {Phi0:8.4f} | {a0['chi'].mean():7.4f} {b0['chi']:7.4f} | "
              f"{a1['Ns'][j][i2]/1e3:9.1f} {b1['Ns'][i2]/1e3:7.1f} | "
              f"{eps1*1e6:11.0f} {b1['eps'][i3]*1e6:8.0f}")

    print(); print(L); print("三、M1 收敛性"); print(L)
    def quick(fe_, tag, gap=0.0):
        t0 = time.time()
        FM = fe_.calibrate_FM(FS_INSTALL, gap_s=gap)
        a0 = solve_FA_2d(fe_, 0.0, FM, gap_s=gap)
        a1 = solve_FA_2d(fe_, FA_REF, FM, gap_s=gap)
        Phi = (a1['FS'].mean() - a0['FS'].mean()) / (a1['FA'].mean() - a0['FA'].mean())
        j = fe_.n_cells // 2
        i = np.argmin(np.abs(a1['x_c'] - 200.0))
        de = (a1['eps'][fe_.iy_center[j], i] - a0['eps'][fe_.iy_center[j], i]) * 1e6
        print(f"  [{tag:>12}] DOF={fe_.ndof:>7d} {time.time()-t0:5.1f}s  "
              f"Φ={Phi:.4f}  χ={a0['chi'].mean():.4f}  Δε(200mm)={de:6.1f} µε")
        return Phi, de
    for nc in (5, 9, 15, 25):
        quick(RootFE(n_cells=nc, m_y=4), f"窗口{nc}cell")
    for my in (2, 4, 6, 8):
        quick(RootFE(n_cells=9, m_y=my), f"m_y={my}")
    for xm in (600.0, 900.0, 1400.0):
        quick(RootFE(n_cells=9, m_y=4, geom=Geom(x_max=xm)), f"x_max={xm:.0f}")

    print(); print(L); print("四、远场与梁理论对照（检查周向广义周期自由度是否正确）"); print(L)
    fe = RootFE(n_cells=9, m_y=4, geom=Geom(x_max=1400.0))
    FM = fe.calibrate_FM(FS_INSTALL)
    a0 = solve_FA_2d(fe, 0.0, FM); a1 = solve_FA_2d(fe, FA_REF, FM)
    dFA = a1['FA'].mean() - a0['FA'].mean()
    beam = dFA / (fe.g.pitch * fe.g.t_wall * fe.m.E_lam_x) * 1e6
    for xp in (700.0, 900.0, 1100.0):
        i = np.argmin(np.abs(a1['x_c'] - xp))
        de = (a1['eps'][:, i] - a0['eps'][:, i]).mean() * 1e6
        print(f"  x={xp:6.0f} mm  Δε(FE 周向平均)={de:7.1f} µε   梁理论={beam:7.1f} µε   "
              f"偏差 {abs(de-beam)/beam*100:4.1f}%")

    print(); print(L); print("五、提取三个关键量（V2.0 报告只能假设、这里是解出来的）"); print(L)
    fe = RootFE(n_cells=15, m_y=4)
    print(f"  {'gap_s [mm]':>10} | {'Φ':>6} | {'χ':>6} | {'F_A,open':>9} | {'安装后端面层压压应变':>12}")
    rows = []
    for gap in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
        FM = fe.calibrate_FM(FS_INSTALL, gap_s=gap)
        a0 = solve_FA_2d(fe, 0.0, FM, gap_s=gap)
        a1 = solve_FA_2d(fe, FA_REF, FM, gap_s=gap)
        Phi = (a1['FS'].mean() - a0['FS'].mean()) / (a1['FA'].mean() - a0['FA'].mean())
        chi = a0['chi'].mean()
        FAopen = FS_INSTALL / (1 - Phi) / 1e3
        e0 = a0['eps'][fe.iy_center[7], 0] * 1e6
        print(f"  {gap:10.2f} | {Phi:6.3f} | {chi:6.3f} | {FAopen:7.0f} kN | {e0:10.0f} µε")
        rows.append([gap, Phi, chi, FAopen, e0])
    np.savetxt('out/verify_gap.csv', np.array(rows), delimiter=',',
               header='gap_s_mm,Phi,chi,FA_open_kN,eps_face_ue', comments='')
    print("\n  传力长度 1/λ = %.1f mm（解析）" % (1 / fe.lam_analytic))
    print("  钢/层压轴向刚度分配 = %.1f%% / %.1f%%" % (
        fe.m.E_steel * fe.g.A_steel / (fe.m.E_steel * fe.g.A_steel + fe.m.E_lam_x * fe.g.A_lam_cell) * 100,
        fe.m.E_lam_x * fe.g.A_lam_cell / (fe.m.E_steel * fe.g.A_steel + fe.m.E_lam_x * fe.g.A_lam_cell) * 100))


if __name__ == '__main__':
    main()
