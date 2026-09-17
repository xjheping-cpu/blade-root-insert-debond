# -*- coding: utf-8 -*-
"""
拔出通道的两个关键追问：
  Q1 界面剪应力的峰值到底在哪一端？这决定脱粘从哪头开始长，
     进而决定光纤该盯哪一段、信号长什么样。
  Q2 相邻拔出后，邻位的界面应力是不是按其外载同比例上升？
     这决定级联有多快。
再给出两端脱粘的对照信号。
"""
from __future__ import annotations
import numpy as np
from root_model import RootFE
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


FS_INSTALL = 420e3
FA_GRAV, FA_OPER, FA_EXT = 74e3, 159e3, 339e3


def ishear(fe, res, j):
    u = res['u']
    b = fe.ndof_plate + j * fe.n_ins_node
    us = u[b: b + fe.n_ins_node]
    ul = u[2 * fe._node(fe.iy_center[j], np.arange(fe.n_ins_node))]
    return fe.x_ins, fe._k_q * (us - ul) / (np.pi * fe.g.D_ins)


def insert_force(fe, res, j):
    EA = fe.m.E_steel * fe.g.A_steel
    b = fe.ndof_plate + j * fe.n_ins_node
    us = res['u'][b: b + fe.n_ins_node]
    xc = 0.5 * (fe.x_ins[:-1] + fe.x_ins[1:])
    return xc, EA * np.diff(us) / np.diff(fe.x_ins)


def solve_w(fe, FM, FA, **kw):
    r = fe.solve(0.0, FM, **kw); r1 = fe.solve(-1e-3, FM, **kw)
    k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3); w = 0.0
    for _ in range(15):
        if abs(r['FA'].mean() - FA) < 30:
            break
        w += (FA - r['FA'].mean()) / k
        r = fe.solve(w, FM, **kw)
    return w, r


def main():
    L = "=" * 100
    fe = RootFE(n_cells=25, m_y=4)
    j = 12

    print(L); print("Q1 界面剪应力峰值在哪一端，随端面构造怎么变"); print(L)
    print(f"  {'gap_s':>7} {'χ':>7} | {'峰值τ':>9} {'峰值位置':>9} | "
          f"{'端面段 0-150 峰值':>17} "
          f"{'埋入端 %.0f-%.0f 峰值' % (_G.L_ins - 90, _G.L_ins):>19} | 谁主导")
    for gap in (0.0, 0.02, 0.05, 0.10, 0.20):
        FM = fe.calibrate_FM(FS_INSTALL, gap_s=gap)
        _, r0 = solve_w(fe, FM, 0.0, gap_s=gap)
        _, re = solve_w(fe, FM, FA_EXT, gap_s=gap)
        x, t = ishear(fe, re, j)
        pk = np.abs(t).max(); xpk = x[np.argmax(np.abs(t))]
        m1 = x <= 150; m2 = x >= 400
        p1, p2 = np.abs(t[m1]).max(), np.abs(t[m2]).max()
        who = "埋入端" if p2 > p1 else "端面段"
        print(f"  {gap:7.2f} {r0['chi'].mean():7.3f} | {pk:8.1f}MPa {xpk:8.0f}mm | "
              f"{p1:16.1f} {p2:18.1f} | {who}")

    print()
    print("  同一位置的螺套轴力 N_s(x)，看载荷是在哪一段被交出去的（gap_s=0，极端工况）：")
    FM = fe.calibrate_FM(FS_INSTALL)
    _, re = solve_w(fe, FM, FA_EXT)
    xc, Ns = insert_force(fe, re, j)
    idx = [np.argmin(np.abs(xc - v)) for v in (5, 50, 105, 200, 300, 400, 450, 485)]
    print("  x [mm]      " + "".join(f"{xc[i]:8.0f}" for i in idx))
    print("  N_s [kN]    " + "".join(f"{Ns[i]/1e3:8.0f}" for i in idx))
    print("  → 螺柱力在 0~105 mm 的啮合段进入螺套，端面接触把大部分顶回去；")
    print("    螺套只把其中一部分（≈" + f"{Ns[idx[3]]/1e3:.0f} kN）带到深处，再在埋入端交给层压。")

    print(); print(L); print("Q2 邻位界面应力是否按外载同比例上升"); print(L)
    fe2 = RootFE(n_cells=41, m_y=4)
    FM2 = fe2.calibrate_FM(FS_INSTALL)
    jj, tot = 20, FA_EXT * 41
    print(f"  {'已拔出':>7} {'邻位外载':>11} {'邻位峰值τ':>12} {'τ/τ(k=0)':>10} "
          f"{'F_A/F_A(k=0)':>13} {'埋入端τ':>10}")
    t_ref = None
    for k in (0, 1, 2, 3, 5, 8):
        deb = ({(jj + i) % 41: _G.L_ins
                for i in range(-(k // 2), k - k // 2)} if k else {})
        r = fe2.solve(0.0, FM2, debond=deb); r1 = fe2.solve(-1e-3, FM2, debond=deb)
        kk = (r1['FA'].sum() - r['FA'].sum()) / (-1e-3); w = 0.0
        for _ in range(15):
            if abs(r['FA'].sum() - tot) < 1e3:
                break
            w += (tot - r['FA'].sum()) / kk
            r = fe2.solve(w, FM2, debond=deb)
        nb = (jj - k // 2 - 1) % 41
        x, t = ishear(fe2, r, nb)
        pk = np.abs(t).max()
        pe = np.abs(t[x >= 400]).max()
        if t_ref is None:
            t_ref, f_ref = pk, r['FA'][nb]
        print(f"  {k:6d} {r['FA'][nb]/1e3:10.0f}kN {pk:11.1f}MPa {pk/t_ref:9.2f} "
              f"{r['FA'][nb]/f_ref:12.2f} {pe:9.1f}MPa")

    print(); print(L); print("Q3 两端脱粘的对照：OFDR 信号谁更好认"); print(L)
    print('  端面侧脱粘 = 从 x=0 向内长；埋入端脱粘 = 从 x=%.0f 向外长。'
          % _G.L_ins)
    FMg = fe.calibrate_FM(FS_INSTALL)
    _, rg = solve_w(fe, FMg, FA_GRAV)
    e0 = rg['eps'][fe.iy_center[j]] * 1e6
    xcp = rg['x_c']
    fw = (xcp >= 0) & (xcp <= 40)
    print()
    print(f"  {'脱粘长度':>9} | {'端面侧 端面Δ':>14} {'最大|Δ|':>9} {'位置':>8} | "
          f"{'埋入端 端面Δ':>14} {'最大|Δ|':>9} {'位置':>8}")
    for d in (30, 50, 100, 200, 300, 400):
        _, ra = solve_w(fe, FMg, FA_GRAV, debond={j: float(d)})
        ea = ra['eps'][fe.iy_center[j]] * 1e6 - e0
        _, rb = solve_w(fe, FMg, FA_GRAV, debond={j: (fe.g.L_ins - d, fe.g.L_ins)})
        eb = rb['eps'][fe.iy_center[j]] * 1e6 - e0
        m = xcp <= _G.L_ins + 30.0
        ia = np.argmax(np.abs(ea[m])); ib = np.argmax(np.abs(eb[m]))
        print(f"  {d:8.0f}mm | {ea[fw].mean():13.1f} {ea[m][ia]:8.1f} {xcp[m][ia]:7.0f}mm | "
              f"{eb[fw].mean():13.1f} {eb[m][ib]:8.1f} {xcp[m][ib]:7.0f}mm")
    print()
    print("  判读：两种脱粘的信号位置不同，因此不但都能看见，还能区分是从哪一端长的。")
    print('  埋入端脱粘的信号出现在螺套末端前后，仍在根部舱'
          '（0~%.0f mm）内，光纤够得着。' % _G.x_baffle)


if __name__ == '__main__':
    main()
