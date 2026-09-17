# -*- coding: utf-8 -*-
"""
与既有间隙位移监测方案（ONYX ecoPITCH 类）的可比性计算。

该类方案测量叶根端面与轮毂法兰之间的相对位移，分辨率 0.1 mm。
本脚本回答：脱粘发展到什么程度，该位移才达到 0.1 mm，
以及同一时刻光纤读数是多少。两者的先后关系决定了方法的相对价值。

同时输出单个 cell 的轴向刚度随脱粘长度的变化，用于说明位移量的来源。
"""
from __future__ import annotations
import numpy as np
from geometry import Geom
from root_model import RootFE

_G = Geom()              # 几何自 config/ 读入，换机型只改配置
L_INS = _G.L_ins
X_FIB = L_INS + 30.0     # 光纤轴向覆盖上限【推断：埋深再延 30】

FS_INSTALL = 420e3
FA_GRAV, FA_OPER, FA_EXT = 74e3, 159e3, 339e3
GAP_RES = 0.10           # 竞品位移分辨率 mm
TAU_ULT_M = 20.0


def face_disp(fe, res, j):
    """第 j 位叶根层压端面相对法兰平面的轴向位移 [mm]，正值为张开。"""
    u = res['u']
    n0 = fe._node(fe.iy_center[j], 0)
    return u[2 * n0] - res['w_flange'][j]


def solve_w(fe, FM, FA, **kw):
    r = fe.solve(0.0, FM, **kw); r1 = fe.solve(-1e-3, FM, **kw)
    k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3); w = 0.0
    for _ in range(15):
        if abs(r['FA'].mean() - FA) < 30:
            break
        w += (FA - r['FA'].mean()) / k
        r = fe.solve(w, FM, **kw)
    r['w_flange'] = np.full(fe.n_cells, w)
    return w, r


def main():
    L = '=' * 100
    fe = RootFE(n_cells=25, m_y=4)
    FM = fe.calibrate_FM(FS_INSTALL)
    j = 12
    xc_all = None

    print(L); print('一、单个螺套脱粘：端面相对位移 vs 光纤读数'); print(L)
    print('  载荷工况：重力摆振幅值（F_A = 74 kN），即回访时叶片水平的标准载荷')
    print()
    print(f"  {'脱粘':>6} {'剩余承载裕度':>12} {'端面相对位移':>13} {'占 0.1 mm 分辨率':>16} "
          f"{'光纤 |Δε|':>11} {'占 4σ 判据':>12}")
    _, r0 = solve_w(fe, FM, FA_GRAV)
    xc_all = r0['x_c']
    d0 = face_disp(fe, r0, j)
    e0 = r0['eps'][fe.iy_center[j]] * 1e6
    rows = []
    for d in (0, 30, 60, 100, 150, 200, 250, 300, 350,
              L_INS - 90, L_INS - 50, L_INS - 20):
        kw = dict(debond={j: (fe.g.L_ins - d, fe.g.L_ins)}) if d else {}
        _, rd = solve_w(fe, FM, FA_GRAV, **kw)
        gap = face_disp(fe, rd, j) - d0
        ee = rd['eps'][fe.iy_center[j]] * 1e6 - e0
        m = xc_all <= X_FIB
        sig = float(np.abs(ee[m]).max())
        cap = TAU_ULT_M * np.pi * fe.g.D_ins * (fe.g.L_ins - d) / 1e3
        rows.append((d, cap / (FA_EXT / 1e3), gap, sig))
        print(f"  {d:5.0f}mm {cap/(FA_EXT/1e3):11.1f} {gap*1000:10.1f} µm "
              f"{abs(gap)/GAP_RES*100:14.1f}% {sig:10.0f} µε {sig/13.2*100:10.0f}%")

    print()
    print('  注：4σ 判据 = 13.2 µε（模式 B，5 年基线）。位移分辨率按 0.1 mm。')

    print(); print(L); print('二、相邻多个螺套同时脱粘（更接近真实的批次性缺陷）'); print(L)
    fe2 = RootFE(n_cells=41, m_y=4)
    FM2 = fe2.calibrate_FM(FS_INSTALL)
    jj = 20
    _, rb = solve_w(fe2, FM2, FA_OPER)
    db = face_disp(fe2, rb, jj)
    eb = rb['eps'][fe2.iy_center[jj]] * 1e6
    print('  载荷工况：额定挥舞（F_A = 159 kN）')
    print()
    print(f"  {'同时脱粘个数':>12} {'各自脱粘长度':>12} {'端面相对位移':>13} "
          f"{'达 0.1 mm?':>11} {'光纤 |Δε|':>11}")
    for k, dlen in ((1, 300.0), (3, 300.0), (5, 300.0), (5, 400.0),
                    (9, 400.0), (9, 460.0), (15, 460.0)):
        deb = {(jj + i) % 41: (fe2.g.L_ins - dlen, fe2.g.L_ins)
               for i in range(-(k // 2), k - k // 2)}
        _, rk = solve_w(fe2, FM2, FA_OPER, debond=deb)
        gap = face_disp(fe2, rk, jj) - db
        ee = rk['eps'][fe2.iy_center[jj]] * 1e6 - eb
        sig = float(np.abs(ee[rk['x_c'] <= X_FIB]).max())
        ok = '是' if abs(gap) >= GAP_RES else '否'
        print(f"  {k:11d} {dlen:11.0f}mm {gap*1000:10.1f} µm {ok:>10} {sig:10.0f} µε")

    print(); http = None
    print(L); print('三、cell 轴向刚度随脱粘长度的变化（位移量的来源）'); print(L)
    print(f"  {'脱粘':>6} {'cell 轴向刚度':>14} {'相对完好':>10}")
    k0 = None
    for d in (0, 100, 200, 300, 400, 460):
        kw = dict(debond={j: (fe.g.L_ins - d, fe.g.L_ins)}) if d else {}
        ra = fe.solve(0.0, FM, **kw); rb2 = fe.solve(-1e-3, FM, **kw)
        kk = abs((rb2['FA'].mean() - ra['FA'].mean()) / (-1e-3))
        if k0 is None:
            k0 = kk
        print(f"  {d:5.0f}mm {kk/1e3:12.0f} kN/mm {kk/k0*100:9.1f}%")
    print()
    print('  刚度几乎不变，是因为载荷仍由 117 个完好位共同承担；')
    print('  单个位置脱粘不改变整环刚度，端面相对位移因此极小。')


if __name__ == '__main__':
    main()
