# -*- coding: utf-8 -*-
"""
缺陷工况仿真：在整环模型上，把 V2.0 报告列出的失效模式逐个加进去，
看根部舱内表面的轴向应变场怎么变。输出存 out/defect_cases.npz 供检测与作图使用。

工况族：
  载荷  L0 停机叶片竖直向下（≈零载）
        L1 停机叶片水平，重力摆振弯矩 +7 MN·m（回访用的"已知标准载荷"）
        L2 同上反向 -7 MN·m（叶轮转 180°，用于半波不对称判据）
        L3 运行工况：挥舞均值 15 MN·m + 摆振 ±7 MN·m
  缺陷  D0 完好
        D1 单根螺柱断裂（最大受拉位）
        D2 单根螺柱断裂（中性位，考察定位灵敏度对位置的依赖）
        D3 单个螺柱预紧力衰减至 40%（考察预紧力可见性）
        D4 螺套脱粘 50 / 100 / 200 / 300 mm
        D5 相邻两根断裂
"""
from __future__ import annotations
import time
import numpy as np
from root_model import RootFE, Geom
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置
L_INS = _G.L_ins

FS_INSTALL = 420.0e3
M_GRAV = 7.0e9          # N·mm，叶片水平时的重力摆振弯矩（80 m 级示例）
M_FLAP = 15.0e9         # N·mm，额定附近挥舞均值
J_DEF = 29              # 缺陷所在 cell：+M_edge 下的最大受拉位
J_NEU = 0               # 中性位
GAP_S = 0.0             # 螺套端面与层压端面齐平（基准构造；gap 敏感性另算）


def run():
    t_all = time.time()
    fe = RootFE(n_cells=_G.n_bolt, m_y=4)
    FM = fe.calibrate_FM(FS_INSTALL, gap_s=GAP_S)
    th = fe.iy_center * fe.dy / fe.g.R_bc
    print(f"整环模型 DOF={fe.ndof}，安装轴力 {FS_INSTALL/1e3:.0f} kN（输入 F_M={FM/1e3:.1f} kN），"
          f"缺陷位 j={J_DEF} (θ={np.degrees(th[J_DEF]):.1f}°)")

    loads = {
        'L0_zero':   dict(M_flap=0.0,    M_edge=0.0),
        'L1_grav_p': dict(M_flap=0.0,    M_edge=+M_GRAV),
        'L2_grav_n': dict(M_flap=0.0,    M_edge=-M_GRAV),
        'L3_oper':   dict(M_flap=M_FLAP, M_edge=+M_GRAV),
    }
    FMv = np.full(_G.n_bolt, FM)
    FM_weak = FMv.copy(); FM_weak[J_DEF] = FM * 0.40
    defects = {
        'D0_intact':  dict(),
        'D1_break':   dict(broken=[J_DEF]),
        'D2_break_n': dict(broken=[J_NEU]),
        'D3_weak40':  dict(F_M=FM_weak),
        'D4_deb50':   dict(debond={J_DEF: 50.0}),
        'D4_deb100':  dict(debond={J_DEF: 100.0}),
        'D4_deb200':  dict(debond={J_DEF: 200.0}),
        'D4_deb300':  dict(debond={J_DEF: 300.0}),
        'D5_break2':  dict(broken=[J_DEF, J_DEF + 1]),
    }

    out = {}
    p_prev = {}
    for lk, lv in loads.items():
        for dk, dv in defects.items():
            key = f"{lk}|{dk}"
            kw = dict(F_M=FMv, gap_s=GAP_S)
            kw.update(dv)
            t0 = time.time()
            r = fe.solve_for_moment(p0=p_prev.get(lk), **lv, **kw)
            p_prev[lk] = r['flange']
            # 只存需要的量：每个螺套中心线的轴向应变剖面 + cell 力 + 螺套轴力
            out[key + '|eps_c'] = r['eps'][fe.iy_center % fe.ny, :].astype(np.float32)
            mid = (fe.iy_center + fe.m_y // 2) % fe.ny
            out[key + '|eps_m'] = r['eps'][mid, :].astype(np.float32)
            out[key + '|FA'] = r['FA']
            out[key + '|FS'] = r['FS']
            out[key + '|Cl'] = r['Cl']
            out[key + '|Cs'] = r['Cs']
            out[key + '|nopen'] = np.array([int((r['state'] == 'open').sum())])
            print(f"  {key:24s} {time.time()-t0:5.1f}s  "
                  f"F_A[{J_DEF}]={r['FA'][J_DEF]/1e3:7.1f} kN  "
                  f"F_A[{J_DEF+1}]={r['FA'][J_DEF+1]/1e3:7.1f}  "
                  f"张口 {int((r['state']=='open').sum()):3d} 个")

    out['x_c'] = r['x_c']
    out['theta'] = th
    out['meta'] = np.array([FS_INSTALL, FM, M_GRAV, M_FLAP, J_DEF, J_NEU, GAP_S, fe.g.pitch])
    np.savez_compressed('out/defect_cases.npz', **out)
    print(f"\n总耗时 {time.time()-t_all:.0f}s，已存 out/defect_cases.npz")

    # --------------------------------------------------------------- 关键数字
    xc = r['x_c']
    def prof(key, j):
        return out[key + '|eps_c'][j] * 1e6

    print("\n" + "=" * 96)
    print("断柱信号：缺陷位 j=29 的轴向应变剖面变化 [µε]")
    print("=" * 96)
    _xp = (5, 20, 50, 100, 200, 300, L_INS - 40, L_INS - 10,
           L_INS + 70, 700)
    idx = [np.argmin(np.abs(xc - v)) for v in _xp]
    hdr = "  " + "".join(f"{xc[i]:8.0f}" for i in idx)
    print("  x [mm]        " + hdr)
    for lk in loads:
        a = prof(f"{lk}|D0_intact", J_DEF); b = prof(f"{lk}|D1_break", J_DEF)
        print(f"  {lk:11s} 完好" + "".join(f"{a[i]:8.0f}" for i in idx))
        print(f"  {'':11s} 断柱" + "".join(f"{b[i]:8.0f}" for i in idx))
        print(f"  {'':11s} 差值" + "".join(f"{b[i]-a[i]:8.0f}" for i in idx))

    print("\n" + "=" * 96)
    print("载荷改道：断柱后相邻螺套分担的变化（L1 重力工况，标称 F_A=74 kN）")
    print("=" * 96)
    a = out['L1_grav_p|D0_intact|FA']; b = out['L1_grav_p|D1_break|FA']
    for d in range(-4, 5):
        j = (J_DEF + d) % _G.n_bolt
        print(f"  j={j:3d} (Δ={d:+2d})  完好 {a[j]/1e3:7.2f} kN → 断柱后 {b[j]/1e3:7.2f} kN  "
              f"变化 {(b[j]-a[j])/1e3:+6.2f} kN ({(b[j]-a[j])/74e3*100:+5.1f}% 标称)")
    print(f"  合计变化 {np.sum(b-a)/1e3:+.2f} kN（应≈0，载荷守恒）")

    print("\n" + "=" * 96)
    print("预紧力可见性：单柱预紧力降到 40% 的信号（L0 零外载）")
    print("=" * 96)
    a = prof("L0_zero|D0_intact", J_DEF); b = prof("L0_zero|D3_weak40", J_DEF)
    c = prof("L0_zero|D1_break", J_DEF)
    print("  x [mm]        " + hdr)
    print("  完好        " + "".join(f"{a[i]:8.0f}" for i in idx))
    print("  预紧 40%    " + "".join(f"{b[i]:8.0f}" for i in idx))
    print("  断柱        " + "".join(f"{c[i]:8.0f}" for i in idx))

    print("\n" + "=" * 96)
    print("脱粘：传力起升点内移（L1 重力工况，减去完好基线）")
    print("=" * 96)
    print("  x [mm]        " + hdr)
    base = prof("L1_grav_p|D0_intact", J_DEF)
    for d in (50, 100, 200, 300):
        v = prof(f"L1_grav_p|D4_deb{d}", J_DEF) - base
        print(f"  脱粘 {d:3d} mm " + "".join(f"{v[i]:8.0f}" for i in idx))


if __name__ == '__main__':
    run()
