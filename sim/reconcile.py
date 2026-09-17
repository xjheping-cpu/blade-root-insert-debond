# -*- coding: utf-8 -*-
"""
把 M1（二维）与 M2（三维）不一致的两处查清楚：

  差异 1  载荷改道份额：三维 j±1 得 +52%，二维 +32%。
          怀疑是边界条件不同（三维固定法兰位移，二维原来是按目标弯矩重解），
          以及窗口宽度。这里用完全相同的条件重做。

  差异 2  界面剪切刚度：三维由网格自然算出 k_q_eff ≈ 29900 N/mm/mm
          （传力长度 63 mm），二维用 t_lam_shear=16 mm 的剪滞当量厚度假设出
          k_q = 100098（34.5 mm）。这条假设是二维模型里最弱的一环，
          这里按三维标定后重算，看对检测结论的影响。
"""
from __future__ import annotations
import numpy as np
from root_model import RootFE, Mat
from geometry import Geom
_G = Geom()          # 几何自 config/ 读入，换机型只改配置


FS = 420e3
FA_T = 74e3


def uniform_w_for(fe, FM, FA_target, **kw):
    """固定均匀法兰位移，割线迭代到目标平均外载。"""
    r = fe.solve(0.0, FM, **kw)
    r1 = fe.solve(-1e-3, FM, **kw)
    k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3)
    w = 0.0
    for _ in range(15):
        if abs(r['FA'].mean() - FA_target) < 30:
            break
        w += (FA_target - r['FA'].mean()) / k
        r = fe.solve(w, FM, **kw)
    return w, r


def part1():
    print("=" * 96)
    print("差异 1：同条件下的载荷改道（固定法兰位移，与 M2 完全一致）")
    print("=" * 96)
    print(f"  {'窗口':>8} {'j断':>9} {'j±1':>8} {'j±2':>8} {'j±3':>8} {'j±4':>8}   备注")
    for nc in (9, 15, 25, 41):
        fe = RootFE(n_cells=nc, m_y=4)
        FM = fe.calibrate_FM(FS)
        w, r0 = uniform_w_for(fe, FM, FA_T)
        j = nc // 2
        r1 = fe.solve(w, FM, broken=[j])           # 同一个 w，只是拆掉螺柱
        d = (r1['FA'] - r0['FA']) / FA_T
        row = [d[j]] + [0.5 * (d[(j + k) % nc] + d[(j - k) % nc]) for k in (1, 2, 3, 4)]
        note = 'M2 用的窗口' if nc == 9 else ''
        print(f"  {nc:6d}cell {row[0]:9.3f} {row[1]:8.3f} {row[2]:8.3f} {row[3]:8.3f} "
              f"{row[4]:8.3f}   {note}")
    print(f"  {'三维 M2':>8} {-1.522:9.3f} {0.522:8.3f} {0.100:8.3f} {0.033:8.3f} {0.014:8.3f}   9cell")
    print()
    print("  说明：二维原来的 -1.000 是按目标弯矩重解得到的（断柱后法兰会微调），")
    print("        固定法兰位移时断柱位会被邻位拖着压回去，因此 < -1。")


def part2():
    print()
    print("=" * 96)
    print("差异 2：界面剪切刚度按 M2 标定后的影响")
    print("=" * 96)
    # 反推能给出 1/λ = 63 mm 的 t_lam_shear
    base = Mat()
    EA_s = base.E_steel * RootFE(n_cells=5).g.A_steel
    EA_l = base.E_lam_x * RootFE(n_cells=5).g.A_lam_cell
    target_lam = 1 / 63.0
    kq_t = target_lam ** 2 / (1 / EA_s + 1 / EA_l)
    per = np.pi * _G.D_ins
    comp = per / kq_t                      # = t_adh/G_adh + t_lam/G_lam
    t_lam = (comp - base.t_adh / base.G_adh) * base.G_lam
    print(f"  为得到 M2 的 1/λ=63 mm，需 k_q={kq_t:.0f} N/mm/mm，"
          f"对应剪滞当量厚度 t_lam_shear={t_lam:.1f} mm（原假设 16 mm）")
    m2 = Mat(t_lam_shear=float(t_lam))
    print()
    print(f"  {'量':38s} {'原 (16mm)':>12} {'标定后':>12} {'变化':>10}")
    rows = []
    for tag, mat in (('orig', base), ('cal', m2)):
        fe = RootFE(n_cells=15, m_y=4, mat=mat)
        FM = fe.calibrate_FM(FS)
        w0, r0 = uniform_w_for(fe, FM, 0.0)
        w1, r1 = uniform_w_for(fe, FM, FA_T)
        j = 7
        rb = fe.solve(w0, FM, broken=[j])
        fw = (r0['x_c'] >= 0) & (r0['x_c'] <= 40)
        face0 = r0['eps'][fe.iy_center[j], fw].mean() * 1e6
        faceb = rb['eps'][fe.iy_center[j], fw].mean() * 1e6
        Phi = (r1['FS'].mean() - r0['FS'].mean()) / (r1['FA'].mean() - r0['FA'].mean())
        # 预紧力灵敏度
        FMw = np.full(15, FM); FMw[j] = FM * 0.4
        rw = fe.solve(w0, FMw)
        s_FM = (rw['eps'][fe.iy_center[j], fw].mean() * 1e6 - face0) / (-0.6 * FS) * 1e3
        # 脱粘可辨：找凹陷幅值
        deb = {}
        for d in (30, 50, 100, 200):
            rd = fe.solve(w1, FM, debond={j: float(d)})
            dd = (rd['eps'][fe.iy_center[j]] - r1['eps'][fe.iy_center[j]]) * 1e6
            m = r0['x_c'] <= 400
            deb[d] = dd[m].min()
        rows.append(dict(tag=tag, lam=1 / fe.lam_analytic, Phi=Phi, chi=r0['chi'].mean(),
                         face0=face0, sig=faceb - face0, s_FM=s_FM, deb=deb))
    a, b = rows
    def line(name, ka, kb, fmt='{:12.1f}'):
        print(f"  {name:38s} " + fmt.format(ka) + " " + fmt.format(kb) +
              f" {((kb-ka)/abs(ka)*100 if ka else 0):9.1f}%")
    line('传力长度 1/λ [mm]', a['lam'], b['lam'])
    line('螺栓载荷系数 Φ', a['Phi'], b['Phi'], '{:12.4f}')
    line('端面夹紧分配 χ', a['chi'], b['chi'], '{:12.4f}')
    line('安装态端面特征 [µε]', a['face0'], b['face0'])
    line('断柱端面信号 [µε]', a['sig'], b['sig'])
    line('预紧力灵敏度 [µε/kN]', a['s_FM'], b['s_FM'], '{:12.3f}')
    for d in (30, 50, 100, 200):
        line(f'脱粘 {d} mm 凹陷幅值 [µε]', a['deb'][d], b['deb'][d])
    print()
    print("  结论按以上数字判断（见报告 §2.4）。")


if __name__ == '__main__':
    part1()
    part2()
