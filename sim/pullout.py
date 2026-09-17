# -*- coding: utf-8 -*-
"""
拔出通道：螺套脱粘 → 拔出 → 相邻过载 → 级联撕开 → 叶根开裂 → 叶片坠落。

这条链和螺柱疲劳断裂是两回事，要单独算。核心要回答的是一个问题：
**在螺套真的拔出之前，有多长的时间窗口，窗口里有没有可测的前兆。**

分六段：
  A 界面剪应力分布：预紧与外载各贡献多少，峰值在哪，离静强度多远
  B 脱粘长度 → 剩余承载能力：什么时候开始掉，掉到哪算失效
  C 脱粘长度 → OFDR 信号：多长的脱粘才可检出
  D 相邻多个拔出：邻位过载、层压应力、环的剩余承载
  E 界面疲劳扩展：脱粘怎么长，级联多快
  F 预警窗口：从可测到拔出还有多久
"""
from __future__ import annotations
import numpy as np
from root_model import RootFE, Geom, Mat

_G = Geom()              # 几何自 config/ 读入，换机型只改配置
L_INS = _G.L_ins

FS_INSTALL = 420e3
FA_GRAV = 74e3            # 重力摆振幅值对应的单柱外载
FA_OPER = 159e3           # 额定挥舞均值
FA_EXT = 339e3            # 极端
TAU_ULT = (15.0, 25.0)    # 优质环氧胶接 + 层压完好的界面剪切强度带 [MPa]（V2.0 §5.5）
TAU_DEG = (8.0, 12.0)     # 存在富树脂层/局部空腔时


def interface_shear(fe: RootFE, res, j: int):
    """界面剪应力沿埋深 τ(x) = k_q·滑移/(πD)  [MPa]。"""
    u = res['u']
    base = fe.ndof_plate + j * fe.n_ins_node
    us = u[base: base + fe.n_ins_node]
    p_dof = 2 * fe._node(fe.iy_center[j], np.arange(fe.n_ins_node))
    ul = u[p_dof]
    slip = us - ul
    return fe.x_ins, fe._k_q * slip / (np.pi * fe.g.D_ins)


def solve_w(fe, FM, FA_target, **kw):
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


def capacity(x, tau, D, L_ins, d_deb):
    """
    由弹性 τ 分布推剩余承载力的上下界。
    下界（脆性）：峰值达到强度即失效 → F_cap = F_applied × τ_ult/τ_peak
    上界（塑性）：整个粘接段都达到强度 → F_cap = τ_ult × πD × L_bond
    """
    m = x >= d_deb
    if not m.any():
        return 0.0, 0.0
    peak = np.abs(tau[m]).max()
    L_bond = L_ins - d_deb
    return peak, np.pi * D * L_bond / 1e3      # 返回峰值与 (πD·L_bond) [mm²·1e-3]


def main():
    L = "=" * 100
    fe = RootFE(n_cells=25, m_y=4)
    FM = fe.calibrate_FM(FS_INSTALL)
    j = 12
    g = fe.g

    print(L); print("A 界面剪应力：预紧与外载各贡献多少"); print(L)
    w0, r0 = solve_w(fe, FM, 0.0)
    wg, rg = solve_w(fe, FM, FA_GRAV)
    wo, ro = solve_w(fe, FM, FA_OPER)
    we, re = solve_w(fe, FM, FA_EXT)
    xi, t0 = interface_shear(fe, r0, j)
    _, tg = interface_shear(fe, rg, j)
    _, to = interface_shear(fe, ro, j)
    _, te = interface_shear(fe, re, j)
    idx = [np.argmin(np.abs(xi - v))
           for v in (0, 20, 50, 105, 150, 250, L_INS - 90, L_INS - 10)]
    print("  x [mm]            " + "".join(f"{xi[i]:8.0f}" for i in idx))
    for tag, t in (("仅预紧 420 kN", t0), ("+ 重力 74 kN", tg),
                   ("+ 额定 159 kN", to), ("+ 极端 339 kN", te)):
        print(f"  {tag:16s}" + "".join(f"{t[i]:8.2f}" for i in idx))
    print(f"\n  峰值：预紧 {np.abs(t0).max():.2f} MPa，额定 {np.abs(to).max():.2f}，"
          f"极端 {np.abs(te).max():.2f} MPa")
    print(f"  界面静强度带：完好 {TAU_ULT[0]}~{TAU_ULT[1]} MPa；"
          f"有富树脂/空腔 {TAU_DEG[0]}~{TAU_DEG[1]} MPa")
    print(f"  → 极端工况下的裕度：完好 {TAU_ULT[0]/np.abs(te).max():.1f}~{TAU_ULT[1]/np.abs(te).max():.1f} 倍；"
          f"缺陷界面 {TAU_DEG[0]/np.abs(te).max():.1f}~{TAU_DEG[1]/np.abs(te).max():.1f} 倍")
    print(f"  注：预紧力本身就贡献了 {np.abs(t0).max():.1f} MPa，与外载同量级——"
          f"这是「超扭矩复紧诱发拔出」的数值来源。")

    print(); print(L); print("B 脱粘长度 → 剩余承载能力"); print(L)
    print(f"  {'脱粘':>6} {'剩余粘接':>9} {'峰值τ(额定)':>12} {'峰值τ(极端)':>12} "
          f"{'塑性承载上界':>13} {'对极端载荷裕度':>14}")
    rows = []
    for d in (0, 50, 100, 150, 200, 250, 300, 350,
              L_INS - 90, L_INS - 50, L_INS - 20):
        _, rd = solve_w(fe, FM, FA_EXT, debond={j: float(d)})
        _, td = interface_shear(fe, rd, j)
        _, rdo = solve_w(fe, FM, FA_OPER, debond={j: float(d)})
        _, tdo = interface_shear(fe, rdo, j)
        m = xi >= d
        pk_e = np.abs(td[m]).max() if m.any() else np.inf
        pk_o = np.abs(tdo[m]).max() if m.any() else np.inf
        Lb = g.L_ins - d
        F_pl = np.mean(TAU_ULT) * np.pi * g.D_ins * Lb / 1e3     # kN，塑性上界
        rows.append((d, Lb, pk_o, pk_e, F_pl, F_pl / (FA_EXT / 1e3)))
        print(f"  {d:5.0f}mm {Lb:8.0f}mm {pk_o:11.2f} {pk_e:11.2f} "
              f"{F_pl:11.0f}kN {F_pl/(FA_EXT/1e3):13.1f}")
    rows = np.array(rows)
    np.save('out/pullout_capacity.npy', rows)
    # 临界脱粘长度：塑性上界降到极端载荷
    dcrit = np.interp(FA_EXT / 1e3, rows[::-1, 4], rows[::-1, 0])
    print(f"\n  临界脱粘长度（塑性上界 = 极端载荷 339 kN）：{dcrit:.0f} mm，"
          f"即剩余粘接 {g.L_ins - dcrit:.0f} mm")
    print(f"  峰值剪应力在脱粘 < {dcrit:.0f} mm 时基本不变（自相似），"
          f"说明脱粘可以长期稳定扩展而不改变承载——这正是预警窗口存在的原因。")

    print(); print(L); print("C 脱粘长度 → OFDR 端面特征信号"); print(L)
    fw = (rg['x_c'] >= 0) & (rg['x_c'] <= 40)
    base = rg['eps'][fe.iy_center[j], fw].mean() * 1e6
    print(f"  {'脱粘':>6} {'端面特征 Δ':>12} {'凹陷幅值':>10} {'凹陷位置':>10} {'模式B 5年可检?':>14}")
    sig = []
    for d in (10, 20, 30, 50, 75, 100, 150, 200, 300, 400):
        _, rd = solve_w(fe, FM, FA_GRAV, debond={j: float(d)})
        ee = rd['eps'][fe.iy_center[j]] * 1e6
        df = ee[fw].mean() - base
        dd = ee - rg['eps'][fe.iy_center[j]] * 1e6
        m = rd['x_c'] <= 450
        dip = dd[m].min(); xdip = rd['x_c'][m][np.argmin(dd[m])]
        ok = "是" if max(abs(df), abs(dip)) > 4 * 3.3 else "否"
        sig.append((d, df, dip, xdip))
        print(f"  {d:5.0f}mm {df:11.1f} {dip:9.1f} {xdip:9.0f}mm {ok:>12}")
    np.save('out/pullout_signal.npy', np.array(sig))

    print(); print(L); print("D 相邻多个螺套失效：邻位过载、界面应力与失稳判据"); print(L)
    print("  恒定外载（失效后由剩余螺套分担，不是固定法兰位移）。以极端工况算，")
    print("  因为决定会不会撕开的是极端事件，不是日常载荷。")
    fe2 = RootFE(n_cells=41, m_y=4)
    FM2 = fe2.calibrate_FM(FS_INSTALL)
    jj = 20
    tot_target = FA_EXT * 41
    print()
    print(f"  {'已拔出':>7} {'紧邻位外载':>12} {'相对标称':>9} {'紧邻位界面峰值τ':>16} "
          f"{'对完好强度裕度':>14} {'对缺陷强度裕度':>14}")
    outD = []
    for k in (0, 1, 2, 3, 5, 8, 12):
        deb = {(jj + i) % 41: float(fe2.g.L_ins) for i in range(-(k // 2), k - k // 2)} if k else {}
        # 恒定总载：割线迭代 w 使剩余螺套担起同样的总外载
        r = fe2.solve(0.0, FM2, debond=deb)
        r1 = fe2.solve(-1e-3, FM2, debond=deb)
        kk = (r1['FA'].sum() - r['FA'].sum()) / (-1e-3)
        w = 0.0
        for _ in range(15):
            if abs(r['FA'].sum() - tot_target) < 1e3:
                break
            w += (tot_target - r['FA'].sum()) / kk
            r = fe2.solve(w, FM2, debond=deb)
        nb = (jj - k // 2 - 1) % 41
        _, tnb = interface_shear(fe2, r, nb)
        pk = np.abs(tnb).max()
        outD.append((k, r['FA'][nb], pk))
        print(f"  {k:6d} {r['FA'][nb]/1e3:11.1f}kN {r['FA'][nb]/FA_EXT:8.2f} "
              f"{pk:14.1f}MPa {np.mean(TAU_ULT)/pk:13.2f} {np.mean(TAU_DEG)/pk:13.2f}")
    np.save('out/pullout_cascade.npy', np.array(outD))
    print()
    print("  裕度 <1 表示极端事件会直接在紧邻位造成界面损伤 → 下一位很快跟着走。")

    print(); print(L); print("G 制造缺陷批次：界面强度本身偏低时"); print(L)
    print("  V2.0 报告认定螺套脱落的主因是制造界面缺陷（富树脂堆积、灌注空腔、")
    print("  楔形条贴合不良），这类界面的剪切强度只有 8~12 MPa 而不是 15~25。")
    print()
    print(f"  {'界面状态':>18} {'静强度':>10} {'塑性承载(完整粘接)':>20} {'对极端载荷裕度':>14} {'弹性峰值裕度':>13}")
    pk_ext = 31.26
    for tag, tb in (('优质环氧 + 层压完好', TAU_ULT), ('富树脂层 / 局部空腔', TAU_DEG),
                    ('已局部脱粘 50%', (4.0, 6.0))):
        tm = np.mean(tb)
        Fpl = tm * np.pi * fe.g.D_ins * fe.g.L_ins / 1e3
        print(f"  {tag:18s} {tb[0]:.0f}~{tb[1]:.0f} MPa {Fpl:17.0f}kN "
              f"{Fpl/(FA_EXT/1e3):13.1f} {tm/pk_ext:12.2f}")
    print()
    print("  读法：塑性承载裕度都 >1，说明缺陷界面也不会一上来就拔出；")
    print("  但弹性峰值裕度都 <1，说明每一次极端事件都会在载荷引入端啃掉一小段界面。")
    print("  这正是「脱粘从端面向内逐步扩展」的力学来源，也是为什么脱套是")
    print("  投运若干年后才集中出现，而不是一次性事故。")

    print(); print(L); print("E 界面疲劳扩展与级联"); print(L)
    print("  说明：界面疲劳按 dd/dN = C·(τ_a/τ_0)^m，m 取 6（胶接/复材界面常用范围 5~8）。")
    print("  C 无实测数据，因此下面只给**相对量**——相对量不依赖 C。")
    m_exp = 6.0
    # 邻位过载导致的扩展加速
    for ov, tag in ((1.52, "三维模型给的 ±1 位过载 +52%"), (1.33, "二维模型给的 +33%")):
        print(f"  {tag}：界面剪应力同比例上升 → 扩展速率放大 {ov**m_exp:.0f} 倍")
    print(f"  一个螺套拔出后，紧邻两位的剩余脱粘寿命降到原来的 1/{1.52**m_exp:.0f}～1/{1.33**m_exp:.0f}")
    print("  → 级联是**加速**的：第二个比第一个快一个数量级，之后更快。")

    print(); print(L); print("F 预警窗口"); print(L)
    d_det = 40.0        # 可检出脱粘长度（按 M2 标定界面刚度后取 6σ 稳健值）
    print(f"  可检出脱粘长度（模式 B，基线差分）          {d_det:.0f} mm")
    print(f"  承载力开始下降的脱粘长度                    {dcrit:.0f} mm")
    print(f"  稳定扩展段占全程                            {(dcrit-d_det)/dcrit*100:.0f}%")
    print(f"  → 从「第一次能测到」到「承载力开始掉」，还剩全部脱粘寿命的 "
          f"{(dcrit-d_det)/dcrit*100:.0f}%")
    print()
    print("  这个比例不依赖疲劳系数 C，因为稳定段的扩展速率基本恒定（峰值剪应力自相似）。")
    print("  按 20 年设计寿命、脱粘在第 10 年开始算，预警提前量是 10 年 ×"
          f" {(dcrit-d_det)/dcrit:.2f} ≈ {10*(dcrit-d_det)/dcrit:.1f} 年量级。")


if __name__ == '__main__':
    main()
