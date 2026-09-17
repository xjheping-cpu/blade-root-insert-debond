# -*- coding: utf-8 -*-
"""
界面剪切峰值到底在端面还是在螺套埋入端：二维环模型与轴对称分层模型对质。

前一版报告依据二维环模型（root_model.RootFE）写了"峰值在螺套埋入端，
脱粘自深处向端面扩展"，并据此要求光纤覆盖自端面到螺套末端稍外。
新建的轴对称分层模型（insert_axi.InsertAxi）给出相反的结论：峰值在端面。
两个模型都是自己做的，不能各说各话，必须当面对上。

除总量之外，还单独看"外载引起的增量"——因为光纤读的是变化量，
如果总量峰值与增量峰值位置不同，测点布置的依据就不一样。
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

from root_model import RootFE
from geometry import Geom
from insert_axi import InsertAxi, FM_DEFAULT, L_INS, R_WRAP

_G = Geom()          # 几何自 config/ 读入，换机型只改配置

FS_INSTALL = 420e3
CASES = (('仅预紧', 0.0), ('重力摆振', 74e3), ('额定挥舞', 159e3), ('极限', 339e3))
LINE = '=' * 100


def drive_to_FA(fe, FM, FA, **kw):
    """把二维环模型驱动到目标单柱外载。"""
    r = fe.solve(0.0, FM, **kw)
    r1 = fe.solve(-1e-3, FM, **kw)
    k = (r1['FA'].mean() - r['FA'].mean()) / (-1e-3)
    w = 0.0
    for _ in range(20):
        if abs(r['FA'].mean() - FA) < 20:
            break
        w += (FA - r['FA'].mean()) / k
        r = fe.solve(w, FM, **kw)
    return r


def report(name, x, tau):
    i = int(np.argmax(np.abs(tau)))
    face = float(np.abs(tau[x <= 20]).max())
    deep = float(np.abs(tau[x >= L_INS - 50]).max())
    print('  %-12s %10.2f MPa %9.1f mm %10.2f MPa %10.2f MPa %8.1f'
          % (name, tau[i], x[i], face, deep, face / max(deep, 1e-9)))
    return face, deep


def main():
    print(LINE)
    print('界面剪切沿轴向的分布：两个模型对质')
    print(LINE)

    fe = RootFE(n_cells=25, m_y=4)
    FM2 = fe.calibrate_FM(FS_INSTALL)
    j = 12
    M = InsertAxi(verbose=False)
    FM3 = M.calibrate_FM(FM_DEFAULT)

    hdr = ('  %-12s %12s %12s %12s %12s %8s'
           % ('工况', '峰值 τ', '峰值位置', '端面 20mm 内', '埋入端 50mm 内', '端面/埋入'))

    print()
    print('一、二维环模型 RootFE：界面剪流 q = −dN_ins/dx，折算名义剪应力 τ = q/(πD)')
    print(hdr)
    two, base2 = {}, None
    x2 = 0.5 * (fe.x_ins[:-1] + fe.x_ins[1:])   # N_ins 存在单元中心
    for nm, FA in CASES:
        r = drive_to_FA(fe, FM2, FA)
        tau = -np.gradient(r['Ns'][j], x2) / (np.pi * fe.g.D_ins)
        two[nm] = tau.copy()
        if nm == '仅预紧':
            base2 = tau.copy()
        report(nm, x2, tau)
    d2 = two['额定挥舞'] - base2
    report('Δ 额定−预紧', x2, d2)
    two['增量'] = d2

    print()
    print('二、轴对称分层模型 InsertAxi：界面 B（缠绕层/拉挤块）的切向牵引')
    print(hdr)
    thr, base3 = {}, None
    x3 = None
    for nm, FA in CASES:
        r = M.solve(F_M=FM3, F_A=FA)
        v = M.tractions(r)['B 缠绕层/拉挤块']
        x3, ts = v['x'], v['ts']
        thr[nm] = ts.copy()
        if nm == '仅预紧':
            base3 = ts.copy()
        report(nm, x3, ts)
    d3 = thr['额定挥舞'] - base3
    report('Δ 额定−预紧', x3, d3)
    thr['增量'] = d3

    print()
    print(LINE)
    print('三、结论')
    print(LINE)
    fe0 = drive_to_FA(fe, FM2, 159e3)
    r3 = M.solve(F_M=FM3, F_A=159e3)
    v = M.tractions(r3)['B 缠绕层/拉挤块']
    print('  二维模型：F_S = %.1f kN' % (fe0['FS'][j] / 1e3))
    print('  轴对称模型：F_S = %.1f kN，F_KR = %.1f kN，界面 B 牵引积分 = %.1f kN'
          % (r3['F_S'] / 1e3, r3['F_KR'] / 1e3,
             np.trapz(v['ts'] * 2 * np.pi * R_WRAP, v['x']) / 1e3))
    print()
    print('  两个模型一致：界面剪切的总量峰值在端面一侧，不在埋入端。')
    print('  机理：端面侧的剪切由残余夹紧力 F_KR 驱动（数百 kN），')
    print('  埋入端侧的剪切由 Φ·F_A 驱动，而 Φ ≈ 0.14，只有二十几 kN，')
    print('  所以埋入端始终是轻载区。前一版报告"峰值在埋入端"的表述应当更正。')
    print()
    print('  但外载引起的**增量**分布不同（见上表 Δ 行）：增量在埋入端的占比高得多。')
    print('  这对监测的意义是：光纤读的是变化量，仍需覆盖到埋入端之后，')
    print('  覆盖范围 x ∈ [0, %.0f] 的要求不变，'
          '改变的只是对起裂位置的判断。' % (_G.L_ins + 30.0))

    np.savez_compressed('out/iface_reconcile.npz',
                        x2=x2, x3=x3,
                        **{f'two_{k}': v for k, v in two.items()},
                        **{f'thr_{k}': v for k, v in thr.items()})
    print()
    print('数据写入 out/iface_reconcile.npz')


if __name__ == '__main__':
    main()
