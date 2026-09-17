# -*- coding: utf-8 -*-
"""
M2′ 模型对【推断】参数的敏感性。

模型里有一批参数没有实测支撑，只能取文献量级值。报告要用模型的结论，
就必须说清楚这些参数取错了会怎样。本脚本逐个扰动，看四个结论量怎么动：

  Φ            载荷系数（决定端面张开载荷）
  τ_face       端面处界面 B 的切向牵引峰值（决定会不会起裂）
  a_50         剩余寿命降到一半时的脱粘长度（决定预警阈值设在哪）
  ΔF_S@400     脱粘 400 mm 时螺柱轴力的相对变化（决定常规检查能否发现）

另外单独查一个几何冲突：缠绕层的名义外径大于螺栓节距，
相邻螺套的缠绕层不可能同时成立。取缠绕层外径等于节距（减薄）重算。
"""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

import insert_axi as IA
import insert_study  # noqa: F401
from insert_study import Tee, set_elastic
from insert_life import sweep, life_fraction
from insert_axi import (InsertAxi, Coh, FM_DEFAULT, L_INS, K_STUD,
                        P_PITCH)

FA_OPER = 159e3
LINE = '=' * 104


def one_case(name, *, r_wrap=None, scale_ts=1.0, scale_G=1.0, k_stud=None,
             note=''):
    """跑一个参数组合，返回四个结论量。"""
    r_wrap0, kstud0 = IA.R_WRAP, IA.K_STUD
    if r_wrap is not None:
        IA.R_WRAP = r_wrap
    if k_stud is not None:
        IA.K_STUD = k_stud
    ifs = tuple(Coh(c.name, IA.R_WRAP if c.name.startswith('B') else c.r,
                    c.K, c.tn0 * scale_ts, c.ts0 * scale_ts,
                    c.GIc * scale_G, c.GIIc * scale_G, c.eta)
                for c in IA.IFACES)
    try:
        M = InsertAxi(ifaces=ifs, verbose=False)
        FM = M.calibrate_FM(FM_DEFAULT)
        r0 = M.solve(F_M=FM, F_A=0.0)
        r1 = M.solve(F_M=FM, F_A=FA_OPER)
        phi = (r1['F_S'] - r0['F_S']) / FA_OPER
        v = M.tractions(r0)['B 缠绕层/拉挤块']
        tau = float(np.abs(v['ts'][v['x'] <= 20]).max())
        a_end = L_INS - 20.0        # 与 insert_life.py 同一口径
        aa = np.round(np.arange(0, a_end + 6.0, 12.5), 3)
        d = sweep(M, FM, 'B', FA_OPER, aa)
        fr, _ = life_fraction(aa, d['G'], 6, a_end)
        a50 = float(np.interp(0.50, fr[::-1], aa[::-1]))
        i4 = int(np.argmin(np.abs(aa - 400)))
        dFS = 100 * (d['FS'][i4] / d['FS'][0] - 1)
        out = (name, phi, tau, a50, dFS, note)
    finally:
        IA.R_WRAP, IA.K_STUD = r_wrap0, kstud0
    return out


def main():
    t0 = time.time()
    log = Tee('out/insert_sens.txt')
    log('M2′ 模型对【推断】参数的敏感性')
    log('生成时间：2026-09-12')
    log('')
    log(LINE)
    log('%-34s %8s %12s %10s %14s' %
        ('参数组合', 'Φ', 'τ_face [MPa]', 'a_50 [mm]', 'ΔF_S@400mm'))
    log(LINE)

    rows = []
    cases = [
        dict(name='基准', note='缠绕层 6.0 mm，界面 B 强度 20 MPa'),
        dict(name='缠绕层减薄至 Ø%.1f（几何自洽）' % P_PITCH,
             r_wrap=P_PITCH / 2,
             note='相邻螺套缠绕层不再重叠'),
        dict(name='界面强度 −50%', scale_ts=0.5),
        dict(name='界面强度 +50%', scale_ts=1.5),
        dict(name='断裂能 −50%', scale_G=0.5),
        dict(name='断裂能 +50%', scale_G=1.5),
        dict(name='螺柱刚度 −30%', k_stud=K_STUD * 0.7,
             note='夹紧长度取长'),
        dict(name='螺柱刚度 +30%', k_stud=K_STUD * 1.3),
    ]
    for c in cases:
        try:
            r = one_case(**c)
            rows.append(r)
            log('%-34s %8.4f %12.2f %10.0f %13.2f%%   %s'
                % (r[0], r[1], r[2], r[3], r[4], r[5]))
        except Exception as e:
            log('%-34s  失败：%s' % (c['name'], e))

    log(LINE)
    base = rows[0]
    log('')
    log('相对基准的变化幅度：')
    log('%-34s %10s %10s %10s %10s' %
        ('参数组合', 'ΔΦ', 'Δτ_face', 'Δa_50', 'ΔΔF_S'))
    for r in rows[1:]:
        log('%-34s %9.1f%% %9.1f%% %9.1f%% %9.1f%%'
            % (r[0], 100 * (r[1] / base[1] - 1), 100 * (r[2] / base[2] - 1),
               100 * (r[3] / base[3] - 1),
               100 * (r[4] / base[4] - 1) if base[4] else float('nan')))
    log('')
    log('读法：')
    log('  Φ 与 τ_face 只由几何与弹性常数决定，与界面强度、断裂能无关——')
    log('  这两个量是"线弹性结论"，可信度最高。')
    log('  a_50 由 G(a) 的形状决定，而 G(a) 也是线弹性量，所以预警阈值这个结论')
    log('  同样不依赖强度与断裂能的取值，只依赖几何与模量。')
    log('  界面强度与断裂能影响的是"什么时候起裂、多久走完"的绝对时标，')
    log('  那正是本模型不回答、必须由子部件试验回答的部分。')
    log('')
    log('用时 %.0f s' % (time.time() - t0))
    log.close()


if __name__ == '__main__':
    main()
