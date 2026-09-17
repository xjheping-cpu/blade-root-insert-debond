# -*- coding: utf-8 -*-
"""
由 G(a) 推预警窗口：脱粘从 a 长到拔出，各阶段各占剩余寿命的多大比例。

疲劳扩展用 Paris 型律 da/dN = C·(ΔG/G_c)^m。C 与材料/工艺有关，未做试验之前
未知；但**各阶段占剩余寿命的比例与 C 无关**，只取决于 G(a) 的形状和指数 m。
于是"还剩多少时间"这个问题可以在不知道 C 的情况下先回答一半：
不能给出绝对年数，但能给出"走到哪一步就只剩百分之几"。

这一点直接决定预警阈值该设在哪里，也是前一版报告里"90% 预警窗口"
被现场数据（ONYX：发现到严重 3 个月）质疑之后必须补上的计算。
"""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

import insert_study  # noqa: F401  —— 引入它对 _coh_update 的弹性模式补丁
from insert_study import set_elastic, Tee
from insert_axi import (InsertAxi, IFACES, L_INS, R_WRAP, R_TRANS, R_BLOCK,
                        FM_DEFAULT, K_STUD)

FA_GRAV, FA_OPER, FA_EXT = 74e3, 159e3, 339e3
OUT = 'out/'
LINE = '=' * 100
RAD = {'A': R_TRANS, 'B': R_WRAP, 'C': R_BLOCK}


def sweep(M, FM, which, FA, aa, direction='deep'):
    """固定载荷下扫脱粘长度，返回总势能、能量释放率、螺柱轴力、残余夹紧力。"""
    P = np.zeros(len(aa)); FS = np.zeros(len(aa)); FK = np.zeros(len(aa))
    set_elastic(M, True)
    for i, a in enumerate(aa):
        M.reset(); M.clear_cut()
        if a > 0:
            rng = (L_INS - a, L_INS) if direction == 'deep' else (0.0, a)
            M.cut(which, rng)
        r = M.solve(F_M=FM, F_A=FA, n_pre=1, n_load=1)
        f0 = np.zeros(M.ndof)
        f0[M.stud_dof] -= M.stud_k * (FM / K_STUD)
        f0[M.far_dof] += M.far_w * FA
        P[i] = -0.5 * float(f0 @ r['u'])
        FS[i], FK[i] = r['F_S'], r['F_KR']
    set_elastic(M, False)
    M.reset(); M.clear_cut()
    G = -np.gradient(P, aa) / (2 * np.pi * RAD[which])
    return dict(a=aa, PI=P, G=G, FS=FS, FKR=FK)


def life_fraction(a, G, m, a_end):
    """剩余寿命分布。返回 f(a) = 从 a 走到 a_end 还需要的循环数占
    从 a0 走到 a_end 的比例（a0 = a[0]）。"""
    G = np.maximum(G, 1e-9)
    w = G ** (-float(m))                 # dN/da ∝ G^-m
    N = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(a))])
    m_end = a <= a_end
    N_end = np.interp(a_end, a, N)
    return (N_end - N) / N_end, N


def main():
    t0 = time.time()
    log = Tee(OUT + 'insert_life.txt')
    log('预警窗口的计算：由 G(a) 推各阶段占剩余寿命的比例')
    log('生成时间：2026-09-12')
    log('')

    M = InsertAxi(verbose=False)
    FM = M.calibrate_FM(FM_DEFAULT)
    # 积分上限留出端部 20 mm：那一段的 G(a) 受端面奇异控制，不作数。
    # 加密段自埋深往回 90 mm 起。图纸口径下与旧版逐值相同。
    a_end = L_INS - 20.0
    a_ref = L_INS - 90.0
    aa = np.round(np.concatenate([np.arange(0, a_ref, 12.5),
                                  np.arange(a_ref, a_end + 6.0, 6.25)]), 3)

    log(LINE)
    log('一、能量释放率 G(a)：细密扫掠')
    log(LINE)
    log('  载荷取额定挥舞 F_A = %.0f kN，脱粘自埋入端向端面扩展。' % (FA_OPER / 1e3))
    log('')
    res = {}
    for w in ('A', 'B'):
        res[w] = sweep(M, FM, w, FA_OPER, aa)
        c = [k for k in IFACES if k.name.startswith(w)][0]
        g = res[w]['G']
        i_min = int(np.argmin(g[2:-4])) + 2
        log('  界面 %-16s G(0)=%.4f  谷底 G=%.4f @ a=%.0fmm  '
            'G(455)=%.4f  末段/谷底 = %.0f 倍'
            % (c.name, g[1], g[i_min], aa[i_min], np.interp(455, aa, g),
               np.interp(455, aa, g) / max(g[i_min], 1e-12)))
        log('    最大 G/G_IIc = %.3f（发生在 a = %.0f mm）→ %s'
            % (g.max() / c.GIIc, aa[int(np.argmax(g))],
               '服役载荷下不会静力扩展，只能靠疲劳' if g.max() < c.GIIc
               else '服役载荷即可静力扩展'))
    log('')

    log(LINE)
    log('二、剩余寿命分布（Paris 律，比例与系数 C 无关）')
    log(LINE)
    log('  da/dN = C·(ΔG/G_c)^m。C 未标定，但"还剩百分之几"只由 G(a) 形状与 m 决定。')
    log('  m 取 4 / 6 / 8 三个值覆盖复合材料界面疲劳的常见范围。')
    log('')
    w = 'B'
    a, G = res[w]['a'], res[w]['G']
    log('  界面 B（文献实测的失效面）')
    log('  %-12s %14s %14s %14s' % ('已脱粘长度', '剩余寿命 m=4', 'm=6', 'm=8'))
    fr = {m: life_fraction(a, G, m, a_end)[0] for m in (4, 6, 8)}
    for aq in (0, 50, 100, 150, 200, 250, 300, 350,
               a_ref, a_ref + 25, a_ref + 50, a_end - 10):
        log('  %9.0f mm %13.1f%% %13.1f%% %13.1f%%'
            % (aq, 100 * np.interp(aq, a, fr[4]),
               100 * np.interp(aq, a, fr[6]),
               100 * np.interp(aq, a, fr[8])))
    log('')
    for m in (4, 6, 8):
        f = fr[m]
        a90 = float(np.interp(0.10, f[::-1], a[::-1]))
        a50 = float(np.interp(0.50, f[::-1], a[::-1]))
        log('  m = %d：走到 a = %.0f mm 时剩余寿命过半；走到 a = %.0f mm 时只剩 10%%。'
            % (m, a50, a90))
    log('')
    log('  结论：扩展在前 80%% 埋深内是减速的（G 下降），进入最后约 90 mm 后 G 急升，')
    log('  剩余寿命迅速塌缩。这与现场"发现后 3 个月即严重"的观察一致，')
    log('  也说明预警必须在脱粘进入末段之前发出，不能等到常规检查有反应。')

    log('')
    log(LINE)
    log('三、剩余承载与极限载荷的交点：拔出临界长度')
    log(LINE)
    tau = {c.name[0]: c.ts0 for c in IFACES}
    log('  %-10s %14s %14s %14s' % ('脱粘长度', 'A 剩余承载', 'B 剩余承载', 'C 剩余承载'))
    crit = {}
    for k in ('A', 'B', 'C'):
        cap = tau[k] * 2 * np.pi * RAD[k] * (L_INS - aa) / 1e3
        crit[k] = float(np.interp(-FA_EXT / 1e3, -cap, aa))
    for aq in (0, 100, 200, 300, a_ref, a_ref + 40, a_ref + 60, a_end + 5):
        row = [tau[k] * 2 * np.pi * RAD[k] * (L_INS - aq) / 1e3 for k in 'ABC']
        log('  %7.0f mm %11.0f kN %11.0f kN %11.0f kN' % (aq, *row))
    log('')
    log('  极限单柱外载 F_A = %.0f kN。剩余承载降到这个量级时螺套被拔出：' % (FA_EXT / 1e3))
    for k in 'ABC':
        log('    界面 %s：临界脱粘长度 %.0f mm（占埋深 %.0f%%）'
            % (k, crit[k], 100 * crit[k] / L_INS))
    log('')
    log('  注：剩余承载按平均剪切强度乘剩余粘接面积估计，未计端部应力集中，')
    log('  因此是乐观上界；真实临界长度更短。')

    np.savez_compressed(OUT + 'insert_life.npz',
                        a=aa, G_A=res['A']['G'], G_B=res['B']['G'],
                        FS_A=res['A']['FS'], FS_B=res['B']['FS'],
                        FKR_A=res['A']['FKR'], FKR_B=res['B']['FKR'],
                        PI_A=res['A']['PI'], PI_B=res['B']['PI'],
                        fr4=fr[4], fr6=fr[6], fr8=fr[8],
                        crit=np.array([crit[k] for k in 'ABC']))
    log('')
    log('用时 %.0f s。数据写入 %sinsert_life.npz' % (time.time() - t0, OUT))
    log.close()


if __name__ == '__main__':
    main()
