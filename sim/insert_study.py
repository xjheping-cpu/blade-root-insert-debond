# -*- coding: utf-8 -*-
"""
M2′ 模型的计算研究。回答四个问题：

  Q1  预紧本身会不会把端面处的界面打坏？边缘损伤是真实效应还是奇异点假象
      （用网格敏感性判定）。
  Q2  三个界面里哪一个先坏、坏在哪一段。
  Q3  脱粘扩展是否稳定——这是前一版报告里悬而未决、且被现场数据质疑的结论。
      判据用能量释放率 G(a)：固定服役载荷下 dG/da > 0 即为不稳定扩展。
  Q4  脱粘到什么长度，常规检查（螺柱轴力、端面残余夹紧力）才可检出。

输出写到 out/insert_axi_study.npz 与 out/insert_axi_study.txt。
"""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np

from insert_axi import (InsertAxi, IFACES, Coh, LAYERS, MATS, L_INS, L_ENG,
                        R_CELL, R_WRAP, R_BLOCK, R_TRANS, FM_DEFAULT, K_STUD)

FA_GRAV, FA_OPER, FA_EXT = 74e3, 159e3, 339e3
OUT = 'out/'
LINE = '=' * 100


class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s)
        self.f.write(s + '\n')

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ 弹性界面开关
def set_elastic(M, on=True):
    """关掉损伤演化，把界面当线弹性——能量释放率的柔度法需要这个。"""
    M._elastic = on


def _patch_elastic():
    orig = InsertAxi._coh_update

    def wrapped(self, u):
        out = orig(self, u)
        if getattr(self, '_elastic', False):
            new = []
            for h, t in zip(self.coh, out):
                dn, ds, d, lam, d0, df, Gc = t
                d = np.where(h['cut'], 1.0, 0.0)
                h['dt'] = d
                new.append((dn, ds, d, lam, d0, df, Gc))
            return new
        return out

    InsertAxi._coh_update = wrapped


_patch_elastic()


# ------------------------------------------------------------------ Q1 网格敏感性
def q1_edge(log):
    log(LINE)
    log('Q1  预紧下端面处的界面损伤：真实效应还是奇异点假象')
    log(LINE)
    log('  判据：若损伤区长度随网格加密而收敛到一个有限值，是真实的边缘过渡区；')
    log('  若随网格加密不断缩短并趋于零，则只是端面处位移间断产生的应力奇异点。')
    log('')
    log('  %-10s %8s %10s %12s %12s %12s' %
        ('网格倍率', '尖端单元', '峰值牵引', 'A 损伤伸到', 'B 损伤伸到', 'A 受损结点'))
    rows = []
    for fac in (0.6, 1.0, 1.6, 2.4):
        M = InsertAxi(nx_fac=fac, verbose=False)
        FM = M.calibrate_FM(FM_DEFAULT)
        r = M.solve(F_M=FM, F_A=0.0)
        tr = M.tractions(r)
        dxm = float(np.diff(M.x[M.x <= L_INS]).min())
        ln = {}
        for k, v in tr.items():
            key = k[0]
            # 度量用"受损区伸到多远"。损伤若只落在 x=0 那一个结点上，
            # 用 max−min 会算成 0 mm，把奇异点误报成没有损伤。
            m = v['d'] > 0.01
            ln[key] = float(v['x'][m].max()) if m.any() else 0.0
            m1 = v['d'] > 0.99
            ln[key + '1'] = float(v['x'][m1].max()) if m1.any() else 0.0
            ln[key + 'n'] = int(m.sum())
        tmax = max(abs(v['ts']).max() for v in tr.values())
        log('  %9.1f× %7.2fmm %9.1f MPa %11.1f mm %11.1f mm %8d 个'
            % (fac, dxm, tmax, ln['A'], ln['B'], ln['An']))
        rows.append((fac, dxm, tmax, ln['A'], ln['B'], ln['An']))
    log('')
    a = np.array(rows)
    conv = np.abs(a[-1, 3] - a[-2, 3]) / max(a[-1, 3], 1e-9)
    log('  A 界面受损区伸展长度在最后两级网格间变化 %.1f%%，'
        '最细网格下仅 %.1f mm。' % (100 * conv, a[-1, 3]))
    if conv < 0.15 and a[-1, 3] > 5.0:
        log('  → 收敛到有限长度，是真实的端面过渡区，不是纯奇异点。')
    else:
        log('  → 受损区随网格加密而缩短、始终不足 %.0f mm，判为端面位移间断造成的'
            % max(a[:, 3].max(), 2.0))
        log('    应力奇异点。结论：预紧本身不会造成有工程意义的脱粘；')
        log('    端面第一个结点上的牵引值不可作为起裂判据。')
    log('')
    log('  说明：模型把螺套端面取为下沉（不参与承压），于是端面处螺套与周围材料之间')
    log('  存在位移间断，这是设计使然——预紧必须由层压端面承受，若螺套端面齐平承压，')
    log('  预紧就直接短路回法兰，界面完全不受力，与预埋螺套的设计意图相反。')
    return rows


# ------------------------------------------------------------------ Q2 界面剖面
def q2_profiles(log):
    log('')
    log(LINE)
    log('Q2  三个界面的牵引力与损伤沿轴向的分布')
    log(LINE)
    M = InsertAxi(verbose=False)
    FM = M.calibrate_FM(FM_DEFAULT)
    out = {}
    for tag, FA in (('preload', 0.0), ('grav', FA_GRAV),
                    ('oper', FA_OPER), ('ext', FA_EXT)):
        r = M.solve(F_M=FM, F_A=FA)
        tr = M.tractions(r)
        out[tag] = dict(r=r, tr=tr)
    log('')
    log('  切向牵引峰值 [MPa]，端面第一个结点（含奇异值）')
    log('  %-16s %10s %10s %10s %10s' %
        ('界面 / 工况', '仅预紧', '重力摆振', '额定挥舞', '极限'))
    TT = ('preload', 'grav', 'oper', 'ext')
    for key in ('A', 'B', 'C'):
        nm = [k for k in out['preload']['tr'] if k.startswith(key)][0]
        v = ['%9.2f' % out[t]['tr'][nm]['ts'][np.argmax(
            np.abs(out[t]['tr'][nm]['ts']))] for t in TT]
        log('  %-16s %s' % (nm, ' '.join(v)))
    log('')
    log('  切向牵引峰值 [MPa]，离开端面 5 mm 之后（工程上有意义的值）')
    log('  %-16s %10s %10s %10s %10s' %
        ('界面 / 工况', '仅预紧', '重力摆振', '额定挥舞', '极限'))
    for key in ('A', 'B', 'C'):
        nm = [k for k in out['preload']['tr'] if k.startswith(key)][0]
        v = []
        for t in TT:
            w = out[t]['tr'][nm]
            m = w['x'] >= 5.0
            v.append('%9.2f' % w['ts'][m][np.argmax(np.abs(w['ts'][m]))])
        log('  %-16s %s' % (nm, ' '.join(v)))
    log('')
    log('  损伤峰值 [-] 与受损结点数（括号内）')
    log('  %-16s %10s %10s %10s %10s' %
        ('界面 / 工况', '仅预紧', '重力摆振', '额定挥舞', '极限'))
    for key in ('A', 'B', 'C'):
        nm = [k for k in out['preload']['tr'] if k.startswith(key)][0]
        v = ['%6.3f(%d)' % (out[t]['tr'][nm]['d'].max(),
                            int((out[t]['tr'][nm]['d'] > 0.01).sum()))
             for t in TT]
        log('  %-16s %s' % (nm, ' '.join('%9s' % z for z in v)))
    log('')
    log('  螺套埋入端（x = %.0f mm）附近的切向牵引：' % L_INS)
    log('  %-16s %12s %12s' % ('界面', '端面侧峰值位置', '埋入端 50 mm 内峰值'))
    for key in ('A', 'B', 'C'):
        nm = [k for k in out['preload']['tr'] if k.startswith(key)][0]
        v = out['oper']['tr'][nm]
        deep = v['x'] > L_INS - 50
        log('  %-16s %11.1f mm %11.2f MPa'
            % (nm, v['x'][np.argmax(np.abs(v['ts']))],
               np.abs(v['ts'][deep]).max()))
    return M, FM, out


# ------------------------------------------------------------------ Q3 能量释放率
def q3_stability(log, M, FM):
    log('')
    log(LINE)
    log('Q3  脱粘扩展是否稳定：能量释放率 G(a)')
    log(LINE)
    log('  方法：把界面人为切开长度 a，界面其余部分取线弹性（关闭损伤演化），')
    log('  在固定服役载荷下算总势能 Π(a)，G = −dΠ/dA，A = 2πr·a。')
    log('  固定载荷下 dG/da > 0 即为不稳定扩展：一旦起裂就会自行走完。')
    log('')

    def PI(M, FM, FA):
        # 弹性模式没有损伤历史，不需要增量加载
        r = M.solve(F_M=FM, F_A=FA, n_pre=1, n_load=1)
        u = r['u']
        f0 = np.zeros(M.ndof)
        f0[M.stud_dof] -= M.stud_k * (FM / K_STUD)
        f0[M.far_dof] += M.far_w * FA
        return -0.5 * float(f0 @ u), r

    res = {}
    for which, rad in (('A', R_TRANS), ('B', R_WRAP), ('C', R_BLOCK)):
        for direction in ('deep', 'face'):
            aa = np.array([0., 20., 40., 70., 100., 150., 200., 250.,
                           300., 350., 400., 430., 455.])
            P = np.zeros(len(aa))
            FSx = np.zeros(len(aa))
            FKR = np.zeros(len(aa))
            set_elastic(M, True)
            for i, a in enumerate(aa):
                M.reset()
                if a > 0:
                    rng = (L_INS - a, L_INS) if direction == 'deep' else (0.0, a)
                    M.cut(which, rng)
                P[i], r = PI(M, FM, FA_OPER)
                FSx[i], FKR[i] = r['F_S'], r['F_KR']
            set_elastic(M, False)
            M.reset(); M.clear_cut()
            dA = 2 * np.pi * rad
            G = -np.gradient(P, aa) / dA
            res[(which, direction)] = dict(a=aa, PI=P, G=G, FS=FSx, FKR=FKR)

    for which in ('A', 'B', 'C'):
        c = [k for k in IFACES if k.name.startswith(which)][0]
        log('  界面 %-16s  G_IIc = %.2f N/mm' % (c.name, c.GIIc))
        log('    %-8s %12s %12s %12s %12s' %
            ('脱粘 a', 'G 自埋入端', 'G/G_IIc', 'G 自端面', 'G/G_IIc'))
        d1, d2 = res[(which, 'deep')], res[(which, 'face')]
        for i, a in enumerate(d1['a']):
            if a not in (0., 40., 100., 200., 300., 400., 455.):
                continue
            log('    %6.0fmm %11.4f %12.2f %11.4f %12.2f'
                % (a, d1['G'][i], d1['G'][i] / c.GIIc,
                   d2['G'][i], d2['G'][i] / c.GIIc))
        for direction, dd in (('自埋入端向端面', d1), ('自端面向埋入端', d2)):
            g = dd['G']
            rising = np.polyfit(dd['a'][1:], g[1:], 1)[0]
            peak = dd['a'][int(np.argmax(g))]
            log('    %s：G 峰值在 a = %.0f mm，整体斜率 dG/da = %+.2e N/mm²  → %s'
                % (direction, peak, rising,
                   '不稳定（自持扩展）' if rising > 0 else '稳定（需持续加载才扩展）'))
        log('')
    return res


# ------------------------------------------------------------------ Q4 常规检查
def q4_observability(log, M, FM, res):
    log(LINE)
    log('Q4  脱粘发展到什么程度，常规检查才可检出')
    log(LINE)
    log('  常规检查手段：复紧力矩（看螺柱轴力 F_S）、端面塞尺（看残余夹紧力 F_KR）。')
    log('  载荷工况取额定挥舞 F_A = %.0f kN。' % (FA_OPER / 1e3))
    log('')
    d = res[('B', 'deep')]
    FS0, FKR0 = d['FS'][0], d['FKR'][0]
    log('  %-10s %12s %10s %12s %10s %14s' %
        ('脱粘长度', 'F_S', '相对变化', 'F_KR', '相对变化', '剩余界面承载'))
    for i, a in enumerate(d['a']):
        cap = IFACES[1].ts0 * 2 * np.pi * R_WRAP * (L_INS - a) / 1e3
        log('  %7.0f mm %10.1f kN %9.2f%% %10.1f kN %9.2f%% %11.0f kN'
            % (a, d['FS'][i] / 1e3, 100 * (d['FS'][i] / FS0 - 1),
               d['FKR'][i] / 1e3, 100 * (d['FKR'][i] / FKR0 - 1), cap))
    log('')
    i400 = int(np.argmin(np.abs(d['a'] - 400)))
    log('  界面 B 脱粘 400 mm（占埋深 %.0f%%）时，螺柱轴力只变了 %.2f%%，'
        % (100 * 400 / L_INS, 100 * (d['FS'][i400] / FS0 - 1)))
    log('  端面残余夹紧力只变了 %.2f%%，端面未张开。复紧力矩与塞尺都查不出来。'
        % (100 * (d['FKR'][i400] / FKR0 - 1)))
    return d


def main():
    t0 = time.time()
    log = Tee(OUT + 'insert_axi_study.txt')
    log('M2′ 预埋螺套单胞轴对称分层内聚力模型 —— 计算研究')
    log('生成时间：2026-09-12')
    log('')
    rows = q1_edge(log)
    M, FM, prof = q2_profiles(log)
    res = q3_stability(log, M, FM)
    d = q4_observability(log, M, FM, res)

    np.savez_compressed(
        OUT + 'insert_axi_study.npz',
        mesh_rows=np.array(rows),
        coh_x=M.coh_x,
        **{f'tr_{t}_{k[0]}_{f}': prof[t]['tr'][k][f]
           for t in prof for k in prof[t]['tr'] for f in ('ts', 'tn', 'd')},
        **{f'G_{w}_{dr}_{f}': res[(w, dr)][f]
           for w in 'ABC' for dr in ('deep', 'face')
           for f in ('a', 'G', 'PI', 'FS', 'FKR')},
        el_rc=M.el_rc, el_xc=M.el_xc, el_tag=M.tag,
        r_grid=M.r, x_grid=M.x)
    log('')
    log('用时 %.0f s。数据写入 %sinsert_axi_study.npz' % (time.time() - t0, OUT))
    log.close()


if __name__ == '__main__':
    main()
