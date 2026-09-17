# -*- coding: utf-8 -*-
"""叶根连接的几何参数。

几何不写死在代码里，从 JSON 配置读入。字段的默认值在本模块 import 时装载，
所以 `Geom()`、`Geom(x_max=1400.0)` 这类既有写法一律照旧可用。

选哪一份配置，按以下次序：
    1. 环境变量 BLADEROOT_GEOM 指向的文件
    2. config/geometry_oem.json  （本机自有机型，不进版本库）
    3. config/geometry_generic.json  （随仓库发布的代表性算例）

    from geometry import Geom
    g = Geom()                                   # 默认配置
    g = Geom.load('config/geometry_generic.json')  # 指定一份
    g = Geom(x_max=1400.0)                       # 覆盖单项

派生量（壁厚、节距、筋宽、净截面积等）一律算出，配置里不重复写死，
免得改了一处忘了另一处。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = os.path.join(_HERE, 'config')
GENERIC_CONFIG = os.path.join(_CFG, 'geometry_generic.json')
OEM_CONFIG = os.path.join(_CFG, 'geometry_oem.json')


def default_config_path() -> str:
    """按环境变量 → 自有机型 → 通用算例的次序选配置。"""
    p = os.environ.get('BLADEROOT_GEOM')
    if p:
        return p
    return OEM_CONFIG if os.path.exists(OEM_CONFIG) else GENERIC_CONFIG


def _read(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return {
        'n_bolt': int(d['n_bolt']),
        'R_bc': float(d['D_bc']) / 2.0,
        'D_out': float(d['D_out']),
        'D_in': float(d['D_in']),
        'L_ins': float(d['L_ins']),
        'D_ins': float(d['D_ins']),
        'd_bore': float(d['d_bore']),
        'l_eng': float(d['l_eng']),
        'x_max': float(d.get('x_max', 900.0)),
        'x_baffle': float(d.get('x_baffle', 1.5 * d['L_ins'])),
        'name': d.get('name', os.path.basename(path)),
        'description': d.get('description', ''),
    }


_D = _read(default_config_path())


@dataclass(frozen=True)
class Geom:
    """叶根连接几何。长度一律 mm。默认值取自 default_config_path() 指向的配置。"""
    n_bolt: int = _D['n_bolt']        # 整环螺套数
    R_bc: float = _D['R_bc']          # 螺栓圆半径
    D_out: float = _D['D_out']        # 根端外径
    D_in: float = _D['D_in']          # 根端内径
    L_ins: float = _D['L_ins']        # 螺套埋深
    D_ins: float = _D['D_ins']        # 螺套外径
    d_bore: float = _D['d_bore']      # 螺套内孔（螺纹小径）
    l_eng: float = _D['l_eng']        # 螺纹啮合长度
    x_max: float = _D['x_max']        # 模型轴向长度，远场夹支
    x_baffle: float = _D['x_baffle']  # 根部挡板距端面的距离，仅作图示意
    name: str = _D['name']
    description: str = _D['description']

    # ---------------------------------------------------------------- 派生量
    @property
    def D_bc(self) -> float:
        return 2.0 * self.R_bc

    @property
    def t_wall(self) -> float:
        return (self.D_out - self.D_in) / 2.0

    @property
    def pitch(self) -> float:
        return 2.0 * np.pi * self.R_bc / self.n_bolt

    @property
    def ligament(self) -> float:
        """相邻螺套钢体之间的净筋宽。整条传力路径上最薄的一处。"""
        return self.pitch - self.D_ins

    @property
    def r_inner_off(self) -> float:
        """螺栓轴线到内表面（光纤面）的径向距离。"""
        return self.R_bc - self.D_in / 2.0

    @property
    def t_over_R(self) -> float:
        """壁厚与曲率半径之比。扇形块展平成直块的误差量级由它决定。"""
        return self.t_wall / self.R_bc

    # 螺套外围的分层构造：过渡层 0.5、缠绕层 6、拉挤块 7，合计 13.5 mm。
    # 这三项是工艺推断值，与机型无关，所以作为常量放在这里。
    LAYER_BUILDUP = 13.5

    @property
    def r_block_out(self) -> float:
        """拉挤块外半径。整个分层构造的最外缘，必须落在壁厚之内。"""
        return self.D_ins / 2.0 + self.LAYER_BUILDUP

    @property
    def A_steel(self) -> float:
        return np.pi / 4.0 * (self.D_ins ** 2 - self.d_bore ** 2)

    @property
    def A_lam_cell(self) -> float:
        """单个螺栓间距内、螺套段的层压净截面积。"""
        return self.pitch * self.t_wall - np.pi / 4.0 * self.D_ins ** 2

    # ---------------------------------------------------------------- 读取
    @classmethod
    def load(cls, path: str | None = None, **over) -> 'Geom':
        g = cls(**_read(path or default_config_path()))
        g.check()
        return replace(g, **over) if over else g

    def check(self) -> None:
        """几何自洽性。配置写错时尽早报出来，不要等到求解阶段。"""
        if self.D_out <= self.D_in:
            raise ValueError('D_out 必须大于 D_in')
        if self.ligament <= 0:
            raise ValueError('筋宽为负：螺套外径 %.1f 超过节距 %.2f'
                             % (self.D_ins, self.pitch))
        if self.d_bore >= self.D_ins:
            raise ValueError('内孔 %.1f 不小于螺套外径 %.1f'
                             % (self.d_bore, self.D_ins))
        if self.l_eng > self.L_ins:
            raise ValueError('啮合长度 %.1f 超过埋深 %.1f'
                             % (self.l_eng, self.L_ins))
        if not (self.D_in / 2.0 < self.R_bc < self.D_out / 2.0):
            raise ValueError('螺栓圆不在壁厚范围内')
        # 螺套连同外围分层必须整体落在壁厚之内。差这一条会让三维分层模型的
        # 网格退化，而症状要到单轴校验才暴露，且误差是 1e5 量级、看不出成因。
        blk = self.r_block_out
        inner, outer = self.r_inner_off, self.t_wall - self.r_inner_off
        if inner < blk or outer < blk:
            raise ValueError(
                '分层构造超出壁厚：拉挤块外半径 %.1f，'
                '而轴线到内表面 %.1f、到外表面 %.1f'
                % (blk, inner, outer))

    def summary(self) -> str:
        return (
            '%s（%s）\n'
            '  螺套 %d 个，埋深 %.0f mm，外径 %.0f mm，内孔 %.1f mm，啮合 %.0f mm\n'
            '  螺栓圆 Ø%.0f，根端 Ø%.0f / Ø%.0f，壁厚 %.1f mm\n'
            '  节距 %.2f mm，筋宽 %.2f mm（壁厚的 1/%.1f），t/R = %.3f'
            % (self.name, self.description, self.n_bolt, self.L_ins, self.D_ins,
               self.d_bore, self.l_eng, self.D_bc, self.D_out, self.D_in,
               self.t_wall, self.pitch, self.ligament,
               self.t_wall / self.ligament, self.t_over_R))


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print('配置：%s' % os.path.normpath(default_config_path()))
    print(Geom.load(sys.argv[1] if len(sys.argv) > 1 else None).summary())
