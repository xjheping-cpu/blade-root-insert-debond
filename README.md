# blade-root-insert-debond

风电叶片叶根**预埋螺套（insert）界面脱粘**的有限元仿真，以及用分布式光纤在叶根内表面
检出该脱粘的正演模型。

> **English summary.** Finite-element simulation of adhesive debonding at the embedded
> insert interface in a wind turbine blade root, plus a forward model for detecting that
> debond with distributed fibre-optic strain sensing on the root inner surface.
> Code comments and documentation are in Chinese. The geometry is read from a JSON
> config; the bundled example is a self-defined representative geometry and does **not**
> correspond to any particular machine or manufacturer's drawing.

---

## 这个仓库解决什么问题

叶根靠一圈预埋螺套把叶片载荷传给轮毂。载荷经螺柱传入螺套，再经**螺套外表面那层胶**
传给周围玻璃钢层压。这层胶是整条传力路径上强度裕度最低的环节，也是最不可及的一处——
它埋在上百毫米厚的叶根壁内部，从叶片外表面和叶根舱里都无从直接观察。

在役的检查手段（复紧力矩、超声轴力抽检、法兰间隙、标记线）量的都是螺柱与法兰端面。
本仓库的计算回答一个可以纯由数值给出的问题：**给定这种连接构造，螺柱那头的读数
能在多大程度上反映界面的状态。**

结论不依赖界面强度与断裂能这两个尚未实测的材料参数——它们只决定起裂时刻与扩展速率，
不决定载荷如何在螺柱与端面承压之间分配。这一点由仓库内的敏感性分析给出。

## 内容

| 目录 | 内容 |
|---|---|
| `sim/` | 模型、求解与图件脚本。二维展开壳模型、轴对称分层内聚力模型、三维分层扇形块模型、寿命积分、光纤正演、检出判据，以及方法图与结果图的生成 |
| `config/` | 几何配置。`geometry_generic.json` 为随仓库发布的代表性算例 |
| `results/data/` | 计算结果 `.npz` |
| `results/figures/` | 图件 `.png` |

`sim/` 下的模块按扁平模块名彼此 import（`from root_model import ...`），
所以它们必须待在同一层，不要拆成包。

**不含**：商业推广材料、客户方案与其生成脚本，以及第三方版权图件。

## 两套模型

**M1　轴对称分层单胞。** 把螺套周围的五层构造显式分开：钢体、树脂富集过渡层、
玻纤缠绕层、拉挤块、叶根层压。三个界面全部用双线性牵引-分离内聚力单元建出，
采用 Benzeggagh–Kenane 混合模式判据，损伤不可逆；端面按单侧接触（主动集）处理，
载荷分级施加。用柔度法由该模型取界面的能量释放率 *G*(*a*)。

**M3　三维分层扇形块。** 覆盖数个螺套、含真实曲率，用于正演叶根内表面的应变场，
即光纤实际测得的那一面。与 M1 在界面剪切总量的峰值位置上交叉校核。

两者相互独立：一个给驱动力与寿命，一个给可观测量。

## 几何配置

几何不写死在代码里：

```bash
cd sim
python geometry.py                                    # 看当前生效的几何
BLADEROOT_GEOM=../config/my_machine.json python ...   # 换一套几何
```

选哪份配置，按 `BLADEROOT_GEOM` 环境变量 → `config/geometry_oem.json`（若存在）
→ `config/geometry_generic.json` 的次序。第二项是留给自有机型的，仓库里没有，
也被 `.gitignore` 排除，不会被误提交。

派生量（壁厚、节距、相邻螺套之间的净筋宽、*t/R*）一律由基本尺寸算出，
配置里不重复写死。加载时做自洽检查：筋宽为负、内孔大于外径、啮合长度超过埋深等
一律在求解之前报错。

`config/geometry_generic.json` 是**自拟的代表性参数**，用于让本仓库可复现，
不对应任何具体机型，也不取自任何整机厂图纸。选值时保留了这一类连接的关键特征：
相邻螺套钢体之间的筋宽很窄，不足壁厚的十二分之一。

换用自有机型时复制一份配置改数即可，求解器与图件脚本都不必动。

## 运行

```bash
pip install -r requirements.txt
cd sim
python verify.py          # 模型自检：解析对照、网格收敛、远场与梁理论对照
python insert_study.py    # 网格敏感性、界面牵引分布、能量释放率 G(a)
python insert_life.py     # Paris 律寿命分数积分
python sector3d.py        # 父模型 M2
python sector3d_layered.py  # 子模型 M3，耗时与内存最大
```

`results/` 下的存档即由上列脚本按 `config/geometry_generic.json` 算出，
全套约一小时。

`sector3d_layered.py` 是耗时与内存的大头，用 `pypardiso` 直接法求解，
峰值内存数十 GB 量级。其余脚本在普通工作站上分钟量级即可跑完。

## 许可

[Apache License 2.0](LICENSE)。

引用方式见 [`CITATION.cff`](CITATION.cff)。
