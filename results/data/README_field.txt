sector_layered.npz —— 叶根分层螺套连接三维场数据
================================================================================
由 sector3d_layered.py 生成（sector3d.Sector3D 的分层子类）。

1. 网格
--------------------------------------------------------------------------------
  x  (nx_n,)  轴向结点坐标 mm，0 = 叶根端面/法兰面
  y  (ny,)    周向展开结点坐标 mm，周期，节距 86.95 mm
  r  (nr_n,)  径向结点坐标 mm，0 = 叶根内表面（光纤位置），112 = 外表面
  xc (nx_e,)  yc (ny,)  rc (nr_e,)   对应的单元形心坐标
  注意 y 方向结点数 = 单元数 = ny（周期，最后一列单元绕回 y=0）。

  结点编号：nid = (iy * nx_n + ix) * nr_n + ir     (ix∈[0,nx_n), iy∈[0,ny), ir∈[0,nr_n))
  单元编号：e   = (ix * ny + iy) * nr_e + ir
  所以 <case>_u[nid] 就是结点 (ix,iy,ir) 的位移 (u_x, u_y, u_r)。

2. 三维渲染：展开坐标 → 真实空间
--------------------------------------------------------------------------------
      真实半径   R      = R_IN + r           (mm)   # r=R_INS_C 处正好是螺栓圆半径
      周向角     theta  = y / R_bc            (rad)
      轴向       X      = x                   (mm)
      笛卡尔     (X, R*sin(theta), R*cos(theta))

  python 示例（把单元场铺到真实圆筒上）：
      d = np.load('sector_layered.npz')
      R  = 1494.0 + d['rc']
      th = d['yc'] / 1550.0
      X  = d['xc']
      # 网格 (nx_e, ny, nr_e)
      Xg = X[:, None, None] * np.ones((1, len(th), len(R)))
      Yg = R[None, None, :] * np.sin(th)[None, :, None]
      Zg = R[None, None, :] * np.cos(th)[None, :, None]
      F  = d['debB_400_eps_xx']          # 任意一个单元场

  本模型只建了 5 个 cell（周向 16.1 度），要画整环就按 86.95 mm 的节距平铺。

3. 材料
--------------------------------------------------------------------------------
  mat (nx_e, ny, nr_e) int8，取值查 mat_names：
      0 lam   叶根层压                d >= 52.0
      1 steel 钢衬套/实心芯           d_bore/2 <= d < D_ins/2（盲孔底以后全钢）
      2 void  空螺孔                  d < 18.75 且 x < 130
      3 wrap  玻纤束缠绕层            过渡层以外 6 mm（过渡层并入其中）
      4 block 拉挤 GFRP 块            45.0 <= d < 52.0
  d = 到最近螺套轴线的距离，轴线在 r=R_INS_C、每 P_PITCH 一根。

4. 工况与场
--------------------------------------------------------------------------------
  工况名：intact, broken, debB_200, debB_400, debB_460, debB_400_x3, debC_400
  每个工况 <case> 有：
    <case>_eps_xx   (nx_e, ny, nr_e) float32  单元形心轴向应变（拉为正）
    <case>_svm      (nx_e, ny, nr_e) float32  单元 von Mises 应力 MPa
    <case>_tau_max  (nx_e, ny, nr_e) float32  单元剪应力合量
                     tau = sqrt(txy^2 + txr^2 + tyr^2)（不是主应力差的一半）
    <case>_u        (n_node, 3)      float32  结点位移 (u_x,u_y,u_r) mm
    <case>_surf_eps (nx_e, ny)       float32  内表面 r=0 的轴向应变（光纤读数来源）
    <case>_FS       标量  监测 cell 的螺柱轴力 N
    <case>_FKR      标量  监测 cell 的端面残余夹紧力 N
    <case>_FS_cells / <case>_FKR_cells / <case>_FA_cells  (n_cell,) 各 cell 的值
    <case>_Ns       (nx_e,)  监测 cell 的螺套钢体轴力沿 x 的分布 N
    <case>_slip     (nx_n,)  钢体与周围复合材料的横截面平均轴向位移之差 mm
    <case>_eps_in_cl(nx_e,)  中心螺套轴线所在 y 面、r=0 的轴向应变
    <case>_debond   (nx_e, ny, nr_e) bool  被判为脱粘（剪切失效）的单元
    <case>_w        标量  法兰面轴向位移 mm

5. 约定与告诫
--------------------------------------------------------------------------------
  · 脱粘用"剪切模量降到 1e-4 倍"等效，不是接触/内聚力单元：不能张开闭合，
    脱粘尖端的应力峰值受单元尺寸控制，只能当量级读。
  · 应变单位是无量纲（乘 1e6 得微应变），应力单位 MPa，位移 mm，力 N。
  · 预紧、外载：F_M 按 F_S=420 kN 标定，法兰位移按完好态单柱 F_A=159 kN 标定，
    所有缺陷工况沿用同一个法兰位移（远场载荷不变）。
