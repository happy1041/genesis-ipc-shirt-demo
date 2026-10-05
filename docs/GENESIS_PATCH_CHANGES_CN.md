# Genesis 补丁改动与仿真影响

这份补丁同时包含接触模型改动、状态访问接口和运行辅助修改。直接撤掉整份补丁再比较成片，会同时改变摩擦与机器人内部接触，不能把差异全部归因于数值求解或 checkpoint。当前 demo 主程序也不能不做适配就直接运行于原版 Genesis。

本文根据[补丁源码](../patches/genesis-world-8b1dba2-current.patch)及 demo 主程序核对。补丁基线为官方提交 [`8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44`](https://github.com/Genesis-Embodied-AI/Genesis/commit/8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44)，共改 7 个文件、增加 298 行、删除 25 行。版本、结果哈希和应用验证见[补丁 manifest](../patches/genesis-world-8b1dba2-current.manifest.json)。

## 文件与作用

以下路径相对于外部 Genesis checkout，不是本 demo 仓库。

| 修改文件 | 主要修改 | 对当前仿真的意义 |
| --- | --- | --- |
| `genesis/engine/couplers/ipc_coupler/coupler.py` | 摩擦接触对、机器人内部接触开关、FEM 状态访问、虚拟抓取接口 | 包含直接改变物理的部分，也包含默认不启用的实验接口 |
| `genesis/engine/materials/FEM/cloth.py` | 增加 `self_friction_mu` | 允许布—布摩擦与布—桌、布—机器人摩擦分开设置 |
| `genesis/engine/entities/fem_entity.py` | `get_state()` 只在需要梯度时保留查询历史 | 减少前向仿真的 GPU 内存积累 |
| `genesis/engine/solvers/fem_solver.py` | `save_ckpt()` 只在需要梯度时保留反向计算历史 | 减少 GPU 历史快照，仍保留前向步进所需的环形缓冲区复制 |
| `genesis/engine/scene.py` | 注册渲染状态完成后的回调 | 提供真实 IPC 姿态的视觉同步入口 |
| `genesis/vis/visualizer.py` | 在视觉更新后执行回调 | 使显示的机器人外观与 IPC 碰撞体姿态对应 |
| `genesis/options/solvers.py` | 接触选项枚举由 `ipc/isometric` 改为 `ipc/al-ipc` | 当前使用 `ipc`，没有切换到 `al-ipc` |

## 直接影响物理的改动

### 布料自摩擦与接触对摩擦

原版按实体材料摩擦系数的几何平均组合接触对：`mu_ij = sqrt(mu_i * mu_j)`。同一布料的自接触因此使用布料自己的 `friction_mu`。新补丁增加两个入口：

- `Cloth.self_friction_mu` 只覆盖布料自接触。
- 实体的 `_ipc_pair_friction_overrides` 可以只覆盖指定的实体接触对。

当前 55k 条件为布料材料摩擦 1、桌面材料摩擦 1、机器人材料摩擦 2；另指定布—布摩擦 2、布—桌接触对摩擦 1。因此布—机器人保留几何平均值 `sqrt(2)`。

若删除该功能并继续用原材料参数，布—布摩擦会从 2 变成 1。理论上这会改变层间滑动阻力和褶皱展开过程，但不能仅从系数变化推断最终折形一定更好。当前布—桌覆盖值 1 与原组合值恰好相同；此前布—桌 0.5 实验则依赖独立接触对覆盖。不能用同一组原材料参数在原版上保留这些不同条件。

### 同一机器人实体的 IPC 自碰撞

原版的 `enable_rigid_rigid_contact=True` 会启用 ABD 刚体接触，也包含同一实体的接触元素自配对。一个机器人的多个耦合腕掌/夹片链接共享一个 ABD 接触元素，因此这种自配对可以让相邻机器人部件进入 IPC 接触求解。

补丁保留不同刚体实体之间的接触，并关闭同一刚体实体的接触自配对。它没有增加或替代完整的机器人自碰撞检测；PATH/IK 的几何合法性仍需单独检查。

本机当前连续 demo 的运行记录中 `ipc_rigid_rigid_contact=True`。因此撤掉这个改动会改变参与求解的接触对，可能影响夹片闭合、IPC 代理姿态与求解稳定性。具体影响大小需实验，源码核对不能证明某次失败由它导致。

## 接口与辅助改动

### 虚拟抓取接口

补丁增加软位置约束与硬绑定的注册、启用和清除接口。软约束通过 animator 更新顶点目标；硬绑定若启用，会在 IPC `advance()` 前写入选定顶点的位置和速度。这些接口在启用时会直接改变运动。

当前 demo 的 `virtual_grasp=False`，硬绑定掩码没有启用，`_apply_fem_hard_bindings()` 在状态写回前返回；当前抓取依靠实际夹爪接触与摩擦。软位置约束类型仍会在建场景时注册，默认顶点掩码为 0。按实现意图，这时不应产生虚拟抓取力；类型注册与回调改变了运行路径，未做消融实验前不能声称其数值结果绝对相同。

### FEM 状态访问与 checkpoint

补丁接入 `FiniteElementStateAccessorFeature`，暴露 FEM 位置和速度的读写通道。主程序的 checkpoint 保存通过 `copy_to()` 读取状态，恢复则调用 `copy_from()` 写入状态。

连续仿真中的读取不应给布料施加额外力。从 checkpoint 恢复涉及重新建立世界和写回状态；该接口本身并不证明所有原生求解器内部状态都已恢复，也不证明续跑与连续运行完全相同。

### 视觉同步

新增的 `register_post_visual_state_callback()` 在求解器视觉变换更新后执行。当前 `IPCActualVisualSynchronizer` 用它把 `_vgeoms_render_T` 改为实际 IPC 刚体姿态，避免显示模型与碰撞代理错位。这条当前回调修改的是渲染变换，没有向 IPC 世界写入布料或刚体物理状态。

### 前向仿真的 GPU 历史管理

原版 `FEMEntity.get_state()` 每次把查询结果保存在 `_queried_states`；`FEMSolver.save_ckpt()` 为每个历史窗口保留 GPU 张量。补丁在 `requires_grad=False` 时省去这些用于梯度计算的历史，仍保留前向环形缓冲区的复制。

理论上这些修改不改变当前前向运动方程，主要防止长时间运行积累显存。这不是用户保存到磁盘的 checkpoint 格式，也不是新增阻尼。显存布局和执行时序仍可能影响浮点数重复性，不能由此承诺逐字节一致。

## 补丁没有改什么

补丁没有修改 libuIPC 的原生 Newton、线性求解器或 CCD 实现，没有改布料弹性/弯曲本构公式，也没有加入布料应力松弛或阻尼算法。当前使用的 `contact_constitution='ipc'` 在新旧枚举中都存在；求解容差和迭代次数由 demo 主程序传入。

因此“布片很弹”或“褶皱铺平慢”不能直接解释为这份补丁重写了这些算法。自摩擦改动与层间滑动相关，但材料参数、几何、抓取压缩和数值收敛也需要各自的证据。

## 恢复原版后能否直接对比

当前主程序构造 `Cloth` 时传入 `self_friction_mu`，原版材料类没有这个字段，且选项模型禁止额外字段，会先报参数错误。checkpoint 状态访问和视觉回调也依赖补丁接口。

适配到原版后，如果同时撤掉自摩擦与机器人接触规则，运行的物理条件已经改变；看到结果不同不能单独证明辅助接口有问题。若研究辅助改动的影响，应先保持相同摩擦、接触对和轨迹，再逐组比较视觉回调、状态读取或历史管理。比较布料位置、接触与折形指标，不只看视频或要求所有数组逐字节相同。

## 已验证范围

已从官方完整提交取得基线，执行 `git apply --check`、实际应用、全部 7 个结果文件与当前源码的逐字节比较、反向应用检查和 Python 编译。结果在补丁 manifest 中记录。

本次说明来自源码分析，没有备份/切换正在使用的 Genesis，也没有执行原版与补丁版的新 GPU 对照。因此上文有关物理影响的大小是待验证推断，不是新的实测结论。
