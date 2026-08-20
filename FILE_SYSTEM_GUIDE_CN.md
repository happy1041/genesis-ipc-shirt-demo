# Genesis IPC 叠衣 Demo 文件导航

本文只回答一个问题：想查看某次 Demo 的配置、轨迹、资产、诊断结果时，应从哪里找。

## 1. 整体数据流

```text
SIM1 官方资产 + episode_000000 原始轨迹
                 │
                 ▼
运行预设 run_*.sh（给本次实验定参数）
                 │
                 ▼
run_three_folds_arc.sh / run_contact_grasp_test.sh / run_fast.sh / run.sh
                 │
                 ▼
run_genesis_ipc.py（建场景、IPC、轨迹修正、录制、checkpoint）
                 │
                 ▼
outputs/episode_000000/<run>.*（视频、JSON、TCP CSV、布料状态）
                 │
                 ▼
diagnostics/（关键帧、定量指标、人工验收、实验索引）
```

## 2. 最推荐的阅读顺序

1. `README_CN.md`：项目背景、运行方式、空间约定。
2. `run_g3v2_full_fast_8000.sh`：目前人工认可的 8000 面 G3 V2 工作预设。
3. `run_three_folds_arc.sh`：三折参数如何变成 Python 命令行参数。
4. `run_contact_grasp_test.sh`：接触抓取的公共配置。
5. `run_fast.sh`、`run.sh`：最终环境、GPU、轨迹、网格和 Python 入口。
6. `run_genesis_ipc.py`：真正执行物理仿真、轨迹修正、录像和诊断的核心。
7. `diagnostics/README_CN.md`：如何评价一次运行，而不只凭单个视角观感。

如果只想复盘一个已经跑完的结果，不必先读 Python：直接查看该结果的 `.json`、
`_tcp_trajectory.csv` 和 `_multiview.mp4`。

## 3. 顶层目录和文件职责

```text
genesis_ipc_shirt_demo/
├── run_genesis_ipc.py              # 唯一核心仿真程序
├── run.sh                          # 完整网格/完整 IPC 的最底层入口
├── run_fast.sh                     # 快速求解器及可选降采样网格
├── run_contact_grasp_test.sh       # 自然接触抓取公共参数
├── run_three_folds_arc.sh          # 当前三折轨迹参数总装层
├── run_g3v2_full_fast_8000.sh      # 当前 8000 面工作基线预设
├── run_current_best_high_quality.sh# 高质量候选，不代表已经验收
├── run_13767_tuned_full.sh         # 13767 面调参候选
├── run_*                           # 历史实验、消融、单折和 checkpoint 入口
├── diagnostics/                    # 自动诊断与实验索引
├── outputs/episode_000000/         # 所有生成资产、视频、状态和报告
├── README_CN.md                    # 总说明
├── PROJECT_AUDIT_AND_ROADMAP_CN.md # 项目审计与路线图
├── IPC_DHAT_8000_DEBUG_CN.md       # contact dHat 专项记录
├── KEYFRAME_DIAGNOSTICS_CN.md      # 关键帧人工检查说明
└── THIRD_FOLD_WORK_STATUS_CN.md     # 第三折历史状态记录
```

`run_*` 数量较多，其中不少是历史实验。判断“当前正在用什么”时，不要按文件名猜，
应先看具体预设，再看对应运行输出的 JSON。

## 4. 配置到底在哪里

配置有四层，优先级从高到低为：

1. 运行命令显式附加的 CLI 参数；
2. 最外层预设脚本设置的环境变量；
3. 中间脚本中的默认值；
4. `run_genesis_ipc.py` 的 argparse 默认值。

因此，README 或 Python 默认值不一定等于某次运行的实际值。**已经完成的运行以
`outputs/episode_000000/<run>.json` 为最终事实来源。** JSON 会记录网格、求解器、
接触参数、轨迹修正、帧数、抓取摘要等实际配置。

目前最适合入门查看的预设是：

```text
run_g3v2_full_fast_8000.sh
```

它明确指定了：

- 8000 面短袖；
- fast solver；
- SIM1 原始 Acone URDF；
- `contact_d_hat=0.002 m`；
- 60 帧初始静置；
- 第一折、第二折和第三折的当前轨迹修正；
- 第三折后的张开、撤离和 120 帧稳定观察。

查看脚本生效值：

```bash
cd /home/happy1041/Workspace/Up3/genesis_ipc_shirt_demo
sed -n '1,180p' run_g3v2_full_fast_8000.sh
```

## 5. 原始轨迹与修改后轨迹

### 5.1 原始轨迹

当前固定使用：

```text
/home/happy1041/work/Sim1SameTrajectory/scale_fold_shirt_in_domain/
  episode_000000/episode_000000_sim1_replay.npz
```

这是 60 Hz、19 维关节轨迹，包含左右机械臂和夹指。`run.sh` 中的 `TRAJECTORY`
变量决定实际读取哪个 NPZ。

### 5.2 我们的轨迹适配在哪里

当前没有维护一份“手工改过的另一个 NPZ”。第一折宽度、第二折弧线/落点、第三折
抓点/放置/撤离等修改，由 `run_genesis_ipc.py` 根据命令行参数在运行时生成。
这些参数主要由 `run_three_folds_arc.sh` 转发，外层预设再覆盖其默认值。

这意味着：

- 原始 NPZ 始终可用于公平回放；
- 某个 Genesis Demo 的真实轨迹 = 原始 NPZ + 该次 JSON 中记录的修正参数；
- 只复制 NPZ 不能复现我们的 Demo，还必须保留预设脚本或运行 JSON。

### 5.3 实际执行后的 TCP 轨迹

每次运行会输出：

```text
outputs/episode_000000/<run>_tcp_trajectory.csv
```

这是修正和 IK 后真正执行的左右 TCP 位姿，比原始关节 NPZ 更适合检查某一帧的高度、
平移和朝向。可配合：

```text
diagnostics/query_tcp_frame.py
analyze_trajectory_tcp.py
```

## 6. 资产在哪里

上游 SIM1 资产：

```text
/home/happy1041/Workspace/SIM1/assets/acone/acone.urdf
/home/happy1041/Workspace/SIM1/assets/cloth/short-shirt.usdc
```

Genesis 使用的导出/降采样资产：

```text
outputs/episode_000000/assets/short-shirt.obj        # 原始约 13767 面
outputs/episode_000000/assets/short-shirt-8000f.obj  # 8000 面调试主力
outputs/episode_000000/assets/*.vertex_map.npz       # 降采样到原网格的映射
outputs/episode_000000/assets/*.garment_atlas.npz    # 领口、腰摆、袖口、双层语义
outputs/episode_000000/assets/acone_collision*/      # 机器人碰撞代理实验
```

相关生成脚本：

- `export_usdc_mesh.py`：从 SIM1 USDC 导出 OBJ；
- `prepare_lowres_shirt.py`：生成 2500/8000 等降采样网格；
- `prepare_acone_collision_proxy.py`：生成机器人碰撞代理；
- `diagnostics/build_garment_atlas.py`：建立短袖语义区域和双层对应关系。

## 7. 核心 Python 里看什么

`run_genesis_ipc.py` 较长，建议按关键词搜索：

```bash
rg -n "add_argument|contact_d_hat|settle|first_fold|second_fold|third_fold" run_genesis_ipc.py
rg -n "checkpoint|record_multi_view|tcp_trajectory|keyframe" run_genesis_ipc.py
```

主要职责包括：

- 按名称校验 19 维轨迹到 URDF 关节的映射；
- 解码官方夹爪开合量；
- 创建 Genesis FEM Cloth 和 IPC 接触场景；
- 将机器人作为规定运动边界回放；
- 运行第一、二、三折的程序化轨迹修正和 IK；
- 保存布料状态、checkpoint、TCP CSV、关键帧和多视角视频；
- 输出接触抓取与折叠诊断摘要。

## 8. 一次运行会产出什么

假设 `RUN_LABEL=my_test`，典型输出为：

```text
outputs/episode_000000/
├── my_test.mp4                     # 整体斜视
├── my_test_overhead.mp4            # 俯视
├── my_test_shirt_bottom.mp4        # 从腰摆看向领口
├── my_test_right_grasp.mp4         # 右 TCP 抓取近景
├── my_test_multiview.mp4           # 四宫格汇总
├── my_test.json                    # 本次真实配置和运行摘要（最重要）
├── my_test_tcp_trajectory.csv      # 实际执行 TCP 轨迹
├── my_test_settled_cloth.npz       # 最终布料状态
├── my_test_keyframes/              # 多视角关键帧和 contact sheet
├── my_test_second_fold_state/      # 第二折关键状态快照与摘要
├── my_test_third_fold_state/       # 第三折关键状态快照与摘要
└── my_test_evaluation/             # Gate 评价和人工复核表
```

输出目录已经包含大量历史实验。快速查找时优先看：

```text
outputs/episode_000000/experiment_index.md
outputs/episode_000000/experiment_index.csv
```

不要仅根据 MP4 文件名判断其配置；同名逻辑可能在不同时间由不同外层环境变量运行。

## 9. Checkpoint 体系

Checkpoint 位于：

```text
outputs/episode_000000/checkpoints/
```

一个 checkpoint 通常由三部分组成：

```text
<name>.pkl           # Genesis 场景状态
<name>.ipc_state.npz # IPC 原生状态
<name>.meta.json     # 源帧、资产和参数签名
```

`run_third_fold_from_checkpoint.sh` 可跳过已验收的前两折，只运行第三折后缀。
若网格、接触参数或前缀轨迹发生变化，签名不兼容时不应强行复用旧 checkpoint。

## 10. 自动诊断体系

入口：

```bash
./diagnostics/run_diagnostics.sh \
  outputs/episode_000000/<run>.json
```

主要文件：

- `diagnostics/fold_evaluation_spec.json`：各折必须观察的帧和 Gate；
- `diagnostics/analyze_semantic_states.py`：位移、叠层和双层布料分析；
- `diagnostics/evaluate_run.py`：决定 PASS/FAIL/INCOMPLETE；
- `diagnostics/capture_provenance.py`：代码、资产、环境和 GPU 追溯；
- `diagnostics/index_runs.py`：更新统一实验索引；
- `diagnostics/README_CN.md`：完整诊断流程。

诊断应按“最早失败 Gate”推进：第一折未通过时冻结第二、三折；第二折未通过时不调
第三折。数值日志用于定位，但抓层错误、穿模、擦碰和折片落点仍必须查看多视角关键帧。

## 11. 空间描述统一约定

所有讨论均以短袖自身为参考：

- 上/下：领口侧 / 腰摆侧；
- 左/右：穿衣者本人的左袖 / 右袖；
- 正面/背面：胸腹侧 / 后背侧；
- 高/低：垂直桌面的高度。

不要用画面左右、相机远近或“屏幕里的左机械臂”描述衣服位置。

## 12. 最实用的复盘方法

对任意一次结果，按以下顺序看：

1. `<run>_multiview.mp4`：发现肉眼可见的大问题；
2. `<run>_keyframes/*contact_sheet*`：定位问题第一次出现的关键帧；
3. `<run>.json`：确认该次实际参数，而不是读取脚本默认值；
4. `<run>_tcp_trajectory.csv`：确认夹爪实际高度、落点和姿态；
5. `*_fold_state/` 和 evaluation：判断整体平移、抓层、叠层和稳定性；
6. 只改一个参数族，使用新 `RUN_LABEL` 运行并比较。

当前建议把 `run_g3v2_full_fast_8000.sh` 当作可读的工作起点，而不是把整个输出目录
中最新生成的文件自动视为“最佳版本”。高面数和 slow solver 的结果属于验证候选，
只有通过同一套关键帧和 Gate 后才能晋升为基线。
