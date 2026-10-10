# 渲染微调后的新仿真结果

Isaac 只离线读取已保存状态。修改 plan、生成新 NPZ 并运行物理后，应使用该次物理输出的 `stage_physics.replay_states.npz`，机器人视觉变换也必须从同一份 replay 导出。

以下命令在仓库根目录执行。`$GENESIS_PYTHON` 是安装本仓库 Genesis 的 Python，`$ISAAC_PYTHON` 是安装 Isaac Sim 的 Python。输出目录必须是新的空路径。

## 1. 从新 replay 导出 Genesis 视频和机器人视觉状态

```bash
"$GENESIS_PYTHON" tools/run_scene527_reproduction.py \
  --mode replay --stage 03_second_fold \
  --replay-override /absolute/path/to/new/stage_physics.replay_states.npz \
  --output outputs/new_second_fold_visuals
```

输出目录 `outputs/new_second_fold_visuals/03_second_fold/` 包含 `stage_multiview.mp4` 和 `robot_visuals.npz`。源帧和机械臂姿态全部取自新 replay；`--stage` 用于选择相同机器人、拓扑与场景的配置。需要只补机器人视觉文件时，在命令末尾添加 `--robot-visuals-only`；这仍需初始化 Genesis 可视化环境，但不推进物理也不生成视频。

底层入口 `src/run_genesis_ipc.py` 同样支持 `--replay-states ... --dump-robot-visuals ... --robot-visuals-only`，可接到已有 Workbench 运行包的离线回放命令。请保持该运行包的网格、机器人和初始放置参数。

## 2. 准备并运行新结果的 Isaac hero

```bash
"$GENESIS_PYTHON" tools/prepare_scene527_isaac.py \
  --job outputs/new_second_fold_isaac \
  --isaac-python "$ISAAC_PYTHON" \
  --replay /absolute/path/to/new/stage_physics.replay_states.npz \
  --robot-visuals outputs/new_second_fold_visuals/03_second_fold/robot_visuals.npz

"$GENESIS_PYTHON" outputs/new_second_fold_isaac/controller.py run \
  --job outputs/new_second_fold_isaac
```

第一条命令只做 CPU 准备；也可添加 `--run` 接着启动离线渲染。输出为 `outputs/new_second_fold_isaac/deliverables/hero.mp4`，普通 RaytracedLighting、带阴影、1080p/60fps。渲染本地帧从 0 编号，原仿真帧号保存在 `source_frames.json`。

`PREPARATION.json` 记录新 replay、机器人视觉文件的路径和哈希，`config.json` 记录实际帧数及渲染输入，避免误用冻结参考回放。新 replay 长度无需等于原阶段长度。

### 机器人视觉几何顺序

Genesis 固定关节合并可能改变视觉几何排列。工具默认用新 replay 第一帧和冻结回放同源帧的机器人姿态匹配几何顺序，超过误差阈值会停止。若源帧重新编号，可以通过 `--reference-frame N` 指定冻结回放里同姿态的帧。

如果新 replay 一开始的姿态就大幅改变，先对同一机器人配置的一份未修改姿态 replay 做上述 CPU 准备，获得 `robot_mapping.json`，随后传 `--robot-mapping path/to/robot_mapping.json`。这个文件是一组 USD 几何序号到 Genesis 几何序号的排列，只能在机器人模型、视觉几何和合并设置相同时复用。

### 当前拓扑范围

这里复用交付包的 55k 冻结 USD 场景，仅支持相同 27,811 顶点的 Scene527 衣服及相同拓扑顺序。8k/13k 可在 Genesis/Workbench 中调试；将它们用于此 Isaac 场景需要另建对应衣服 USD 网格，不能把低面数位置数组直接套入 55k 场景。

## 多阶段独立复验的兼容入口

`--stage-results path/to/stage_results` 保留之前的“第一折参考回放 + 新跑后六段”组装方式。每个阶段子目录须同时有 `stage_physics.replay_states.npz` 和 `robot_visuals.npz`，且帧范围与冻结 bundle 一致。它不能和 `--replay` 一起使用。微调阶段长度或任意单段结果应使用上面的 `--replay` 模式。
