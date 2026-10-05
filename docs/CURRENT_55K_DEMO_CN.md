# 当前 55k 分段版的参数和边界

约 1:13 的版本来自七个 checkpoint 接续阶段，指令帧为 0–4372、60 fps。随附 NPZ 是最终阶段使用的**累计关节命令**；七段轨迹的对应前缀已逐元素核对相同。物理状态不包含在此仓库中。

从各段运行记录与基础参数文件读取的共同条件：

| 项目 | 数值 |
| --- | --- |
| 衣服 | 27,811 顶点／55,068 三角面 |
| 动作频率 | 60 fps |
| IPC 子步 | 2 |
| Young 模量 | 20,000 Pa |
| 弯曲参数 | 40 |
| 布—布摩擦 | 2 |
| 布—桌接触对摩擦 | 1 |
| 接触 `dHat` | 0.0015 m，来自基础预设 |
| 厚度 | 0.0001 m，来自基础预设 |
| 初始衣服位置 | `(0.667, 0.015, 0.93)` m |

最终阶段的运行命令还指定 `--action-plan-trajectory`、`--allow-material-change-on-checkpoint`，并加载上一阶段 checkpoint。把这些末段参数直接用于从初态运行，与七段物理续接**不是同一实验**。源运行记录在原工作区 `outputs/workbench/scene527_55068/final_push/20260929_press_at_yend_mu1_home/table_mu1/full_action/run_bundle/manifest.json`；该路径只用于追溯，不是本仓库依赖。

本包现有`genesis-world-8b1dba2-current.patch`可从官方基线重建本机当前7个修改的Genesis源码文件，应用结果已逐字节验证。`coupler.py` SHA256为`c403be43580383ccd018df1386984ea5ebb2683511431711747856407012065b`，与本机55k全程记录匹配。源码重建通过不代表完整物理重复性验证；环境、资产、阶段checkpoint与此前观测到的数值重复性差异仍需分别核验。
