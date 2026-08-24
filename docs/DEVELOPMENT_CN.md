# 开发与复现

## 外部依赖

- Genesis World commit `8b1dba2`
- SIM1 commit `973476b`
- episode 000000 replay trajectory
- SIM1 short-shirt USD和Acone URDF/visual meshes
- 一个安装Genesis、libuIPC、trimesh、Pillow等依赖的Python环境
- 导出USD时需要包含`pxr`的SIM1 Python环境

应用Genesis补丁：

```bash
git -C /path/to/genesis-world checkout 8b1dba2
git -C /path/to/genesis-world apply --check \
  /path/to/this-repo/patches/genesis-world-8b1dba2-local.patch
git -C /path/to/genesis-world apply \
  /path/to/this-repo/patches/genesis-world-8b1dba2-local.patch
```

该补丁是2026-08-16至08-18调试期间的累计工作区差异，并非已经最小化的单功能
修复。当前纯摩擦主线关闭了virtual grasp，但补丁仍包含早期soft virtual grasp
实验代码。完整范围、限制和后续拆分计划见[补丁说明](../patches/README_CN.md)。

## 环境变量

```bash
cp .env.example .env
# 编辑.env后：
./scripts/check_setup.sh
```

也可以直接设置 `GENESIS_PYTHON`、`TRAJECTORY`、`SHIRT_OBJ`、`ROBOT_URDF`。
`GENESIS_GPU`可选；未设置时由Genesis选择设备，不再写死本机GPU UUID。
`scripts/run_demo.sh`、`scripts/check_setup.sh`会自动读取仓库根目录的`.env`。
资产包目录和协作者操作见[资产交接](ASSETS_CN.md)。

## 资产准备

```bash
./scripts/prepare_assets.sh
```

脚本会把短袖USD导出到 `outputs/.../assets/short-shirt.obj`，并从干净的SIM1
Acone URDF生成项目所需的IPC腕掌凸包和运行时URDF。生成物包含本机绝对路径，
因此只存在于被Git忽略的outputs中，不进入公开仓库。

## 运行架构

```text
configs/*.args
       ↓
scripts/run_demo.sh       物理/离线渲染编排
       ↓
scripts/_launch.sh        环境、GPU锁、资产和Python入口
       ↓
src/run_genesis_ipc.py    场景、轨迹修正、IPC、诊断、录制
```

正式运行先关闭相机录制完成物理并保存replay states，再单独渲染。原因是实时
GPU录制会改变CUDA调度，使临界接触求解走到不同数值分支。

## 配置与实验

正式参数唯一真源为 `configs/dhat_1p5_pure_friction.args`。临时覆盖示例：

```bash
RUN_LABEL=test_settle \
  ./scripts/run_demo.sh --post-release-settle-frames 60
```

建议一次只改变一个参数族，并使用新的`RUN_LABEL`。旧实验脚本已从主分支删除，
可从Git标签 `pre-reorg-20260824` 查阅。

## 代码边界

- `src/run_genesis_ipc.py`仍是单文件核心；本轮只整理仓库，不同时拆分5000行物理
  逻辑，避免改变已验证行为。
- `diagnostics/`只读输出JSON/NPZ并生成评价结果。
- `tools/assets/`只负责生成外部资产派生物。
- `patches/`记录上游Genesis尚未合入的IPC和渲染同步修改。

## 已知可移植性限制

- 仓库不分发SIM1数据和visual meshes；许可需由项目负责人确认。
- 没有选定开源LICENSE；当前仅用于研究审阅。
- 完整GPU运行无法由普通CI覆盖，需要人工执行preflight和短程smoke。
