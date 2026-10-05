# Genesis＋libuIPC 叠衣 demo：组内代码包

本仓库包含Scene527双臂叠衣的代码与**4373帧55k七段版复现数据**：衣服、双X5模型/碰撞网格、阶段checkpoint、原始replay、摩擦参数和运行入口。Isaac只读取保存状态做离线渲染。完整说明见[55k复现指南](docs/SCENE527_REPRODUCTION_CN.md)。

```bash
git lfs pull
export GENESIS_PYTHON=/path/to/patched-genesis-env/bin/python
./scripts/run_scene527_reproduction.sh --verify
```

查看包内`reproduction/scene527_55k_73s/reference/hero.mp4`，或按指南重渲染/重新仿真。下面的`run_demo.sh`默认入口仍保留历史13k资产约定；新版结果使用上面的专用入口。

## 从哪里看

- `src/run_genesis_ipc.py`：Genesis 场景、libuIPC 接触、机器人轨迹执行、checkpoint/replay 与诊断。
- `tools/contact_dump.py`：可选的原始 IPC 接触记录辅助模块；主程序的该项诊断只在指定 `--contact-dump-frames` 时调用。
- `scripts/`：运行和环境检查入口；`configs/dhat_1p5_pure_friction.args` 是历史参数基线，当前 55k 成片的参数摘要见 [`docs/CURRENT_55K_DEMO_CN.md`](docs/CURRENT_55K_DEMO_CN.md)。
- `patches/genesis-world-8b1dba2-current.patch`：官方 `8b1dba2` 基线到当前运行源码的完整补丁，涉及 7 个文件；应用后逐文件字节匹配已验证。[逐项改动与仿真影响](docs/GENESIS_PATCH_CHANGES_CN.md)。
- `data/scene527_55k_73s_joint_commands.npz`：当前约 1:13 分段版的累计机器人命令，见 [`data/README.md`](data/README.md)。
- `reproduction/scene527_55k_73s/`：完整冻结输入、七段状态与回放、参考hero及Isaac静态场景。
- `scripts/run_scene527_reproduction.sh`：数据验证、原状态回放、单段恢复、重新运行阶段链或单进程全程的入口。

这里的“代码”主要是 `src/` 与 `tools/assets/` 下的 Python 文件，以及 `scripts/` 下负责启动、准备资产和检查环境的 Shell 脚本。`configs/*.args` 是参数文本；`requirements/` 是依赖版本记录；`patches/*.patch` 是对外部 Genesis 源码的修改，不是独立程序。`data/*.npz` 是逐帧机器人命令数据，不是代码，也不包含衣服的运动状态。

## 如何启动基础入口

在 Linux/NVIDIA GPU 环境中使用与 `pyuipc==0.0.25` 兼容的 Genesis；补丁的基线与安装说明在 [`docs/DEVELOPMENT_CN.md`](docs/DEVELOPMENT_CN.md)。

```bash
cp .env.example .env
# 编辑 .env 中的 SIM1、Genesis、Python、轨迹及外部资产路径
./scripts/check_setup.sh
./scripts/run_demo.sh --physics-only
```

上述默认入口保留了旧 13k 参数与资产约定。要研究随附的 55k 命令轨迹，需将 `TRAJECTORY` 指向 `data/scene527_55k_73s_joint_commands.npz`，并提供匹配的 **55,068 面衣服**与**双 X5 URDF/碰撞网格**；具体数值覆盖见 [`docs/CURRENT_55K_DEMO_CN.md`](docs/CURRENT_55K_DEMO_CN.md)。不要将不同拓扑的抓点编号或 checkpoint 混用。

新补丁包含 IPC 状态访问、视觉同步、布料自摩擦、指定接触对摩擦，以及前向仿真的 GPU 状态历史管理修改。已从官方 `8b1dba2` 提交的原始文件应用补丁，确认全部 7 个结果文件与当前运行源码逐字节相同；`check_setup.sh` 同步检查这些文件的哈希。补丁验证不包含新 GPU 仿真。七段成片通过 checkpoint 接续，单独的机器人命令不会重建已折好的衣服；复现仍需外部资产和对应阶段状态。

## 提交范围

[`docs/SOURCE_FILES.txt`](docs/SOURCE_FILES.txt)记录基础代码选择；[来源记录](docs/PROVENANCE_CN.md)与复现包`bundle.json`记录哈希和源运行。包不含Workbench或整个SIM1/Genesis/虚拟环境；只收集实际需要的资产和冻结结果，大文件由Git LFS管理。原仓库`diagnostics/`与历史说明保留。研究共享声明见[`NOTICE.md`](NOTICE.md)。
