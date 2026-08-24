# Genesis 本地补丁说明

## 补丁来源

`genesis-world-8b1dba2-local.patch` 不是一个经过最小化验证的单功能补丁，而是
demo 调试期间 Genesis 工作区相对官方 commit `8b1dba2` 的累计差异快照。

这些修改在 2026-08-16 至 2026-08-18 期间逐步形成，并于 2026-08-24 整理
公开仓库时统一导出。补丁涉及：

- `genesis/engine/couplers/ipc_coupler/coupler.py`
- `genesis/engine/scene.py`
- `genesis/options/solvers.py`
- `genesis/vis/visualizer.py`

## 当前主线与历史功能

当前正式配置 `configs/dhat_1p5_pure_friction.args` 使用真实夹指接触与摩擦，
`virtual_grasp` 关闭。补丁中仍包含早期 soft virtual grasp/FEM vertex constraint
实验所需的代码；保留这些代码不表示当前结果使用了软抓取。

补丁还包含 IPC affine-body 状态访问、视觉模型跟随 IPC actual pose、录制/渲染
时序和求解参数接口等修改。当前仓库尚未完成逐项删减实验，因此不能保证补丁中的
每一行都是纯摩擦主线必需，也不应把它描述为上游 Genesis 的通用修复。

## 应用方式

只应对官方 Genesis commit `8b1dba2` 应用：

```bash
git -C /path/to/genesis-world checkout 8b1dba2
git -C /path/to/genesis-world apply --check \
  /path/to/this-repo/patches/genesis-world-8b1dba2-local.patch
git -C /path/to/genesis-world apply \
  /path/to/this-repo/patches/genesis-world-8b1dba2-local.patch
```

如果 `--check` 失败，不要强行应用；应先确认 Genesis commit 和工作树是否干净。

## 后续清理任务

公开交接的理想状态是把累计补丁拆成：

1. 纯摩擦主线实际需要的最小补丁；
2. soft virtual grasp 历史实验补丁。

拆分前必须分别执行 trajectory preflight、短程 GPU smoke 和完整三折回归，避免
删除看似停用但仍参与 IPC 初始化或离线渲染的代码。
