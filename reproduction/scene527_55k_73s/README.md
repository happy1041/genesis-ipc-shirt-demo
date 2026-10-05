# 4373帧55k复现数据

运行与参数解释见仓库[`docs/SCENE527_REPRODUCTION_CN.md`](../../docs/SCENE527_REPRODUCTION_CN.md)。

`bundle.json`记录相对路径、来源和SHA256。先执行`git lfs pull`，再用`GENESIS_PYTHON=/path/to/env/bin/python ./scripts/run_scene527_reproduction.sh --verify`验证文件。checkpoint压缩包为源三件套的无损副本；源URDF原文保留，网格相对路径已经齐全。
