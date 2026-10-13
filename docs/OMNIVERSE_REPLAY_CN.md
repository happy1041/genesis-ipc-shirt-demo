# Isaac / Omniverse 离线渲染源码与回放对齐

渲染源码、控制器及静态场景已随复现包上传；它们位于 `reproduction/scene527_55k_73s/isaac/`，不是外部私有依赖。需要 NVIDIA Isaac Sim 环境，当前交接链按 Isaac Sim 5.0 使用。

## 代码入口

|文件|用途|
|---|---|
|[renderer.py](../reproduction/scene527_55k_73s/isaac/renderer.py)|创建 SimulationApp、打开 USD、逐帧写布料顶点与机器人视觉矩阵、Replicator RGB、FFmpeg 编码|
|[controller.py](../reproduction/scene527_55k_73s/isaac/controller.py)|启动渲染进程、分块进度和视频拼接|
|[prepare_scene527_isaac.py](../tools/prepare_scene527_isaac.py)|解包场景、重定位路径、组装回放、匹配机器人几何顺序并生成运行配置|
|[settings.json](../reproduction/scene527_55k_73s/isaac/settings.json)|冻结相机、分辨率、光照和渲染参数；运行时由准备工具重定位路径|
|[run_genesis_ipc.py](../src/run_genesis_ipc.py)|物理状态记录，以及从同一 replay 导出实际机器人视觉变换|

`renderer.py` 的 `cloth.GetPointsAttr().Set(...)` 和 `op.Set(Gf.Matrix4d(transforms[frame, i].T.tolist()))` 是布料和机器人更新位置的关键两处。源码为冻结复现输入，准备工具会复制到新 job，不建议直接运行仍含参考配置路径的模板。

## 运行随包的完整参考回放

在仓库根目录执行，先确保 `git lfs pull` 完成。`GENESIS_PYTHON` 用安装了本项目依赖的 Python；`ISAAC_PYTHON` 用能够导入 Isaac Sim 的 Python。

```bash
export GENESIS_PYTHON=/path/to/genesis-env/bin/python
export ISAAC_PYTHON=/path/to/isaac-env/bin/python
bash scripts/render_scene527_isaac.sh \
  --job outputs/scene527_isaac_hero \
  --isaac-python "$ISAAC_PYTHON" --run
```

输出为 `outputs/scene527_isaac_hero/deliverables/hero.mp4`，普通 RaytracedLighting、1080p/60fps。省略 `--run` 只准备文件。新仿真或新轨迹结果请按[新 replay 渲染指南](RENDER_NEW_REPLAY_CN.md)提供同源 `--replay` 与 `--robot-visuals`，避免继续播放包内参考状态。

## 渲染与穿模有什么关系

这条渲染链暂停 timeline，每次 Replicator 调用使用 `pause_timeline=True, delta_time=0.0`，并检查时间没有推进。Isaac 不重新求解衣服接触，画面来自 Genesis/libuipc 保存的状态。

但画面仍可能产生视觉穿模：

1. **帧不一致**：衣服和机器人来自不同运行或不同 source frame。准备工具检查同源帧数组；不要单独拼接一条旧机器人动画。
2. **姿态来源不同**：关节目标姿态不一定等于 IPC 实际碰撞刚体姿态。使用同一 replay 导出的实际视觉变换，不用另一条轨迹重新做 FK 替代。
3. **变换或几何顺序错误**：USD 的矩阵约定、父节点平移、visual 的局部 origin、固定关节合并后的几何排列均需匹配。准备工具会输出 `robot_mapping.json`；它仅能用于相同机器人几何和合并设置。
4. **visual 与 collision 不一致**：物理碰撞网格可能比可见夹片小，实际接触无穿透也可能看起来插进布里。把实际 collision proxy 叠加到相同帧检查。
5. **衣服拓扑或桌面坐标不一致**：顶点数相同仍不保证顶点顺序相同。此交付场景匹配指定55k拓扑与桌面，换机器人/衣服/桌面必须同步更新USD几何及映射。

排查时选一帧，在 Genesis 和 Isaac 中核对同一 `source_frame` 的布料顶点、夹指实际矩阵和桌面高度。如果保存状态与碰撞代理已经相交，检查物理；如果保存状态正确但两个渲染结果不同，检查上述回放变换。仅凭视频不能认定是哪一层出错。

本入口复用已验证的渲染实现；此次补充入口和说明只做CPU命令/编译验证，没有追加GPU渲染，也不承诺接收方新机器人已经兼容。
