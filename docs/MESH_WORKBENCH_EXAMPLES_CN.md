# 8k / 13k Workbench 选点示例

仓库附带匹配拓扑的真实第一折末段回放、TCP 和 atlas。可直接查看布层、切帧、选择左右抓放点并导出 JSON。

|版本|顶点 / 面|示例及来源清单|
|---|---|---|
|8k|4090 / 8000|[8k manifest](../reproduction/workbench_examples/8k/manifest.json)|
|13k|7021 / 13767|[13k manifest](../reproduction/workbench_examples/13k/manifest.json)|

## 启动

在已安装 Workbench 依赖的仓库根目录执行；输出目录须尚不存在：

```bash
git lfs pull
python tools/prepare_workbench.py --mesh-example 13k --output outputs/edit_13k
python tools/trajectory_workbench/server.py --config outputs/edit_13k/workbench.json --port 8766
```

打开 `http://127.0.0.1:8766`。8k 换成 `--mesh-example 8k --output outputs/edit_8k`，服务端也指向该目录。
运行器将仓库相对路径绑定到本次 checkout；无需原作者的绝对目录。保存的点在本次输出目录的 `edits/`。

## 样本范围和后续物理

两版均来自 `20260927_no_spread_8k_13767_source_mm` 的同名网格运行，保留源帧 **770–829，共60帧**，默认显示829。所有布料位置、机器人关节/夹片变换和TCP均从原回放直接截取；没有重模拟、缩放或重排顶点。新建 atlas 使用对应 OBJ 的原始顶点顺序。

这些是**查看/选点样本**，不附带物理续跑 checkpoint；画面只绘制布料和TCP标记，不提供完整机器人外观。界面中的参考路径是记录的TCP，不能把这些末态坐标当作物理状态恢复文件。

需要用8k/13k继续微调并仿真时，按[面数切换指南](CLOTH_MESH_SWITCH_CN.md)从该拓扑初态跑到目标阶段，保存自己的 checkpoint/replay/TCP，再用这些路径建立 Workbench 配置。随后沿[plan → IK → NPZ 指南](PLAN_TO_TRAJECTORY_CN.md)执行；55k七段 checkpoint 不能恢复到这两种网格。

55k原入口仍为 `python tools/prepare_workbench.py --after-stage 03_second_fold --output outputs/edit_55k`，会准备其可续跑checkpoint和指令前缀。

## 已执行检查

两个样本均核对原运行OBJ与交付OBJ的SHA256、顶点/面数、60帧长度、TCP帧一致性和atlas顶点顺序；WorkbenchData载入、CPU画面生成、布料射线选点通过。未追加GPU物理运行。

来源哈希及样本文件哈希在上表manifest中。维护者可以用 `tools/package_mesh_workbench_examples.py SOURCE_BATCH` 从原始数据重新提取到空的样本目录。
