# 切换衣服面数：8k / 13k / 55k

**切换面数就是切换物理网格资产。** 本分支已附带同一短袖的三个分辨率，无需重新建模。只有需要其他面数或款式时，才需要简化、细分或制作新网格。

## 1. 下载并选择资产

在仓库根目录执行 `git lfs pull`。三个OBJ均为原始毫米坐标，当前运行器以 `scale=0.001` 转换为米；不要预先再缩小一次。

|版本|顶点数|三角面数|资产|
|---|---:|---:|---|
|8k，历史对照|4090|8000|[short-shirt-8000f.obj](../reproduction/scene527_55k_73s/assets/cloth/short-shirt-8000f.obj)|
|13k，常规调试|7021|13767|[short-shirt-13767f.obj](../reproduction/scene527_55k_73s/assets/cloth/short-shirt-13767f.obj)|
|55k，高分辨率复验|27811|55068|[short-shirt-55068f.obj](../reproduction/scene527_55k_73s/assets/cloth/short-shirt-55068f.obj)|

资产数量和SHA256见[网格清单](../reproduction/scene527_55k_73s/assets/cloth/mesh_variants.json)。目录名含55k，是历史交付包名称；具体使用哪个文件由参数决定。

## 2. 替换运行参数，从初态开始

在已配置好的 `src/run_genesis_ipc.py` 完整命令中，替换衣服参数。以下以13k为例，只是参数片段：

```bash
--shirt-obj reproduction/scene527_55k_73s/assets/cloth/short-shirt-13767f.obj \
--expected-cloth-vertices 7021 \
--expected-cloth-faces 13767
```

8k或55k按表替换文件名及两个数量。使用新的输出目录，移除旧拓扑的checkpoint恢复参数；从目标网格初态运行，后续阶段恢复本轮新生成的checkpoint。

**随包七套checkpoint、参考replay和默认bundle仍只对应55k。** `tools/run_scene527_reproduction.py` 的55k复现流程不会因为新资产存在就自动切换；要用该编排跑其他网格，需要建立独立bundle，更新网格/拓扑/哈希并重建状态链，不能直接套用旧checkpoint。

## 3. 同步检查抓取和渲染

已附[8k/13k Workbench真实布态示例](MESH_WORKBENCH_EXAMPLES_CN.md)，包含匹配replay、TCP、atlas及配置；可直接启动选点查看。它们不附物理续跑checkpoint。

- 机器人轨迹可作初猜，先检查新网格的尺寸、摆放和抓取效果，再跑全程。面数变化会影响接触和褶皱，不保证旧轨迹仍夹得住。
- 面ID、顶点ID、atlas和布料状态不能跨拓扑直接复用。Workbench须导入匹配网格与replay，重新选点或建立材料点对应。
- Isaac/Genesis离线回放须使用同一拓扑；Isaac准备工具的 `cloth_obj` 也要对应替换。
- E、B、密度和摩擦先保持一致，不按面数比例缩放。先做网格对照，再单独标定材料。

即使顶点/面数相同，顶点顺序或连接关系不同也可能使旧状态失效。`--allow-material-change-on-checkpoint` 不能用于跨拓扑恢复。

本次补充核验了8k/13k文件哈希、顶点/三角面数量及有效面索引；没有新增这两种网格的全程复现验收。55k原交付基线和结果保持原样。
