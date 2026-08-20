# Genesis IPC 短袖叠衣当前状态

更新时间：2026-08-17（Asia/Shanghai）

## 2026-08-17 关键控制问题修复

第三折放手附近的整臂瞬移已经修复，并从 frame-619 checkpoint 与初始状态
各完成一次 8000 面物理验证。旧运行在 frame 959→960 的右 TCP 单帧跳变为
`246.354 mm`；修复后 frame 920--979 最大单帧位移为 `1.763 mm`，frame 960
约为 `1.726 mm`。

直接根因是：普通第三折 correction weights 在 frame 960 归零后，轨迹预计算
错误回退到原始公开轨迹，漏掉仍应活跃的 `third_fold_release_hold`。当前实现会在
frame 920 捕获释放位姿，保持到 945，并在 946--979 沿上一帧修正姿态的同一
IK 分支竖直上撤约 40 mm。

同时修复了 release-hold 冻结 finger DOF 的问题。连续 IK 只继承机械臂姿态，
四个夹指每帧重新合并当前 source finger command。完整验证中 frame 960 右夹指
实际位置约为 `43.66/43.57 mm`，frame 979 为 `44.02/43.97 mm`，不再停在
闭合时的约 `1.7--1.8 mm`。

第三折闭合前的目标穿桌也已修复：延长 approach lift 到 post-close lift 接管，
frame 683/686 的 IPC 代理桌面净空分别为 `4.64/5.64 mm`；完整第三折范围内
target 与 proxy 的最小桌面净空均为正。第三折抓取在 checkpoint 后缀与完整运行
中均为 `PASS`。

新增 CPU-only `--trajectory-preflight-only` Gate，检查 frame 920--979 的 TCP、
右臂关节、保持/上撤路径和夹指 source command 连续性；预检不会再初始化 CUDA
IPC。唯一复现入口为：

```bash
./run_g3v2_majorfix_8000.sh
```

验证产物：

- checkpoint 后缀：`outputs/episode_000000/g3v2_checkpoint619_third_majorfix_8000_*`
- 初始状态完整运行：`outputs/episode_000000/g3v2_full_majorfix_8000_*`

仍未处理的已知问题：完整流程 frame 420--439 的第二折目标/IPC 代理仍有明显
分叉，自动碰撞 Gate、第二折主体位移和叠层 Gate 因此前序问题继续为 `FAIL`；
第三折最终覆盖率约 `43.97%`，略低于 45% 阈值。它们不属于本次释放瞬移、
夹指冻结和第三折穿桌修复范围。

## 当前结论

13,767 面短袖、fast solver、原始 URDF 的全流程已经能够稳定完成三次接触抓取，
并在放手、机械臂撤离后继续观察 120 个仿真步。第一、二折可作为当前可靠基线；
第三折能够抓起并覆盖，但最终仍是偏厚的隆起布团，不是规整平铺的成品形态。

当前最佳完整运行是：

- 标签：`final_13767f_fast_tuned_full`
- 视频：`outputs/episode_000000/final_13767f_fast_tuned_full_multiview.mp4`
- 指标：`outputs/episode_000000/final_13767f_fast_tuned_full.json`
- 第三折前 checkpoint：
  `outputs/episode_000000/checkpoints/final_13767f_fast_tuned_full_pre_third.pkl`
- 三折接触抓取均为 `PASS`，没有夹空或中途滑脱判定。

## 当前默认参数

唯一完整启动入口为：

```bash
./run_13767_tuned_full.sh LABEL fast 0.100 0.060 0.006
```

主要参数：

- 第一折重叠修正：`-0.015 m`
- 第二折重叠：`0.100 m`
- 第二折落放松弛：`0.060`
- 第二折落放抬升：`0.006 m`
- 第三折抓取抬升：`0.080 m`
- 第三折向领口侧落点修正：`0.100 m`
- 第三折夹爪水平化比例：`0.800`
- 放手后张开保持 10 步、撤离 40 步、稳定观察 120 步。

## 最佳运行的定量结果

- 完整 946 帧耗时：`668.94 s`
- 平均墙钟速度：`1.414 FPS`
- 第三折静止半区中位平面位移：`36.93 mm`
- 第三折移动半区对静止半区的最终覆盖率：`55.28%`
- 帧 941 的 5%--95% 高度跨度：`99.54 mm`
- 帧 941 的 90% 平面包围面积：`0.06628 m²`

第三折自动判定仍为 `FAIL`，原因仅是静止半区位移超过当前严格阈值
`30 mm`；覆盖率已经高于 `45%` 阈值。这个判定用于筛选候选，不应被表述为
“整个任务完全失败”。

## 已拒绝的第三折候选

### `tune13767_third_flat_v3`

同时将抬升改为 `70 mm`、领口修正改为 `75 mm`、水平化改为 `0.9`。
虽然静止半区位移略降到 `33.06 mm`，覆盖率却降到 `44.16%`，高度跨度增到
`103.66 mm`，视觉上也没有更平，因此拒绝。

### `tune13767_third_level90_v4`

只把水平化比例从 `0.8` 改为 `0.9`，其余保持最佳配置：

- 静止半区位移：`40.64 mm`（变差）
- 覆盖率：`48.12%`（变差）
- 高度跨度：`103.20 mm`（变差）
- 平面包围面积：`0.06697 m²`

因此默认值继续保持 `0.8`，不采用 V4。

## 诊断闭环

每次候选运行都应依次执行：

1. 检查四视角关键帧中的抓取、抬升、跨越、落放和撤离。
2. 检查 `contact_grasp_summary`，排除夹空和滑脱。
3. 比较第三折静止区位移、覆盖率、高度跨度和平面面积。
4. 运行：

   ```bash
   ./diagnostics/run_diagnostics.sh outputs/episode_000000/LABEL.json
   ```

5. 只有视觉与指标同时改善才写回默认参数；否则保留为被拒绝实验。

13,767 面网格的 garment atlas 已生成在：

- `outputs/episode_000000/assets/short-shirt.garment_atlas.json`
- `outputs/episode_000000/assets/short-shirt.garment_atlas.npz`

实验索引会自动更新到：

- `outputs/episode_000000/experiment_index.csv`
- `outputs/episode_000000/experiment_index.md`

## 下一阶段建议

不再继续只调单个落点比例。第三折隆起主要发生在搬运末段（约帧
`840--900`），下一阶段应将“翻转”和“落放”拆成两个独立轨迹段：先在更高位置
完成翻面，再以近似竖直的小位移放下；必要时增加一只手对已折主体的轻压保持。
这样比继续加大水平化或把落点向领口深压更可能减少堆积，同时避免多层布料过度
重叠引发 IPC 接触刚化或数值伪影。
