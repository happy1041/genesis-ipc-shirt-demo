# 布料观感与 IPC 数值问题交接（RTX 5060 Laptop）

## 当前可复现稳定版本

运行：

```bash
RUN_LABEL=my_fabric_run ./run_g3v2_fabric_render_5060m.sh
```

已完成的参考产物是 `g3v2_cloth_t01_fabric_render_v2_8000`：980/980 帧、四路单视角、multiview、JSON 和关键帧均完整生成。物理参数为 shell radius 0.1 mm、dHat 2 mm、E 20 kPa、cloth/table/robot friction 1/1/2、substeps 2。录制时机器人灰色 visual 使用 IPC actual affine-body pose。

注意：长运行不要依赖无人读取的交互 PTY。推荐将 stdout/stderr 重定向到日志，并用 `setsid` 后台运行，否则高频 Genesis FPS 日志可能填满缓冲区。

## 已确认的观感原因

- `contact_d_hat=2 mm` 主导桌面和叠层的可见空气间隔；渲染器并没有按 thickness 挤出表面。
- 8000-face QEM 网格边长不均匀，最长边约 60 mm，只能形成宽大折脊，缺少细褶自由度。
- 当前是各向同性 strain-limiting shell，不表达针织物经纬方向性和弯曲迟滞。
- 高 cloth/self friction 能稳定抓取，但会锁住层间滑移和鼓包。
- 纯色、全局平滑法线和原 ambient=0.45 会强化橡胶观感。当前代码已将 ambient 调为 0.26，并对 cloth 设置 roughness=1、normal_diff_clamp=42°。

## 失败实验与上游阻塞

以下实验均保持或尝试保持三折轨迹，但不能作为候选：

1. `t=.1 mm, dHat=1 mm, E=100k, cloth/table friction=.4/.6`：五次抓取全部失败。
2. `t=.1 mm, dHat=2 mm, E=40k, friction=.4/.6/5, substeps=2`：约 frame 482，CUDA CCD 报极小负 TOI 后断言退出。
3. 同上 `substeps=4`：约 frame 326，`cudaErrorMemoryAllocation`；5060M 8 GB 不适合该配置。
4. `E=30k, friction=.5/.7/4`：约 frame 325，line-search energy 变 NaN。
5. `E=20k, friction=.7/.8/2.857`：约第一折后段，CCD 报 `TOI=-3.4e-12` 并断言退出。

微负 TOI 属于浮点舍入量级，但 libuipc 当前 CUDA backend 在 `global_trajectory_filter.cu:95` 直接断言。若继续物理改进，优先在隔离分支复现并评估上游修复（容差/夹紧），不要在 demo 主分支反复扩大参数矩阵。

## 稳定版本的验收结果

- 完整运行且无 CUDA 错误。
- 5 次抓取中 4 次 PASS；第一次右手 `follow_projection=0.7395`，略低于约 0.75 门槛，但方向余弦 0.99986，不是完全掉抓。
- 第三折覆盖率 50.62% 通过；静止上片中位位移 31.84 mm，略高于 30 mm 门槛，因此严格 verdict 为 FAIL。
- 最终高度 P50/P90/P99 为 26.2/68.2/103.1 mm。渲染改动改善了折痕可读性，但没有改变物理蓬松度。

## 推荐后续顺序

1. 先给第一次右手抓取增加少量鲁棒余量，并重复至少 3 个相同 seed/run，避免把边缘随机分支当成功。
2. 在 libuipc 隔离复现中处理微负 TOI；修复前不要再做全流程低摩擦 sweep。
3. 物理稳定后测试均匀 remesh，而不是当前 QEM 8k 网格；5060M 上先做 10k–14k，监控峰值显存。
4. 最后增加织物 albedo/normal 纹理。纹理只改善材质读感，不能代替接触和网格修复。

## 2026-08-21：13,767 面稳定三折基线

现有两套 RTX 5060 Laptop 可复现配置：

```bash
# 快速功能基线：dHat 2 mm
./run_g3v2_13767_dhat2_soft_5060m.sh

# 高质量基线：dHat 1.5 mm，必须使用完整求解器
./run_g3v2_13767_dhat15_slow_5060m.sh
```

两者都使用 13,767 面原始模型、0.1 mm shell radius、2 substeps、6 点 soft virtual grasp、strength 20，并完整完成五次抓取和三折释放。

- dHat 2 mm fast：980/980，无 CUDA 错误；第二折 layering PASS，第三折覆盖 62.8%，但静止上片中位移动 33.98 mm，严格 third motion FAIL。
- dHat 1.5 mm slow：980/980，无 TOI/NaN/CUDA；第二折 layering PASS，第三折 motion PASS。最终高度 P50/P90/P99 为 23.6/55.2/71.1 mm，相比 2 mm 的 26.3/61.6/81.2 mm 更薄。
- dHat 1.5 mm 若使用 fast solver，会在第二折约 frame 484 触发 `TOI=-1.8e-12` 断言。完整求解器（Newton 50、line search 8、linear tol 1e-3）能稳定跨过，并保持 frame480 左右抓取误差低于约 10 mm。
- 软抓取避免了高分辨率模型首次左手纯摩擦抓取失败；代价是局部会出现较平滑的张力膜，但没有单点飞布或释放后继续拖拽。

### 首次夹空修复与 dHat 1 mm 试验

原 soft grasp 会永久保留 attach 瞬间的布片相对偏移；首次左右选点离 TCP 约 19--36 mm，因此布虽随动，画面仍像隔空抓取。正式 1.5 mm 脚本现增加：

- `--soft-grasp-first-capture-frames 20`：在约 0.33 秒内将首次所夹布片质心平滑导入 TCP 夹取中心，同时保留布片形状。
- `--first-grasp-right-depth 0.003`：首次闭合窗口右夹爪整体下降 3 mm，消除右侧残余 3--6 mm 可见空气缝。

最终 `g3v2_13767f_dhat15_capture20_rightdepth3_slow_full` 完成980帧、五抓五放且无CUDA错误；视觉上左右布片均从指缝连续伸出，无明显穿模。耗时1684.10秒（28.07分钟、0.582 source FPS）。最终相对桌面高度P50/P90/P99/max为25.95/62.07/88.14/109.50 mm；这属于折后高度包络，不是材料厚度，材料shell radius仍为0.1 mm。

dHat 1.0 mm 的slow/capture20/right-depth3试验：第一折220帧通过；扩展到第二折时在frame451、右手第二次attach后触发`cudaErrorMemoryAllocation`。这不是TOI/NaN或掉抓，而是RTX 5060 Laptop 8 GB上的接触/求解工作集溢出。因此1 mm暂不提供完整运行脚本，需先优化显存。

## 2026-08-22：RTX 5070 Ti、dHat 1 mm 与可靠离线录制

RTX 5070 Ti 16 GB 能跨过原先 8 GB GPU 的 frame451 显存失败，但直接沿用腕部刚性旋转仍会在第二、第三折产生约15 cm的软约束长力臂。当前修复为：

- 第二折两手仅传递 TCP 平移，保留 attach 时的世界坐标片内偏移；
- 第三折右手仍传递旋转，但布片绕 `TCP_LOCAL` 而不是腕部/link origin 旋转；
- 首次抓取继续使用 capture20 与右手下探3 mm，保持视觉上布片位于指缝内。

最终 checkpoint 连续物理运行 `g3v2_13767f_dhat10_tcp_rotation_finalphysics` 完成 source frame 584--979：frame780/840/900 soft-grasp mean error分别3.8/4.3/4.8 mm，最终覆盖率67.06%，五次抓取全部发生并释放。静止上片累计中位位移30.65 mm，仅比严格30 mm门槛高0.65 mm，因此自动 verdict 仍为FAIL，但三折功能与视觉均已完成。

实时四视角录制会在750--780段改变CUDA调度并放大软约束数值分支：同配置无录制frame780误差约3--4 mm，实时录制可升至43--58 mm并锁死。正式录制因此改成两阶段：第一遍`--no-record --dump-replay-states`保存每帧布料、robot q和IPC实际affine transform；第二遍`--replay-states`只渲染、不推进物理。推荐入口：

```bash
./run_g3v2_13767_dhat10_offline_record_5070ti.sh
```

本次最终四视角视频严格为980帧、60 FPS、16.333秒。后缀物理396帧耗时355.02秒（1.115 source FPS）；此前连续四视角实时跑完整段为25.36分钟/0.644 source FPS，而按最终视频实际采用的checkpoint前缀与后缀计，物理生成约19分钟。纯渲染396帧四视角耗时8.68秒（场景初始化另计）。最终相对桌面高度包络P50/P90/P95/P99/max为20.05/52.71/59.20/74.19/96.82 mm；在同时含上下折片的10 mm XY网格中，局部stack span的P50/P90约54.64/71.28 mm。材料输入仍是0.1 mm shell radius；这些折后高度/局部span不是材料厚度。
