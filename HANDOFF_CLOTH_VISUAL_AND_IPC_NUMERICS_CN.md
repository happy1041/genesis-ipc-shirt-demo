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
