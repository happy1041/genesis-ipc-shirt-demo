# Genesis IPC 叠衣 Demo 仓库交接（2026-08-24）

## 当前可交付结论

当前最佳主线是 **13,767 面短袖、dHat=1.5 mm、slow solver、全程纯摩擦**。
正式入口为：

```bash
./run_g3v2_13767_dhat15_pure_friction.sh
```

该入口明确设置 `USE_SOFT_VIRTUAL_GRASP=0`，先在无录制模式下保存980帧物理
状态，再离线渲染四视角视频，避免GPU录制改变IPC求解分支。

最近一次验证结果：

- 980/980帧完成，无CUDA、TOI或NaN错误；
- 物理耗时1110.61秒（18分31秒），平均0.882 FPS；
- 第一折和第二折四次实体摩擦抓取均通过；
- 第三折实际夹起、翻面、释放并通过最终覆盖护栏，但夹点持续抽滑约90 mm，
  严格抓取事件判定为FAIL；
- 第二折层叠覆盖率77.23%，层叠PASS；第二折边缘p90位移132.86 mm，
  motion gate FAIL；
- 第三折最终覆盖率46.49%（阈值45%），余量仅1.49个百分点；
- 最终离桌高度中位21.65 mm、p90 49.66 mm；10 mm网格局部叠层包络
  中位53.47 mm、p90 64.96 mm；
- 全程没有soft virtual grasp、隔空牵引或明显视觉穿模。

因此可以汇报为“全程纯摩擦的三折功能Demo已经完成”，但不能汇报为
“五次抓取全部严格通过”或“最终形状已经鲁棒收敛”。

## Clone后的外部依赖

本仓库不复制Genesis和SIM1源码，也不提交9.8 GB运行输出。复现时需要：

- Genesis World commit `8b1dba2`；
- 对该commit应用本仓库补丁：

  ```bash
  git -C /path/to/genesis-world checkout 8b1dba2
  git -C /path/to/genesis-world apply \
    /path/to/genesis_ipc_shirt_demo/patches/genesis-world-8b1dba2-local.patch
  ```

- SIM1 commit `973476b`及episode 000000轨迹/短袖/Acone资产；本机SIM1工作树很脏，
  交接时必须从干净commit和明确的数据包重新验证，不能复制整个当前目录；
- 通过环境变量显式提供路径，例如：

  ```bash
  SIM1_ROOT=/path/to/SIM1 \
  GENESIS_ENV=/path/to/genesis-ipc-env \
  GENESIS_GPU=<optional-gpu-uuid> \
    ./run_g3v2_13767_dhat15_pure_friction.sh
  ```

项目输出、视频、checkpoint和replay states都位于 `outputs/`，默认不进入Git。
正式上传前还需要决定是否单独发布一个小型“已验证结果包”，并附SHA-256清单。

## 参数消融结论

| 配置 | 结论 |
|---|---|
| dHat=2.0 mm | 稳定，但多层接触空气垫明显，成品更厚、更像充气橡胶 |
| dHat=1.5 mm | 当前稳定主线；比旧2 mm结果的最终高度/局部包络低约17--21% |
| dHat=1.0 mm slow | 纯摩擦长程在frame251出现`cudaMallocAsync`峰值分配失败 |
| dHat=1.0 mm fast | 在frame181触发约`-3.7e-13`的微负TOI断言 |
| soft virtual grasp | 能提高完成率，但会隔空牵引，已从正式主线移除 |

dHat是接触势能开始介入的距离，不是材料厚度。1.5→1.0 mm在同配置静置
状态中将最低层间位置降低约0.5 mm、中位高度降低约1.0 mm；完整折叠后的
厚度还由落点、覆盖、抽滑、摩擦锁定和沉降共同决定。

## 当前主要问题

1. **第三折抽滑。** 第三抓并非夹空，但690→941帧patch-TCP分离增长约90 mm。
2. **第二折边缘拖动。** 中央主体中位位移合格，p90由边缘/翻折区域拉高至
   133 mm，需要改进评价语义或轨迹。
3. **成品仍厚。** 局部叠层中位约53 mm；当前最终释放后没有额外自由沉降。
4. **dHat=1 mm不稳定。** 既有微负TOI，也有接触工作区显存峰值。
5. **结果尚未做重复性验证。** 目前是单seed、单次完整验证。
6. **项目依赖本地Genesis修改。** upstream Genesis 8b1dba2不能直接运行本Demo；
   必须应用 `patches/genesis-world-8b1dba2-local.patch`。
7. **SIM1数据不可直接放进仓库。** 轨迹和原始资产来自外部SIM1目录，需要确认
   许可证、下载方式和是否允许重新分发。
8. **旧文档包含历史结论。** README和路线图保留了调试过程，引用结果时应以本文件
   和最新JSON为准。

## 建议向学长学姐确认的问题

1. Demo验收是否要求“五次抓取全部严格随TCP”，还是最终三折成功即可？第三折
   90 mm抽滑是否必须优先修复？
2. 规定运动机器人边界（direct qpos replay）+实体摩擦抓取是否可接受，还是必须
   使用PD/双向动力学机器人？
3. Genesis/libuIPC是否有正式的kinematic affine-body接口，能避免当前
   `SoftTransformConstraint`造成IPC代理与Genesis FK的有限偏差？
4. 是否允许针对不同接触对分别设置dHat与摩擦：布-布/布-桌较低，布-夹爪较高？
5. 对约`1e-13`微负TOI断言，上游建议是数值容差、时间步、CCD修复，还是只能
   增大dHat？`cudaMallocAsync`峰值是否有接触候选/allocator诊断接口？
6. 当前壳厚参数在libuIPC中是radius，0.1 mm对应约0.2 mm总壳厚；项目报告是否
   采用这一口径？目标T恤克重和材料标定值是什么？
7. 当前各向同性Baraff-Witkin壳是否足够，还是需要针织物各向异性/塑性弯曲？
8. 是否有人工认可的目标折后状态或SIM1成功轨迹，可用来标定覆盖率、厚度和
   stationary-region阈值？
9. 仓库应采用哪种方式管理Genesis修改：独立fork、submodule固定commit，还是
   当前的patch文件？
10. SIM1轨迹、短袖网格、Acone URDF和输出视频哪些可以上传到公开仓库？仓库应
    放个人账号、课题组组织还是校内GitLab？

## 建议的后续顺序

1. 第三抓采用纯物理的“轻触/扫褶/闭合”或小幅0.25--0.5 mm overclose，先消除
   抽滑，不恢复soft grasp；
2. 最终释放后增加30--60帧自由沉降，复测厚度与卷边；
3. 如需更自然的布感，再分别测试bending 10→8、table friction 1→0.8；
4. 最后做至少3次重复运行和matched dHat 1.5/2.0 A/B；
5. 渲染侧独立增加低幅织物normal/roughness纹理，不与物理改参混在同一实验。

## 汇报时可以直接使用的表述

> 我把Genesis IPC叠衣Demo整理成了可交接Git仓库。目前13,767面、dHat=1.5 mm、
> slow solver、全程实体摩擦的版本可以完成三折，980帧物理约18.5分钟，最终
> 四视角视频已生成。相较旧2 mm版本，最终高度和局部叠层包络下降约17--21%。
> soft virtual grasp已从正式主线移除，第一、第二折四次摩擦抓取均通过；第三折
> 能完成翻面和释放，但存在约90 mm持续抽滑，最终覆盖率46.5%且余量较小。
> 当前希望确认验收标准、Genesis本地修改的仓库管理方式、dHat/摩擦能否按接触对
> 配置，以及SIM1资产和视频的发布权限。
