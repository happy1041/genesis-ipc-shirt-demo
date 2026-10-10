# Workbench：选点、plan、IK 与轨迹导出

仓库提供浏览器Workbench、动作插值、双X5逆运动学和运行包导出。抓放点由人在实际布态上选择，再设置夹爪位置、姿态、开度和动作时长；生成轨迹后用物理仿真检验夹持。

## 1. 准备与打开Workbench

先按[复现指南](SCENE527_REPRODUCTION_CN.md)配置Genesis/libuipc并执行 `git lfs pull`。在该Python环境安装 `numpy scipy trimesh opencv-python pillow numba rtree`。以下命令均在仓库根目录执行，`python` 指已配置的Genesis解释器。

```bash
python tools/prepare_workbench.py --after-stage 03_second_fold --output outputs/edit_third
python tools/trajectory_workbench/server.py --config outputs/edit_third/workbench.json --port 8766
```

浏览器访问 `http://127.0.0.1:8766`。这会导入第二折参考回放、匹配55k atlas、TCP和1939帧checkpoint，适合规划第三折第一段。`--after-stage` 可选择下面七段之一。输出目录必须是新目录。

准备工具校验并解包checkpoint，建立匹配的轨迹前缀和本地绝对路径配置。CPU预览直接读取布料回放，不依赖重新渲染视频；夹取表面点不是TCP本身，需考虑夹片偏移和姿态。续跑动作从checkpoint末帧开始。

## 2. 已有plan示例

|阶段|plan|
|---|---|
|第一折|[01_first_fold](../reproduction/scene527_55k_73s/source_records/01_first_fold/plan.json)|
|回拉|[02_first_pull](../reproduction/scene527_55k_73s/source_records/02_first_pull/plan.json)|
|第二折|[03_second_fold](../reproduction/scene527_55k_73s/source_records/03_second_fold/plan.json)|
|第三折右手|[04_third_right](../reproduction/scene527_55k_73s/source_records/04_third_right/plan.json)|
|第三折左手|[05_third_left](../reproduction/scene527_55k_73s/source_records/05_third_left/plan.json)|
|扫平|[06_sweep](../reproduction/scene527_55k_73s/source_records/06_sweep/plan.json)|
|下压、拖动、复原|[07_press_slide_home](../reproduction/scene527_55k_73s/source_records/07_press_slide_home/plan.json)|

`start_frame/reference_frame` 指初态源帧；`actions` 按顺序定义动作。`duration_frames` 是时长；`pos` 是世界坐标TCP位置（米），`quat` 为WXYZ，`opening` 是每指开度（米）。原plan保留完整字段，可复制后编辑；历史plan的起始帧应与导入checkpoint对应。

## 3. plan → 插值 → IK

```bash
python tools/compile_plan.py --config outputs/edit_third/workbench.json \
  --plan reproduction/scene527_55k_73s/source_records/04_third_right/plan.json --ik
```

终端输出新动作目录 `outputs/edit_third/action_plans/<时间>/`，包含 `plan.json`、`compiled.json`、`seed.npz` 和 `ik/result.json`。通过检查才输出 `ik/robot_q.npz`；被拒绝的轨迹只用于诊断。

- [action_plan.py](../tools/trajectory_workbench/action_plan.py)：分段位置插值、SLERP姿态插值、弧形/端点速度和开闭爪事件，检查速度/加速度。
- [action_plan_ik.py](../tools/trajectory_workbench/action_plan_ik.py)：用URDF、TCP偏移和19关节seed逐帧求IK；依赖同目录 `ik_preflight.py`。
- [action_service.py](../tools/trajectory_workbench/action_service.py)：Workbench保存/编译/IK/导出的HTTP服务。

IK结果包含 `robot_q`、`source_frames`、`joint_names`、`accepted`。这是机械臂指令，不包含衣服运动。初始seed取自checkpoint，避免起点跳变；IK合法不等于布料抓取成功。

## 4. 导出可运行轨迹

将下面 `<时间>` 替换成上一条命令输出的动作目录名：

```bash
python tools/trajectory_workbench/action_plan_runtime.py export \
  --config outputs/edit_third/workbench.json \
  --joints outputs/edit_third/action_plans/<时间>/ik/robot_q.npz \
  --output outputs/edit_third/run_bundle
```

运行包会校验历史指令前缀、起始关节和checkpoint，拼入新阶段，输出仿真用 `joint_q/openness` 轨迹、manifest及 `run.sh`。设置复现环境中的 `SIM1_ROOT`（本包reproduction目录）、`GENESIS_PYTHON` 等后执行 `bash outputs/edit_third/run_bundle/run.sh`。导出动作本身不会启动物理。

完成新阶段后，可用 `action_plan_runtime.py import --config ... --manifest ... --output ...` 导入新末态，继续下一阶段。具体产物以运行包manifest为准。

## 范围与数据约束

示例使用随包55k资产、回放、atlas和checkpoint。换拓扑参考[面数切换指南](CLOTH_MESH_SWITCH_CN.md)，须生成匹配atlas/状态。Workbench为研究调试界面，人工选点和参数化动作需逐阶段验收。原七段plan是历史来源记录，不保证用新版工具重新生成后与冻结4373帧轨迹逐字节相同。
