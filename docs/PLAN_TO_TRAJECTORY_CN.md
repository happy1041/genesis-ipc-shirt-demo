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

## Workbench界面怎么用

### 查看当前衣服

顶部 `Source frame` 是源轨迹帧号，可直接输入；`−1/+1` 和 `−10/+10` 用于逐帧查看。准备工具默认停在所选阶段的末帧，例如第二折末态1939。选点前先确认帧号与衣服形状，避免把运动中途的位置当作下一段初态。选点会保持当前帧；`初始衣服` 是主动切换查看初态的按钮。

`高级模式` 展开时间线、速度图和材料点信息。CPU预览默认只显示布料，机械臂未显示不代表仿真缺少机器人。坐标读数以世界坐标为准，不要把屏幕上下左右当作XYZ方向。

### 选择抓点和落点

1. 点击右侧 `左手` 或 `右手`，确定正在编辑哪只手。
2. 在 `点击模式` 选择 `选抓点`，点击布料。右侧列出命中的布层候选及面编号、Z高度；点击候选后，再点 `确认当前层为本手抓点`。
3. 改为 `选落点`，点击目标位置，或填写落点XYZ（界面单位mm），点击 `确认本手落点`。空白处按桌面Z=800mm投影，所需空中释放高度应显式填写。
4. 切换另一只手重复操作，最后点击抓放点保存按钮。可以仅保存已调整的点，不必强制凑齐四点。

页面状态栏显示保存的JSON路径，通常在工作目录的 `edits/` 中。抓点记录布料表面位置、材料面及选中帧；落点记录世界坐标参考。**保存点位不会自动生成新动作轨迹**：需要把点位用于修改plan的TCP目标，补上夹片偏移、抓取深度、姿态、开闭时序，再执行第3节的编译和IK。

同一个屏幕位置可能命中多层布料。候选中的 `surface_layer` 是材料标签，不应单独作为当前第几层的判断；结合候选Z与实际布态选择。表面点也不等于夹爪TCP，直接照抄其Z可能导致漏抓或碰桌。

### 修改已有轨迹节点

在配置提供可编辑节点时，先选手，再选右侧 `语义轨迹节点`，拖动画面圆点，或填写 `ΔXYZ` / `World XYZ` 后点击对应应用按钮。界面单位是mm，plan内部单位是m。`重置本手` 清除该手的当前节点修改。

自动准备的参考配置将已完成阶段的起止节点锁定，因此不能拖动这些节点；它用于查看旧结果、选下一段点位。要重新规划，复制下方合适的plan、修改 `actions`，通过命令行生成新轨迹。节点编辑模式的 `保存草案` 输出修改记录，`运行本手顺序IK预检` 检查该节点方案；完整双手动作plan使用第3节的入口。

PATH红色通常表示位移/时长导致速度或加速度超限；可延长动作或调整路径。IK拒绝时检查 `result.json` / 日志中的位置、姿态、限位与碰撞信息，修改目标后生成新的动作目录。不同模式的保存文件用途不同，按状态栏给出的路径取文件。

### 推荐的一次修改流程

打开上一段末态 → 选择并保存抓放点 → 复制下一段plan并修改动作 → `compile_plan.py --ik` → 查看PATH/IK报告 → 导出run_bundle → 跑物理 → 导入新末态继续。

如运行在远程机器，服务默认只监听localhost，可使用SSH端口转发：`ssh -L 8766:127.0.0.1:8766 用户@服务器`，再在自己电脑打开 `http://127.0.0.1:8766`。关闭服务可在启动终端按Ctrl+C；已保存文件保留，未保存的页面编辑应先导出。

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
