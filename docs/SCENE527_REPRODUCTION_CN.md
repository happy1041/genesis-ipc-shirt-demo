# 55k Scene527：完整数据与运行入口

本包对应约1:13的**七段checkpoint接续版**，0–4372共4373帧、60fps。此前只上传代码和机器人命令，确实不足以复现折叠后的布态。现在将物理运行所需的55k衣服、双X5 URDF及全部引用网格、七个阶段的边界状态、命令轨迹、实际参数和原始replay集中到`reproduction/scene527_55k_73s/`。

这里的参考成片不是此前4148帧的另一版连续仿真。帧数和控制流程不应混用。

## 缺少的五项具体是什么

| 项目 | 包中内容与作用 |
| --- | --- |
| 55k衣服 | `assets/cloth/short-shirt-55068f.obj`，27,811顶点、55,068面。保留源网格字节及单位；runner按0.001缩放并应用初始姿态 |
| 双X5机器人与碰撞资产 | `assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf`及其16个唯一mesh引用，含机械臂CAD、夹片和腕掌IPC凸包 |
| 匹配的分阶段状态 | `checkpoints/*.tar.gz`，每个无损压缩包包含同名`.pkl`、`.ipc_state.npz`、`.meta.json`三件套 |
| 新版摩擦组合 | 普通布料摩擦1、桌面1、机器人2；布—布独立系数2；布—桌实际系数1；布—机器人实际系数√2。逐段参数保存在`bundle.json` |
| 4373帧控制流程 | `controls/trajectory.npz`是完整累计关节与开度命令；新运行入口按帧号加载状态并执行各段，无需Workbench |

`.pkl`保存Genesis场景/求解器状态，`.ipc_state.npz`保存布料位置与速度、IPC刚体仿射姿态与速度，`.meta.json`保存帧号、时间、上一帧关节姿态及配置/命令前缀签名。机器人轨迹只说明“夹爪怎么动”，不能代替已经折好的衣服状态。

前六段布—桌系数由`sqrt(1×1)=1`得到，没有独立override；末段显式设为1，并保留源命令中的`--allow-material-change-on-checkpoint`以允许签名`None→1`。不要把七段全部改为显式override再忽略签名错误。

## 阶段表

| 阶段 | 执行帧 | 输入历史边界帧 | 输出边界帧 |
| --- | --- | --- | --- |
| `first_fold` | 0–739 | 从平铺初态开始 | 739 |
| `first_pull` | 740–1135 | 739 | 1135 |
| `second_fold` | 1136–1939 | 1135 | 1939 |
| `third_right` | 1940–2614 | 1939 | 2614 |
| `third_left` | 2615–3292 | 2614 | 3292 |
| `sweep` | 3293–3592 | 3292 | 3592 |
| `press_slide_home` | 3593–4372 | 3592 | 4372 |

每段都使用同一完整累计轨迹，`--frames`指定累计终点+1。加载器从checkpoint索引+1继续；切成从0重新编号的短轨迹会破坏校验。已核对七个checkpoint的命令前缀与该累计轨迹一致。

## 文件布局与大文件下载

```text
reproduction/scene527_55k_73s/
  bundle.json                 相对路径、逐文件SHA256、源参数、阶段与签名
  configs/source_preset.args  原运行基础预设
  controls/trajectory.npz     4373帧机器人命令
  assets/                    衣服、原始URDF和局部凸包
  transport/                 仅URDF实际引用的其他网格，保留相对目录结构
  checkpoints/               七个无损压缩的状态三件套
  reference_replays/         七段原始物理回放，可直接离线渲染
  source_records/            原参数/plan/manifest、物理指标及TCP
  reference/hero.mp4          原速、相机拉近1/3的Isaac参考成片
  isaac/                     离线renderer/controller、3层USD场景及纹理压缩包
```

大文件使用Git LFS。首次下载需：

```bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --branch handoff/scene527-ipc-demo-20261005 \
  https://github.com/happy1041/genesis-ipc-shirt-demo.git
cd genesis-ipc-shirt-demo
git lfs pull
```

目前完整包位于这个交付分支，默认`main`仍是历史版本。已有克隆应先切换到该分支。约1.43GiB的大文件均需实际下载。[数据与原结果索引](../reproduction/scene527_55k_73s/INDEX.md)提供每段checkpoint、replay、plan、指标、TCP及原物理/渲染日志的直接入口。

如果获得的是网页下载的zip或LFS指针文本，应改用Git克隆并执行`git lfs pull`。`bundle.json`与验证入口会检查实际文件大小和哈希，不会把指针文本当作模型或checkpoint。

## 环境

使用Linux/NVIDIA GPU、Python3.12、`pyuipc==0.0.25`、Genesis官方`8b1dba2`加仓库[当前补丁](../patches/genesis-world-8b1dba2-current.patch)。本机记录环境为Torch2.8.0+cu128、Genesis1.3.2、Quadrants1.2.0；包版本快照见`requirements/`。根据目标GPU安装适合的Torch/CUDA轮子，再检查补丁结果哈希和运行环境。

独立Genesis环境与Isaac环境仍是软件依赖；不要把本机`envs/`复制到Git。物理入口提供完整OBJ/URDF/命令路径，不需要另找整套SIM1资产。运行参数中的`--sim1-root`指向包根目录，仅兼容原runner接口。

```bash
export GENESIS_PYTHON=/path/to/genesis-env/bin/python
./scripts/run_scene527_reproduction.sh --verify
```

验证是CPU文件检查，不启动物理仿真。它检查所有文件、URDF网格引用、checkpoint压缩包内容、命令前缀和回放帧号。

## 查看原结果与重新运行

直接查看`reference/hero.mp4`可看到本次要复现的成片。原物理状态可以通过Genesis离线重渲染；例如只渲染第二折：

```bash
./scripts/run_scene527_reproduction.sh --mode replay --stage second_fold
```

下面从原第二折输入checkpoint做短恢复检查，仅推进两帧：

```bash
./scripts/run_scene527_reproduction.sh \
  --mode reference-boundaries --stage second_fold --smoke-frames 2 --physics-only
```

`--mode reference-boundaries`逐段加载本包中的原历史边界状态，适合单段复验。去掉`--stage`或设置`--stage all`可运行七段；各段物理结束后默认用保存状态生成Genesis六合一。`--physics-only`跳过视频，`--headless`显式关闭前台viewer，`--dry-run`只打印命令。

若要从头重做整条checkpoint链，下一段使用本次新生成的上一段状态：

```bash
./scripts/run_scene527_reproduction.sh --mode chain --physics-only
```

若要验证单进程从初态连续运动，不加载分段状态：

```bash
./scripts/run_scene527_reproduction.sh --mode continuous --physics-only
```

后二者是新的动力学运行，不能把它们叫作原七段布态的逐帧复现。当前状态格式没有证明保存了全部原生求解器接触缓存，跨机器也存在浮点数差异；应分别比较原replay、边界状态和新仿真。

输出进入新的`outputs/scene527_reproduction/<时间模式>/`目录。`RUN.json`记录实际命令；每段保存物理指标、TCP、replay、checkpoint以及按选项生成的Genesis视频。入口不会读取仓库`.env`中的旧资产路径。

## 保持资产与签名一致

URDF字节哈希属于checkpoint签名，不能随意改写其中的mesh路径。本包保留原URDF原文，并重建其相对`transport/`路径；不需要篡改checkpoint元数据。衣服/轨迹主路径迁移通过`--allow-checkpoint-path-relocation`允许，内容哈希仍由包验证。

全部checkpoint都是源文件的无损副本；源记录中的本机绝对路径只用于追溯，运行入口使用`bundle.json`的相对路径。不要把其他模型拓扑、后续69秒版的轨迹或其他摩擦试验的checkpoint放进这条链。

## Isaac hero重渲染

除参考成片外，本包也包含对应的三层USD场景、完整静态场景/纹理压缩包、4373帧机器人视觉变换和离线renderer/controller。CPU检查确认场景的442个外部资产都位于冻结scene目录中；`OmniPBR.mdl`由Isaac运行环境提供。准备工具只重写两个USD sublayer路径，保持相机、材质、照明与保存状态。

```bash
"$GENESIS_PYTHON" tools/prepare_scene527_isaac.py \
  --isaac-python /path/to/isaac5-env/bin/python \
  --job outputs/scene527_isaac_hero
```

这一步在CPU上解包场景、合并七段布料状态并生成本机渲染配置，约需额外数GiB临时磁盘空间；不推进物理、不启动Isaac GPU渲染。成功后按打印的controller命令渲染，或准备命令追加`--run`。需要Isaac Sim5.0环境，以及CPU环境中的NumPy/PyAV、Pillow、ffmpeg（或imageio-ffmpeg），NVIDIA驱动与`nvidia-smi`。

输出为普通RaytracedLighting、带阴影、1080p60fps单hero，相机沿原视线拉近1/3。Isaac用于回放，不负责布料物理。新机器GPU兼容、渲染效果和运行耗时需另外验收。

## 本包已经做过的验证

[验证摘要](../reproduction/scene527_55k_73s/VALIDATION.json)记录范围：全部文件和URDF引用核对、七套状态及命令前缀核对、4项CPU测试和语法检查均通过。已在迁移后的包路径下用原1135帧状态执行1136–1137帧、原3592帧状态执行3593–3594帧；两次前台GPU短跑均正常退出、状态有限并保存新checkpoint。

Isaac准备工具实际生成了独立job；合并后的`cloth_pos`/`source_frames`与原hero输入完全相同，机器人视觉数据和相机配置保持相同。CPU USD依赖检查确认全部3层/442个资产位于新job中，controller配置检查通过。本次没有重跑完整4373帧物理，也没有重新执行整片Isaac GPU渲染。

Git LFS实际上传完成后，在独立目录通过HTTPS重新克隆交付分支，不使用本机SSH认证配置；抽样下载55k衣服与1135帧checkpoint，文件大小、SHA256和checkpoint三件套成员哈希均与原输入一致。该检查没有重复下载全部大文件。
