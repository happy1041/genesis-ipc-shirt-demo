# 自动诊断与修复闭环

## 目标

诊断系统回答三个问题：这次运行是否可复现、失败最早发生在哪一折、下一次只应
修改哪一个参数族。它不会用“日志里出现 PASS”替代视觉确认，也不会把不同网格、
不同初态或不同轨迹的结果混成一次迭代。

## 一条命令

```bash
./diagnostics/run_diagnostics.sh outputs/episode_000000/<run>.json
```

该命令不启动仿真，会生成：

- `<run>.provenance.json`：代码/资产哈希、仓库状态、Python 包和 GPU；
- `<run>.semantic_state_analysis.json`：衣服局部区域、两表面关键点、逐帧位移和叠层投影；
- `<run>_evaluation/evaluation.{json,md}`：Gate 状态；
- `evaluation.json.next_action`：最早未批准 Gate 的唯一下一目标、允许改动和被冻结的下游；
- `<run>_evaluation/manual_review.json`：必须填写的人工四视角检查表；
- `experiment_index.{csv,md}`：所有运行的统一索引。

## 衣服局部语义

首次使用某个网格时生成 atlas：

```bash
/home/happy1041/work/genesis-ipc-env/bin/python \
  diagnostics/build_garment_atlas.py \
  outputs/episode_000000/assets/short-shirt-8000f.obj
```

atlas 使用原 OBJ 坐标记录：

- raw Z 最小/最大：腰摆/领口；
- raw X：宽度负侧/中部/正侧；
- raw Y 经 Genesis `Rx(-90°)` 后：朝上/朝桌面的两层；
- 腰摆中点、领口中点和两袖端等成对表面关键点。

穿着者左/右的 raw-X 符号必须从资产一次性校准，不能看某个相机画面猜测。

## 每轮实验的状态机

1. 选择尚未 PASS 的最早 Gate；后面的折暂不调。
2. 写出一个可证伪假设，例如“第二折右手进场擦到第一折折片”。
3. 只改一个参数族，例如净空轨迹；网格固定为 8000 面。
4. 从已批准 checkpoint 跑最短后缀，自动保存所有强制关键帧。
5. 先看四视角接触表，再看接触/位移/叠层数值；任何一项缺失均为 INCOMPLETE。
6. 候选通过人工复核才标 PASS，并创建下一 Gate checkpoint。
7. 8000 面三段均通过后，才运行约 14000 面完整三折与重复性验证。

评价器按最早未批准 Gate 自动路由下一轮目标。例如关键帧缺失时只允许补录证据，
接触失败时只允许调抓取，第二折主体位移失败时只允许调第二折松弛/净空/弧线。
`blocked_downstream_gates` 中的折在当前 Gate 通过前不得继续调参。

推荐命名：`g<gate>_<single-hypothesis>_<mesh>f_seed<seed>_<date>`。

## 自动指标的边界

- RGB 蓝色掩码只发现明显漂移或塌缩，机器人遮挡时不能证明抓层正确；
- atlas 投影重叠能发现“两个折片并排而未叠层”，阈值需由一条人工认可结果标定；
- 表面关键点距离能提示正/背层分离，但袖口、领口等开放边界仍需看近景；
- 2500 面只做启动/checkpoint 冒烟测试，不用于抓取和折叠结论。

## Checkpoint 规则

新 checkpoint 会额外携带 frame 0 的诊断参考布料状态；加载后会立即保存 checkpoint
源帧的状态和关键帧，因此后缀运行不再因为缺 frame 0/332/583 而得到空指标。旧
checkpoint 仍可加载，但缺少上述参考时应被标记为 INCOMPLETE，不能据此批准 Gate。
