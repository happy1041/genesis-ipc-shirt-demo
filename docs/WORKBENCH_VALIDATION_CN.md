# Workbench交付检查（2026-10-10）

## 微调交接补充检查

- 页面 `/api/run-ik` 实际HTTP调用返回200及IK报告：1939末态参考段右手804帧预检完成。修复启动脚本接收页面环境变量，并默认使用服务的Python解释器。报告仍保留原轨迹离桌距离等诊断提示。
- 新增选点转换工具；测试覆盖偏移单位、只改落点保留抓取、跨帧拒绝、重复绑定拒绝和原plan不变。合计24项测试通过。
- 新结果Isaac准备已用实际第二折804帧测试：生成的布料数组与输入完全一致，机器人视觉数组按匹配排列后完全一致，源帧1136–1939完整保存。此项为CPU准备验证，未把旧参考回放当新结果。
- 8k/13k样本各60帧、拓扑分别4090/8000与7021/13767，哈希、TCP帧数、Workbench绘图及射线选点通过；原55k准备入口回归通过。
- 可用 `python tools/validate_workbench_roundtrip.py --output outputs/short_validation --physics` 显式执行三帧微调、IK、导出、checkpoint续跑及Genesis回放检查；默认不加 `--physics` 时只做CPU准备与IK导出。
- 三帧新物理结果已额外导出机器人视觉状态，并实际渲染Isaac hero（1920×1080/60fps/3帧）；完整解码与抽帧可见性检查通过。随后新checkpoint导入Workbench、1942末帧选点/预览通过；导入时同步更新选点绑定帧，避免仍指向旧1939帧。

- 使用随包03_second_fold的1939 checkpoint、参考replay、55k网格和atlas成功生成可迁移的Workbench配置及历史指令前缀。
- Workbench默认帧射线选点检查通过（命中8层候选），CPU预览生成并检查；补齐轨迹编辑器配置及limits。
- 04_third_right原plan插值通过；CPU完整IK通过，生成676行关节轨迹（含1939锚点，后续1940–2614）。
- 运行包导出通过，校验checkpoint指令前缀，生成累计轨迹和run.sh。
- 仓库根目录执行 `python -m unittest discover -s tools/trajectory_workbench -p 'test_*.py'`：20项通过。
- 已追加三帧端到端物理验证：从1939 checkpoint续跑1940–1942，右TCP上移0.1mm的测试动作经过IK、导出、前台物理及Genesis回放，三帧布态有限且帧数正确。此测试验证流程接通，不代表新抓取动作已经完成效果验收。

测试使用已安装并打补丁的Genesis环境；环境安装仍遵循复现指南。交付包含核心编辑器及生成链，历史实验专用批处理脚本不作为Workbench入口。
