# 机器人命令轨迹

`scene527_55k_73s_joint_commands.npz`：4373 帧、60 fps；包含 `joint_q` `(4373,19)`、`openness` `(4373,2)` 和 `joint_names` `(19,)`。`joint_q` 为双臂加夹爪的关节命令，`openness` 为两手开度；文件没有布料位置或物理 checkpoint。

来源：原工作区 `outputs/workbench/scene527_55068/final_push/20260929_press_at_yend_mu1_home/table_mu1/full_action/run_bundle/trajectory.npz`。原文件 SHA256：`fc95729acd82e2293f31535fe073e449eac5be3c8a807a2c5c34bbbfab090464`。拷贝后应保持相同哈希。

此轨迹从 SIM1 动作改编；在向仓库外共享前，应确认组内对轨迹数据和相关资产的分享范围。仅凭本文件无法重演七段 checkpoint 接续后的布态。
