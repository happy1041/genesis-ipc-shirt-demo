# 代码包来源与验收范围

本分支在GitHub仓库原有`main`提交`bb94a344c831a8ced995c669385a843558e2e567`上导入本地交付包`f39cf09`，保留原仓库历史。交付包于2026-10-04从本机源码快照整理，来源旧仓库HEAD为`92a36d3`，主要代码来自其尚未提交的工作树。文件选择见[`SOURCE_FILES.txt`](SOURCE_FILES.txt)。

机械复制后，把依赖快照中旧机器的Genesis editable路径改为说明注释。交付仓库不包含`tools/trajectory_workbench/`，可选接触诊断辅助模块独立放在`tools/contact_dump.py`；主程序仅改了对应加载路径与两条CLI说明。

2026-10-05，新生成的`patches/genesis-world-8b1dba2-current.patch`替换了旧补丁。它由官方完整提交`8b1dba2fc99d0eff9ab7cc4b4bbc87685688fa44`与本机当前源码直接比较生成，SHA256为`355102f4710081590a080fd0f94bbc64cab4b1adf7324439df70a5a8d77793ae`。全部7个应用结果文件与当前源码逐字节一致；文件哈希及验证项见同目录manifest。`scripts/check_setup.sh`已更新对应哈希。

主要代码 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `src/run_genesis_ipc.py` | `f49b94d8561ec93e2e8d86a5c9f3ae223703060062fd4b0ca9bd680b4646b9ee` |
| `tools/contact_dump.py` | `d3b6f36eff24db46ea35f05adb7ad8b4ae1d7ce32c9543236f761d8966594793` |
| `data/scene527_55k_73s_joint_commands.npz` | `fc95729acd82e2293f31535fe073e449eac5be3c8a807a2c5c34bbbfab090464` |

从原七个分段运行记录核对，随附累计轨迹的前缀关节命令和双手开度逐元素一致。整理期间曾运行 Workbench 40 项 CPU 单元测试；它们随 Workbench 一起移除，不再作为当前精简仓库的测试覆盖范围。当前仓库只进行主程序 Python 语法与 Shell 语法检查，不代替 GPU 上的完整折叠测试。

未包含SIM1/双X5模型、55k衣服、Isaac场景、视频、checkpoint或replay。新补丁验证了当前Genesis源码的重建；本次未执行新GPU仿真，也未在独立安装环境中复跑完整成片。
