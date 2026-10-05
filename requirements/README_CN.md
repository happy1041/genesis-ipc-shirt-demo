# Python依赖快照

`verified-linux-py312-rtx50-cu130-20260827.txt` 是成功环境的完整 `pip freeze`
快照，用于审计和复建，不是跨平台、跨GPU通用的 `requirements.txt`。

已验证组合：

```text
Ubuntu 24.04 x86_64
Python 3.12.3
NVIDIA driver 580.173.02
RTX 50系列，compute capability 12.0
torch 2.11.0+cu130
pyuipc 0.0.25
```

当前Torch wheel包含 `sm_75/80/86/90/100/120`。CUDA wheel版本并不单独决定
兼容性；接收方GPU的compute capability、驱动和wheel包含的架构必须同时兼容。
因此，不同GPU不要盲目照装cu130快照，应先选择该GPU支持的Torch/CUDA组合，
再保持 `pyuipc==0.0.25` 并执行仓库检查和GPU smoke test。

如果机器与验证平台相近，可在全新的Python 3.12虚拟环境中尝试：

```bash
python -m pip install -r requirements/verified-linux-py312-rtx50-cu130-20260827.txt
python -m pip install --no-deps -e /path/to/patched/genesis-world
```

Genesis不在锁文件中，因为它必须使用指定源码快照或官方commit加仓库patch；
`SIM1_PYTHON`中的`pxr`也属于独立资产转换环境。锁文件同样不包含GPU驱动、
系统OpenGL、CUDA系统库和项目资产。

迁移时，版本完全一致表示“已验证环境”；版本不同并不一定不能运行，但只能视为
新的兼容性组合，必须重新完成20帧GPU smoke和980帧physics-only验收。

## 为什么暂不升级pyuipc 0.0.26

0.0.26保留了本项目使用的大部分Python函数签名，但改变了布料本构的数值语义，
因此“能够import和运行”不代表与0.0.25物理等价：

- membrane element从面积乘厚度的volume measure改为area measure，stretch与shear
  stiffness分别携带不同的厚度因子；
- raw `DiscreteShellBending.apply_to(..., bending_stiffness=k)` 改为per-area stiffness，
  后端不再额外乘厚度；
- 官方迁移提交将旧raw-kappa调用改为`kappa * thickness`来保持等效弯曲刚度；
- solver、contact kappa、CUDA graph与Stiff-GIPC数值路径也同时发生变化。

当前Genesis仍把`Cloth.bending_stiffness`原值直接传给raw bending接口。对本基线
`thickness=0.0001 m`、`bending=10`，若直接换0.0.26而不适配，同一个数值的等效
弯曲项会相差约`1/thickness = 10000`倍。升级必须在独立环境中先修改Genesis
材料映射并重新标定/回归，不能只把锁文件中的版本号改成0.0.26。
