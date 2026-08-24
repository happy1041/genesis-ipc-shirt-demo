# 资产交接

## 仓库为什么不包含资产

公开仓库只保存代码、参数和资产转换工具。SIM1衣服、Acone机器人模型和轨迹的
再分发许可尚未确认，因此不要直接提交到Git历史；运行视频、checkpoint和
`outputs/`也不应提交。

## 发给协作者的资产包

建议在仓库外制作一个 `shirt-demo-assets` 目录：

```text
shirt-demo-assets/
├── SIM1/
│   └── assets/
│       ├── cloth/short-shirt.usdc
│       └── acone/
│           ├── acone.urdf
│           └── meshes/...
└── episode_000000/
    └── episode_000000_sim1_replay.npz
```

`acone.urdf`引用的全部mesh必须保持原有相对目录结构。若对方已有完整的SIM1
checkout，优先直接使用它，无需重复打包SIM1。

发送资产前应先向资产权利人确认是否允许分享。可通过校内网盘或受控共享盘发送；
不要把授权不明的资产上传到公开GitHub Release或Git LFS。

## 接收方操作

```bash
git clone https://github.com/happy1041/genesis-ipc-shirt-demo.git
cd genesis-ipc-shirt-demo
cp .env.example .env
```

编辑 `.env`，让 `SIM1_ROOT` 和 `EPISODE_ROOT` 指向解压后的资产，同时填写
Genesis和两个Python环境的路径。然后检查：

```bash
./scripts/check_setup.sh
```

检查通过后，正式运行：

```bash
./scripts/run_demo.sh
```

脚本会自动生成被Git忽略的派生资产：

- `outputs/episode_000000/assets/short-shirt.obj`
- `outputs/episode_000000/assets/acone_ipc/acone_ipc.urdf`
- IPC collision proxy meshes

这些派生文件可能包含当前机器的绝对路径，不应从一台机器复制到另一台机器；在接收
方机器上重新生成即可。

## 平台范围

正式验证平台是Ubuntu/Linux和NVIDIA CUDA。Windows用户应使用WSL2，并在`.env`
中填写 `/mnt/c/...` 或WSL文件系统路径；原生Windows目前没有经过支持或验证。
