# Genesis IPC Shirt-Folding Demo

Dual-arm replay of a three-fold short-shirt task using Genesis FEM cloth and
libuIPC contact. The current public baseline uses **physical gripper contact and
friction only**; virtual cloth attachments are disabled.

## Current baseline

- 13,767-face shirt mesh exported from SIM1
- shell radius `0.1 mm`, `dHat=1.5 mm`
- 2 substeps, full/slow IPC solver
- `E=20 kPa`, bending `10`, density `800 kg/m³`
- cloth/table/robot friction `1 / 1 / 2`
- physics is saved first; video is rendered offline from replay states

Latest verified run: 980/980 frames, 18.5 minutes on an RTX 5070 Ti. All three
folds complete, but the final right grasp slides about 90 mm and remains the
main robustness issue. See [current status](docs/STATUS_CN.md).

## Quick start

This repository depends on external SIM1 assets and a patched Genesis checkout.
The repository alone is not enough to run the demo. Obtain the asset bundle from
the project owner, then read [asset handoff](docs/ASSETS_CN.md) and
[setup and architecture](docs/DEVELOPMENT_CN.md).

```bash
cp .env.example .env
# Edit .env with the local asset and Python environment paths.
./scripts/check_setup.sh
./scripts/run_demo.sh
```

Run only physics or only render an existing state trajectory:

```bash
./scripts/run_demo.sh --physics-only
./scripts/run_demo.sh --render-only
```

The validated CLI preset is stored in
[`configs/dhat_1p5_pure_friction.args`](configs/dhat_1p5_pure_friction.args).
Append CLI arguments to override it for an experiment.

## Repository layout

```text
src/            core simulator
scripts/        public run and asset-preparation entry points
configs/        versioned CLI presets
diagnostics/    run evaluation and provenance tools
tools/assets/   asset conversion and collision-proxy generation
docs/           status, development and evaluation documentation
patches/        required Genesis patch
outputs/        generated results; ignored by Git
```

## Important limitations

- Clone is not self-contained: SIM1 trajectory, shirt, and Acone visual meshes
  are external and their redistribution permission is not assumed.
- Apply `patches/genesis-world-8b1dba2-local.patch` to Genesis commit `8b1dba2`.
  This is a cumulative development snapshot, not yet a minimal patch; it still
  contains disabled soft-virtual-grasp experiment support. See
  [patch notes](patches/README_CN.md).
- `dHat=1.0 mm` is not stable in the current high-resolution pure-friction run.
- No project license has been selected yet; this repository is currently shared
  for research review, not as a redistribution grant.
- The latest result is a single full run, not a multi-seed robustness claim.
- Ubuntu/Linux with NVIDIA CUDA is the validated platform. Windows users should
  use WSL2; native Windows is not currently supported.

## Documentation

- [Current results and open questions](docs/STATUS_CN.md)
- [Asset bundle and collaborator handoff](docs/ASSETS_CN.md)
- [Genesis local patch scope and history](patches/README_CN.md)
- [Setup, architecture and external dependencies](docs/DEVELOPMENT_CN.md)
- [Evaluation protocol](docs/EVALUATION_CN.md)
- [Short project history](docs/HISTORY_CN.md)
