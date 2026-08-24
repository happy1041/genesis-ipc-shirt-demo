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
Read [setup and architecture](docs/DEVELOPMENT_CN.md) first.

```bash
export SIM1_ROOT=/path/to/SIM1
export EPISODE_ROOT=/path/to/episode_000000
export GENESIS_ENV=/path/to/genesis-ipc-env
export SIM1_PYTHON=/path/to/sim1-python

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
- `dHat=1.0 mm` is not stable in the current high-resolution pure-friction run.
- No project license has been selected yet; this repository is currently shared
  for research review, not as a redistribution grant.
- The latest result is a single full run, not a multi-seed robustness claim.

## Documentation

- [Current results and open questions](docs/STATUS_CN.md)
- [Setup, architecture and external dependencies](docs/DEVELOPMENT_CN.md)
- [Evaluation protocol](docs/EVALUATION_CN.md)
- [Short project history](docs/HISTORY_CN.md)
