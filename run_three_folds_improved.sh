#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/episode_000000}"
mkdir -p "$OUTPUT_ROOT"

# Shirt-centric trajectory corrections:
# 1) first fold: high transfer, then near-vertical placement;
# 2) second fold: both staggered approaches clear the first folded flap, then
#    lift before horizontal transfer so the shirt body stays on the table. The
#    two outer panels are advanced from opposite directions toward the central
#    body before release. This makes the two folded panels and the base share
#    one footprint (three folded shirt panels / six cloth surfaces) instead of
#    lying side by side;
# 3) third fold: robot-right traverses high, then recovers the public grasp
#    depth before closing so both front and back shirt layers enter the pinch;
#    its final placement is lowered before release so the flap is supported by
#    the folded stack instead of being released while still under tension.
# The exported closed shirt surface is about 0.72 m^2.  rho=800 kg/m^3 at
# 0.2 mm thickness gives about 160 g/m^2 (roughly 115 g total), instead of
# the previous unrealistically light 56 g/m^2.  The extra inertia reduces
# whole-shirt drag without making the table artificially sticky, so the
# transported flap can still settle under gravity.
CONTACT_MESH_FACES="${CONTACT_MESH_FACES:-8000}" \
  "$PROJECT_ROOT/run_contact_grasp_test.sh" \
  --frames 980 \
  --seed 0 \
  --camera-view overhead \
  --cloth-rho 800 \
  --table-friction 0.6 \
  --first-fold-tcp-lift 0.015 \
  --first-fold-transfer-lift 0.07 \
  --first-fold-stack-overlap 0.045 \
  --second-fold-left-approach-lift 0.06 \
  --second-fold-right-approach-lift 0.08 \
  --second-fold-transport-lift 0.08 \
  --second-fold-placement-relax 0.0 \
  --second-fold-stack-overlap 0.030 \
  --third-fold-right-grasp-lift 0.04 \
  --third-fold-right-grasp-depth 0.0 \
  --third-fold-right-grasp-lateral 0.012 \
  --third-fold-right-grasp-world-x -0.095 \
  --third-fold-right-grasp-world-y 0.0 \
  --third-fold-placement-depth 0.03 \
  --debug-third-fold-dir "$OUTPUT_ROOT/genesis_ipc_three_folds_centered_8000f_third_fold_state" \
  --ipc-constraint-strength-translation 100 \
  --ipc-constraint-strength-rotation 500 \
  --output "$OUTPUT_ROOT/genesis_ipc_three_folds_centered_8000f.mp4" \
  "$@"
