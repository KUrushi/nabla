#!/usr/bin/env bash
# 一時的な実験（merge する前に消す）: 再現の MLIR を nabla と同じフラグ（target-cpu=host）で N 回コンパイルする
set -u
N="${GN_REPRO_RUNS:-60}"
for f in docs/iree-repros/group-norm-grad-multiplier.mlir docs/iree-repros/group-norm-grad-normalize-first.mlir; do
  fail=0
  for ((i = 0; i < N; i++)); do
    "${NABLA_IREE_HOME}/bin/iree-compile" --iree-input-type=stablehlo --iree-input-demote-f64-to-f32=false \
      --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu \
      --iree-llvmcpu-target-cpu=host --iree-dispatch-creation-fuse-multi-use=false \
      "$f" -o /dev/null 2>/dev/null || fail=$((fail + 1))
  done
  echo "[gn-repro] RESULT $(basename "$f") failures=${fail}/${N}"
done
