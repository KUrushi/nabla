#!/usr/bin/env python3
"""group-norm（Flax の nn.GroupNorm）を Flax と突き合わせるフィクスチャを作る。

issue #260。形式と方針は generate.py（#204）と同じ（固定のパラメータ・入力・余接線 ct で
loss = sum(out * ct) の出力と、パラメータ・入力についての勾配を記録する。f32）。generate.py の
case / rnd をそのまま使うので、同じディレクトリに置く。

各ケースの :config は nabla 側が group-norm を作るための設定（:num-groups :group-size :use-scale
:use-bias :epsilon。Lisp の値で、指定しないものは nil）。入力は平均 3・標準偏差 2 にずらす（generate.py の
layer-norm と同じ。速い分散の桁落ちの分だけ許容誤差を緩める。group-norm-test.lisp）。

注: JAX / Flax は Linux の Rosetta VM では import できないので、macOS arm64 の venv で生成する（issue #241）。

使い方 (jax[cpu]==0.10.2 と flax の入った venv で):
    python3 tests/fixtures/layers/generate-group-norm.py > tests/fixtures/layers/flax-group-norm.lisp
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flax
import flax.linen as nn
import jax
import numpy as np

from generate import case, rnd


def lisp(value):
    if value is None:
        return "nil"
    if value is True:
        return "t"
    if value is False:
        return "nil"
    return str(value)


def group_norm_case(rng, name, x_shape, num_groups=32, group_size=None, use_scale=True, use_bias=True):
    module = nn.GroupNorm(num_groups=num_groups, group_size=group_size, epsilon=1e-6,
                          use_scale=use_scale, use_bias=use_bias)
    channels = x_shape[-1]
    params = {}
    if use_scale:
        params["scale"] = rnd(rng, channels, scale=0.5, shift=1.0)
    if use_bias:
        params["bias"] = rnd(rng, channels)
    x = rnd(rng, *x_shape, scale=2.0, shift=3.0)
    ct = rnd(rng, *x_shape)
    text = case(name, module, params, x, ct)
    config = (f":config (:num-groups {lisp(num_groups)} :group-size {lisp(group_size)} "
              f":use-scale {lisp(use_scale)} :use-bias {lisp(use_bias)} :epsilon 1f-6)")
    return text[:-1] + " " + config + ")"


def main():
    rng = np.random.default_rng(260)
    cases = [
        # NHWC: 統計は先頭のバッチ軸以外（空間と群の中のチャンネル）
        group_norm_case(rng, "num-groups", (2, 3, 4, 6), num_groups=2),
        # group_size で指定する（Flax は num_groups=None を明示する）
        group_norm_case(rng, "group-size", (2, 5, 6), num_groups=None, group_size=3),
        # rank 2: 空間の軸が無く、群の中のチャンネルだけで統計を取る
        group_norm_case(rng, "rank2", (3, 8), num_groups=4),
        # rank 1: バッチ軸も無い（Flax の reduction_axes の既定は [-1] だけ）
        group_norm_case(rng, "rank1", (6,), num_groups=3),
        # 群が 1 つ（バッチ軸以外の全部で正規化）と、群の大きさが 1（instance norm）
        group_norm_case(rng, "one-group", (2, 3, 4), num_groups=1),
        group_norm_case(rng, "instance", (2, 5, 4), num_groups=None, group_size=1),
        # 大きさ 1 の軸（空間の軸が 1 × 1）、既定の num_groups=32（C = 64）、要素が 1 つの群（rank 2・group_size 1。
        # 出力も勾配も 0）
        group_norm_case(rng, "unit-axes", (1, 1, 1, 8), num_groups=4),
        group_norm_case(rng, "default-32-groups", (2, 2, 64)),
        group_norm_case(rng, "single-element-groups", (3, 4), num_groups=None, group_size=1),
        group_norm_case(rng, "no-scale", (2, 3, 6), num_groups=3, use_scale=False),
        group_norm_case(rng, "no-bias", (2, 3, 6), num_groups=3, use_bias=False),
        group_norm_case(rng, "no-scale-no-bias", (2, 3, 6), num_groups=3, use_scale=False, use_bias=False),
    ]
    print(";;;; Flax (flax.linen) の nn.GroupNorm との突き合わせ用フィクスチャ（issue #260）。")
    print(";;;; 生成: tests/fixtures/layers/generate-group-norm.py。形式は generate.py と generate-group-norm.py の docstring。")
    print(";;;; macOS arm64（Apple Silicon）で x64 無効で生成した。")
    print(f'(:versions "jax {jax.__version__} flax {flax.__version__}"')
    print(" :cases (")
    for c in cases:
        print("  " + c)
    print(" ))")


if __name__ == "__main__":
    main()
