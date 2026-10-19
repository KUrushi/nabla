#!/usr/bin/env python3
"""conv（Flax の nn.Conv）を Flax と突き合わせるフィクスチャを作る。

issue #207。形式と方針は generate.py（#204）と同じ（固定のパラメータ・入力・余接線 ct で
loss = sum(out * ct) の出力と、パラメータ・入力についての勾配を記録する。f32）。generate.py の
case / rnd をそのまま使うので、同じディレクトリに置く。

各ケースの :config は nabla 側が conv を作るための設定（:strides :padding :kernel-dilation
:groups）。padding は "SAME" / "VALID" / 明示の ((low high) ...)。:init-stats は Flax の init を
1回サンプルした kernel の形・標準偏差・最大絶対値。

使い方 (jax[cpu]==0.10.2 と flax の入った venv で):
    python3 tests/fixtures/layers/generate-conv.py > tests/fixtures/layers/flax-conv.lisp
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from generate import case, rnd


def conv_case(rng, name, x_shape, kernel_size, features, in_features, strides=1, padding="SAME",
              kernel_dilation=1, groups=1):
    k = tuple(kernel_size)
    module = nn.Conv(features, k, strides=strides, padding=padding, kernel_dilation=kernel_dilation,
                     feature_group_count=groups)
    kernel = rnd(rng, *k, in_features // groups, features)
    x = rnd(rng, *x_shape)
    bias = rnd(rng, features)
    out = module.apply({"params": {"kernel": kernel, "bias": bias}}, x)
    ct = rnd(rng, *out.shape)
    text = case(name, module, {"kernel": kernel, "bias": bias}, x, ct)
    s = strides if isinstance(strides, (tuple, list)) else (strides,) * len(k)
    d = kernel_dilation if isinstance(kernel_dilation, (tuple, list)) else (kernel_dilation,) * len(k)
    if isinstance(padding, str):
        pad = f'"{padding}"'
    else:
        pad = "(" + " ".join(f"({lo} {hi})" for lo, hi in padding) + ")"
    config = (f':config (:strides ({" ".join(map(str, s))}) :padding {pad} '
              f':kernel-dilation ({" ".join(map(str, d))}) :groups {groups})')
    return text[:-1] + " " + config + ")"


def main():
    rng = np.random.default_rng(207)
    cases = [
        conv_case(rng, "same", (2, 6, 7, 3), (3, 3), 4, 3),
        # 偶数の入力・stride 2: SAME の pad が左右非対称 (0 1)
        conv_case(rng, "same-stride2", (2, 6, 7, 3), (3, 3), 4, 3, strides=(2, 2)),
        conv_case(rng, "same-even-kernel", (1, 5, 6, 2), (2, 4), 3, 2),
        conv_case(rng, "valid-stride", (2, 7, 8, 3), (2, 3), 5, 3, strides=(2, 1), padding="VALID"),
        conv_case(rng, "explicit", (2, 5, 6, 3), (3, 2), 4, 3, padding=((1, 2), (0, 1))),
        conv_case(rng, "groups", (2, 5, 5, 4), (3, 3), 6, 4, groups=2),
        conv_case(rng, "depthwise", (2, 5, 5, 4), (3, 3), 4, 4, groups=4),
        conv_case(rng, "kernel-dilation", (2, 8, 7, 2), (3, 3), 3, 2, kernel_dilation=(2, 1)),
        conv_case(rng, "unbatched", (6, 7, 3), (3, 3), 4, 3, strides=(2, 1)),
        conv_case(rng, "conv1d", (2, 9, 3), (3,), 4, 3, strides=2),
    ]
    key = jax.random.PRNGKey(207)
    kernel = nn.Conv(16, (3, 3), feature_group_count=2).init(key, jnp.zeros((1, 8, 8, 16)))["params"]["kernel"]
    shape = " ".join(str(d) for d in kernel.shape)
    print(";;;; Flax (flax.linen.Conv) との突き合わせ用フィクスチャ（issue #207）。")
    print(";;;; 生成: tests/fixtures/layers/generate-conv.py。形式は generate.py と generate-conv.py の docstring。")
    print(f'(:versions "jax {jax.__version__} flax {flax.__version__}"')
    print(" :cases (")
    for c in cases:
        print("  " + c)
    print(" )")
    print(" :init-stats (")
    print(f"  (:conv-kernel ({shape}) {float(kernel.std())!r} {float(jnp.abs(kernel).max())!r})")
    print(" ))")


if __name__ == "__main__":
    main()
