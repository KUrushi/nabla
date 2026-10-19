#!/usr/bin/env python3
"""convolution の勾配（issue #203）用の JAX フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX に依存しない（CLAUDE.md）。
フィクスチャを作るときにだけ使う開発用スクリプトで、生成した conv-grad.lisp をコミットする。
jax.lax.conv_general_dilated を channels-last（NHWC / HWIO / NHWC。空間次元の数に
よって NWC / NDHWC なども）で呼び、jax.vjp で余接線 ct に対する lhs と rhs の勾配
（dlhs, drhs）を記録する。値はすべて f32、入力は numpy の固定 seed。

出力形式: 1つの s式。値は f32 の u32 ビットパターン（row-major）。
  ((name :strides (..) :padding ((lo hi) ..) :lhs-dilation (..) :rhs-dilation (..)
         :groups G :batch-groups B
         :lhs ((shape) (bits)) :rhs (..) :out (..) :ct (..) :dlhs (..) :drhs (..)) ...)

使い方: python3 tests/fixtures/conv/generate-grad.py > tests/fixtures/conv/conv-grad.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

# (name, lhs shape, rhs shape, strides, padding, lhs dilation, rhs dilation, feature groups, batch groups)
CASES = [
    ("basic-2d", (2, 5, 6, 3), (3, 3, 3, 4), (1, 1), ((1, 1), (1, 1)), (1, 1), (1, 1), 1, 1),
    ("strided-2d", (1, 7, 8, 2), (2, 3, 2, 5), (2, 3), ((0, 1), (2, 0)), (1, 1), (1, 1), 1, 1),
    ("dilated-2d", (2, 6, 5, 2), (3, 2, 2, 3), (1, 2), ((1, 2), (0, 1)), (2, 1), (1, 2), 1, 1),
    ("grouped-2d", (2, 5, 5, 4), (2, 2, 2, 6), (1, 1), ((0, 0), (0, 0)), (1, 1), (1, 1), 2, 1),
    ("depthwise-2d", (1, 6, 6, 3), (3, 3, 1, 3), (1, 1), ((1, 1), (1, 1)), (1, 1), (1, 1), 3, 1),
    ("conv-1d", (3, 9, 2), (4, 2, 3), (2,), ((1, 2),), (1,), (1,), 1, 1),
    ("conv-3d", (1, 4, 4, 4, 2), (2, 2, 2, 2, 3), (1, 2, 1), ((0, 0), (1, 0), (0, 1)),
     (1, 1, 1), (1, 1, 1), 1, 1),
    ("negative-padding-2d", (1, 6, 6, 2), (3, 3, 2, 2), (1, 1), ((-1, 2), (0, -1)),
     (1, 1), (1, 1), 1, 1),
    # batch_group_count > 1 は、2階微分で現れる畳み込み（rhs の転置が G > 1 のとき）の形。
    ("batch-groups-1d", (4, 6, 3), (2, 3, 6), (2,), ((1, 0),), (1,), (1,), 1, 2),
    ("batch-groups-2d", (4, 5, 5, 2), (2, 2, 2, 4), (1, 2), ((0, 1), (1, 0)), (1, 1), (1, 2), 1, 2),
]


def bits(x) -> list[int]:
    return [int(v) for v in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(x) -> str:
    x = np.asarray(x)
    return f"({' '.join(str(d) for d in x.shape)}) ({' '.join(str(v) for v in bits(x))})"


def specs(spatial: int):
    sp = "DHW"[-spatial:] if spatial > 1 else "W"
    return ("N" + sp + "C", sp + "IO", "N" + sp + "C")


def main():
    rng = np.random.default_rng(203)
    print(";;;; convolution の勾配（issue #203）用の JAX フィクスチャ。")
    print(";;;;")
    print(f";;;; jax {jax.__version__}, jax_enable_x64={jax.config.jax_enable_x64}, f32。")
    print(";;;; tests/fixtures/conv/generate-grad.py で生成した。手で編集しない。")
    print("(")
    for name, lshape, rshape, strides, padding, ldil, rdil, groups, bgroups in CASES:
        lhs = (rng.random(lshape) - 0.5).astype(np.float32)
        rhs = (rng.random(rshape) - 0.5).astype(np.float32)

        def conv(x, w):
            return jax.lax.conv_general_dilated(
                x, w, window_strides=strides, padding=padding,
                lhs_dilation=ldil, rhs_dilation=rdil, dimension_numbers=specs(len(strides)),
                feature_group_count=groups, batch_group_count=bgroups)

        out, vjp = jax.vjp(conv, jnp.asarray(lhs), jnp.asarray(rhs))
        ct = (rng.random(out.shape) - 0.5).astype(np.float32)
        dlhs, drhs = vjp(jnp.asarray(ct))
        pad = " ".join(f"({lo} {hi})" for lo, hi in padding)
        ints = lambda t: "(" + " ".join(str(v) for v in t) + ")"
        print(f" ({name} :strides {ints(strides)} :padding ({pad}) :lhs-dilation {ints(ldil)}"
              f" :rhs-dilation {ints(rdil)} :groups {groups} :batch-groups {bgroups}")
        print(f"  :lhs ({arr(lhs)})")
        print(f"  :rhs ({arr(rhs)})")
        print(f"  :out ({arr(np.asarray(out))})")
        print(f"  :ct ({arr(ct)})")
        print(f"  :dlhs ({arr(np.asarray(dlhs))})")
        print(f"  :drhs ({arr(np.asarray(drhs))}))")
    print(")")


if __name__ == "__main__":
    main()
