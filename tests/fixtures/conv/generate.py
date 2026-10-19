#!/usr/bin/env python3
"""convolution プリミティブ（issue #197）用の JAX フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX に依存しない（CLAUDE.md）。
フィクスチャを作るときにだけ使う開発用スクリプトで、生成した conv.lisp をコミットする。
jax.lax.conv_general_dilated を channels-last（NHWC / HWIO / NHWC。空間次元の数に
よって NWC / NDHWC なども）で呼び、f32 の入力と出力を記録する。入力は numpy の固定 seed。

出力形式: 1つの s式。値は f32 の u32 ビットパターン（row-major）。
  ((name :strides (..) :padding ((lo hi) ..) :lhs-dilation (..) :rhs-dilation (..)
         :groups G
         :lhs ((shape) (bits)) :rhs ((shape) (bits)) :out ((shape) (bits))) ...)

使い方: python3 tests/fixtures/conv/generate.py > tests/fixtures/conv/conv.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

# (name, lhs shape, rhs shape, strides, padding, lhs dilation, rhs dilation, groups)
CASES = [
    ("basic-2d", (2, 5, 6, 3), (3, 3, 3, 4), (1, 1), ((1, 1), (1, 1)), (1, 1), (1, 1), 1),
    ("strided-2d", (1, 7, 8, 2), (2, 3, 2, 5), (2, 3), ((0, 1), (2, 0)), (1, 1), (1, 1), 1),
    ("dilated-2d", (2, 6, 5, 2), (3, 2, 2, 3), (1, 2), ((1, 2), (0, 1)), (2, 1), (1, 2), 1),
    ("grouped-2d", (2, 5, 5, 4), (2, 2, 2, 6), (1, 1), ((0, 0), (0, 0)), (1, 1), (1, 1), 2),
    ("depthwise-2d", (1, 6, 6, 3), (3, 3, 1, 3), (1, 1), ((1, 1), (1, 1)), (1, 1), (1, 1), 3),
    ("conv-1d", (3, 9, 2), (4, 2, 3), (2,), ((1, 2),), (1,), (1,), 1),
    ("conv-3d", (1, 4, 4, 4, 2), (2, 2, 2, 2, 3), (1, 2, 1), ((0, 0), (1, 0), (0, 1)),
     (1, 1, 1), (1, 1, 1), 1),
    ("negative-padding-2d", (1, 6, 6, 2), (3, 3, 2, 2), (1, 1), ((-1, 2), (0, -1)),
     (1, 1), (1, 1), 1),
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
    rng = np.random.default_rng(197)
    print(f";;;; conv プリミティブ（issue #197）用の JAX フィクスチャ。")
    print(f";;;;")
    print(f";;;; jax {jax.__version__}, jax_enable_x64={jax.config.jax_enable_x64}, f32。")
    print(f";;;; tests/fixtures/conv/generate.py で生成した。手で編集しない。")
    print("(")
    for name, lshape, rshape, strides, padding, ldil, rdil, groups in CASES:
        lhs = (rng.random(lshape) - 0.5).astype(np.float32)
        rhs = (rng.random(rshape) - 0.5).astype(np.float32)
        out = jax.lax.conv_general_dilated(
            jnp.asarray(lhs), jnp.asarray(rhs), window_strides=strides, padding=padding,
            lhs_dilation=ldil, rhs_dilation=rdil, dimension_numbers=specs(len(strides)),
            feature_group_count=groups)
        pad = " ".join(f"({lo} {hi})" for lo, hi in padding)
        ints = lambda t: "(" + " ".join(str(v) for v in t) + ")"
        print(f" ({name} :strides {ints(strides)} :padding ({pad}) :lhs-dilation {ints(ldil)}"
              f" :rhs-dilation {ints(rdil)} :groups {groups}")
        print(f"  :lhs ({arr(lhs)})")
        print(f"  :rhs ({arr(rhs)})")
        print(f"  :out ({arr(np.asarray(out))}))")
    print(")")


if __name__ == "__main__":
    main()
