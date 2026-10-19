#!/usr/bin/env python3
"""expm1 / log1p / sin / cos を JAX (jax.lax.expm1 / log1p / sin / cos) と突き合わせるフィクスチャを作る。

issue #223。nabla のライブラリ本体は Python / JAX に依存しない（CLAUDE.md）ので、
これはフィクスチャを作るときにだけ使う開発用スクリプト。生成した2つのファイルを
コミットし、venv 自体はコミットしない。各行は (dtype 入力のビット 出力1のビット 出力2のビット)
で、tests/fixtures/sqrt/ と同じ形式。

  jax-expm1-log1p.lisp : (dtype x expm1(x) log1p(x))
  jax-sin-cos.lisp     : (dtype x sin(x) cos(x))

±0・±inf・NaN を含む特殊値は、nabla 側のテストで厳密（符号つき）に比べる。
f32 / f64 / bf16 の非正規数の入力は XLA CPU が 0 に丸める（flush-to-zero）ので入れない
（nabla の eager と IREE は丸めない）。f16 の非正規数も同じ扱いにして、dtype の最小の正規数
未満の非零の入力はすべて落とす。

sin / cos の入力は |x| <= 100 に限る。XLA の f32 の sin / cos は多項式で、引数を pi/2 の
倍数に縮約する部分の精度が x の大きさに比例して悪くなる（XLA は x 自体を f32 の
定数 pi/2 の数個の部分和で引く）。nabla の eager は double で計算して丸めるので、
|x| が数百を超えると、sin / cos がほぼ 0 になる点の近くで XLA と atol の範囲を超えて食い違う
ことがある。これは nabla の誤りではなく範囲縮約の実装の差なので、定義域を |x| <= 100 に
絞って比べる。

使い方 (x86_64 の VM で、jax[cpu]==0.10.2 の入った venv から。x64 を有効にして f64 も作る):
    python3 tests/fixtures/unary/generate.py
tests/fixtures/unary/ の下に2つのファイルを書く。
"""

import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

jax.config.update("jax_enable_x64", True)

DTYPES = [("f32", np.float32), ("f64", np.float64), ("bf16", jnp.bfloat16), ("f16", np.float16)]
TINY = {"f32": np.finfo(np.float32).tiny, "f64": np.finfo(np.float64).tiny,
        "bf16": float(ml_dtypes.finfo(ml_dtypes.bfloat16).smallest_normal),
        "f16": np.finfo(np.float16).tiny}


def bits(a):
    a = np.asarray(a)
    view = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return [int(v) for v in a.view(view)]


def inputs(name, dt, specials, randoms, limit=None):
    xs = np.concatenate([np.array(specials, dtype=np.float64)] + randoms)
    x = np.asarray(jnp.asarray(xs).astype(dt))
    xf = x.astype(np.float64)
    keep = (xf == 0) | np.isnan(xf) | (np.abs(xf) >= float(TINY[name]))
    if limit is not None:
        keep &= np.isnan(xf) | np.isinf(xf) | (np.abs(xf) <= limit)
    return jnp.asarray(x[keep])


def write(path, columns, fns, rng, specials, make_randoms, limit=None):
    with open(path, "w") as f:
        f.write(f";;;; JAX (jax.lax.{' / jax.lax.'.join(columns)}) との突き合わせ用フィクスチャ（issue #223）。\n")
        f.write(f";;;; 生成: tests/fixtures/unary/generate.py（jax {jax.__version__}）。"
                f"各行は (dtype 入力 {' '.join(columns)})、値はビットパターン。\n")
        f.write("(\n")
        for name, dt in DTYPES:
            x = inputs(name, dt, specials, make_randoms(rng), limit)
            outs = [bits(fn(x)) for fn in fns]
            for row in zip(bits(x), *outs):
                f.write(f" (:{name} {' '.join(str(v) for v in row)})\n")
        f.write(")\n")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(223)
    inf, nan = float("inf"), float("nan")

    expm1_log1p_specials = [0.0, -0.0, inf, -inf, nan, -1.0, -2.0, -0.5, 1e-10, -1e-10, 1e-5, -1e-5,
                            1.0, 0.5, 10.0, -10.0, 100.0, -100.0, 88.5, 705.0, 1e308, 1e30, -1e30, 1e-30, 0.1,
                            -0.9999, 3.0, 65504.0, -65504.0]

    def expm1_log1p_randoms(rng):
        return [rng.uniform(-5.0, 5.0, 60), rng.uniform(-1.0, 1.0, 40),
                np.exp(rng.uniform(-20.0, 3.0, 20)), -1.0 + np.exp(rng.uniform(-8.0, 0.0, 20))]

    write(os.path.join(here, "jax-expm1-log1p.lisp"), ["expm1", "log1p"],
          [jax.lax.expm1, jax.lax.log1p], rng, expm1_log1p_specials, expm1_log1p_randoms)

    sin_cos_specials = [0.0, -0.0, inf, -inf, nan, 1.0, -1.0, 0.5, 3.14159265, 1.5707964, 100.0, -100.0,
                        1e-5, -1e-5, 1e-10, 50.0, 6.2831855, -2.0]

    def sin_cos_randoms(rng):
        return [rng.uniform(-10.0, 10.0, 60), rng.uniform(-100.0, 100.0, 40), rng.uniform(-1.0, 1.0, 40)]

    write(os.path.join(here, "jax-sin-cos.lisp"), ["sin", "cos"],
          [jax.lax.sin, jax.lax.cos], rng, sin_cos_specials, sin_cos_randoms, limit=100.0)


if __name__ == "__main__":
    main()
