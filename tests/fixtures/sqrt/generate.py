#!/usr/bin/env python3
"""sqrt / rsqrt を JAX (jax.lax.sqrt / jax.lax.rsqrt) と突き合わせるフィクスチャを作る。

issue #196。nabla のライブラリ本体は Python / JAX に依存しない（CLAUDE.md）ので、
これはフィクスチャを作るときにだけ使う開発用スクリプト。生成した jax-sqrt.lisp を
コミットし、venv 自体はコミットしない。各行は (dtype 入力のビット sqrt のビット
rsqrt のビット)。±0・±inf・NaN・負の値・非正規数を含む。

使い方 (jax[cpu]==0.10.2 の入った venv で):
    python3 tests/fixtures/sqrt/generate.py > tests/fixtures/sqrt/jax-sqrt.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def bits(a):
    a = np.asarray(a)
    view = {2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize]
    return [int(v) for v in a.view(view)]


def main() -> None:
    rng = np.random.default_rng(196)
    specials = [0.0, -0.0, float("inf"), float("-inf"), float("nan"), -4.0, -1e-3,
                4.0, 9.0, 0.25, 1e-40, 1e-30, 1e30, 3.0, 2.0, 0.5, 1e-5, 65504.0, 1e-8]
    print(";;;; JAX (jax.lax.sqrt / jax.lax.rsqrt) との突き合わせ用フィクスチャ（issue #196）。")
    print(";;;; 生成: tests/fixtures/sqrt/generate.py。各行は (dtype 入力 sqrt rsqrt)、値はビットパターン。")
    print("(")
    for name, dt in [("f32", np.float32), ("f64", np.float64),
                     ("bf16", jnp.bfloat16), ("f16", np.float16)]:
        xs = np.concatenate([np.array(specials, dtype=np.float64),
                             rng.uniform(1e-3, 100.0, 60), rng.uniform(0, 1, 40),
                             np.exp(rng.uniform(-30, 30, 40))])
        x = jnp.asarray(xs).astype(dt)
        s = jax.lax.sqrt(x)
        r = jax.lax.rsqrt(x)
        for xb, sb, rb in zip(bits(x), bits(s), bits(r)):
            print(f" (:{name} {xb} {sb} {rb})")
    print(")")


if __name__ == "__main__":
    main()
