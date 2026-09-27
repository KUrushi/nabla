#!/usr/bin/env python3
"""bf16 / f16 の RNE 変換を JAX の数値と突き合わせるフィクスチャを作る。

issue #38（u4）。nabla のライブラリ本体は Python / JAX に依存しない
（CLAUDE.md）ので、これはフィクスチャを作るときにだけ使う開発用スクリプト。
生成した jax-cross-check.lisp をコミットし、venv 自体はコミットしない。

使い方 (jax の入った venv で):
    python3 tests/fixtures/float16/generate.py > tests/fixtures/float16/jax-cross-check.lisp
"""

import random

import jax.numpy as jnp
import numpy as np


def bits32_to_bf16_bits(bits32: int) -> int:
    x = np.array([bits32], dtype=np.uint32).view(np.float32)
    return int(jnp.asarray(x, dtype=jnp.bfloat16).view(jnp.uint16)[0])


def bits32_to_f16_bits(bits32: int) -> int:
    x = np.array([bits32], dtype=np.uint32).view(np.float32)
    return int(jnp.asarray(x, dtype=jnp.float16).view(jnp.uint16)[0])


def main() -> None:
    rng = random.Random(38)
    # 全ビットパターンを一様に選ぶと bf16/f16 とも NaN や極端な値ばかりに
    # 偏らないよう、[-4, 4] あたりの「普通の」値を厚めに混ぜる。
    samples = []
    for _ in range(300):
        samples.append(rng.getrandbits(32))
    for _ in range(300):
        x = np.float32(rng.uniform(-4.0, 4.0))
        samples.append(int(x.view(np.uint32)))

    print(";;;; JAX (jnp.asarray(x, bfloat16/float16).view(uint16)) と")
    print(";;;; 突き合わせた bf16 / f16 の変換結果（issue #38）。")
    print(";;;;")
    print(";;;; tests/fixtures/float16/generate.py で生成した。手で編集しない。")
    print(";;;; 各要素は (bits32 bf16-bits f16-bits)。bits32 が NaN のパターンは")
    print(";;;; ビットパターンが処理系依存になりうるので除いてある。")
    print("(")
    for bits32 in samples:
        x = np.array([bits32], dtype=np.uint32).view(np.float32)[0]
        if np.isnan(x):
            continue
        bf16_bits = bits32_to_bf16_bits(bits32)
        f16_bits = bits32_to_f16_bits(bits32)
        print(f" ({bits32} {bf16_bits} {f16_bits})")
    print(")")


if __name__ == "__main__":
    main()
