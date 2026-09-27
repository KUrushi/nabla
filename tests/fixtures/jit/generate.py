#!/usr/bin/env python3
"""defjit の end-to-end 完了条件（issue #34 / #35）用の MLP フィクスチャを作る。

nabla のライブラリ本体は Python / JAX に依存しない（CLAUDE.md）ので、これは
フィクスチャを作るときにだけ使う開発用スクリプト。生成した mlp.lisp を
コミットし、venv 自体はコミットしない。

nabla 側の定義（examples/jit.lisp / tests/iree/jit-test.lisp の %mlp）:

    h      = tanh(dot(x, w1) + broadcast(b1))          shape (B H)
    logits = dot(h, w2) + broadcast(b2)                shape (B C)
    out1   = reduce-max(logits, axes=(1))              shape (B,)
    out2   = reduce-sum(reshape(logits, (B*C,)))        shape ()

bf16 は nabla が実際に emit する数値と合わせるため、dot は
preferred_element_type=float32 で計算してから bfloat16 に丸め（#54/#59）、
reduce-sum は float32 に上げてから丸める（#64）。それ以外の要素ごとの演算
（+、tanh、reduce-max）は bf16 のまま行う（IREE が要素ごとの演算を融合して
1回だけ丸めるのに対し、この JAX 側は演算ごとに丸めるので、テスト側の
許容誤差を rtol × (1 + eqn数) に緩めて吸収する。tests/iree/jit-test.lisp
参照）。

出力形式: 1つの読みやすい s式 (:f32 (:inputs (...) :outputs (...)) :bf16 (...))。
値はすべて整数のビットパターン（f32 は u32、bf16 は u16）にして、Lisp の
リーダが浮動小数点フォーマットに依存しないようにする。f32 は
nb::%make-single-float で、bf16 はビット列をそのまま (unsigned-byte 16) の
配列としてデコードする。

使い方 (jax の入った venv で):
    python3 tests/fixtures/jit/generate.py > tests/fixtures/jit/mlp.lisp
"""

import numpy as np
import jax.numpy as jnp

B, D, H, C = 2, 3, 4, 2


def f32_bits(x: np.ndarray) -> list[int]:
    return [int(b) for b in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def bf16_bits(x) -> list[int]:
    return [int(b) for b in np.asarray(jnp.asarray(x, dtype=jnp.bfloat16).view(jnp.uint16)).reshape(-1)]


def lisp_array(name: str, shape: tuple[int, ...], bits: list[int]) -> str:
    shape_str = "(" + " ".join(str(d) for d in shape) + ")"
    bits_str = "(" + " ".join(str(b) for b in bits) + ")"
    return f"({name} {shape_str} {bits_str})"


def main() -> None:
    rng = np.random.default_rng(34)
    x = rng.uniform(-1.0, 1.0, size=(B, D)).astype(np.float32)
    w1 = rng.uniform(-1.0, 1.0, size=(D, H)).astype(np.float32)
    b1 = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float32)
    w2 = rng.uniform(-1.0, 1.0, size=(H, C)).astype(np.float32)
    b2 = rng.uniform(-1.0, 1.0, size=(C,)).astype(np.float32)

    # --- f32: そのまま JAX の float32 で計算する ---
    h_f32 = jnp.tanh(jnp.dot(x, w1) + b1[None, :])
    logits_f32 = jnp.dot(h_f32, w2) + b2[None, :]
    out1_f32 = jnp.max(logits_f32, axis=1)
    out2_f32 = jnp.sum(logits_f32.reshape(-1))

    # --- bf16: 入力を bf16 に丸めてから、nabla の数値方針で計算する ---
    x_bf16 = jnp.asarray(x, dtype=jnp.bfloat16)
    w1_bf16 = jnp.asarray(w1, dtype=jnp.bfloat16)
    b1_bf16 = jnp.asarray(b1, dtype=jnp.bfloat16)
    w2_bf16 = jnp.asarray(w2, dtype=jnp.bfloat16)
    b2_bf16 = jnp.asarray(b2, dtype=jnp.bfloat16)

    dot1_bf16 = jnp.dot(x_bf16, w1_bf16, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    h_bf16 = jnp.tanh(dot1_bf16 + b1_bf16[None, :])
    dot2_bf16 = jnp.dot(h_bf16, w2_bf16, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    logits_bf16 = dot2_bf16 + b2_bf16[None, :]
    out1_bf16 = jnp.max(logits_bf16, axis=1)
    out2_bf16 = jnp.sum(logits_bf16.reshape(-1).astype(jnp.float32)).astype(jnp.bfloat16)

    print(";;;; defjit の end-to-end 完了条件（issue #34 / #35）用の MLP フィクスチャ。")
    print(";;;;")
    print(";;;; tests/fixtures/jit/generate.py で生成した。手で編集しない。")
    print(";;;; 形式: (:f32 (:inputs (...) :outputs (...)) :bf16 (...))。")
    print(";;;; 各配列は (name shape bits) で、bits は f32 なら u32、bf16 なら")
    print(";;;; u16 のビットパターンのリスト（row-major）。")
    print("(:f32")
    print(" (:inputs")
    print("  (" + lisp_array("x", (B, D), f32_bits(x)))
    print("   " + lisp_array("w1", (D, H), f32_bits(w1)))
    print("   " + lisp_array("b1", (H,), f32_bits(b1)))
    print("   " + lisp_array("w2", (H, C), f32_bits(w2)))
    print("   " + lisp_array("b2", (C,), f32_bits(b2)) + ")")
    print("  :outputs")
    print("  (" + lisp_array("out1", (B,), f32_bits(out1_f32)))
    print("   " + lisp_array("out2", (), f32_bits(out2_f32)) + "))")
    print(" :bf16")
    print(" (:inputs")
    print("  (" + lisp_array("x", (B, D), bf16_bits(x_bf16)))
    print("   " + lisp_array("w1", (D, H), bf16_bits(w1_bf16)))
    print("   " + lisp_array("b1", (H,), bf16_bits(b1_bf16)))
    print("   " + lisp_array("w2", (H, C), bf16_bits(w2_bf16)))
    print("   " + lisp_array("b2", (C,), bf16_bits(b2_bf16)) + ")")
    print("  :outputs")
    print("  (" + lisp_array("out1", (B,), bf16_bits(out1_bf16)))
    print("   " + lisp_array("out2", (), bf16_bits(out2_bf16)) + ")))")


if __name__ == "__main__":
    main()
