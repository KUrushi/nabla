#!/usr/bin/env python3
"""2層 MLP の per-example 勾配（issue #138）用の JAX フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX に依存しない（CLAUDE.md）。
これはフィクスチャを作るときにだけ使う開発用スクリプトで、生成した
mlp-per-example.lisp をコミットする。

nabla 側の定義（examples/mlp.lisp の MAKE-MLP-EXAMPLE-LOSS）と同じ、1サンプル
（x: (D)、y: (C) の one-hot）の損失を JAX で書く:

    h      = tanh(x w1 + b1)                         (H)
    logits = h w2 + b2                               (C)
    m      = stop_gradient(max(logits))              ()
    logp   = (logits - m) - log(sum(exp(logits - m)))
    loss   = -sum(y * logp)                          ()

jax.vmap(jax.grad(loss), in_axes=(None, 0, 0)) で、パラメータをバッチせず x と y だけを
バッチした per-example 勾配を求める。初期値とデータは numpy の固定 seed。f32 のまま
（jax_enable_x64 は有効にしない。tests/fixtures/train と同じ）。

出力形式: 1つの s式。値は f32 の u32 ビットパターン（row-major）。
  (:jax-version "..." :x64 nil
   :inputs ((x shape bits) (y ...) (w1 ...) (b1 ...) (w2 ...) (b2 ...))
   :grads  ((w1 shape bits) (b1 ...) (w2 ...) (b2 ...)))   ; 形は (N ...param-shape)

使い方 (jax の入った環境で):
    python3 tests/fixtures/per-example/generate.py > tests/fixtures/per-example/mlp-per-example.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

N, D, H, C = 6, 2, 8, 2


def bits(x) -> list[int]:
    return [int(b) for b in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(name, x) -> str:
    x = np.asarray(x)
    return f"({name} ({' '.join(str(d) for d in x.shape)}) ({' '.join(str(b) for b in bits(x))}))"


def example_loss(params, x, y):
    w1, b1, w2, b2 = params
    h = jnp.tanh(x @ w1 + b1)
    logits = h @ w2 + b2
    m = jax.lax.stop_gradient(jnp.max(logits))
    shifted = logits - m
    logp = shifted - jnp.log(jnp.sum(jnp.exp(shifted)))
    return -jnp.sum(y * logp)


def main() -> None:
    assert not jax.config.jax_enable_x64
    rng = np.random.default_rng(138)
    centers = rng.choice([-1.0, 1.0], size=(N, D))
    x = (centers + 0.3 * rng.standard_normal((N, D))).astype(np.float32)
    labels = (centers[:, 0] * centers[:, 1] > 0).astype(np.int64)
    y = np.eye(C, dtype=np.float32)[labels]
    params = (
        rng.uniform(-0.5, 0.5, size=(D, H)).astype(np.float32),
        rng.uniform(-0.5, 0.5, size=(H,)).astype(np.float32),
        rng.uniform(-0.5, 0.5, size=(H, C)).astype(np.float32),
        rng.uniform(-0.5, 0.5, size=(C,)).astype(np.float32),
    )
    inputs = [("x", x), ("y", y), ("w1", params[0]), ("b1", params[1]), ("w2", params[2]), ("b2", params[3])]
    grads = jax.jit(jax.vmap(jax.grad(example_loss), in_axes=(None, 0, 0)))(params, x, y)

    print(";;;; 2層 MLP の per-example 勾配（issue #138）用の JAX フィクスチャ。")
    print(";;;;")
    print(";;;; tests/fixtures/per-example/generate.py で生成した。手で編集しない。")
    print(";;;; 値は f32 の u32 ビットパターン。形式は generate.py の docstring を参照。")
    print(f'(:jax-version "{jax.__version__}" :x64 nil')
    print(" :inputs")
    print(" (" + "\n  ".join(arr(n, v) for n, v in inputs) + ")")
    print(" :grads")
    print(" (" + "\n  ".join(arr(n, g) for n, g in zip(["w1", "b1", "w2", "b2"], grads)) + "))")


if __name__ == "__main__":
    main()
