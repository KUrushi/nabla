#!/usr/bin/env python3
"""scan を使った Elman RNN の学習（issue #141）用の JAX フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX に依存しない（CLAUDE.md）。
フィクスチャを作るときにだけ使う開発用スクリプトで、生成した rnn-sgd.lisp をコミットする。
examples/rnn.lisp の MAKE-RNN-LOSS と同じ式を jax.lax.scan で書く:

    h'   = tanh(h wh + x_t wx + b)        h: (B H)  x_t: (B D)  xs: (T B D)
    out  = h_T wo + bo                    (B O)
    loss = mean((out - y)^2)

f32（jax_enable_x64 は無効）。初期値とデータは numpy の固定 seed。STEPS 回
p <- p - LR * grad を行い、各ステップの更新前の損失、1ステップ目の勾配、
5 ステップ後と最終ステップ後のパラメータを記録する。
出力ファイルの先頭のコメントに jax のバージョンと x64 フラグを書く。

出力形式: 1つの s式。値は f32 の u32 ビットパターン（row-major）。
  (:lr <u32> :steps N
   :inputs ((xs shape bits) (y ..) (wh ..) (wx ..) (b ..) (wo ..) (bo ..))
   :losses (bits ...)          ; ステップ k の更新前の損失 k=0..N-1
   :grads  ((wh ..) (wx ..) (b ..) (wo ..) (bo ..))   ; 1ステップ目（更新前）の勾配
   :params5 ((wh ..) ...)      ; 5ステップ後のパラメータ
   :params-final ((wh ..) ...))  ; N ステップ後のパラメータ

使い方: python3 tests/fixtures/rnn/generate.py > tests/fixtures/rnn/rnn-sgd.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

T, B, D, H, O = 8, 4, 4, 8, 2
STEPS = 30
LR = np.float32(0.3)
NAMES = ["wh", "wx", "b", "wo", "bo"]


def bits(x) -> list[int]:
    return [int(v) for v in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(name, x) -> str:
    x = np.asarray(x)
    return f"({name} ({' '.join(str(d) for d in x.shape)}) ({' '.join(str(v) for v in bits(x))}))"


def loss_fn(params, xs, y):
    wh, wx, b, wo, bo = params

    def step(h, x):
        return jnp.tanh(h @ wh + x @ wx + b[None, :]), None

    h, _ = jax.lax.scan(step, jnp.zeros((B, H), jnp.float32), xs)
    pred = h @ wo + bo[None, :]
    return jnp.sum((pred - y) ** 2) / jnp.float32(B * O)


def uniform(rng, scale, *shape):
    return (scale * (rng.random(shape) - 0.5)).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(141)
    xs = uniform(rng, 2.0, T, B, D)
    y = xs.mean(axis=0)[:, :O].astype(np.float32)
    params = (uniform(rng, 0.6, H, H), uniform(rng, 1.0, D, H), uniform(rng, 0.2, H),
              uniform(rng, 1.0, H, O), uniform(rng, 0.2, O))
    inputs = [("xs", xs), ("y", y)] + list(zip(NAMES, params))

    vg = jax.jit(jax.value_and_grad(loss_fn))
    losses, first_grads, params5 = [], None, None
    for k in range(STEPS):
        loss, grads = vg(params, xs, y)
        if k == 0:
            first_grads = grads
        params = tuple(np.asarray(p - LR * g, dtype=np.float32) for p, g in zip(params, grads))
        losses.append(float(loss))
        if k == 4:
            params5 = params

    def group(ps):
        return "(" + " ".join(arr(n, p) for n, p in zip(NAMES, ps)) + ")"

    print(";;;; scan を使った Elman RNN の SGD 学習（issue #141）用の JAX フィクスチャ。")
    print(";;;;")
    print(f";;;; jax {jax.__version__}, jax_enable_x64={jax.config.jax_enable_x64}, f32。")
    print(";;;; tests/fixtures/rnn/generate.py で生成した。手で編集しない。")
    print("(:lr", bits(LR)[0], ":steps", STEPS)
    print(" :inputs")
    print(" (" + "\n  ".join(arr(n, v) for n, v in inputs) + ")")
    print(" :losses", "(" + " ".join(str(v) for v in bits(np.array(losses, dtype=np.float32))) + ")")
    print(" :grads", group(first_grads))
    print(" :params5", group(params5))
    print(" :params-final", group(params) + ")")


if __name__ == "__main__":
    main()
