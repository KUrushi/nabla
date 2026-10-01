#!/usr/bin/env python3
"""2層 MLP の SGD 学習（issue #88）用の JAX フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX に依存しない（CLAUDE.md）。
これはフィクスチャを作るときにだけ使う開発用スクリプトで、生成した
mlp-sgd.lisp をコミットする。

nabla 側の定義（examples/mlp.lisp の MLP-LOSS）と同じ式を JAX で書く:

    h      = tanh(x w1 + b1)                         (N H)
    logits = h w2 + b2                               (N C)
    m      = stop_gradient(max(logits, axis=1))      (N)
    logp   = (logits - m) - log(sum(exp(logits - m), axis=1))
    loss   = -sum(y * logp) / N                      ()

初期値とデータは numpy の固定 seed で作る。f32 のまま STEPS 回
p <- p - LR * grad を行い、各ステップの更新前の損失と、更新後のパラメータを
記録する。bf16 は SGD の更新を f32 で累積する前提で比べるため、
フェーズ1のような bf16 フィクスチャは作らない（f32 のみ）。

出力形式: 1つの s式。値は f32 の u32 ビットパターン（row-major）。
  (:lr <u32> :steps N
   :inputs ((x shape bits) (y ...) (w1 ...) (b1 ...) (w2 ...) (b2 ...))
   :losses (bits ...)                       ; ステップ k の更新前の損失
   :params (((w1 ..) (b1 ..) (w2 ..) (b2 ..)) ...))  ; ステップ k の更新後、k=1..N

使い方 (jax の入った環境で):
    python3 tests/fixtures/train/generate.py > tests/fixtures/train/mlp-sgd.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np

N, D, H, C = 16, 2, 8, 2
STEPS = 5
LR = np.float32(0.5)


def bits(x) -> list[int]:
    return [int(b) for b in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(name, x) -> str:
    x = np.asarray(x)
    return f"({name} ({' '.join(str(d) for d in x.shape)}) ({' '.join(str(b) for b in bits(x))}))"


def loss_fn(params, x, y):
    w1, b1, w2, b2 = params
    h = jnp.tanh(x @ w1 + b1[None, :])
    logits = h @ w2 + b2[None, :]
    m = jax.lax.stop_gradient(jnp.max(logits, axis=1))
    shifted = logits - m[:, None]
    logp = shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=1))[:, None]
    return -jnp.sum(y * logp) / jnp.float32(N)


def main() -> None:
    rng = np.random.default_rng(88)
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

    vg = jax.jit(jax.value_and_grad(loss_fn))
    losses, history = [], []
    for _ in range(STEPS):
        loss, grads = vg(params, x, y)
        params = tuple(np.asarray(p - LR * g, dtype=np.float32) for p, g in zip(params, grads))
        losses.append(float(loss))
        history.append(params)

    names = ["w1", "b1", "w2", "b2"]
    print(";;;; 2層 MLP の SGD 学習（issue #88）用の JAX フィクスチャ。")
    print(";;;;")
    print(";;;; tests/fixtures/train/generate.py で生成した。手で編集しない。")
    print(";;;; 値は f32 の u32 ビットパターン。形式は generate.py の docstring を参照。")
    print("(:lr", bits(LR)[0], ":steps", STEPS)
    print(" :inputs")
    print(" (" + "\n  ".join(arr(n, v) for n, v in inputs) + ")")
    print(" :losses", "(" + " ".join(str(b) for b in bits(np.array(losses, dtype=np.float32))) + ")")
    print(" :params")
    print(" (" + "\n  ".join("(" + " ".join(arr(n, p) for n, p in zip(names, ps)) + ")" for ps in history) + "))")


if __name__ == "__main__":
    main()
