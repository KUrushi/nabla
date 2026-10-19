#!/usr/bin/env python3
"""オプティマイザ（issue #205）用の Optax フィクスチャを作る。

nabla のライブラリ本体・テストの実行時は Python / JAX / Optax に依存しない（CLAUDE.md）。
これはフィクスチャを作るときにだけ使う開発用スクリプトで、生成した optimizer.lisp を
コミットする。使ったバージョン: jax 0.10.2, optax 0.2.8。

パラメータは (w (3 2), b (2)) の2つ組、勾配は固定の乱数から作った STEPS 個の同じ形の組。
ケースごとに optax の変換を STEPS 回適用し、各ステップの更新量・更新後のパラメータ
（update を apply_updates したもの）・最後の状態の葉（tree_leaves の順）を記録する。
勾配の大きさをステップごとに変えて、clip_by_global_norm が効くステップと効かない
ステップの両方を含める。スケジュールは step 0.. の値を記録する。

出力形式: 1つの s式。f32 は u32 のビットパターン、int32 の葉は整数そのまま。
  (:steps N
   :params ((w shape bits) (b shape bits))      ; 初期値
   :grads  (((w ..) (b ..)) ...)                 ; ステップ k の勾配
   :cases ((name :updates (((w ..) (b ..)) ...) ; ステップ k の更新量
                 :params  (((w ..) (b ..)) ...) ; ステップ k の更新後のパラメータ
                 :state   ((:f32 shape bits) | (:i32 shape ints) ...)) ...)
   :schedules ((name (bits ...)) ...))          ; step 0.. の値

使い方 (jax と optax の入った環境で):
    python3 tests/fixtures/optimizer/generate.py > tests/fixtures/optimizer/optimizer.lisp
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

STEPS = 6
GRAD_SCALES = [1.0, 1.0, 0.05, 1.5, 0.3, 2.0]


def bits(x) -> list[int]:
    return [int(b) for b in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def shape_of(x) -> str:
    return "(" + " ".join(str(d) for d in np.asarray(x).shape) + ")"


def arr(name, x) -> str:
    return f"({name} {shape_of(x)} ({' '.join(str(b) for b in bits(x))}))"


def pair(tree) -> str:
    return "(" + " ".join(arr(n, t) for n, t in zip(["w", "b"], tree)) + ")"


def leaf(x) -> str:
    x = np.asarray(x)
    if x.dtype == np.int32:
        return f"(:i32 {shape_of(x)} ({' '.join(str(int(v)) for v in x.reshape(-1))}))"
    return f"(:f32 {shape_of(x)} ({' '.join(str(b) for b in bits(x))}))"


CASES = {
    "sgd": lambda: optax.sgd(0.1),
    "sgd-momentum": lambda: optax.sgd(0.1, momentum=0.9),
    "sgd-nesterov": lambda: optax.sgd(0.1, momentum=0.9, nesterov=True),
    "sgd-linear-schedule": lambda: optax.sgd(optax.linear_schedule(0.5, 0.1, 4)),
    "sgd-warmup-cosine": lambda: optax.sgd(
        optax.warmup_cosine_decay_schedule(0.0, 0.1, 2, 6, end_value=0.01), momentum=0.9),
    "adam": lambda: optax.adam(1e-3),
    "adam-custom": lambda: optax.adam(0.01, b1=0.8, b2=0.9, eps=1e-6, eps_root=1e-9),
    "adam-cosine": lambda: optax.adam(optax.cosine_decay_schedule(0.01, 5, alpha=0.1)),
    "adamw": lambda: optax.adamw(0.01, weight_decay=0.1),
    "clip-by-global-norm": lambda: optax.clip_by_global_norm(1.0),
    "clip-adam": lambda: optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.01)),
}

SCHEDULES = {
    "constant": (optax.constant_schedule(0.3), 12),
    "linear": (optax.linear_schedule(1.0, 0.2, 5, transition_begin=2), 12),
    "cosine-decay": (optax.cosine_decay_schedule(0.5, 6, alpha=0.1), 12),
    "warmup-cosine-decay": (optax.warmup_cosine_decay_schedule(0.0, 1.0, 3, 10, end_value=0.1), 14),
}


def main() -> None:
    rng = np.random.default_rng(205)
    params = (rng.uniform(-1, 1, size=(3, 2)).astype(np.float32),
              rng.uniform(-1, 1, size=(2,)).astype(np.float32))
    grads = [tuple((scale * rng.standard_normal(p.shape)).astype(np.float32) for p in params)
             for scale in GRAD_SCALES]

    print(";;;; オプティマイザ（issue #205）用の Optax フィクスチャ。")
    print(";;;;")
    print(";;;; tests/fixtures/optimizer/generate.py で生成した。手で編集しない。")
    print(f"(:steps {STEPS}")
    print(" :params", pair(params))
    print(" :grads (" + "\n  ".join(pair(g) for g in grads) + ")")
    print(" :cases (")
    for name, make in CASES.items():
        tx = make()
        state = tx.init(params)
        p = params
        updates_log, params_log = [], []
        for g in grads:
            u, state = tx.update(g, state, p)
            p = optax.apply_updates(p, u)
            updates_log.append(pair(u))
            params_log.append(pair(p))
        print(f"  ({name}")
        print("   :updates (" + "\n    ".join(updates_log) + ")")
        print("   :params (" + "\n    ".join(params_log) + ")")
        print("   :state (" + "\n    ".join(leaf(x) for x in jax.tree_util.tree_leaves(state)) + "))")
    print(" )")
    print(" :schedules (")
    for name, (fn, n) in SCHEDULES.items():
        values = [float(fn(jnp.int32(i))) for i in range(n)]
        print(f"  ({name} ({' '.join(str(b) for b in bits(np.array(values, dtype=np.float32)))}))")
    print("))")


if __name__ == "__main__":
    main()
