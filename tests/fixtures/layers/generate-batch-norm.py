#!/usr/bin/env python3
"""batch-norm を Flax の nn.BatchNorm と突き合わせるフィクスチャを作る（issue #260）。

nabla のライブラリ本体・テストの実行時は Python / JAX / Flax に依存しない（CLAUDE.md）。フィクスチャを
作るときにだけ使う開発用スクリプトで、生成した flax-batch-norm.lisp をコミットする。

各ケースは、固定のパラメータ（scale / bias）と移動平均の初期値（batch_stats の mean / var）から、
学習モード（use_running_average=False）を STEPS ステップ、入力 x.<i> を変えて続けて呼ぶ。ステップごとに
  - 出力 out.<i>
  - 更新後の移動平均 mean.<i> / var.<i>（apply(..., mutable=["batch_stats"]) が返すもの）
  - loss = sum(out * ct.<i>) の scale / bias / x についての勾配 grad.scale.<i> など
を記録する（パラメータはステップ間で変えない。移動平均だけが前のステップの値を引き継ぐ）。最後に、
推論モード（use_running_average=True）で最後の移動平均を使った出力 eval.out と、その勾配 eval.grad.* を
記録する。f32（jax_enable_x64 は無効）。

出力形式: 1つの s式。配列は (名前 形状 値) で、値は f32 の u32 ビットパターン。
  (:versions "jax x.y.z flax a.b.c"
   :cases ((:name "batch-norm" :axis -1 :momentum 0.99 :epsilon 1e-5 :use-scale t :use-bias t :steps 3
            :arrays (("scale" (5) (...)) ...)) ...))

使い方 (jax[cpu]==0.10.2 と flax の入った venv で):
    python3 tests/fixtures/layers/generate-batch-norm.py > tests/fixtures/layers/flax-batch-norm.lisp
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def bits(x):
    return [int(v) for v in np.asarray(x, dtype=np.float32).view(np.uint32).reshape(-1)]


def arr(name, x):
    x = np.asarray(x)
    shape = " ".join(str(d) for d in x.shape)
    return f'("{name}" ({shape}) ({" ".join(str(v) for v in bits(x))}))'


def rnd(rng, *shape, scale=1.0, shift=0.0):
    return (scale * rng.standard_normal(shape) + shift).astype(np.float32)


def lisp_bool(b):
    return "t" if b else "nil"


def case(name, rng, x_shape, axis=-1, momentum=0.99, epsilon=1e-5, use_scale=True, use_bias=True, steps=3,
         x_scale=2.0, x_shift=3.0):
    module = nn.BatchNorm(axis=axis, momentum=momentum, epsilon=epsilon, use_scale=use_scale, use_bias=use_bias)
    rank = len(x_shape)
    axes = [axis] if isinstance(axis, int) else list(axis)
    feature_axes = sorted(a % rank for a in axes)
    feature_shape = tuple(x_shape[a] for a in feature_axes)

    params = {}
    if use_scale:
        params["scale"] = rnd(rng, *feature_shape, scale=0.5, shift=1.0)
    if use_bias:
        params["bias"] = rnd(rng, *feature_shape)
    # 移動平均の初期値は 0 / 1（Flax の init）でなく任意の値にする（momentum の向きを確かめるため）
    stats = {"mean": rnd(rng, *feature_shape), "var": np.abs(rnd(rng, *feature_shape)) + 0.5}
    items = [arr(k, v) for k, v in params.items()]
    items += [arr("mean.init", stats["mean"]), arr("var.init", stats["var"])]

    def train_loss(p, x, s, ct):
        out, updated = module.apply({"params": p, "batch_stats": s}, x, use_running_average=False,
                                    mutable=["batch_stats"])
        return jnp.sum(out * ct), (out, updated["batch_stats"])

    def eval_loss(p, x, s, ct):
        out = module.apply({"params": p, "batch_stats": s}, x, use_running_average=True)
        return jnp.sum(out * ct)

    p = {k: jnp.asarray(v) for k, v in params.items()}
    s = {k: jnp.asarray(v) for k, v in stats.items()}
    for i in range(steps):
        x = rnd(rng, *x_shape, scale=x_scale, shift=x_shift)
        ct = rnd(rng, *x_shape)
        (_, (out, s)), (gp, gx) = jax.value_and_grad(train_loss, argnums=(0, 1), has_aux=True)(p, x, s, ct)
        items += [arr(f"x.{i}", x), arr(f"ct.{i}", ct), arr(f"out.{i}", out),
                  arr(f"mean.{i}", s["mean"]), arr(f"var.{i}", s["var"])]
        items += [arr(f"grad.{k}.{i}", gp[k]) for k in params]
        items.append(arr(f"grad.x.{i}", gx))

    x = rnd(rng, *x_shape, scale=x_scale, shift=x_shift)
    ct = rnd(rng, *x_shape)
    out = module.apply({"params": p, "batch_stats": s}, x, use_running_average=True)
    gp, gx = jax.grad(eval_loss, argnums=(0, 1))(p, x, s, ct)
    items += [arr("eval.x", x), arr("eval.ct", ct), arr("eval.out", out)]
    items += [arr(f"eval.grad.{k}", gp[k]) for k in params]
    items.append(arr("eval.grad.x", gx))

    axis_lisp = str(axis) if isinstance(axis, int) else "(" + " ".join(str(a) for a in axis) + ")"
    return (f'(:name "{name}" :axis {axis_lisp} :momentum {momentum!r} :epsilon {epsilon!r} '
            f':use-scale {lisp_bool(use_scale)} :use-bias {lisp_bool(use_bias)} :steps {steps}\n'
            f'  :arrays ({" ".join(items)}))')


def main():
    rng = np.random.default_rng(260)
    cases = [
        # 既定（Flax と同じ momentum 0.99・epsilon 1e-5・最後の軸）。[バッチ 特徴]
        case("batch-norm", rng, (8, 5)),
        # 画像（NHWC）。統計はバッチと空間の軸 (0 1 2) で取る
        case("batch-norm-nhwc", rng, (2, 3, 4, 5)),
        # チャネルが軸 1（NCHW のような配置）。momentum と epsilon も既定と変える
        case("batch-norm-axis1", rng, (4, 3, 2, 2), axis=1, momentum=0.9, epsilon=1e-3),
        # 特徴の軸が2つ（scale / bias は (3 4)）
        case("batch-norm-two-feature-axes", rng, (6, 3, 4), axis=(1, 2)),
        # scale / bias なし
        case("batch-norm-no-scale-no-bias", rng, (8, 5), use_scale=False, use_bias=False),
        # 分散が epsilon と同程度（平均は 0 にして速い分散の桁落ちを避ける）
        case("batch-norm-tiny-variance", rng, (8, 4), x_scale=3e-3, x_shift=0.0),
    ]
    print(f'(:versions "jax {jax.__version__} flax {flax.__version__}"')
    print(" :cases (" + "\n         ".join(cases) + "))")


if __name__ == "__main__":
    main()
